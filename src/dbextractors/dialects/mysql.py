"""``MySQLDialect`` — the adapter for MySQL as a source.

MySQL is the largest source by volume (73 % of the data, see ARCHITECTURE.md).
It comes fourth in the migration order on purpose — if the golden test turns out
to have holes, let that show on 21 tables rather than on 254.

**Implemented ahead of the other dialects.** The golden test has to run against
real tables, but the new component does not start at all without a concrete
dialect, so one dialect had to exist first. The remaining three follow in the
planned order.

Inherits from ``DictTypeMapDialect`` — the MySQL type map is a declarative dict,
unlike Firebird where the sign of ``scale`` also decides the PostgreSQL type.
"""

from __future__ import annotations

import logging
import socket
import urllib.parse
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, List, Sequence

import pandas as pd
from sqlalchemy import text

from .base import (
    ABRA_COUNTER_LENGTH,
    FEATURE_ABRA_WATERMARK,
    FEATURE_HASH_DIFF,
    FEATURE_KEYSET,
    ColumnDef,
    DictTypeMapDialect,
    SizeEstimate,
    SurrogateKey,
    TableRef,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine

_log = logging.getLogger(__name__)


class MySQLDialect(DictTypeMapDialect):
    """Adapter for MySQL. Driver and quoting are exactly those of the predecessors."""

    name: str = "mysql"
    default_port: int = 3306
    #: Exactly as the predecessors spell it (mysql-connector-python).
    sqlalchemy_driver: str = "mysql+mysqlconnector"
    quote_char: str = "`"
    FEATURES: frozenset = frozenset({FEATURE_KEYSET, FEATURE_HASH_DIFF, FEATURE_ABRA_WATERMARK})

    #: MySQL returns text as UTF-8; the cascade ``cp1250 -> utf-8`` must **not**
    #: be generalised to here (see the decision at `coerce.decode_bytes_columns`).
    decode_encodings: tuple = ("utf-8",)

    #: Types that have to be read as text. Without this, pandas infers
    #: ``float64`` for an eighteen-digit EAN in a ``varchar`` column and the
    #: number loses precision. The list is taken verbatim from the predecessor
    #: (``TEXT_LIKE_MYSQL_TYPES``).
    text_like_types: frozenset = frozenset(
        {
            "char",
            "varchar",
            "tinytext",
            "text",
            "mediumtext",
            "longtext",
            "enum",
            "set",
            "binary",
            "varbinary",
        }
    )

    #: MySQL can store a date that is not NULL and does not exist either.
    zero_datetime_literal: str | None = "0000-00-00 00:00:00"

    #: The driver's default for cursors opened on the connection. **It is not
    #: what makes reading stream any more**: SQLAlchemy 2 opens every cursor of
    #: this dialect as ``cursor(buffered=True)`` and so overrides it, which is why
    #: `iter_batches` opens its own unbuffered cursor on the DBAPI connection.
    #: Kept because a cursor opened without an explicit ``buffered`` must still
    #: not pull a whole result into RAM. Under SQLAlchemy 1.4 this alone was the
    #: difference, measured on 6 M rows: first batch after 15.53 s and a peak RSS
    #: of 4 667 MB, against 0.41 s and 318 MB (see `scripts/bench_hash_memory.py`).
    connect_args: ClassVar[dict] = {"buffered": False}

    #: The server waits for a slow client only ``net_write_timeout`` seconds
    #: (MySQL defaults to 60) and then closes the connection — which the client
    #: sees as ``2013 Lost connection to MySQL server during query``. Unbuffered
    #: batch reading exposes this fully: the conversion of a whole batch happens
    #: between two ``fetch`` calls, and on wide tables that is a noticeable pause.
    #: Documented by two long extractions that died this way in production.
    #:
    #: Ten minutes is comfortably above what converting one batch costs (the
    #: worst measured throughput is 3 644 rows/s on 48 columns, i.e. ~14 s for a
    #: batch of 50 000). It does not replace read recovery (`core.reading`) — it
    #: only removes the most frequent cause.
    session_sql: ClassVar[tuple] = ("SET SESSION net_write_timeout = 600, net_read_timeout = 600",)

    #: Source type (lowercase) -> PostgreSQL type. Taken verbatim from the
    #: predecessor's ``MYSQL_TO_PG_TYPES``. ``tinyint`` is ``INTEGER`` here, not
    #: ``BOOLEAN`` — the promotion to boolean is decided from the data (an enum
    #: of 0/1), not from the type.
    type_map: ClassVar[dict] = {
        "int": "INTEGER",
        "integer": "INTEGER",
        "bigint": "BIGINT",
        "smallint": "INTEGER",
        "mediumint": "INTEGER",
        "tinyint": "INTEGER",
        "bit": "BOOLEAN",
        "bool": "BOOLEAN",
        "boolean": "BOOLEAN",
        "float": "REAL",
        "double": "DOUBLE PRECISION",
        "double precision": "DOUBLE PRECISION",
        "decimal": "NUMERIC",
        "dec": "NUMERIC",
        "numeric": "NUMERIC",
        "fixed": "NUMERIC",
        "char": "TEXT",
        "varchar": "VARCHAR",
        "tinytext": "TEXT",
        "text": "TEXT",
        "mediumtext": "TEXT",
        "longtext": "TEXT",
        "enum": "TEXT",
        "set": "TEXT",
        "binary": "TEXT",
        "varbinary": "TEXT",
        "blob": "BYTEA",
        "tinyblob": "BYTEA",
        "mediumblob": "BYTEA",
        "longblob": "BYTEA",
        "date": "DATE",
        "datetime": "TIMESTAMP",
        "timestamp": "TIMESTAMP",
        "time": "TIME",
        "year": "INTEGER",
        "json": "JSONB",
    }

    # -- connecting --------------------------------------------------------

    def build_conn_str(self, params: dict, host: str, port: int) -> str:
        """The SQLAlchemy URL. The password **must** be escaped or an `@` in it breaks the URL."""
        user = urllib.parse.quote_plus(str(params.get("user", "")))
        password = urllib.parse.quote_plus(str(params.get("password", "")))
        database = params.get("database", "")
        # `or`, not `get(key, default)`. Configuration parsing deliberately does
        # not fill in a charset — which port or encoding a database expects is
        # the dialect's knowledge, not the parser's — so a configuration without
        # one arrives here as an explicit ``None``. With `get(…, default)` that
        # `None` wins and the URL becomes ``?charset=None``, which the driver
        # rejects with "Character set 'None' unsupported".
        charset = params.get("charset") or "utf8mb4"
        return (
            f"{self.sqlalchemy_driver}://{user}:{password}@{host}:{port}/{database}"
            f"?charset={charset}"
        )

    def probe(self, host: str, port: int, timeout: float = 10.0) -> bool:
        """Is the source reachable? The predecessors' ``is_mysql_accessible``.

        TCP only, no login: this decides between a direct connection and a
        tunnel, and for that it is enough to know whether the port is open.
        """
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError as err:
            _log.debug("MySQL %s:%s is not reachable: %s", host, port, err)
            return False

    # -- introspection -----------------------------------------------------

    def introspect_columns(self, engine: Engine, ref: TableRef) -> List[ColumnDef]:
        """Columns from ``INFORMATION_SCHEMA.COLUMNS``, in the source's own order.

        The order is part of the contract — the golden test compares it right
        after the row count. ``target_name`` is **not** filled in here: that is
        `core.naming`'s job, and the frozen target column names depend on it.
        """
        query = text(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = COALESCE(:db, DATABASE()) AND TABLE_NAME = :tbl "
            "ORDER BY ORDINAL_POSITION"
        )
        with engine.connect() as con:
            rows = con.execute(
                query, {"db": ref.database or ref.schema, "tbl": ref.name}
            ).fetchall()

        if not rows:
            raise RuntimeError(
                f"Table {ref.name!r} does not exist in the source or has no columns. "
                "Check source_name and database in the configuration."
            )

        out = []
        for name, data_type, nullable, ordinal in rows:
            source_type = str(data_type).lower()
            out.append(
                ColumnDef(
                    name=name,
                    source_type=source_type,
                    pg_type=self.type_map.get(source_type, "TEXT"),
                    nullable=str(nullable).upper() == "YES",
                    ordinal=int(ordinal),
                )
            )
        return out

    def estimate_size(
        self,
        engine: Engine,
        ref: TableRef,
        where: str | None = None,
        *,
        known_total_rows: int | None = None,
        sample_size: int = 100,
    ) -> SizeEstimate:
        """``COUNT(*)`` plus a sample to estimate the row size.

        **A failure is not swallowed.** The predecessor returns ``(0, 0)`` when
        the sample is empty — which is legitimate, the table really can be empty
        — but it does let an exception from ``COUNT(*)`` through, and that is the
        part that matters: an unreachable source must never look like an empty
        table.
        """
        where_sql = f" WHERE ({where})" if where else ""
        table = self.quote_ident(ref.name)

        with engine.connect() as con:
            if known_total_rows is not None:
                total_rows = int(known_total_rows)
                method = "known_total_rows"
            else:
                total_rows = int(
                    con.execute(text(f"SELECT COUNT(*) FROM {table}{where_sql}")).scalar() or 0
                )
                method = "count_and_sample"

            if total_rows == 0:
                return SizeEstimate(rows=0, size_mb=0.0, method=method)

            sample = pd.read_sql(f"SELECT * FROM {table}{where_sql} LIMIT {int(sample_size)}", con)

        if sample.empty:
            return SizeEstimate(rows=total_rows, size_mb=0.0, method=method)

        avg_row_size = sample.memory_usage(deep=True).sum() / len(sample)
        size_mb = round((avg_row_size * total_rows) / 1024 / 1024, 2)
        return SizeEstimate(rows=total_rows, size_mb=float(size_mb), method=method)

    # -- SQL generation ----------------------------------------------------

    def render_select(
        self,
        columns: Sequence[str],
        ref: TableRef,
        where: str | None = None,
        order_by: Sequence[str] | None = None,
        surrogate: SurrogateKey | None = None,
        *,
        column_types: dict | None = None,
        convert_nchar: bool = False,
    ) -> str:
        clause = self.render_select_clause(
            columns, surrogate, column_types=column_types, convert_nchar=convert_nchar
        )
        sql = f"SELECT {clause} FROM {self.quote_ident(ref.name)}"
        if where:
            sql += f" WHERE ({where})"
        if order_by:
            sql += " ORDER BY " + ", ".join(self.quote_ident(c) for c in order_by)
        return sql

    def render_hash_expr(
        self,
        hashed_columns: Sequence[str],
        alias: str,
        column_types: dict | None = None,
    ) -> str:
        """``SHA2(CONCAT_WS('||', COALESCE(CAST(`c` AS CHAR), '')…), 256)``.

        **Byte-for-byte identical to the predecessor's ``build_hash_expr``** —
        and that is the whole proof that ``row_hash`` does not change: the same
        expression in the same engine over the same data returns the same digest.
        Checked by `tests/dialects/test_mysql.py`.

        ``column_types`` is not needed by MySQL (only MSSQL has it, because of
        binary types); it is accepted for the sake of the shared signature.
        """
        del column_types
        parts = ", ".join(
            f"COALESCE(CAST({self.quote_ident(c)} AS CHAR), '')" for c in hashed_columns
        )
        return f"SHA2(CONCAT_WS('||', {parts}), 256) AS {self.quote_ident(alias)}"

    def sql_literal(self, value: Any) -> str:
        """As in the base class, but a backslash is doubled as well.

        MySQL is the only one of the four dialects that by default
        (``NO_BACKSLASH_ESCAPES`` off) treats ``\\`` inside a string as an escape
        character. Without doubling it, a key ending in a backslash would eat the
        closing apostrophe.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        escaped = str(value).replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'"

    def render_abra_counter(self, key_expr: str) -> str:
        """A ``BINARY`` cast makes the comparison against the literal byte-wise."""
        return f"CAST(REVERSE(SUBSTRING({key_expr}, 1, {ABRA_COUNTER_LENGTH})) AS BINARY)"

    def render_abra_suffix(self, key_expr: str) -> str:
        return f"SUBSTRING({key_expr}, {ABRA_COUNTER_LENGTH + 1})"

    # -- reading -----------------------------------------------------------

    def iter_batches(self, engine: Engine, sql: str, batch_size: int) -> Iterator[pd.DataFrame]:
        """Streams the result in batches from one query on an unbuffered cursor.

        One query rather than paging: the predecessors' ``LIMIT/OFFSET`` made the
        source skip a growing prefix for every page (with 15 M rows and a batch of
        50 000 that is 302 queries with an average offset of 7.5 M), and a single
        result is consistent, so no duplicate can arise from data shifting between
        pages.

        **Not ``pd.read_sql(chunksize=)``.** That goes through SQLAlchemy, and
        SQLAlchemy 2's ``mysqlconnector`` dialect opens every cursor as
        ``cursor(buffered=True)`` — overriding ``connect_args`` — and has no
        server-side cursor to ask for instead. The first ``fetchmany`` then waits
        until the whole result is in client RAM: an ``id_watermark`` delta of
        42.7 M rows was killed at 14 GiB without yielding a single batch. So the
        cursor is opened here, on the DBAPI connection, unbuffered by name.
        The ``connect`` listener with ``session_sql`` still runs —
        ``raw_connection()`` checks out through the same pool.

        The frames are built the way pandas builds them in ``read_sql`` (see
        `_frame`), and an empty result still yields one empty frame with the
        column names, as ``read_sql`` does.
        """
        size = int(batch_size)
        raw = engine.raw_connection()
        cursor = None
        drained = False
        disconnected = False
        # Taken before the query: it is what `KILL QUERY` needs if the result is
        # abandoned, and asking a connection mid-stream is not possible.
        connection_id = getattr(raw.dbapi_connection, "connection_id", None)
        try:
            try:
                cursor = raw.cursor(buffered=False)
                cursor.execute(sql)
                columns = [d[0] for d in cursor.description or ()]
            except Exception as err:
                disconnected = _is_disconnect(engine, err, raw, cursor)
                _raise_as_sqlalchemy(engine, sql, err, disconnected)
                raise

            yielded = False
            while True:
                try:
                    rows = cursor.fetchmany(size)
                except Exception as err:
                    disconnected = _is_disconnect(engine, err, raw, cursor)
                    _raise_as_sqlalchemy(engine, sql, err, disconnected)
                    raise
                if not rows:
                    break
                yielded = True
                yield _frame(rows, columns)
            drained = True
            if not yielded:
                yield _frame([], columns)
        finally:
            _release(
                engine,
                raw,
                cursor,
                drained=drained,
                disconnected=disconnected,
                connection_id=connection_id,
            )


# --- Helper functions -------------------------------------------------------


def _frame(rows: Sequence[Any], columns: List[str]) -> pd.DataFrame:
    """Rows from the driver -> a frame, **as ``pd.read_sql`` builds it**.

    ``read_sql`` hands each ``fetchmany`` to ``DataFrame`` construction with
    ``coerce_float=True`` and no ``dtype``; ``from_records`` with the same
    argument goes through the same object-array inference, so ``Decimal`` stays
    ``object``, an integer column with a NULL becomes ``float64``, ``TIME``
    becomes ``timedelta64[ns]`` and ``BIGINT UNSIGNED`` ``uint64`` — exactly what
    the conversion layer was written against. The one step of ``read_sql`` left
    out is converting tz-aware columns to UTC: the driver returns MySQL
    ``DATETIME``/``TIMESTAMP`` naive, so there is never such a column. Parity is
    checked live against ``read_sql`` in ``tests/dialects/test_source_db.py``.
    """
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(list(rows), columns=columns, coerce_float=True)


def _is_disconnect(engine: Engine, err: BaseException, raw: Any, cursor: Any) -> bool:
    """SQLAlchemy's own verdict on whether the error means the connection is gone."""
    try:
        return bool(engine.dialect.is_disconnect(err, raw.dbapi_connection, cursor))  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - a classifier must never mask the error
        return False


def _raise_as_sqlalchemy(engine: Engine, sql: str, err: Exception, disconnected: bool) -> None:
    """Re-raise a driver error wrapped the way SQLAlchemy wraps it on its own path.

    Read recovery (`core.reading.is_temporary_error`) decides on SQLAlchemy's
    exception taxonomy, and the text in the log stays the familiar
    ``(mysql.connector.errors.OperationalError) 2013 …``. Reading past
    SQLAlchemy must not quietly turn a retryable drop into a fatal error.
    Anything that is not a driver error returns, for the caller to re-raise
    unchanged.
    """
    from sqlalchemy import exc as sa_exc

    dbapi = getattr(engine.dialect, "loaded_dbapi", None) or engine.dialect.dbapi
    base_error = getattr(dbapi, "Error", None)
    if base_error is None or not isinstance(err, base_error):
        return
    raise sa_exc.DBAPIError.instance(
        sql,
        None,
        err,
        base_error,
        connection_invalidated=disconnected,
        dialect=engine.dialect,
    ) from err


def _release(
    engine: Engine,
    raw: Any,
    cursor: Any,
    *,
    drained: bool,
    disconnected: bool,
    connection_id: Any,
) -> None:
    """Close the cursor and the connection without draining an abandoned result.

    When the result was read to the end this is an ordinary close. When it was
    not — the consumer stopped early, or something failed mid-stream — the
    unbuffered result is still coming down the wire, and neither close is
    cheap: ``cursor.close()`` refuses with *Unread result found*, and closing
    the connection makes the client library read and discard every remaining
    row first (measured: 8.4 s for 10 M narrow rows; the 42.7 M-row delta
    would take minutes). So the query is cancelled first with ``KILL QUERY``
    from a second, short connection, and the connection is then discarded
    rather than returned. A connection that is already gone is simply
    discarded. Nothing here raises: it runs in ``finally`` and must not replace
    the error that brought it here.
    """
    if drained:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            raw.close()
        return

    pending = getattr(getattr(raw, "dbapi_connection", None), "unread_result", True)
    if pending and not disconnected and connection_id is not None:
        try:
            with engine.connect() as side:
                side.exec_driver_sql(f"KILL QUERY {int(connection_id)}")
            _log.debug("The abandoned query on connection %s was cancelled.", connection_id)
        except Exception as err:
            _log.warning(
                "⚠️ The abandoned source query could not be cancelled (%s); "
                "closing the connection will read out the rest of the result.",
                type(err).__name__,
            )
    try:
        raw.invalidate()
    except Exception as err:  # pragma: no cover - best effort
        _log.debug("Discarding the source connection failed: %s", type(err).__name__)


__all__ = ["MySQLDialect"]
