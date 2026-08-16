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
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, List, Optional, Sequence

import pandas as pd
from sqlalchemy import text

from .base import (
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
    FEATURES: frozenset = frozenset({FEATURE_KEYSET, FEATURE_HASH_DIFF})

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
    zero_datetime_literal: Optional[str] = "0000-00-00 00:00:00"

    #: **Without this the whole result is pulled into the client's RAM.**
    #: SQLAlchemy 1.4 takes a buffered cursor for `mysqlconnector`, so
    #: `read_sql(chunksize=)` does hand out batches, but only once everything has
    #: been downloaded. Measured on 6 M rows: first batch after 15.53 s and a
    #: peak RSS of 4 667 MB, against 0.41 s and 318 MB — **14.7x less memory**
    #: (see `scripts/bench_hash_memory.py`).
    #:
    #: The orchestrator runs 3 pipelines at once, so the difference is threefold
    #: at the peak.
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
        where: Optional[str] = None,
        *,
        known_total_rows: Optional[int] = None,
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
        where: Optional[str] = None,
        order_by: Optional[Sequence[str]] = None,
        surrogate: Optional[SurrogateKey] = None,
        *,
        column_types: Optional[dict] = None,
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
        column_types: Optional[dict] = None,
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

    # -- reading -----------------------------------------------------------

    def iter_batches(self, engine: Engine, sql: str, batch_size: int) -> Iterator[pd.DataFrame]:
        """Reads the result in batches from one query rather than by paging.

        ``pd.read_sql(chunksize=)`` holds a single cursor and takes
        ``batch_size`` rows from it at a time. Against the predecessors'
        ``LIMIT/OFFSET`` paging that has two advantages: the source does not have
        to skip a growing prefix for every next page (with 15 M rows and a batch
        of 50 000 that is 302 queries with an average offset of 7.5 M), and the
        result is consistent, so no duplicate can arise from data shifting
        between pages.

        How much this saves in practice still has to be measured against a live
        source, not estimated.
        """
        with engine.connect() as con:
            yield from pd.read_sql(sql, con, chunksize=int(batch_size))


__all__ = ["MySQLDialect"]
