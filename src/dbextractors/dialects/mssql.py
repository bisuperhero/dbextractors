"""``MSSQLDialect`` — the adapter for MSSQL as a source.

In the migration order from ARCHITECTURE.md MSSQL comes **second** (after
PostgreSQL): that source has 16 tables and is the only one today that knows
``partition_by_source`` (a target table assembled from several sources) and
multi-source. It also lights up the `full_by_source` strategy, which until then
had only been exercised against a fake source.

Inherits from ``DictTypeMapDialect`` — the type map is a declarative dict.

Both predecessors (variant C and variant B) have the type map and
``build_hash_expr`` **identical character for character**, so there was nothing
to merge. They differ in three small things, and variant C wins all three:

- ``probe`` retries three times with a delay in variant C (variant B tries once).
  The reason is in its own comment: a short TCP outage would otherwise drop the
  run into the SSH branch, which reports a confusing error when no tunnel is
  configured on purpose.
- ``connect_args`` additionally carries ``login_timeout``.
- ``estimate_size`` understands ``known_total_rows`` (the fingerprint has already
  counted the rows, so there is no point asking a second time).

Four things that keep MSSQL from being just another ``DictTypeMapDialect``
subclass:

- **square-bracket quoting** ``[ ]`` — ``quote_char`` is a single character and
  does not stretch to a bracket pair, so ``quote_ident()`` is overridden.
- **paging** has no ``LIMIT``; it has ``TOP (n)`` and ``OFFSET … ROWS FETCH NEXT``.
- **``COUNT_BIG``** in the fingerprint — ``COUNT`` is an ``INT`` and above
  2.1 billion rows it **fails** with an overflow rather than rounding quietly.
- **binary types in the hash** — ``image``/``binary``/``varbinary`` cannot be cast
  to ``NVARCHAR(MAX)`` directly, which is why ``render_hash_expr`` is the only one
  with ``column_types`` in its signature.

## The encoding is cp1250, not UTF-8

The source system keeps its databases in cp1250. The predecessors handle that
twice — ``connect_args={'charset': 'cp1250'}`` on the connection and the cascade
``cp1250 -> utf-8 -> latin-1`` for the remaining bytes — and both are kept here.
The order of the cascade must **not** be generalised to the other dialects:
cp1250 rejects only five bytes, so UTF-8 text passes through it as mojibake.

The ``SOURCE_DB.charset`` key is **not read** for MSSQL, even though the frozen
contract knows it. That is not an oversight — it is today's behaviour, and all 16
pipelines of this source have ``utf8mb4`` in that key, i.e. the name of a MySQL
encoding. Details at ``build_conn_str``.

## Known limitation: NVARCHAR text arrives cut short

**With this connection charset, ``NVARCHAR``, ``NCHAR`` and ``NTEXT`` values are
truncated at the first character that latin-1 cannot express — silently, with no
error and no replacement character.** ``'příliš žluťoučký kůň'`` arrives as
``'p'``; a pure-ASCII value is unaffected.

It is inherited rather than introduced: both predecessors open the connection
with the same charset, so the target tables have been receiving the truncated
text for as long as they have existed. Changing it is a parity decision and not
a bugfix — no charset value reads the N-types and the legacy single-byte columns
correctly at the same time. The measurements are in ``docs/legacy-compat.md``;
the behaviour is pinned by ``tests/dialects/test_source_db.py`` under "the MSSQL
client charset".

**``LOAD_SETTINGS.convert_nchar_to_varchar`` opts one table out of it.** With it
on, the N-typed columns are wrapped in ``CONVERT(VARCHAR(MAX), …)`` in the
generated SELECT, so the server converts them through their own collation and
they arrive whole over the same cp1250 connection; the legacy single-byte
columns are not touched, which is why this is the only remedy that does not
trade one corruption for another. It is off by default and per table, because
turning it on changes what lands in the target and the table has to be reloaded
for the change to reach the rows already there. See `render_nchar_convert`.
"""

from __future__ import annotations

import logging
import socket
import time
import urllib.parse
from typing import TYPE_CHECKING, ClassVar, Iterator, List, Optional, Sequence

import pandas as pd
from sqlalchemy import text

from dbextractors.core import secrets

from .base import (
    FEATURE_HASH_DIFF,
    FEATURE_KEYSET,
    FEATURE_PARTITION_BY_SOURCE,
    ColumnDef,
    DictTypeMapDialect,
    SizeEstimate,
    SurrogateKey,
    TableRef,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine

_log = logging.getLogger(__name__)

#: The schema searched when the configuration does not name one. It is the
#: default value of the ``src_schema`` parameter across the predecessors.
DEFAULT_SOURCE_SCHEMA = "dbo"

#: The encoding the connection is opened with. The source keeps its databases in
#: cp1250 and both predecessors hard-code that in ``connect_args`` —
#: ``SOURCE_DB.charset`` is not read for MSSQL, see ``build_conn_str``.
#:
#: **This value costs the N-types their contents** (see the module docstring):
#: pymssql converts UCS-2 to the single-byte charset as latin-1 and then decodes
#: the result as cp1250, so an ``NVARCHAR`` value is cut at its first character
#: outside latin-1 and a character *inside* latin-1 arrives as a different one.
#: Setting it to ``"UTF-8"`` rescues the N-types and mangles every legacy
#: ``VARCHAR`` under the source's CP1250 collation instead — measured across
#: seven candidate values, none of which is right for both. The way out is not a
#: different charset but ``LOAD_SETTINGS.convert_nchar_to_varchar``, which leaves
#: this value alone and converts the N-types on the server.
DEFAULT_CHARSET = "cp1250"

#: Types that ``CAST(… AS NVARCHAR(MAX))`` will not convert. They have to go
#: through ``CONVERT(…, VARBINARY(MAX), …, 1)``, i.e. to hex.
BINARY_TYPES: frozenset = frozenset({"image", "binary", "varbinary"})


class MSSQLDialect(DictTypeMapDialect):
    """Adapter for MSSQL. The only dialect that knows partitioning and multi-source today."""

    name: str = "mssql"
    default_port: int = 1433
    #: Exactly as the predecessors spell it (pymssql, not pyodbc).
    sqlalchemy_driver: str = "mssql+pymssql"
    #: The opening bracket only — ``quote_ident`` is overridden, because one
    #: character does not cover a ``[`` … ``]`` pair. It is kept here to document
    #: the intent.
    quote_char: str = "["
    FEATURES: frozenset[str] = frozenset(
        {FEATURE_KEYSET, FEATURE_HASH_DIFF, FEATURE_PARTITION_BY_SOURCE}
    )

    #: The cascade taken verbatim from the predecessors (``decode_bytes_columns``).
    #: The source is cp1250, but not everything in it really is cp1250 — hence
    #: the two further steps.
    decode_encodings: tuple = ("cp1250", "utf-8", "latin-1")

    #: Types read as text. **Binary types are deliberately absent**:
    #: ``apply_text_dtypes`` decodes bytes as UTF-8 with replacement characters,
    #: whereas the predecessors put them through the ``decode_encodings``
    #: cascade — and that yields different text. For ``image``/``varbinary``
    #: columns it would change the contents of the target.
    text_like_types: frozenset = frozenset(
        {"char", "varchar", "nchar", "nvarchar", "text", "ntext", "xml"}
    )

    #: The national-character types: stored as UCS-2, hence the ones the
    #: ``cp1250`` connection mangles. ``LOAD_SETTINGS.convert_nchar_to_varchar``
    #: converts exactly these and nothing else — single-byte ``varchar``/``char``/
    #: ``text`` are read correctly today and wrapping them would break them.
    #: MSSQL is the only dialect that fills this in; see `render_nchar_convert`.
    NCHAR_TYPES: frozenset = frozenset({"nchar", "nvarchar", "ntext"})

    #: MSSQL has no zero date (unlike MySQL).
    zero_datetime_literal: Optional[str] = None

    #: The hash is computed by the source through ``HASHBYTES`` — see
    #: ``render_hash_expr``.
    hash_in_pandas: bool = False

    #: ``charset`` is hard-coded to cp1250 and **deliberately ignores
    #: ``SOURCE_DB.charset`` from the configuration** — see ``build_conn_str``.
    #:
    #: **It truncates every ``NVARCHAR``/``NCHAR``/``NTEXT`` value at the first
    #: character latin-1 cannot express**, and it is kept anyway because the
    #: predecessors do the same and because dropping it corrupts the legacy
    #: single-byte columns instead. A table that needs those columns whole sets
    #: ``LOAD_SETTINGS.convert_nchar_to_varchar`` rather than changing this. See
    #: ``DEFAULT_CHARSET`` and ``docs/legacy-compat.md`` before changing this line.
    #:
    #: ``login_timeout`` comes from variant C. Without it, logging in to a
    #: memory-starved server hangs on the driver's default timeout; the TCP probe
    #: passes meanwhile, because the port is listening.
    connect_args: ClassVar[dict] = {"charset": DEFAULT_CHARSET, "login_timeout": 30}

    #: Source type (lowercase) -> PostgreSQL type. Taken verbatim from variant
    #: C's ``MSSQL_TO_PG_TYPES``; variant B has the same map character for
    #: character, so nothing had to be merged.
    #:
    #: Two things that look like mistakes and are not: ``money`` is ``NUMERIC``
    #: here (unlike a PostgreSQL source, where psycopg2 returns it as a localised
    #: string — pymssql returns a ``Decimal``), and the binary types go to
    #: ``TEXT`` rather than ``BYTEA``, because that is what is in the target today.
    type_map: ClassVar[dict] = {
        "int": "INTEGER",
        "integer": "INTEGER",
        "bigint": "BIGINT",
        "smallint": "INTEGER",
        "tinyint": "INTEGER",
        "bit": "BOOLEAN",
        "float": "DOUBLE PRECISION",
        "real": "REAL",
        "decimal": "NUMERIC",
        "numeric": "NUMERIC",
        "money": "NUMERIC",
        "smallmoney": "NUMERIC",
        "char": "TEXT",
        "varchar": "VARCHAR",
        "nchar": "TEXT",
        "nvarchar": "TEXT",
        "text": "TEXT",
        "ntext": "TEXT",
        "binary": "TEXT",
        "varbinary": "TEXT",
        "image": "TEXT",
        "date": "DATE",
        "datetime": "TIMESTAMP",
        "datetime2": "TIMESTAMP",
        "smalldatetime": "TIMESTAMP",
        "datetimeoffset": "TIMESTAMP",
        "time": "TIME",
        "uniqueidentifier": "TEXT",
        "xml": "TEXT",
        "json": "JSONB",
    }

    # -- connecting --------------------------------------------------------

    def build_conn_str(self, params: dict, host: str, port: int) -> str:
        """The SQLAlchemy URL. Password and database name both **must** be escaped.

        Escaping the database name comes from variant C's ``_build_engine`` — in a
        multi-source run the name comes from ``DATABASES`` in the configuration,
        so it can contain anything MSSQL allows in a name.

        ``SOURCE_DB.charset`` is **deliberately ignored** here and the encoding is
        taken from ``connect_args`` (cp1250). It looks like a breach of the frozen
        contract, but it is the opposite: this is exactly what the MSSQL extractor
        does today — both predecessors hard-code cp1250 in ``connect_args`` and
        never read the key from the configuration.

        Why it matters: **all 16 pipelines against this source have
        ``"charset": "utf8mb4"`` in their configuration** (verified in their
        config blocks). That is the name of a *MySQL* encoding, which pymssql does
        not know — plainly a leftover from a template. If this dialect started
        honouring it, text decoding would change in all 16 tables at once.

        MySQL is the other way round: there ``charset`` is still used in the URL,
        so ``MySQLDialect.build_conn_str`` does read it.
        """
        user = urllib.parse.quote_plus(str(params.get("user", "")))
        password = urllib.parse.quote_plus(str(params.get("password", "")))
        database = urllib.parse.quote_plus(str(params.get("database", "")))
        return f"{self.sqlalchemy_driver}://{user}:{password}@{host}:{port}/{database}"

    def probe(
        self,
        host: str,
        port: int,
        timeout: float = 10.0,
        *,
        attempts: int = 3,
        retry_delay: float = 3.0,
    ) -> bool:
        """Is the source reachable? TCP only, no login.

        It is the only one of the four dialects that tries **repeatedly**, and
        that is variant C's intent: a short outage would otherwise skip past the
        (already retrying) direct-connection branch and fall into the SSH branch,
        which reports a confusing error when no tunnel is configured on purpose.
        """
        for attempt in range(1, int(attempts) + 1):
            try:
                with socket.create_connection((host, int(port)), timeout=timeout):
                    return True
            except OSError as err:
                if attempt < attempts:
                    _log.warning(
                        "MSSQL %s:%s is not reachable (attempt %d/%d): %s — retrying in %.1f s.",
                        host,
                        port,
                        attempt,
                        attempts,
                        secrets.redact(err),
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    _log.error(
                        "MSSQL %s:%s is not reachable even after %d attempts: %s",
                        host,
                        port,
                        attempts,
                        secrets.redact(err),
                    )
        return False

    # -- introspection -----------------------------------------------------

    def introspect_columns(self, engine: Engine, ref: TableRef) -> List[ColumnDef]:
        """Columns from ``INFORMATION_SCHEMA.COLUMNS``, in the source's own order.

        ``TABLE_CATALOG`` is part of the filter too — in a multi-source run the
        same table is read from several databases, and without the catalog the
        query would return the columns of whichever one the open connection
        happens to point at. The order is part of the contract; the golden test
        compares it right after the row count.

        ``target_name`` is **not** filled in here — that is `core.naming`'s job,
        and the frozen target column names depend on it.
        """
        podminky = ["TABLE_SCHEMA = :sch", "TABLE_NAME = :tbl"]
        parametry: dict = {"sch": _schema(ref), "tbl": ref.name}
        if ref.database:
            podminky.insert(0, "TABLE_CATALOG = :db")
            parametry["db"] = ref.database

        query = text(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE {' AND '.join(podminky)} "
            "ORDER BY ORDINAL_POSITION"
        )
        with engine.connect() as con:
            rows = con.execute(query, parametry).fetchall()

        if not rows:
            raise RuntimeError(
                f"Table [{_schema(ref)}].[{ref.name}] does not exist in the source or has "
                "no columns. Check source_name and source_schema in the configuration."
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
        self._warn_about_binary(ref, out)
        return out

    @staticmethod
    def _warn_about_binary(ref: TableRef, columns: List[ColumnDef]) -> None:
        """Warn that a table with a binary column will most likely not load.

        The predecessors' type map sends ``image``/``binary``/``varbinary`` to
        ``TEXT``, but binary data does not fit into CSV and `COPY` ends with
        "need to escape, but no escapechar set". **Today's extractor fails on it
        too** — verified on one such table, where both components fail with the
        same error — so this is not a regression but an inherited limitation.
        None of the 16 pipelines against this source has such a table.

        Letting it fail as late as `COPY` is needlessly obscure: the error then
        talks about CSV instead of about a column. It warns rather than raises —
        a column that is entirely NULL passes, and killing such a run would be
        worse than the limitation itself.
        """
        binarni = [c.name for c in columns if c.source_type in BINARY_TYPES]
        if not binarni:
            return
        _log.warning(
            "Table [%s].[%s] has binary columns (%s). The type map sends them to TEXT, "
            'and unless they are empty COPY fails with "need to escape, but no escapechar '
            "set\" — today's extractor fails on this too. Drop them with the "
            "excluded_columns key in the TABLE section.",
            _schema(ref),
            ref.name,
            ", ".join(binarni),
        )

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

        ``known_total_rows`` earns its keep right here: the fingerprint
        (``render_fingerprint``) has already counted the rows, so a second
        ``COUNT(*)`` over a table of millions is pure waste.

        **A failure is not swallowed.** The predecessors return ``(0, 0)`` even
        when ``COUNT(*)`` fails — an unreachable source then passes itself off as
        an empty table and, with ``empty_rows_ok``, overwrites the target with
        nothing. An empty sample is a different matter and is legitimate: the
        table really can be empty.
        """
        where_sql = f" WHERE ({where})" if where else ""
        table = self.qualified(ref)

        with engine.connect() as con:
            if known_total_rows is not None:
                total_rows = int(known_total_rows)
                method = "known_total_rows"
            else:
                total_rows = int(
                    con.execute(text(f"SELECT COUNT_BIG(*) FROM {table}{where_sql}")).scalar() or 0
                )
                method = "count_and_sample"

            if total_rows == 0:
                return SizeEstimate(rows=0, size_mb=0.0, method=method)

            sample = pd.read_sql(
                f"SELECT * FROM {table}{where_sql} ORDER BY (SELECT NULL) "
                f"OFFSET 0 ROWS FETCH NEXT {int(sample_size)} ROWS ONLY",
                con,
            )

        if sample.empty:
            return SizeEstimate(rows=total_rows, size_mb=0.0, method=method)

        avg_row_size = sample.memory_usage(deep=True).sum() / len(sample)
        size_mb = round((avg_row_size * total_rows) / 1024 / 1024, 2)
        return SizeEstimate(rows=total_rows, size_mb=float(size_mb), method=method)

    # -- SQL generation ----------------------------------------------------

    def quote_ident(self, name: str) -> str:
        """Quote an identifier with square brackets (``[name]``).

        Overridden because ``SourceDialect.quote_ident`` assumes one character on
        each side and MSSQL needs a ``[`` / ``]`` pair.
        """
        return f"[{name.strip()}]"

    def qualified(self, ref: TableRef) -> str:
        """``[schema].[table]``. An empty schema means ``dbo``."""
        return f"{self.quote_ident(_schema(ref))}.{self.quote_ident(ref.name)}"

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
        sql = f"SELECT {clause} FROM {self.qualified(ref)}"
        if where:
            sql += f" WHERE ({where})"
        if order_by:
            sql += " ORDER BY " + ", ".join(self.quote_ident(c) for c in order_by)
        return sql

    def render_nchar_convert(self, column_name: str) -> str:
        """``CONVERT(VARCHAR(MAX), [col]) AS [col]`` — the N-type read that survives cp1250.

        The server converts UCS-2 to single-byte text using the **column's own
        collation** (``SQL_Czech_CP1250_CI_AS`` on this source), and the result
        then travels over a ``cp1250`` connection as the cp1250 it already is.
        That is the whole trick: the value never passes through the latin-1 step
        the driver would otherwise take, so nothing is cut short.

        Measured on ``dbo.unicode_edge``: ``'příliš žluťoučký kůň'`` arrives whole
        instead of as ``'p'``. What CP1250 cannot express is degraded rather than
        truncated — Cyrillic and emoji become ``?``, and a character CP1250 has a
        base letter for is folded onto it (``'a-ø-b'`` -> ``'a-o-b'``), which is
        the collation's best-fit mapping, not ours to choose.

        ``VARCHAR(MAX)`` rather than a sized ``VARCHAR``: ``NTEXT`` has no length
        this could be derived from, and a too-small size would silently truncate
        — which is the very failure the key exists to remove.
        """
        quoted = self.quote_ident(column_name)
        return f"CONVERT(VARCHAR(MAX), {quoted}) AS {quoted}"

    def _col_cast(self, column_name: str, source_type: str) -> str:
        """The cast of one column for ``render_hash_expr``.

        ``image``/``binary``/``varbinary`` cannot be cast to ``NVARCHAR(MAX)``
        directly — MSSQL would produce nonsense (or fail on the type mismatch) —
        so the detour goes through ``VARBINARY(MAX)`` and style ``1``, i.e. hex.
        """
        if source_type.lower() in BINARY_TYPES:
            return (
                f"ISNULL(CONVERT(NVARCHAR(MAX), "
                f"CONVERT(VARBINARY(MAX), {self.quote_ident(column_name)}), 1), '')"
            )
        return f"ISNULL(CAST({self.quote_ident(column_name)} AS NVARCHAR(MAX)), '')"

    def render_hash_expr(
        self,
        hashed_columns: Sequence[str],
        alias: str,
        column_types: Optional[dict] = None,
    ) -> str:
        """``HASHBYTES('SHA2_256', CONCAT_WS('||', …))`` as 64 characters of hex.

        **Byte-for-byte identical to the predecessors' ``build_hash_expr``** (the
        same in variant C and variant B) — and that is the whole proof that
        ``row_hash`` does not change with the migration: the same expression in
        the same engine over the same data returns the same digest.

        It is the only dialect that needs ``column_types`` — without them a binary
        column would not be recognised and the hash would be computed from
        something other than it is today. The key is the **source** column name,
        the value its source type.

        Two properties that are easy to overlook and must stay: ``HASHBYTES`` over
        ``NVARCHAR`` hashes UTF-16LE (not UTF-8, as the pandas path does), and
        ``CONVERT(…, 2)`` returns hex in **upper case**. Both are what is stored
        in the target today.
        """
        types = column_types or {}
        parts = ", ".join(self._col_cast(c, str(types.get(c, ""))) for c in hashed_columns)
        return (
            f"CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONCAT_WS('||', {parts})), 2)"
            f" AS {self.quote_ident(alias)}"
        )

    def render_fingerprint(
        self,
        ref: TableRef,
        *,
        pk: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        aggregate_columns: Sequence[str] = (),
        where: Optional[str] = None,
    ) -> str:
        """As in the base class, but with ``COUNT_BIG`` and a qualified table name.

        ``COUNT(*)`` is an ``INT`` in MSSQL: above 2 147 483 647 rows it **fails**
        with an overflow error rather than returning an imprecise number. That is
        why the predecessors use ``COUNT_BIG``.
        """
        sql = super().render_fingerprint(
            TableRef(name=ref.name, schema=None),
            pk=pk,
            timestamp_column=timestamp_column,
            aggregate_columns=aggregate_columns,
            where=where,
        )
        sql = sql.replace("COUNT(*) AS fp_count", "COUNT_BIG(*) AS fp_count", 1)
        return sql.replace(f"FROM {self.quote_ident(ref.name)}", f"FROM {self.qualified(ref)}", 1)

    # -- reading -----------------------------------------------------------

    def iter_batches(self, engine: Engine, sql: str, batch_size: int) -> Iterator[pd.DataFrame]:
        """Reads the result in batches from one query rather than by paging.

        ``pymssql`` pulls rows from the TDS stream as it goes, so ``chunksize``
        really does stream and needs no counterpart to MySQL's ``buffered=False``.
        **To be confirmed by measurement on the first live run** — MySQL looked
        exactly like this until it turned out that SQLAlchemy 1.4 holds a buffered
        cursor underneath it and 6 M rows cost 4.7 GB (see
        `scripts/bench_hash_memory.py`).
        """
        with engine.connect() as con:
            yield from pd.read_sql(sql, con, chunksize=int(batch_size))


def _schema(ref: TableRef) -> str:
    """The source schema. Empty or missing means ``dbo``.

    The same reading as for PostgreSQL: an empty value in YAML means "not filled
    in", not "a schema whose name is the empty string".
    """
    return ref.schema or DEFAULT_SOURCE_SCHEMA


__all__ = ["MSSQLDialect", "DEFAULT_SOURCE_SCHEMA", "DEFAULT_CHARSET", "BINARY_TYPES"]
