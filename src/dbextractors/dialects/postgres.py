"""``PostgresDialect`` — the adapter for PostgreSQL as a source.

PostgreSQL comes **first** in the migration order: the PostgreSQL source has
only 21 tables, which makes it the smallest real dialect, and it fixes a known
bug on the way (``_deleted_in_source`` is missing in 14 of the 15 predecessor
files even though the dbt layer depends on it).

Not to be confused with the target side (``core/target_pg.py``) — this module is
about PostgreSQL as a *source* of data, not as the place data is written to. The
target is always PostgreSQL, no matter what the data is read from.

Inherits from ``DictTypeMapDialect`` — the type map is a declarative dict and is
often close to the identity (source and target are the same database system).

## For PostgreSQL the hash is computed in pandas, not in SQL

This is the only difference that reaches into the core, and **it is not a
choice** — it is a constraint imposed by what is already in the target. The
PostgreSQL predecessor has no ``build_hash_expr`` at all: it computes the hash
with ``hashlib.sha256`` over the concatenated values, and that digest is what is
stored in the target for every PostgreSQL table using the hash strategy.

If the dialect started computing the hash in SQL — and PostgreSQL is perfectly
able to — the digest would come out **different**, and the first run after the
migration would evaluate the whole table as changed for all ~58 PostgreSQL
pipelines. This was measured: the pandas path and the SQL path disagree (4 of 51
tables match, see `scripts/verify_hash_parity.py`).

Hence ``hash_in_pandas = True`` and ``render_hash_expr`` raising
``UnsupportedFeature``. The strategy picks its path from that — and above all
uses the **same** path for seeding and for the diff.

## ``source_schema`` and its two different readings

Two of the predecessors read it differently, and it is **not** the same thing:

- variant B: ``table_config.get('source_schema', 'public')``
- variant A: ``table_config.get('source_schema') or 'public'``

They diverge when the key **is** present in the configuration but empty: variant
B takes the empty string (and a query against ``""."table"`` then fails), while
variant A falls back to ``public``. Variant A wins — an empty value in YAML
almost certainly means "not filled in", not "a schema whose name is the empty
string". `core.config` settles it; by the time a value arrives here it is already
decided, and an empty ``ref.schema`` means ``public``.
"""

from __future__ import annotations

import logging
import socket
import urllib.parse
from typing import TYPE_CHECKING, ClassVar, Iterator, List, Optional, Sequence

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
    UnsupportedFeature,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine

_log = logging.getLogger(__name__)

#: The schema searched when the configuration does not name one.
DEFAULT_SOURCE_SCHEMA = "public"


class PostgresDialect(DictTypeMapDialect):
    """Adapter for PostgreSQL as a source. The smallest real dialect, migrated first."""

    name: str = "postgres"
    default_port: int = 5432
    #: Exactly as the predecessors spell it.
    sqlalchemy_driver: str = "postgresql+psycopg2"
    quote_char: str = '"'
    FEATURES: frozenset = frozenset({FEATURE_KEYSET, FEATURE_HASH_DIFF})

    #: The hash is computed by pandas, not by the source — see the module
    #: docstring. It is not a choice, it is what is stored in the target.
    hash_in_pandas: bool = True

    #: PostgreSQL returns text as `str`, not as bytes — the decoding cascade
    #: never gets here.
    decode_encodings: tuple = ("utf-8",)

    #: Types that have to be read as text even though they look numeric.
    text_like_types: frozenset = frozenset(
        {"character varying", "character", "text", "citext", "uuid"}
    )

    #: Source type (lowercase) -> PostgreSQL type. Taken verbatim from the
    #: predecessor's ``PG_TO_PG_TYPES``. It is not the identity: ``smallint`` is
    #: promoted to ``INTEGER`` and the geometric and network types fall back to
    #: ``TEXT``. The single deviation is ``money`` — the reason is in the comment
    #: next to it.
    type_map: ClassVar[dict] = {
        "smallint": "INTEGER",
        "integer": "INTEGER",
        "int": "INTEGER",
        "bigint": "BIGINT",
        "boolean": "BOOLEAN",
        "real": "REAL",
        "double precision": "DOUBLE PRECISION",
        "numeric": "NUMERIC",
        "decimal": "NUMERIC",
        # **A deviation from the predecessor**, which maps `money` to NUMERIC.
        # psycopg2 returns `money` as a *localised string* — with the locale's
        # currency symbol and decimal separator — so a `COPY` into NUMERIC would
        # fail with "invalid input syntax". The
        # predecessor never meets this, because a full load does not use the type
        # map at all and the column ends up as text — which is also what is in
        # the target today.
        "money": "TEXT",
        "character varying": "VARCHAR",
        "character": "TEXT",
        "text": "TEXT",
        "citext": "TEXT",
        "uuid": "TEXT",
        "date": "DATE",
        "timestamp without time zone": "TIMESTAMP",
        "timestamp with time zone": "TIMESTAMP",
        "time without time zone": "TIME",
        "time with time zone": "TIME",
        "interval": "TEXT",
        "json": "JSONB",
        "jsonb": "JSONB",
        "bytea": "BYTEA",
        "bit": "TEXT",
        "bit varying": "TEXT",
        "xml": "TEXT",
        "inet": "TEXT",
        "cidr": "TEXT",
        "macaddr": "TEXT",
        "macaddr8": "TEXT",
        "tsvector": "TEXT",
        "tsquery": "TEXT",
        "point": "TEXT",
        "line": "TEXT",
        "lseg": "TEXT",
        "box": "TEXT",
        "path": "TEXT",
        "polygon": "TEXT",
        "circle": "TEXT",
        "array": "TEXT",
        "user-defined": "TEXT",
    }

    # -- connecting --------------------------------------------------------

    def build_conn_str(self, params: dict, host: str, port: int) -> str:
        """The SQLAlchemy URL. The password **must** be escaped or an `@` in it breaks the URL.

        ``charset`` is not passed for PostgreSQL: the database decides the
        encoding and psycopg2 follows it, unlike MySQL where it is a connection
        parameter.
        """
        user = urllib.parse.quote_plus(str(params.get("user", "")))
        password = urllib.parse.quote_plus(str(params.get("password", "")))
        database = params.get("database", "")
        return f"{self.sqlalchemy_driver}://{user}:{password}@{host}:{port}/{database}"

    def probe(self, host: str, port: int, timeout: float = 10.0) -> bool:
        """Is the source reachable? TCP only, no login.

        This decides between a direct connection and a tunnel, and for that it is
        enough to know whether the port is open.
        """
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError as err:
            _log.debug("PostgreSQL %s:%s is not reachable: %s", host, port, err)
            return False

    # -- introspection -----------------------------------------------------

    def introspect_columns(self, engine: Engine, ref: TableRef) -> List[ColumnDef]:
        """Columns from ``information_schema.columns``, in the source's own order.

        The order is part of the contract — the golden test compares it right
        after the row count. ``target_name`` is **not** filled in here: that is
        `core.naming`'s job.
        """
        query = text(
            "SELECT column_name, data_type, is_nullable, ordinal_position "
            "FROM information_schema.columns "
            "WHERE table_schema = :sch AND table_name = :tbl "
            "ORDER BY ordinal_position"
        )
        with engine.connect() as con:
            rows = con.execute(query, {"sch": _schema(ref), "tbl": ref.name}).fetchall()

        if not rows:
            raise RuntimeError(
                f"Table {_schema(ref)}.{ref.name!r} does not exist in the source or has "
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

        **A deviation from variant B, which stays silent.** Its version returns
        ``(0, 0)`` when the sample is empty **or** when ``COUNT(*)`` fails — so an
        unreachable source passes itself off as an empty table, which combined
        with ``empty_rows_ok`` overwrites the target with nothing. Variant A
        wins: the exception is let through, because a failed estimate has to fail
        loudly.
        """
        where_sql = f" WHERE ({where})" if where else ""
        table = self.qualified(ref)

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

    def qualified(self, ref: TableRef) -> str:
        """``"schema"."table"``. An empty schema means ``public``."""
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

    def render_hash_expr(
        self,
        hashed_columns: Sequence[str],
        alias: str,
        column_types: Optional[dict] = None,
    ) -> str:
        """**Does not exist.** For PostgreSQL the hash is computed in pandas.

        See the module docstring. PostgreSQL could do it in SQL, but the result
        would differ from the digests currently in the target, and the first run
        after the migration would evaluate the whole table as changed for some
        58 pipelines.
        """
        raise UnsupportedFeature(
            "postgres computes row_hash in pandas, not on the source side — otherwise the "
            "digest would differ from what is stored in the target (see dialects/postgres.py)."
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
        """As in the base class, only with a qualified table name."""
        sql = super().render_fingerprint(
            TableRef(name=ref.name, schema=None),
            pk=pk,
            timestamp_column=timestamp_column,
            aggregate_columns=aggregate_columns,
            where=where,
        )
        return sql.replace(f"FROM {self.quote_ident(ref.name)}", f"FROM {self.qualified(ref)}", 1)

    # -- reading -----------------------------------------------------------

    def iter_batches(self, engine: Engine, sql: str, batch_size: int) -> Iterator[pd.DataFrame]:
        """Reads through a **named cursor** so the result is not held in RAM.

        Without it psycopg2 pulls the whole result to the client before pandas
        sees the first batch — the same trap as with MySQL, where 6 M rows cost
        4.7 GB (see `scripts/bench_hash_memory.py`). psycopg2 only streams
        through a server-side cursor, which SQLAlchemy 1.4 switches on with
        `stream_results`.
        """
        with engine.connect().execution_options(
            stream_results=True, max_row_buffer=int(batch_size)
        ) as con:
            yield from pd.read_sql(sql, con, chunksize=int(batch_size))


def _schema(ref: TableRef) -> str:
    """The source schema. Empty or missing means ``public``.

    Variant A's ``or`` wins over variant B's ``get(…, 'public')`` — an empty
    value in YAML means "not filled in", not "a schema whose name is the empty
    string".
    """
    return ref.schema or DEFAULT_SOURCE_SCHEMA


__all__ = ["PostgresDialect", "DEFAULT_SOURCE_SCHEMA"]
