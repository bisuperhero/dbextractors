"""``FirebirdDialect`` — the adapter for Firebird as a source.

In the migration order from ARCHITECTURE.md Firebird comes **last**: it shares
55 % of its lines with MySQL (the lowest of all the pairs, see the similarity
table in ARCHITECTURE.md) and its introspection goes through the ``RDB$`` system
tables rather than ``information_schema``. It also unlocks the one strategy that
had until then only been exercised against a fake source — ``parent_incremental``.

It inherits straight from ``SourceDialect``, not from ``DictTypeMapDialect``. The
reason is ``map_type()``: Firebird types its columns with numeric codes, and the
resulting PostgreSQL type also depends on the **sign** of ``scale``
(``_fb_type_to_pg(fb_type, fb_subtype, fb_scale)`` in the predecessor) — type 7
(SMALLINT) is ``NUMERIC`` when ``fb_scale < 0`` and ``INTEGER`` otherwise. A plain
``source_type -> pg_type`` dict cannot express that.

## ``source_type`` is the type's name, not that numeric code

Introspection **translates the code into a name** (``varchar``, ``timestamp``,
``blob``) and hides the numbers in ``ColumnDef.meta``. It looks like a pointless
intermediate layer, but the rest of the package asks ``source_type`` for a name:
`coerce.DATE_LIKE_TYPES`, `coerce._TIME_TYPES` and
`SourceDialect.text_like_types` all compare strings. If ``"37"`` stayed here, the
silent consequence would be that Firebird's date columns were not treated as
dates. ``map_type()`` therefore reads ``meta``, not ``source_type`` — only there
are all three values that decide the type.

## What Firebird does not have

- **hashing on the source side** — the predecessor has no ``build_hash_expr`` for
  Firebird at all. ``render_hash_expr()`` is not overridden here: it inherits the
  version from ``SourceDialect``, which raises ``UnsupportedFeature``. Hence
  ``hash_in_pandas = True`` — ``row_hash`` is computed by pandas, just as for a
  PostgreSQL source. Today's Firebird extractor does not compute the hash at all
  (the column is not in the target), so this is an **added column**.
- **``partition_by_source``** — and with it the `full_by_source` strategy. A side
  effect: the base ``render_fingerprint`` is never called for Firebird, which is
  just as well — the ``CAST(… AS DECIMAL(38,6))`` in it only works from Firebird 4
  onwards.
- **``_deleted_in_source`` and ``row_hash`` in the target.** Firebird is one of
  the two predecessor files that lack ``_deleted_in_source`` entirely, even though
  the dbt layer depends on it. This package has it, because it is the standard for
  **all** strategies.

Firebird is also addressed by the **path to the database file** rather than by a
name — hence the private ``_normalize_db_path``.
"""

from __future__ import annotations

import logging
import random
import re
import socket
import time
import urllib.parse
from datetime import date, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import pandas as pd
from sqlalchemy import text

from dbextractors.core import secrets

from .base import (
    FEATURE_KEYSET,
    FEATURE_PARENT_INCREMENTAL,
    ColumnDef,
    SizeEstimate,
    SourceDialect,
    SurrogateKey,
    TableRef,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

#: The encoding the connection is opened with when ``SOURCE_DB.charset`` does not
#: name one. Taken verbatim as the predecessor's default (``conn_params``).
DEFAULT_CHARSET = "WIN1250"

#: The epoch that date columns held as a **day count** are counted from in this
#: Firebird schema (a Delphi-era convention). Taken verbatim from the
#: predecessor's ``FIREBIRD_EPOCH``. See ``cutoff_value``.
FIREBIRD_EPOCH = date(1899, 12, 30)

#: The BLOB type code. Split out because the resulting PostgreSQL type is decided
#: by its subtype: ``1`` is text, anything else is binary.
BLOB_TYPE_CODE = 261

#: Firebird's numeric type code -> the name the rest of the package knows it by.
#: The range is exactly the one ``_fb_type_to_pg`` knows in the predecessor — a
#: code outside the list ends up as ``TEXT`` there in exactly the same way.
FB_TYPE_NAMES: dict = {
    7: "smallint",
    8: "integer",
    10: "float",
    12: "date",
    13: "time",
    14: "char",
    16: "bigint",
    23: "boolean",
    27: "double precision",
    35: "timestamp",
    37: "varchar",
    BLOB_TYPE_CODE: "blob",
}


class FirebirdDialect(SourceDialect):
    """Adapter for Firebird. The most distant dialect, migrated last."""

    name: str = "firebird"
    default_port: int = 3050
    #: Exactly as the predecessors spell it.
    sqlalchemy_driver: str = "firebird+fdb"
    quote_char: str = '"'
    #: Without ``hash_diff`` (the source cannot hash) and without
    #: ``partition_by_source``. ``keyset`` is here: the predecessor has it
    #: (`_build_keyset_query`) and drove it from the ``pagination_mode`` key in
    #: exactly the same way as the other dialects. **This package does not read
    #: that key** — the feature flag is what decides, see `full._keyset_usable`
    #: and docs/legacy-compat.md, "Keys that are accepted and do nothing".
    FEATURES: frozenset = frozenset({FEATURE_KEYSET, FEATURE_PARENT_INCREMENTAL})

    #: The cascade taken verbatim from the predecessor (``decode_bytes_columns``).
    #: It happens to be the same as MSSQL's and for the same reason: the source is
    #: Central European, but not everything in it really is cp1250.
    decode_encodings: tuple = ("cp1250", "utf-8", "latin-1")

    #: **Deliberately empty.** The Firebird predecessor has no counterpart to
    #: ``TEXT_LIKE_MYSQL_TYPES``, so forcing a text dtype would change values
    #: against today — and with ``fdb`` it is not a risk anyway: both ``CHAR`` and
    #: ``VARCHAR`` come out of it as ``str``, not as a number.
    text_like_types: frozenset = frozenset()

    #: Firebird has no zero date (unlike MySQL).
    zero_datetime_literal: Optional[str] = None

    #: The source cannot compute the hash, so pandas does — as for PostgreSQL.
    #: See the module docstring; it is an **added column** against today.
    hash_in_pandas: bool = True

    #: For Firebird the encoding is passed in the URL (``?charset=``), not in
    #: ``connect_args`` — hence the emptiness here.
    connect_args: ClassVar[dict] = {}

    # -- connecting --------------------------------------------------------

    def build_conn_str(self, params: dict, host: str, port: int) -> str:
        """Build the SQLAlchemy URL for ``firebird+fdb``.

        Firebird is addressed by the **path to the database file**, not by a name.
        The path is normalised here rather than at the caller: forgetting
        ``_normalize_db_path`` must not be possible, and ``database`` is not used
        for anything else anyway.

        The path goes into the URL **unescaped**, verbatim as in the predecessor.
        That is deliberate: an absolute path starts with a slash, so
        ``…{port}/{path}`` becomes ``…{port}//var/db/x.fdb`` — and the double
        slash is exactly what tells ``fdb`` the path is absolute. Escaping would
        turn it into ``%2F`` and the database would not be found.

        Unlike MSSQL, ``SOURCE_DB.charset`` **is** read — the predecessor really
        does put it into the URL for Firebird, and the ``WIN1250`` default applies
        only when the configuration says nothing.
        """
        user = urllib.parse.quote_plus(str(params.get("user", "")))
        password = urllib.parse.quote_plus(str(params.get("password", "")))
        db_path = self._normalize_db_path(params.get("database") or "")
        charset = params.get("charset") or DEFAULT_CHARSET
        return (
            f"{self.sqlalchemy_driver}://{user}:{password}@{host}:{port}/{db_path}"
            f"?charset={charset}"
        )

    def probe(self, host: str, port: int, timeout: float = 10.0) -> bool:
        """Is the source reachable? TCP only, no login (``is_firebird_accessible``).

        A single variant across the predecessors and a single attempt — of the
        four, only MSSQL retries, and there the reason is documented. It should
        not be carried over here: Firebird is reached through an SSH tunnel, so
        "the port does not answer" more likely means "the tunnel is not up yet",
        and that is another layer's problem.
        """
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError as err:
            _log.debug("🔌 Firebird %s:%s is not reachable: %s", host, port, err)
            return False

    # -- introspection -----------------------------------------------------

    def introspect_columns(self, engine: Engine, ref: TableRef) -> List[ColumnDef]:
        """Columns from ``RDB$RELATION_FIELDS`` / ``RDB$FIELDS``.

        The query is taken verbatim from the predecessor, including three details
        that look decorative and are not:

        - ``TRIM(…)`` — ``RDB$RELATION_NAME`` is a ``CHAR(31)``, so it is compared
          with trailing spaces and without the trim would match nothing.
        - ``RDB$SYSTEM_FLAG`` — without it the result would include system columns,
          which have no business being in the target.
        - **the table name in upper case** — Firebird stores unquoted identifiers
          as ``UPPERCASE``. In the ``SELECT`` the name is quoted exactly as the
          configuration spells it, so if someone writes it in lower case there,
          introspection passes and reading fails. The predecessor behaves the same.

        ``target_name`` is **not** filled in here — that is `core.naming`'s job,
        and the frozen target column names depend on it.
        """
        query = text(
            """
            SELECT
                TRIM(rf.RDB$FIELD_NAME),
                f.RDB$FIELD_TYPE,
                COALESCE(f.RDB$FIELD_SUB_TYPE, 0),
                COALESCE(f.RDB$FIELD_SCALE, 0)
            FROM RDB$RELATION_FIELDS rf
            JOIN RDB$FIELDS f ON f.RDB$FIELD_NAME = rf.RDB$FIELD_SOURCE
            WHERE TRIM(rf.RDB$RELATION_NAME) = :tbl
              AND (rf.RDB$SYSTEM_FLAG IS NULL OR rf.RDB$SYSTEM_FLAG = 0)
            ORDER BY rf.RDB$FIELD_POSITION
            """
        )
        with engine.connect() as con:
            rows = _retry(
                lambda: con.execute(query, {"tbl": ref.name.upper()}).fetchall(),
                label="reading RDB$RELATION_FIELDS",
            )

        if not rows:
            raise RuntimeError(
                f"Table {ref.name!r} does not exist in the source or has no columns "
                f"(it was looked up in upper case, as {ref.name.upper()!r}). "
                "Check source_name in the configuration."
            )

        out: List[ColumnDef] = []
        for ordinal, (name, fb_type, fb_subtype, fb_scale) in enumerate(rows, start=1):
            meta = {
                "fb_type": _as_int(fb_type),
                "fb_subtype": _as_int(fb_subtype),
                "fb_scale": _as_int(fb_scale),
            }
            out.append(
                ColumnDef(
                    name=str(name).strip(),
                    source_type=FB_TYPE_NAMES.get(meta["fb_type"], "unknown"),
                    pg_type=fb_type_to_pg(**meta),
                    # ``RDB$NULL_FLAG`` is not read by the predecessor and is used
                    # nowhere — the target creates every column nullable.
                    nullable=True,
                    ordinal=ordinal,
                    meta=meta,
                )
            )

        self._log_blob_columns(ref, out)
        return out

    @staticmethod
    def _log_blob_columns(ref: TableRef, columns: List[ColumnDef]) -> None:
        """List the BLOB columns, the way the predecessor does.

        It is not a warning about an error — a text BLOB goes to ``TEXT`` and a
        binary one to ``BYTEA``, both without trouble. It is information about
        where the volume of transferred data suddenly balloons.
        """
        bloby = sorted(c.name for c in columns if c.meta.get("fb_type") == BLOB_TYPE_CODE)
        if bloby:
            _log.warning(
                "🗂️ Table %r has BLOB columns (%d): %s", ref.name, len(bloby), ", ".join(bloby)
            )

    def map_type(self, column: ColumnDef) -> str:
        """Firebird's numeric type code (+ subtype, scale) -> PostgreSQL type.

        It overrides the abstract ``SourceDialect.map_type`` instead of inheriting
        from ``DictTypeMapDialect``, because the **sign** of ``scale`` decides as
        well: ``NUMERIC(p,s)`` is stored in Firebird as an integer with a negative
        ``scale``, so type 7/8/16 can just as easily be ``INTEGER`` as ``NUMERIC``.
        If the scale were ignored, every decimal value in the target would be off
        by orders of magnitude — a silent data error, not a crash.

        The values are read from ``column.meta``; when they are not there (the
        column did not come through this dialect's introspection) the result is
        ``TEXT`` — the same fallback the predecessor has for an unknown code. The
        decision itself lives in :func:`fb_type_to_pg`, so that it can be compared
        with the predecessor line by line.
        """
        return fb_type_to_pg(
            fb_type=_as_int(column.meta.get("fb_type")),
            fb_subtype=_as_int(column.meta.get("fb_subtype")),
            fb_scale=_as_int(column.meta.get("fb_scale")),
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
        """``COUNT(*)`` plus the row size **from metadata**, not from a sample.

        Firebird is the only one of the four: the other dialects read 100 rows and
        measure them in pandas, this one sums the declared column lengths from
        ``RDB$FIELDS`` (the predecessor's ``estimate_result_size_mb``). The
        predecessor comments it as "no data page reads", and that makes sense — it
        is the only source reached through a tunnel, so even a hundred rows cost
        something. ``sample_size`` is therefore **unused**; it stays in the
        signature for the shared interface.

        The estimate is coarser as a result (a ``BLOB`` is counted flat at 200 B
        and a ``VARCHAR`` at its declared rather than actual length), but it only
        serves to decide the batch size.

        **A failure is not swallowed**: ``_retry`` lets the exception through after
        three attempts. ``rows=0`` is never returned because asking failed — only
        because the table really holds nothing.
        """
        del sample_size
        where_sql = f" WHERE ({where})" if where else ""
        table = self.quote_ident(ref.name)

        velikost_query = text(
            """
            SELECT SUM(
                CASE f.RDB$FIELD_TYPE
                    WHEN 261 THEN 200
                    WHEN 37  THEN COALESCE(f.RDB$FIELD_LENGTH, 50)
                    WHEN 14  THEN COALESCE(f.RDB$FIELD_LENGTH, 10)
                    ELSE COALESCE(f.RDB$FIELD_LENGTH, 8)
                END
            )
            FROM RDB$RELATION_FIELDS rf
            JOIN RDB$FIELDS f ON f.RDB$FIELD_NAME = rf.RDB$FIELD_SOURCE
            WHERE TRIM(rf.RDB$RELATION_NAME) = :tbl
              AND (rf.RDB$SYSTEM_FLAG IS NULL OR rf.RDB$SYSTEM_FLAG = 0)
            """
        )

        with engine.connect() as con:
            if known_total_rows is not None:
                total_rows = int(known_total_rows)
                method = "known_total_rows"
            else:
                total_rows = int(
                    _retry(
                        lambda: con.execute(
                            text(f"SELECT COUNT(*) FROM {table}{where_sql}")
                        ).scalar(),
                        label="COUNT(*) of the size estimate",
                    )
                    or 0
                )
                method = "count_and_schema"

            if total_rows == 0:
                return SizeEstimate(rows=0, size_mb=0.0, method=method)

            # The 256 B fallback comes from the predecessor: a relation without
            # readable metadata (typically a view) has no column lengths, but the
            # estimate is still worth having.
            avg_row_bytes = int(
                _retry(
                    lambda: con.execute(velikost_query, {"tbl": ref.name.upper()}).scalar(),
                    label="row size estimate from metadata",
                )
                or 256
            )

        size_mb = round((avg_row_bytes * total_rows) / 1024 / 1024, 2)
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
        """The whole SELECT against a Firebird source, identifiers in double quotes.

        ``ref.schema`` is ignored — Firebird has no schemas and the predecessor
        never qualifies the table.
        """
        clause = self.render_select_clause(
            columns, surrogate, column_types=column_types, convert_nchar=convert_nchar
        )
        sql = f"SELECT {clause} FROM {self.quote_ident(ref.name)}"
        if where:
            sql += f" WHERE ({where})"
        if order_by:
            sql += " ORDER BY " + ", ".join(self.quote_ident(c) for c in order_by)
        return sql

    def source_ident(self, name: str) -> str:
        """A name from the configuration or a default, put into Firebird's shape.

        Firebird keeps unquoted names in **upper case**, so `"id"` is a different —
        and non-existent — column from `"ID"`. Names from introspection are already
        upper case; this is for the ones that come from elsewhere — the three
        names in `parent_incremental._source_where`: ``incremental_parent_table``,
        ``incremental_parent_key_column`` (default `parent_id`) and
        ``incremental_parent_id_column`` (default `id`).

        Only a name written entirely in lower case is touched, the same rule
        SQLAlchemy's ``denormalize_name`` uses. A column created in quotes as
        ``"mixedCase"`` is left alone — rewriting it would be wrong.

        Documented in production: ``SELECT "id" FROM "SOME_TABLE"`` ended with
        ``SQLCODE -206, Column unknown - id``.
        """
        cisty = name.strip()
        return self.quote_ident(cisty.upper() if cisty.islower() else cisty)

    def cutoff_value(self, cutoff: Any) -> Any:
        """The cutoff as a **day count since 30 December 1899**, not as a date.

        The date columns the incremental window is computed over here are not
        ``DATE`` — they are integers counting days since that epoch (a Delphi-era
        inheritance). The predecessor computes it in the shared
        ``_compute_incremental_cutoff`` and uses ``cutoff_numeric`` exclusively:
        **every** Firebird path (``incremental`` and ``incremental_by_parent``)
        compares a number with a number.

        The conversion deliberately lives in the dialect rather than in the
        strategy — otherwise code that has nothing to do with Firebird would know
        about the Firebird epoch as well.

        The limitation this inherits: if some table had its window over a genuine
        ``DATE`` column, this comparison would be wrong. No such table exists today
        and the predecessor would fail on it just the same; if one appeared, that
        calls for a new key in ``LOAD_SETTINGS``, not for guessing from the type.
        """
        # The order of the branches is not arbitrary: ``datetime`` is a subclass of
        # ``date``, so the other way round it would fall into a ``datetime - date``
        # subtraction and fail on the types.
        if isinstance(cutoff, datetime):
            cutoff = cutoff.date()
        if isinstance(cutoff, date):
            return (cutoff - FIREBIRD_EPOCH).days
        return cutoff

    # -- reading -----------------------------------------------------------

    def iter_batches(self, engine: Engine, sql: str, batch_size: int) -> Iterator[pd.DataFrame]:
        """Reads the result in batches from one query rather than by paging.

        The predecessor pages through ``FIRST/SKIP``, which on a large table means
        quadratic work on the source side (``SKIP`` has to walk a growing prefix
        every time). One cursor with ``chunksize`` avoids that and gives a
        consistent view as well, so no duplicate can appear between batches.

        **Not verified against a live source** — this session had no access to a
        Firebird instance. MySQL looked exactly like this until it turned out that
        SQLAlchemy 1.4 holds a buffered cursor underneath it and 6 M rows cost
        4.7 GB (see `scripts/bench_hash_memory.py`). With ``fdb`` that is not a
        risk in itself (it fetches by ``arraysize``), but it has to be measured.
        """
        with engine.connect() as con:
            for batch in pd.read_sql(sql, con, chunksize=int(batch_size)):
                yield self._denormalize_columns(batch, engine)

    @staticmethod
    def _denormalize_columns(batch: pd.DataFrame, engine: Engine) -> pd.DataFrame:
        """Give the column names back the letter case the source uses.

        SQLAlchemy sets ``requires_name_normalize`` for Firebird, so a name stored
        in the database entirely in upper case reaches the caller in lower case.
        Introspection from ``RDB$RELATION_FIELDS`` returns the original upper case
        — and ``ctx.name_map`` is built on those.

        Without lining the two up, the map misses, the rename to target names
        silently does not happen, and the failure comes two steps later::

            KeyError: "['_name'] not in index"

        Documented in production: introspection returned ``['ID', 'NAME', …]`` and
        the driver ``['id', 'name', …]``.

        It uses the ``denormalize_name`` of the very dialect that did the
        normalisation — its own inverse, so the two cannot drift apart. The
        hand-written rule is only a fallback for a dialect that does not have one.
        """
        convert = getattr(getattr(engine, "dialect", None), "denormalize_name", None)
        if not callable(convert):
            # The same rule SQLAlchemy uses: only names that are entirely lower
            # case are touched. A column created in quotes as ``"mixedCase"`` is
            # returned as it is, and rewriting it would be wrong.
            def convert(name: Any) -> Any:
                return name.upper() if isinstance(name, str) and name.islower() else name

        changes = {}
        for column in batch.columns:
            new = convert(column)
            if new and new != column:
                changes[column] = new
        return batch.rename(columns=changes) if changes else batch

    # -- other -------------------------------------------------------------

    def _normalize_db_path(self, database: str) -> str:
        """Collapse repeated backslashes back into one.

        Taken verbatim from the predecessor's ``_normalize_db_path``. The
        configuration holds a Windows path with single backslashes, but passing
        the dict between Mage blocks occasionally doubles them
        (``D:\\\\data\\\\...``). ``fdb`` needs the original form, and a disk path
        never legitimately contains a doubled separator, so the repair is safe.
        """
        if not isinstance(database, str):
            return database
        opraveno = re.sub(r"\\{2,}", r"\\", database)
        if opraveno != database:
            _log.warning(
                "🩹 Collapsed doubled backslashes in the DB path: %r -> %r", database, opraveno
            )
        return opraveno


# --- Helpers ----------------------------------------------------------------


def _as_int(value: Any) -> int:
    """A Firebird code as an ``int``. ``None`` is 0, same as in the predecessor."""
    return int(value) if value is not None else 0


def fb_type_to_pg(fb_type: Any = 0, fb_subtype: Any = 0, fb_scale: Any = 0) -> str:
    """Firebird type code -> PostgreSQL type. Verbatim the predecessor's ``_fb_type_to_pg``.

    A standalone function rather than a method precisely so that it can be
    compared with the predecessor line by line — this is the only place in the
    package where a type is decided not by a name but by a triple of numbers.
    """
    fb_type = _as_int(fb_type)
    fb_subtype = _as_int(fb_subtype)
    fb_scale = _as_int(fb_scale)

    if fb_type == 7:  # SMALLINT
        return "NUMERIC" if fb_scale < 0 else "INTEGER"
    if fb_type == 8:  # INTEGER
        return "NUMERIC" if fb_scale < 0 else "INTEGER"
    if fb_type == 16:  # INT64 / NUMERIC / DECIMAL
        if fb_subtype in (1, 2):
            return "NUMERIC"
        return "NUMERIC" if fb_scale < 0 else "BIGINT"
    if fb_type == 10:  # FLOAT
        return "REAL"
    if fb_type == 27:  # DOUBLE PRECISION
        return "DOUBLE PRECISION"
    if fb_type == 14:  # CHAR
        return "TEXT"
    if fb_type == 37:  # VARCHAR
        return "VARCHAR"
    if fb_type == BLOB_TYPE_CODE:
        return "TEXT" if fb_subtype == 1 else "BYTEA"
    if fb_type == 35:  # TIMESTAMP
        return "TIMESTAMP"
    if fb_type == 12:  # DATE
        return "DATE"
    if fb_type == 13:  # TIME
        return "TIME"
    if fb_type == 23:  # BOOLEAN (FB 3.0+)
        return "BOOLEAN"
    return "TEXT"


def _retry(
    fn: Callable[[], _T],
    *,
    label: str,
    attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.25,
) -> _T:
    """Repeat an operation with exponential backoff. Verbatim the predecessor's ``with_retry``.

    Of the four dialects only Firebird has this, and with reason: it is reached
    through an SSH tunnel, so a "connection reset" in the middle of introspection
    is not a theoretical possibility. Once the attempts are exhausted the
    exception is **let through** — a partial result is never carried on with.

    The jitter comes from the predecessor: the orchestrator runs three pipelines
    at once, so without it all three would try to connect at the same instant.
    """
    for attempt in range(1, int(attempts) + 1):
        try:
            return fn()
        except Exception as err:
            if attempt >= attempts:
                _log.error(
                    "🛑 %s failed even after %d attempts: %s", label, attempts, secrets.redact(err)
                )
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 1 + jitter * (random.random() * 2 - 1)
            _log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1f s.",
                label,
                attempt,
                attempts,
                secrets.redact(err),
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "FirebirdDialect",
    "DEFAULT_CHARSET",
    "FIREBIRD_EPOCH",
    "FB_TYPE_NAMES",
    "fb_type_to_pg",
]
