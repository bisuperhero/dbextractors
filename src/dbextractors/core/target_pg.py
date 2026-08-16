"""Writing into the target PostgreSQL. The largest module here, and 100% shared.

**The target side is always PostgreSQL** (ARCHITECTURE.md). It is at the same
time the one large block of code that is identical across all four dialects,
which is why it fits here without adapters. ``target_pg`` knows nothing about
the source: it is handed a DataFrame and column metadata.

The module works over a **bare psycopg2 connection**, not over
``mage_ai.io.postgres``. The predecessor goes through ``loader.export()``, which
does three things at once (DDL, conversion, writing) and each of them has to be
steered separately here. Whatever had to be replicated from it carries a pointer
to its origin in Mage 0.9.79.

## What turned out differently from the original brief

The brief lists "writing through ``loader.export()`` in batches (INSERT)" as a
banned practice and blames it for the full load running at 4 430 rows/s. **The
full load sends no INSERTs at all.**
``mage_ai.io.postgres.Postgres.upload_dataframe`` has two branches and only uses
INSERT (``cursor.executemany`` over ``df.iterrows()``) when it is given
``unique_constraints`` **and** ``unique_conflict_method``. The full load passes
neither, so it goes through ``COPY … FROM STDIN``. Only the incremental and
hash-diff paths pass them — 18 call sites in the predecessor, and there the
criticism holds literally.

Practical consequence for this module: ``COPY`` is not a speed-up over the
predecessor, it is **keeping** its write path without the rest of ``export()``.
The speed-up lies elsewhere — see ``scripts/bench_write_path.py``.

## The COPY format is inherited, not chosen

``upload_dataframe`` writes like this (mage_ai/io/postgres.py)::

    df.to_csv(buffer, header=False, index=False, na_rep='')
    COPY … FROM STDIN (FORMAT csv, DELIMITER ',', NULL '', FORCE_NULL(<all columns>))

``NULL ''`` together with ``FORCE_NULL`` on **every** column means that an empty
string in the DataFrame ends up as ``NULL`` in the target. That loses
information, but it is the state of all ~670 tables today; changing it would show
up in the data, not only in the schema. `EMPTY_STRING_AS_NULL` governs it.

The single deviation from Mage is **quoting: ``QUOTE_ALL`` instead of
``QUOTE_MINIMAL``**. With the default quoting, pandas leaves a value holding a
lone ``\r`` unquoted (``lineterminator`` only carries ``\n``) and ``COPY``
rejects it with "unquoted carriage return found in data". Mage suffers from the
same thing; all that protects it is `coerce.normalize_carriage_returns` one step
higher. Verified against PG 17.9 that wherever ``QUOTE_MINIMAL`` passes,
``QUOTE_ALL`` stores the very same value in the database; the payload is ~21%
larger and ``to_csv`` is no slower.

## Columns every target table must have

The predecessor is uneven here: ``_deleted_in_source`` is fully maintained by
only 8 of its 15 files, another 5 merely declare it, and Firebird has none at
all. In this package it is the standard for every strategy.

The types are read off the target, not chosen: ``_timestamp`` is ``TEXT`` in all
109 tables that carry it, because the predecessor writes
``datetime.now(timezone.utc).isoformat()`` into it.
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

from dbextractors.core import coerce

if TYPE_CHECKING:  # pragma: no cover
    from dbextractors.core.strategies.base import TargetRef

_log = logging.getLogger(__name__)


class TargetError(RuntimeError):
    """A target-side error that is pointless to retry without changing the request."""


#: Columns the package adds to every target table.
#:
#: The types are not a choice — they are read off the target. ``_timestamp`` is
#: ``TEXT`` because the predecessor writes an ISO 8601 string into it; in all 109
#: tables that have it, the type is ``text``. ``_deleted_in_source`` is a nullable
#: ``boolean`` without a default in today's tables (Mage created it from the
#: dataframe), but the predecessor's ALTER branch creates it as ``NOT NULL DEFAULT
#: false`` — and that is our choice too: it is stricter, breaks nothing, and 133
#: dbt models are spared from handling NULL because of it.
MANAGED_COLUMNS: dict = {
    "_timestamp": "TEXT",
    "_deleted_in_source": "BOOLEAN NOT NULL DEFAULT false",
}

#: Type of the hash column. In the target it is ``text`` in all 22 tables with one.
ROW_HASH_TYPE = "TEXT"

#: Prefix of the columns Mage adds to the dataframe. The predecessor drops them by
#: prefix rather than by an explicit list (``c.startswith('_mage_')``), so do we.
MAGE_COLUMN_PREFIX = "_mage_"

#: An empty string lands in the target as ``NULL``. Inherited from ``FORCE_NULL``
#: in ``mage_ai.io.postgres.upload_dataframe`` — see the module docstring.
EMPTY_STRING_AS_NULL = True

#: How many rows get serialised into the CSV buffer at a time. This is not the
#: batch size — a batch may hold hundreds of thousands of rows; this keeps the
#: memory peak independent of it. The orchestrator runs 3 pipelines side by side,
#: so the peak is tripled.
COPY_CHUNK_ROWS = 10_000

#: Prefix of shadow tables. It has to be recognisable — leftovers from crashed runs
#: are cleaned up by it.
SHADOW_PREFIX = "dbx_shadow_"

#: The column that tells which source database a row came from. It is added **only**
#: for multi-source tables (today MSSQL) — adding it everywhere would change the DDL
#: of all ~670 tables, and the frozen target column names exist precisely to prevent
#: that. Agreed.
SOURCE_COLUMN = "_source"
SOURCE_COLUMN_TYPE = "TEXT"

#: Where the package keeps source fingerprints. The predecessor stores them through
#: a function living in the client repository, so the behaviour differs per repo.
FINGERPRINT_TABLE = "dbx_source_fingerprints"

#: Maximum identifier length in PostgreSQL. Anything longer is silently truncated,
#: which is dangerous for shadow tables: two different tables would get one name.
MAX_IDENTIFIER_LENGTH = 63


# --- Identifiers ------------------------------------------------------------


def quote_ident(name: str) -> str:
    """Quote an identifier for PostgreSQL.

    Taken verbatim from the predecessor's PostgreSQL loader — the only variant
    across all of it. Note that this is the **target** side; the source side has
    ``SourceDialect.quote_ident``, where the quoting character differs.
    """
    return '"' + name.replace('"', '""') + '"'


def qualify(target: TargetRef) -> str:
    """``"schema"."table"`` — the fully qualified, quoted name."""
    return f"{quote_ident(target.schema)}.{quote_ident(target.table)}"


def _column_list(columns: Sequence[str]) -> str:
    return ", ".join(quote_ident(c) for c in columns)


# --- Target table schema -----------------------------------------------------


@dataclass
class PgSchema:
    """The real schema of the target table, as ``information_schema`` reports it.

    ``exists=False`` means "the table is not there yet" — a legitimate state
    before the first run, not an error. The predecessor does not tell the two
    apart: it catches `Exception`, logs a warning and returns empty lists, so an
    unreachable database looks exactly like a missing table.
    """

    exists: bool = False
    columns: List[str] = field(default_factory=list)
    types: dict = field(default_factory=dict)
    integer_columns: Set[str] = field(default_factory=set)


#: Map from ``information_schema.data_type`` to the type that overwrites
#: ``overwrite_types``. The order of the tests carries meaning, as in the
#: predecessor: ``bigint`` has to be tested before ``int``, because
#: ``'int' in 'bigint'`` is true.
_DB_TYPE_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("bigint",), "BIGINT"),
    (("int", "serial"), "INTEGER"),
    (("bool",), "BOOLEAN"),
    (("json",), "JSONB"),
    (("timestamp",), "TIMESTAMP"),
    (("date",), "DATE"),
    (("numeric", "decimal"), "NUMERIC"),
    (("float", "double", "real"), "DOUBLE PRECISION"),
)

_INTEGER_DB_TYPES = frozenset({"integer", "bigint", "smallint"})
_INTEGER_TARGET_TYPES = frozenset({"INTEGER", "BIGINT", "SMALLINT"})


def load_pg_schema(
    conn: Any,
    target: TargetRef,
    overwrite_types: dict,
    integer_columns_mapping: dict,
) -> PgSchema:
    """Load the real target schema and **overwrite the types from it**.

    Both maps are mutated in place, just as in the predecessor's
    ``_load_pg_schema``. The point is that the target may already have a column
    promoted to `BIGINT`, and the guess taken from the source catalogue would pull
    it back down to `INTEGER`.

    **Deviation:** the predecessor swallows any exception and returns an empty
    list. Here a missing table is recognised by the query itself (`exists=False`)
    and an error is allowed to propagate — a failure must be loud. A silent empty
    list leads to `align_df_columns_to_db` being skipped and the batch being
    written with the columns in the wrong order.
    """
    schema = PgSchema()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (target.schema, target.table),
        )
        rows = cur.fetchall()

    if not rows:
        _log.debug("🔍 Target table %s does not exist yet.", qualify(target))
        return schema

    schema.exists = True
    schema.columns = [r[0] for r in rows]
    schema.types = {r[0]: str(r[1]).lower() for r in rows}
    schema.integer_columns = {c for c, t in schema.types.items() if t in _INTEGER_DB_TYPES}

    for col_name, pg_type in schema.types.items():
        for needles, new_type in _DB_TYPE_RULES:
            if any(needle in pg_type for needle in needles):
                overwrite_types[col_name] = new_type
                break

    for col, typ in list(overwrite_types.items()):
        if typ in _INTEGER_TARGET_TYPES:
            integer_columns_mapping[col] = typ

    return schema


def existing_column_order(conn: Any, target: TargetRef) -> List[str]:
    """Target column names in the order in which they really sit in the table.

    Empty list when the table does not exist. Its purpose is that dbextractors
    **never reorders** an existing target table: the predecessor builds the order
    of the managed columns in six different ways, so the pipeline name gives no
    clue which of them a given table got. Asking the target is the only reliable
    way.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (target.schema, target.table),
        )
        return [r[0] for r in cur.fetchall()]


def table_exists(conn: Any, target: TargetRef) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{target.schema}.{target.table}",))
        return cur.fetchone()[0] is not None


def ensure_schema(conn: Any, schema: str) -> None:
    """``CREATE SCHEMA IF NOT EXISTS`` — ``export()`` does this on every batch."""
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")


def ensure_generated_columns(
    conn: Any,
    target: TargetRef,
    *,
    include_row_hash: bool = True,
    hash_column: str = "row_hash",
) -> List[str]:
    """Add ``_timestamp``, ``_deleted_in_source`` and optionally the hash column.

    Returns the list of columns actually added (empty when nothing was missing).

    ``include_row_hash`` exists in only one of the predecessor's variants; the
    other 7 files do not have it. Rather than keeping a forked variant, it is
    taken over as an option with the safe default ``True``.

    **Deviation:** the predecessor wraps the whole block in ``except Exception``
    and on failure only logs at DEBUG. A missing table is a legitimate reason for
    the ALTER not to happen — an unreachable database or missing privileges is
    not. The two are told apart here.
    """
    if not table_exists(conn, target):
        _log.debug("🔍 %s does not exist; DDL will add the generated columns.", qualify(target))
        return []

    wanted: List[Tuple[str, str]] = list(MANAGED_COLUMNS.items())
    if include_row_hash:
        wanted.insert(0, (hash_column, ROW_HASH_TYPE))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (target.schema, target.table),
        )
        existing = {r[0] for r in cur.fetchall()}

        missing = [(name, ddl) for name, ddl in wanted if name not in existing]
        if not missing:
            return []

        clauses = ", ".join(f"ADD COLUMN {quote_ident(name)} {ddl}" for name, ddl in missing)
        cur.execute(f"ALTER TABLE {qualify(target)} {clauses}")

    added = [name for name, _ in missing]
    _log.info("🧩 Columns added to %s: %s", qualify(target), ", ".join(added))
    return added


def columns_missing_in_target(conn: Any, target: TargetRef, columns: Sequence[str]) -> List[str]:
    """Columns the source sends and the target table does not have.

    Keeps the input order. When the target does not exist, returns an empty list —
    the table is yet to be created and will hold exactly what we hand it.

    The comparison is **case-insensitive**. A column that differs from the target
    one only in case is not missing; treating it as missing would throw data away
    over cosmetics.
    """
    existing = {c.lower() for c in existing_column_order(conn, target)}
    if not existing:
        return []
    return [c for c in columns if c.lower() not in existing]


def adopt_new_source_columns(
    conn: Any,
    target: TargetRef,
    columns: Sequence[str],
    *,
    overwrite_types: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Add columns the source sends and the target lacks. Returns the added ones.

    Called **only from the full load** (and `full_by_source`), where the table is
    rewritten wholesale anyway, so widening its shape costs nothing. The other
    strategies drop the extra column — see `columns_missing_in_target`.

    This has to happen **before the shadow table**, not at swap time: the shadow
    is created through ``LIKE <target>``, so a column added to the target
    beforehand is inherited by it and `swap_shadow_table` then carries it over.
    Same ordering as with `ensure_generated_columns`.
    """
    missing = columns_missing_in_target(conn, target, columns)
    if not missing:
        return []

    overwrite_types = overwrite_types or {}
    parts = []
    for name in missing:
        if name in overwrite_types:
            pg_type = overwrite_types[name]
        elif df is not None and name in df.columns:
            pg_type = infer_pg_type(df[name], infer_dtype(df[name], skipna=True))
        else:
            raise TargetError(
                f"Target {qualify(target)} has no column {name!r} and its type cannot be "
                "derived — it is neither in the batch nor in overwrite_types."
            )
        parts.append(f"ADD COLUMN {quote_ident(name)} {pg_type}")

    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {qualify(target)} {', '.join(parts)}")

    _log.info("📝 Columns adopted from the source into %s: %s", qualify(target), ", ".join(missing))
    return missing


# --- DDL --------------------------------------------------------------------
# A replica of what `mage_ai.io.postgres.Postgres` does inside `export()`.
# Without it there would be no way to write through COPY, because somebody has
# to create the table.


def infer_pg_type(column: pd.Series, dtype: str) -> str:
    """Derive a PostgreSQL column type from a pandas column.

    A replica of ``mage_ai.io.postgres.Postgres.get_type`` (Mage 0.9.79). It
    returns the types in lower case exactly as that one does — these are names
    that end up in ``CREATE TABLE``, so the frozen target naming applies here too.

    One thing is not preserved: on an unknown dtype Mage **prints to stdout**
    ``Invalid datatype provided`` and returns ``text``. Here a warning is logged
    instead; a print from a library never shows up in the Mage UI, so today that
    information is lost entirely.
    """
    if dtype in ("mixed", "unknown-array", "complex"):
        return _infer_object_type(column, dtype)
    if dtype in ("datetime", "datetime64"):
        try:
            if column.dt.tz:
                return "timestamptz"
        except AttributeError:
            pass
        return "timestamp"
    if dtype == "time":
        try:
            if column.dt.tz:
                return "timetz"
        except AttributeError:
            pass
        return "time"
    if dtype == "date":
        return "date"
    if dtype in ("string", "categorical"):
        return "text"
    if dtype == "bytes":
        return "bytea"
    if dtype in ("floating", "decimal", "mixed-integer-float"):
        return "double precision"
    if dtype in ("integer", "int64"):
        return _infer_integer_width(column)
    if dtype == "boolean":
        return "boolean"
    if dtype in ("timedelta", "timedelta64", "period"):
        return "bigint"
    if dtype == "empty":
        return "text"
    if dtype == "object":
        return "JSONB"

    _log.warning("⚠️ Unknown pandas dtype %r on column %r — using text.", dtype, column.name)
    return "text"


def _infer_integer_width(column: pd.Series) -> str:
    """``smallint`` / ``integer`` / ``bigint`` by the value range in the batch.

    Careful, this is a genuine source of risk: the decision is made from **one
    batch**, so a table created from the first 50 000 rows may get `smallint` and
    fail on the millionth. The predecessor handles it elsewhere — after the first
    batch it promotes to `BIGINT` based on `update_stats` (``INT_THRESHOLD``).
    That is kept identical so that `overwrite_types` remains the single place
    where a type is overridden.
    """
    max_int, min_int = column.max(), column.min()
    if np.int16(max_int) == max_int and np.int16(min_int) == min_int:
        return "smallint"
    if np.int32(max_int) == max_int and np.int32(min_int) == min_int:
        return "integer"
    return "bigint"


def _infer_object_type(column: pd.Series, dtype: str) -> str:
    values = column[column.notnull()].values
    if len(values) == 0:
        raise TargetError(
            f"Column {column.name!r} has dtype {dtype!r} and nothing but empty values — "
            "the target type cannot be derived from it. Declare it in LOAD_SETTINGS."
        )

    non_empty = [v for v in values if not isinstance(v, list) or v]
    if not non_empty:
        return "JSONB"

    value = non_empty[0]
    if not isinstance(value, list):
        return "JSONB"
    if not value:
        return "text[]"
    item = value[0]
    if isinstance(item, dict):
        return "JSONB"
    item_series = pd.Series(data=item)
    item_dtype = infer_dtype(item_series, skipna=True)
    if item_dtype == "object":
        return "text[]"
    return f"{infer_pg_type(item_series, item_dtype)}[]"


def build_create_table(
    df: pd.DataFrame,
    target: TargetRef,
    *,
    overwrite_types: Optional[dict] = None,
    columns: Optional[Sequence[str]] = None,
) -> str:
    """Assemble the ``CREATE TABLE`` statement for the target table.

    A replica of ``gen_table_creation_query`` (Mage 0.9.79) with one difference:
    column names are **not touched**. Mage runs ``clean_name`` over them again,
    but ``export()`` already renamed them earlier, so the second call is
    idempotent. Here the names arrive finished from `core.naming` and further
    sanitising could only change them silently.
    """
    overwrite_types = overwrite_types or {}
    names = list(columns) if columns is not None else list(df.columns)

    parts = []
    for name in names:
        if name in overwrite_types:
            pg_type = overwrite_types[name]
        elif name in df.columns:
            pg_type = infer_pg_type(df[name], infer_dtype(df[name], skipna=True))
        else:
            raise TargetError(
                f"Column {name!r} is neither in the batch nor in overwrite_types — "
                "the target type cannot be derived from it."
            )
        parts.append(f"{quote_ident(name)} {pg_type}")

    return f"CREATE TABLE IF NOT EXISTS {qualify(target)} ({', '.join(parts)})"


def create_table(
    conn: Any,
    target: TargetRef,
    df: pd.DataFrame,
    *,
    overwrite_types: Optional[dict] = None,
    columns: Optional[Sequence[str]] = None,
) -> None:
    ensure_schema(conn, target.schema)
    with conn.cursor() as cur:
        cur.execute(
            build_create_table(df, target, overwrite_types=overwrite_types, columns=columns)
        )


# --- Batch preparation -------------------------------------------------------


def drop_mage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns Mage adds. The only variant across the predecessor."""
    return df.drop(
        columns=[c for c in df.columns if c.startswith(MAGE_COLUMN_PREFIX)], errors="ignore"
    )


def align_df_columns_to_db(
    export_df: pd.DataFrame, db_cols: Sequence[str], show_debug: bool = False
) -> pd.DataFrame:
    """Align the batch columns with what the target table has — names and order.

    **The order is part of the contract**, not cosmetics. The golden test compares
    it as its second level, right after the row count.

    A rewrite of the predecessor, which has a single variant. Only the second
    branch is peculiar: when the target has ``_type`` and the batch has ``type``,
    it is renamed. That is glue for exactly the underscore prefix that the frozen
    target names carry — the batch goes through
    `core.naming.create_column_mapping`, which does not add the prefix, while in
    the target `mage_ai` did add it. In this core the prefix is already added by
    `core.naming.target_column_names`, so the branch has nothing to catch; it
    stays for tables created by the older code.
    """
    if not db_cols:
        return export_df

    df_lower_to_actual = {c.lower(): c for c in export_df.columns}
    rename_map = {}
    for db_col in db_cols:
        db_key = db_col.lower()
        actual = df_lower_to_actual.get(db_key)
        if actual is not None:
            if actual != db_col:
                rename_map[actual] = db_col
            continue
        if db_key.startswith("_"):
            alt = df_lower_to_actual.get(db_key[1:])
            if alt is not None:
                rename_map[alt] = db_col

    if rename_map:
        export_df = export_df.rename(columns=rename_map)
        if show_debug:
            _log.warning("📋 Columns renamed to match the target: %s", rename_map)

    return export_df.reindex(columns=list(db_cols), fill_value=None)


def _integer_bound_columns(
    integer_columns_mapping: Optional[dict],
    overwrite_types: Optional[dict],
    db_integer_cols: Optional[Sequence[str]],
) -> list:
    """Every column that ends up in an integer column of the target.

    Three sources, because `prepare_export_df` is handed the information three
    times over and they do not have to agree: ``integer_columns_mapping`` drives
    `coerce.convert_integer_columns`, ``overwrite_types`` drives
    `coerce.sanitize_integer_columns`, and ``db_integer_cols`` is what the target
    table actually has. A column named by any of them is integer-bound.

    Used **only** by the ``strict_integer_precision`` check. The inherited
    ``integer_cols`` list inside `prepare_export_df` deliberately stays as it is:
    it decides which columns get sanitised and reformatted for the CSV, so
    widening it would change what is written.
    """
    names: list = []
    for column, target_type in (integer_columns_mapping or {}).items():
        if str(target_type).upper() in coerce.INTEGER_TYPES and column not in names:
            names.append(column)
    for column, target_type in (overwrite_types or {}).items():
        if (target_type or "").upper() in _INTEGER_TARGET_TYPES and column not in names:
            names.append(column)
    for column in db_integer_cols or ():
        if column not in names:
            names.append(column)
    return names


def prepare_export_df(
    batch_df: pd.DataFrame,
    meta: dict,
    all_columns: Sequence[str],
    date_like_columns: set,
    integer_columns_mapping: dict,
    overwrite_types: dict,
    orig_type_map: dict,
    db_integer_cols: Optional[Sequence[str]] = None,
    show_debug: bool = False,
    *,
    ensure_ascii: bool = True,
    strict_integer_precision: bool = False,
) -> pd.DataFrame:
    """Prepare a batch for writing. The step order is inherited, not arbitrary.

    ``replace_pdna_with_none`` is called four times on purpose: every preceding
    step is able to produce ``pd.NA`` again.

    A single variant across the predecessor. ``ensure_ascii`` was added as a
    parameter — see the reasoning at `coerce.to_jsonb`.

    ``strict_integer_precision`` (``LOAD_SETTINGS``, off by default) adds one
    check, and its position in the sequence is the whole point. It sits directly
    after the third ``replace_pdna_with_none``, because that is the **last step
    at which an integer-bound column is still a float**:

    * both places the precision goes are already behind it — the read, which
      makes an integer column with a NULL ``float64`` before this function is
      ever called, and that very ``replace_pdna_with_none``, which demotes an
      intact ``Int64`` back to ``float64``;
    * everything after it turns those columns into something a vectorised
      comparison cannot read — ``fix_column_values`` casts them to ``Int64``
      (rounding already applied), `coerce.sanitize_integer_columns` to ``object``
      Python ints, and ``fmt_int_for_csv`` to strings. Checking at the very end
      would mean parsing text row by row, which rule 4 forbids.

    With the key off this costs one boolean test.
    """
    for key, value in meta.items():
        batch_df[key] = value

    ordered = sorted(all_columns) if isinstance(all_columns, set) else list(all_columns)
    for col in ordered:
        if col not in batch_df.columns:
            batch_df[col] = None

    batch_df = batch_df.where(pd.notnull(batch_df), None)
    batch_df = coerce.replace_pdna_with_none(batch_df)
    batch_df = coerce.fix_invalid_dates(batch_df, date_like_columns)
    batch_df = coerce.replace_pdna_with_none(batch_df)
    batch_df = coerce.convert_time_columns(batch_df, overwrite_types, orig_type_map)
    batch_df = coerce.convert_integer_columns(batch_df, integer_columns_mapping)
    batch_df = coerce.replace_pdna_with_none(batch_df)

    if strict_integer_precision:
        coerce.check_integer_precision(
            batch_df,
            _integer_bound_columns(integer_columns_mapping, overwrite_types, db_integer_cols),
        )

    export_df = drop_mage_cols(batch_df)
    export_df = coerce.fix_column_values(
        export_df, overwrite_types, show_debug, ensure_ascii=ensure_ascii
    )

    integer_cols = [
        c for c, t in (overwrite_types or {}).items() if (t or "").upper() in _INTEGER_TARGET_TYPES
    ]
    for col in db_integer_cols or ():
        if col not in integer_cols:
            integer_cols.append(col)

    export_df = coerce.sanitize_integer_columns(export_df, integer_cols, show_debug)
    export_df = coerce.normalize_carriage_returns(export_df)

    for col in integer_cols:
        if col in export_df.columns:
            export_df[col] = export_df[col].astype("object").map(coerce.fmt_int_for_csv)

    return coerce.replace_pdna_with_none(export_df)


# --- Writing: COPY is the only path ------------------------------------------


def _serialize_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Dicts, lists and ``ndarray`` to JSON — otherwise ``repr`` ends up in the CSV.

    A replica of the ``serialize_obj`` branch in
    ``mage_ai.io.postgres.upload_dataframe``. Most of the work was already done by
    `coerce.fix_column_values` (for JSONB-typed columns); this is the safety net
    for columns that are not in `overwrite_types`.
    """
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        if not non_null.map(lambda v: isinstance(v, (dict, list, np.ndarray))).any():
            continue
        df[col] = series.map(_serialize_value)
    return df


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (dict, np.ndarray)):
        return coerce.to_jsonb(value, ensure_ascii=False)
    if isinstance(value, list):
        rendered = coerce.to_jsonb(value, ensure_ascii=False)
        return _clean_array_value(rendered)
    return value


def _clean_array_value(value: Any) -> Any:
    """``[1, 2]`` -> ``{1, 2}`` — the PostgreSQL array literal.

    A replica of ``Postgres._clean_array_value``. Without it, ``COPY`` into a
    ``text[]`` column would fail with "malformed array literal".
    """
    if not isinstance(value, str) or len(value) < 2:
        return value
    if value[0] == "[" and value[-1] == "]":
        return value.replace("[", "{").replace("]", "}")
    return value


class _CsvChunkReader:
    """A file-like object for ``copy_expert`` that produces the CSV in pieces.

    ``psycopg2`` asks for data through ``read(size)``, so the whole DataFrame does
    not have to sit in memory as text. It is serialised `COPY_CHUNK_ROWS` rows at
    a time — the memory peak is then independent of the batch size, which matters
    because the orchestrator runs 3 pipelines side by side.

    The CSV is produced by ``DataFrame.to_csv``, not by hand-rolled quoting — that
    would have to match character by character, otherwise the data in the target
    changes. The only extra parameter compared to Mage is ``QUOTE_ALL``; for why,
    see the module docstring.
    """

    def __init__(self, df: pd.DataFrame, chunk_rows: int = COPY_CHUNK_ROWS) -> None:
        self._chunks = _iter_csv_chunks(df, chunk_rows)
        self._buffer = ""
        #: How far into the buffer we have got. **Must not be replaced by
        #: re-slicing** ``self._buffer = self._buffer[size:]``: psycopg2 asks for
        #: 8 kB at a time, so on a hundred-megabyte piece that would mean twelve
        #: thousand copies of the whole buffer. Measured at 209 s against 9 s.
        self._offset = 0
        self._exhausted = False

    def _fill(self, size: int) -> None:
        """Top the buffer up to at least ``size`` characters from the current position."""
        while len(self._buffer) - self._offset < size and not self._exhausted:
            try:
                self._compact()
                self._buffer += next(self._chunks)
            except StopIteration:
                self._exhausted = True

    def _compact(self) -> None:
        """Drop the part already read. Once per piece, not once per read."""
        if self._offset:
            self._buffer = self._buffer[self._offset :]
            self._offset = 0

    def read(self, size: Optional[int] = -1) -> str:
        if size is None or size < 0:
            rest = self._buffer[self._offset :] + "".join(self._chunks)
            self._buffer = ""
            self._offset = 0
            self._exhausted = True
            return rest

        self._fill(size)
        out = self._buffer[self._offset : self._offset + size]
        self._offset += len(out)
        if self._offset >= len(self._buffer):
            self._buffer = ""
            self._offset = 0
        return out

    def readline(self, size: Optional[int] = -1) -> str:  # pragma: no cover
        """``copy_expert`` uses ``read``; this is here only to complete the interface."""
        while "\n" not in self._buffer[self._offset :] and not self._exhausted:
            try:
                self._buffer += next(self._chunks)
            except StopIteration:
                self._exhausted = True
        idx = self._buffer.find("\n", self._offset)
        if idx < 0:
            return self.read(-1)
        out = self._buffer[self._offset : idx + 1]
        self._offset = idx + 1
        return out


def _iter_csv_chunks(df: pd.DataFrame, chunk_rows: int) -> Iterator[str]:
    for start in range(0, len(df), chunk_rows):
        piece = df.iloc[start : start + chunk_rows]
        yield piece.to_csv(header=False, index=False, na_rep="", quoting=csv.QUOTE_ALL)


def build_copy_sql(target: TargetRef, columns: Sequence[str]) -> str:
    """``COPY … FROM STDIN`` in exactly the shape Mage sends it.

    ``FORCE_NULL`` on every column together with ``NULL ''`` is the reason an
    empty string lands in the target as ``NULL``. See `EMPTY_STRING_AS_NULL`.
    """
    cols = _column_list(columns)
    options = ["FORMAT csv", "DELIMITER ','", "NULL ''"]
    if EMPTY_STRING_AS_NULL:
        options.append(f"FORCE_NULL({cols})")
    return f"COPY {qualify(target)} ({cols}) FROM STDIN ({', '.join(options)})"


def copy_from_stdin(
    conn: Any,
    target: TargetRef,
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    chunk_rows: int = COPY_CHUNK_ROWS,
) -> int:
    """**The only permitted write path.**

    Returns the number of rows written.

    ``columns`` dictates the order — the batch is reindexed onto them, so an extra
    column is dropped and a missing one is filled in as ``NULL``. That is
    deliberate: the column order is part of the contract and must not depend on
    the order in which the source returned them.
    """
    cols = list(columns)
    if not cols:
        raise TargetError("COPY without a column list — the column order is part of the contract.")

    frame = df.reindex(columns=cols)
    if frame.empty:
        return 0

    frame = _serialize_objects(frame.copy())
    reader = _CsvChunkReader(frame, chunk_rows)

    with conn.cursor() as cur:
        cur.copy_expert(build_copy_sql(target, cols), reader)

    return len(frame)


def manual_insert_fallback(
    conn: Any,
    export_df: pd.DataFrame,
    target: TargetRef,
    resolved_pk: str,
    conflict_cols: Optional[Sequence[str]] = None,
) -> int:
    """The emergency path when ``COPY`` does not go through. **Not an alternative.**

    It has to be called explicitly and always logs a warning — otherwise the
    information that the run took the slow path would be lost. ``conflict_cols``
    exists in only one of the predecessor's variants; the other 8 files lack it,
    so it is taken over with the default ``None``.

    The predecessor uses ``psycopg2.extras.execute_batch``; that sends several
    INSERTs in one round trip, but it is still row by row. Hence a fallback.
    """
    from psycopg2.extras import execute_batch

    columns = list(export_df.columns)
    if not columns:
        return 0

    _log.warning(
        "⚠️ Writing into %s takes the emergency path (row-by-row INSERT, %d rows). "
        "COPY did not go through — look for the reason in the exception above this record.",
        qualify(target),
        len(export_df),
    )

    safe_df = export_df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(lambda x: str(x) if x is not None else None)

    keys = list(conflict_cols) if conflict_cols else ([resolved_pk] if resolved_pk else [])
    conflict_clause = ""
    if keys:
        updates = ", ".join(
            f"{quote_ident(c)} = EXCLUDED.{quote_ident(c)}" for c in columns if c not in keys
        )
        conflict_clause = f" ON CONFLICT ({_column_list(keys)}) DO UPDATE SET {updates}"

    sql = (
        f"INSERT INTO {qualify(target)} ({_column_list(columns)}) "
        f"VALUES ({', '.join(['%s'] * len(columns))}){conflict_clause}"
    )
    rows = [tuple(row) for row in safe_df.to_numpy()]
    with conn.cursor() as cur:
        execute_batch(cur, sql, rows)

    return len(rows)


# --- Indexes -----------------------------------------------------------------


def index_name(target: TargetRef, resolved_pk: str) -> str:
    """``idx_<table>_<pk>_unique`` — the inherited name, weaknesses included.

    It is not schema-qualified, so two identically named tables in two schemas ask
    for the same index name. In PostgreSQL that is harmless (indexes live in the
    table's schema), but truncation at 63 characters is not — hence the guard.
    """
    name = f"idx_{target.table}_{resolved_pk}_unique"
    if len(name) > MAX_IDENTIFIER_LENGTH:
        digest = uuid.uuid5(uuid.NAMESPACE_OID, name).hex[:8]
        keep = MAX_IDENTIFIER_LENGTH - len(digest) - 1
        name = f"{name[:keep]}_{digest}"
    return name


def create_unique_pk_index(
    conn: Any,
    target: TargetRef,
    resolved_pk: str,
    all_columns_ordered: Sequence[str],
    fatal: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> bool:
    """Create the unique index over the primary key. Without it upsert cannot work.

    Called twice: right after the table is created (cheap over an empty table and
    it survives a cancelled run) and at the end (``IF NOT EXISTS`` is idempotent).
    With ``fatal=False`` a failure is only logged — the final call is the one that
    has to raise.

    ``CONCURRENTLY`` requires autocommit and psycopg2 refuses to switch the mode
    with an open transaction, hence the ``commit()`` before it.

    **Over a partitioned target ``CONCURRENTLY`` is left out** — PostgreSQL cannot
    do it there ("cannot create index on partitioned table … concurrently"). A
    hard-coded ``CONCURRENTLY`` meant that two partitioned tables never got their
    unique index at all: the call from ``full_by_source`` passes ``fatal=False``,
    so the failure was merely logged, the run finished green, and only upsert did
    not work over them. A plain ``CREATE INDEX`` holds ``ACCESS EXCLUSIVE`` while
    it builds; on a partitioned parent that is acceptable, because it is built
    right after creation over an empty table.

    Note: a unique index over a partitioned table has to contain the partition
    key. With ``partition_by_source`` the caller therefore passes ``columns=[pk,
    _source]``.
    """
    if not resolved_pk or resolved_pk not in set(all_columns_ordered):
        if resolved_pk:
            _log.warning(
                "⚠️ PK column %r is not among the target columns — no index is created.",
                resolved_pk,
            )
        return False

    # `columns` is here because of multi-source: the same key repeats across the
    # different source databases, so on its own it is not unique and the index has
    # to cover the pair (pk, _source). Without the parameter the behaviour is
    # unchanged.
    index_columns = list(columns) if columns else [resolved_pk]
    name = quote_ident(index_name(target, "_".join(index_columns)))
    partitioned = is_partitioned(conn, target)
    concurrent = "" if partitioned else " CONCURRENTLY"
    query = (
        f"CREATE UNIQUE INDEX{concurrent} IF NOT EXISTS {name} "
        f"ON {qualify(target)} ({_column_list(index_columns)})"
    )
    try:
        if partitioned:
            with conn.cursor() as cur:
                cur.execute(query)
        else:
            conn.commit()
            original_autocommit = conn.autocommit
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute(query)
            finally:
                conn.autocommit = original_autocommit
    except Exception as err:
        if partitioned:
            # This branch runs inside the caller's transaction. Without the
            # rollback it would stay aborted under `fatal=False` and the next,
            # unrelated statement would be the one to fail.
            conn.rollback()
        _log.error("🛑 The unique index over %r could not be created: %s", resolved_pk, err)
        _log.error("💡 By hand: %s", query.replace(" CONCURRENTLY", ""))
        if fatal:
            raise
        return False

    _log.info("📝 The unique index over %r is in place.", resolved_pk)
    return True


_UNIQUE_CHECK_SQL = """
SELECT COUNT(*) FROM (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = %(rel)s::regclass
      AND contype IN ('u', 'p')
      AND %(pk)s = ANY(
        SELECT attname FROM pg_attribute
        WHERE attrelid = conrelid AND attnum = ANY(conkey)
      )
    UNION
    SELECT 1 FROM pg_index i
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
    WHERE i.indrelid = %(rel)s::regclass
      AND i.indisunique = true
      AND a.attname = %(pk)s
) AS unique_checks
"""


def assert_unique_pk_index(conn: Any, target: TargetRef, resolved_pk: str) -> None:
    """Verify that the target exists and has a unique index/constraint over the PK.

    **Must run before any expensive work against the source** — a missing index
    should fail within seconds, not after a ``COUNT(*)`` over the incremental
    window.

    **Deviation:** the predecessor assembles the query as an f-string with
    ``resolved_pk`` inside it. Here it is a bound parameter. The column name does
    come from the configuration rather than from a user, but an apostrophe in it
    would break the query as written today.
    """
    if not resolved_pk:
        raise TargetError(
            "assert_unique_pk_index without a PK — the caller has to make sure there is one."
        )

    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{target.schema}.{target.table}",))
        if cur.fetchone()[0] is None:
            raise TargetError(
                f"Target table {qualify(target)} does not exist — an incremental run has "
                "nothing to update. Run a full load first."
            )

        cur.execute(
            _UNIQUE_CHECK_SQL,
            {"rel": f"{target.schema}.{target.table}", "pk": resolved_pk},
        )
        if cur.fetchone()[0] == 0:
            raise TargetError(
                f"{qualify(target)}.{resolved_pk} is missing a UNIQUE/PRIMARY KEY. Without "
                "one, upsert does not work. Run a full load, or create the index by hand: "
                f"CREATE UNIQUE INDEX {index_name(target, resolved_pk)} "
                f"ON {qualify(target)} ({quote_ident(resolved_pk)})"
            )


def ensure_unique_index(
    conn: Any, target: TargetRef, resolved_pk: str, all_columns_ordered: Sequence[str]
) -> None:
    """Create the index if it is missing, then verify it straight away."""
    create_unique_pk_index(conn, target, resolved_pk, all_columns_ordered, fatal=False)
    assert_unique_pk_index(conn, target, resolved_pk)


def key_index_name(target: TargetRef, column: str) -> str:
    """``idx_<table>_<column>`` — a non-unique index, same guard on the length."""
    name = f"idx_{target.table}_{column}"
    if len(name) > MAX_IDENTIFIER_LENGTH:
        digest = uuid.uuid5(uuid.NAMESPACE_OID, name).hex[:8]
        name = f"{name[: MAX_IDENTIFIER_LENGTH - len(digest) - 1]}_{digest}"
    return name


def ensure_key_index(conn: Any, target: TargetRef, column: str) -> None:
    """A plain index over the key. `replace_by_key` needs it on a partitioned target.

    Over a partitioned table a unique index cannot stand on the key alone
    (PostgreSQL requires the partition key in it), so deleting by key would be a
    seq scan **across every partition** without this index. A non-unique index
    over an arbitrary column of a partitioned table is allowed.
    """
    name = quote_ident(key_index_name(target, column))
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {name} ON {qualify(target)} ({quote_ident(column)})"
        )


def replace_by_key(
    conn: Any, staging: TargetRef, target: TargetRef, columns: Sequence[str], pk: str
) -> int:
    """``DELETE`` by key + ``INSERT`` from the staging table. **In one transaction.**

    A replacement for ``INSERT … ON CONFLICT DO UPDATE`` on a partitioned target,
    for two reasons:

    1. ``ON CONFLICT (pk)`` has nothing to lean on. PostgreSQL requires a unique
       index over a partitioned table to contain the partition key, so an index
       over ``pk`` alone cannot exist there.
    2. If the conflict key were merely widened to ``(pk, partition_key)``, a row
       whose partition value changes in the source would **not be updated but
       duplicated** — the old one would stay in the old partition. Deleting by the
       key alone finds it no matter where it sits.

    Returns the number of inserted rows. The caller must have autocommit off —
    between the ``DELETE`` and the ``INSERT`` the target is incomplete.

    **Duplicates in staging are not caught here.** ``ON CONFLICT`` fails on them
    ("command cannot affect row a second time"); this path inserts both, and only
    the unique index catches them — that is, only if they share the partition
    value too. A source returning the same key twice is a configuration error.
    """
    if conn.autocommit:
        raise TargetError(
            "replace_by_key requires autocommit to be off — otherwise the target would be "
            "visibly incomplete between the DELETE and the INSERT."
        )

    cols = _column_list(columns)
    key = quote_ident(pk)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {qualify(target)} t USING {qualify(staging)} s "
            f"WHERE t.{key} = s.{key}"
        )
        deleted = int(cur.rowcount)
        cur.execute(f"INSERT INTO {qualify(target)} ({cols}) SELECT {cols} FROM {qualify(staging)}")
        inserted = int(cur.rowcount)

    _log.info("✅ Replaced by %r: %s deleted, %s inserted.", pk, f"{deleted:,}", f"{inserted:,}")
    return inserted


# --- Shadow table and the atomic full load -----------------------------------
#
# Replaces the `DROP TABLE … CASCADE` + stream of the predecessor, which drops
# 102 views and on an interruption leaves the table truncated (documented:
# 11.4 of 18.8 M rows).
#
# **The swap is not done through RENAME.** The original brief prescribed it and
# promised the views would survive — they do not. PostgreSQL keeps the table OID
# in the view definition, not the name, so after
# `ALTER TABLE t RENAME TO t_old; ALTER TABLE shadow RENAME TO t` the view stays
# bound to `t_old` and `pg_get_viewdef` rewrites itself accordingly. Verified on
# PG 17.9: after the swap the view returns the old data. That is worse than the
# `DROP CASCADE` it replaces — that one kills views visibly, this would freeze
# them silently.
#
# What does work is `TRUNCATE` + `INSERT … SELECT` in one transaction: the view
# stays bound, sees the new data, and a `ROLLBACK` leaves the target untouched
# (verified as well). Measured at 523 000 rows/s, i.e. ~36 s of an exclusive lock
# on a table with 18.8 M rows.


def shadow_ref(target: TargetRef) -> TargetRef:
    """Address of the shadow table. The random suffix is there for concurrency.

    The orchestrator runs 3 pipelines side by side and nothing guarantees that two
    of them will not touch the same table. A fixed name would mean they overwrite
    each other's shadow; hence a UUID rather than something like ``_tmp``.
    """
    from dbextractors.core.strategies.base import TargetRef as _TargetRef

    suffix = uuid.uuid4().hex[:8]
    room = MAX_IDENTIFIER_LENGTH - len(SHADOW_PREFIX) - len(suffix) - 1
    name = f"{SHADOW_PREFIX}{target.table[:room]}_{suffix}"
    return _TargetRef(schema=target.schema, table=name)


def create_shadow_table(
    conn: Any,
    target: TargetRef,
    *,
    df: Optional[pd.DataFrame] = None,
    overwrite_types: Optional[dict] = None,
    columns: Optional[Sequence[str]] = None,
) -> TargetRef:
    """Create the shadow table the full load writes into instead of the target.

    When the target exists, its structure is taken over through ``LIKE …
    INCLUDING DEFAULTS``. Indexes are **deliberately not copied**: the shadow is
    only filled and dropped, so maintaining them would be pointless and they would
    slow ``COPY`` down. The target keeps its own indexes, because ``TRUNCATE``
    does not remove them.

    When the target does not exist, the schema is derived from the batch — that
    needs either ``df`` or ``columns`` together with ``overwrite_types``.
    """
    shadow = shadow_ref(target)
    ensure_schema(conn, target.schema)

    if table_exists(conn, target):
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE {qualify(shadow)} "
                f"(LIKE {qualify(target)} INCLUDING DEFAULTS INCLUDING GENERATED)"
            )
    elif df is not None or columns is not None:
        create_table(
            conn,
            shadow,
            df if df is not None else pd.DataFrame(columns=list(columns or ())),
            overwrite_types=overwrite_types,
            columns=columns,
        )
    else:
        raise TargetError(
            f"Target table {qualify(target)} does not exist and the shadow has nothing to "
            "be built from — pass `df`, or `columns` together with `overwrite_types`."
        )

    _log.info("📝 Shadow table %s created for %s.", qualify(shadow), qualify(target))
    return shadow


def swap_shadow_table(conn: Any, shadow: TargetRef, target: TargetRef, spec: Any = None) -> int:
    """Swap the shadow contents into the target. **One transaction, no CASCADE.**

    Returns the number of rows swapped over. The caller must have
    ``conn.autocommit`` off — otherwise TRUNCATE and INSERT would be two
    transactions and the target would be empty in between.

    When the target does not exist there is nothing to rescue (a view over a
    missing table cannot exist) and the shadow is simply renamed. That is the one
    situation in which RENAME is fine.

    With `spec` (`partitioning.PartitionSpec`) **RENAME is not used**: renaming
    would turn a plain shadow into a plain target, silently discarding the
    requested partitioning. Instead a partitioned parent is created from the shape
    of the shadow. On a partitioned target, partitions for the values actually
    present in the shadow are created before the insert — without them ``INSERT``
    would fail with "no partition found for row".
    """
    if conn.autocommit:
        raise TargetError(
            "swap_shadow_table requires autocommit to be off — otherwise the target would "
            "be visibly empty between the TRUNCATE and the INSERT."
        )

    if not table_exists(conn, target):
        if spec is None:
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE {qualify(shadow)} RENAME TO {quote_ident(target.table)}")
                cur.execute(f"SELECT count(*) FROM {qualify(target)}")
                rows = cur.fetchone()[0]
            _log.info("🔁 Target %s did not exist; shadow renamed into place.", qualify(target))
            return int(rows)

        from dbextractors.core import partitioning

        partitioning.create_parent(conn, target, shadow, spec)

    if spec is not None and is_partitioned(conn, target):
        from dbextractors.core import partitioning

        partitioning.ensure_partitions_from_table(conn, target, spec, shadow)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (target.schema, target.table),
        )
        target_cols = [r[0] for r in cur.fetchall()]

        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (shadow.schema, shadow.table),
        )
        shadow_cols = {r[0] for r in cur.fetchall()}

        missing = [c for c in target_cols if c not in shadow_cols]
        if missing:
            raise TargetError(
                f"Shadow {qualify(shadow)} is missing columns {missing} that the target has. "
                "The swap would blank them out."
            )

        cols = _column_list(target_cols)
        cur.execute(f"TRUNCATE {qualify(target)}")
        cur.execute(f"INSERT INTO {qualify(target)} ({cols}) SELECT {cols} FROM {qualify(shadow)}")
        rows = cur.rowcount

    _log.info("🔁 %s rows swapped from the shadow into %s.", f"{rows:,}", qualify(target))
    return int(rows)


def drop_shadow_table(conn: Any, shadow: TargetRef) -> None:
    """Clean the shadow table up. Has to happen on failure too.

    Without ``CASCADE`` on purpose: nothing should depend on the shadow, and if
    something does, we want to know instead of dropping it silently.
    """
    if not shadow.table.startswith(SHADOW_PREFIX):
        raise TargetError(
            f"drop_shadow_table was given {shadow.table!r}, which is not a shadow table "
            f"(the {SHADOW_PREFIX!r} prefix is missing). A guard against dropping the target."
        )
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {qualify(shadow)}")


def find_stale_shadows(conn: Any, schema: str) -> List[str]:
    """Shadow tables somebody left behind.

    Does not clean them up — only returns them. With three pipelines running side
    by side, automatic cleanup could delete the shadow of a run in progress.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name LIKE %s ORDER BY table_name",
            (schema, f"{SHADOW_PREFIX}%"),
        )
        return [r[0] for r in cur.fetchall()]


# --- Dedup without a Python set ----------------------------------------------
#
# The predecessor has `seen_keys = set()`, holding every PK of the whole table as
# Python strings in RAM — several GB at 15 M rows, and the orchestrator runs three
# pipelines side by side, so the peak is tripled.


class SeenKeys:
    """A record of already seen primary keys in a temporary PostgreSQL table.

    Python memory is bounded by one batch, not by the size of the table.

    **A fix over the predecessor:** its code compares incomparable things —
    ``seen_keys`` is filled with strings (``.astype(str)``) but the raw column is
    filtered (``batch_df[pk].isin(seen_keys)``). With an integer PK it therefore
    never matches and the dedup is dead code. Here both sides are compared as
    text, so the dedup really works. Consequence: wherever OFFSET instability
    produced duplicates, there will now be fewer rows — which is exactly what that
    code was supposed to do.
    """

    def __init__(self, conn: Any, pk_column: str) -> None:
        self._conn = conn
        self._pk = pk_column
        self._table = f"dbx_seen_pk_{uuid.uuid4().hex[:8]}"
        self._open = False

    def __enter__(self) -> SeenKeys:
        with self._conn.cursor() as cur:
            cur.execute(f"CREATE TEMP TABLE {quote_ident(self._table)} (k TEXT PRIMARY KEY)")
        self._open = True
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Clean the temporary table up. **Must not mask the exception it cleans up after.**

        `SeenKeys` is used as a context manager inside a ``try``, so `__exit__`
        runs **before** the caller's ``except``. When the write failed, the
        transaction is aborted and this ``DROP`` ends in
        ``InFailedSqlTransaction`` — and that error would replace the original
        one. Exactly that happened on a table with a ``money`` column: the report
        showed nothing but "current transaction is aborted" and the real cause was
        lost.

        A failure is therefore **logged** (swallowing it silently is forbidden)
        and not raised. The temporary table disappears with the connection anyway.
        """
        if not self._open:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {quote_ident(self._table)}")
        except Exception as err:
            _log.warning(
                "⚠️ The temporary table %s could not be cleaned up (%s). It will disappear "
                "with the connection; the original error is not being masked by this.",
                self._table,
                err,
            )
        self._open = False

    def filter_new(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows whose PK has not appeared in any previous batch."""
        if self._pk not in batch_df.columns or batch_df.empty:
            return batch_df

        keys = batch_df[self._pk].dropna().astype(str)
        if keys.empty:
            return batch_df

        unique_keys = keys.drop_duplicates()
        reader = _CsvChunkReader(unique_keys.to_frame(name="k"))
        staging = f"{self._table}_in"
        with self._conn.cursor() as cur:
            cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS {quote_ident(staging)} (k TEXT)")
            cur.execute(f"TRUNCATE {quote_ident(staging)}")
            cur.copy_expert(
                f"COPY {quote_ident(staging)} (k) FROM STDIN (FORMAT csv, DELIMITER ',')", reader
            )
            cur.execute(
                f"INSERT INTO {quote_ident(self._table)} (k) "
                f"SELECT DISTINCT k FROM {quote_ident(staging)} "
                f"ON CONFLICT (k) DO NOTHING RETURNING k"
            )
            fresh = {r[0] for r in cur.fetchall()}

        removed = len(unique_keys) - len(fresh)
        if removed <= 0:
            return batch_df

        _log.info("🧹 Dedup by %r: %d duplicates dropped.", self._pk, removed)
        return batch_df[batch_df[self._pk].astype(str).isin(fresh)]


# --- Tracking deleted rows ---------------------------------------------------


def _live_pk_reader(live_pk_source: Any) -> Any:
    """The source of live PKs as a file-like object for ``copy_expert``.

    Accepts anything readable (the predecessor's CSV in ``/tmp``), a pandas
    ``Series``, a single-column ``DataFrame`` or a plain iterable of keys. The
    predecessor ships the list through a file, which is a pointless detour via
    disk — a temporary table is the mandated way.
    """
    if hasattr(live_pk_source, "read"):
        return live_pk_source
    if isinstance(live_pk_source, pd.DataFrame):
        return _CsvChunkReader(live_pk_source.iloc[:, :1])
    if isinstance(live_pk_source, pd.Series):
        return _CsvChunkReader(live_pk_source.to_frame())
    return _CsvChunkReader(pd.DataFrame({"k": list(live_pk_source)}))


def apply_live_pk_snapshot(
    conn: Any,
    target: TargetRef,
    resolved_pk: str,
    live_pk_source: Any,
    pk_pg_type: str = "text",
    source_label: Optional[str] = None,
    show_debug: bool = False,
) -> int:
    """Project into the target which rows still exist in the source.

    Returns the number of rows whose ``_deleted_in_source`` changed.

    **No strategy calls this today** — `hash_diff` marks deletions through
    `mark_deleted_in_source` directly, because its diff already holds the live
    keys, and the incremental path deliberately does not sweep (see
    ``strategies/incremental.py``). It is kept as the building block for an
    opt-in sweep and mirrors the predecessor's helper of the same name.

    ``source_label`` is used only by MSSQL today, as the only dialect with
    multi-source and partitioning; elsewhere it is ``None``.

    **Deviation:** on an error the predecessor does ``rollback()`` and ``raise``,
    but ``commit()`` itself beforehand. Here the transaction is left to the caller
    — a module that commits on its own cannot be composed into a larger
    transaction.
    """
    if not resolved_pk:
        raise TargetError("apply_live_pk_snapshot without a PK — there is nothing to match on.")

    temp_table = f"dbx_live_pk_{uuid.uuid4().hex[:8]}"
    pk_type_sql = pk_pg_type or "text"

    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TEMP TABLE {quote_ident(temp_table)} "
            f"({quote_ident(resolved_pk)} {pk_type_sql})"
        )
        cur.copy_expert(
            f"COPY {quote_ident(temp_table)} ({quote_ident(resolved_pk)}) "
            f"FROM STDIN (FORMAT csv, DELIMITER ',')",
            _live_pk_reader(live_pk_source),
        )
        cur.execute(f"CREATE INDEX ON {quote_ident(temp_table)} ({quote_ident(resolved_pk)})")
        cur.execute(f"ANALYZE {quote_ident(temp_table)}")

        if show_debug:
            cur.execute(f"SELECT count(*) FROM {quote_ident(temp_table)}")
            _log.warning("🔢 Live PK snapshot: %s keys.", f"{cur.fetchone()[0]:,}")

        changed = mark_deleted_in_source(
            conn, target, resolved_pk, temp_table, source_label=source_label
        )
        cur.execute(f"DROP TABLE IF EXISTS {quote_ident(temp_table)}")

    return changed


def timestamp_expr(conn: Any, target: TargetRef) -> str:
    """``NOW()`` in the shape that fits the target's ``_timestamp`` column type.

    Two generations of tables have that column **differently** and PostgreSQL does
    not convert between them on assignment:

    - dbextractors creates it as ``TEXT`` (`MANAGED_COLUMNS`) — that is how
      `mage_ai.io.postgres` produced it too, because an ISO string went into the
      frame;
    - the predecessor's add-column branch creates it as ``TIMESTAMP``
      (``ADD COLUMN _timestamp TIMESTAMP``, 8 files).

    A hard-coded ``NOW()::text`` therefore fails on the second generation::

        column "_timestamp" is of type timestamp without time zone
        but expression is of type text

    A missing column or table yields the text variant — the caller will write
    nothing there anyway.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = '_timestamp'",
            (target.schema, target.table),
        )
        row = cur.fetchone()

    if row and str(row[0]).lower().startswith("timestamp"):
        return "NOW()"
    return "NOW()::text"


def mark_deleted_in_source(
    conn: Any,
    target: TargetRef,
    resolved_pk: str,
    live_pk_table: str,
    source_label: Optional[str] = None,
    *,
    pk_as_text: bool = False,
) -> int:
    """Mark rows that vanished and unmark those that came back.

    In the predecessor this sits inside `apply_live_pk_snapshot`. It is pulled out
    because **every** strategy has to be able to do it, not only the ones that
    take a snapshot today — 5 of the 15 files merely declare the column and never
    flip it, so it is permanently ``FALSE`` and 133 dbt models are looking at a
    lie.

    ``pk_as_text`` compares the key through ``tgt.pk::text``. `SourceHashSnapshot`
    needs that, as it keeps the key as ``TEXT`` so it can be compared across
    types; `apply_live_pk_snapshot` creates its temporary table in the right type
    straight away, so it stays with the direct comparison.
    """
    source_filter = f" AND tgt.{quote_ident('_source')} = %s" if source_label else ""
    params: Tuple = (source_label,) if source_label else ()
    live = quote_ident(live_pk_table)
    pk = quote_ident(resolved_pk)
    tgt_pk = f"tgt.{pk}::text" if pk_as_text else f"tgt.{pk}"
    now = timestamp_expr(conn, target)

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {qualify(target)} tgt "
            f'SET "_deleted_in_source" = FALSE, "_timestamp" = {now} '
            f"FROM {live} src "
            f'WHERE {tgt_pk} = src.{pk} AND tgt."_deleted_in_source" IS TRUE{source_filter}',
            params,
        )
        revived = cur.rowcount

        cur.execute(
            f"UPDATE {qualify(target)} tgt "
            f'SET "_deleted_in_source" = TRUE, "_timestamp" = {now} '
            f'WHERE tgt."_deleted_in_source" IS FALSE{source_filter} '
            f"AND NOT EXISTS (SELECT 1 FROM {live} src WHERE src.{pk} = {tgt_pk})",
            params,
        )
        deleted = cur.rowcount

    _log.info(
        "🗑️ %s: %s marked as deleted, %s revived.",
        qualify(target),
        f"{deleted:,}",
        f"{revived:,}",
    )
    return int(revived) + int(deleted)


# --- Snapshot of source hashes -----------------------------------------------


class SourceHashSnapshot:
    """``(pk, hash)`` pairs from the source in a temporary PostgreSQL table.

    Replaces the predecessor's `target_hashes = {}`, which holds the **entire
    target snapshot** in memory (`fetchmany(50000)` over every row) and then
    compares through ``chunk.iterrows()``. With ~530 tables and three concurrent
    pipelines that is precisely the banned approach.

    The target is in PostgreSQL anyway and the snapshot gets here by `COPY`, so
    the diff is a single ``LEFT JOIN``. The same table doubles as the list of live
    keys for `mark_deleted_in_source` — the source is not read twice.
    """

    def __init__(self, conn: Any, pk_column: str, hash_column: str) -> None:
        self._conn = conn
        self._pk = pk_column
        self._hash = hash_column
        self.table = f"dbx_src_hash_{uuid.uuid4().hex[:8]}"
        self._open = False

    def __enter__(self) -> SourceHashSnapshot:
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TEMP TABLE {quote_ident(self.table)} "
                f"({quote_ident(self._pk)} TEXT, {quote_ident(self._hash)} TEXT)"
            )
        self._open = True
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._open:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {quote_ident(self.table)}")
        self._open = False

    def add(self, df: pd.DataFrame) -> int:
        """Add a batch. The key is stored as text so it can be compared across types."""
        from dbextractors.core import hashing

        if df.empty:
            return 0
        frame = pd.DataFrame(
            {
                self._pk: hashing.pk_series_to_str(df[self._pk]),
                self._hash: df[self._hash].astype("object"),
            }
        )
        reader = _CsvChunkReader(frame)
        with self._conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {quote_ident(self.table)} "
                f"({quote_ident(self._pk)}, {quote_ident(self._hash)}) "
                f"FROM STDIN (FORMAT csv, DELIMITER ',')",
                reader,
            )
        return len(frame)

    def index(self) -> None:
        """Index and ``ANALYZE`` — without them the planner picks a nested loop."""
        with self._conn.cursor() as cur:
            cur.execute(f"CREATE INDEX ON {quote_ident(self.table)} ({quote_ident(self._pk)})")
            cur.execute(f"ANALYZE {quote_ident(self.table)}")

    def diff(self, target: TargetRef) -> dict:
        """Compute the difference against the target **in a single query**.

        ``IS DISTINCT FROM`` matters: ``hash <> hash`` would return NULL with a
        NULL on either side and the row would drop out of the result — exactly the
        rows that seeding missed.
        """
        pk, hash_col = quote_ident(self._pk), quote_ident(self._hash)
        in_target = f"t.{pk} IS NOT NULL"
        differs = f"t.{hash_col} IS DISTINCT FROM s.{hash_col}"
        sql = (
            f"SELECT count(*) FILTER (WHERE t.{pk} IS NULL), "
            f"       count(*) FILTER (WHERE {in_target} AND {differs}), "
            f"       count(*) FILTER (WHERE {in_target} AND NOT ({differs})), "
            f"       count(*) "
            f"FROM {quote_ident(self.table)} s "
            f"LEFT JOIN {qualify(target)} t ON t.{pk}::text = s.{pk}"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql)
            added, changed, matched, total = cur.fetchone()
        return {
            "added": int(added),
            "changed": int(changed),
            "matched": int(matched),
            "source_rows": int(total),
        }

    def changed_keys(self, target: TargetRef, batch_size: int) -> Iterator[List[str]]:
        """Keys that changed or were added — batch by batch.

        Batching is deliberate: on a large table the list of changed keys can run
        into millions and has no business sitting in memory in one piece.
        """
        pk, hash_col = quote_ident(self._pk), quote_ident(self._hash)
        sql = (
            f"SELECT s.{pk} FROM {quote_ident(self.table)} s "
            f"LEFT JOIN {qualify(target)} t ON t.{pk}::text = s.{pk} "
            f"WHERE t.{pk} IS NULL OR t.{hash_col} IS DISTINCT FROM s.{hash_col}"
        )
        with self._conn.cursor(name=f"cursor_{self.table}") as cur:
            cur.itersize = int(batch_size)
            cur.execute(sql)
            batch: List[str] = []
            for (key,) in cur:
                batch.append(key)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch


# --- Per-source slice: `_source`, partitions and fingerprints ----------------
#
# All of this exists in only one of the predecessor's extractors, the MSSQL one,
# which fills a single target table from N source databases. It belongs here
# because it is pure PostgreSQL DDL — a strategy should say what, not how.


def is_partitioned(conn: Any, target: TargetRef) -> bool:
    """Is the target split into partitions? ``relkind = 'p'``.

    It asks **reality**, not the configuration. An existing plain table is never
    silently rewritten into a partitioned one: `partition_by_source` only decides
    for a table that does not exist yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relkind = 'p' FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s",
            (target.schema, target.table),
        )
        row = cur.fetchone()
    return bool(row and row[0])


def _source_spec():
    """Partitioning by ``_source`` expressed as a `PartitionSpec`.

    The import is lazy on purpose: `core.partitioning` builds on this module, so
    the opposite direction may only be reached at run time.
    """
    from dbextractors.core.partitioning import MODE_LIST, PartitionSpec

    return PartitionSpec(column=SOURCE_COLUMN, mode=MODE_LIST, default_partition=False)


def partition_name(target: TargetRef, source_label: str) -> str:
    """A deterministic and safe partition name for one source.

    Anything longer than ``MAX_IDENTIFIER_LENGTH`` is silently truncated by
    PostgreSQL, and two sources sharing a long prefix would end up with the same
    partition. A long name is therefore shortened and given a digest — the same
    way shadow tables are handled.
    """
    from dbextractors.core import partitioning

    return partitioning.partition_ref(target, _source_spec(), source_label).table


def create_partitioned_table(conn: Any, target: TargetRef, like: TargetRef) -> None:
    """Create the target as ``PARTITION BY LIST (_source)`` from another table's shape.

    A narrowing of `partitioning.create_parent` down to the per-source split that
    `full_by_source` supports. The general split by ``partition_by`` goes straight
    to `partitioning`.
    """
    from dbextractors.core import partitioning

    partitioning.create_parent(conn, target, like, _source_spec())


def ensure_partition(conn: Any, target: TargetRef, source_label: str) -> TargetRef:
    """Make sure a partition for one source exists and return its address.

    This is what makes a new source database self-service: it shows up in the
    control layer and its partition is created here, with no manual DDL.
    """
    from dbextractors.core import partitioning

    return partitioning.ensure_partition_for(conn, target, _source_spec(), source_label)


def ensure_source_index(conn: Any, target: TargetRef) -> None:
    """An index over ``_source`` for deleting a whole slice.

    ``DELETE … WHERE _source = %s`` cannot use a unique index whose first column
    is the primary key — without this it is a seq scan over the entire table. A
    partitioned target does not need it; there the slice is cut off by partition
    pruning.
    """
    if is_partitioned(conn, target):
        return
    name = f"idx_{target.table}__source"[:MAX_IDENTIFIER_LENGTH]
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {quote_ident(name)} "
            f"ON {qualify(target)} ({quote_ident(SOURCE_COLUMN)})"
        )


def replace_source_slice(
    conn: Any,
    target: TargetRef,
    staging: TargetRef,
    columns: Sequence[str],
    source_label: Optional[str],
) -> Tuple[int, int]:
    """Replace one source's slice with the staging contents. **In one transaction.**

    Returns ``(deleted, written)``; on a partitioned target deleted is ``-1``,
    because ``TRUNCATE`` does not report a count.

    A partitioned target is cut by ``TRUNCATE`` of a single partition: that is
    O(1) and leaves no dead rows, unlike deleting ~66% of a two-million-row table
    every night.
    """
    deleted = -1
    if source_label and is_partitioned(conn, target):
        part = ensure_partition(conn, target, source_label)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {qualify(part)}")
    else:
        with conn.cursor() as cur:
            if source_label:
                cur.execute(
                    f"DELETE FROM {qualify(target)} WHERE {quote_ident(SOURCE_COLUMN)} = %s",
                    (source_label,),
                )
            else:
                cur.execute(f"DELETE FROM {qualify(target)}")
            deleted = int(cur.rowcount)

    cols = _column_list(columns)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {qualify(target)} ({cols}) SELECT {cols} FROM {qualify(staging)}")
        written = int(cur.rowcount)

    _log.info(
        "✅ Slice %r: %s deleted, %s written.",
        source_label or "-",
        "partition truncated" if deleted < 0 else f"{deleted:,}",
        f"{written:,}",
    )
    return deleted, written


def target_holds_source(conn: Any, target: TargetRef, source_label: str) -> bool:
    """Does the target still hold this source's slice?

    The fingerprint alone is not enough: the first database of a rebuilt table
    creates it again, so from the second one onwards the table is back and a stale
    fingerprint would skip all the remaining databases. Documented in production —
    that is how 3 rows out of 220 972 were left in the target.
    """
    if not table_exists(conn, target):
        return False
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT EXISTS (SELECT 1 FROM {qualify(target)} "
            f"WHERE {quote_ident(SOURCE_COLUMN)} = %s)",
            (source_label,),
        )
        return bool(cur.fetchone()[0])


# --- Source fingerprints -----------------------------------------------------


def ensure_fingerprint_store(conn: Any, schema: str) -> None:
    """Create the fingerprint table if it is missing.

    A fingerprint is a cheap aggregate query over the source; when it has not
    changed since last time, the database does not have to be pulled at all. The
    predecessor computes it itself but stores and reads it through a function
    living in the **client repository** — so the behaviour differs per repo and
    would be missing from a shared package. Here the package has its own store and
    the client has to supply nothing.
    """
    ensure_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {quote_ident(schema)}.{quote_ident(FINGERPRINT_TABLE)} ("
            f"  table_name TEXT NOT NULL,"
            f"  source_label TEXT NOT NULL,"
            f"  fingerprint TEXT NOT NULL,"
            f"  parts JSONB,"
            f"  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            f"  PRIMARY KEY (table_name, source_label))"
        )


def read_fingerprint(conn: Any, target: TargetRef, source_label: str) -> Optional[str]:
    """The last stored fingerprint, or ``None`` when there is none."""
    ensure_fingerprint_store(conn, target.schema)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT fingerprint FROM {quote_ident(target.schema)}."
            f"{quote_ident(FINGERPRINT_TABLE)} WHERE table_name = %s AND source_label = %s",
            (target.table, source_label),
        )
        row = cur.fetchone()
    return row[0] if row else None


def write_fingerprint(
    conn: Any,
    target: TargetRef,
    source_label: str,
    fingerprint: str,
    parts: Optional[dict] = None,
) -> None:
    """Store the fingerprint. Written **after** a successful transfer, not before.

    If it were stored earlier, an interrupted run would leave behind a fingerprint
    for a source whose data is not in the target — and the next run would skip it
    as unchanged.
    """
    ensure_fingerprint_store(conn, target.schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {quote_ident(target.schema)}.{quote_ident(FINGERPRINT_TABLE)} "
            f"(table_name, source_label, fingerprint, parts) VALUES (%s, %s, %s, %s) "
            f"ON CONFLICT (table_name, source_label) DO UPDATE SET "
            f"fingerprint = EXCLUDED.fingerprint, parts = EXCLUDED.parts, updated_at = now()",
            # `default=str` because of the numpy types coming from pandas: `int64`
            # and `Timestamp` are not JSON serialisable, and the fingerprint is
            # only bookkeeping, not data.
            (target.table, source_label, fingerprint, json.dumps(parts or {}, default=str)),
        )
