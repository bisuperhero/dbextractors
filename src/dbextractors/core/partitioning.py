"""Partitioning of the target table — LIST by column value, RANGE by time.

Turned on with the ``partition_by`` key in ``LOAD_SETTINGS``. **Without it nothing
is partitioned** — it is a new key in the frozen configuration contract with a safe
default, so none of the ~670 existing tables changes shape::

    LOAD_SETTINGS:
      partition_by:
        column: order_date
        mode: range_month        # list | range_day | range_month | range_year
        default_partition: true  # catch-all partition, on by default

    # shorthand for LIST by a column
    LOAD_SETTINGS:
      partition_by: _source

The existing ``partition_by_source: true`` keeps working and means exactly
``partition_by: {column: _source, mode: list}``.

## Why an existing table is never rebuilt

``relkind`` cannot be changed in PostgreSQL, so converting an ordinary table into a
partitioned one means ``DROP`` + ``CREATE`` + refill. 102 views hang off the target
tables, ``DROP … CASCADE`` is forbidden by the project's performance rules, and
``RENAME`` is out of the question too (a view holds on to the OID, not the name).
Partitioning is therefore applied **only to a table that does not exist yet**; for
an existing one the decision follows reality (``relkind = 'p'``), not the
configuration, and a mismatch is merely logged.

## Why partitions are created from the staging table

Every strategy first pours the data into a staging/shadow table and only then
projects it into the target with a single set-based statement. Partitions are
created in that intermediate step, from the values actually present in staging
(``SELECT DISTINCT``) — not from the min–max range. A range would turn one stray
date from 1900 into a thousand empty partitions; distinct creates exactly those that
are needed.

The catch-all ``DEFAULT`` partition is the safety net for whatever was missed:
without it, a single unroutable row brings the run down with "no partition found for
row". ``NULL``s in the partition key, which fit no range, land in it too.

## The trap this closes: upsert against a partitioned target

PostgreSQL requires a unique index on a partitioned table to contain the partition
key. Over ``partition_by: order_date`` there can therefore be no index on ``id``
alone, and ``INSERT … ON CONFLICT (id)`` **has nothing to lean on**. Merely widening
the key to ``(id, order_date)`` would mean that a row whose date changes at the
source is not updated but duplicated — the old one would stay in the old partition.

That is why upserting strategies against a partitioned target do not go through
``ON CONFLICT`` but through ``DELETE`` by key plus ``INSERT``
(`target_pg.replace_by_key`). The delete finds the old row regardless of which
partition it sits in, and the insert routes it by the new value.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from dbextractors.core.target_pg import (
    MAX_IDENTIFIER_LENGTH,
    SOURCE_COLUMN,
    TargetError,
    ensure_schema,
    qualify,
    quote_ident,
)

if TYPE_CHECKING:  # pragma: no cover
    from dbextractors.core.strategies.base import TargetRef

MODE_LIST = "list"
MODE_RANGE_DAY = "range_day"
MODE_RANGE_MONTH = "range_month"
MODE_RANGE_YEAR = "range_year"

RANGE_MODES = (MODE_RANGE_DAY, MODE_RANGE_MONTH, MODE_RANGE_YEAR)
VALID_MODES = (MODE_LIST, *RANGE_MODES)

#: The ``date_trunc`` unit for each range mode.
_TRUNC_UNIT = {
    MODE_RANGE_DAY: "day",
    MODE_RANGE_MONTH: "month",
    MODE_RANGE_YEAR: "year",
}

DEFAULT_SUFFIX = "default"


@dataclass(frozen=True)
class PartitionSpec:
    """A parsed ``partition_by``. A value object with no behaviour."""

    column: str
    mode: str = MODE_LIST
    default_partition: bool = True

    @property
    def is_range(self) -> bool:
        return self.mode in RANGE_MODES

    def as_dict(self) -> dict:
        return {
            "column": self.column,
            "mode": self.mode,
            "default_partition": self.default_partition,
        }


def parse_spec(value: Any, *, legacy_by_source: bool = False) -> Optional[PartitionSpec]:
    """``partition_by`` from the configuration -> `PartitionSpec`, or ``None``.

    ``legacy_by_source`` is the existing ``partition_by_source: true``. It applies
    only when ``partition_by`` says nothing — whoever writes both meant the more
    specific one.

    Errors are ``ValueError``; `core.config` converts them into ``ConfigError``, so
    that this module need not depend on the configuration layer.
    """
    if value is None or value is False or value == "":
        return PartitionSpec(column=SOURCE_COLUMN, mode=MODE_LIST) if legacy_by_source else None

    if isinstance(value, str):
        name = value.strip()
        if not name:
            raise ValueError("partition_by: empty column name.")
        return PartitionSpec(column=name, mode=MODE_LIST)

    if not isinstance(value, dict):
        raise ValueError(
            f"partition_by must be a column name or a mapping, not {type(value).__name__}."
        )

    column = str(value.get("column") or "").strip()
    if not column:
        raise ValueError("partition_by: `column` is missing.")

    mode = str(value.get("mode") or MODE_LIST).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"partition_by.mode {mode!r} is unknown. Allowed: {', '.join(VALID_MODES)}."
        )

    from dbextractors.core.config import is_truthy

    return PartitionSpec(
        column=column,
        mode=mode,
        default_partition=is_truthy(value.get("default_partition", True)),
    )


def from_settings(settings: dict) -> Optional[PartitionSpec]:
    """`PartitionSpec` from ``ctx.settings``, as a strategy sees it.

    ``settings`` is a flat dict (``dataclasses.asdict`` in the entry point), so the
    spec arrives here as a mapping and is reassembled. Strategies thereby stay
    dict-driven, as the frozen contract requires.
    """
    value = (settings or {}).get("partition_by")
    legacy = bool((settings or {}).get("partition_by_source"))
    return parse_spec(value, legacy_by_source=legacy)


# --- Names ------------------------------------------------------------------


def _slug(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(value).lower()).strip("_") or "x"


def _fit(target: TargetRef, suffix: str, seed: str) -> str:
    """``<table>__<suffix>``, shortened so that it stays unambiguous.

    PostgreSQL silently truncates anything longer than ``MAX_IDENTIFIER_LENGTH``, so
    two values with a long common prefix would end up in the same partition. A digest
    of the **source** value keeps them apart.
    """
    name = f"{target.table}__{suffix}"
    if len(name) > MAX_IDENTIFIER_LENGTH:
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: MAX_IDENTIFIER_LENGTH - 9]}_{digest}"
    return name


def period_start(spec: PartitionSpec, value: Any) -> _dt.date:
    """Start of the period the value falls into."""
    if isinstance(value, _dt.datetime):
        day = value.date()
    elif isinstance(value, _dt.date):
        day = value
    else:
        day = _dt.date.fromisoformat(str(value)[:10])

    if spec.mode == MODE_RANGE_DAY:
        return day
    if spec.mode == MODE_RANGE_MONTH:
        return day.replace(day=1)
    if spec.mode == MODE_RANGE_YEAR:
        return day.replace(month=1, day=1)
    raise ValueError(f"period_start makes no sense for mode {spec.mode!r}.")


def period_bounds(spec: PartitionSpec, value: Any) -> Tuple[_dt.date, _dt.date]:
    """``[from, to)`` of one period. The upper bound is **exclusive**, as PostgreSQL wants."""
    start = period_start(spec, value)
    if spec.mode == MODE_RANGE_DAY:
        return start, start + _dt.timedelta(days=1)
    if spec.mode == MODE_RANGE_MONTH:
        return start, (start.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
    return start, start.replace(year=start.year + 1)


def partition_suffix(spec: PartitionSpec, value: Any) -> str:
    """The partition name suffix for one value."""
    if not spec.is_range:
        return _slug(value)
    start = period_start(spec, value)
    if spec.mode == MODE_RANGE_DAY:
        return f"{start.year:04d}_{start.month:02d}_{start.day:02d}"
    if spec.mode == MODE_RANGE_MONTH:
        return f"{start.year:04d}_{start.month:02d}"
    return f"{start.year:04d}"


def partition_ref(target: TargetRef, spec: PartitionSpec, value: Any) -> TargetRef:
    """Address of the partition for one value. Deterministic."""
    from dbextractors.core.strategies.base import TargetRef as _TargetRef

    suffix = partition_suffix(spec, value)
    return _TargetRef(schema=target.schema, table=_fit(target, suffix, str(value)))


def default_ref(target: TargetRef) -> TargetRef:
    """Address of the catch-all ``DEFAULT`` partition."""
    from dbextractors.core.strategies.base import TargetRef as _TargetRef

    return _TargetRef(schema=target.schema, table=_fit(target, DEFAULT_SUFFIX, DEFAULT_SUFFIX))


# --- DDL --------------------------------------------------------------------


def create_parent(conn: Any, target: TargetRef, like: TargetRef, spec: PartitionSpec) -> None:
    """Creates the target as a partitioned parent, shaped like another table.

    A partitioned parent cannot be produced by renaming an ordinary table, so the
    shape is taken through ``LIKE`` from the staging table and the parent is created
    empty.
    """
    _assert_column(conn, like, spec)
    strategy = "LIST" if not spec.is_range else "RANGE"
    ensure_schema(conn, target.schema)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE {qualify(target)} (LIKE {qualify(like)} INCLUDING DEFAULTS) "
            f"PARTITION BY {strategy} ({quote_ident(spec.column)})"
        )
    _log().info(
        "📝 Target %s created as partitioned %s by %r.",
        qualify(target),
        strategy,
        spec.column,
    )


def ensure_default_partition(
    conn: Any, target: TargetRef, spec: PartitionSpec
) -> Optional[TargetRef]:
    """Creates the catch-all ``DEFAULT`` partition if the spec asks for it.

    Without it, a single row that fits no partition brings the run down — including
    a ``NULL`` in the partition key, which belongs to no range.
    """
    if not spec.default_partition:
        return None
    part = default_ref(target)
    if _partition_exists(conn, target, part):
        return part
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {qualify(part)} PARTITION OF {qualify(target)} DEFAULT")
    _log().info("📝 Catch-all partition %s created.", qualify(part))
    return part


def ensure_partition_for(
    conn: Any, target: TargetRef, spec: PartitionSpec, value: Any
) -> TargetRef:
    """Ensures the partition for one value exists and returns its address.

    This is what makes a new source or a new month self-service: it appears in the
    data and the partition is created here, with no manual DDL.

    ``IF NOT EXISTS`` would not be enough. Renaming a partitioned table leaves its
    partitions under their original names, still attached to the renamed parent;
    ``CREATE`` would then be a silent no-op and the write would fail with "no
    partition found for row". That is why **whose** partition it is gets checked too.
    """
    part = partition_ref(target, spec, value)
    if _partition_exists(conn, target, part):
        return part

    if spec.is_range:
        start, end = period_bounds(spec, value)
        sql = (
            f"CREATE TABLE {qualify(part)} PARTITION OF {qualify(target)} "
            f"FOR VALUES FROM (%s) TO (%s)"
        )
        params: Tuple[Any, ...] = (start, end)
    else:
        sql = f"CREATE TABLE {qualify(part)} PARTITION OF {qualify(target)} FOR VALUES IN (%s)"
        params = (value,)

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    except Exception as err:
        raise TargetError(
            f"Partition {qualify(part)} could not be created ({err}). The most common "
            f"cause: {qualify(target)} is partitioned differently from what the current "
            f"partition_by says (mode {spec.mode!r} over {spec.column!r}). Changing the "
            "partitioning of an existing table is a manual migration; a run will not do it."
        ) from err

    _log().info("📝 Partition %s created for %r.", qualify(part), value)
    return part


def ensure_partitions_from_table(
    conn: Any, target: TargetRef, spec: PartitionSpec, source_table: TargetRef
) -> List[TargetRef]:
    """Creates partitions for every value present in the staging table.

    It goes by ``SELECT DISTINCT``, not by the min–max range: one stray date from
    1900 would otherwise mean a thousand empty partitions.
    """
    ensure_default_partition(conn, target, spec)

    column = quote_ident(spec.column)
    if spec.is_range:
        expr = f"date_trunc('{_TRUNC_UNIT[spec.mode]}', {column}::timestamp)::date"
    else:
        expr = column

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT {expr} FROM {qualify(source_table)} WHERE {column} IS NOT NULL"
        )
        values = [r[0] for r in cur.fetchall()]

    return [ensure_partition_for(conn, target, spec, v) for v in values if v is not None]


def partition_key(conn: Any, target: TargetRef) -> Optional[str]:
    """What the target is really partitioned by, or ``None`` for an ordinary table.

    Takes only the first column of the key — the package never creates more than one.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.attname FROM pg_partitioned_table pt "
            "JOIN pg_class c ON c.oid = pt.partrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            # `partattrs` is an int2vector, and those are indexed **from zero**.
            "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = pt.partattrs[0] "
            "WHERE n.nspname = %s AND c.relname = %s",
            (target.schema, target.table),
        )
        row = cur.fetchone()
    return row[0] if row else None


def effective_spec(
    conn: Any, target: TargetRef, spec: Optional[PartitionSpec], log: Any = None
) -> Optional[PartitionSpec]:
    """The partitioning that actually applies. For an existing table, the table decides.

    Three cases:

    * the target does not exist → the configuration applies and the table is created
      partitioned;
    * the target is partitioned by the same column → it applies;
    * the target is an ordinary table → it **does not apply**, it is only logged.
      Converting means ``DROP`` + ``CREATE`` and views hang off the targets; that is
      a manual migration.

    Partitioning by a **different** column fails loudly: silently writing into a table
    partitioned differently from what the configuration says is worse than failing.
    """
    from dbextractors.core.target_pg import table_exists

    if spec is None:
        return None
    if not table_exists(conn, target):
        return spec

    actual = partition_key(conn, target)
    if actual is None:
        if log:
            log(
                "warning",
                "⚠️ partition_by is set (%s), but %s is an ordinary table — "
                "partitioning will not be applied. Converting is a manual migration "
                "(DROP + CREATE); a run will not do it.",
                spec.column,
                qualify(target),
            )
        return None

    if actual != spec.column:
        raise TargetError(
            f"{qualify(target)} is partitioned by {actual!r}, but partition_by says "
            f"{spec.column!r}. The write would go into a differently partitioned table. "
            "Align the configuration with the target, or rebuild the table by hand."
        )
    return spec


def unique_index_columns(spec: Optional[PartitionSpec], pk: str) -> List[str]:
    """Columns of the unique index over a (possibly partitioned) target.

    PostgreSQL requires a unique index on a partitioned table to contain the
    partition key. Without a `spec`, the key alone remains.
    """
    if spec is None or spec.column == pk:
        return [pk]
    return [pk, spec.column]


# --- Helpers ----------------------------------------------------------------


def _log():
    import logging

    return logging.getLogger("dbextractors.core.partitioning")


def _partition_exists(conn: Any, target: TargetRef, part: TargetRef) -> bool:
    """Does ``part`` exist — and is it really a partition of **our** parent?"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.relname FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "LEFT JOIN pg_inherits i ON i.inhrelid = c.oid "
            "LEFT JOIN pg_class p ON p.oid = i.inhparent "
            "WHERE n.nspname = %s AND c.relname = %s",
            (part.schema, part.table),
        )
        row = cur.fetchone()

    if row is None:
        return False
    if row[0] != target.table:
        what = f"a partition of table {row[0]!r}" if row[0] else "an ordinary table"
        raise TargetError(
            f"{qualify(part)} already exists and it is {what}. Rename it or drop it "
            f"before {qualify(target)} is built again."
        )
    return True


def _assert_column(conn: Any, like: TargetRef, spec: PartitionSpec) -> None:
    """The partition key must exist in the table — otherwise the configuration is wrong.

    It fails loudly and immediately. Left to PostgreSQL, the same message would only
    arrive after staging had been filled, i.e. after the whole source was downloaded.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
            (like.schema, like.table, spec.column),
        )
        if cur.fetchone() is None:
            raise TargetError(
                f"partition_by.column {spec.column!r} is not in the target table. "
                "Check the name (in the target it has been through `core.naming` and "
                "may carry an underscore) and whether the column selection excluded it."
            )


__all__ = [
    "MODE_LIST",
    "MODE_RANGE_DAY",
    "MODE_RANGE_MONTH",
    "MODE_RANGE_YEAR",
    "RANGE_MODES",
    "VALID_MODES",
    "PartitionSpec",
    "parse_spec",
    "from_settings",
    "period_start",
    "period_bounds",
    "partition_suffix",
    "partition_ref",
    "default_ref",
    "create_parent",
    "ensure_default_partition",
    "ensure_partition_for",
    "ensure_partitions_from_table",
    "partition_key",
    "effective_spec",
    "unique_index_columns",
]
