"""Building the aggregate expressions for level 4 — per-column checksums.

Three requirements shape those expressions:

1. **Nothing may be pulled into memory.** Tables reach 15 M+ rows. Everything is
   computed by aggregates in SQL; one row per table comes back to Python.
2. **It has to be deterministic** — independent of row order and of floating
   point, and it has to tell ``NULL`` apart from the empty string.
3. **Where the comparison is necessarily approximate, the report has to say so**
   rather than let it pass silently.

## How order independence is achieved

For text and other non-aggregable types, ``sum`` over 32-bit slices of MD5 is
used. Addition is commutative, so row order does not matter. Verified on
PG 17.9: ``('a','b','c')`` and ``('c','a','b')`` both give 3929464487.

**Two independent slices** of the same MD5 are taken (characters 1–8 and 9–16).
A single 32-bit sum would collide far too readily over millions of rows; two
slices together with the counts and with ``min``/``max`` are enough.

## How floating point is handled

``sum(x)`` over ``double precision`` depends on the order of addition.
``sum(x::numeric)`` does not — ``numeric`` adds exactly. Verified:
``0.1+0.2+0.3`` gives 0.6000000000000001 as ``float8``, and exactly 0.6 through
``::numeric``.

``NaN`` and ``Infinity`` are kept out of the sum. PostgreSQL 17 does support
them in ``numeric``, but a single ``NaN`` would swallow the whole sum and throw
away every bit of information about the rest of the column. They are counted
separately instead.

## How NULL is handled

``count(*)`` and ``count(column)`` differ by exactly the number of ``NULL``s. An
empty string counts towards ``count(column)``, a ``NULL`` does not — so the two
are told apart. Verified: over ``('a', '', NULL)`` the result is ``count(v)=2``,
``count(*)=3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Collection


class Family(str, Enum):
    """Class of a type. Decides which aggregates make sense."""

    NUMERIC_EXACT = "numeric_exact"
    NUMERIC_FLOAT = "numeric_float"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"
    TEXTUAL = "textual"
    BINARY = "binary"
    JSON_CANONICAL = "json_canonical"
    JSON_LOOSE = "json_loose"
    OTHER = "other"


#: Map keyed by `information_schema.columns.data_type` / `udt_name`.
_FAMILY_BY_TYPE: dict[str, Family] = {
    "smallint": Family.NUMERIC_EXACT,
    "integer": Family.NUMERIC_EXACT,
    "bigint": Family.NUMERIC_EXACT,
    "numeric": Family.NUMERIC_EXACT,
    "decimal": Family.NUMERIC_EXACT,
    "money": Family.NUMERIC_EXACT,
    "real": Family.NUMERIC_FLOAT,
    "double precision": Family.NUMERIC_FLOAT,
    "boolean": Family.BOOLEAN,
    "date": Family.TEMPORAL,
    "time without time zone": Family.TEMPORAL,
    "time with time zone": Family.TEMPORAL,
    "timestamp without time zone": Family.TEMPORAL,
    "timestamp with time zone": Family.TEMPORAL,
    "interval": Family.TEMPORAL,
    "text": Family.TEXTUAL,
    "character varying": Family.TEXTUAL,
    "character": Family.TEXTUAL,
    "uuid": Family.TEXTUAL,
    "name": Family.TEXTUAL,
    "citext": Family.TEXTUAL,
    "xml": Family.TEXTUAL,
    "bytea": Family.BINARY,
    "jsonb": Family.JSON_CANONICAL,
    "json": Family.JSON_LOOSE,
}

#: Families whose comparison is necessarily approximate. The report must say so.
APPROXIMATE_FAMILIES: frozenset[Family] = frozenset({Family.JSON_LOOSE, Family.OTHER})

APPROXIMATION_NOTES: dict[Family, str] = {
    Family.JSON_LOOSE: (
        "the `json` type (unlike `jsonb`) remembers the original spelling including "
        "whitespace and key order, so two semantically equal values hash differently — "
        "the comparison is approximate"
    ),
    Family.OTHER: (
        "an unknown or composite type (array, composite, enum) is compared through its "
        "textual representation — the comparison is approximate"
    ),
}


def classify(data_type: str, udt_name: str = "") -> Family:
    """Puts a PostgreSQL type into a family."""
    normalized = (data_type or "").strip().lower()
    if normalized in _FAMILY_BY_TYPE:
        return _FAMILY_BY_TYPE[normalized]
    # `information_schema` reports arrays as 'ARRAY' with udt_name '_text' etc.
    if normalized == "array":
        return Family.OTHER
    if normalized == "user-defined":
        return _FAMILY_BY_TYPE.get((udt_name or "").strip().lower(), Family.OTHER)
    return Family.OTHER


def quote_ident(name: str) -> str:
    """Quotes an identifier for PostgreSQL.

    Doubling the quotes inside the name matters: target tables have columns such
    as ``_type`` or ``_order``, and even more exotic names cannot be ruled out.
    """
    return '"' + name.replace('"', '""') + '"'


def _hash_slices(expr: str) -> tuple[str, str]:
    """Two independent 32-bit slices of MD5, summed over all rows."""
    # The trailing `::text` matters: in PostgreSQL, sum() over bigint returns
    # `numeric`, which psycopg2 hands back as `Decimal` — and that is not JSON
    # serialisable. It would only show up when the report is written, that is at
    # the end of a batch of over 250 tables. Text is also consistent with the
    # rest of the metrics.
    return (
        f"sum(('x' || substr(md5({expr}), 1, 8))::bit(32)::bigint)::text",
        f"sum(('x' || substr(md5({expr}), 9, 8))::bit(32)::bigint)::text",
    )


@dataclass
class ColumnAggregates:
    """Aggregates for one column: alias -> SQL expression."""

    column: str
    family: Family
    expressions: dict[str, str] = field(default_factory=dict)
    approximate: bool = False
    note: str = ""


def build_column_aggregates(
    column: str, family: Family, *, cast_to_text: bool = False
) -> ColumnAggregates:
    """Builds the aggregate expressions for one column from its family.

    ``cast_to_text`` is for columns that have a **different type** on each side
    and are compared through their textual form
    (`CompareOptions.compare_type_mismatch_as_text`). The family is then forced
    to `TEXTUAL`, because the original one cannot be relied on — the two sides
    disagree about it. The cast is applied to the column itself rather than
    inside the individual expressions: `sum(length(col))` would fail over
    `integer`.
    """
    col = quote_ident(column)
    if cast_to_text:
        col = f"({col})::text"
        family = Family.TEXTUAL
    agg = ColumnAggregates(column=column, family=family)
    exprs = agg.expressions

    # Common to every family. `count(*) - count(col)` tells NULL apart from the
    # empty string.
    exprs["nulls"] = f"count(*) - count({col})"

    if family is Family.NUMERIC_EXACT:
        exprs["sum"] = f"sum({col}::numeric)::text"
        exprs["min"] = f"min({col})::text"
        exprs["max"] = f"max({col})::text"

    elif family is Family.NUMERIC_FLOAT:
        # NaN and Infinity are kept out of the sum — a single NaN would swallow
        # it and throw away the information about the whole rest of the column.
        finite = (
            f"{col} = {col} AND {col} <> 'Infinity'::float8 " f"AND {col} <> '-Infinity'::float8"
        )
        exprs["sum"] = f"sum({col}::numeric) FILTER (WHERE {finite})::text"
        exprs["min"] = f"min({col}) FILTER (WHERE {finite})::text"
        exprs["max"] = f"max({col}) FILTER (WHERE {finite})::text"
        exprs["nonfinite"] = f"count(*) FILTER (WHERE {col} IS NOT NULL AND NOT ({finite}))"

    elif family is Family.BOOLEAN:
        exprs["true"] = f"count(*) FILTER (WHERE {col})"
        exprs["false"] = f"count(*) FILTER (WHERE NOT {col})"

    elif family is Family.TEMPORAL:
        exprs["min"] = f"min({col})::text"
        exprs["max"] = f"max({col})::text"
        exprs["h1"], exprs["h2"] = _hash_slices(f"{col}::text")

    elif family is Family.BINARY:
        # md5(bytea) is a built-in function, no need to go through ::text.
        exprs["h1"], exprs["h2"] = _hash_slices(f"encode({col}, 'hex')")
        exprs["bytes"] = f"sum(octet_length({col}))::text"

    else:
        # TEXTUAL, JSON_CANONICAL, JSON_LOOSE, OTHER
        as_text = col if cast_to_text else f"{col}::text"
        exprs["h1"], exprs["h2"] = _hash_slices(as_text)
        exprs["min"] = f"min({as_text})"
        exprs["max"] = f"max({as_text})"
        if family is Family.TEXTUAL:
            exprs["chars"] = f"sum(length({as_text}))::text"

    if family in APPROXIMATE_FAMILIES:
        agg.approximate = True
        agg.note = APPROXIMATION_NOTES[family]

    return agg


def build_checksum_query(
    relation_sql: str,
    columns: list[tuple[str, Family]],
    *,
    text_coerced: Collection[str] = (),
) -> tuple[str, list[ColumnAggregates]]:
    """One query that computes the aggregates of every column at once.

    Deliberately **one query per table**, not one per column: over a table with
    15 M rows every pass is expensive, and splitting it per column would need as
    many passes as there are columns.

    Returns:
        A pair (SQL, aggregate descriptions). The descriptions define the column
        order in the result.
    """
    coerced = set(text_coerced)
    aggregates = [
        build_column_aggregates(name, family, cast_to_text=name in coerced)
        for name, family in columns
    ]

    select_parts = ["count(*) AS row_count"]
    for index, agg in enumerate(aggregates):
        for metric, expression in agg.expressions.items():
            select_parts.append(f"{expression} AS c{index}__{metric}")

    sql = "SELECT " + ", ".join(select_parts) + f" FROM {relation_sql}"
    return sql, aggregates


def parse_checksum_row(row: tuple, aggregates: list[ColumnAggregates]) -> dict:
    """Takes the single-row result apart again, back into per-column values."""
    values: dict = {"row_count": row[0]}
    position = 1
    for index, agg in enumerate(aggregates):
        per_column: dict = {}
        for metric in agg.expressions:
            per_column[metric] = row[position]
            position += 1
        values[agg.column] = per_column
        del index
    return values
