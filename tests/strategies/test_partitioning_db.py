"""Target partitioning against a **live** PostgreSQL, through the real strategies.

A mock would only verify string assembly here. What matters — that a row really ends up
in the right partition, that a new month creates its own partition, and that an upsert
over a partitioned target does **not** duplicate a row whose partition value changes —
can only be verified against a database.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from dbextractors.core import partitioning, target_pg
from dbextractors.core.strategies.base import TargetRef
from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.incremental import IncrementalStrategy
from fakes import FakeDialect, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("label", "varchar", "TEXT"),
    ("doc_date", "date", "DATE"),
    ("date_modified", "timestamp", "TIMESTAMP"),
]

BY_MONTH = {
    "primary_column": "id",
    "partition_by": {"column": "doc_date", "mode": "range_month"},
}


def _batch(rows) -> pd.DataFrame:
    """``rows`` is a list of ``(id, doc_date)``; the rest is derived."""
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "label": [f"row {r[0]}" for r in rows],
            "doc_date": [r[1] for r in rows],
            "date_modified": [datetime(2026, 8, 11, 12, 0)] * len(rows),
        }
    )


def _ctx(conn, schema, batches, *, settings=None):
    return make_context(
        conn,
        schema,
        dialect=FakeDialect(batches),
        columns=make_columns(COLUMNS),
        settings={**BY_MONTH, **(settings or {})},
    )


def _target_ref(schema: str) -> TargetRef:
    return TargetRef(schema=schema, table="cil")


def _partitions(conn, schema: str) -> list:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname FROM pg_inherits i "
            "JOIN pg_class c ON c.oid = i.inhrelid "
            "JOIN pg_class p ON p.oid = i.inhparent "
            "JOIN pg_namespace n ON n.oid = p.relnamespace "
            "WHERE n.nspname = %s AND p.relname = 'cil' ORDER BY 1",
            (schema,),
        )
        return [r[0] for r in cur.fetchall()]


def _rows_with_partition(conn, schema: str) -> list:
    """``(id, partition name)`` — where the row actually lives."""
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, tableoid::regclass::text FROM "{schema}"."cil" ORDER BY id')
        return [(r[0], r[1].split(".")[-1].strip('"')) for r in cur.fetchall()]


# --- Creating the target ----------------------------------------------------


def test_without_a_key_the_target_stays_plain(conn, schema) -> None:
    """The default state. None of the ~670 existing tables may change shape."""
    ctx = _ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])], settings={"partition_by": None})
    FullLoadStrategy().run(ctx)

    assert target_pg.is_partitioned(conn, _target_ref(schema)) is False
    assert _partitions(conn, schema) == []


def test_full_load_creates_a_partitioned_target(conn, schema) -> None:
    ctx = _ctx(conn, schema, [_batch([(1, date(2026, 8, 4)), (2, date(2026, 9, 15))])])

    result = FullLoadStrategy().run(ctx)

    assert result.rows_written == 2
    assert target_pg.is_partitioned(conn, _target_ref(schema)) is True
    assert partitioning.partition_key(conn, _target_ref(schema)) == "doc_date"
    assert _partitions(conn, schema) == ["cil__2026_08", "cil__2026_09", "cil__default"]


def test_rows_land_in_the_right_partition(conn, schema) -> None:
    ctx = _ctx(
        conn,
        schema,
        [_batch([(1, date(2026, 8, 4)), (2, date(2026, 8, 31)), (3, date(2026, 9, 1))])],
    )
    FullLoadStrategy().run(ctx)

    assert _rows_with_partition(conn, schema) == [
        (1, "cil__2026_08"),
        (2, "cil__2026_08"),
        (3, "cil__2026_09"),
    ]


def test_a_new_month_creates_its_own_partition(conn, schema) -> None:
    """This is why partitions are created from the data rather than by hand-written DDL."""
    FullLoadStrategy().run(_ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])]))
    assert _partitions(conn, schema) == ["cil__2026_08", "cil__default"]

    FullLoadStrategy().run(_ctx(conn, schema, [_batch([(1, date(2026, 11, 4))])]))

    assert _partitions(conn, schema) == ["cil__2026_08", "cil__2026_11", "cil__default"]
    assert _rows_with_partition(conn, schema) == [(1, "cil__2026_11")]


def test_null_in_the_partition_key_lands_in_the_catch_all(conn, schema) -> None:
    """Without a catch-all partition this single row would bring the whole run down."""
    ctx = _ctx(conn, schema, [_batch([(1, date(2026, 8, 4)), (2, None)])])

    FullLoadStrategy().run(ctx)

    assert _rows_with_partition(conn, schema) == [(1, "cil__2026_08"), (2, "cil__default")]


def test_without_the_catch_all_a_null_brings_the_run_down(conn, schema) -> None:
    """Evidence that the catch-all partition is not decoration."""
    ctx = _ctx(
        conn,
        schema,
        [_batch([(1, None)])],
        settings={
            "partition_by": {
                "column": "doc_date",
                "mode": "range_month",
                "default_partition": False,
            }
        },
    )

    with pytest.raises(Exception, match="no partition of relation"):
        FullLoadStrategy().run(ctx)


def test_list_by_column_value(conn, schema) -> None:
    """The second basic case: a column with an enumerated set of values."""
    ctx = _ctx(
        conn,
        schema,
        [_batch([(1, date(2026, 8, 4)), (2, date(2026, 9, 4))])],
        settings={"partition_by": {"column": "label", "mode": "list"}},
    )
    FullLoadStrategy().run(ctx)

    assert partitioning.partition_key(conn, _target_ref(schema)) == "label"
    assert _partitions(conn, schema) == ["cil__default", "cil__row_1", "cil__row_2"]


# --- The existing table decides for itself ----------------------------------


def test_an_existing_plain_table_is_not_rebuilt(conn, schema) -> None:
    """The conversion is DROP + CREATE and targets have views on them — a run will not do it."""
    FullLoadStrategy().run(
        _ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])], settings={"partition_by": None})
    )
    assert target_pg.is_partitioned(conn, _target_ref(schema)) is False

    result = FullLoadStrategy().run(_ctx(conn, schema, [_batch([(2, date(2026, 8, 4))])]))

    assert result.rows_written == 1, "the run must go through, only without partitioning"
    assert target_pg.is_partitioned(conn, _target_ref(schema)) is False


def test_partitioning_by_a_different_column_fails(conn, schema) -> None:
    """Silently writing into a differently partitioned table is worse than failing."""
    FullLoadStrategy().run(_ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])]))

    ctx = _ctx(
        conn,
        schema,
        [_batch([(2, date(2026, 8, 4))])],
        settings={"partition_by": {"column": "label", "mode": "list"}},
    )
    with pytest.raises(target_pg.TargetError, match="is partitioned by"):
        FullLoadStrategy().run(ctx)


def test_an_unknown_column_fails_loudly(conn, schema) -> None:
    ctx = _ctx(
        conn,
        schema,
        [_batch([(1, date(2026, 8, 4))])],
        settings={"partition_by": {"column": "no_such_column", "mode": "list"}},
    )
    with pytest.raises(target_pg.TargetError, match="is not in the target table"):
        FullLoadStrategy().run(ctx)


# --- Indexes ----------------------------------------------------------------


def _indexes(conn, schema: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname, i.indisunique, "
            "  array_agg(a.attname ORDER BY k.ord) "
            "FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_class t ON t.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(att, ord) ON true "
            "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.att "
            "WHERE n.nspname = %s AND t.relname = 'cil' "
            "GROUP BY c.relname, i.indisunique",
            (schema,),
        )
        return {r[0]: (r[1], r[2]) for r in cur.fetchall()}


def test_the_unique_index_contains_the_partition_key(conn, schema) -> None:
    """Otherwise PostgreSQL will not allow a unique index over a partitioned table."""
    FullLoadStrategy().run(_ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])]))

    indexes = _indexes(conn, schema)
    unique = [columns for is_unique, columns in indexes.values() if is_unique]
    assert unique == [["id", "doc_date"]]


def test_a_partitioned_target_also_has_a_plain_index_over_the_key(conn, schema) -> None:
    """Deletion in `replace_by_key` leans on it — without it that is a seq scan."""
    FullLoadStrategy().run(_ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])]))

    indexes = _indexes(conn, schema)
    plain = [columns for is_unique, columns in indexes.values() if not is_unique]
    assert ["id"] in plain


# --- Upsert over a partitioned target ---------------------------------------


INCREMENTAL_SETTINGS = {
    **BY_MONTH,
    "updated_at_column": "date_modified",
    "days_back": 3650,
}


def test_upsert_does_not_duplicate_a_row_with_a_new_partition_value(conn, schema) -> None:
    """**This is the trap that keeps a partitioned target off ON CONFLICT.**

    If the conflict key were merely widened to ``(id, doc_date)``, a row whose date
    changes in the source would not be updated but duplicated — the old one would stay
    in the old partition and dbt would see two.
    """
    FullLoadStrategy().run(
        _ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])], settings=INCREMENTAL_SETTINGS)
    )
    assert _rows_with_partition(conn, schema) == [(1, "cil__2026_08")]

    # The same key, but the date moved into a different month.
    moved = _batch([(1, date(2026, 9, 20))])
    IncrementalStrategy().run(_ctx(conn, schema, [moved], settings=INCREMENTAL_SETTINGS))

    assert _rows_with_partition(conn, schema) == [(1, "cil__2026_09")], "the row must move"

    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{schema}"."cil" WHERE id = 1')
        assert cur.fetchone()[0] == 1, "duplicate left behind in the old partition"


def test_upsert_updates_a_row_within_the_same_partition(conn, schema) -> None:
    FullLoadStrategy().run(
        _ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])], settings=INCREMENTAL_SETTINGS)
    )

    change = _batch([(1, date(2026, 8, 20))])
    change.loc[0, "label"] = "after change"
    IncrementalStrategy().run(_ctx(conn, schema, [change], settings=INCREMENTAL_SETTINGS))

    with conn.cursor() as cur:
        cur.execute(f'SELECT id, label, doc_date FROM "{schema}"."cil"')
        assert cur.fetchall() == [(1, "after change", date(2026, 8, 20))]


def test_upsert_adds_a_new_row_and_a_new_partition(conn, schema) -> None:
    FullLoadStrategy().run(
        _ctx(conn, schema, [_batch([(1, date(2026, 8, 4))])], settings=INCREMENTAL_SETTINGS)
    )

    IncrementalStrategy().run(
        _ctx(conn, schema, [_batch([(2, date(2027, 1, 9))])], settings=INCREMENTAL_SETTINGS)
    )

    assert _rows_with_partition(conn, schema) == [(1, "cil__2026_08"), (2, "cil__2027_01")]
