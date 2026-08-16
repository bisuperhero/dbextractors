"""Golden test of partitioning: a partitioned target has to give **the same
thing** as a plain one.

For the other features the golden test compares the old component against the
new one. That cannot be done here — partitioning is a new capability the old side
does not have. The meaningful question is therefore a different one, but just as
hard: *does anything about the data change when the same table is loaded once
plainly and once split up?*

All five levels are compared — row count, **column names and order**, types,
per-column checksums and `row_hash` per row — with the same comparator the real
golden run uses.
"""

from __future__ import annotations

from datetime import date
from typing import Iterator

import pandas as pd
import pytest

from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.golden import compare as compare_mod
from dbextractors.golden.model import Relation
from fakes import FakeDialect, make_columns, make_context

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("label", "varchar", "TEXT"),
    ("issued_on", "date", "DATE"),
    ("price", "decimal", "NUMERIC"),
]

#: Data deliberately crossing both a month and a year boundary, and with a NULL
#: in the partition key — that is, all three paths by which a row lands in a
#: partition.
ROWS = [
    (1, date(2025, 12, 31)),
    (2, date(2026, 1, 1)),
    (3, date(2026, 1, 31)),
    (4, date(2026, 2, 1)),
    (5, None),
    (6, date(2026, 8, 14)),
]

# Both tests load real tables through `rw_conn_txn` and compare them with the
# real comparator, so the whole module needs a live PostgreSQL.
pytestmark = pytest.mark.needs_pg


@pytest.fixture()
def rw_conn_txn(dsn: str) -> Iterator:
    """A writable connection with autocommit **off**.

    The golden `rw_conn` has autocommit on, but the strategies refuse it — the
    shadow table swap rests on it being off.
    """
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.fixture()
def schema(rw_conn_txn) -> Iterator[str]:
    from dbextractors.golden import scratch

    name = scratch.scratch_schema_name("partition_golden")
    scratch.create_schema(rw_conn_txn, name)
    rw_conn_txn.commit()
    try:
        yield name
    finally:
        rw_conn_txn.rollback()
        scratch.drop_schema(rw_conn_txn, name)
        rw_conn_txn.commit()


def _batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [r[0] for r in ROWS],
            "label": [f"row {r[0]}" for r in ROWS],
            "issued_on": [r[1] for r in ROWS],
            "price": [round(1.5 * r[0], 2) for r in ROWS],
        }
    )


def _load(conn, schema: str, table: str, partition_by) -> None:
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([_batch()]),
        columns=make_columns(COLUMNS),
        settings={"primary_column": "id", "partition_by": partition_by},
        table=table,
    )
    FullLoadStrategy().run(ctx)


def test_a_partitioned_target_gives_the_same_as_a_plain_one(rw_conn_txn, ro_conn, schema: str):
    _load(rw_conn_txn, schema, "plain", None)
    _load(rw_conn_txn, schema, "split", {"column": "issued_on", "mode": "range_month"})
    rw_conn_txn.commit()

    report = compare_mod.compare_tables(
        ro_conn,
        Relation(schema=schema, table="plain"),
        Relation(schema=schema, table="split"),
        label="partitioning: plain vs. split target",
    )

    assert report.first_failed_level is None, "\n".join(
        f"{level.level}: {'OK' if level.passed else 'DIFF'} {level.notes}"
        for level in report.levels
    )
    assert report.error is None
    # A safeguard that the comparison really ran rather than being skipped.
    assert not any(level.skipped for level in report.levels)
    assert len(report.levels) == 5


def test_the_split_target_really_is_split(rw_conn_txn, schema: str) -> None:
    """Were the partitioning silently not created, the test above would pass and
    mean nothing."""
    from dbextractors.core import partitioning, target_pg
    from dbextractors.core.strategies.base import TargetRef

    _load(rw_conn_txn, schema, "split", {"column": "issued_on", "mode": "range_month"})
    rw_conn_txn.commit()

    target = TargetRef(schema=schema, table="split")
    assert target_pg.is_partitioned(rw_conn_txn, target) is True
    assert partitioning.partition_key(rw_conn_txn, target) == "issued_on"

    with rw_conn_txn.cursor() as cur:
        cur.execute(
            "SELECT c.relname FROM pg_inherits i "
            "JOIN pg_class c ON c.oid = i.inhrelid "
            "JOIN pg_class p ON p.oid = i.inhparent "
            "JOIN pg_namespace n ON n.oid = p.relnamespace "
            "WHERE n.nspname = %s AND p.relname = 'split' ORDER BY 1",
            (schema,),
        )
        assert [r[0] for r in cur.fetchall()] == [
            "split__2025_12",
            "split__2026_01",
            "split__2026_02",
            "split__2026_08",
            "split__default",
        ]
