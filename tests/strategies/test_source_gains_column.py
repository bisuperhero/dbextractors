"""The source gained a column the target does not have. What each strategy does about it.

Seen in production on 14 Aug 2026::

    column "promo_code" of relation "dbx_shadow_<table>_..."
    does not exist

The mechanism is the same one as for missing managed columns
(`test_legacy_target_all_strategies.py`): the staging table is created with
``LIKE <target>``, so it inherits the target's shape, and ``COPY`` fails on the
column the source sends on top of it. What differs is the **correct reaction**,
which is why this lives in its own file.

The predecessor aligned each batch to the target's columns and dropped the extra
column **silently**. The decision here:

- **full load** **adopts** the column — the table is rewritten in full anyway, so
  widening the shape costs nothing and from the next run onwards the target matches;
- **every other strategy** **drops** it, because they touch existing data and are not
  supposed to change the target's shape — but they drop it **loudly**, not silently
  as before.

The test is parametrised over the list of strategies for the same reason as its
neighbour: so that the next addition fails rather than being forgotten.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import pytest

from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.hash_diff import HashDiffStrategy
from dbextractors.core.strategies.id_watermark import IdWatermarkStrategy
from dbextractors.core.strategies.incremental import IncrementalStrategy
from dbextractors.core.strategies.parent_incremental import ParentIncrementalStrategy
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

#: The last column is the one the target **does not** have — in production it was
#: a promo code column on an order-items table.
COLUMNS = [
    ("id", "int", "INTEGER"),
    ("parent_id", "int", "INTEGER"),
    ("changed_at", "date", "DATE"),
    ("descr", "varchar", "TEXT"),
    ("promo_code", "varchar", "TEXT"),
]
NEW_COLUMN = "promo_code"

TODAY = date.today()
IN_WINDOW = TODAY - timedelta(days=1)

#: Strategies that touch existing data — they drop the extra column.
DROPPING = [
    (
        "incremental",
        IncrementalStrategy,
        {"primary_column": "id", "updated_at_column": "changed_at", "days_back": 7},
    ),
    (
        "id_watermark",
        IdWatermarkStrategy,
        {"primary_column": "id", "primary_key_column": "id", "resume_full_load": True},
    ),
    (
        "parent_incremental",
        ParentIncrementalStrategy,
        {
            "primary_column": "id",
            "incremental_parent_table": "orders",
            "incremental_parent_date_column": "changed_at",
            "incremental_lookback_hours": 7 * 24,
        },
    ),
    (
        "hash_diff",
        HashDiffStrategy,
        {"primary_column": "id", "hash_column": "row_hash"},
    ),
]


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "parent_id": [10, 10, 10],
            "changed_at": [IN_WINDOW, IN_WINDOW, IN_WINDOW],
            "descr": ["a", "b", "c"],
            NEW_COLUMN: ["DISCOUNT10", "DISCOUNT20", None],
        }
    )


def _source_hashes() -> list:
    """Row hashes exactly as `FakeHashSource` computes them — that is, as the source does.

    ``fakes.source_row_hash``, not ``hashing.row_hash``: the fake's source-side
    hash is deliberately distinguishable from the pandas computation, and a
    target seeded here has to look like one seeded through the source path.
    """
    from fakes import source_row_hash

    source = _source_rows()
    columns = list(source.columns)
    return [source_row_hash([row[c] for c in columns]) for _, row in source.iterrows()]


def _target_without_new_column(conn, schema: str) -> None:
    """The target in the shape a run left it in before the source gained the column.

    It **does** have the managed columns — otherwise the test would be measuring the
    failure from the neighbouring file rather than this one.

    ``row_hash`` is filled with **the source's hashes**, not random strings. Either of
    those would push `hash_diff` onto a full load and the test would then measure full
    load behaviour under a false name: an empty column reads as "the hashes have to be
    seeded", nonsense values as "the hashes have drifted" (mismatch above
    `RESEED_MISMATCH_RATIO`). This way two rows are unchanged and one is new.
    """
    hashes = dict(zip(_source_rows()["id"], _source_hashes(), strict=False))
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{schema}"."cil" ('
            "id INTEGER, parent_id INTEGER, changed_at DATE, descr TEXT, "
            # `_timestamp` is TEXT because that is what `MANAGED_COLUMNS` makes it.
            # A target whose column is a real TIMESTAMP is a different story, covered
            # by `test_target_timestamp_type.py`.
            "row_hash TEXT, _timestamp TEXT, "
            "_deleted_in_source BOOLEAN NOT NULL DEFAULT false)"
        )
        for row in (
            (1, 10, IN_WINDOW, "original", hashes[1]),
            (2, 10, IN_WINDOW, "original", hashes[2]),
        ):
            cur.execute(
                f'INSERT INTO "{schema}"."cil" (id, parent_id, changed_at, descr, row_hash) '
                "VALUES (%s, %s, %s, %s, %s)",
                row,
            )
        cur.execute(f'CREATE UNIQUE INDEX idx_cil_id_unique ON "{schema}"."cil" (id)')
        cur.execute(f'CREATE TABLE "{schema}"."orders" (id integer, changed_at date)')
        cur.execute(f'INSERT INTO "{schema}"."orders" VALUES (10, %s)', (IN_WINDOW,))
    conn.commit()


def _target_columns(conn, schema: str) -> list:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'cil' ORDER BY ordinal_position",
            (schema,),
        )
        return [r[0] for r in cur.fetchall()]


def _context(conn, schema, settings):
    source = FakeHashSource(
        _source_rows(),
        pk="id",
        parents=pd.DataFrame({"id": [10], "changed_at": [IN_WINDOW]}),
    )
    return make_context(
        conn,
        schema,
        dialect=source,
        columns=make_columns(COLUMNS),
        settings=settings,
    )


@pytest.mark.parametrize("name,cls,settings", DROPPING, ids=[s[0] for s in DROPPING])
def test_extra_column_is_dropped_and_the_run_continues(
    conn, schema, caplog, name, cls, settings
) -> None:
    """The run must not fail and the target's shape must not change."""
    _target_without_new_column(conn, schema)
    ctx = _context(conn, schema, settings)

    with caplog.at_level(logging.WARNING):
        result = cls().run(ctx)

    assert NEW_COLUMN not in _target_columns(
        conn, schema
    ), f"{name}: a strategy that touches existing data must not change the target's shape"
    assert result.rows_written > 0, f"{name}: not a single row was written"


@pytest.mark.parametrize("name,cls,settings", DROPPING, ids=[s[0] for s in DROPPING])
def test_dropping_a_column_is_visible_in_the_log(conn, schema, caplog, name, cls, settings) -> None:
    """Dropping silently is exactly what used to hide a hole in the data.

    The message has to name the **specific** column — "something was dropped" helps
    nobody work out what is missing in the warehouse.
    """
    _target_without_new_column(conn, schema)
    ctx = _context(conn, schema, settings)

    with caplog.at_level(logging.WARNING):
        cls().run(ctx)

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        NEW_COLUMN in m for m in messages
    ), f"{name}: dropping column {NEW_COLUMN!r} is not visible in the log. Messages: {messages}"


def test_full_load_adopts_the_column(conn, schema) -> None:
    """Full load rewrites the whole table, so it takes the extra column with it."""
    _target_without_new_column(conn, schema)
    ctx = _context(conn, schema, {"primary_column": "id"})

    result = FullLoadStrategy().run(ctx)

    columns = _target_columns(conn, schema)
    assert NEW_COLUMN in columns, "full load is meant to adopt the source column, not drop it"
    assert result.rows_written == 3

    with conn.cursor() as cur:
        cur.execute(f'SELECT {NEW_COLUMN} FROM "{schema}"."cil" ORDER BY id')
        values = [r[0] for r in cur.fetchall()]
    assert values == [
        "DISCOUNT10",
        "DISCOUNT20",
        None,
    ], "the column was added but holds no data — the swap dropped it"


def test_full_load_does_not_reorder_the_original_columns(conn, schema) -> None:
    """An added column goes **at the end**. Column order is part of the frozen contract."""
    _target_without_new_column(conn, schema)
    before = _target_columns(conn, schema)
    ctx = _context(conn, schema, {"primary_column": "id"})

    FullLoadStrategy().run(ctx)

    after = _target_columns(conn, schema)
    assert (
        after[: len(before)] == before
    ), f"the order of the original columns changed: {before} -> {after}"
    assert after[len(before) :] == [NEW_COLUMN]
