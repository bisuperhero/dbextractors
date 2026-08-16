"""Every strategy must be able to write into a target created by an older extractor.

This test exists because the same bug turned up **three times in a row** in different
strategies — `full`, `parent_incremental`, `incremental` — and every time it was a run
against a real source that found it, not the tests. The cause is shared: the staging
(shadow) table is created with ``LIKE <target>``, so it inherits whatever the target is
missing, and ``COPY`` of the first batch fails on::

    column "row_hash" of relation "dbx_shadow_..." does not exist

The columns therefore have to be added **before** the staging table is created. Tests
written one strategy at a time did not catch it, because each new strategy started out
without that test.

**The parametrisation is taken from the registry** (`strategies.base.STRATEGIES`), not
from a list kept here by hand. The hand-written list claimed completeness while holding
three of the six strategies; now a strategy that is added to the registry and not to
`SETTINGS` fails `test_the_settings_cover_the_whole_registry` instead of quietly not
being covered.

This is not an edge case but the **starting state of the migration**: variant B's
Firebird extractor has no `_deleted_in_source` at all, and five more predecessor files
only declare it, so the first run of dbextractors over such a table would fail.
"""

from __future__ import annotations

import importlib
from datetime import date, timedelta

import pandas as pd
import pytest

from dbextractors.core.strategies.base import STRATEGIES
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("parent_id", "int", "INTEGER"),
    ("changed_at", "date", "DATE"),
    ("descr", "varchar", "TEXT"),
]

TODAY = date.today()
IN_WINDOW = TODAY - timedelta(days=1)


def _strategy_classes() -> dict:
    """Every class the registry can hand out, keyed by its own ``name``.

    The registry maps `load_method` to ``module:Class`` and two of its keys
    (`hash`, `hash_diff`) point at the same class, so it is deduplicated by class
    — running the same one twice would only look like wider coverage.
    """
    classes = {}
    for path in set(STRATEGIES.values()):
        module_name, _, class_name = path.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        classes[cls.name] = cls
    return classes


STRATEGY_CLASSES = _strategy_classes()

#: What each strategy needs in order to reach its own staging path at all:
#: `id_watermark` a watermark column, `parent_incremental` a parent, `incremental`
#: a date column. `full`, `hash_diff` and `full_by_source` need nothing beyond the
#: key.
SETTINGS = {
    "full": {"primary_column": "id"},
    "hash_diff": {"primary_column": "id"},
    "full_by_source": {"primary_column": "id"},
    "incremental": {"primary_column": "id", "updated_at_column": "changed_at", "days_back": 7},
    "id_watermark": {"primary_column": "id", "primary_key_column": "id", "resume_full_load": True},
    "parent_incremental": {
        "primary_column": "id",
        "incremental_parent_table": "orders",
        "incremental_parent_date_column": "changed_at",
        "incremental_lookback_hours": 7 * 24,
    },
}


def test_the_settings_cover_the_whole_registry() -> None:
    """A new strategy has to be given settings here, or this fails.

    The point of the whole file: the bug it guards against turned up three times
    in a row, each time in a strategy that had been written after the test.
    """
    assert set(SETTINGS) == set(STRATEGY_CLASSES), (
        "the registry and the settings table have drifted apart — add the new strategy "
        "to SETTINGS so that it is covered too"
    )


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "parent_id": [10, 10, 10],
            "changed_at": [IN_WINDOW, IN_WINDOW, IN_WINDOW],
            "descr": ["a", "b", "c"],
        }
    )


def _legacy_target(conn, schema: str) -> None:
    """The target exactly as the old side left it: the source's columns and nothing else.

    No `row_hash`, no `_deleted_in_source` — for variant B's Firebird tables this is the
    real starting state, not an invented case.
    """
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{schema}"."cil" '
            "(id INTEGER, parent_id INTEGER, changed_at DATE, descr TEXT)"
        )
        for row in ((1, 10, IN_WINDOW, "original"), (2, 10, IN_WINDOW, "original")):
            cur.execute(f'INSERT INTO "{schema}"."cil" VALUES (%s, %s, %s, %s)', row)
        # A unique index over the PK is a precondition of the upsert and real targets
        # have one — what is missing here are the mandatory columns, not the index.
        cur.execute(f'CREATE UNIQUE INDEX idx_cil_id_unique ON "{schema}"."cil" (id)')
        cur.execute(f'CREATE TABLE "{schema}"."orders" (id integer, changed_at date)')
        cur.execute(f'INSERT INTO "{schema}"."orders" VALUES (10, %s)', (IN_WINDOW,))
    conn.commit()


@pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
def test_legacy_target_without_mandatory_columns(conn, schema, name) -> None:
    cls = STRATEGY_CLASSES[name]
    settings = SETTINGS[name]
    _legacy_target(conn, schema)
    source = FakeHashSource(
        _source_rows(),
        pk="id",
        parents=pd.DataFrame({"id": [10], "changed_at": [IN_WINDOW]}),
    )
    ctx = make_context(
        conn,
        schema,
        dialect=source,
        columns=make_columns(COLUMNS),
        settings=settings,
    )

    result = cls().run(ctx)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'cil'",
            (schema,),
        )
        columns = {r[0] for r in cur.fetchall()}

    assert "row_hash" in columns, f"{name}: `row_hash` has to be added before staging"
    assert (
        "_deleted_in_source" in columns
    ), f"{name}: `_deleted_in_source` is mandatory for every strategy"
    assert result.rows_written > 0, f"{name}: not a single row was written"
