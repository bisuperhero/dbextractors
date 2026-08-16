"""Tests for `IncrementalStrategy` against a live target PostgreSQL.

The most valuable one here is `test_window_never_sees_a_change_older_than_days_back` —
it demonstrates a data gap the current behaviour can produce and that shows up
nowhere.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from dbextractors.core.strategies.base import StrategyError
from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.incremental import (
    DEFAULT_DAYS_BACK,
    IncrementalStrategy,
    compute_incremental_cutoff,
)
from fakes import FakeDialect, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("name", "varchar", "TEXT"),
    ("date_modified", "timestamp", "TIMESTAMP"),
]

SETTINGS = {
    "primary_column": "id",
    "updated_at_column": "date_modified",
    "created_at_column": "date_created",
    "days_back": 7,
}


def _batch(ids, when=None) -> pd.DataFrame:
    when = when or datetime(2026, 8, 11, 12, 0, 0)
    return pd.DataFrame(
        {
            "id": list(ids),
            "name": [f"row {i}" for i in ids],
            "date_modified": [when] * len(list(ids)),
        }
    )


def _rows(conn, schema: str) -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_name" FROM "{schema}"."cil" ORDER BY id')
        return cur.fetchall()


def _fill_target(conn, schema, ids, columns=None) -> None:
    """A full load, so the incremental run has something to build on.

    ``columns`` has to match what the incremental context is later given — the
    staging table is created with ``LIKE <target>``, so an extra column would have
    nowhere to be written.
    """
    columns = columns or COLUMNS
    batch = _batch(ids)
    for name, _typ, _pg in columns:
        if name not in batch.columns:
            batch[name] = None
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([batch]),
        columns=make_columns(columns),
        settings={"primary_column": "id"},
    )
    FullLoadStrategy().run(ctx)


def test_increment_updates_and_adds(conn, schema) -> None:
    _fill_target(conn, schema, [1, 2, 3])

    changed = _batch([2, 4])
    changed["name"] = ["changed 2", "new 4"]
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([changed]),
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )

    result = IncrementalStrategy().run(ctx)

    assert result.is_incremental is True
    assert result.rows_read == 2
    assert _rows(conn, schema) == [
        (1, "row 1"),
        (2, "changed 2"),
        (3, "row 3"),
        (4, "new 4"),
    ]


def test_empty_window_changes_nothing(conn, schema) -> None:
    _fill_target(conn, schema, [1, 2])
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([], rows=0),
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )

    result = IncrementalStrategy().run(ctx)

    assert result.data_present is False
    assert _rows(conn, schema) == [(1, "row 1"), (2, "row 2")]


def test_missing_target_falls_back_to_full(conn, schema) -> None:
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([_batch([1, 2])]),
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )

    result = IncrementalStrategy().run(ctx)

    assert result.fallback_reason == "the target table does not exist"
    assert result.load_method == "full"
    assert len(_rows(conn, schema)) == 2


def test_empty_target_falls_back_to_full(conn, schema) -> None:
    _fill_target(conn, schema, [1])
    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE "{schema}"."cil"')
    conn.commit()

    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([_batch([9])]),
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )
    result = IncrementalStrategy().run(ctx)

    assert result.fallback_reason == "the target table is empty"


def test_a_failure_midway_through_staging_leaves_the_target_untouched(conn, schema) -> None:
    """A window that dies mid-read must project **nothing** into the target.

    Reading the window and writing it are separated by a staging table exactly so
    that a half-read window cannot be seen: the only statement that touches the
    target is the closing ``INSERT … ON CONFLICT``. Were the staging table
    committed as it filled, or the upsert run per batch, the target would end up
    holding an arbitrary prefix of the window -- and the run would report a
    failure while the data had already moved. `full` has the same test; this path
    writes to a live target rather than a shadow one, so it needs its own.
    """
    from dbextractors.core import target_pg

    _fill_target(conn, schema, [1, 2, 3])
    before = _rows(conn, schema)

    class Explodes(FakeDialect):
        def iter_batches(self, engine, sql, batch_size):
            changed = _batch([2])
            changed["name"] = ["changed 2"]
            yield changed
            raise RuntimeError("the connection dropped mid-window")

    ctx = make_context(
        conn,
        schema,
        dialect=Explodes(rows=2),
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )

    with pytest.raises(RuntimeError, match="dropped mid-window"):
        IncrementalStrategy().run(ctx)

    # Deliberately without a `conn.rollback()` first: the caller is left holding
    # this connection, and the strategy owes it a state it can carry on from. A
    # rollback here would also hide a staging table the strategy failed to clean
    # up, because an uncommitted one is visible only inside its own transaction.
    assert _rows(conn, schema) == before
    assert target_pg.find_stale_shadows(conn, schema) == []


def test_window_condition_uses_coalesce_and_the_zero_date(conn, schema) -> None:
    """``COALESCE(NULLIF(updated, '<zero date>'), created)`` — taken from the predecessor.

    Without ``NULLIF`` a row with a zero `updated_at` would drop out of the window,
    because `COALESCE` would never reach `created_at`.
    """
    columns = [*COLUMNS, ("date_created", "timestamp", "TIMESTAMP")]
    _fill_target(conn, schema, [1], columns)
    batch = _batch([1])
    batch["date_created"] = None
    dialect = FakeDialect([batch])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(columns),
        settings=SETTINGS,
    )
    IncrementalStrategy().run(ctx)

    where = dialect.estimate_calls[-1]
    assert "COALESCE(NULLIF(`date_modified`, '0000-00-00 00:00:00'), `date_created`)" in where
    assert ">=" in where


def test_window_without_created_at_uses_updated_only(conn, schema) -> None:
    _fill_target(conn, schema, [1])
    dialect = FakeDialect([_batch([1])])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )
    IncrementalStrategy().run(ctx)

    assert dialect.estimate_calls[-1].startswith("`date_modified` >=")


def test_where_from_the_configuration_is_appended(conn, schema) -> None:
    _fill_target(conn, schema, [1])
    dialect = FakeDialect([_batch([1])])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
        where="deleted = 0",
    )
    IncrementalStrategy().run(ctx)

    assert dialect.estimate_calls[-1].endswith("AND (deleted = 0)")


# --- Validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"primary_column": None}, "is missing from LOAD_SETTINGS"),
        ({"updated_at_column": None}, "is missing from LOAD_SETTINGS"),
        ({"updated_at_column": "does_not_exist"}, "is not in the source table"),
        ({"created_at_column": "does_not_exist"}, "is not in the source table"),
    ],
)
def test_validation_fails_instead_of_a_silent_zero(conn, schema, override, message) -> None:
    """The predecessor returns an empty result in these cases and the run goes green."""
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([]),
        columns=make_columns(COLUMNS),
        settings={**SETTINGS, "created_at_column": None, **override},
    )
    with pytest.raises(StrategyError, match=message):
        IncrementalStrategy().validate(ctx)


# --- Cutoff ------------------------------------------------------------------


def test_cutoff_from_days_back() -> None:
    cutoff, numeric, lookback = compute_incremental_cutoff({"days_back": 7}, {})
    assert cutoff == (datetime.now() - timedelta(days=7)).date()
    assert numeric is None
    assert lookback is None


def test_lookback_beats_days_back() -> None:
    cutoff, _numeric, lookback = compute_incremental_cutoff(
        {"days_back": 30, "incremental_lookback_hours": 6}, {}
    )
    assert lookback == 6
    assert cutoff == (datetime.now() - timedelta(hours=6)).date()


def test_runtime_variable_beats_the_configuration() -> None:
    _cutoff, _numeric, lookback = compute_incremental_cutoff(
        {"incremental_lookback_hours": 24}, {"incremental_lookback_hours": 2}
    )
    assert lookback == 2


def test_deep_cutoff_without_any_setting() -> None:
    """1 January of the previous or the current year, depending on whether July has passed."""
    cutoff, _numeric, lookback = compute_incremental_cutoff({}, {})
    today = date.today()
    expected_year = today.year - 1 if today.month < 7 else today.year
    assert cutoff == date(expected_year, 1, 1)
    assert lookback is None


def test_invalid_lookback_falls_back_to_the_deep_cutoff() -> None:
    import logging

    _cutoff, _numeric, lookback = compute_incremental_cutoff(
        {"incremental_lookback_hours": "nonsense"}, {}, logger=logging.getLogger("test")
    )
    assert lookback is None


def test_window_never_sees_a_change_older_than_days_back(conn, schema) -> None:
    """**A data gap the current behaviour can produce.**

    The window is computed from the time of the run, not from the state of the
    target. When a pipeline has not run for longer than ``days_back``, rows changed
    in that gap do not fit into the window — and the next run again looks only
    ``days_back`` back, so they are **never picked up**. The test demonstrates it on
    the condition: the cutoff is always relative to today, never to the last
    successful run.
    """
    cutoff, _numeric, _lookback = compute_incremental_cutoff({"days_back": 7}, {})
    assert cutoff > date(2020, 1, 1)
    assert (date.today() - cutoff).days == 7
    # A change from before the cutoff does not fit into the window, even if the
    # target was last updated a month ago.
    old_change = date.today() - timedelta(days=30)
    assert old_change < cutoff


def test_the_reported_metrics_are_exactly_these(conn, schema) -> None:
    """The metric set, pinned — because the docstring has drifted from it twice.

    Once it claimed this strategy flipped `_deleted_in_source`, which nothing
    did; once it claimed a `stale_by_days` metric and a warning, and
    `stale_by_days` appeared nowhere in the package. Both were readable,
    plausible and false, and both were found by reading rather than by a failing
    test.

    An exact comparison rather than a subset: a metric that is added has to be
    described where the strategy is described, and a metric that disappears has
    to fail here rather than leave the prose behind. Changing this set is meant
    to be a deliberate act.
    """
    _fill_target(conn, schema, [1, 2])
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([_batch([2])]),
        columns=make_columns(COLUMNS),
        settings=dict(SETTINGS, created_at_column=None),
    )

    result = IncrementalStrategy().run(ctx)

    assert set(result.phase_metrics) == {
        "cutoff",
        "mode",
        "days_back",
        "lookback_hours",
        "estimated_rows",
    }


def test_default_days_back() -> None:
    assert DEFAULT_DAYS_BACK == 14


# --- The window label in the log ---------------------------------------------
#
# Found during a pilot run: the log said
#
#     Incremental window from 2026-01-01 (days_back 14)
#
# The window was right — a deep cutoff — but the label next to it came from a
# different branch. Fourteen days back from 14 August is 31 July, not 1 January,
# so the message pointed the wrong way while investigating "why did it transfer
# that many rows". The cause was that each strategy assembled its own label and
# knew only two of the three branches: `incremental` knew lookback + days_back,
# `parent_incremental` knew lookback + deep. Both are now derived from
# `window_mode`.


@pytest.mark.parametrize(
    "settings,lookback,expected_mode,expected_label",
    [
        ({"incremental_lookback_hours": 6}, 6, "lookback", "lookback 6 h"),
        ({"days_back": 7}, None, "days_back", "days_back 7"),
        # An empty configuration: `days_back` is missing altogether, so the deep
        # cutoff applies. (At run time `config.parse` substitutes 14, but that is
        # a layer above this decision.)
        ({}, None, "deep", "deep cutoff"),
        # Variant B's shape (see docs/legacy-compat.md): `incremental_date_column`
        # switches off the `days_back` branch even though `config.parse`
        # substitutes 14. This is the case from the pilot run.
        (
            {"incremental_date_column": "corrected_at", "days_back": 14},
            None,
            "deep",
            "deep cutoff",
        ),
        (
            {"incremental_parent_table": "invoices", "days_back": 14},
            None,
            "deep",
            "deep cutoff",
        ),
    ],
)
def test_window_label_matches_the_mode(settings, lookback, expected_mode, expected_label) -> None:
    from dbextractors.core.strategies.incremental import window_label, window_mode

    assert window_mode(settings, lookback) == expected_mode
    assert window_label(settings, lookback) == expected_label


def test_deep_cutoff_label_does_not_mention_days_back() -> None:
    """Exactly the message from the pilot — the cutoff is January, the label must not say 14 days."""
    from dbextractors.core.strategies.incremental import window_label

    settings = {"incremental_date_column": "corrected_at", "days_back": 14}
    cutoff, _numeric, lookback = compute_incremental_cutoff(settings, {})

    assert cutoff.month == 1 and cutoff.day == 1, "a deep cutoff is 1 January"
    assert "days_back" not in window_label(settings, lookback)


# --- Variant B's window shape -------------------------------------------------
#
# Variant B describes the window with a different set of keys and builds a
# **different condition** out of them (see docs/legacy-compat.md). They are not
# aliases: `COALESCE(a,0) >= c OR COALESCE(b,0) >= c` takes a row when either of
# the two columns is above the boundary, whereas `COALESCE(NULLIF(a,...), b) >= c`
# only looks at the fallback column when the primary one is NULL. A row with an
# old `corrected_at` and a fresh `created_at` is therefore taken by one and
# dropped by the other.


def test_variant_b_keys_give_a_window_joined_with_or(conn, schema) -> None:
    from datetime import date

    from dbextractors.core.strategies.incremental import IncrementalStrategy

    source = FakeDialect([])
    ctx = make_context(
        conn,
        schema,
        dialect=source,
        columns=make_columns(COLUMNS),
        settings={
            "primary_column": "id",
            "incremental_date_column": "changed_at",
            "incremental_date_column_fallback": "created_at",
        },
    )

    where = IncrementalStrategy()._build_where(ctx, date(2026, 8, 6))

    assert " OR " in where
    assert "COALESCE" in where
    assert "NULLIF" not in where


def test_current_keys_stay_on_coalesce(conn, schema) -> None:
    """Anything with `updated_at_column` must not get a different condition than today."""
    from datetime import date

    from dbextractors.core.strategies.incremental import IncrementalStrategy

    source = FakeDialect([])
    ctx = make_context(
        conn,
        schema,
        dialect=source,
        columns=make_columns(COLUMNS),
        settings={
            "primary_column": "id",
            "updated_at_column": "changed_at",
            "created_at_column": "created_at",
        },
    )

    where = IncrementalStrategy()._build_where(ctx, date(2026, 8, 6))

    assert " OR " not in where
    assert "COALESCE" in where


def test_parent_table_routes_to_parent_incremental() -> None:
    """Variant B's child tables carry `load_method: incremental` and are told apart
    only by the presence of `incremental_parent_table`."""
    from dbextractors.core.strategies.base import resolve_strategy

    assert resolve_strategy("incremental", {}).name == "incremental"
    assert (
        resolve_strategy("incremental", {"incremental_parent_table": "orders"}).name
        == "parent_incremental"
    )
    # The key only applies to `incremental` — for the other methods it has nothing
    # to change.
    assert resolve_strategy("full", {"incremental_parent_table": "X"}).name == "full"
    assert resolve_strategy("hash", {"incremental_parent_table": "X"}).name == "hash_diff"


# --- Deep cutoff vs days_back -------------------------------------------------


def test_variant_b_keys_give_a_deep_cutoff_even_when_days_back_was_substituted() -> None:
    """`config.parse` substitutes `days_back=14` even when the configuration lacks it.

    Variant B's predecessor has no `days_back` branch at all — only lookback, or a
    deep cutoff. If the substituted default won, the deep branch would be
    unreachable for variant B.

    Documented in production: the package computed today - 14 days and transferred
    0 rows, while the old side used 1 January and transferred 394.
    """
    cutoff, _numeric, lookback = compute_incremental_cutoff(
        {"incremental_date_column": "corrected_at", "days_back": 14}, {}
    )
    assert (cutoff.month, cutoff.day) == (1, 1)
    assert lookback is None


def test_child_of_a_parent_gets_the_deep_cutoff_too() -> None:
    """A child table has no date column of its own — its shape is recognised through
    the parent. If the decision went by `incremental_date_column` alone, the child
    would get a different window from its parent and the two would drift apart."""
    cutoff, _numeric, _lookback = compute_incremental_cutoff(
        {"incremental_parent_table": "orders", "days_back": 14}, {}
    )
    assert (cutoff.month, cutoff.day) == (1, 1)


def test_days_back_still_applies_where_variant_b_keys_are_absent() -> None:
    """The other deployments know no deep cutoff — their behaviour must not change."""
    cutoff, _numeric, _lookback = compute_incremental_cutoff({"days_back": 7}, {})
    assert cutoff == (datetime.now() - timedelta(days=7)).date()


def test_lookback_beats_variant_b_deep_cutoff_as_well() -> None:
    """Delta mode wins over both — otherwise a warm-up run could not be switched to
    a short window."""
    cutoff, _numeric, lookback = compute_incremental_cutoff(
        {"incremental_date_column": "corrected_at"}, {"incremental_lookback_hours": 24}
    )
    assert lookback == 24
    assert cutoff == (datetime.now() - timedelta(hours=24)).date()
