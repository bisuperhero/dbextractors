"""Tests for `ParentIncrementalStrategy` against a live target PostgreSQL.

This is the only strategy that **deletes** from the target, so what it deletes and
what it leaves alone is the bulk of the testing here. The predecessor commits its
`DELETE` before the stream, so a stream failure leaves a hole in the target;
`test_a_stream_failure_does_not_delete_the_window` proves that cannot happen here.

The parent table is an ordinary table in the same target schema -- it is created by
hand here, because in a real run its own pipeline had loaded it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from dbextractors.core.strategies.base import StrategyError
from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.parent_incremental import ParentIncrementalStrategy
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("parent_id", "int", "INTEGER"),
    ("descr", "varchar", "TEXT"),
]

#: The window here is driven by ``incremental_lookback_hours``, not ``days_back``. On
#: the parent path that is the only lever the predecessor has (lookback, or the deep
#: cutoff -- the ``days_back`` branch does not exist there at all). This used to say
#: ``days_back: 7``, which encoded behaviour that diverged from the predecessor.
SETTINGS = {
    "primary_column": "id",
    "incremental_parent_table": "orders",
    "incremental_parent_date_column": "changed_at",
    "incremental_lookback_hours": 7 * 24,
}

TODAY = date.today()
IN_WINDOW = TODAY - timedelta(days=2)
OUT_OF_WINDOW = TODAY - timedelta(days=60)


def _source_rows(rows=((1, 10, "a"), (2, 10, "b"), (3, 20, "c"))) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "parent_id": [r[1] for r in rows],
            "descr": [r[2] for r in rows],
        }
    )


def _parents(rows=((10, IN_WINDOW), (20, OUT_OF_WINDOW)), column="changed_at") -> pd.DataFrame:
    """The parent **in the source** -- the child is narrowed from it by a subquery."""
    return pd.DataFrame({"id": [r[0] for r in rows], column: [r[1] for r in rows]})


def _fake(rows=None, parents=None) -> FakeHashSource:
    return FakeHashSource(
        _source_rows() if rows is None else rows,
        pk="id",
        parents=_parents() if parents is None else parents,
    )


def _ctx(conn, schema, dialect, *, settings=None, columns=None):
    return make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(columns or COLUMNS),
        settings={**SETTINGS, **(settings or {})},
    )


def _parent_in_target(conn, schema, rows=((10, IN_WINDOW), (20, OUT_OF_WINDOW))) -> None:
    """The parent in the target. In a real run its own pipeline had loaded it there."""
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{schema}"."orders" (id integer, changed_at date)')
        for ident, when in rows:
            cur.execute(f'INSERT INTO "{schema}"."orders" VALUES (%s, %s)', (ident, when))
    conn.commit()


def _rows(conn, schema) -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, parent_id, descr FROM "{schema}"."cil" ORDER BY id')
        return cur.fetchall()


def test_only_the_parent_window_is_replaced(conn, schema) -> None:
    """Children of a parent outside the window must be neither deleted nor reloaded."""
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema)

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert result.phase_metrics["deleted"] == 2, "only the children of parent 10 go"
    assert result.rows_written == 2
    assert _rows(conn, schema) == [(1, 10, "a"), (2, 10, "b"), (3, 20, "c")]


def test_a_change_inside_the_window_lands(conn, schema) -> None:
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema)
    source.update(1, "descr", "changed")

    ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert _rows(conn, schema)[0] == (1, 10, "changed")


def test_a_deleted_child_inside_the_window_disappears(conn, schema) -> None:
    """This is why it deletes: the child has no change date, so nothing else notices."""
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema)
    source.delete(2)

    ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert _rows(conn, schema) == [(1, 10, "a"), (3, 20, "c")]


def test_a_deleted_child_outside_the_window_stays(conn, schema) -> None:
    """The mirror of the previous one: outside the window nothing is deleted, even
    when the source no longer has the row."""
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema)
    source.delete(3)

    ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert (3, 20, "c") in _rows(conn, schema)


def test_the_fallback_date_column_widens_the_window(conn, schema) -> None:
    """``incremental_parent_date_column_fallback`` is added with ``OR``.

    In some tables the change date is filled in only in one of the two columns.
    """
    parents = pd.DataFrame(
        {
            "id": [10, 20],
            "changed_at": [None, OUT_OF_WINDOW],
            "created_at": [IN_WINDOW, OUT_OF_WINDOW],
        }
    )
    source = _fake(parents=parents)
    FullLoadStrategy().run(_ctx(conn, schema, source))
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{schema}"."orders" (id integer, changed_at date, created_at date)'
        )
        cur.execute(
            f'INSERT INTO "{schema}"."orders" VALUES (10, NULL, %s), (20, %s, %s)',
            (IN_WINDOW, OUT_OF_WINDOW, OUT_OF_WINDOW),
        )
    conn.commit()

    result = ParentIncrementalStrategy().run(
        _ctx(
            conn,
            schema,
            source,
            settings={"incremental_parent_date_column_fallback": "created_at"},
        )
    )

    assert result.phase_metrics["deleted"] == 2, "parent 10 entered the window via the fallback"


def test_an_empty_window_deletes_nothing(conn, schema) -> None:
    source = _fake(parents=_parents(rows=((10, OUT_OF_WINDOW), (20, OUT_OF_WINDOW))))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema, rows=((10, OUT_OF_WINDOW), (20, OUT_OF_WINDOW)))

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 0
    assert result.data_present is False
    assert len(_rows(conn, schema)) == 3, "an empty window must not touch the target"


def test_a_stream_failure_does_not_delete_the_window(conn, schema) -> None:
    """**The most important test of this strategy.**

    The predecessor commits the `DELETE` and only then streams, so a stream failure
    leaves the target with the rows deleted and nothing put back. Here both are in one
    transaction, so after a failure the target has to be exactly what it was.
    """
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema)
    before = _rows(conn, schema)
    source.scan_error = RuntimeError("the connection dropped mid-stream")

    class FailsWhileReading(type(source)):  # type: ignore[misc]
        def iter_batches(self, engine, sql, batch_size):
            yield from ()
            raise RuntimeError("the connection dropped mid-stream")

    source.__class__ = FailsWhileReading
    with pytest.raises(RuntimeError, match="dropped"):
        ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    # No `conn.rollback()` before the assertions: rolling back here would clean up
    # after the strategy, and a staging table it forgot would disappear with it
    # (an uncommitted one is visible only inside its own transaction). The caller
    # holds this very connection after a failure.
    from dbextractors.core import target_pg

    assert _rows(conn, schema) == before, "after a failure not a single row may be missing"
    assert target_pg.find_stale_shadows(conn, schema) == []


def test_without_a_parent_in_the_configuration_it_fails(conn, schema) -> None:
    source = _fake()
    ctx = make_context(
        conn,
        schema,
        dialect=source,
        columns=make_columns(COLUMNS),
        settings={"primary_column": "id"},
    )

    with pytest.raises(StrategyError, match="incremental_parent_table"):
        ParentIncrementalStrategy().run(ctx)


def test_without_a_reference_column_it_fails(conn, schema) -> None:
    """A child with no column pointing at the parent cannot narrow the window -- say so now."""
    columns = [("id", "int", "INTEGER"), ("descr", "varchar", "TEXT")]
    source = FakeHashSource(pd.DataFrame({"id": [1], "descr": ["a"]}), pk="id")

    with pytest.raises(StrategyError, match="parent_id"):
        ParentIncrementalStrategy().run(_ctx(conn, schema, source, columns=columns))


def test_a_missing_parent_in_the_target_falls_back_to_full(conn, schema) -> None:
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is not None
    assert "parent table" in result.fallback_reason
    assert len(_rows(conn, schema)) == 3


def test_a_missing_target_falls_back_to_full(conn, schema) -> None:
    source = _fake()

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason == "the target child table does not exist"
    assert len(_rows(conn, schema)) == 3


def test_an_empty_target_is_not_backfilled(conn, schema) -> None:
    """**Pins a known gap, on purpose.** An existing but empty target is *not* a full load.

    `incremental` and `id_watermark` treat an empty target as "there is nothing to
    build on" and fall back to a full load. This strategy checks only that the
    table **exists** -- so over an empty target it transfers the window and
    nothing else, and the run is green with a partial table.

    That is not an oversight in the rewrite, it is parity: the predecessor
    (``load_incremental_by_parent`` in variant B's Firebird extractor) gates on
    ``to_regclass`` alone and has no row count anywhere. Adding a fallback here
    would transfer a different set of rows than the old side does, and parity
    with the predecessor outranks the symmetry between our own strategies -- see
    the module docstring of `strategies.parent_incremental`.

    The test therefore asserts the gap. Should it ever be closed deliberately,
    this is the test to rewrite, and the reason it exists is so that closing it
    cannot happen by accident.
    """
    source = _fake()
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _parent_in_target(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f'DELETE FROM "{schema}"."cil"')
    conn.commit()

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is None, "an empty target does not switch to a full load"
    assert _rows(conn, schema) == [(1, 10, "a"), (2, 10, "b")]
    assert (3, 20, "c") not in _rows(conn, schema), "the row outside the window stays missing"


def test_days_back_does_not_affect_the_parent_window(conn, schema) -> None:
    """`days_back` is ignored on the parent path -- the deep cutoff applies.

    The predecessor has no `days_back` branch there, but `config.parse` substitutes
    `days_back=14` even where the configuration does not have it. If that substituted
    default won, the child would get a "today - 14 days" window instead of "since
    1 January" and would silently transfer nothing.

    Documented in production: 0 rows transferred against 394 on the old side, with
    `status = completed`.
    """
    # A parent changed 60 days ago: outside "today - 14 days", inside the deep cutoff
    # (1 January). `incremental_lookback_hours=None` turns the delta mode off so that the
    # window is decided by exactly the branch this test is about.
    without_lookback = {"incremental_lookback_hours": None, "days_back": 14}
    source = _fake(parents=_parents(rows=((10, OUT_OF_WINDOW),)))
    FullLoadStrategy().run(_ctx(conn, schema, source, settings=without_lookback))
    _parent_in_target(conn, schema, rows=((10, OUT_OF_WINDOW),))

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source, settings=without_lookback))

    assert (
        result.rows_written > 0
    ), "the parent is outside `days_back` but inside the deep cutoff -- it belongs in the window"


def test_a_legacy_target_without_row_hash_is_extended_before_staging(conn, schema) -> None:
    """A target created by an older extractor has no `row_hash` -- dbextractors must add it.

    The staging table is created with ``LIKE <target>``, so it inherits whatever the
    target is missing, and the `COPY` of the first batch fails with::

        column "row_hash" of relation "dbx_shadow_<table>_..." does not exist

    So it has to be added **before** the staging table, not just before the
    `INSERT ... SELECT` -- by then it is too late. `full` has its own counterpart in the
    full-load tests; the parent path was missing one.
    """
    source = _fake()
    # The target exactly as the old side left it: source columns and nothing else.
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{schema}"."cil" (id INTEGER, parent_id INTEGER, descr TEXT)')
        for row in ((1, 10, "a"), (2, 10, "b"), (3, 20, "c")):
            cur.execute(f'INSERT INTO "{schema}"."cil" VALUES (%s, %s, %s)', row)
    conn.commit()
    _parent_in_target(conn, schema)

    result = ParentIncrementalStrategy().run(_ctx(conn, schema, source))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'cil'",
            (schema,),
        )
        columns = {r[0] for r in cur.fetchall()}

    assert "row_hash" in columns
    assert "_deleted_in_source" in columns, "every strategy owes the target this column"
    assert result.rows_written == 2
