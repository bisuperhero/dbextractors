"""Tests for `IdWatermarkStrategy` against a live target PostgreSQL.

The strategy is simple, but it has two properties that break easily and silently:
the watermark is taken from the **target** (not from the source), and rows below it
are never transferred. Both have their own test here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core.strategies.base import StrategyError
from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.id_watermark import IdWatermarkStrategy
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("name", "varchar", "TEXT"),
]

SETTINGS = {"primary_column": "id"}


def _source_rows(ids=(1, 2, 3)) -> pd.DataFrame:
    return pd.DataFrame({"id": list(ids), "name": [f"row {i}" for i in ids]})


def _ctx(conn, schema, dialect, *, settings=None, columns=None, where=None):
    return make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(columns or COLUMNS),
        settings={**SETTINGS, **(settings or {})},
        where=where,
    )


def _rows(conn, schema) -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_name" FROM "{schema}"."cil" ORDER BY id')
        return cur.fetchall()


def test_only_rows_above_the_watermark_are_transferred(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": 4, "name": "new"})
    source.insert({"id": 5, "name": "next"})

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 2, "the old rows 1-3 must not be read again"
    assert result.rows_written == 2
    assert result.phase_metrics["watermark"] == "3"
    assert _rows(conn, schema) == [
        (1, "row 1"),
        (2, "row 2"),
        (3, "row 3"),
        (4, "new"),
        (5, "next"),
    ]


def test_no_changes_transfers_nothing(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 0
    assert result.rows_written == 0
    assert result.data_present is False


def test_a_change_below_the_watermark_is_not_transferred(conn, schema) -> None:
    """A property of the strategy, not a bug -- which is exactly why it is tested.

    Anyone who needs changes picked up should use `hash` or `incremental`. The test is
    here so that reading the code does not leave it looking like an unfinished feature.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.update(2, "name", "changed")

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 0
    assert _rows(conn, schema)[1] == (2, "row 2"), "the change goes unnoticed"


def test_a_late_insert_below_the_watermark_is_never_picked_up(conn, schema) -> None:
    """The second documented blind spot, and a different one from a late *change*.

    A row that appears in the source with a key lower than one already in the
    target is not an update of anything -- it is missing from the target
    altogether, and no later run will ever fetch it, because every run starts
    above ``MAX(pk)``. With a sequence that cannot happen; with hand-assigned
    keys it can, and then the table is quietly short a row for good.
    """
    source = FakeHashSource(_source_rows(ids=(1, 3)), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": 2, "name": "filled in later"})
    source.insert({"id": 4, "name": "new"})

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 1, "only the row above the watermark is read"
    assert _rows(conn, schema) == [(1, "row 1"), (3, "row 3"), (4, "new")]


def test_a_deleted_row_is_not_marked(conn, schema) -> None:
    """`_deleted_in_source` stays `FALSE` here -- documented, and the column still exists.

    Marking deletions would mean reading every key in the source, which is the
    entire saving this strategy exists for. What is **not** optional is the
    column: 133 dbt models read it, so a table loaded this way has to have it,
    filled with `FALSE`.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.delete(2)

    IdWatermarkStrategy().run(_ctx(conn, schema, source))

    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_deleted_in_source" FROM "{schema}"."cil" ORDER BY id')
        assert cur.fetchall() == [(1, False), (2, False), (3, False)]


def test_the_where_clause_from_the_configuration_applies(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source, where="1=1"))
    source.insert({"id": 4, "name": "new"})

    IdWatermarkStrategy().run(_ctx(conn, schema, source, where="1=1"))

    assert any("1=1" in sql and "> 3" in sql for sql in source.seen_sql)


def test_an_empty_target_falls_back_to_full(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))
    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE "{schema}"."cil"')
    conn.commit()

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is not None
    assert "is empty" in result.fallback_reason
    assert len(_rows(conn, schema)) == 3


def test_a_missing_target_falls_back_to_full(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason == "the target table does not exist"
    assert len(_rows(conn, schema)) == 3


def test_without_a_primary_key_it_fails(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    ctx = make_context(conn, schema, dialect=source, columns=make_columns(COLUMNS), settings={})

    with pytest.raises(StrategyError, match="primary_column"):
        IdWatermarkStrategy().run(ctx)


def test_a_dialect_without_keyset_support_fails(conn, schema) -> None:
    """``required_features`` is checked before the source is touched."""
    from dbextractors.dialects.base import UnsupportedFeature

    source = FakeHashSource(_source_rows(), pk="id")
    source.FEATURES = frozenset()

    with pytest.raises(UnsupportedFeature, match="keyset"):
        IdWatermarkStrategy().run(_ctx(conn, schema, source))


def test_a_text_key_is_escaped(conn, schema) -> None:
    """The watermark goes into SQL as a literal, so an apostrophe in it has to survive."""
    columns = [("code", "varchar", "TEXT"), ("name", "varchar", "TEXT")]
    data = pd.DataFrame({"code": ["a", "b'c"], "name": ["x", "y"]})
    source = FakeHashSource(data, pk="code")
    settings = {"primary_column": "code"}

    FullLoadStrategy().run(_ctx(conn, schema, source, settings=settings, columns=columns))
    source.estimate_calls.clear()
    source.insert({"code": "z", "name": "new"})
    result = IdWatermarkStrategy().run(
        _ctx(conn, schema, source, settings=settings, columns=columns)
    )

    assert any("> 'b''c'" in (where or "") for where in source.estimate_calls)
    assert result.rows_written == 1, "'z' > \"b'c\", so it has to be transferred"


# --- checkpoints ---------------------------------------------------------------
#
# A delta used to be one transaction: a run that died three quarters of the way
# through rolled everything back, and every retry read the same delta from the
# same `MAX(pk)` again. In production that was a 42.7 M-row delta that never got
# anywhere.


class _FailingSource(FakeHashSource):
    """Dies after handing over ``fail_after`` batches — like an OOM kill or a dead source.

    The error is deliberately not a connection error, so read recovery does not
    retry it: the point is what the *strategy* leaves behind.
    """

    def __init__(self, *args, fail_after: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_after = fail_after

    def iter_batches(self, engine, sql, batch_size):
        for handed, batch in enumerate(super().iter_batches(engine, sql, batch_size)):
            if self.fail_after is not None and handed >= self.fail_after:
                raise RuntimeError("the source died mid-delta")
            yield batch


def _shadows(conn, schema) -> list:
    from dbextractors.core import target_pg

    return target_pg.find_stale_shadows(conn, schema)


def _committed_ids(dsn, schema) -> list:
    """What another session sees — i.e. only what was really committed."""
    import psycopg2

    with psycopg2.connect(dsn) as other, other.cursor() as cur:
        cur.execute(f'SELECT id FROM "{schema}"."cil" ORDER BY id')
        ids = [r[0] for r in cur.fetchall()]
    other.close()
    return ids


def _seeded(conn, schema, source) -> None:
    FullLoadStrategy().run(_ctx(conn, schema, source))
    for i in range(4, 14):
        source.insert({"id": i, "name": f"new {i}"})


def test_a_large_delta_is_committed_in_checkpoints(conn, schema, dsn) -> None:
    source = _FailingSource(_source_rows(), pk="id")
    _seeded(conn, schema, source)

    result = IdWatermarkStrategy().run(
        _ctx(conn, schema, source, settings={"batch_size": 2, "checkpoint_rows": 4})
    )

    assert result.rows_read == 10
    assert result.rows_written == 10
    assert result.phase_metrics["checkpoints"] == 2, "after rows 4 and 8; the last 2 at the end"
    assert _committed_ids(dsn, schema) == list(range(1, 14))
    assert _shadows(conn, schema) == []


def test_a_failure_after_checkpoints_keeps_them_and_the_next_run_resumes(conn, schema, dsn) -> None:
    """The guarantee the checkpoints exist for, end to end.

    Four batches of two arrive, then the source dies. The checkpoint after six
    rows (ids 4–9) is committed; the batch staged after it (ids 10–11) is rolled
    back. The next run starts above 9 — not above 3 — reads only what is left,
    and the target ends with every row exactly once.
    """
    source = _FailingSource(_source_rows(), pk="id", fail_after=4)
    _seeded(conn, schema, source)
    settings = {"batch_size": 2, "checkpoint_rows": 6}

    with pytest.raises(RuntimeError, match="mid-delta"):
        IdWatermarkStrategy().run(_ctx(conn, schema, source, settings=settings))

    assert _committed_ids(dsn, schema) == list(range(1, 10))
    assert _shadows(conn, schema) == [], "the failed run must not leave its staging table"

    source.fail_after = None
    source.estimate_calls.clear()
    result = IdWatermarkStrategy().run(_ctx(conn, schema, source, settings=settings))

    assert result.phase_metrics["watermark"] == "9"
    assert any("> 9" in (where or "") for where in source.estimate_calls)
    assert result.rows_read == 4
    assert _committed_ids(dsn, schema) == list(range(1, 14))
    assert _rows(conn, schema)[-1] == (13, "new 13")


def test_checkpoints_off_keeps_the_delta_all_or_nothing(conn, schema, dsn) -> None:
    """``checkpoint_rows: 0`` is the behaviour before 2.1.2: nothing survives a failure."""
    source = _FailingSource(_source_rows(), pk="id", fail_after=4)
    _seeded(conn, schema, source)

    with pytest.raises(RuntimeError, match="mid-delta"):
        IdWatermarkStrategy().run(
            _ctx(conn, schema, source, settings={"batch_size": 2, "checkpoint_rows": 0})
        )

    assert _committed_ids(dsn, schema) == [1, 2, 3]
    assert _shadows(conn, schema) == []


def test_the_default_checkpoint_does_not_split_a_small_delta(conn, schema) -> None:
    from dbextractors.core.strategies.id_watermark import DEFAULT_CHECKPOINT_ROWS

    source = _FailingSource(_source_rows(), pk="id")
    _seeded(conn, schema, source)

    result = IdWatermarkStrategy().run(_ctx(conn, schema, source, settings={"batch_size": 2}))

    assert DEFAULT_CHECKPOINT_ROWS == 5_000_000
    assert result.phase_metrics["checkpoints"] == 0
    assert result.rows_written == 10


@pytest.mark.parametrize("value", ["many", 2.5, True, [4]])
def test_a_checkpoint_size_that_is_not_a_whole_number_fails(conn, schema, value) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source))

    with pytest.raises(StrategyError, match="checkpoint_rows"):
        IdWatermarkStrategy().run(_ctx(conn, schema, source, settings={"checkpoint_rows": value}))


def test_an_unordered_read_never_checkpoints(conn, schema, dsn) -> None:
    """With a surrogate key the read carries no ``ORDER BY``.

    A committed part of an unordered stream is an arbitrary subset, and resuming
    above its ``MAX(pk)`` would skip whatever below that key had not arrived
    yet. So the delta stays one transaction, whatever ``checkpoint_rows`` says.
    """
    import dataclasses

    from dbextractors.dialects.base import SurrogateKey

    source = _FailingSource(_source_rows(), pk="id")
    _seeded(conn, schema, source)
    ctx = dataclasses.replace(
        _ctx(conn, schema, source, settings={"batch_size": 2, "checkpoint_rows": 4}),
        surrogate=SurrogateKey(enabled=True, alias="_uni_surrogate_key", expr="CONCAT(id, name)"),
    )
    source.fail_after = 4

    with pytest.raises(RuntimeError, match="mid-delta"):
        IdWatermarkStrategy().run(ctx)

    assert not any("ORDER BY" in sql for sql in source.seen_sql[-1:])
    assert _committed_ids(dsn, schema) == [1, 2, 3]
