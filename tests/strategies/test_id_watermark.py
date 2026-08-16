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
