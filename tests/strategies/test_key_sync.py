"""Tests for `KeySyncStrategy` against a live target PostgreSQL.

What matters here is the pair of things a deletion check must get right and a
plain "rows were marked" assertion does not show:

- **it reads only the key** from the source and **fetches no row** from it, however
  many rows the table has — the whole reason this strategy exists next to
  `hash_diff`;
- **it can never mark from a partial or empty view of the source**, because the
  cost of being wrong is every live row hidden from the models.

The source is `FakeHashSource`, which answers a bare ``SELECT <pk>`` with the key
column alone, the way a database would.
"""

from __future__ import annotations

import re
from typing import ClassVar

import pandas as pd
import psycopg2.extensions
import pytest

from dbextractors.core.strategies.base import StrategyError, resolve_strategy
from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.key_sync import KeySyncStrategy
from dbextractors.dialects.base import SurrogateKey
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("name", "varchar", "TEXT"),
    ("price", "decimal", "NUMERIC"),
]
SETTINGS = {"primary_column": "id"}


def _source_rows(ids=(1, 2, 3)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(ids),
            "name": [f"row {i}" for i in ids],
            "price": [float(i) * 10 for i in ids],
        }
    )


def _ctx(conn, schema, dialect, *, settings=None, table_cfg=None, where=None):
    return make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={**SETTINGS, **(settings or {})},
        table_cfg=table_cfg,
        where=where,
    )


def _seed(conn, schema, source) -> None:
    FullLoadStrategy().run(_ctx(conn, schema, source))


def _flags(conn, schema) -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_deleted_in_source" FROM "{schema}"."cil" ORDER BY id')
        return cur.fetchall()


def _snapshot_of_target(conn, schema) -> list:
    """Everything except the managed columns — to prove nothing else was touched."""
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_name", price, row_hash FROM "{schema}"."cil" ORDER BY id')
        return cur.fetchall()


# --- Marking ----------------------------------------------------------------


def test_a_key_missing_from_the_source_is_marked_and_nothing_else_moves(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    before = _snapshot_of_target(conn, schema)
    source.delete(2)

    result = KeySyncStrategy().run(_ctx(conn, schema, source))

    assert _flags(conn, schema) == [(1, False), (2, True), (3, False)]
    assert _snapshot_of_target(conn, schema) == before, "key_sync must not rewrite row content"
    assert result.load_method == "key_sync"
    assert result.rows_read == 2
    assert result.rows_written == 0
    assert result.data_present is True
    assert result.fallback_reason is None
    assert result.phase_metrics == {
        "source_keys": 2,
        "target_rows": 3,
        "new_in_source": 0,
        "deleted_marked": 1,
        "revived": 0,
    }


def test_a_key_that_comes_back_is_unmarked(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.delete(2)
    KeySyncStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": 2, "name": "row 2", "price": 20.0})

    result = KeySyncStrategy().run(_ctx(conn, schema, source))

    assert _flags(conn, schema) == [(1, False), (2, False), (3, False)]
    assert result.phase_metrics["revived"] == 1
    assert result.phase_metrics["deleted_marked"] == 0


def test_an_unchanged_source_changes_nothing(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)

    result = KeySyncStrategy().run(_ctx(conn, schema, source))

    assert _flags(conn, schema) == [(1, False), (2, False), (3, False)]
    assert result.data_present is False
    assert result.phase_metrics["deleted_marked"] == 0
    assert result.phase_metrics["revived"] == 0


def test_a_new_source_key_is_counted_and_not_fetched(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.insert({"id": 4, "name": "new", "price": 40.0})
    source.insert({"id": 5, "name": "newer", "price": 50.0})

    result = KeySyncStrategy().run(_ctx(conn, schema, source))

    assert result.phase_metrics["new_in_source"] == 2
    assert result.phase_metrics["source_keys"] == 5
    assert [row[0] for row in _flags(conn, schema)] == [1, 2, 3], "the new rows must not be loaded"
    assert not source.download_sql


def test_it_reads_only_the_key_and_in_order(conn, schema) -> None:
    """One ``SELECT <pk>`` ordered by it — no hash, no other column, no ``IN (…)`` fetch."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.scan_sql.clear()
    source.download_sql.clear()

    KeySyncStrategy().run(_ctx(conn, schema, source))

    assert source.scan_sql == ["SELECT `id` FROM `zdroj` ORDER BY `id`"]
    assert not source.download_sql


def test_the_scan_is_streamed_in_batches(conn, schema) -> None:
    source = FakeHashSource(_source_rows(range(1, 8)), pk="id")
    _seed(conn, schema, source)
    source.delete(7)

    result = KeySyncStrategy().run(_ctx(conn, schema, source, settings={"batch_size": 2}))

    assert result.phase_metrics["source_keys"] == 6
    assert result.phase_metrics["deleted_marked"] == 1
    assert _flags(conn, schema)[-1] == (7, True)


def test_the_where_clause_narrows_the_key_scan(conn, schema) -> None:
    """Same contract as `hash_diff`: a row the filter hides is, to the sync, gone."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)

    KeySyncStrategy().run(_ctx(conn, schema, source, where="`id` > 1"))

    assert source.scan_sql[-1].startswith("SELECT `id` FROM `zdroj` WHERE `id` > 1")
    assert _flags(conn, schema) == [(1, True), (2, False), (3, False)]


# --- The SQL, verbatim --------------------------------------------------------


class _RecordingCursor(psycopg2.extensions.cursor):
    log: ClassVar[list] = []

    def execute(self, query, vars=None):
        type(self).log.append(("execute", str(query), vars))
        return super().execute(query, vars)

    def copy_expert(self, sql, file, size=8192):
        type(self).log.append(("copy", str(sql), None))
        return super().copy_expert(sql, file, size)


def _normalised(statements: list, schema: str) -> list:
    out = []
    for kind, sql, params in statements:
        sql = re.sub(r"dbx_src_keys_[0-9a-f]{8}", "dbx_src_keys_XXXXXXXX", " ".join(sql.split()))
        sql = sql.replace(schema, "<schema>")
        out.append((kind, sql, params))
    return out


def test_the_statements_it_sends_to_the_target(conn, schema) -> None:
    """Pins the exact SQL: keys go to a temp table, flags flip by key — no row is selected
    out of the target."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.delete(2)

    _RecordingCursor.log = []
    conn.cursor_factory = _RecordingCursor
    try:
        KeySyncStrategy().run(_ctx(conn, schema, source))
    finally:
        conn.cursor_factory = None

    sent = [s for s in _normalised(_RecordingCursor.log, schema) if "dbx_src_keys" in s[1]]
    assert sent == [
        ("execute", 'CREATE TEMP TABLE "dbx_src_keys_XXXXXXXX" ("id" TEXT)', None),
        (
            "copy",
            'COPY "dbx_src_keys_XXXXXXXX" ("id") FROM STDIN (FORMAT csv, DELIMITER \',\')',
            None,
        ),
        ("execute", 'CREATE INDEX ON "dbx_src_keys_XXXXXXXX" ("id")', None),
        ("execute", 'ANALYZE "dbx_src_keys_XXXXXXXX"', None),
        (
            "execute",
            'SELECT count(*) FROM "dbx_src_keys_XXXXXXXX" s WHERE NOT EXISTS '
            '(SELECT 1 FROM "<schema>"."cil" t WHERE t."id"::text = s."id")',
            None,
        ),
        (
            "execute",
            'UPDATE "<schema>"."cil" tgt SET "_deleted_in_source" = FALSE, "_timestamp" = '
            'NOW()::text FROM "dbx_src_keys_XXXXXXXX" src WHERE tgt."id"::text = src."id" '
            'AND tgt."_deleted_in_source" IS TRUE',
            (),
        ),
        (
            "execute",
            'UPDATE "<schema>"."cil" tgt SET "_deleted_in_source" = TRUE, "_timestamp" = '
            'NOW()::text WHERE tgt."_deleted_in_source" IS FALSE AND NOT EXISTS '
            '(SELECT 1 FROM "dbx_src_keys_XXXXXXXX" src WHERE src."id" = tgt."id"::text)',
            (),
        ),
        ("execute", 'DROP TABLE IF EXISTS "dbx_src_keys_XXXXXXXX"', None),
    ]


# --- Refusals -----------------------------------------------------------------


def test_a_failed_scan_marks_nothing(conn, schema) -> None:
    """A partial key set would flag every key not yet read — so the run raises."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)

    def boom(engine, sql, batch_size):
        raise RuntimeError("connection lost")
        yield  # pragma: no cover

    source.iter_batches = boom  # type: ignore[method-assign]

    with pytest.raises(Exception, match="connection lost"):
        KeySyncStrategy().run(_ctx(conn, schema, source, settings={"read_retry_attempts": 1}))

    assert _flags(conn, schema) == [(1, False), (2, False), (3, False)]


def test_an_empty_source_is_refused_unless_empty_rows_ok(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.rows = source.rows.iloc[0:0]

    with pytest.raises(StrategyError, match="returned no keys"):
        KeySyncStrategy().run(_ctx(conn, schema, source))
    assert _flags(conn, schema) == [(1, False), (2, False), (3, False)]

    result = KeySyncStrategy().run(
        _ctx(conn, schema, source, table_cfg={"empty_rows_ok": True}),
    )
    assert result.phase_metrics["deleted_marked"] == 3


def test_a_missing_target_is_reported_and_nothing_is_created(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")

    result = KeySyncStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason == "the target table does not exist"
    assert result.rows_read == 0
    assert result.data_present is False
    assert not source.scan_sql, "the source must not be touched when there is nothing to compare"
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'"{schema}"."cil"',))
        assert cur.fetchone()[0] is None


def test_a_surrogate_key_is_refused(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    ctx = _ctx(conn, schema, source)
    ctx.surrogate = SurrogateKey(enabled=True, alias="id", expr="a || b")

    with pytest.raises(StrategyError, match="surrogate key is not supported"):
        KeySyncStrategy().run(ctx)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"primary_column": None}, "is missing from LOAD_SETTINGS"),
        ({"primary_column": "nope"}, "is not among the table's columns"),
    ],
)
def test_validation(conn, schema, override, message) -> None:
    source = FakeHashSource(_source_rows(), pk="id")

    with pytest.raises(StrategyError, match=message):
        KeySyncStrategy().run(_ctx(conn, schema, source, settings=override))


def test_it_is_registered_under_its_name() -> None:
    assert isinstance(resolve_strategy("key_sync"), KeySyncStrategy)
    assert isinstance(resolve_strategy(" Key_Sync "), KeySyncStrategy)
