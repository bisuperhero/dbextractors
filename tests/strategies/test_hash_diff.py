"""Tests for `HashDiffStrategy` against a live target PostgreSQL.

This strategy serves ~530 of the ~670 tables, so it is tested differently from
the others: the source table **really changes** between two runs
(`FakeHashSource` holds the data and answers both kinds of query) and the checks
cover not only the result in the target but also **how many rows were
transferred**. Without the second part, an implementation that quietly transfers
the whole table every time would pass as well.

The most valuable one here is `test_second_run_with_no_changes_transfers_nothing`
-- it is the gate for the whole strategy and at the same time the one property
the current code has no way of breaking loudly: if the hashes drifted apart, the
run is green and everything simply gets transferred.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core.strategies.base import StrategyError
from dbextractors.core.strategies.full import FullLoadStrategy
from dbextractors.core.strategies.hash_diff import HashDiffStrategy
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


def _ctx(conn, schema, dialect, *, settings=None, columns=None, where=None):
    return make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(columns or COLUMNS),
        settings={**SETTINGS, **(settings or {})},
        where=where,
    )


def _seed(conn, schema, dialect, **kwargs) -> None:
    """A full load, so the diff has something to build on -- and above all so the
    hash is in the target.

    This is the part everything else rests on: the full load has to fill
    `row_hash` by **the same route** the diff later computes it. If it did not,
    the second run would find absolutely everything changed.
    """
    FullLoadStrategy().run(_ctx(conn, schema, dialect, **kwargs))


def _rows(conn, schema, table="cil") -> list:
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT id, "_name", price, "_deleted_in_source" '
            f'FROM "{schema}"."{table}" ORDER BY id'
        )
        return cur.fetchall()


def _hashes(conn, schema, table="cil") -> dict:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, row_hash FROM "{schema}"."{table}"')
        return dict(cur.fetchall())


# --- Basic behaviour --------------------------------------------------------


def test_second_run_with_no_changes_transfers_nothing(conn, schema) -> None:
    """The gate for this strategy: with no source changes, not a single row.

    If the hash were computed one way while seeding and another way in the diff,
    this test fails on `rows_read` -- and it is the only place where that shows.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    before = _hashes(conn, schema)

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 0
    assert result.rows_written == 0
    assert result.fallback_reason is None
    assert result.load_method == "hash"
    assert result.phase_metrics["matched"] == 3
    assert result.phase_metrics["changed"] == 0
    assert _hashes(conn, schema) == before
    assert not source.download_sql, "with no changes the source must not be touched twice"


def test_the_seed_stores_the_source_hash_not_the_pandas_one(conn, schema) -> None:
    """The full load that seeds `row_hash` must take the **source** path.

    The two real paths do not agree (4 of 51 tables, see ``core/hashing.py``), so
    a seed that quietly recomputed the hash in pandas (``force_recompute=True``
    on a dialect whose source computes it) would store fingerprints no later diff
    can match — and the whole table would transfer forever. The fake's
    source-side hash carries a recognisable prefix precisely so this test can
    tell the two paths apart; without it,
    `test_second_run_with_no_changes_transfers_nothing` passed even when both
    sides had silently switched to pandas together.
    """
    from fakes import SOURCE_HASH_PREFIX

    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)

    hashes = _hashes(conn, schema)

    assert hashes, "the seed stored no hashes at all"
    assert all(
        h is not None and h.startswith(SOURCE_HASH_PREFIX) for h in hashes.values()
    ), f"the seed did not go through the source path: {hashes}"


def test_only_the_changed_row_is_transferred(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.update(2, "name", "changed")

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 1, "only the changed row should be fetched, not the whole table"
    assert result.rows_written == 1
    assert result.phase_metrics == {
        "source_rows": 3,
        "added": 0,
        "changed": 1,
        "matched": 2,
        "deleted_marked": 0,
    }
    assert _rows(conn, schema) == [
        (1, "row 1", 10.0, False),
        (2, "changed", 20.0, False),
        (3, "row 3", 30.0, False),
    ]


def test_a_changed_row_gets_a_new_hash(conn, schema) -> None:
    """Otherwise the next run would find it changed again -- and so on forever."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.update(2, "name", "changed")
    HashDiffStrategy().run(_ctx(conn, schema, source))

    second = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert second.rows_read == 0
    assert second.phase_metrics["matched"] == 3


def test_a_new_row_is_added(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.insert({"id": 4, "name": "new", "price": 40.0})

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.phase_metrics["added"] == 1
    assert result.rows_written == 1
    assert _rows(conn, schema)[-1] == (4, "new", 40.0, False)


def test_a_deleted_row_is_marked(conn, schema) -> None:
    """`_deleted_in_source` is mandatory for **every** strategy.

    A vanishing row leaves no trace in the hashes -- it is recognised only by its
    key being absent from the source snapshot.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.delete(2)

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.phase_metrics["deleted_marked"] == 1
    assert _rows(conn, schema) == [
        (1, "row 1", 10.0, False),
        (2, "row 2", 20.0, True),
        (3, "row 3", 30.0, False),
    ]


def test_a_returning_row_is_unmarked(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.delete(2)
    HashDiffStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": 2, "name": "row 2", "price": 20.0})

    HashDiffStrategy().run(_ctx(conn, schema, source))

    assert _rows(conn, schema)[1] == (2, "row 2", 20.0, False)


def test_deletions_are_marked_even_when_nothing_else_changed(conn, schema) -> None:
    """A run without a single hash change must not skip marking deletions."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.delete(3)

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.rows_written == 0
    assert result.phase_metrics["deleted_marked"] == 1
    assert _rows(conn, schema)[2][3] is True


# --- Fetching in batches ----------------------------------------------------


def test_fetching_happens_in_batches(conn, schema) -> None:
    """``hash_download_batch_size`` controls how many keys go into one ``IN (...)``.

    Five of seven rows change, not all of them: if not a single one matched, the
    emergency brake would fire and it would run as a full load (see
    `test_drifted_hashes_trigger_a_full_load_and_reseed`).
    """
    source = FakeHashSource(_source_rows(range(1, 8)), pk="id")
    _seed(conn, schema, source)
    for key in range(1, 6):
        source.update(key, "name", f"change {key}")

    result = HashDiffStrategy().run(
        _ctx(conn, schema, source, settings={"hash_download_batch_size": 2})
    )

    assert result.rows_read == 5
    assert len(source.download_sql) == 3, "five keys two at a time are three queries"
    assert len(_rows(conn, schema)) == 7


def test_the_where_clause_applies_to_the_fetch_as_well(conn, schema) -> None:
    """The configured condition has to appear in both queries, not just the scan.

    If it applied only to the scan, the fetch could pull in a row that does not
    belong in the target at all.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source, where="1=1")
    source.update(2, "name", "changed")

    HashDiffStrategy().run(_ctx(conn, schema, source, where="1=1"))

    assert all("1=1" in sql for sql in source.scan_sql)
    assert all("1=1" in sql for sql in source.download_sql)


def test_a_reserved_key_name(conn, schema) -> None:
    """PK ``order`` is ``_order`` in the target, but ``order`` in the source query.

    The frozen target column-naming contract turns a reserved word into a column
    with a leading underscore. The strategy works with the key in target
    vocabulary, so it has to be translated back before querying the source --
    otherwise it would ask for a column the source does not have.
    """
    columns = [("order", "int", "INTEGER"), ("name", "varchar", "TEXT")]
    data = pd.DataFrame({"order": [1, 2], "name": ["a", "b"]})
    source = FakeHashSource(data, pk="order")
    settings = {"primary_column": "order"}

    _seed(conn, schema, source, settings=settings, columns=columns)
    source.update(2, "name", "changed")
    result = HashDiffStrategy().run(_ctx(conn, schema, source, settings=settings, columns=columns))

    assert result.rows_written == 1
    assert all("`order` IN (" in sql for sql in source.download_sql)
    with conn.cursor() as cur:
        cur.execute(f'SELECT "_order", "_name" FROM "{schema}"."cil" ORDER BY "_order"')
        assert cur.fetchall() == [(1, "a"), (2, "changed")]


# --- Falling back to a full load --------------------------------------------


def test_a_missing_target_falls_back_to_full(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason == "the target table does not exist"
    assert result.rows_written == 3
    assert len(_rows(conn, schema)) == 3


def test_an_empty_target_falls_back_to_full(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE "{schema}"."cil"')
    conn.commit()

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason == "the target table is empty"
    assert len(_rows(conn, schema)) == 3


def test_an_all_null_hash_falls_back_to_full(conn, schema) -> None:
    """A target without hashes would look entirely changed -- a full load is cheaper."""
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    with conn.cursor() as cur:
        cur.execute(f'UPDATE "{schema}"."cil" SET row_hash = NULL')
    conn.commit()

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is not None
    assert "all NULL" in result.fallback_reason
    assert all(h is not None for h in _hashes(conn, schema).values())


def test_drifted_hashes_trigger_a_full_load_and_reseed(conn, schema) -> None:
    """When not a single row matches and almost all differ, the hashes have drifted.

    That is a different state from "the data changed": fetching the whole table
    in batches through ``IN (...)`` would be many times more expensive than a
    full load, and the hashes would stay drifted next time as well.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    with conn.cursor() as cur:
        cur.execute(f'UPDATE "{schema}"."cil" SET row_hash = \'other path\'')
    conn.commit()

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is not None
    assert "drifted" in result.fallback_reason
    assert not source.download_sql, "instead of fetching by key it should run a full load"
    # And after the reseed, the second run transfers nothing.
    assert HashDiffStrategy().run(_ctx(conn, schema, source)).rows_read == 0


def test_a_partial_mismatch_does_not_trigger_full(conn, schema) -> None:
    """The counterpart of the previous test: 1 of 3 rows differing is ordinary change.

    Without this test, a brake that fires all the time would pass too.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    with conn.cursor() as cur:
        cur.execute(f'UPDATE "{schema}"."cil" SET row_hash = \'other\' WHERE id = 2')
    conn.commit()

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is None
    assert result.rows_read == 1


def test_a_failed_scan_falls_back_to_full(conn, schema) -> None:
    """A transient loss of connection mid-scan is not a failed run but a full load.

    The predecessor logs it as a warning on purpose (so it does not raise a false
    error in the error tracker); on top of that it has to be visible in
    `fallback_reason`.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.scan_error = RuntimeError("Lost connection to MySQL server")

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is not None
    assert "Lost connection" in result.fallback_reason
    assert len(_rows(conn, schema)) == 3


def test_ignore_hash_falls_back_to_full(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)

    result = HashDiffStrategy().run(_ctx(conn, schema, source, settings={"ignore_hash": True}))

    assert result.fallback_reason is not None
    assert "ignore_hash" in result.fallback_reason


def test_a_failure_during_the_fetch_leaves_the_target_and_no_debris(conn, schema) -> None:
    """A crash while fetching the changed rows must not leave half a diff behind.

    The fetch goes through a staging table and only an ``INSERT … ON CONFLICT``
    projects it into the target, so a failure mid-fetch has to leave the target
    exactly as it was — data, hashes and `_deleted_in_source` alike — and no
    stale staging table. Unlike a failed *scan* (which falls back to a full
    load), a failed fetch propagates: at that point the diff already touched the
    target connection and the only safe answer is a rollback.
    """
    from dbextractors.core import target_pg

    class ExplodesOnDownload(FakeHashSource):
        def iter_batches(self, engine, sql, batch_size):
            if " IN (" in sql:
                raise RuntimeError("the connection dropped during the fetch")
            yield from super().iter_batches(engine, sql, batch_size)

    source = ExplodesOnDownload(_source_rows(), pk="id")
    _seed(conn, schema, source)
    before_rows = _rows(conn, schema)
    before_hashes = _hashes(conn, schema)
    source.update(2, "name", "changed")

    with pytest.raises(RuntimeError, match="dropped during the fetch"):
        HashDiffStrategy().run(_ctx(conn, schema, source))

    # Checked **before** this test rolls anything back itself. Rolling back first
    # cleans up after the strategy, and the assertions then hold even for a
    # strategy that cleans up nothing: an uncommitted staging table is visible to
    # its own transaction, so a missing rollback shows in `find_stale_shadows`
    # only while that transaction is still open. In a real run the caller carries
    # on with the same connection, which is exactly this state — and the rollback
    # in the middle is why the mutation "drop the cleanup from `_download`" used
    # to survive.
    assert _rows(conn, schema) == before_rows
    assert _hashes(conn, schema) == before_hashes
    assert target_pg.find_stale_shadows(conn, schema) == []
    conn.rollback()


# --- Validation -------------------------------------------------------------


def test_a_missing_primary_key_fails(conn, schema) -> None:
    """**A deviation from the predecessor**, which quietly runs a full load when
    the PK is missing.

    A full load is the most expensive possible answer to a typo in YAML.
    """
    source = FakeHashSource(_source_rows(), pk="id")
    ctx = make_context(conn, schema, dialect=source, columns=make_columns(COLUMNS), settings={})

    with pytest.raises(StrategyError, match="primary_column"):
        HashDiffStrategy().run(ctx)


def test_a_key_outside_the_columns_fails(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")
    ctx = _ctx(conn, schema, source, settings={"primary_column": "does_not_exist"})

    with pytest.raises(StrategyError, match="is not among the table's columns"):
        HashDiffStrategy().run(ctx)


def test_a_surrogate_key_passes_even_though_the_source_lacks_it(conn, schema) -> None:
    """A surrogate key is computed by an expression, so it is never among the
    source columns.

    The case that surfaced this was a junction table with no `id` of its own: the
    key `_uni_surrogate_key` is built by `CONCAT` over three columns. Validation
    looked for it among the introspected columns, and the pipeline could not be
    started because of it.
    """
    import dataclasses

    from dbextractors.dialects.base import SurrogateKey

    source = FakeHashSource(_source_rows(), pk="id")
    ctx = _ctx(conn, schema, source, settings={"primary_column": "_uni_surrogate_key"})
    ctx = dataclasses.replace(
        ctx,
        surrogate=SurrogateKey(
            enabled=True,
            alias="_uni_surrogate_key",
            expr="CONCAT(product_id, '|', parameter_id, '|', position)",
        ),
    )

    HashDiffStrategy().validate(ctx)


def test_a_surrogate_with_a_different_alias_does_not_excuse_an_unknown_key(conn, schema) -> None:
    """The exception applies only to the key the surrogate really produces."""
    import dataclasses

    from dbextractors.dialects.base import SurrogateKey

    source = FakeHashSource(_source_rows(), pk="id")
    ctx = _ctx(conn, schema, source, settings={"primary_column": "does_not_exist"})
    ctx = dataclasses.replace(
        ctx,
        surrogate=SurrogateKey(enabled=True, alias="_uni_surrogate_key", expr="CONCAT(a, b)"),
    )

    with pytest.raises(StrategyError, match="is not among the table's columns"):
        HashDiffStrategy().validate(ctx)


def test_a_disabled_hash_fails(conn, schema) -> None:
    """``compute_row_hash=False`` and a hash diff are mutually exclusive -- say so
    straight away."""
    source = FakeHashSource(_source_rows(), pk="id")
    ctx = _ctx(conn, schema, source, settings={"compute_row_hash": False})

    with pytest.raises(StrategyError, match="compute_row_hash"):
        HashDiffStrategy().run(ctx)


def test_a_missing_unique_index_fails_before_the_source_is_read(conn, schema) -> None:
    """Without the index the upsert does not work -- and it shows before the
    source is touched."""
    from dbextractors.core import target_pg
    from dbextractors.core.strategies.base import TargetRef

    source = FakeHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    with conn.cursor() as cur:
        name = target_pg.index_name(TargetRef(schema, "cil"), "id")
        cur.execute(f'DROP INDEX "{schema}".{name}')
    conn.commit()
    source.seen_sql.clear()

    with pytest.raises(target_pg.TargetError, match="UNIQUE"):
        HashDiffStrategy().run(_ctx(conn, schema, source))

    assert source.seen_sql == [], "the index must be checked before the source scan"


# --- A dialect that computes the hash in pandas -----------------------------


class PandasHashSource(FakeHashSource):
    """A source like PostgreSQL: it cannot compute the hash on its own side.

    This is not hypothetical -- it is the only way to match the fingerprints that
    are in the target today for ~58 PG pipelines.
    """

    hash_in_pandas = True

    def render_hash_expr(self, hashed_columns, alias, column_types=None):
        from dbextractors.dialects.base import UnsupportedFeature

        raise UnsupportedFeature("this source does not compute hashes")


def test_a_pandas_hash_works_end_to_end(conn, schema) -> None:
    """The seed and the diff have to take **the same** route, or the second run
    never settles.

    This is the test that would fail if the full load seeded the hash from the
    source expression while the diff computed it in pandas (or the other way
    round).
    """
    source = PandasHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 0
    assert result.phase_metrics["matched"] == 3
    assert not any("SHA2" in sql for sql in source.scan_sql)


def test_a_pandas_hash_finds_a_change(conn, schema) -> None:
    source = PandasHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source)
    source.update(2, "name", "changed")

    result = HashDiffStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 1
    assert _rows(conn, schema)[1] == (2, "changed", 20.0, False)


def test_a_pandas_hash_respects_the_column_selection(conn, schema) -> None:
    """``hash_exclude_columns`` has to apply where pandas computes the hash too.

    Without `hashed_columns_override` the whole batch would be hashed and a change
    in an excluded column would transfer the table for nothing.
    """
    settings = {"hash_exclude_columns": ["price"]}
    source = PandasHashSource(_source_rows(), pk="id")
    _seed(conn, schema, source, settings=settings)
    source.update(2, "price", 999.0)

    result = HashDiffStrategy().run(_ctx(conn, schema, source, settings=settings))

    assert result.rows_read == 0, "a change in an excluded column must not change the hash"
