"""Tests for `FullBySourceStrategy` against a live target PostgreSQL.

The three most valuable ones here: that one source's slice is replaced **without
touching the others**, that the fingerprint does not skip a source whose data is
missing from the target, and that the fingerprint is written only after a
successful transfer.

Partitioning is tested against a real database -- a mock would only verify string
assembly, and the part that matters (that `TRUNCATE` on a partition empties
exactly one slice) would pass even when broken.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import StrategyError, TargetRef
from dbextractors.core.strategies.full_by_source import FullBySourceStrategy
from fakes import SOURCE_HASH_PREFIX, FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "int", "INTEGER"),
    ("label", "varchar", "TEXT"),
]


def _source_rows(ids=(1, 2), prefix="a") -> pd.DataFrame:
    return pd.DataFrame({"id": list(ids), "label": [f"{prefix}{i}" for i in ids]})


def _ctx(conn, schema, dialect, *, label, settings=None, first=True, last=True, fingerprint=None):
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"primary_column": "id", **(settings or {})},
    )
    ctx.source_label = label
    ctx.is_first_source = first
    ctx.is_last_source = last
    ctx.fingerprint_cfg = dict(fingerprint or {})
    return ctx


def _rows(conn, schema) -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, label, "_source" FROM "{schema}"."cil" ORDER BY "_source", id')
        return cur.fetchall()


def _load(conn, schema, label, data, **kwargs):
    source = FakeHashSource(data, pk="id")
    result = FullBySourceStrategy().run(_ctx(conn, schema, source, label=label, **kwargs))
    return source, result


# --- Replacing a slice ------------------------------------------------------


def test_two_sources_into_one_table(conn, schema) -> None:
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"), last=False)
    _load(conn, schema, "source_b", _source_rows((1, 2), "b"), first=False)

    assert _rows(conn, schema) == [
        (1, "a1", "source_a"),
        (2, "a2", "source_a"),
        (1, "b1", "source_b"),
        (2, "b2", "source_b"),
    ]


def test_replacing_a_slice_leaves_the_others_alone(conn, schema) -> None:
    """This is why the strategy exists -- otherwise `full` would do."""
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"), last=False)
    _load(conn, schema, "source_b", _source_rows((1, 2), "b"), first=False)

    _load(conn, schema, "source_b", _source_rows((7, 8), "b"), first=False)

    assert _rows(conn, schema) == [
        (1, "a1", "source_a"),
        (2, "a2", "source_a"),
        (7, "b7", "source_b"),
        (8, "b8", "source_b"),
    ]


def test_the_same_key_in_two_sources_does_not_collide(conn, schema) -> None:
    """The unique index has to be over ``(pk, _source)``, not over the key alone."""
    _load(conn, schema, "source_a", _source_rows((1,), "a"), last=False)
    _load(conn, schema, "source_b", _source_rows((1,), "b"), first=False)

    assert len(_rows(conn, schema)) == 2


def test_without_a_source_label_no_source_column_is_added(conn, schema) -> None:
    """A single-source table must not gain an extra column -- the naming contract."""
    _load(conn, schema, None, _source_rows((1, 2), "a"))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'cil'",
            (schema,),
        )
        assert "_source" not in {r[0] for r in cur.fetchall()}


def test_an_empty_source_clears_only_its_own_slice(conn, schema) -> None:
    """The database was emptied upstream -- its rows go, other sources' rows stay."""
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"), last=False)
    _load(conn, schema, "source_b", _source_rows((1, 2), "b"), first=False)

    _load(conn, schema, "source_b", pd.DataFrame({"id": [], "label": []}), first=False)

    assert _rows(conn, schema) == [(1, "a1", "source_a"), (2, "a2", "source_a")]


def test_an_empty_source_without_a_target_fails(conn, schema) -> None:
    """An empty source plus a missing target must not report success."""
    source = FakeHashSource(pd.DataFrame({"id": [], "label": []}), pk="id")

    with pytest.raises(StrategyError, match="empty_rows_ok"):
        FullBySourceStrategy().run(_ctx(conn, schema, source, label="source_a"))


def test_a_source_that_streamed_nothing_does_not_replace_the_slice(conn, schema) -> None:
    """The estimate reported rows, the source returned none -- the slice must not be emptied."""
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"))
    before = _rows(conn, schema)

    source = FakeHashSource(pd.DataFrame({"id": [], "label": []}), pk="id")
    source._rows = 2  # the estimate ran before the source was emptied
    with pytest.raises(StrategyError, match="streamed none"):
        FullBySourceStrategy().run(_ctx(conn, schema, source, label="source_a"))

    conn.rollback()
    assert _rows(conn, schema) == before


def test_a_failure_midway_through_the_stream_leaves_every_slice_alone(conn, schema) -> None:
    """A stream that dies mid-way must take neither its own slice nor anyone else's.

    Documented in `_replace_slice`: the source is streamed into a staging table
    and only the closing swap touches the target, so this is the one strategy
    where a failure cannot even empty the slice being rewritten -- the other
    databases share the table, and their rows have nothing to do with this
    source's outage.
    """
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"), last=False)
    _load(conn, schema, "source_b", _source_rows((3, 4), "b"), first=False)
    before = _rows(conn, schema)

    class ExplodesMidStream(FakeHashSource):
        def iter_batches(self, engine, sql, batch_size):
            if "fp_count" in sql:
                yield from super().iter_batches(engine, sql, batch_size)
                return
            for batch in super().iter_batches(engine, sql, batch_size):
                yield batch
                raise RuntimeError("the connection dropped mid-stream")

    source = ExplodesMidStream(_source_rows((1, 2), "new"), pk="id")
    with pytest.raises(RuntimeError, match="dropped mid-stream"):
        FullBySourceStrategy().run(_ctx(conn, schema, source, label="source_a", first=False))

    # Before this test rolls anything back: a rollback here would clean up after
    # the strategy and hide a staging table it left behind, which is visible only
    # inside its own transaction.
    assert _rows(conn, schema) == before
    assert target_pg.find_stale_shadows(conn, schema) == []


# --- Partitioning -----------------------------------------------------------


def test_a_partition_is_created_and_replaced(conn, schema) -> None:
    settings = {"partition_by_source": True}
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"), settings=settings, last=False)
    _load(conn, schema, "source_b", _source_rows((1, 2), "b"), settings=settings, first=False)

    target = TargetRef(schema=schema, table="cil")
    assert target_pg.is_partitioned(conn, target)
    assert len(_rows(conn, schema)) == 4

    _load(conn, schema, "source_b", _source_rows((9,), "b"), settings=settings, first=False)

    assert _rows(conn, schema) == [
        (1, "a1", "source_a"),
        (2, "a2", "source_a"),
        (9, "b9", "source_b"),
    ]


def test_partitioning_does_not_rebuild_an_existing_plain_table(conn, schema) -> None:
    """Rebuilding the target would drop views -- reality beats configuration."""
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"))
    target = TargetRef(schema=schema, table="cil")
    assert not target_pg.is_partitioned(conn, target)

    _, result = _load(
        conn, schema, "source_a", _source_rows((3,), "a"), settings={"partition_by_source": True}
    )

    assert result.phase_metrics["partitioned"] is False
    assert not target_pg.is_partitioned(conn, target)


def test_partition_by_source_without_a_label_fails(conn, schema) -> None:
    source = FakeHashSource(_source_rows(), pk="id")

    with pytest.raises(StrategyError, match="source_label"):
        FullBySourceStrategy().run(
            _ctx(conn, schema, source, label=None, settings={"partition_by_source": True})
        )


def test_the_partition_name_is_safe(conn, schema) -> None:
    """PostgreSQL truncates anything over 63 characters silently -- two sources would merge."""
    target = TargetRef(schema=schema, table="very_long_target_table_name_in_the_warehouse")
    long_name = "very_long_source_database_name_that_does_not_fit_the_limit"

    name = target_pg.partition_name(target, long_name)

    assert len(name) <= target_pg.MAX_IDENTIFIER_LENGTH
    assert name != target_pg.partition_name(target, long_name + "_other")


# --- Fingerprints -----------------------------------------------------------


FINGERPRINT = {"mode": "count", "primary_column": "id"}


def test_an_unchanged_source_is_skipped(conn, schema) -> None:
    source, _ = _load(conn, schema, "source_a", _source_rows((1, 2), "a"), fingerprint=FINGERPRINT)

    ctx = _ctx(conn, schema, source, label="source_a", fingerprint=FINGERPRINT)
    result = FullBySourceStrategy().run(ctx)

    assert result.load_method == "unchanged"
    assert result.phase_metrics["skipped"] is True
    assert result.rows_read == 0


def test_a_changed_source_is_downloaded(conn, schema) -> None:
    source, _ = _load(conn, schema, "source_a", _source_rows((1, 2), "a"), fingerprint=FINGERPRINT)
    source.insert({"id": 3, "label": "a3"})

    ctx = _ctx(conn, schema, source, label="source_a", fingerprint=FINGERPRINT)
    result = FullBySourceStrategy().run(ctx)

    assert result.rows_written == 3
    assert result.load_method == "full_by_source"


def test_a_matching_fingerprint_with_a_missing_slice_downloads_again(conn, schema) -> None:
    """A documented case from the predecessor: this left 3 rows out of 220 972 in the target.

    The first database of a rebuilt table recreates it, so from the second one on the
    table is back -- and the fingerprint alone would then skip every remaining database.
    """
    source, _ = _load(conn, schema, "source_a", _source_rows((1, 2), "a"), fingerprint=FINGERPRINT)
    with conn.cursor() as cur:
        cur.execute(f'DELETE FROM "{schema}"."cil" WHERE "_source" = %s', ("source_a",))
    conn.commit()

    ctx = _ctx(conn, schema, source, label="source_a", fingerprint=FINGERPRINT)
    result = FullBySourceStrategy().run(ctx)

    assert result.load_method == "full_by_source", "the slice is missing, skipping is not allowed"
    assert len(_rows(conn, schema)) == 2


def test_the_fingerprint_is_written_only_after_the_transfer(conn, schema) -> None:
    """If it were written earlier, an interrupted run would skip the source next time."""
    source = FakeHashSource(_source_rows((1, 2), "a"), pk="id")

    class FailsWhileReading(type(source)):  # type: ignore[misc]
        def iter_batches(self, engine, sql, batch_size):
            if "fp_count" in sql:
                yield from super().iter_batches(engine, sql, batch_size)
                return
            raise RuntimeError("the connection dropped")

    ctx = _ctx(conn, schema, source, label="source_a", fingerprint=FINGERPRINT)
    source.__class__ = FailsWhileReading
    with pytest.raises(RuntimeError, match="dropped"):
        FullBySourceStrategy().run(ctx)

    conn.rollback()
    target = TargetRef(schema=schema, table="cil")
    assert target_pg.read_fingerprint(conn, target, "source_a") is None


def test_force_reload_ignores_the_fingerprint(conn, schema) -> None:
    source, _ = _load(conn, schema, "source_a", _source_rows((1, 2), "a"), fingerprint=FINGERPRINT)

    settings = {"force_reload": True}
    ctx = _ctx(conn, schema, source, label="source_a", settings=settings, fingerprint=FINGERPRINT)
    result = FullBySourceStrategy().run(ctx)

    assert result.load_method == "full_by_source"


def test_a_failing_fingerprint_does_not_bring_the_run_down(conn, schema) -> None:
    """The fingerprint is an optimisation. When it fails, the load runs normally -- just dearer."""
    source = FakeHashSource(_source_rows((1, 2), "a"), pk="id")

    class FingerprintFails(type(source)):  # type: ignore[misc]
        def render_fingerprint(self, ref, **kwargs):
            return "SELECT this_is_not_sql"

        def iter_batches(self, engine, sql, batch_size):
            if "this_is_not_sql" in sql:
                raise RuntimeError("the fingerprint failed")
            yield from super().iter_batches(engine, sql, batch_size)

    source.__class__ = FingerprintFails
    ctx = _ctx(conn, schema, source, label="source_a", fingerprint=FINGERPRINT)

    result = FullBySourceStrategy().run(ctx)

    assert result.rows_written == 2


def test_a_failed_fingerprint_read_does_not_leave_the_transaction_aborted(
    conn, schema, monkeypatch
) -> None:
    """A swallowed exception has to clean the connection up, or something unrelated fails later.

    The target runs with autocommit off, so a failed statement leaves the transaction
    aborted. Without ``rollback()`` the next statement on the same connection
    (`table_exists`) failed with ``InFailedSqlTransaction`` -- an error unrelated to the
    original one. That is exactly how a production pipeline fell over.
    """
    source, _ = _load(conn, schema, "source_a", _source_rows((1, 2), "a"), fingerprint=FINGERPRINT)

    def failing_read(target_conn, target, source_label):
        with target_conn.cursor() as cur:
            cur.execute("SELECT this_is_not_a_column")

    monkeypatch.setattr(target_pg, "read_fingerprint", failing_read)
    ctx = _ctx(conn, schema, source, label="source_a", fingerprint=FINGERPRINT)

    result = FullBySourceStrategy().run(ctx)

    assert result.load_method == "full_by_source", "the fingerprint was not read, so download"
    assert result.rows_written == 2


# --- row_hash: the column yes, the value no ---------------------------------


def _hashes(conn, schema) -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT row_hash FROM "{schema}"."cil" ORDER BY id')
        return [r[0] for r in cur.fetchall()]


def test_the_row_hash_column_exists_but_stays_empty(conn, schema) -> None:
    """The only strategy where the column exists and the value is not computed.

    The predecessor leaves it out on this one path deliberately -- "no row hashing...
    the cheapest shape a memory-constrained SQL Server can serve". It still creates the
    column, so the target has to have it too: the dbt layer counts on tables having it.
    """
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"))

    assert _hashes(conn, schema) == [None, None]


def test_compute_row_hash_turns_it_on(conn, schema) -> None:
    """A new capability is an option in LOAD_SETTINGS with a safe default."""
    _load(conn, schema, "source_a", _source_rows((1, 2), "a"), settings={"compute_row_hash": True})

    hashes = _hashes(conn, schema)
    # The prefix proves the value came through the **source** path rather than
    # being recomputed in pandas — see `fakes.SOURCE_HASH_PREFIX`. Without that
    # check this only asserted "something hash-shaped is there", which a switch
    # of path would have satisfied just as well.
    assert all(h.startswith(SOURCE_HASH_PREFIX) for h in hashes), hashes
    assert all(len(h) == len(SOURCE_HASH_PREFIX) + 64 for h in hashes), hashes
    assert hashes[0] != hashes[1], "different rows must have different hashes"


def test_the_other_strategies_still_compute_the_hash(conn, schema) -> None:
    """The inverted default **must not** leak into `full`."""
    from dbextractors.core.strategies.full import FullLoadStrategy

    source = FakeHashSource(_source_rows((1, 2), "a"), pk="id")
    FullLoadStrategy().run(_ctx(conn, schema, source, label=None))

    with conn.cursor() as cur:
        cur.execute(f'SELECT count(row_hash) FROM "{schema}"."cil"')
        assert cur.fetchone()[0] == 2


def test_the_default_survives_a_real_configuration(conn, schema) -> None:
    """**A regression caught by the golden test.** A fake context cannot see this bug.

    `LoadSettingsConfig` is a dataclass, so `dataclasses.asdict` puts the
    ``compute_row_hash`` key into `ctx.settings` **every time**. As long as it had a
    plain default of ``True``, the inverted default of `full_by_source` never applied --
    and that only showed up against a live source, not here.
    """
    from dbextractors.core import config as config_module
    from dbextractors.core.strategies.full import _hash_column
    from dbextractors.core.strategies.full_by_source import _hash_plan
    from dbextractors.entrypoint import _settings_dict

    parsed = config_module.parse(
        {
            "TABLE": {"source_name": "t", "output_name": "cil", "output_schema": schema},
            "SOURCE_DB": {"user": "u", "password": "p", "database": "d", "host": "h"},
            "LOAD_SETTINGS": {"load_method": "full_by_source", "primary_column": "id"},
        }
    )
    settings = _settings_dict(parsed.load_settings)
    assert settings["compute_row_hash"] is None, "a silent configuration must stay silent"

    source = FakeHashSource(_source_rows((1,), "a"), pk="id")
    ctx = _ctx(conn, schema, source, label="source_a", settings=settings)

    assert _hash_plan(ctx) == ("row_hash", False)
    # And `full` with the same configuration still computes the hash.
    assert _hash_column(ctx) == "row_hash"


def test_turning_it_off_in_the_configuration_drops_the_column_only_for_full(conn, schema) -> None:
    """``compute_row_hash: false`` removes the column for `full`, but not here.

    For `full_by_source` the column stays, because the predecessor creates it too --
    just without a value.
    """
    from dbextractors.core.strategies.full import _hash_column
    from dbextractors.core.strategies.full_by_source import _hash_plan

    source = FakeHashSource(_source_rows((1,), "a"), pk="id")
    ctx = _ctx(conn, schema, source, label="source_a", settings={"compute_row_hash": False})

    assert _hash_column(ctx) is None
    assert _hash_plan(ctx) == ("row_hash", False)
