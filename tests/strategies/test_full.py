"""Tests for `FullLoadStrategy` against a live target PostgreSQL.

The source is fake (see `conftest.FakeDialect`), the target is real -- the
strategy stands or falls on whether `COPY`, the shadow table and the atomic
swap actually work, and a mock proves none of that.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import StrategyError, resolve_strategy
from dbextractors.core.strategies.full import FullLoadStrategy
from fakes import FakeDialect, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [("id", "int", "INTEGER"), ("name", "varchar", "TEXT"), ("price", "decimal", "NUMERIC")]


def _batch(start: int, count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(start, start + count),
            "name": [f"row {i}" for i in range(start, start + count)],
            "price": [round(1.5 * i, 2) for i in range(start, start + count)],
        }
    )


def _rows(conn, schema: str, table: str = "cil") -> list:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_name", price FROM "{schema}"."{table}" ORDER BY id')
        return cur.fetchall()


def _columns(conn, schema: str, table: str = "cil") -> list:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (schema, table),
        )
        return [r[0] for r in cur.fetchall()]


def test_full_load_creates_the_target_and_writes(conn, schema) -> None:
    dialect = FakeDialect([_batch(1, 5)])
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))

    result = FullLoadStrategy().run(ctx)

    assert result.rows_read == 5
    assert result.rows_written == 5
    assert result.load_method == "full"
    assert len(_rows(conn, schema)) == 5


def test_column_order_and_names_follow_the_naming_contract(conn, schema) -> None:
    """``name`` is a reserved word, so in the target it must be ``_name``."""
    dialect = FakeDialect([_batch(1, 2)])
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))
    FullLoadStrategy().run(ctx)

    assert _columns(conn, schema) == [
        "id",
        "_name",
        "price",
        "row_hash",
        "_timestamp",
        "_deleted_in_source",
    ]


def test_several_batches_are_assembled_together(conn, schema) -> None:
    dialect = FakeDialect([_batch(1, 3), _batch(4, 3), _batch(7, 2)])
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))

    result = FullLoadStrategy().run(ctx)

    assert result.rows_written == 8
    assert [r[0] for r in _rows(conn, schema)] == list(range(1, 9))


def test_dedup_drops_a_repeated_key(conn, schema) -> None:
    """Unstable paging can hand back the same row twice."""
    dialect = FakeDialect([_batch(1, 3), _batch(2, 3)])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"primary_column": "id"},
    )

    result = FullLoadStrategy().run(ctx)

    assert result.rows_read == 6
    assert result.rows_written == 4
    assert [r[0] for r in _rows(conn, schema)] == [1, 2, 3, 4]


def test_second_run_replaces_the_target_without_doubling_it(conn, schema) -> None:
    columns = make_columns(COLUMNS)
    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(1, 5)]), columns=columns)
    )
    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(100, 2)]), columns=columns)
    )

    assert [r[0] for r in _rows(conn, schema)] == [100, 101]


def test_a_view_over_the_target_survives_the_second_run(conn, schema) -> None:
    """**This is the main reason there is no DROP CASCADE.**

    The current full load takes down 102 views. The new path has to leave them
    alive and bound to the target -- not to a set-aside copy.
    """
    columns = make_columns(COLUMNS)
    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(1, 3)]), columns=columns)
    )
    with conn.cursor() as cur:
        cur.execute(f'CREATE VIEW "{schema}"."v_target" AS SELECT id FROM "{schema}"."cil"')
    conn.commit()

    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(50, 2)]), columns=columns)
    )

    with conn.cursor() as cur:
        cur.execute(f'SELECT id FROM "{schema}"."v_target" ORDER BY id')
        assert [r[0] for r in cur.fetchall()] == [50, 51]


def test_failure_midway_leaves_the_target_untouched(conn, schema) -> None:
    """An interrupted run must not leave the target truncated -- documented at
    11.4 of 18.8 M rows."""
    columns = make_columns(COLUMNS)
    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(1, 4)]), columns=columns)
    )

    class Explodes(FakeDialect):
        def iter_batches(self, engine, sql, batch_size):
            yield _batch(100, 2)
            raise RuntimeError("the source failed midway")

    with pytest.raises(RuntimeError, match="failed midway"):
        FullLoadStrategy().run(
            make_context(conn, schema, dialect=Explodes(rows=6), columns=columns)
        )

    assert [r[0] for r in _rows(conn, schema)] == [1, 2, 3, 4]
    assert target_pg.find_stale_shadows(conn, schema) == []


def test_a_failed_size_estimate_fails_loudly(conn, schema) -> None:
    """Size-estimate failures must fail loudly. Never ``total_rows = 0``."""
    dialect = FakeDialect([_batch(1, 3)])
    dialect.estimate_error = RuntimeError("source unreachable")
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))

    with pytest.raises(RuntimeError, match="unreachable"):
        FullLoadStrategy().run(ctx)


def test_skip_size_estimate_asks_the_source_nothing_and_still_loads(conn, schema) -> None:
    """`skip_size_estimate` means "no estimate", it never means "zero rows".

    On a 15 M row table the `COUNT(*)` costs minutes and feeds nothing but the
    batch auto-tuning, so a pipeline is allowed to switch it off. What must not
    happen is the two being confused: no estimate is `None`, and a `0` would send
    the run down the empty-source branch -- where `empty_rows_ok` then replaces
    the target with an empty table and the run comes out green. That is rule 5's
    failure mode reached through a configuration key instead of through an error,
    and `empty_rows_ok` is set here so that nothing would stand in its way.
    """
    dialect = FakeDialect([_batch(1, 4)])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"skip_size_estimate": True},
        table_cfg={"empty_rows_ok": True},
    )

    result = FullLoadStrategy().run(ctx)

    assert dialect.estimate_calls == [], "the source must not be counted at all"
    assert result.rows_written == 4
    assert result.phase_metrics["estimated_rows"] is None
    assert len(_rows(conn, schema)) == 4


def test_an_estimate_with_rows_and_a_source_that_streams_none_keeps_the_target(
    conn, schema
) -> None:
    """The estimate says there are rows, the source hands over no batch at all.

    The contradiction is real: a connection that dies before the first batch, or
    a source whose data vanished between the `COUNT(*)` and the read. Only the
    estimate decides whether the source counts as empty, so without this branch
    the run would swap in a shadow table that was never filled -- and with
    `empty_rows_ok` set nothing else would stop it. That is rule 5's failure
    mode reached from the other side: not a zero from a failed estimate, but a
    zero from a read that never happened.

    `empty_rows_ok` is on here on purpose. It is the setting that permits an
    empty *source* to empty the target; it must not be read as permission to
    empty it after a read that produced nothing.
    """
    columns = make_columns(COLUMNS)
    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(1, 3)]), columns=columns)
    )

    result = FullLoadStrategy().run(
        make_context(
            conn,
            schema,
            dialect=FakeDialect([], rows=7),
            columns=columns,
            table_cfg={"empty_rows_ok": True},
        )
    )

    assert result.rows_written == 0
    assert result.data_present is False
    assert [r[0] for r in _rows(conn, schema)] == [1, 2, 3]
    assert target_pg.find_stale_shadows(conn, schema) == []


def test_empty_source_without_empty_rows_ok_does_not_wipe_the_target(conn, schema) -> None:
    """An empty read must not overwrite the target with an empty table."""
    columns = make_columns(COLUMNS)
    FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([_batch(1, 3)]), columns=columns)
    )

    result = FullLoadStrategy().run(
        make_context(conn, schema, dialect=FakeDialect([], rows=0), columns=columns)
    )

    assert result.data_present is False
    assert len(_rows(conn, schema)) == 3


def test_empty_source_with_empty_rows_ok_creates_an_empty_target(conn, schema) -> None:
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([], rows=0),
        columns=make_columns(COLUMNS),
        table_cfg={"empty_rows_ok": True},
    )
    result = FullLoadStrategy().run(ctx)

    assert result.rows_written == 0
    assert _columns(conn, schema)[:3] == ["id", "_name", "price"]
    assert len(_rows(conn, schema)) == 0


# --- Type promotion out of the first batch ----------------------------------
#
# Both promotions are decided by the **data**, because the catalogue cannot tell
# either of them: a MySQL `int` column may hold values that no longer fit into
# `INTEGER`, and an `enum('0','1')` arrives as `TEXT`. Their failure mode is
# silent until the offending value shows up -- `COPY` then stops with "value out
# of range" halfway through a batch and the whole run is lost.
#
# The tests therefore write values that would **not go through** without the
# promotion. Asserting only on the DDL would let a promotion pass that computes
# the right type and never applies it, which is precisely the mutation
# ("`_promote_types` returns the types unchanged") that survived the suite.


def _column_types(conn, schema: str, table: str = "cil") -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        )
        return dict(cur.fetchall())


def test_a_value_over_int_max_promotes_the_column_to_bigint(conn, schema) -> None:
    """The source calls it `int`, the batch holds a value `INTEGER` cannot store."""
    batch = _batch(1, 2)
    batch["id"] = [1, 5_000_000_000]
    ctx = make_context(conn, schema, dialect=FakeDialect([batch]), columns=make_columns(COLUMNS))

    result = FullLoadStrategy().run(ctx)

    assert result.rows_written == 2
    assert _column_types(conn, schema)["id"] == "bigint"
    assert [r[0] for r in _rows(conn, schema)] == [1, 5_000_000_000]


def test_a_value_just_under_the_threshold_leaves_the_column_alone(conn, schema) -> None:
    """The mirror image: promoting everything would be as wrong as promoting nothing.

    `INT_PROMOTION_THRESHOLD` is 90 % of `INT_MAX`, so the promotion has room
    before the first value that would actually overflow. A column that stays
    below it keeps `INTEGER` -- 4 bytes against 8 across ~670 tables.
    """
    from dbextractors.core.strategies.full import INT_PROMOTION_THRESHOLD

    batch = _batch(1, 2)
    batch["id"] = [1, INT_PROMOTION_THRESHOLD - 1]
    ctx = make_context(conn, schema, dialect=FakeDialect([batch]), columns=make_columns(COLUMNS))

    FullLoadStrategy().run(ctx)

    assert _column_types(conn, schema)["id"] == "integer"


def test_an_enum_of_zero_and_one_is_promoted_to_boolean(conn, schema) -> None:
    """`enum` maps to `TEXT`; that it is a flag can only be seen in the data.

    The dbt layer reads these columns as booleans, so a `TEXT` holding ``'0'``
    and ``'1'`` is not merely untidy -- it is a different type on the other side
    of the warehouse.
    """
    columns = [("id", "int", "INTEGER"), ("active", "enum", "TEXT")]
    batch = pd.DataFrame({"id": [1, 2], "active": [0, 1]})
    ctx = make_context(conn, schema, dialect=FakeDialect([batch]), columns=make_columns(columns))

    FullLoadStrategy().run(ctx)

    assert _column_types(conn, schema)["active"] == "boolean"
    with conn.cursor() as cur:
        cur.execute(f'SELECT active FROM "{schema}"."cil" ORDER BY id')
        assert cur.fetchall() == [(False,), (True,)]


def test_an_enum_with_another_value_stays_text(conn, schema) -> None:
    """Only ``0``/``1`` is a flag. Anything else and the promotion would lose data."""
    columns = [("id", "int", "INTEGER"), ("active", "enum", "TEXT")]
    batch = pd.DataFrame({"id": [1, 2, 3], "active": [0, 1, 2]})
    ctx = make_context(conn, schema, dialect=FakeDialect([batch]), columns=make_columns(columns))

    FullLoadStrategy().run(ctx)

    assert _column_types(conn, schema)["active"] == "text"


def test_the_hash_expression_is_appended_after_the_columns(conn, schema) -> None:
    dialect = FakeDialect([_batch(1, 2)])
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))
    FullLoadStrategy().run(ctx)

    sql = dialect.seen_sql[0]
    assert sql.startswith("SELECT `id`, `name`, `price`, SHA2(")
    assert " FROM `zdroj`" in sql


def test_compute_row_hash_false_turns_the_hash_off(conn, schema) -> None:
    dialect = FakeDialect([_batch(1, 2)])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"compute_row_hash": False},
    )
    FullLoadStrategy().run(ctx)

    assert "SHA2" not in dialect.seen_sql[0]
    assert "row_hash" not in _columns(conn, schema)


def test_deleted_in_source_is_false_after_a_full_load(conn, schema) -> None:
    """``_deleted_in_source`` is mandatory for every strategy -- ``full`` included,
    not just the hash path."""
    ctx = make_context(
        conn, schema, dialect=FakeDialect([_batch(1, 3)]), columns=make_columns(COLUMNS)
    )
    FullLoadStrategy().run(ctx)

    with conn.cursor() as cur:
        cur.execute(f'SELECT DISTINCT "_deleted_in_source" FROM "{schema}"."cil"')
        assert cur.fetchall() == [(False,)]


def test_a_unique_index_over_the_pk_is_created(conn, schema) -> None:
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([_batch(1, 3)]),
        columns=make_columns(COLUMNS),
        settings={"primary_column": "id"},
    )
    FullLoadStrategy().run(ctx)
    target_pg.assert_unique_pk_index(conn, ctx.target, "id")


def test_validation_rejects_a_pk_outside_the_columns(conn, schema) -> None:
    ctx = make_context(
        conn,
        schema,
        dialect=FakeDialect([_batch(1, 1)]),
        columns=make_columns(COLUMNS),
        settings={"primary_column": "does_not_exist"},
    )
    with pytest.raises(StrategyError, match="primary key"):
        FullLoadStrategy().validate(ctx)


def test_validation_rejects_autocommit_being_on(conn, schema) -> None:
    ctx = make_context(conn, schema, dialect=FakeDialect([]), columns=make_columns(COLUMNS))
    conn.commit()
    conn.autocommit = True
    try:
        with pytest.raises(StrategyError, match="autocommit"):
            FullLoadStrategy().validate(ctx)
    finally:
        conn.autocommit = False


def test_resolve_strategy() -> None:
    assert resolve_strategy("full").name == "full"
    assert resolve_strategy("incremental").name == "incremental"


# --- resume_full_load is a watermark, not a full load -------------------------
#
# Under `resume_full_load` the predecessor runs `SELECT MAX(pk)` on the target
# and fetches only the rows above it -- which is literally `IdWatermarkStrategy`.
# Without routing, that switch got lost on the way and the whole table was
# rebuilt: on one production table, the difference between 823 thousand and
# 9.9 million rows.


def test_resume_full_load_routes_to_the_watermark() -> None:
    assert resolve_strategy("full", {"resume_full_load": True}).name == "id_watermark"


def test_resume_full_load_also_accepts_a_string() -> None:
    """Configurations arrive from YAML, where `true` easily ends up as text.

    ``"ano"`` is in the list because deployed pipelines contain it; the parser
    still accepts that legacy spelling, so the test has to keep covering it.
    """
    for value in ("True", "true", "1", "ano"):
        assert resolve_strategy("full", {"resume_full_load": value}).name == "id_watermark"


def test_without_resume_it_stays_full() -> None:
    assert resolve_strategy("full", {}).name == "full"
    assert resolve_strategy("full", {"resume_full_load": False}).name == "full"
    assert resolve_strategy("full", None).name == "full"


def test_forced_full_load_beats_resume() -> None:
    """`forced_full_load` means "trust nothing the target remembers".

    A watermark is exactly what must not be trusted -- it takes `MAX(pk)` from
    the target -- so a forced reload would otherwise quietly turn into fetching
    a handful of rows.
    """
    settings = {"resume_full_load": True, "force_reload": True}

    assert resolve_strategy("full", settings).name == "full"


def test_resume_does_not_apply_to_other_methods() -> None:
    """The key is called `resume_full_load` and applies to the full load only."""
    assert resolve_strategy("hash", {"resume_full_load": True}).name == "hash_diff"
    assert resolve_strategy("incremental", {"resume_full_load": True}).name == "incremental"


def test_resolve_strategy_fails_on_a_typo() -> None:
    """The predecessor quietly falls back to a full load in such a case."""
    with pytest.raises(StrategyError, match="Unknown load_method"):
        resolve_strategy("incrementall")


def test_the_estimate_only_shrinks_the_batch() -> None:
    """The configured value is a **ceiling**, not a suggestion -- unlike in the
    predecessor.

    The predecessor also **grows** the batch from the estimate: on a table
    estimated at 418 MB it turns 100,000 rows into 764,635. Measurement shows
    that buys nothing -- a large batch costs 2.8x the memory and is 9% slower.
    """
    from dbextractors.core.strategies.base import resolve_batch_size
    from dbextractors.dialects.base import SizeEstimate

    light = SizeEstimate(rows=800_000, size_mb=418.0, method="test")
    assert resolve_batch_size({"batch_size": 100_000}, {}, light) == 100_000


def test_the_estimate_shrinks_the_batch_on_a_heavy_table() -> None:
    """Shrinking stays -- because of tables with 12 kB per row.

    There, 100,000 rows are 1.2 GB in a single batch, and that is exactly the
    situation ``TARGET_BATCH_MB`` was introduced for in the predecessor.
    """
    from dbextractors.core.strategies.base import TARGET_BATCH_MB, resolve_batch_size
    from dbextractors.dialects.base import SizeEstimate

    heavy = SizeEstimate(rows=100_000, size_mb=1200.0, method="test")
    batch = resolve_batch_size({"batch_size": 100_000}, {}, heavy)

    assert batch < 100_000
    assert batch == int(100_000 / (1200.0 / TARGET_BATCH_MB))


def test_the_batch_never_drops_below_the_minimum() -> None:
    from dbextractors.core.strategies.base import MIN_BATCH_SIZE, resolve_batch_size
    from dbextractors.dialects.base import SizeEstimate

    huge = SizeEstimate(rows=1000, size_mb=100_000.0, method="test")
    assert resolve_batch_size({"batch_size": 100_000}, {}, huge) == MIN_BATCH_SIZE


def test_a_legacy_target_without_row_hash_gets_the_column_and_the_values(conn, schema) -> None:
    """A target created by an older extractor has no `row_hash` -- dbextractors
    has to be able to catch it up.

    The shadow is made with ``LIKE <target>``, so it inherits whatever the target
    is missing, and ``COPY`` then fails on `column "row_hash" ... does not exist`.
    This is not a corner case: variant B's Firebird extractor has no
    `_deleted_in_source` at all, so the very first dbextractors run over such a
    table would always fail (`_deleted_in_source` is mandatory for every
    strategy).
    """
    source = FakeDialect([_batch(1, 5)])
    ctx = make_context(conn, schema, dialect=source, columns=make_columns(COLUMNS), settings={})

    # The target exactly as the old side left it: source columns and nothing more.
    # `name` is a reserved word, so in the target it is `_name`.
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{schema}"."cil" '
            '(id INTEGER, "_name" TEXT, price NUMERIC, "_timestamp" TEXT)'
        )
    conn.commit()

    result = FullLoadStrategy().run(ctx)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'cil'",
            (schema,),
        )
        columns = {r[0] for r in cur.fetchall()}
        cur.execute(f'SELECT count(*), count("row_hash") FROM "{schema}"."cil"')
        row_count, with_hash = cur.fetchone()

    assert "row_hash" in columns
    assert "_deleted_in_source" in columns
    assert result.rows_written == row_count
    # The values must not be lost by adding the column only after the swap.
    assert with_hash == row_count


def test_a_surrogate_key_passes_validation_in_full_too(conn, schema) -> None:
    """The same exception as in `hash_diff` -- a surrogate key is not in the source.

    The `hash` strategy can fall back to full (empty target, drifted hashes), so
    the check has to be lenient on both sides. Otherwise a pipeline passes
    `hash_diff` validation and fails one step later in `full`, which is exactly
    what happened to a junction table with no key of its own.
    """
    import dataclasses

    from dbextractors.dialects.base import SurrogateKey

    source = FakeDialect([_batch(1, 3)])
    ctx = make_context(
        conn,
        schema,
        dialect=source,
        columns=make_columns(COLUMNS),
        settings={"primary_column": "_uni_surrogate_key"},
    )
    ctx = dataclasses.replace(
        ctx,
        surrogate=SurrogateKey(enabled=True, alias="_uni_surrogate_key", expr="CONCAT(id, price)"),
    )

    FullLoadStrategy().validate(ctx)


def test_reading_is_ordered_by_the_pk(conn, schema) -> None:
    """Without ``ORDER BY pk`` recovery after a lost connection is not correct.

    `core.reading` resumes from the last key it saw. In an unordered stream that
    means nothing -- the server may send rows with lower keys too, and those
    would be discarded. The run would still finish green, so it is a silent loss
    of data. Ordering is therefore part of the contract, not an optimisation.
    """
    dialect = FakeDialect([_batch(1, 3)])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"primary_column": "id"},
    )

    FullLoadStrategy().run(ctx)

    reads = [s for s in dialect.seen_sql if s.startswith("SELECT")]
    assert reads, "the strategy read nothing"
    assert all(
        "ORDER BY `id`" in s for s in reads
    ), f"reads are not ordered by the PK, recovery would lose rows: {reads}"


def test_without_a_pk_nothing_is_ordered(conn, schema) -> None:
    """With no key there is no ordering -- recovery is impossible anyway, so the
    plan should not change."""
    dialect = FakeDialect([_batch(1, 3)])
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))

    FullLoadStrategy().run(ctx)

    reads = [s for s in dialect.seen_sql if s.startswith("SELECT")]
    assert all("ORDER BY" not in s for s in reads), reads
