"""Tests for `core.target_pg` against a **live** PostgreSQL.

What sits here is the part a mock cannot verify: that `COPY` gives back the same
value it was handed, that the swap keeps the views, and that the dedup and the
bookkeeping of deleted rows really change the rows they should.

Everything happens in a scratch schema prefixed ``dbx_golden_``, which is
dropped after the test. Without a database the tests skip rather than fail.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import TargetRef

pytestmark = pytest.mark.needs_pg


def _ref(schema: str, table: str = "target") -> TargetRef:
    return TargetRef(schema=schema, table=table)


def _fetch(conn, sql: str, params=None) -> list:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


# --- COPY: there and back ---------------------------------------------------

#: The values writing traditionally breaks on: CR, quotes, a tab, a backslash,
#: an empty string against NULL, and unicode — the whole ugly sample, run there
#: and back.
UGLY = [
    "ordinary text",
    'a quote " in the middle',
    'both "quotes" around',
    "comma, inside",
    "semicolon; inside",
    "tab\tinside",
    "new\nline",
    "CR\r\nbreak",
    "lone\rCR",
    "back \\ slash",
    "doubled \\\\ slash",
    "\\N as text",
    "naïve Fußgängerübergänge",
    "已经 中文",
    "emoji 🚚 in the middle",
    "   spaces at the edges   ",
    "",
    None,
]


@pytest.fixture()
def target(conn, schema: str) -> TargetRef:
    ref = _ref(schema)
    df = pd.DataFrame({"id": [1], "text": ["x"]})
    target_pg.create_table(conn, ref, df, overwrite_types={"id": "INTEGER", "text": "TEXT"})
    return ref


def test_copy_preserves_the_values(conn, target) -> None:
    df = pd.DataFrame({"id": range(len(UGLY)), "text": UGLY})
    written = target_pg.copy_from_stdin(conn, target, df, ["id", "text"])
    assert written == len(UGLY)

    rows = _fetch(conn, f"SELECT id, text FROM {target_pg.qualify(target)} ORDER BY id")
    received = [r[1] for r in rows]

    # An empty string ends up as NULL — the inherited `FORCE_NULL` behaviour.
    expected = [None if v == "" else v for v in UGLY]
    assert received == expected


def test_an_empty_string_is_null_in_the_target(conn, target) -> None:
    """Inherited from ``FORCE_NULL`` in mage. It loses information, but it is
    today's behaviour.

    Changing it would change the data in all ~670 tables — which is why it has a
    test and a constant of its own, `EMPTY_STRING_AS_NULL`, not just a comment.
    """
    assert target_pg.EMPTY_STRING_AS_NULL is True
    df = pd.DataFrame({"id": [1, 2], "text": ["", None]})
    target_pg.copy_from_stdin(conn, target, df, ["id", "text"])
    rows = _fetch(conn, f"SELECT text FROM {target_pg.qualify(target)} ORDER BY id")
    assert [r[0] for r in rows] == [None, None]


def test_copy_respects_the_column_order(conn, target) -> None:
    """The batch arrives in a different order; it has to go into the target in
    the target's."""
    df = pd.DataFrame({"text": ["a"], "id": [7]})
    target_pg.copy_from_stdin(conn, target, df, ["id", "text"])
    assert _fetch(conn, f"SELECT id, text FROM {target_pg.qualify(target)}") == [(7, "a")]


def test_copy_fills_a_missing_column_in_as_null(conn, target) -> None:
    df = pd.DataFrame({"id": [1]})
    target_pg.copy_from_stdin(conn, target, df, ["id", "text"])
    assert _fetch(conn, f"SELECT id, text FROM {target_pg.qualify(target)}") == [(1, None)]


def test_copy_in_pieces_writes_everything(conn, target) -> None:
    """Chunking keeps the memory peak down — but it must lose nothing and
    duplicate nothing."""
    df = pd.DataFrame({"id": range(5000), "text": [f"line {i}" for i in range(5000)]})
    written = target_pg.copy_from_stdin(conn, target, df, ["id", "text"], chunk_rows=137)
    assert written == 5000
    assert _fetch(
        conn, f"SELECT count(*), count(DISTINCT id) FROM {target_pg.qualify(target)}"
    ) == [(5000, 5000)]


def test_copy_of_an_empty_batch_does_nothing(conn, target) -> None:
    assert target_pg.copy_from_stdin(conn, target, pd.DataFrame({"id": []}), ["id", "text"]) == 0


# --- The target table's schema ----------------------------------------------


def test_load_pg_schema_for_a_table_that_does_not_exist(conn, schema: str) -> None:
    """A missing table is not an error — it is the state before the first run."""
    out = target_pg.load_pg_schema(conn, _ref(schema, "absent"), {}, {})
    assert out.exists is False
    assert out.columns == []


def test_load_pg_schema_overwrites_the_types(conn, schema: str) -> None:
    """The target may already have a column promoted to BIGINT; an estimate from
    the source must not knock it back down."""
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE {target_pg.qualify(ref)} "
            "(id BIGINT, price NUMERIC, active BOOLEAN, data JSONB, seen_at TIMESTAMP)"
        )
    overwrite = {"id": "INTEGER"}
    integers: dict = {}
    out = target_pg.load_pg_schema(conn, ref, overwrite, integers)

    assert out.exists is True
    assert out.columns == ["id", "price", "active", "data", "seen_at"]
    assert overwrite["id"] == "BIGINT"
    assert overwrite["price"] == "NUMERIC"
    assert overwrite["active"] == "BOOLEAN"
    assert overwrite["data"] == "JSONB"
    assert overwrite["seen_at"] == "TIMESTAMP"
    assert integers == {"id": "BIGINT"}
    assert out.integer_columns == {"id"}


def test_ensure_generated_columns(conn, schema: str) -> None:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER)")

    added = target_pg.ensure_generated_columns(conn, ref)
    assert added == ["row_hash", "_timestamp", "_deleted_in_source"]

    types = dict(
        _fetch(
            conn,
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (ref.schema, ref.table),
        )
    )
    assert types["_timestamp"] == "text"
    assert types["_deleted_in_source"] == "boolean"
    assert types["row_hash"] == "text"

    # The second time round there is nothing left to add.
    assert target_pg.ensure_generated_columns(conn, ref) == []


def test_deleted_in_source_is_not_null_with_a_default(conn, schema: str) -> None:
    """133 dbt models hang off this column; a NULL has no business being in
    it."""
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER)")
    target_pg.ensure_generated_columns(conn, ref)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {target_pg.qualify(ref)} (id) VALUES (1)")
    assert _fetch(conn, f"SELECT _deleted_in_source FROM {target_pg.qualify(ref)}") == [(False,)]


def test_ensure_generated_columns_without_the_hash(conn, schema: str) -> None:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER)")
    assert target_pg.ensure_generated_columns(conn, ref, include_row_hash=False) == [
        "_timestamp",
        "_deleted_in_source",
    ]


# --- Shadow table and swap --------------------------------------------------


def _target_with_a_view(conn, schema: str) -> TargetRef:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER, v TEXT)")
        cur.execute(f"INSERT INTO {target_pg.qualify(ref)} VALUES (1, 'old')")
        cur.execute(f'CREATE VIEW "{schema}"."the_view" AS SELECT * FROM {target_pg.qualify(ref)}')
    return ref


def test_the_swap_keeps_the_view(conn, schema: str) -> None:
    """**This is why the swap is not done with RENAME.**

    After ``ALTER TABLE … RENAME`` the view stays bound to the old table,
    because PostgreSQL holds on to the OID, not the name — verified in
    `test_rename_redirects_the_view`. `TRUNCATE` + `INSERT SELECT` leaves it
    bound to the target.
    """
    ref = _target_with_a_view(conn, schema)
    shadow = target_pg.create_shadow_table(conn, ref)
    df = pd.DataFrame({"id": [1, 2], "v": ["new", "new2"]})
    target_pg.copy_from_stdin(conn, shadow, df, ["id", "v"])

    assert target_pg.swap_shadow_table(conn, shadow, ref) == 2
    target_pg.drop_shadow_table(conn, shadow)

    assert _fetch(conn, f'SELECT v FROM "{schema}"."the_view" ORDER BY id') == [
        ("new",),
        ("new2",),
    ]


def test_rename_redirects_the_view(conn, schema: str) -> None:
    """A control test: it documents why `swap_shadow_table` does not use RENAME.

    Should PostgreSQL ever start behaving differently, this test fails and there
    is a reason to go back to RENAME — it is an order of magnitude faster.
    """
    ref = _target_with_a_view(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{schema}"."other" (id INTEGER, v TEXT)')
        cur.execute(f'INSERT INTO "{schema}"."other" VALUES (1, \'new\')')
        cur.execute(f"ALTER TABLE {target_pg.qualify(ref)} RENAME TO target_old")
        cur.execute(f'ALTER TABLE "{schema}"."other" RENAME TO target')

    assert _fetch(conn, f'SELECT v FROM "{schema}"."the_view"') == [("old",)]


def test_truncate_is_rolled_back_in_postgresql(conn, schema: str) -> None:
    """A control test for the premise the swap rests on, not a test of the swap.

    `swap_shadow_table` is only allowed to `TRUNCATE` the target because a
    rolled-back `TRUNCATE` puts the rows back — unlike in databases where DDL
    commits implicitly. Should that ever stop holding, the swap has to be
    rebuilt, and this test is where it shows first. What the swap itself does
    with a failure is
    `test_a_failing_swap_leaves_the_target_and_the_view_intact`.
    """
    ref = _target_with_a_view(conn, schema)
    shadow = target_pg.create_shadow_table(conn, ref)
    target_pg.copy_from_stdin(conn, shadow, pd.DataFrame({"id": [9], "v": ["x"]}), ["id", "v"])
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {target_pg.qualify(ref)}")
    conn.rollback()

    assert _fetch(conn, f"SELECT v FROM {target_pg.qualify(ref)}") == [("old",)]


def test_a_failing_swap_leaves_the_target_and_the_view_intact(conn, schema: str) -> None:
    """`TRUNCATE` and `INSERT … SELECT` are **one** transaction — through the real swap.

    The failure is a real one from inside the database: the target carries a
    `CHECK` the shadow's rows violate, and ``LIKE … INCLUDING DEFAULTS`` does not
    copy constraints, so the shadow accepts them and only the
    ``INSERT … SELECT`` falls over — with the `TRUNCATE` already issued.

    That is the only arrangement in which the swap's transaction discipline is
    visible. Were a `commit()` to appear anywhere between the two statements, the
    emptying would survive the rollback and the target would be left blank with
    the view serving nothing — the state a full load must never produce
    (documented at 11.4 of 18.8 M rows).
    """
    import psycopg2

    ref = _target_with_a_view(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {target_pg.qualify(ref)} ADD CONSTRAINT small_id CHECK (id < 5)")
    conn.commit()

    shadow = target_pg.create_shadow_table(conn, ref)
    target_pg.copy_from_stdin(conn, shadow, pd.DataFrame({"id": [9], "v": ["new"]}), ["id", "v"])

    with pytest.raises(psycopg2.errors.CheckViolation):
        target_pg.swap_shadow_table(conn, shadow, ref)
    conn.rollback()

    assert _fetch(conn, f"SELECT v FROM {target_pg.qualify(ref)}") == [("old",)]
    assert _fetch(conn, f'SELECT v FROM "{schema}"."the_view"') == [("old",)]
    # The shadow was created after the last commit, so the rollback took it with
    # it — nothing to clean up, and nothing left for the next run to trip over.
    assert target_pg.find_stale_shadows(conn, schema) == []


class _CursorThatDies:
    """A cursor that hands everything to the real one until the statement it is waiting for.

    A `psycopg2` cursor is used as a context manager and the wrapper is what the
    swap gets, so both halves of the protocol have to be forwarded.
    """

    def __init__(self, cursor, fails_on: str) -> None:
        self._cursor = cursor
        self._fails_on = fails_on

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._cursor.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def execute(self, sql, params=None):
        if self._fails_on in sql:
            raise ConnectionError(f"the connection dropped before: {sql}")
        return self._cursor.execute(sql, params or ())


class _ConnectionThatDies:
    """The target connection, but a chosen statement never reaches the server.

    Simulates the half of the failures that happen **outside** the database —
    a dropped connection, a killed process — which cannot be provoked with a
    constraint. `commit` is counted rather than blocked: the swap is not supposed
    to commit at all, and counting says so directly.
    """

    def __init__(self, conn, fails_on: str) -> None:
        self._conn = conn
        self._fails_on = fails_on
        self.commits = 0

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def cursor(self):
        return _CursorThatDies(self._conn.cursor(), self._fails_on)

    def commit(self):
        self.commits += 1
        return self._conn.commit()


def test_a_swap_cut_off_after_the_truncate_leaves_the_target_intact(conn, schema: str) -> None:
    """The connection dies between the `TRUNCATE` and the `INSERT`.

    The counterpart of `test_a_failing_swap_leaves_the_target_and_the_view_intact`
    for a failure the database never sees. It also states the rule positively:
    the swap commits **nothing** — the decision when the new data becomes visible
    belongs to the caller, which is what lets the strategies put the swap and
    their own bookkeeping into a single transaction.
    """
    ref = _target_with_a_view(conn, schema)
    shadow = target_pg.create_shadow_table(conn, ref)
    target_pg.copy_from_stdin(conn, shadow, pd.DataFrame({"id": [9], "v": ["new"]}), ["id", "v"])
    conn.commit()

    wrapped = _ConnectionThatDies(conn, fails_on="INSERT INTO")
    with pytest.raises(ConnectionError):
        target_pg.swap_shadow_table(wrapped, shadow, ref)
    conn.rollback()

    assert wrapped.commits == 0, "the swap must not commit; that is the caller's decision"
    assert _fetch(conn, f"SELECT v FROM {target_pg.qualify(ref)}") == [("old",)]
    assert _fetch(conn, f'SELECT v FROM "{schema}"."the_view"') == [("old",)]


def test_a_swap_without_an_existing_target_renames(conn, schema: str) -> None:
    """When there is no target, there can be no view over it — RENAME is fine."""
    ref = _ref(schema)
    df = pd.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    shadow = target_pg.create_shadow_table(conn, ref, df=df)
    target_pg.copy_from_stdin(conn, shadow, df, ["id", "v"])
    assert target_pg.swap_shadow_table(conn, shadow, ref) == 2
    assert _fetch(conn, f"SELECT count(*) FROM {target_pg.qualify(ref)}") == [(2,)]


def test_the_swap_requires_autocommit_to_be_off(conn, schema: str) -> None:
    ref = _target_with_a_view(conn, schema)
    shadow = target_pg.create_shadow_table(conn, ref)
    conn.commit()
    conn.autocommit = True
    try:
        with pytest.raises(target_pg.TargetError, match="autocommit"):
            target_pg.swap_shadow_table(conn, shadow, ref)
    finally:
        conn.autocommit = False


def test_the_swap_refuses_a_shadow_with_a_missing_column(conn, schema: str) -> None:
    """The swap would blank the column out — better to fail."""
    ref = _target_with_a_view(conn, schema)
    shadow = target_pg.shadow_ref(ref)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(shadow)} (id INTEGER)")
    with pytest.raises(target_pg.TargetError, match="is missing columns"):
        target_pg.swap_shadow_table(conn, shadow, ref)


def test_find_stale_shadows(conn, schema: str) -> None:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER)")
    shadow = target_pg.create_shadow_table(conn, ref)
    assert target_pg.find_stale_shadows(conn, schema) == [shadow.table]
    target_pg.drop_shadow_table(conn, shadow)
    assert target_pg.find_stale_shadows(conn, schema) == []


# --- Dedup ------------------------------------------------------------------


def test_seen_keys_drops_duplicates_across_batches(conn) -> None:
    with target_pg.SeenKeys(conn, "id") as seen:
        first = seen.filter_new(pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}))
        assert first["id"].tolist() == [1, 2, 3]

        second = seen.filter_new(pd.DataFrame({"id": [3, 4], "v": ["c2", "d"]}))
        assert second["id"].tolist() == [4]


def test_seen_keys_works_for_integers_too(conn) -> None:
    """In the predecessors the dedup is dead code for an integer PK.

    ``seen_keys`` is filled with strings (``.astype(str)``) while the raw column
    is filtered — so ``isin`` never matches. Here both sides are compared as
    text, so the dedup really works.
    """
    with target_pg.SeenKeys(conn, "id") as seen:
        seen.filter_new(pd.DataFrame({"id": [10, 20]}))
        assert seen.filter_new(pd.DataFrame({"id": [10, 20]})).empty


def test_seen_keys_ignores_a_batch_without_the_pk(conn) -> None:
    with target_pg.SeenKeys(conn, "id") as seen:
        df = pd.DataFrame({"other": [1, 2]})
        assert seen.filter_new(df) is df


def test_seen_keys_cleans_up_after_itself(conn) -> None:
    with target_pg.SeenKeys(conn, "id") as seen:
        name = seen._table
        seen.filter_new(pd.DataFrame({"id": [1]}))
    assert _fetch(conn, "SELECT to_regclass(%s)", (name,)) == [(None,)]


# --- Bookkeeping of deleted rows --------------------------------------------


@pytest.fixture()
def target_with_flag(conn, schema: str) -> TargetRef:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE {target_pg.qualify(ref)} "
            '(id INTEGER, "_deleted_in_source" BOOLEAN NOT NULL DEFAULT false, '
            '"_timestamp" TEXT)'
        )
        cur.execute(f"INSERT INTO {target_pg.qualify(ref)} (id) VALUES (1), (2), (3)")
    return ref


def test_the_snapshot_marks_the_rows_that_disappeared(conn, target_with_flag) -> None:
    changed = target_pg.apply_live_pk_snapshot(conn, target_with_flag, "id", [1, 2], "integer")
    assert changed == 1
    rows = _fetch(
        conn,
        f'SELECT id, "_deleted_in_source" FROM {target_pg.qualify(target_with_flag)} ORDER BY id',
    )
    assert rows == [(1, False), (2, False), (3, True)]


def test_the_snapshot_restores_rows_that_came_back(conn, target_with_flag) -> None:
    target_pg.apply_live_pk_snapshot(conn, target_with_flag, "id", [1], "integer")
    changed = target_pg.apply_live_pk_snapshot(conn, target_with_flag, "id", [1, 2, 3], "integer")
    assert changed == 2
    rows = _fetch(
        conn, f'SELECT "_deleted_in_source" FROM {target_pg.qualify(target_with_flag)} ORDER BY id'
    )
    assert [r[0] for r in rows] == [False, False, False]


def test_the_snapshot_accepts_a_series_as_well_as_a_dataframe(conn, target_with_flag) -> None:
    target_pg.apply_live_pk_snapshot(conn, target_with_flag, "id", pd.Series([1, 2, 3]), "integer")
    rows = _fetch(conn, f'SELECT "_deleted_in_source" FROM {target_pg.qualify(target_with_flag)}')
    assert all(not r[0] for r in rows)


def test_the_snapshot_fails_without_a_pk(conn, target_with_flag) -> None:
    with pytest.raises(target_pg.TargetError, match="without a PK"):
        target_pg.apply_live_pk_snapshot(conn, target_with_flag, "", [1])


# --- Indexes ----------------------------------------------------------------


def test_assert_unique_pk_index_fails_without_the_table(conn, schema: str) -> None:
    with pytest.raises(target_pg.TargetError, match="does not exist"):
        target_pg.assert_unique_pk_index(conn, _ref(schema, "absent"), "id")


def test_assert_unique_pk_index_fails_without_the_index(conn, schema: str) -> None:
    """It has to fail **before** any expensive work over the source."""
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER)")
    with pytest.raises(target_pg.TargetError, match="is missing a UNIQUE"):
        target_pg.assert_unique_pk_index(conn, ref, "id")


def test_assert_unique_pk_index_accepts_a_primary_key(conn, schema: str) -> None:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER PRIMARY KEY)")
    target_pg.assert_unique_pk_index(conn, ref, "id")


def test_create_unique_pk_index(conn, schema: str) -> None:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER)")
    conn.commit()

    assert target_pg.create_unique_pk_index(conn, ref, "id", ["id"]) is True
    target_pg.assert_unique_pk_index(conn, ref, "id")
    # The second time round it passes too — IF NOT EXISTS.
    assert target_pg.create_unique_pk_index(conn, ref, "id", ["id"]) is True


def test_no_index_is_created_when_the_pk_is_not_among_the_columns(conn, schema: str) -> None:
    ref = _ref(schema)
    assert target_pg.create_unique_pk_index(conn, ref, "id", ["other"]) is False


def _partitioned_target(conn, schema: str) -> TargetRef:
    """A target partitioned by ``_source`` with a single partition."""
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE {target_pg.qualify(ref)} "
            f'(id INTEGER, "{target_pg.SOURCE_COLUMN}" TEXT NOT NULL) '
            f'PARTITION BY LIST ("{target_pg.SOURCE_COLUMN}")'
        )
    target_pg.ensure_partition(conn, ref, "company_a")
    conn.commit()
    return ref


def test_create_unique_pk_index_over_a_partitioned_target(conn, schema: str) -> None:
    """PostgreSQL refuses ``CONCURRENTLY`` over a partitioned table.

    Documented impact: two partitioned target tables were left without a unique
    index, because the call from `full_by_source` is `fatal=False` — the failure
    was only logged, the run finished green, and the upsert simply does not fit
    those tables.
    """
    ref = _partitioned_target(conn, schema)

    result = target_pg.create_unique_pk_index(
        conn, ref, "id", ["id", target_pg.SOURCE_COLUMN], columns=["id", target_pg.SOURCE_COLUMN]
    )

    assert result is True
    target_pg.assert_unique_pk_index(conn, ref, "id")


def test_create_unique_pk_index_over_a_partitioned_target_is_idempotent(conn, schema: str) -> None:
    ref = _partitioned_target(conn, schema)
    columns = ["id", target_pg.SOURCE_COLUMN]

    assert target_pg.create_unique_pk_index(conn, ref, "id", columns, columns=columns) is True
    assert target_pg.create_unique_pk_index(conn, ref, "id", columns, columns=columns) is True


def test_a_failed_index_over_a_partitioned_target_does_not_abort_the_transaction(
    conn, schema: str
) -> None:
    """A unique index over a partitioned table must contain the partition key.

    Without ``_source`` PostgreSQL refuses it — and because this branch runs
    inside a transaction, `create_unique_pk_index` has to roll it back after the
    failure. Otherwise, with ``fatal=False``, the next unrelated statement would
    be the one that failed.
    """
    ref = _partitioned_target(conn, schema)

    assert target_pg.create_unique_pk_index(conn, ref, "id", ["id"], fatal=False) is False

    # The connection is still usable — without the rollback this would fail with
    # `InFailedSqlTransaction`.
    assert target_pg.table_exists(conn, ref) is True


# --- Snapshot of the source hashes ------------------------------------------


@pytest.fixture()
def target_with_hash(conn, schema: str) -> TargetRef:
    ref = _ref(schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {target_pg.qualify(ref)} (id INTEGER, row_hash TEXT)")
        cur.execute(
            f"INSERT INTO {target_pg.qualify(ref)} VALUES "
            "(1, 'aaa'), (2, 'bbb'), (3, 'ccc'), (4, NULL)"
        )
    return ref


def _snapshot(conn, pairs) -> pd.DataFrame:
    return pd.DataFrame({"id": [k for k, _ in pairs], "row_hash": [h for _, h in pairs]})


def test_the_diff_tells_changed_new_and_identical_apart(conn, target_with_hash) -> None:
    with target_pg.SourceHashSnapshot(conn, "id", "row_hash") as snap:
        snap.add(_snapshot(conn, [(1, "aaa"), (2, "CHANGED"), (5, "new")]))
        snap.index()
        result = snap.diff(target_with_hash)

    assert result == {"added": 1, "changed": 1, "matched": 1, "source_rows": 3}


def test_the_diff_treats_a_null_hash_in_the_target_as_a_change(conn, target_with_hash) -> None:
    """``hash <> hash`` would return NULL for a NULL and the row would drop out.

    Those are exactly the rows the seeding missed — they have to be
    transferred.
    """
    with target_pg.SourceHashSnapshot(conn, "id", "row_hash") as snap:
        snap.add(_snapshot(conn, [(4, "something")]))
        snap.index()
        assert snap.diff(target_with_hash)["changed"] == 1


def test_changed_keys_returns_both_changed_and_new(conn, target_with_hash) -> None:
    with target_pg.SourceHashSnapshot(conn, "id", "row_hash") as snap:
        snap.add(_snapshot(conn, [(1, "aaa"), (2, "CHANGED"), (5, "new")]))
        snap.index()
        keys = [k for batch in snap.changed_keys(target_with_hash, 10) for k in batch]

    assert sorted(keys) == ["2", "5"]


def test_changed_keys_comes_in_batches(conn, target_with_hash) -> None:
    """A million-item list of changed keys has no business being in memory at
    once."""
    with target_pg.SourceHashSnapshot(conn, "id", "row_hash") as snap:
        snap.add(_snapshot(conn, [(i, f"h{i}") for i in range(100, 130)]))
        snap.index()
        batches = list(snap.changed_keys(target_with_hash, 7))

    assert [len(b) for b in batches] == [7, 7, 7, 7, 2]


def test_a_second_run_with_no_changes_finds_nothing(conn, target_with_hash) -> None:
    """The headline scenario: with nothing changed in the source, nothing may be
    transferred."""
    with target_pg.SourceHashSnapshot(conn, "id", "row_hash") as snap:
        snap.add(_snapshot(conn, [(1, "aaa"), (2, "bbb"), (3, "ccc")]))
        snap.index()
        assert snap.diff(target_with_hash)["changed"] == 0
        assert snap.diff(target_with_hash)["added"] == 0
        assert list(snap.changed_keys(target_with_hash, 10)) == []


def test_the_snapshot_cleans_up_after_itself(conn, target_with_hash) -> None:
    with target_pg.SourceHashSnapshot(conn, "id", "row_hash") as snap:
        name = snap.table
        snap.add(_snapshot(conn, [(1, "aaa")]))
    assert _fetch(conn, "SELECT to_regclass(%s)", (name,)) == [(None,)]
