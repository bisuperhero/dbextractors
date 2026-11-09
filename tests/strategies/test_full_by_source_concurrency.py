"""`full_by_source` with one source per run: labels, concurrency, the unique index.

The shape this covers: one target table, every source a separate `run()` with its
own ``source_label``, and the caller running several of them at once (three pools
side by side). Every run here gets **its own connection** and closes it without a
commit at the end, exactly as `entrypoint._run_one` does — a check made on the
connection that did the work would see its uncommitted state and miss the bugs
that matter (the unique index used to exist only until ``close()``).

The concurrency tests start the runs behind a barrier, so they really collide in
the DDL that used to fail: ``CREATE SCHEMA``, the partitioned parent, a new
column, the fingerprint ledger. Before the DDL lock, the first two failed with
``UniqueViolation`` on the catalogue (verified on PG 17, 2026-11-09).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List

import pandas as pd
import psycopg2
import pytest

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import StrategyError, TargetRef
from dbextractors.core.strategies.full_by_source import FullBySourceStrategy
from dbextractors.dialects.base import TableRef
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [("id", "int", "INTEGER"), ("label", "varchar", "TEXT")]
PARTITIONED = {"partition_by_source": True}
FINGERPRINT = {"mode": "count", "primary_column": "id"}

#: How long a worker thread may take before the test calls it hung.
JOIN_SECONDS = 60


# --- Scaffolding ------------------------------------------------------------


@pytest.fixture()
def fresh_schema(dsn) -> Iterator[str]:
    """A schema name that does **not** exist yet; dropped afterwards whatever happened."""
    name = f"dbx_golden_fbs_{uuid.uuid4().hex[:8]}"
    yield name
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
    conn.close()


def _connect(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def _data(ids, prefix: str, **extra: List[Any]) -> pd.DataFrame:
    frame = pd.DataFrame({"id": list(ids), "label": [f"{prefix}{i}" for i in ids]})
    for name, values in extra.items():
        frame[name] = values
    return frame


def _run(
    dsn: str,
    schema: str,
    label: str | None,
    data: pd.DataFrame,
    *,
    columns=COLUMNS,
    settings: dict | None = None,
    fingerprint: dict | None = None,
    source: FakeHashSource | None = None,
    database: str = "dory",
    table: str = "cil",
    configured: bool = True,
):
    """One run as `entrypoint._run_one` makes it: own connection, closed at the end.

    ``configured`` says whether the label came from ``LOAD_SETTINGS.source_label``
    (it then reaches ``ctx.settings`` too, as through `entrypoint._settings_dict`)
    or from a database name in the ``DATABASES`` loop.
    """
    source = source if source is not None else FakeHashSource(data, pk="id")
    conn = _connect(dsn)
    try:
        ctx = make_context(
            conn,
            schema,
            dialect=source,
            columns=make_columns(columns),
            settings={
                "primary_column": "id",
                **PARTITIONED,
                "source_label": label if configured else None,
                **(settings or {}),
            },
            table=table,
        )
        # Every pool has a database of the same name; only the label tells them apart.
        ctx.source = TableRef(name="zdroj", database=database)
        ctx.source_label = label
        ctx.is_first_source = True
        ctx.is_last_source = True
        ctx.fingerprint_cfg = dict(fingerprint or {})
        return FullBySourceStrategy().run(ctx)
    finally:
        conn.close()


def _concurrently(jobs: Dict[str, Callable[[], Any]]) -> Dict[str, Any]:
    """Run the jobs at once behind a barrier. Returns ``{name: result or exception}``."""
    barrier = threading.Barrier(len(jobs))
    out: Dict[str, Any] = {}

    def worker(name: str, job: Callable[[], Any]) -> None:
        try:
            barrier.wait()
            out[name] = job()
        except Exception as err:
            out[name] = err

    threads = [threading.Thread(target=worker, args=item, daemon=True) for item in jobs.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_SECONDS)
    assert not any(t.is_alive() for t in threads), "a run hung"
    return out


def _errors(results: Dict[str, Any]) -> Dict[str, str]:
    return {k: f"{type(v).__name__}: {v}" for k, v in results.items() if isinstance(v, Exception)}


def _query(dsn: str, sql: str, params=()) -> list:
    """From a fresh connection — what other sessions see, i.e. what was committed."""
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def _slices(dsn: str, schema: str, table: str = "cil") -> list:
    return _query(
        dsn, f'SELECT "_source", count(*) FROM "{schema}"."{table}" GROUP BY 1 ORDER BY 1'
    )


def _partitions(dsn: str, schema: str, table: str = "cil") -> list:
    return [
        r[0]
        for r in _query(
            dsn,
            "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
            "WHERE i.inhparent = to_regclass(%s) ORDER BY 1",
            (f'"{schema}"."{table}"',),
        )
    ]


def _indexes(dsn: str, schema: str, table: str = "cil") -> list:
    return [
        r[0]
        for r in _query(
            dsn,
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = %s ORDER BY 1",
            (schema, table),
        )
    ]


def _ledger(dsn: str, schema: str) -> list:
    return _query(
        dsn,
        f'SELECT table_name, source_label FROM "{schema}"."{target_pg.FINGERPRINT_TABLE}" '
        "ORDER BY 1, 2",
    )


def _columns(dsn: str, schema: str, table: str = "cil") -> list:
    return [
        r[0]
        for r in _query(
            dsn,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (schema, table),
        )
    ]


def _stale_shadows(dsn: str, schema: str) -> list:
    conn = _connect(dsn)
    try:
        return target_pg.find_stale_shadows(conn, schema)
    finally:
        conn.close()


# --- Labels: one database name, several sources ------------------------------


def test_two_sources_with_the_same_database_name_get_their_own_slices(dsn, fresh_schema) -> None:
    """Every pool's database is called ``dory``; the label is what keeps them apart."""
    _run(dsn, fresh_schema, "finone", _data((1, 2), "a"), fingerprint=FINGERPRINT)
    _run(dsn, fresh_schema, "fintwo", _data((1, 2, 3), "b"), fingerprint=FINGERPRINT)

    assert _slices(dsn, fresh_schema) == [("finone", 2), ("fintwo", 3)]
    assert _partitions(dsn, fresh_schema) == ["cil__finone", "cil__fintwo"]
    assert _ledger(dsn, fresh_schema) == [("cil", "finone"), ("cil", "fintwo")]


def test_rewriting_one_labelled_slice_leaves_the_other_alone(dsn, fresh_schema) -> None:
    _run(dsn, fresh_schema, "finone", _data((1, 2), "a"))
    _run(dsn, fresh_schema, "fintwo", _data((1, 2), "b"))

    _run(dsn, fresh_schema, "fintwo", _data((7,), "b"))

    rows = _query(dsn, f'SELECT "_source", id, label FROM "{fresh_schema}".cil ORDER BY 1, 2')
    assert rows == [("finone", 1, "a1"), ("finone", 2, "a2"), ("fintwo", 7, "b7")]


def test_an_unchanged_labelled_source_is_skipped(dsn, fresh_schema) -> None:
    source = FakeHashSource(_data((1, 2), "a"), pk="id")
    _run(dsn, fresh_schema, "finone", None, source=source, fingerprint=FINGERPRINT)

    result = _run(dsn, fresh_schema, "finone", None, source=source, fingerprint=FINGERPRINT)

    assert result.load_method == "unchanged"


def test_the_fingerprint_is_read_to_the_end(dsn, fresh_schema) -> None:
    """A suspended generator counts as abandoned on MySQL (`KILL QUERY` + a dropped connection)."""

    class RecordsTheEnd(FakeHashSource):
        drained = 0

        def iter_batches(self, engine, sql, batch_size):
            yield from super().iter_batches(engine, sql, batch_size)
            if "fp_count" in sql:
                type(self).drained += 1

    source = RecordsTheEnd(_data((1, 2), "a"), pk="id")
    _run(dsn, fresh_schema, "finone", None, source=source, fingerprint=FINGERPRINT)
    _run(dsn, fresh_schema, "finone", None, source=source, fingerprint=FINGERPRINT)

    assert RecordsTheEnd.drained == 2


def test_reading_the_fingerprint_creates_nothing(dsn, fresh_schema) -> None:
    """Reading used to create the schema and the ledger — DDL in the read phase."""
    conn = _connect(dsn)
    try:
        target = TargetRef(schema=fresh_schema, table="cil")
        assert target_pg.read_fingerprint(conn, target, "finone") is None
        conn.commit()
    finally:
        conn.close()
    assert _query(dsn, "SELECT 1 FROM pg_namespace WHERE nspname = %s", (fresh_schema,)) == []


# --- Concurrency --------------------------------------------------------------


def test_three_runs_into_a_schema_that_does_not_exist(dsn, fresh_schema) -> None:
    """Failed with `UniqueViolation … pg_namespace_nspname_index` before the lock."""
    labels = ("finone", "fintwo", "finthree")
    results = _concurrently(
        {lbl: (lambda lbl=lbl: _run(dsn, fresh_schema, lbl, _data((1, 2), lbl))) for lbl in labels}
    )

    assert _errors(results) == {}
    assert _partitions(dsn, fresh_schema) == sorted(f"cil__{lbl}" for lbl in labels)
    assert _slices(dsn, fresh_schema) == sorted((lbl, 2) for lbl in labels)
    assert _stale_shadows(dsn, fresh_schema) == []


def test_three_runs_into_an_existing_schema_without_the_target(dsn, schema) -> None:
    """Failed with `UniqueViolation … pg_type_typname_nsp_index` (the parent) before the lock."""
    labels = ("finone", "fintwo", "finthree")
    results = _concurrently(
        {lbl: (lambda lbl=lbl: _run(dsn, schema, lbl, _data((1, 2), lbl))) for lbl in labels}
    )

    assert _errors(results) == {}
    assert target_pg.is_partitioned(_connect(dsn), TargetRef(schema=schema, table="cil"))
    assert _partitions(dsn, schema) == sorted(f"cil__{lbl}" for lbl in labels)
    assert _slices(dsn, schema) == sorted((lbl, 2) for lbl in labels)


def test_different_tables_into_a_new_schema_at_once(dsn, fresh_schema) -> None:
    """Different targets have different DDL locks — the schema and the ledger are shared."""
    tables = ("smlouvy", "splatky", "klienti", "udalosti")
    results = _concurrently(
        {
            t: (
                lambda t=t: _run(
                    dsn, fresh_schema, "finone", _data((1,), "a"), table=t, fingerprint=FINGERPRINT
                )
            )
            for t in tables
        }
    )

    assert _errors(results) == {}
    assert _ledger(dsn, fresh_schema) == sorted((t, "finone") for t in tables)


def test_two_runs_that_both_bring_a_new_column(dsn, schema) -> None:
    """The column is added once; neither run fails on "already exists"."""
    _run(dsn, schema, "seed", _data((9,), "s"))
    wider = [*COLUMNS, ("note", "varchar", "TEXT")]

    results = _concurrently(
        {
            lbl: (
                lambda lbl=lbl: _run(
                    dsn, schema, lbl, _data((1, 2), lbl, note=["x", "y"]), columns=wider
                )
            )
            for lbl in ("finone", "fintwo")
        }
    )

    assert _errors(results) == {}
    assert _columns(dsn, schema).count("note") == 1
    rows = _query(dsn, f'SELECT "_source", count(note) FROM "{schema}".cil GROUP BY 1 ORDER BY 1')
    assert rows == [("finone", 2), ("fintwo", 2), ("seed", 0)]


def test_two_runs_with_a_fingerprint_and_no_ledger_yet(dsn, schema) -> None:
    _run(dsn, schema, "seed", _data((9,), "s"))

    results = _concurrently(
        {
            lbl: (
                lambda lbl=lbl: _run(dsn, schema, lbl, _data((1, 2), lbl), fingerprint=FINGERPRINT)
            )
            for lbl in ("finone", "fintwo")
        }
    )

    assert _errors(results) == {}
    assert _ledger(dsn, schema) == [("cil", "finone"), ("cil", "fintwo")]


def test_a_new_column_does_not_wait_for_another_runs_stream(dsn, schema, monkeypatch) -> None:
    """Run A pauses mid-stream; run B adds a column and finishes meanwhile.

    The staging table used to be created ``LIKE`` the target inside the data
    transaction, which held an ``AccessShareLock`` on the target for the whole
    download — and B's ``ALTER TABLE … ADD COLUMN`` would have waited for it. The
    lock timeout is short so that a regression fails here instead of hanging.
    """
    monkeypatch.setenv(target_pg.LOCK_TIMEOUT_ENV, "5s")
    _run(dsn, schema, "seed", _data((9,), "s"))
    paused, release = threading.Event(), threading.Event()

    class PausesMidStream(FakeHashSource):
        def iter_batches(self, engine, sql, batch_size):
            for index, batch in enumerate(super().iter_batches(engine, sql, 1)):
                yield batch
                if index == 0 and "fp_count" not in sql:
                    paused.set()
                    assert release.wait(JOIN_SECONDS), "never released"

    result_a: Dict[str, Any] = {}

    def run_a() -> None:
        try:
            source = PausesMidStream(_data((1, 2, 3), "a"), pk="id")
            result_a["ok"] = _run(dsn, schema, "finone", None, source=source)
        except Exception as err:
            result_a["error"] = err

    thread = threading.Thread(target=run_a, daemon=True)
    thread.start()
    assert paused.wait(JOIN_SECONDS), "run A never started streaming"

    started = time.monotonic()
    wider = [*COLUMNS, ("note", "varchar", "TEXT")]
    _run(dsn, schema, "fintwo", _data((1,), "b", note=["x"]), columns=wider)
    elapsed = time.monotonic() - started

    assert not release.is_set() and thread.is_alive(), "A has to be still mid-stream here"
    release.set()
    thread.join(JOIN_SECONDS)

    assert "error" not in result_a, result_a.get("error")
    assert elapsed < 5, f"B waited {elapsed:.1f}s — for A's stream?"
    assert "note" in _columns(dsn, schema)
    assert _slices(dsn, schema) == [("finone", 3), ("fintwo", 1), ("seed", 1)]


def test_a_held_lock_times_out_and_leaves_the_target_alone(dsn, schema, monkeypatch) -> None:
    """The wait for the DDL lock is bounded, and running into the bound changes nothing."""
    _run(dsn, schema, "finone", _data((1, 2), "a"))
    monkeypatch.setenv(target_pg.LOCK_TIMEOUT_ENV, "300ms")

    holder = _connect(dsn)
    try:
        with holder.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"dbx:ddl:{schema}.cil",),
            )
        with pytest.raises(target_pg.TargetError, match="timed out after 300ms"):
            _run(dsn, schema, "fintwo", _data((5,), "b"))
    finally:
        holder.rollback()
        holder.close()

    assert _slices(dsn, schema) == [("finone", 2)]
    assert _stale_shadows(dsn, schema) == []


def test_an_invalid_lock_timeout_is_refused_by_name(dsn, schema, monkeypatch) -> None:
    monkeypatch.setenv(target_pg.LOCK_TIMEOUT_ENV, "ten minutes")
    with pytest.raises(target_pg.TargetError, match=target_pg.LOCK_TIMEOUT_ENV):
        _run(dsn, schema, "finone", _data((1,), "a"))


# --- Sources whose columns differ ---------------------------------------------


def test_a_source_with_an_extra_column_widens_the_target(dsn, schema) -> None:
    """In the other slices the new column is NULL."""
    _run(dsn, schema, "finone", _data((1,), "a"))
    _run(
        dsn,
        schema,
        "fintwo",
        _data((1,), "b", note=["x"]),
        columns=[*COLUMNS, ("note", "varchar", "TEXT")],
    )

    rows = _query(dsn, f'SELECT "_source", note FROM "{schema}".cil ORDER BY 1')
    assert rows == [("finone", None), ("fintwo", "x")]


def test_a_source_without_a_column_has_null_in_its_slice(dsn, schema) -> None:
    wider = [*COLUMNS, ("note", "varchar", "TEXT")]
    _run(dsn, schema, "finone", _data((1,), "a", note=["x"]), columns=wider)
    _run(dsn, schema, "fintwo", _data((1,), "b"))

    rows = _query(dsn, f'SELECT "_source", note FROM "{schema}".cil ORDER BY 1')
    assert rows == [("finone", "x"), ("fintwo", None)]
    # And rewriting the narrower source again leaves the other slice's value alone.
    _run(dsn, schema, "fintwo", _data((2,), "b"))
    assert _query(dsn, f'SELECT note FROM "{schema}".cil WHERE "_source" = %s', ("finone",)) == [
        ("x",)
    ]


def test_an_incompatible_type_fails_only_that_source_and_says_why(dsn, schema) -> None:
    """The message names the column, both types and the source; nothing changes."""
    _run(
        dsn,
        schema,
        "finone",
        _data((1, 2), "a", amount=[10, 20]),
        columns=[*COLUMNS, ("amount", "int", "INTEGER")],
    )
    before = _query(dsn, f'SELECT "_source", id, amount FROM "{schema}".cil ORDER BY 1, 2')

    with pytest.raises(StrategyError) as err:
        _run(
            dsn,
            schema,
            "fintwo",
            _data((1,), "b", amount=["not a number"]),
            columns=[*COLUMNS, ("amount", "varchar", "TEXT")],
        )

    text = str(err.value)
    assert "fintwo" in text and "'amount'" in text
    assert "varchar" in text and "integer" in text
    assert _query(dsn, f'SELECT "_source", id, amount FROM "{schema}".cil ORDER BY 1, 2') == before
    assert _stale_shadows(dsn, schema) == []


def test_an_incompatible_type_is_found_under_a_renamed_column(dsn, schema) -> None:
    """``order`` is a reserved word and lands as ``_order``: the check must follow the rename."""
    _run(
        dsn,
        schema,
        "finone",
        _data((1,), "a", order=[3]),
        columns=[*COLUMNS, ("order", "int", "INTEGER")],
    )
    assert "_order" in _columns(dsn, schema)

    with pytest.raises(StrategyError, match="column '_order': 1 value"):
        _run(
            dsn,
            schema,
            "fintwo",
            _data((1,), "b", order=["first"]),
            columns=[*COLUMNS, ("order", "varchar", "TEXT")],
        )
    assert _slices(dsn, schema) == [("finone", 1)]


def test_an_incompatible_type_without_a_configured_label_only_warns(dsn, schema, caplog) -> None:
    """The `DATABASES` loop from before 2.4.0 keeps its behaviour: NULL, plus a warning now."""
    import logging

    _run(
        dsn,
        schema,
        "db_a",
        _data((1,), "a", amount=[10]),
        columns=[*COLUMNS, ("amount", "int", "INTEGER")],
        configured=False,
    )
    with caplog.at_level(logging.WARNING, logger="dbextractors.core.strategies.full_by_source"):
        _run(
            dsn,
            schema,
            "db_b",
            _data((1,), "b", amount=["not a number"]),
            columns=[*COLUMNS, ("amount", "varchar", "TEXT")],
            configured=False,
        )

    assert "column 'amount': 1 value(s)" in caplog.text
    assert _query(dsn, f'SELECT "_source", amount FROM "{schema}".cil ORDER BY 1') == [
        ("db_a", 10),
        ("db_b", None),
    ]


def test_text_that_holds_numbers_still_loads_into_an_integer_column(dsn, schema) -> None:
    """Decided by the values, not by the types: nothing is lost, nothing fails."""
    _run(
        dsn,
        schema,
        "finone",
        _data((1,), "a", amount=[10]),
        columns=[*COLUMNS, ("amount", "int", "INTEGER")],
    )
    _run(
        dsn,
        schema,
        "fintwo",
        _data((1, 2), "b", amount=["42", ""]),
        columns=[*COLUMNS, ("amount", "varchar", "TEXT")],
    )
    assert _query(dsn, f'SELECT "_source", id, amount FROM "{schema}".cil ORDER BY 1, 2') == [
        ("finone", 1, 10),
        ("fintwo", 1, 42),
        ("fintwo", 2, None),
    ]


# --- Two labels, one partition name -------------------------------------------


def test_labels_with_the_same_slug_fail_with_both_names(dsn, schema) -> None:
    """``A-B`` and ``a_b`` both map to ``cil__a_b``. The first slice must survive."""
    _run(dsn, schema, "A-B", _data((1, 2), "x"))

    with pytest.raises(target_pg.TargetError) as err:
        _run(dsn, schema, "a_b", _data((5,), "y"))

    assert "'A-B'" in str(err.value) and "'a_b'" in str(err.value)
    assert _slices(dsn, schema) == [("A-B", 2)]
    assert _stale_shadows(dsn, schema) == []


def test_a_label_with_an_apostrophe_is_recognised_as_its_own(dsn, schema) -> None:
    """The partition bound is read back as a quoted literal; quotes must round-trip."""
    _run(dsn, schema, "o'neil", _data((1,), "x"))
    _run(dsn, schema, "o'neil", _data((1, 2), "x"))
    assert _slices(dsn, schema) == [("o'neil", 2)]


# --- The unique index over (pk, _source) --------------------------------------


def test_the_unique_index_survives_the_end_of_the_run(dsn, fresh_schema) -> None:
    """Checked from a **new** connection. It used to be rolled back by ``close()``."""
    _run(dsn, fresh_schema, "finone", _data((1, 2), "a"))

    assert _indexes(dsn, fresh_schema) == ["idx_cil_id__source_unique"]


def test_the_unique_index_is_built_once(dsn, schema) -> None:
    """Every one-source run is the "last" source; only the first one builds it."""
    for label in ("finone", "fintwo", "finone"):
        _run(dsn, schema, label, _data((1, 2), label))
    assert _indexes(dsn, schema) == ["idx_cil_id__source_unique"]


def test_an_index_under_the_predecessors_name_is_recognised(dsn, schema) -> None:
    """The predecessor's multi-source targets have ``idx_<t>_id_source_unique``.

    Asking by name (``IF NOT EXISTS``) would build a second, identical index over
    the whole table on the first run after an upgrade, holding a ``ShareLock``.
    """
    _run(dsn, schema, "finone", _data((1, 2), "a"))
    conn = _connect(dsn)
    with conn.cursor() as cur:
        cur.execute(f'DROP INDEX "{schema}".idx_cil_id__source_unique')
        cur.execute(
            f'CREATE UNIQUE INDEX idx_cil_id_source_unique ON "{schema}".cil (id, "_source")'
        )
    conn.commit()
    conn.close()

    _run(dsn, schema, "fintwo", _data((1,), "b"))

    assert _indexes(dsn, schema) == ["idx_cil_id_source_unique"]


def test_an_index_that_does_not_fit_is_not_taken_for_the_one_needed(dsn, schema) -> None:
    """A unique index over other columns, or a partial one, does not count."""
    _run(dsn, schema, "finone", _data((1, 2), "a"))
    conn = _connect(dsn)
    with conn.cursor() as cur:
        cur.execute(f'DROP INDEX "{schema}".idx_cil_id__source_unique')
        cur.execute(f'CREATE UNIQUE INDEX other_one ON "{schema}".cil (label, "_source")')
    conn.commit()
    conn.close()

    _run(dsn, schema, "fintwo", _data((1,), "b"))

    assert _indexes(dsn, schema) == ["idx_cil_id__source_unique", "other_one"]


def test_an_existing_target_without_the_index_gets_it(dsn, schema) -> None:
    """The state every partitioned target is in after 2.3.x: the first run builds it."""
    _run(dsn, schema, "finone", _data((1, 2), "a"))
    conn = _connect(dsn)
    with conn.cursor() as cur:
        cur.execute(f'DROP INDEX "{schema}".idx_cil_id__source_unique')
    conn.commit()
    conn.close()

    _run(dsn, schema, "fintwo", _data((1,), "b"))

    assert _indexes(dsn, schema) == ["idx_cil_id__source_unique"]


def test_duplicates_in_one_source_fail_only_that_source(dsn, schema) -> None:
    """Once the index exists, a source with a repeated key cannot swap its slice in."""
    _run(dsn, schema, "finone", _data((1, 2), "a"))
    _run(dsn, schema, "fintwo", _data((1, 2), "b"))

    with pytest.raises(psycopg2.errors.UniqueViolation):
        _run(dsn, schema, "fintwo", _data((3, 3), "c"))

    assert _slices(dsn, schema) == [("finone", 2), ("fintwo", 2)]
    assert _stale_shadows(dsn, schema) == []


def test_concurrent_runs_build_one_index(dsn, fresh_schema) -> None:
    labels = ("finone", "fintwo", "finthree")
    results = _concurrently(
        {lbl: (lambda lbl=lbl: _run(dsn, fresh_schema, lbl, _data((1, 2), lbl))) for lbl in labels}
    )
    assert _errors(results) == {}
    assert _indexes(dsn, fresh_schema) == ["idx_cil_id__source_unique"]
