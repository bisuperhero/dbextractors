"""``run()`` end to end: live source -> conversion -> ``COPY`` -> live target.

Until now the only source that was ever tested all the way into the target was
PostgreSQL. For the other three the pipeline stopped at reading — and the audit
showed exactly the class of bug that leaves open: a type that maps fine and
whose value the target then rejects mid-``COPY``.

These tests run the real entry point over the seeded ``paged`` table (five
rows, safe values — the deliberately hostile ``types_wide`` values are pinned
in the dialect tests) into a scratch schema on the live target. They assert the
whole contract of a full load at once:

- the row count survives the trip,
- the management columns exist (``_timestamp``, ``_deleted_in_source``,
  ``row_hash``),
- ``_deleted_in_source`` is ``FALSE`` everywhere after a full load,
- the status frame reports what was actually written.

The scratch schema uses the ``dbx_golden_`` prefix purely so that a crashed run
is picked up by the golden tool's leftover sweep; ``run()`` itself has no
schema restriction.
"""

from __future__ import annotations

import os
import uuid
from typing import Iterator

import psycopg2
import pytest

from target_pin import pin_target

#: Environment switch -> the driver each source needs. Duplicated from
#: ``tests/dialects/source_db.py`` on purpose: that module lives on the
#: dialect tests' own import path, and a sys.path bridge just to share
#: fifteen lines would couple two test layers that fail differently.
_SOURCES_ENV = {
    "mysql": ("DBX_TEST_MYSQL_DSN", "mysql.connector"),
    "mssql": ("DBX_TEST_MSSQL_DSN", "pymssql"),
    "firebird": ("DBX_TEST_FIREBIRD_DSN", "firebird.driver"),
}


def require_source_params(name: str) -> dict:
    variable, module = _SOURCES_ENV[name]
    raw = os.environ.get(variable)
    if not raw:
        pytest.skip(f"{name} is not available; set {variable} (see .env.example)")
    try:
        __import__(module)
    except ImportError:
        pytest.skip(f"{name}: the {module} driver is not installed")
    params: dict = {}
    for token in raw.split():
        key, _, value = token.partition("=")
        params[key.strip()] = value.strip()
    if "port" in params:
        params["port"] = int(params["port"])
    return params


SOURCES = [
    pytest.param("mysql", "paged", marks=pytest.mark.needs_mysql, id="mysql"),
    pytest.param("mssql", "paged", marks=pytest.mark.needs_mssql, id="mssql"),
    pytest.param("firebird", "PAGED", marks=pytest.mark.needs_firebird, id="firebird"),
]


def _target_dsn() -> str:
    dsn = os.environ.get("DBX_GOLDEN_TEST_DSN")
    if not dsn:
        pytest.skip("no target PostgreSQL; set DBX_GOLDEN_TEST_DSN (see .env.example)")
    return dsn


@pytest.fixture
def target_schema() -> Iterator[str]:
    schema = f"dbx_golden_e2e_{uuid.uuid4().hex[:8]}"
    dsn = _target_dsn()
    pin_target(dsn)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        yield schema
    finally:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.close()


@pytest.mark.parametrize(
    ("source", "table"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_full_load_moves_the_seeded_table_into_the_target(source, table, target_schema) -> None:
    from dbextractors import run

    params = require_source_params(source)
    config = {
        "TABLE": {
            "source_name": table,
            "source_schema": "dbo" if source == "mssql" else "",
            "output_schema": target_schema,
            "output_name": "paged",
            "empty_rows_ok": False,
        },
        "SOURCE_DB": params,
        "LOAD_SETTINGS": {
            "load_method": "full",
            "primary_column": "ID" if source == "firebird" else "id",
        },
        "connection_mode": "direct",
    }

    status = run(config, dialect=source)

    # The status frame is the caller's only window into what happened —
    # it has to say what was written, not merely that something ran.
    assert int(status["rows_written"].iloc[0]) == 5

    dsn = _target_dsn()
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{target_schema}"."paged"')
        assert cur.fetchone()[0] == 5

        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'paged'",
            (target_schema,),
        )
        columns = {row[0] for row in cur.fetchall()}
        # The management columns are the contract a large modelling layer
        # is built on; a full load must create every one of them.
        assert {"_timestamp", "_deleted_in_source", "row_hash"} <= columns, columns

        cur.execute(
            f'SELECT count(*) FROM "{target_schema}"."paged" '
            f'WHERE "_deleted_in_source" IS NOT FALSE'
        )
        assert cur.fetchone()[0] == 0

        # NULL must arrive as NULL: row 5 of every seed has updated_at NULL,
        # and a source that turns it into an empty string or the text 'None'
        # corrupts the target while keeping the row count right.
        cur.execute(f'SELECT updated_at FROM "{target_schema}"."paged" WHERE id = 5')
        assert cur.fetchone()[0] is None


@pytest.mark.needs_firebird
def test_abra_watermark_reads_only_newer_counters_from_firebird(target_schema) -> None:
    """`abra_watermark` end to end over a Firebird 2.5 source, seeded ``ABRA_IDS``.

    The source is read-only, so "new rows" are made by taking rows out of the
    target after a full load. The ID column carries a Czech collation, and the
    rows taken out include the counter reversed to ``CH00000`` — under that
    collation it would sort above the watermark in the wrong place; see the seed.
    """
    from dbextractors import run

    params = require_source_params("firebird")

    def config(load: dict) -> dict:
        return {
            "TABLE": {
                "source_name": "ABRA_IDS",
                "output_schema": target_schema,
                "output_name": "abra_ids",
                "empty_rows_ok": True,
            },
            "SOURCE_DB": params,
            "LOAD_SETTINGS": {"primary_column": "ID", **load},
            "connection_mode": "direct",
        }

    run(config({"load_method": "full"}), dialect="firebird")

    newer = ["0100000101", "00000HC101", "000000D101"]
    outside = ["ZZZZZZZ000", "0200000102"]
    dsn = _target_dsn()
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{target_schema}"."abra_ids"')
        assert cur.fetchone()[0] == 7
        cur.execute(
            f'DELETE FROM "{target_schema}"."abra_ids" WHERE id = ANY(%s)', (newer + outside,)
        )

    settings = {"load_method": "abra_watermark", "abra_id_suffixes": ["101"]}
    first = run(config(settings), dialect="firebird")
    second = run(config(settings), dialect="firebird")

    assert int(first["rows_written"].iloc[0]) == 3
    assert int(second["rows_written"].iloc[0]) == 0
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT id FROM "{target_schema}"."abra_ids"')
        ids = {row[0] for row in cur.fetchall()}
    assert ids == {"1000000101", "Z000000101", *newer}, "other suffixes wait for a full load"


@pytest.mark.needs_mysql
def test_id_watermark_streams_a_mysql_delta_in_checkpoints(target_schema, caplog) -> None:
    """The production path of the 42.7 M-row failure, in miniature and end to end.

    Live MySQL through the streaming read, ``id_watermark`` above a target
    trimmed back to ``id <= 2``, a checkpoint every two rows: the delta (ids
    3–5) must land complete, with a checkpoint on the way.
    """
    import logging

    from dbextractors import run

    params = require_source_params("mysql")

    def config(load_method: str) -> dict:
        return {
            "TABLE": {
                "source_name": "paged",
                "source_schema": "",
                "output_schema": target_schema,
                "output_name": "paged",
                "empty_rows_ok": False,
            },
            "SOURCE_DB": params,
            "LOAD_SETTINGS": {
                "load_method": load_method,
                "primary_column": "id",
                "batch_size": 1,
                "checkpoint_rows": 2,
            },
            "connection_mode": "direct",
        }

    run(config("full"), dialect="mysql")
    dsn = _target_dsn()
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'DELETE FROM "{target_schema}"."paged" WHERE id > 2')

    with caplog.at_level(logging.INFO, logger="dbextractors"):
        status = run(config("id_watermark"), dialect="mysql")

    assert int(status["rows_written"].iloc[0]) == 3
    assert any("Checkpoint 1" in r.getMessage() for r in caplog.records)
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT id FROM "{target_schema}"."paged" ORDER BY id')
        assert [r[0] for r in cur.fetchall()] == [1, 2, 3, 4, 5]
        cur.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name LIKE 'dbx_shadow_%%'",
            (target_schema,),
        )
        assert cur.fetchone()[0] == 0


# --- full_by_source over MySQL: one target, several labelled sources ----------
#
# Dory 2 in miniature: every pool is a cluster of its own with a database of the
# same name and the same tables. Here the pools are one table in one database,
# reached under two host names (`127.0.0.1` and `localhost`) and split by
# `where_clause` — the same database name, the same table name, different data;
# only `source_label` tells the slices apart.


@pytest.fixture
def mysql_pools() -> Iterator[dict]:
    """A source table with two pools' rows; dropped afterwards."""
    import mysql.connector

    params = require_source_params("mysql")
    table = f"fbs_{uuid.uuid4().hex[:8]}"
    conn = mysql.connector.connect(
        host=params.get("host", "127.0.0.1"),
        port=params.get("port", 3306),
        user=params["user"],
        password=params["password"],
        database=params["database"],
    )
    cur = conn.cursor()
    cur.execute(
        # `source`, like the tenant tables it stands in for: a reserved word that
        # lands in the target as `_source`, the default name of the source column.
        f"CREATE TABLE {table} (id INT PRIMARY KEY, pool VARCHAR(8) NOT NULL, "
        "amount DECIMAL(10,2), source VARCHAR(32), updating_date DATETIME NOT NULL)"
    )
    rows = [(1, "a", 10.5), (2, "a", 20), (3, "a", None), (4, "b", 1), (5, "b", 2)]
    cur.executemany(
        f"INSERT INTO {table} VALUES (%s, %s, %s, CONCAT('source.', %s), '2026-11-01 08:00:00')",
        [(*row, row[0]) for row in rows],
    )
    conn.commit()

    def execute(sql: str) -> None:
        cur.execute(sql.format(table=table))
        conn.commit()

    try:
        yield {"params": params, "table": table, "execute": execute}
    finally:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        conn.close()


def _pool_config(
    pools: dict, schema: str, label: str, host: str, pool: str, source_column: str = "_system"
) -> dict:
    return {
        "TABLE": {
            "source_name": pools["table"],
            "output_schema": schema,
            "output_name": "events",
            "where_clause": f"pool = '{pool}'",
        },
        "SOURCE_DB": {**pools["params"], "host": host},
        "LOAD_SETTINGS": {
            "load_method": "full_by_source",
            "primary_column": "id",
            "partition_by_source": True,
            "source_label": label,
            "source_column": source_column,
        },
        "FINGERPRINT": {
            "mode": "datsave",
            "primary_column": "id",
            "timestamp_column": "updating_date",
        },
        "connection_mode": "direct",
    }


def _target_rows(schema: str, table: str = "events") -> list:
    with psycopg2.connect(_target_dsn()) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT "_system", id, amount FROM "{schema}"."{table}" ORDER BY 1, 2')
        return cur.fetchall()


@pytest.mark.needs_mysql
def test_mysql_pools_with_labels_assemble_one_target(mysql_pools, target_schema, monkeypatch):
    """Counts, `_source`, one slice rewritten, an unchanged pool skipped, the fingerprint drained."""
    from decimal import Decimal

    from dbextractors import run
    from dbextractors.dialects import mysql as mysql_dialect

    releases: list = []
    original_release = mysql_dialect._release

    def recording_release(*args, **kwargs):
        releases.append(kwargs["drained"])
        return original_release(*args, **kwargs)

    monkeypatch.setattr(mysql_dialect, "_release", recording_release)

    finone = _pool_config(mysql_pools, target_schema, "finone", "127.0.0.1", "a")
    fintwo = _pool_config(mysql_pools, target_schema, "fintwo", "localhost", "b")

    first_a = run(finone, dialect="mysql")
    first_b = run(fintwo, dialect="mysql")

    assert first_a["source"].tolist() == ["finone"] and int(first_a["rows_written"][0]) == 3
    assert first_b["source"].tolist() == ["fintwo"] and int(first_b["rows_written"][0]) == 2
    assert _target_rows(target_schema) == [
        ("finone", 1, Decimal("10.50")),
        ("finone", 2, Decimal("20.00")),
        ("finone", 3, None),
        ("fintwo", 4, Decimal("1.00")),
        ("fintwo", 5, Decimal("2.00")),
    ]

    # One pool changes upstream: only its slice is rewritten, the other is skipped.
    mysql_pools["execute"](
        "UPDATE {table} SET amount = 99, updating_date = '2026-11-02 08:00:00' WHERE id = 4"
    )
    second_a = run(finone, dialect="mysql")
    second_b = run(fintwo, dialect="mysql")

    assert second_a["load_method"].tolist() == ["unchanged"]
    assert second_b["load_method"].tolist() == ["full_by_source"]
    assert _target_rows(target_schema)[3] == ("fintwo", 4, Decimal("99.00"))
    assert len(_target_rows(target_schema)) == 5

    # Every stream — the fingerprint's single row included — was read to the
    # end and closed normally, never cancelled with `KILL QUERY`.
    assert releases and all(releases), releases

    with psycopg2.connect(_target_dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = 'events'",
            (target_schema,),
        )
        assert [r[0] for r in cur.fetchall()] == ["idx_events_id__system_unique"]
        # The source's own `source` column survives as `_source`.
        cur.execute(f'SELECT id, "_source" FROM "{target_schema}".events ORDER BY id')
        assert cur.fetchall() == [(i, f"source.{i}") for i in range(1, 6)]
        cur.execute(
            f'SELECT source_label FROM "{target_schema}".dbx_source_fingerprints ORDER BY 1'
        )
        assert [r[0] for r in cur.fetchall()] == ["finone", "fintwo"]


@pytest.mark.needs_mysql
def test_mysql_pools_run_from_two_threads_at_once(mysql_pools, target_schema) -> None:
    """`run()` called concurrently from threads of one process, each with its own logger.

    The caller runs three pools side by side; this is the guarantee it relies on:
    both runs finish, the target holds both slices, and each host logger gets
    only its own run's records (forwarding is keyed by thread).
    """
    import threading

    from dbextractors import run

    class Captured:
        def __init__(self) -> None:
            self.lines: list = []

        def debug(self, msg):
            self.lines.append(msg)

        info = warning = error = critical = debug

    jobs = {
        "finone": _pool_config(mysql_pools, target_schema, "finone", "127.0.0.1", "a"),
        "fintwo": _pool_config(mysql_pools, target_schema, "fintwo", "localhost", "b"),
    }
    loggers = {label: Captured() for label in jobs}
    results: dict = {}
    barrier = threading.Barrier(len(jobs))

    def worker(label: str) -> None:
        try:
            barrier.wait()
            results[label] = run(jobs[label], dialect="mysql", logger=loggers[label])
        except Exception as err:  # reported below
            results[label] = err

    threads = [threading.Thread(target=worker, args=(label,)) for label in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    errors = {k: repr(v) for k, v in results.items() if isinstance(v, Exception)}
    assert errors == {}
    assert [r[0] for r in _target_rows(target_schema)] == ["finone"] * 3 + ["fintwo"] * 2
    for label, other in (("finone", "fintwo"), ("fintwo", "finone")):
        text = "\n".join(loggers[label].lines)
        assert f"source={label}" in text, text
        assert f"source={other}" not in text, f"{other}'s records reached {label}'s logger"


@pytest.mark.needs_mssql
def test_mssql_multi_source_loop_still_labels_slices_by_database(target_schema) -> None:
    """The pre-2.4.0 shape, unchanged: no `source_label`, `_source` is the database name.

    Plus the one intended difference: the unique index over ``(id, _source)`` is
    now committed, so it is there from a fresh connection.
    """
    from dbextractors import run

    params = require_source_params("mssql")
    config = {
        "TABLE": {
            "source_name": "paged",
            "source_schema": "dbo",
            "output_schema": target_schema,
            "output_name": "paged",
        },
        "SOURCE_DB": {k: v for k, v in params.items() if k != "database"},
        "DATABASES": [params["database"]],
        "LOAD_SETTINGS": {
            "load_method": "full_by_source",
            "primary_column": "id",
            "partition_by_source": True,
            "multi_source": True,
        },
        "FINGERPRINT": {"mode": "count", "primary_column": "id"},
        "connection_mode": "direct",
    }

    first = run(config, dialect="mssql")
    second = run(config, dialect="mssql")

    assert first["source"].tolist() == [params["database"]]
    assert int(first["rows_written"][0]) == 5
    assert second["load_method"].tolist() == ["unchanged"]
    with psycopg2.connect(_target_dsn()) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT DISTINCT "_source" FROM "{target_schema}".paged')
        assert cur.fetchall() == [(params["database"],)]
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = 'paged'",
            (target_schema,),
        )
        assert [r[0] for r in cur.fetchall()] == ["idx_paged_id__source_unique"]


@pytest.mark.needs_mysql
def test_mysql_source_column_collision_fails_before_anything_is_written(
    mysql_pools, target_schema
) -> None:
    """Without `source_column`, the source's `source` would be overwritten by the label."""
    from dbextractors import run
    from dbextractors.entrypoint import SourceExtractionError

    config = _pool_config(mysql_pools, target_schema, "finone", "127.0.0.1", "a", "_source")
    with pytest.raises(SourceExtractionError, match="lands in the target as '_source'") as err:
        run(config, dialect="mysql")

    assert err.value.statuses[0]["source"] == "finone"
    with psycopg2.connect(_target_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'"{target_schema}".events',))
        assert cur.fetchone()[0] is None
