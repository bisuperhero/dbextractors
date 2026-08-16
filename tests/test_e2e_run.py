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
    "firebird": ("DBX_TEST_FIREBIRD_DSN", "fdb"),
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
