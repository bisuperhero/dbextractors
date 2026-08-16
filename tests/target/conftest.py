"""Scaffolding for the `core.target_pg` tests.

Most of the module is SQL, so it is tested against a **live** PostgreSQL. A mock
would only prove that strings can be assembled — not that `TRUNCATE` +
`INSERT SELECT` really keeps the views, or that `COPY` gives back the same value
it was handed.

The database and the safeguard on scratch schemas are taken over from the golden
test (``tests/golden/conftest.py``, ``dbextractors.golden.scratch``) — it is the
same environment, and we do not want two different places deciding where writing
is allowed.
"""

from __future__ import annotations

import os
import pathlib
from typing import Iterator

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@pytest.fixture(scope="session")
def dsn() -> str:
    from dbextractors.golden import session

    explicit = os.environ.get("DBX_GOLDEN_TEST_DSN")
    if explicit:
        return explicit
    _load_dotenv()
    try:
        return session.resolve_dsn()
    except session.SessionError:
        pytest.skip("No PostgreSQL available. Set DBX_GOLDEN_TEST_DSN or POSTGRES_* in .env.")


@pytest.fixture()
def conn(dsn: str) -> Iterator:
    """A writable connection with **autocommit switched off**.

    Off on purpose: `swap_shadow_table` insists on it, and the tests should run
    in the same mode as a live run.
    """
    import psycopg2

    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


@pytest.fixture()
def schema(conn) -> Iterator[str]:
    """A scratch schema prefixed ``dbx_golden_``, dropped after the test.

    ``scratch.scratch_schema`` is deliberately not used: it does not commit, so
    the ``DROP`` would be rolled back when the connection is torn down and the
    schemas would pile up. The ``rollback()`` before the cleanup is necessary —
    a test may leave the transaction in an error state
    (`InFailedSqlTransaction`), and then the `DROP` would not go through.
    """
    from dbextractors.golden import scratch

    name = scratch.scratch_schema_name("target")
    scratch.create_schema(conn, name)
    conn.commit()
    try:
        yield name
    finally:
        conn.rollback()
        scratch.drop_schema(conn, name)
        conn.commit()
