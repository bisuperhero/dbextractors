"""Plumbing for the strategy tests.

A strategy must not know the SQL dialect — which means it can be tested with a
**fake** one. `FakeDialect` hands out prepared batches instead of touching a
source, so these tests only need the target PostgreSQL. Source databases (MySQL,
MSSQL, Firebird) are exercised elsewhere.

The target is real, not a mock: a strategy lives or dies by whether `COPY`, the
shadow table and the upsert actually work. The fake dialect and the context
builders live in ``tests/fakes.py`` so they can be imported when the whole suite
runs, not just this subdirectory.
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
    from dbextractors.golden import scratch

    name = scratch.scratch_schema_name("strategies")
    scratch.create_schema(conn, name)
    conn.commit()
    try:
        yield name
    finally:
        conn.rollback()
        scratch.drop_schema(conn, name)
        conn.commit()
