"""Support fixtures for the dialect tests that need a live database.

Most dialect tests work on strings and need no database. The ones in
``test_postgres_db.py`` verify that a value really does survive `COPY` into the
target — and that cannot be verified on strings.

Both the source and the target are scratch schemas prefixed with
``dbx_golden_``; PostgreSQL happens to be both here, because the same database
serves as source and as target. Nothing outside those two schemas is touched.
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
    from target_pin import pin_target

    explicit = os.environ.get("DBX_GOLDEN_TEST_DSN")
    if explicit:
        return pin_target(explicit)
    _load_dotenv()
    try:
        return pin_target(session.resolve_dsn())
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
    """The schema holding the **source** table."""
    from dbextractors.golden import scratch

    name = scratch.scratch_schema_name("source")
    scratch.create_schema(conn, name)
    conn.commit()
    try:
        yield name
    finally:
        conn.rollback()
        scratch.drop_schema(conn, name)
        conn.commit()


@pytest.fixture()
def cil(conn) -> Iterator[str]:
    """The schema holding the **target** table."""
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
