"""Support fixtures for the golden test.

The tests run against a live PostgreSQL, because the comparison is nine tenths
SQL — a mock would only prove that we can assemble strings, not that what we
assemble gives the right answer.

The database comes from ``DBX_GOLDEN_TEST_DSN``, or from ``.env`` in the repo
root (the same ``POSTGRES_*`` variables the client repositories use). When there
is none, the tests skip — they do not fail.

Everything happens in scratch schemas prefixed with ``dbx_golden_``, which are
dropped afterwards. Nothing else is touched.
"""

from __future__ import annotations

import os
import pathlib
from typing import Iterator

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Loads .env from the repo root, unless the variables are already set."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _resolve_dsn() -> str | None:
    from dbextractors.golden import session
    from target_pin import pin_target

    explicit = os.environ.get("DBX_GOLDEN_TEST_DSN")
    if explicit:
        return pin_target(explicit)

    _load_dotenv()
    try:
        return pin_target(session.resolve_dsn())
    except session.SessionError:
        return None


@pytest.fixture(scope="session")
def dsn() -> str:
    resolved = _resolve_dsn()
    if resolved is None:
        pytest.skip("No PostgreSQL available. Set DBX_GOLDEN_TEST_DSN or POSTGRES_* in .env.")
    return resolved


@pytest.fixture()
def rw_conn(dsn: str) -> Iterator:
    """A connection allowed to write — only for setting up test data."""
    from dbextractors.golden import session

    with session.connect(dsn, read_only=False) as conn:
        yield conn


@pytest.fixture()
def ro_conn(dsn: str) -> Iterator:
    """The connection the comparison runs on. Read-only, as in a real run."""
    from dbextractors.golden import session

    with session.connect(dsn, read_only=True) as conn:
        yield conn


@pytest.fixture()
def workspace(rw_conn) -> Iterator[str]:
    """A scratch schema that is dropped after the test."""
    from dbextractors.golden import scratch

    with scratch.scratch_schema(rw_conn, "test") as schema:
        yield schema
