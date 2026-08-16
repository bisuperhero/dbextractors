"""Shared fixtures for the source databases: MySQL, MSSQL and Firebird.

The target has had a live test for a long time. These three did not, and it
showed: every one of their dialects was covered by string assertions only,
which cannot catch the one bug class that actually bites — a type that maps
correctly on paper and whose **value** does not survive the trip.

Each source is switched on by its own environment variable, so working on the
MySQL dialect does not require running MSSQL. Without the variable the tests
skip; ``docker/compose.yml`` and ``.env.example`` between them provide all
three with no further setup.

Nothing here writes to a source. The sources are read-only fixtures; only the
target is written to, and only into a scratch schema.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pytest

#: Environment variable -> the driver the dialect needs. Both have to be
#: present for the tests to run: a DSN without the driver installed would fail
#: at connect time with a message about the import, which is a confusing way to
#: learn that an optional extra is missing.
SOURCES: dict[str, tuple[str, str]] = {
    "mysql": ("DBX_TEST_MYSQL_DSN", "mysql.connector"),
    "mssql": ("DBX_TEST_MSSQL_DSN", "pymssql"),
    "firebird": ("DBX_TEST_FIREBIRD_DSN", "fdb"),
}


def parse_dsn(raw: str) -> dict[str, Any]:
    """``key=value`` pairs into the dict shape ``SOURCE_DB`` uses.

    Deliberately the libpq spelling rather than a URL: a Firebird database is a
    **path** on the server, and a password may contain characters that a URL
    would then have to escape. Space-separated pairs sidestep both.
    """
    params: dict[str, Any] = {}
    for token in raw.split():
        key, _, value = token.partition("=")
        if not _:
            # The token is not `!r`-ed: a DSN malformed *inside* the password is
            # exactly how this branch is reached, and the message goes into a
            # pytest report. The key name is what the reader needs anyway.
            raise ValueError(f"a DSN element of {len(token)} characters is not key=value")
        params[key.strip()] = value.strip()

    if "port" in params:
        params["port"] = int(params["port"])
    return params


def source_params(name: str) -> Optional[dict[str, Any]]:
    """Connection parameters for one source, or ``None`` when it is switched off."""
    variable, module = SOURCES[name]
    raw = os.environ.get(variable)
    if not raw:
        return None

    __import__(module)  # raises when the optional extra is not installed
    return parse_dsn(raw)


def require(name: str) -> dict[str, Any]:
    """Parameters for one source, skipping the test when it is unavailable."""
    variable, module = SOURCES[name]
    try:
        params = source_params(name)
    except ImportError:
        pytest.skip(f"{name}: the {module} driver is not installed")

    if params is None:
        pytest.skip(f"{name} is not available; set {variable} (see .env.example)")
    return params


def engine_for(name: str) -> Any:
    """A SQLAlchemy engine over the source, built the way the package builds it.

    Goes through the dialect's own ``build_conn_str`` on purpose. A test that
    assembled its own URL would verify the database and not the code that has
    to talk to it — and URL assembly is where three of the four dialects differ
    most.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    from dbextractors.entrypoint import resolve_dialect

    params = require(name)
    dialect = resolve_dialect(name)
    conn_str = dialect.build_conn_str(
        {
            "user": params.get("user"),
            "password": params.get("password"),
            "database": params.get("database"),
            "charset": params.get("charset"),
        },
        params.get("host", "127.0.0.1"),
        int(params.get("port") or dialect.default_port),
    )
    return create_engine(conn_str, poolclass=NullPool, connect_args=dict(dialect.connect_args))
