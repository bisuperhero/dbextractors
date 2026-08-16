"""Tests that a golden-test session really pins its settings on the server.

The audit found that deleting the ``SET`` loop in ``session.connect`` survived
the whole suite: every comparison runs both sides inside the *same* session, so
nothing mismatched — but a report saved to JSON would stop being reproducible
between runs, which is the whole point of pinning. These tests therefore ask the
**server** (``SHOW``) what the session looks like, instead of trusting the dict
in ``session.DETERMINISM_SETTINGS`` — a removed loop leaves the dict intact and
only the server can tell the difference.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_pg

#: The representations these settings pin down are listed in the module
#: docstring of ``session.py``: timestamptz, bytea, float8 and date rendering.
_PINNED = {
    "timezone": "UTC",
    "DateStyle": "ISO, YMD",
    "bytea_output": "hex",
    "extra_float_digits": "3",
}


def _show(conn, name: str) -> str:
    with conn.cursor() as cur:
        cur.execute(f"SHOW {name}")
        return cur.fetchone()[0]


@pytest.mark.parametrize("conn_fixture", ["ro_conn", "rw_conn"])
@pytest.mark.parametrize(("setting", "expected"), sorted(_PINNED.items()))
def test_the_session_pins_determinism_settings(request, conn_fixture, setting, expected):
    """Both connection flavours: the read-only comparing session and the
    read-write setup session must render values identically, otherwise the test
    data itself would differ from what the comparison later reads."""
    conn = request.getfixturevalue(conn_fixture)
    assert _show(conn, setting) == expected


def test_the_comparing_session_is_read_only_on_the_server(ro_conn):
    """The guard lives in the database, not in our code — so ask the database."""
    assert _show(ro_conn, "default_transaction_read_only") == "on"
