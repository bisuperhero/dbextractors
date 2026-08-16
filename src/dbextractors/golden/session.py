"""The PostgreSQL connection used by the golden test — and its two guards.

**Guard 1: determinism.** Several textual representations in PostgreSQL depend
on session settings rather than on the data. Verified on PG 17.9:

| what | depends on | evidence |
|---|---|---|
| ``timestamptz::text`` | ``timezone`` | ``12:00:00+00`` in UTC vs ``13:00:00+01`` in CET |
| ``bytea::text`` | ``bytea_output`` | ``\\x0102`` (hex) vs ``\\001\\002`` (escape) |
| ``float8::text`` | ``extra_float_digits`` | number of digits printed |
| ``date::text`` | ``DateStyle`` | ISO vs German |
| ``interval::text`` | ``IntervalStyle`` | |

Because both sides are compared inside the same session, this alone would not
produce a false mismatch. It would produce something worse: a report saved to
JSON would not be reproducible, and two runs would give different numbers. So it
is pinned down on every connection.

**Guard 2: read-only.** The golden test may only **read** from the production
database. The comparing session therefore runs with
``default_transaction_read_only = on``. That is not just a declaration: any
attempt to write fails with an error from the database, not from a check of ours
that could be bypassed.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Mapping, Optional

from dbextractors.core import secrets, target_conn

if TYPE_CHECKING:  # pragma: no cover
    import psycopg2.extensions

logger = logging.getLogger("dbextractors.golden")

#: Settings pinned at the start of every session. Without them the report is not
#: reproducible between runs.
DETERMINISM_SETTINGS: dict[str, str] = {
    "timezone": "UTC",
    "DateStyle": "ISO, YMD",
    "IntervalStyle": "iso_8601",
    "bytea_output": "hex",
    "extra_float_digits": "3",
    "lc_numeric": "C",
    "lc_monetary": "C",
    # "0" DISABLES the timeout — deliberately. Checksums over a 15 M row table
    # legitimately run for minutes (247 s observed on one wide table), and a
    # comparison that dies on a timeout reports ERROR where the honest answer
    # was "still working". An earlier comment here claimed the opposite of what
    # the value does; an audit caught it. Pin the session's behaviour, not a
    # guess about how long a table is allowed to be.
    "statement_timeout": "0",
}


class SessionError(RuntimeError):
    """A usable session could not be established."""


def dsn_from_env(env: Optional[dict] = None) -> str:
    """Builds a DSN from environment variables.

    ``DBX_GOLDEN_DSN`` takes precedence over everything else. Otherwise the DSN
    is assembled from ``POSTGRES_*`` — the same variables the deployment
    repositories already use, so that credentials are not configured twice.
    """
    source: Mapping[str, str] = os.environ if env is None else env

    explicit = source.get("DBX_GOLDEN_DSN")
    if explicit:
        return explicit

    host = source.get("POSTGRES_HOST")
    database = source.get("POSTGRES_DB")
    user = source.get("POSTGRES_USER")
    if not (host and database and user):
        raise SessionError(
            "No PostgreSQL credentials. Set DBX_GOLDEN_DSN, or "
            "POSTGRES_HOST / POSTGRES_DB / POSTGRES_USER (and POSTGRES_PASSWORD)."
        )

    port = source.get("POSTGRES_PORT", "5432")
    password = source.get("POSTGRES_PASSWORD", "")
    return target_conn.build_dsn(host, port, database, user, password)


def _dsn_field(dsn: str, key: str) -> Optional[str]:
    """One value out of a libpq DSN, quoted or not.

    Both spellings have to be handled. This module now builds its DSN with every
    value quoted — a password containing a space is otherwise unusable, see
    ``target_conn.build_dsn`` — while ``DBX_GOLDEN_DSN`` is written by a person and
    is normally bare. A parser that understood only one of the two would work right
    up until somebody's password had a space in it, and then hand ``int()`` the
    string ``'5432'`` with the quotes still on.
    """
    match = re.search(rf"(?:^|\s){re.escape(key)}=('(?:[^'\\]|\\.)*'|[^\s]+)", dsn)
    if not match:
        return None
    value = match.group(1)
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        # Undo what `build_dsn` escaped: `\\` -> `\`, `\'` -> `'`.
        return re.sub(r"\\(.)", r"\1", value[1:-1])
    return value


def _running_under_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def wsl_windows_host() -> Optional[str]:
    """Address of the Windows host as seen from WSL, that is, the default gateway.

    It is not constant — it changes when WSL restarts, so it must not be written
    into ``.env``. Hence it is looked up at run time.
    """
    try:
        out = subprocess.run(
            ["ip", "route"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"^default via (\S+)", out, re.MULTILINE)
    return match.group(1) if match else None


def _reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_dsn(dsn: Optional[str] = None, *, allow_wsl_fallback: bool = True) -> str:
    """Returns a DSN that can actually be connected to.

    The special case this exists for: when the development PostgreSQL runs on
    Windows and the work happens inside WSL, the LAN address of that **same**
    machine is not reachable from WSL (Windows Firewall blocks it), while the
    gateway of the virtual adapter is. The gateway address also changes whenever
    WSL restarts.

    A substituted address is **always logged**. Silently redirecting somewhere
    other than where the user aimed is unacceptable in a tool meant to be an
    oracle.
    """
    dsn = dsn or dsn_from_env()
    host = _dsn_field(dsn, "host")
    port = int(_dsn_field(dsn, "port") or 5432)

    if not host or _reachable(host, port):
        return dsn

    if not (allow_wsl_fallback and _running_under_wsl()):
        raise SessionError(f"PostgreSQL at {host}:{port} does not answer.")

    gateway = wsl_windows_host()
    if not gateway or not _reachable(gateway, port):
        raise SessionError(
            f"PostgreSQL at {host}:{port} does not answer, and it cannot be "
            f"reached through the WSL gateway either ({gateway or 'not found'})."
        )

    logger.warning(
        "Host %s:%s does not answer from WSL, using gateway %s:%s. "
        "The gateway address changes when WSL restarts, so it is looked up at run time.",
        host,
        port,
        gateway,
        port,
    )
    return dsn.replace(f"host={host}", f"host={gateway}")


@contextmanager
def connect(
    dsn: Optional[str] = None, *, read_only: bool = True
) -> Iterator[psycopg2.extensions.connection]:
    """Opens a session with determinism pinned down.

    Args:
        dsn: When ``None``, it is taken from the environment.
        read_only: ``True`` by default. Writing is only allowed where the harness
            creates its own helper schemas — never while comparing.
    """
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise SessionError(
            "The golden test needs psycopg2: pip install 'mage-db-extractors[golden]'"
        ) from exc

    resolved = resolve_dsn(dsn)
    try:
        conn = psycopg2.connect(resolved)
    except Exception as err:
        # This tool is run by hand and its output is pasted into tickets, so an
        # uncaught psycopg2 traceback here travels further than a Mage log line
        # does. psycopg2 quotes back the token it choked on — with a password
        # containing a space that token *is* the password, see
        # `secrets.dsn_secrets`. Re-raised as `SessionError` so `cli.main` prints
        # one redacted line and exits 2 (ERROR) instead of dumping a traceback;
        # `from None` keeps the unredacted original out of the `__cause__` chain.
        raise SessionError(
            f"Could not connect to PostgreSQL: "
            f"{secrets.redact(err, extra=secrets.dsn_secrets(resolved))}"
        ) from None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for name, value in DETERMINISM_SETTINGS.items():
                cur.execute(f"SET {name} = %s", (value,))
            if read_only:
                # The guard lives on the database side, not in our code. An
                # attempt to write fails with a PostgreSQL error that can neither
                # be overlooked nor worked around.
                cur.execute("SET default_transaction_read_only = on")
        yield conn
    finally:
        conn.close()


def describe_session(conn: psycopg2.extensions.connection) -> dict:
    """What goes into the report, so that a run can be traced and repeated."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT version(), current_database(), current_user, "
            "current_setting('timezone'), current_setting('default_transaction_read_only')"
        )
        version, database, user, timezone, read_only = cur.fetchone()
    return {
        "server_version": version.split(" on ")[0],
        "database": database,
        "user": user,
        "timezone": timezone,
        "read_only": read_only == "on",
    }
