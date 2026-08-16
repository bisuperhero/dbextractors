"""Where the package **writes** — access to the target PostgreSQL.

This module grew out of a production failure::

    [<table>] extraction failed: PostgreSQL at postgres:54333 is not responding.

The cause was not the pipeline configuration but where the package took the target
from. ``entrypoint._target_connection`` called
``dbextractors.golden.session.resolve_dsn``, i.e. the resolver belonging to the
**golden test**, which builds the DSN out of ``POSTGRES_HOST`` + ``POSTGRES_PORT``.
In that deployment, however, ``POSTGRES_PORT`` means something else::

    postgres:
      ports:
        - ${POSTGRES_PORT}:5432     # the port published on the host

Inside the docker network the database listens on 5432 while `POSTGRES_PORT` is
54333. That is precisely why the ``warehouse`` profile in `io_config.yaml` has the
port hard-coded while taking the host from a variable — and why the old extractors,
which went through `io_config.yaml`, hit the right database. In another deployment
the two values happened to coincide, so nothing showed up there.

The order in which the target is looked up is therefore:

1. ``DBX_TARGET_DSN`` — explicit, overrides everything. For cases where writing
   goes somewhere other than where Mage reads.
2. **``io_config.yaml``**, profile ``warehouse``. It can be changed with the
   root-level key ``TARGET_PROFILE`` in the pipeline configuration (more specific,
   applies to a single pipeline) or with the ``DBX_TARGET_PROFILE`` variable (for
   the whole container). This is the same thing the replaced extractors used — so
   dbextractors lands where they landed, including where ``POSTGRES_*`` means
   something else.
3. ``POSTGRES_*`` — for runs **outside** Mage, where there is no `io_config.yaml`.
4. ``DBX_GOLDEN_DSN`` as the last fallback, so that deployments which set it because
   of this bug keep working.

Backwards compatibility: where `io_config.yaml` and ``POSTGRES_*`` say the same
thing, nothing changes — `io_config.yaml` wins with the identical value.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from dbextractors.core import secrets

#: The profile in `io_config.yaml` that the replaced extractors used. The MySQL,
#: MSSQL and Firebird streaming extractors all had `config_profile = 'warehouse'`.
DEFAULT_PROFILE = "warehouse"


class TargetConnectionError(RuntimeError):
    """The target to write to could not be determined."""


def _from_io_config(profile: str) -> Optional[str]:
    """DSN from Mage's `io_config.yaml`, or ``None`` when it cannot be read.

    Returns ``None`` (not an exception) in every case where `io_config.yaml` is
    simply unavailable — a run outside Mage, a missing profile, a missing key. The
    caller then moves on to the next source in the order.
    """
    try:
        from mage_ai.io.config import ConfigFileLoader, ConfigKey
        from mage_ai.settings.repo import get_repo_path
    except ImportError:
        return None

    path = os.path.join(get_repo_path(), "io_config.yaml")
    if not os.path.isfile(path):
        return None

    try:
        loader = ConfigFileLoader(path, profile)
        host = loader.get(ConfigKey.POSTGRES_HOST)
        database = loader.get(ConfigKey.POSTGRES_DBNAME)
        user = loader.get(ConfigKey.POSTGRES_USER)
        if not (host and database and user):
            return None
        port = loader.get(ConfigKey.POSTGRES_PORT) or 5432
        password = loader.get(ConfigKey.POSTGRES_PASSWORD) or ""
    except Exception:
        # A broken or incomplete profile is no reason to bring the run down — it is a
        # reason to reach for the next source. But it has to be visible that it
        # happened: silently moving on is exactly what produced this bug.
        import logging

        logging.getLogger(__name__).warning(
            "⚠️ Profile %r in io_config.yaml could not be read, trying POSTGRES_*.", profile
        )
        return None

    return build_dsn(host, port, database, user, password)


def build_dsn(host: Any, port: Any, database: Any, user: Any, password: Any) -> str:
    """A libpq keyword/value DSN with every value quoted and escaped.

    Interpolating the values bare is what these builders used to do, and it is
    wrong for any value containing a space, a single quote or a backslash — libpq
    parses ``password=my secret`` as a password of ``my`` followed by a second
    keyword ``secret``, and reports ``invalid dsn: missing "=" after "secret"``.

    That was worse than an unusable connection. The error text quotes the offending
    fragment, so **part of the password ended up in the message** — which is how
    this was found: a leak audit triggered it with a password containing a space.
    Redacting the message was the immediate fix; this is the cause.

    Quoting unconditionally is safe: libpq treats ``dbname=x`` and ``dbname='x'``
    identically, so a configuration that connects today connects unchanged. What
    changes is that one which could never connect now can.
    """
    return " ".join(
        f"{keyword}={_quote(value)}"
        for keyword, value in (
            ("host", host),
            ("port", port),
            ("dbname", database),
            ("user", user),
            ("password", password),
        )
    )


def _quote(value: Any) -> str:
    """One libpq value: single-quoted, with ``\\`` and ``'`` backslash-escaped."""
    text = "" if value is None else str(value)
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _from_env_vars(source: Mapping[str, str]) -> Optional[str]:
    host = source.get("POSTGRES_HOST")
    database = source.get("POSTGRES_DB")
    user = source.get("POSTGRES_USER")
    if not (host and database and user):
        return None
    port = source.get("POSTGRES_PORT", "5432")
    password = source.get("POSTGRES_PASSWORD", "")
    return build_dsn(host, port, database, user, password)


def resolve_target_dsn(
    env: Optional[Mapping[str, str]] = None, *, profile: Optional[str] = None
) -> str:
    """Where writing goes. The order of sources is described in the module header.

    ``profile`` is the root-level ``TARGET_PROFILE`` key from the pipeline
    configuration — it takes precedence over the environment variable because it is
    more specific: the variable applies to the whole container, the key to a single
    pipeline.
    """
    source: Mapping[str, str] = os.environ if env is None else env

    explicit = source.get("DBX_TARGET_DSN")
    if explicit:
        return explicit

    dsn = _from_io_config(profile or source.get("DBX_TARGET_PROFILE") or DEFAULT_PROFILE)
    if dsn:
        return dsn

    dsn = _from_env_vars(source)
    if dsn:
        return dsn

    fallback = source.get("DBX_GOLDEN_DSN")
    if fallback:
        return fallback

    raise TargetConnectionError(
        "Could not determine where to write. Tried DBX_TARGET_DSN, profile "
        f"{profile or source.get('DBX_TARGET_PROFILE') or DEFAULT_PROFILE!r} "
        "in io_config.yaml and POSTGRES_HOST / POSTGRES_DB / POSTGRES_USER. "
        "Set at least one of them."
    )


def describe_dsn(dsn: str) -> str:
    """The DSN without the password — for the log. Passwords must never be logged.

    Delegates to `core.secrets.redact` so that there is one redaction, not two. The
    original form tokenised on spaces and dropped the `password=…` token entirely;
    this one **masks** it (`password=***`), so the log still shows that the DSN
    carried a password.
    """
    return secrets.redact(dsn)
