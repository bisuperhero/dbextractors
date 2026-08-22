"""Every path a credential could take to a log line, an exception or a file.

`test_secrets.py` tests the redaction itself. This file tests the **places that
have to call it** — one test per path found by the P5 audit, each asserting that
a canary password, or a private key, is absent from

1. the message of whatever comes out,
2. the rendered traceback (``traceback.format_exception``), because a handler
   that prints one sees more than ``str(err)`` does,
3. the ``__cause__`` chain, because a redacted message chained to an unredacted
   original is not a redaction at all.

Two kinds of test live here and the difference matters when one of them fails:

- **A path that leaked.** It was triggered on purpose (a wrong password, an
  unreachable host, an invalid DSN, a tunnel that could not bind), the leak was
  observed and the test pins the fix.
- **A path that is safe only because a driver happens not to talk about its own
  connection string.** Checked live against psycopg2 2.9.12, mysql-connector,
  pymssql and fdb: none of the four puts the DSN or the URL into a failed
  connection's message. That is not a guarantee, it is today's behaviour of four
  third-party packages — so it is pinned here and a driver upgrade that changes
  it fails a test instead of filling a production log.

The canary is deliberately distinctive: a leak cannot hide inside a long string.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import pytest

from dbextractors.core import secrets

#: Distinctive enough to grep for anywhere in a multi-line traceback.
CANARY = "pw-CANARY-8213"

#: A URL of the shape all four dialects build. `_URL_CREDENTIALS` has to catch it.
CANARY_URL = f"mysql+mysqlconnector://svc:{CANARY}@db.example.com:3306/prod"

#: A DSN of the shape `target_conn` and `golden.session` build.
CANARY_DSN = f"host=db.example.com port=5432 dbname=w user=svc password={CANARY}"

#: A private key, in the shape `ssh` writes and reads. The canary sits in the
#: base64 body, where only a redaction of the whole block removes it.
CANARY_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    f"b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ{CANARY}AAAAAAAAtzc2gt\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


def rendered(err: BaseException) -> str:
    """Message, traceback and the whole ``__cause__`` chain as one string.

    This is what a caller printing the failure actually sees. Asserting only on
    ``str(err)`` would pass for a redacted message chained to an unredacted
    original — the exact mistake `entrypoint` guards against with ``from None``.
    """
    return "".join(traceback.format_exception(type(err), err, err.__traceback__))


def assert_clean(text: str, *, what: str) -> None:
    """The canary — and any recognisable piece of it — is nowhere in ``text``."""
    assert CANARY not in text, f"{what} leaked the password: {text!r}"
    assert "CANARY" not in text, f"{what} leaked a fragment of the password: {text!r}"
    assert "-----BEGIN" not in text, f"{what} leaked a private key: {text!r}"


# --- The redaction's two new shapes -----------------------------------------
#
# Both were added by the P5 audit for a path below. They are tested here rather
# than in test_secrets.py because they only exist for those paths.


@pytest.mark.parametrize(
    "header",
    [
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_redact_masks_a_private_key_in_every_spelling(header: str) -> None:
    """``ssh`` accepts all of these and `core/tunnel.py` logs whatever it is given."""
    end = header.replace("BEGIN", "END")
    text = f"ssh -i {header}\nbody-{CANARY}-body\n{end} -N"

    result = secrets.redact(text)

    assert_clean(result, what="redact")
    # The rest of the command line survives — that line exists so a failing
    # tunnel can be reproduced by hand.
    assert "ssh -i" in result and "-N" in result


def test_redact_masks_a_private_key_whose_end_marker_never_arrived() -> None:
    """Truncated output is the normal way a key reaches a log: a line limit, a
    CRLF file, a message cut off. Everything after an unterminated BEGIN is key
    material."""
    text = f"could not read -----BEGIN OPENSSH PRIVATE KEY-----\nbody-{CANARY}"

    assert_clean(secrets.redact(text), what="redact")


def test_dsn_secrets_returns_the_password_and_its_pieces() -> None:
    """The pieces are what covers psycopg2 quoting a single token back — see
    `secrets.dsn_secrets`."""
    found = secrets.dsn_secrets(f"host=h user=u password=pw {CANARY} tail dbname=d")

    assert f"pw {CANARY} tail" in found
    assert CANARY in found


@pytest.mark.parametrize(
    "dsn",
    [
        f"host=h password={CANARY} dbname=d",
        f"host=h password='{CANARY}' dbname=d",
        f'host=h password="{CANARY}" dbname=d',
        f"host=h password={CANARY}",
    ],
)
def test_dsn_secrets_copes_with_quoting_and_position(dsn: str) -> None:
    assert CANARY in secrets.dsn_secrets(dsn)


def test_dsn_secrets_finds_nothing_in_a_dsn_without_a_password() -> None:
    assert secrets.dsn_secrets("host=h user=u dbname=d") == []


# --- core/tunnel.py ---------------------------------------------------------
#
# `SOURCE_DB.ssh_pkey` is contracted to be a path. When it holds the key itself
# — a paste into YAML, the mistake `normalize_private_key_contents` exists to
# repair a milder form of — the package used to put that value into a log line,
# into `ssh -i` and into an exception message. Triggered: `os.stat` on a
# 400-character "path" fails with ENAMETOOLONG and the whole key was logged.


def test_a_private_key_pasted_into_ssh_pkey_does_not_reach_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dbextractors.core import tunnel

    with caplog.at_level(logging.DEBUG, logger="dbextractors.core.tunnel"):
        tunnel.ensure_private_key_permissions(CANARY_KEY, show_debug=True)

    assert caplog.text, "the path was not exercised — the key was accepted as a file name"
    assert_clean(caplog.text, what="ensure_private_key_permissions")


def test_a_tunnel_that_cannot_come_up_does_not_report_the_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``ssh`` echoes its own ``-i`` argument back when it cannot read it.

    The stderr below is the real thing, copied from an OpenSSH run against a
    tunnel that could not bind; the process is faked only so the test needs
    neither an ``ssh`` binary nor two seconds of waiting. Three exits are checked
    at once: the ``RuntimeError`` message, its traceback, and the debug log line
    that prints the argv.
    """
    import subprocess

    from dbextractors.core import tunnel

    class _FakeProc:
        pid = 424242

        def terminate(self) -> None:
            return None

        def wait(self, timeout: Optional[float] = None) -> int:
            return 0

        def communicate(self, timeout: Optional[float] = None) -> tuple[bytes, bytes]:
            return (
                b"",
                f"Warning: Identity file {CANARY_KEY} not accessible: "
                f"No such file or directory.\n".encode(),
            )

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _FakeProc())
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: False)
    monkeypatch.setattr(tunnel.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(tunnel.os, "killpg", lambda pgid, sig: None)

    source_db = {
        "ssh_address_or_host": "bastion.example.com",
        "ssh_username": "deploy",
        "ssh_pkey": CANARY_KEY,
        "local_bind_address": ["127.0.0.1", 15432],
        "remote_bind_address": ["dbhost.internal", 5432],
        "ssh_wait_timeout": 0.1,
    }

    with (
        caplog.at_level(logging.DEBUG, logger="dbextractors.core.tunnel"),
        pytest.raises(RuntimeError) as err,
        tunnel.open_tunnel(source_db, "ssh", show_debug=True),
    ):
        pass  # pragma: no cover - the tunnel never comes up

    assert "SSH tunnel could not be established" in str(err.value)
    assert_clean(rendered(err.value), what="open_tunnel")
    assert_clean(caplog.text, what="the SSH tunnel debug log")


# --- entrypoint._attach_session_sql -----------------------------------------


def test_a_failing_session_statement_does_not_log_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listener runs on **every** connection, so a driver that quoted its own
    DSN back would leak it once per batch, not once per run.

    The real path was triggered against MySQL (an unknown session variable): the
    listener logs and the run carries on, which is the behaviour that must not
    change. Here the driver error is one that *does* carry the URL, so that the
    redaction is what the assertion depends on.
    """
    from dbextractors import entrypoint

    captured: dict = {}

    def fake_listens_for(target: Any, identifier: str):
        def decorator(fn):
            captured["listener"] = fn
            return fn

        return decorator

    monkeypatch.setattr("sqlalchemy.event.listens_for", fake_listens_for)

    class _Cursor:
        def execute(self, statement: str) -> None:
            raise RuntimeError(f"lost the connection {CANARY_URL}")

        def close(self) -> None:
            return None

    class _Conn:
        def cursor(self) -> _Cursor:
            return _Cursor()

    class _Logger:
        def __init__(self) -> None:
            self.lines: list = []

        def warning(self, message: str, *args: Any) -> None:
            self.lines.append(message % args)

    logger = _Logger()
    dialect = type("_D", (), {"session_sql": ("SET SESSION net_write_timeout = 600",)})()
    entrypoint._attach_session_sql(object(), dialect, logger)

    captured["listener"](_Conn(), None)

    assert logger.lines, "the failure was swallowed — it has to be logged"
    assert_clean("\n".join(logger.lines), what="the session-SQL listener")


# --- entrypoint.run: the funnel ---------------------------------------------


class _FakeStrategy:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def run(self, ctx: Any) -> Any:
        raise self.error


def _run_failing(monkeypatch: pytest.MonkeyPatch, error: BaseException):
    """Run one source whose strategy fails with ``error``, with no database."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from dbextractors import entrypoint
    from dbextractors.core.tunnel import TunnelAddress

    monkeypatch.setattr(
        "dbextractors.core.strategies.base.resolve_strategy",
        lambda _name, _settings=None: _FakeStrategy(error),
        raising=True,
    )
    monkeypatch.setattr(
        entrypoint,
        "build_context",
        lambda parsed, dialect, **kwargs: SimpleNamespace(
            target=SimpleNamespace(schema="s", table="target"),
            target_conn=SimpleNamespace(close=lambda: None),
            source_label=kwargs.get("source_label"),
            log=lambda *a, **k: None,
            target_names=["id"],
            overwrite_types={},
            orig_type_map={},
            surrogate=None,
            where=None,
            debug=False,
        ),
    )

    @contextmanager
    def _fake_open_tunnel(source_db, mode=None, *, probe=None, show_debug=False):
        yield TunnelAddress(host="127.0.0.1", port=54321, mode="ssh")

    monkeypatch.setattr("dbextractors.core.tunnel.open_tunnel", _fake_open_tunnel)

    return entrypoint.run(
        {
            "TABLE": {"source_name": "t", "output_name": "target", "output_schema": "s"},
            "SOURCE_DB": {"user": "u", "password": CANARY, "database": "d", "host": "h"},
            "LOAD_SETTINGS": {"load_method": "full", "primary_column": "id"},
        },
        "mysql",
    )


def test_the_run_funnel_redacts_the_message_the_traceback_and_the_status_frame(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`run` catches per database and the caught exception leaves by four doors.

    The package log (with a full traceback), the Mage log, the ``error`` column
    of the returned DataFrame, and `SourceExtractionError`, which quotes that
    column. One unredacted door is enough, so all four are asserted here.
    """
    from dbextractors import entrypoint

    error = RuntimeError(f"could not read from {CANARY_URL}")

    with (
        caplog.at_level(logging.DEBUG, logger="dbextractors.entrypoint"),
        pytest.raises(entrypoint.SourceExtractionError) as err,
    ):
        _run_failing(monkeypatch, error)

    assert "extraction failed" in caplog.text, "the failure was not logged at all"
    assert "Traceback" in caplog.text, "the traceback is what makes the failure debuggable"
    assert_clean(caplog.text, what="the run log")
    assert_clean(rendered(err.value), what="SourceExtractionError")


def test_the_status_frame_error_column_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frame is not only read in Mage — a pipeline can print it, write it
    onward or attach it to an alert, so this column travels further than a log
    line does."""
    from dbextractors import entrypoint
    from dbextractors.core import status

    error = RuntimeError(f"could not read from {CANARY_URL}")
    row = status.error_status("s.target", "db_a", error)

    assert_clean(row["error"], what="error_status")
    assert row["success"] is False, "redaction must not turn a failure green"

    # And the same through the real `run`, which is what fills the frame.
    with pytest.raises(entrypoint.SourceExtractionError):
        _run_failing(monkeypatch, error)


def test_the_password_alone_is_redacted_even_in_a_shape_no_pattern_knows(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`run` passes ``SOURCE_DB.password`` as ``extra``, so a driver that printed
    the password in some private format is covered too — that is the whole point
    of the ``extra`` list."""
    from dbextractors import entrypoint

    error = RuntimeError(f"login failed, credentials were {{'pass': '{CANARY}'}}")

    with (
        caplog.at_level(logging.DEBUG, logger="dbextractors.entrypoint"),
        pytest.raises(entrypoint.SourceExtractionError) as err,
    ):
        _run_failing(monkeypatch, error)

    assert_clean(caplog.text, what="the run log")
    assert_clean(rendered(err.value), what="SourceExtractionError")


# --- entrypoint._target_connection ------------------------------------------


def test_an_unusable_target_dsn_does_not_report_the_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triggered: psycopg2 answers a DSN it cannot parse by quoting the offending
    token back. With a password containing a space the DSN is invalid *because
    of* the password, and the token quoted back is a piece of it.

        ProgrammingError: invalid dsn: missing "=" after "CANARY" …
    """
    from dbextractors import entrypoint
    from dbextractors.core.target_conn import TargetConnectionError

    monkeypatch.setenv(
        "DBX_TARGET_DSN",
        f"host=127.0.0.1 port=5432 dbname=w user=svc password=pw {CANARY} tail",
    )

    with pytest.raises(TargetConnectionError) as err:
        entrypoint._target_connection()

    assert_clean(rendered(err.value), what="_target_connection")
    assert err.value.__cause__ is None, "a chained original would carry the DSN back into the dump"
    assert err.value.__suppress_context__, "`from None` also suppresses the implicit context"


def test_an_unreachable_target_does_not_report_the_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A syntactically fine DSN that simply does not answer. psycopg2 names the
    host and the port and nothing else — pinned so a driver change is caught."""
    from dbextractors import entrypoint
    from dbextractors.core.target_conn import TargetConnectionError

    monkeypatch.setenv(
        "DBX_TARGET_DSN", f"host=127.0.0.1 port=1 dbname=w user=svc password={CANARY}"
    )

    with pytest.raises(TargetConnectionError) as err:
        entrypoint._target_connection()

    assert_clean(rendered(err.value), what="_target_connection")
    assert "127.0.0.1" in str(err.value), "the host must survive — otherwise it is not debuggable"


# --- core/reading.py --------------------------------------------------------


class _FailingReader:
    """A source that always fails, with an error the caller supplies."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def iter_batches(self, engine: Any, sql: str, batch_size: int):
        raise self.error
        yield  # pragma: no cover - unreachable, makes this a generator


def _connection_error(text: str) -> Exception:
    from sqlalchemy import exc as sa_exc

    return sa_exc.OperationalError("SELECT 1", {}, Exception(text))


def test_a_read_that_cannot_be_resumed_does_not_log_the_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Triggered against MySQL, twice: a missing table (not retried) and a real
    ``KILL`` mid-read (2013 Lost connection). Neither driver message carried a
    credential — but the SQL statement is in there, so this is a text the package
    quotes wholesale and cannot vouch for."""
    from dbextractors.core import reading

    reader = _FailingReader(_connection_error(f"cannot reach {CANARY_URL}"))

    with (
        caplog.at_level(logging.DEBUG, logger="dbextractors.core.reading"),
        pytest.raises(Exception) as err,
    ):
        list(
            reading.read_with_resume(
                reader,
                None,
                100,
                build_sql=lambda key: "SELECT * FROM t",
                pk_in_batch=None,
                # `attempts=1`: nothing has been yielded, so this is now the
                # restart-from-scratch case (`core.reading`) rather than an
                # immediate refusal — the exhausted-attempts failure is what is
                # under test here, not the resume logic itself.
                attempts=1,
            )
        )

    assert "Reading the source failed" in caplog.text
    assert_clean(caplog.text, what="the reading log")
    # The exception itself is re-raised unchanged on purpose: the caller decides
    # on the type. `entrypoint.run` is what redacts it — the funnel test above.
    assert CANARY in str(err.value), "the re-raise is deliberately untouched"


def test_the_resume_warning_does_not_log_the_connection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The other log line in `read_with_resume` — the one printed between
    attempts, which in a long extraction is printed more often than the final
    one."""
    import pandas as pd

    from dbextractors.core import reading

    monkeypatch.setattr(reading.time, "sleep", lambda seconds: None)

    class _FailsOnce:
        def __init__(self) -> None:
            self.calls = 0

        def iter_batches(self, engine: Any, sql: str, batch_size: int):
            self.calls += 1
            yield pd.DataFrame({"id": [1, 2]})
            if self.calls == 1:
                raise _connection_error(f"cannot reach {CANARY_URL}")

    with caplog.at_level(logging.DEBUG, logger="dbextractors.core.reading"):
        list(
            reading.read_with_resume(
                _FailsOnce(),
                None,
                100,
                build_sql=lambda key: "SELECT * FROM t",
                pk_in_batch="id",
            )
        )

    assert "dropped" in caplog.text, "the retry was not exercised"
    assert_clean(caplog.text, what="the resume warning")


# --- core/retry.py ----------------------------------------------------------


def test_the_retry_wrapper_does_not_log_the_connection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`with_retry` wraps connection attempts, so ``err`` here is whatever the
    driver produced — the one text in that module that could carry a URL."""
    from dbextractors.core import retry

    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)

    def always_fails():
        raise RuntimeError(f"cannot reach {CANARY_URL}")

    with (
        caplog.at_level(logging.DEBUG, logger="dbextractors.core.retry"),
        pytest.raises(RuntimeError),
    ):
        retry.with_retry(always_fails, attempts=2, base_delay=0, desc="the connection")

    assert "failed after 2 attempts" in caplog.text
    assert "retrying in" in caplog.text, "both log lines have to be exercised"
    assert_clean(caplog.text, what="the retry log")


# --- core/logging.py --------------------------------------------------------


def test_a_malformed_format_string_does_not_repr_a_credential_into_the_log() -> None:
    """The one place in the package that ``repr()``s log arguments it never
    inspected, and it fires on a typo rather than on anything the author
    considered."""
    from dbextractors.core.logging import adapt

    class _DictLogger:
        """Mage's logger: keyword arguments only, no positional ones."""

        def __init__(self) -> None:
            self.lines: list = []

        def warning(self, message: str, **kwargs: Any) -> None:
            self.lines.append(message)

    logger = _DictLogger()
    # Two arguments for one placeholder — a TypeError, so the fallback branch runs.
    adapt(logger).warning("connecting to %s", CANARY_URL, CANARY_DSN)

    assert logger.lines, "the message was lost"
    assert_clean("\n".join(logger.lines), what="the logger adapter fallback")


# --- The drivers themselves -------------------------------------------------
#
# Safe by accident, not by design: none of the four talks about its own
# connection string on a failed connection. Checked live against a running
# instance of each. Pinned so an upgrade that changes it fails here.


@pytest.mark.parametrize(
    "dialect_name, params, port",
    [
        ("mysql", {"user": "svc", "database": "d"}, 3306),
        ("mssql", {"user": "svc", "database": "d"}, 1433),
        ("postgres", {"user": "svc", "database": "d"}, 5432),
        ("firebird", {"user": "svc", "database": "/var/db/x.fdb", "charset": "WIN1250"}, 3050),
    ],
)
def test_an_engine_never_renders_its_password(dialect_name: str, params: dict, port: int) -> None:
    """SQLAlchemy's ``Engine.__repr__`` hides the password; ``str(url)`` does
    **not**. No module in the package renders ``engine.url`` — this pins the half
    that is relied on.
    """
    import warnings

    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    from dbextractors.entrypoint import resolve_dialect

    dialect = resolve_dialect(dialect_name)
    url = dialect.build_conn_str({**params, "password": CANARY}, "db.example.com", port)
    with warnings.catch_warnings():
        # The SQLAlchemy 1.4 Firebird dialect warns that it is deprecated.
        warnings.simplefilter("ignore")
        engine = create_engine(url, poolclass=NullPool, connect_args=dict(dialect.connect_args))

    assert_clean(repr(engine), what=f"repr of the {dialect_name} engine")
    # The premise: the password really is in the URL, so the assertion above is
    # not passing for the wrong reason.
    assert CANARY in url


def test_psycopg2_does_not_name_the_password_on_a_failed_connection() -> None:
    """Triggered live against PostgreSQL 17: a wrong password, an unreachable
    host and an unknown connection option. None of the three messages quotes the
    password back — but the malformed-DSN message quotes *a token*, which is why
    `secrets.dsn_secrets` exists."""
    import psycopg2

    # Built outside the raising frame on purpose: a traceback prints the source
    # line of the frame it came from, and that line would contain the *name*
    # `CANARY`, failing the assertion for a reason that is not a leak.
    dsn = CANARY_DSN.replace("db.example.com", "127.0.0.1").replace("5432", "1")

    with pytest.raises(psycopg2.Error) as err:
        psycopg2.connect(dsn)

    assert_clean(rendered(err.value), what="psycopg2")
