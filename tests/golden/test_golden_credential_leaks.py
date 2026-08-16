"""Credential leaks in the golden harness — see `tests/core/test_credential_leaks.py`.

The harness deserves its own file because its output is read differently from a
Mage log: it is run by hand, on a terminal, and what it prints gets pasted into a
ticket. The ``--json`` report goes further still — it is a file that outlives the
run and is committed or attached.

The same two categories apply. `test_a_dsn_that_cannot_be_parsed…` pins a leak
that was triggered and fixed; the comparison-report test pins a path that carries
nothing today but writes to disk, which is the wrong place to discover otherwise.
"""

from __future__ import annotations

import traceback
from typing import Any

import pytest

from dbextractors.golden import session

CANARY = "pw-CANARY-8213"

CANARY_URL = f"postgresql+psycopg2://svc:{CANARY}@db.example.com:5432/prod"


def rendered(err: BaseException) -> str:
    """Message, traceback and the whole ``__cause__`` chain as one string."""
    return "".join(traceback.format_exception(type(err), err, err.__traceback__))


def assert_clean(text: str, *, what: str) -> None:
    assert CANARY not in text, f"{what} leaked the password: {text!r}"
    assert "CANARY" not in text, f"{what} leaked a fragment of the password: {text!r}"


# --- golden/session.connect -------------------------------------------------


def test_a_dsn_that_cannot_be_parsed_does_not_report_the_password() -> None:
    """Triggered: psycopg2 answers a DSN it considers malformed by quoting the
    offending token back::

        ProgrammingError: invalid dsn: missing "=" after "CANARY" …

    A password containing a space makes the DSN invalid *because of* the
    password, so that token is a piece of it. Before the fix the exception was
    not caught at all and the interpreter printed the whole traceback to the
    terminal this tool is read on.
    """
    # Built outside the raising frame: a traceback prints the source line of the
    # frame it came from, and that line would contain the *name* `CANARY`.
    dsn = f"host=127.0.0.1 port=5432 dbname=w user=svc password=pw {CANARY} tail"

    with pytest.raises(session.SessionError) as err, session.connect(dsn):
        pass  # pragma: no cover - the connection never opens

    assert_clean(rendered(err.value), what="session.connect")
    assert err.value.__cause__ is None, "a chained original would carry the DSN back into the dump"
    assert err.value.__suppress_context__, "`from None` also suppresses the implicit context"


def test_an_unreachable_host_is_reported_with_host_and_port_but_no_password() -> None:
    """`resolve_dsn` fails before psycopg2 is reached. Safe already — pinned so
    that the host and port, which are what makes it debuggable, keep surviving."""
    dsn = f"host=127.0.0.1 port=1 dbname=w user=svc password={CANARY}"

    with pytest.raises(session.SessionError) as err:
        session.resolve_dsn(dsn, allow_wsl_fallback=False)

    assert_clean(rendered(err.value), what="resolve_dsn")
    assert "127.0.0.1" in str(err.value) and "1" in str(err.value)


# --- golden/compare: the report that goes to disk ---------------------------


@pytest.mark.needs_pg
def test_a_failed_comparison_does_not_write_a_credential_into_the_report(
    ro_conn: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """`compare_tables` catches everything into ``report.error``.

    That string is printed by ``render_table`` and written verbatim into the
    ``--json`` file. The connection is already open by the time anything here can
    fail, which is why nothing observed carries a DSN — but a report on disk is
    the wrong place to find out that something does. A level function is made to
    fail with a URL-bearing error so that the redaction is what the assertion
    depends on.
    """
    from dbextractors.golden import compare as compare_mod
    from dbextractors.golden import report as report_mod
    from dbextractors.golden.compare import CompareOptions, compare_tables
    from dbextractors.golden.model import VERDICT_ERROR, Relation

    def _explode(ctx: Any) -> Any:
        raise RuntimeError(f"the comparison lost its connection {CANARY_URL}")

    monkeypatch.setattr(compare_mod, "_level_column_names", _explode)

    report = compare_tables(
        ro_conn,
        Relation.parse("pg_catalog.pg_class"),
        Relation.parse("pg_catalog.pg_class"),
        CompareOptions(key_columns=["oid"], skip_row_level=True),
    )

    assert report.verdict == VERDICT_ERROR, "the failure was swallowed instead of reported"
    assert_clean(report.error or "", what="report.error")

    # And the same through the two ways that string leaves the process.
    assert_clean(report_mod.render_table(report, verbose=True), what="the rendered report")
    path = tmp_path / "report.json"
    report_mod.to_json(report, str(path))
    assert_clean(path.read_text(encoding="utf-8"), what="the JSON report on disk")
