"""Tests for `dbextractors.core.secrets` — redaction of credentials.

Two parts. The first checks the redaction itself over hand-made strings. The
second matters more: it takes the **real** ``build_conn_str`` of all four
dialects, has each of them build a URL containing a password, and checks that
after redaction the password is nowhere to be found.

Why do it that way: the first part would pass even if one dialect assembled its
URL differently from what the pattern in `core/secrets.py` expects. The second
part depends on the shape of the URL on purpose — when somebody changes
`build_conn_str`, it fails here rather than in a production log.

Nothing here touches the network — `build_conn_str` is a pure function over a
dict.
"""

from __future__ import annotations

import pytest

from dbextractors.core import secrets

# A password with everything that typically breaks a URL or a DSN: an at sign,
# a slash, a space, a quote, a percent sign.
PASSWORD = "a@b/c d'e%f"


# --- redact_url --------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "mysql+mysqlconnector://svc:secretpass@db.example.com:3306/prod",
        "mssql+pymssql://svc:secretpass@db.example.com:1433/prod",
        "postgresql+psycopg2://svc:secretpass@db.example.com:5432/prod",
        "firebird+fdb://svc:secretpass@db.example.com:3050//var/db/x.fdb",
    ],
)
def test_redact_url_hides_the_password_and_keeps_the_rest(url: str) -> None:
    result = secrets.redact_url(url)

    assert "secretpass" not in result
    assert secrets.MASK in result
    # The user, host, port and database stay in the log — without them
    # "it did not connect" cannot be debugged.
    assert "svc" in result
    assert "db.example.com" in result


def test_redact_url_leaves_a_url_without_a_password_alone() -> None:
    url = "postgresql+psycopg2://db.example.com:5432/prod"
    assert secrets.redact_url(url) == url


def test_redact_url_copes_with_several_urls_in_one_text() -> None:
    text = (
        "source mysql+mysqlconnector://a:pass1@h1:3306/d "
        "target postgresql+psycopg2://b:pass2@h2:5432/d"
    )
    result = secrets.redact_url(text)

    assert "pass1" not in result and "pass2" not in result


# --- redact_dsn --------------------------------------------------------------


@pytest.mark.parametrize(
    "dsn",
    [
        "host=h port=5432 dbname=d user=u password=secretpass",
        "password=secretpass host=h",
        "host=h password = secretpass",
        "host=h PASSWORD=secretpass",
        "host=h passwd=secretpass",
        "host=h pwd=secretpass",
        "host=h password='secretpass'",
        'host=h password="secretpass"',
    ],
)
def test_redact_dsn_hides_the_password(dsn: str) -> None:
    result = secrets.redact_dsn(dsn)

    assert "secretpass" not in result
    assert secrets.MASK in result
    assert "host=h" in result


# --- The two shapes the audit caught -----------------------------------------
#
# Both were found by reading the pattern rather than by a failing test, which is
# why they are pinned here: a regression in either is invisible until a log is
# read, and by then the damage is done. They fail in opposite directions — the
# first hides too much, the second too little — and the same edit can bring
# either one back.


def test_an_empty_password_does_not_mask_the_following_token() -> None:
    """``password=`` with nothing after it used to eat the next key=value pair.

    ``\\s*`` swallowed the separating space and ``\\S+`` then matched ``user=svc``,
    so the log lost the user name and claimed a password had been masked where
    there was none. Nothing to hide means nothing to mask.
    """
    result = secrets.redact_dsn("host=h password= user=svc dbname=d")

    assert "user=svc" in result
    assert "dbname=d" in result


def test_a_quoted_password_with_an_escaped_quote_does_not_leak_its_tail() -> None:
    """``password='se\\'cret'`` used to stop at the escape and leave the rest in.

    The quoted alternatives now step over backslash escapes, so the whole value
    goes — and the pair after it survives, which is how one tells a fixed
    pattern from one that simply masks everything to the end of the line.
    """
    result = secrets.redact_dsn(r"host=h password='se\'cret' user=svc")

    assert "cret" not in result, result
    assert "host=h" in result and "user=svc" in result


# --- redact ------------------------------------------------------------------


def test_redact_copes_with_a_url_and_a_dsn_in_one_message() -> None:
    """An exception message often carries both — the source and the target
    connection."""
    text = (
        "could not write from mysql+mysqlconnector://u:sourcepass@h:3306/d "
        "into host=target dbname=w password=targetpass"
    )
    result = secrets.redact(text)

    assert "sourcepass" not in result
    assert "targetpass" not in result


def test_redact_accepts_an_exception_too_not_just_a_string() -> None:
    err = ValueError("bad url: postgresql+psycopg2://u:secretpass@h:5432/d")

    assert "secretpass" not in secrets.redact(err)


def test_redact_extra_hides_the_value_in_any_shape() -> None:
    """`extra` is the safety net for shapes no pattern knows."""
    text = "the driver reports: login failed for {'pass': 'secretpass'}"

    assert "secretpass" not in secrets.redact(text, extra=["secretpass"])


def test_redact_extra_skips_short_values() -> None:
    """A password of ``ab`` would turn the message into a mess of asterisks and
    hide exactly the part somebody is reading the log for."""
    text = "table abc has a bad column"

    assert secrets.redact(text, extra=["ab"]) == text


def test_redact_extra_ignores_an_empty_password() -> None:
    text = "nothing secret here"

    assert secrets.redact(text, extra=[None, ""]) == text


# --- Against the real dialects -----------------------------------------------
#
# Errors must not be swallowed. This is the other side of that rule: an error
# has to be reported **in full, but without the password**.


def _dialects() -> list:
    from dbextractors.dialects.firebird import FirebirdDialect
    from dbextractors.dialects.mssql import MSSQLDialect
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    return [MySQLDialect(), MSSQLDialect(), PostgresDialect(), FirebirdDialect()]


@pytest.mark.parametrize("dialect", _dialects(), ids=lambda d: type(d).__name__)
def test_the_password_disappears_from_every_dialects_url(dialect) -> None:
    conn_str = dialect.build_conn_str(
        {"user": "svc", "password": PASSWORD, "database": "prod", "charset": "utf8mb4"},
        "db.example.com",
        1234,
    )

    # Check the premise: the password really is in the URL (URL-encoded or
    # not), otherwise the test would pass for the wrong reason.
    assert conn_str != secrets.redact(conn_str), (
        f"{type(dialect).__name__}.build_conn_str assembles the URL in a shape "
        f"the redaction does not recognise: {conn_str!r}"
    )

    result = secrets.redact(conn_str, extra=[PASSWORD])

    assert PASSWORD not in result
    # The URL-encoded form must not get through either — `quote_plus` turns the
    # password into `a%40b%2Fc+d%27e%25f`, which is still the password.
    import urllib.parse

    assert urllib.parse.quote_plus(PASSWORD) not in result
    assert "db.example.com" in result
