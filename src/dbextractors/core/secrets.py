"""Redaction of credentials from text that may end up in a log.

The package builds a password into a string in seven places — four dialect
``build_conn_str`` functions (SQLAlchemy URLs) and three paths to the target
(libpq DSNs):

    dialects/mysql.py, mssql.py, postgres.py, firebird.py   → ``scheme://user:pw@host``
    core/target_conn.py, golden/session.py                  → ``… password=pw``

Until this module existed there was a single redaction (`target_conn.describe_dsn`)
and it protected a single place — the "🎯 Target: …" message. Everywhere else it
was enough for the string to reach the text of an exception and the password was
in the log. Typically via ``create_engine``: when a URL is unusable for the
dialect, SQLAlchemy prints the whole of it, password included.

This module is the one place such text passes through. It is not a substitute for
keeping passwords out of strings in the first place — it is the safety net for the
moment when they are already in one and it is heading for the log.
"""

from __future__ import annotations

import re
from typing import Iterable, List

__all__ = ["MASK", "dsn_secrets", "redact", "redact_url", "redact_dsn", "redact_private_key"]

#: What the password is replaced with. Deliberately **not** an empty string — the
#: log should show that a password was there and was masked, not that there was none.
MASK = "***"

#: ``scheme://user:password@host`` — a SQLAlchemy URL.
#:
#: The password is URL-encoded (`urllib.parse.quote_plus` in all four dialects), so
#: it contains neither ``@`` nor ``/`` and can be delimited reliably. The user name
#: is **kept** — it is useful in the log and it is not a secret.
_URL_CREDENTIALS = re.compile(
    r"(?P<prefix>[A-Za-z][A-Za-z0-9+.\-]*://[^:/?#\s@]*):(?P<password>[^@\s/]*)@"
)

#: ``password=…`` in a libpq DSN, including ``passwd``/``pwd`` and quoted variants.
#: The value ends at a space or at the end of the string.
#: Two shapes the audit caught and this pattern now handles:
#:
#: - ``password=`` with an **empty** value used to have its ``\s*`` swallow the
#:   separating space and its ``\S+`` then mask the *following* token
#:   (``password= user=u`` redacted ``user=u``). The lookahead refuses a value
#:   that itself looks like the next ``key=`` pair, so an empty password now
#:   masks nothing — there is nothing to hide.
#: - a quoted value with an escaped quote (``password='ab\'cd'``) used to stop
#:   at the escape and leak the tail; the quoted alternatives now step over
#:   backslash escapes.
_KEYWORD_PASSWORD = re.compile(
    r"(?P<key>\b(?:password|passwd|pwd)\s*=\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?!\w+=)\S+)",
    re.IGNORECASE,
)

#: A PEM private key block, in every spelling ``ssh`` accepts: ``OPENSSH``,
#: ``RSA``, ``DSA``, ``EC``, ``ENCRYPTED`` or none at all.
#:
#: ``SOURCE_DB.ssh_pkey`` is contracted to be a **path**, and with a path there is
#: nothing here to match. The pattern exists for the case where somebody puts the
#: key *itself* in that key — pasting a key into a YAML string is a mistake made
#: often enough that `normalize_private_key_contents` exists to repair its milder
#: form. When it happens, `core/tunnel.py` hands that value to ``os.stat``, to
#: ``ssh -i`` and to an error message, and the private key ends up in the log
#: verbatim. Of everything this module masks it is the only secret that a changed
#: database password does not retire.
#:
#: The second pattern catches a block whose ``-----END-----`` never arrived —
#: truncated output, a key with CRLF endings, a message cut off at a line limit.
#: Everything after an unterminated ``BEGIN`` is key material, so it all goes.
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
    r".*?"
    r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)
_PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*", re.DOTALL)

#: The password **as written** in a libpq DSN — see `dsn_secrets`. Unlike
#: `_KEYWORD_PASSWORD` this one does not stop at whitespace: it runs to the next
#: ``key=`` pair or to the end of the string, so a value containing spaces comes
#: out whole.
_DSN_PASSWORD_VALUE = re.compile(
    r"\b(?:password|passwd|pwd)\s*=\s*"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|.*?(?=\s+\w+\s*=|$))",
    re.IGNORECASE | re.DOTALL,
)

#: Values shorter than this are not masked from `extra` — see `redact`.
_MIN_LITERAL_LENGTH = 4


def redact_url(text: str) -> str:
    """Masks the password in a SQLAlchemy URL."""
    return _URL_CREDENTIALS.sub(rf"\g<prefix>:{MASK}@", text)


def redact_dsn(text: str) -> str:
    """Masks ``password=…`` in a libpq DSN."""
    return _KEYWORD_PASSWORD.sub(rf"\g<key>{MASK}", text)


def redact_private_key(text: str) -> str:
    """Masks any PEM private key block — see `_PRIVATE_KEY_BLOCK`."""
    return _PRIVATE_KEY_HEADER.sub(MASK, _PRIVATE_KEY_BLOCK.sub(MASK, text))


def dsn_secrets(dsn: str) -> List[str]:
    """Literals from a libpq DSN's password, to hand to ``redact(extra=…)``.

    `redact_dsn` masks ``password=…`` **inside a DSN**. It cannot help when a
    *fragment* of the password comes back on its own, with no ``password=`` in
    front of it — and that is exactly what psycopg2 does with a DSN it considers
    malformed. Triggered against psycopg2 2.9.12::

        psycopg2.connect("host=h user=u password=pw CANARY 8213")
        ProgrammingError: invalid dsn: missing "=" after "CANARY" …

    A password containing a space makes the DSN invalid *because of the
    password*, and the token quoted back is a piece of it. So the value is
    returned whole **and** split on whitespace: the whole value covers the
    ordinary case, the pieces cover this one.

    The underlying defect — that the DSN builders do not quote the password, so
    such a password cannot connect at all — is not fixed here. That is a change
    of behaviour, not of what reaches the log.
    """
    found: List[str] = []
    for match in _DSN_PASSWORD_VALUE.finditer(dsn or ""):
        value = match.group("value").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            continue
        found.append(value)
        found.extend(value.split())
    return found


def redact(text: object, extra: Iterable[object] = ()) -> str:
    """Text with credentials removed — for the log and for exception messages.

    It handles all three forms at once, because the caller often does not know
    what is in the text: an exception message can carry a URL and a DSN at the
    same time (the source connection and the target connection in one sentence),
    and a tunnel failure carries an ``ssh`` command line on top of that.

    ``extra`` holds concrete values the caller **knows** to be secret — typically
    the password from the configuration. They are masked literally, so even shapes
    none of the patterns recognise get through (a password on a command line, in
    JSON, in a driver message). Values shorter than four characters are skipped: a
    password of ``a`` would turn the message into an unreadable mess of asterisks
    and would hide exactly the part someone is reading the log for.
    """
    result = redact_private_key(redact_dsn(redact_url(str(text))))

    for value in extra:
        literal = str(value or "")
        if len(literal) >= _MIN_LITERAL_LENGTH:
            result = result.replace(literal, MASK)

    return result
