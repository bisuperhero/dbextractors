"""Retrying operations and waiting for a port.

Both functions were merged from 15 predecessor extractors, which between them held
five variants. The winning variant of both comes from **variant A**'s MySQL
streaming extractor — the same predecessor `core/tunnel.py` follows. The variants
differed only in docstrings and comments; the bodies were functionally identical in
every file that had the function at all (an AST diff told them apart only because
docstrings are AST nodes too).

The signatures are frozen — they were identical across all 15 files — which is why
there is no parameter for the Mage logger here (unlike `core/tunnel.py`, where
`ensure_private_key_permissions`/`normalize_private_key_contents` do not need a
logger but `show_debug` does). Logging goes through the standard
``logging.getLogger(__name__)``; routing it into the Mage logger is a question for
`entrypoint.run()`, not for this module.
"""

from __future__ import annotations

import logging
import random
import socket
import time
from typing import Any, Callable

from dbextractors.core import secrets

_log = logging.getLogger(__name__)


def with_retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.25,
    desc: str = "operation",
) -> Any:
    """Retries ``fn`` with exponential backoff and jitter.

    The signature is taken over unchanged — it was identical across all 15
    predecessor files.

    On the last attempt the exception from ``fn()`` is let through (a bare
    ``raise``, type unchanged); on every other attempt it is logged and the code
    waits ``base_delay * 2**(attempt-1)`` (capped at ``max_delay``), spread randomly
    both ways by a ``jitter`` fraction.

    A project rule applies here: no ``except Exception: pass`` — whenever an
    exception is swallowed it has to appear in the log. Every failed attempt is
    therefore logged, not silently dropped.

    Edge case preserved from the predecessor: with ``attempts <= 0`` the loop body
    never runs and the function returns ``None`` (``fn()`` is never called). That is
    not an explicit ``return`` here or in the old version — just identical behaviour,
    pinned by a characterisation test.
    """
    for attempt in range(1, attempts + 1):
        try:
            _log.warning("🔄 Attempt %d/%d for %s...", attempt, attempts, desc)
            return fn()
        except Exception as err:
            # `fn` is typically a connection attempt, so `err` is whatever the
            # driver produced — the one text in this module that could carry a URL
            # or a DSN. The exception itself is re-raised untouched (the caller
            # needs its type); only what goes into the log is redacted.
            if attempt == attempts:
                _log.error(
                    "🛑 %s failed after %d attempts: %s", desc, attempts, secrets.redact(err)
                )
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 1 + jitter * (random.random() * 2 - 1)
            _log.warning(
                "⚠️ %s failed (attempt %d/%d): %s -> retrying in %.1fs",
                desc,
                attempt,
                attempts,
                secrets.redact(err),
                delay,
            )
            time.sleep(delay)
    return None


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 10) -> bool:
    """Waits until something answers on the port. Used after starting the SSH tunnel.

    Tries a TCP connection every 0.2 s until it succeeds or ``timeout`` runs out.

    Against the predecessor the ``except`` is narrowed from ``Exception`` to
    ``OSError`` — which is exactly what ``socket.create_connection`` raises for an
    unreachable target (``ConnectionRefusedError``, ``TimeoutError``,
    ``socket.gaierror`` — all subclasses of ``OSError``). Behaviour for those cases
    is identical to the old version; the only difference is that a genuine
    programming error (a wrong argument type, say) now propagates out instead of
    disappearing into a silent ``except Exception``.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError as err:
            _log.debug("🔌 Port %s:%s not reachable yet: %s", host, port, err)
            time.sleep(0.2)
    return False
