"""Adapter for the logger handed in by Mage.

Mage does not pass a `logging.Logger` into the block; it passes its own
`DictLogger`, whose signature is ``warning(self, message, **kwargs)`` — **no
positional arguments**. The idiomatic lazy formatting that Python recommends for
logging::

    logger.warning("Full load by source, source=%s", label)

therefore ends in::

    TypeError: DictLogger.warning() takes 2 positional arguments but 3 were given

This is not a corner case. dbextractors logs this way in 48 places across nine
modules and **the first such call brings the whole run down**. Observed in
production: a pipeline died on a lazily formatted warning, and the error handler
that was supposed to report the failure died on the very same thing, so the
original cause was never printed at all.

Rewriting all 48 calls as f-strings would mean giving up lazy formatting
everywhere because of one peculiarity of one host. Instead the logger is wrapped
**once, at the entry point** — and the rest of the package keeps logging normally.

The price of formatting here is that the string is built even for a level that is
ultimately discarded. For dbextractors that does not matter: logging happens
roughly once per batch, not once per row.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = ["LoggerAdapter", "adapt"]

#: The levels anything in dbextractors ever calls.
_LEVELS = ("debug", "info", "warning", "error", "exception", "critical")


class LoggerAdapter:
    """Passes the message on only after the ``%s`` arguments have been substituted.

    Behaves like a logger: it has ``debug``/``info``/``warning``/``error``/
    ``exception``/``critical``. Keyword arguments are passed through unchanged —
    Mage uses them to carry things like ``error=`` into its JSON records.
    """

    __slots__ = ("_target",)

    def __init__(self, target: Any) -> None:
        self._target = target

    def _emit(self, level: str, message: Any, *args: Any, **kwargs: Any) -> None:
        text = message
        if args:
            try:
                text = str(message) % args
            except (TypeError, ValueError):
                # A malformed format string must not bring down a run that would
                # otherwise have finished. The message is emitted as it is, with the
                # arguments appended — a less readable log beats a lost extraction.
                #
                # This branch is the one place in the package that `repr()`s log
                # arguments it never inspected, and it fires on a typo rather than
                # on anything the author considered. Whatever those arguments were
                # meant to be formatted into, they go through the redaction first.
                from dbextractors.core import secrets

                text = secrets.redact(f"{message} {args!r}")
        method = getattr(self._target, level, None) or self._target.info
        method(text, **kwargs)

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("debug", message, *args, **kwargs)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("info", message, *args, **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("warning", message, *args, **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("error", message, *args, **kwargs)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("critical", message, *args, **kwargs)

    def exception(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._emit("exception", message, *args, **kwargs)


def adapt(logger: Optional[Any]) -> Optional[Any]:
    """Wraps the logger if it needs wrapping. ``None`` stays ``None``.

    A standard `logging.Logger` needs no wrapper — it handles positional arguments
    itself, and lazy formatting is a genuine benefit there that there is no reason
    to give up. The wrapper goes on everything else, because "everything else" is
    exactly what behaves differently in production.
    """
    if logger is None or isinstance(logger, (logging.Logger, LoggerAdapter)):
        return logger
    return LoggerAdapter(logger)
