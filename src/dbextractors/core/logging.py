"""How dbextractors logs, and how that reaches the host's logger.

## Levels

Every module logs to its own standard logger under ``dbextractors``
(``logging.getLogger(__name__)``), at the level the message deserves:

- ``DEBUG`` — detail for debugging: generated SQL, hash expressions, individual
  retries of a port probe, column renames.
- ``INFO`` — the normal course of a run: which strategy, the window or
  watermark, row counts, batch progress, the swap, the tunnel up and down, done.
- ``WARNING`` — the run goes on, but something is not as it should be and a
  person should know: a fallback to a full load, dropped columns, an inert or
  unknown key, a retry, the emergency write path, a cleanup that failed.
- ``ERROR`` — a failure, logged just before the exception is raised.

The ``DEBUG`` configuration key still switches on extra output that costs
something to produce (type maps, the ssh command line, a count of the live-key
snapshot); what it switches on is logged at ``DEBUG``.

Under Mage (the 1.0.x line) nearly everything was logged as a warning, because
that was what reached the pipeline log. 2.x does not run under Mage, so the level
says what the message is, not where it has to get to.

## Reaching the host

The package configures no handlers — only a `logging.NullHandler` on the
``dbextractors`` logger, as a library should. A host that captures standard
loggers (Dagster's ``python_logs.managed_python_loggers``, a plain
``logging.basicConfig``) sees everything under ``dbextractors``.

A host that hands a logger to ``run(logger=...)`` instead — Dagster's
``context.log`` or ``get_dagster_logger()`` — gets the records forwarded to it
for the duration of that run (`forward_to`). A forwarded record does not also
propagate further up, so a host that does both does not see every line twice.

## Several runs in one process

A Dagster in-process executor can run two assets in two threads of one process,
each with its own ``context.log``. The ``dbextractors`` logger is process-wide,
so forwarding is organised around that: one shared handler, installed while at
least one run forwards, and a registry of targets keyed by thread, all changed
under a lock. A record goes to the target of the thread that logged it. A record
from a thread that forwards nothing is propagated exactly as it would be without
any forwarding — and only if it passes the level the package logger had before,
because the level is lowered for the forwarding runs' sake, not for theirs.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Dict, Iterator, List, Tuple

__all__ = ["PACKAGE_LOGGER", "forward_to"]

#: The root of the package's logger hierarchy.
PACKAGE_LOGGER = "dbextractors"

_lock = threading.Lock()
#: Thread id -> the stack of (target, level) that thread forwards to. A stack,
#: because a run may call `run` again inside itself; the innermost target wins.
_targets: Dict[int, List[Tuple[Any, int]]] = {}
#: The package logger's own level and propagation from before the first
#: forwarding began; restored when the last one ends.
_saved: Tuple[int, bool] | None = None


class _Dispatcher(logging.Handler):
    """The one handler that forwarding installs on the ``dbextractors`` logger."""

    def __init__(self) -> None:
        super().__init__(logging.NOTSET)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        # `thread` is None when the host switched `logging.logThreads` off.
        stack = _targets.get(record.thread) if record.thread is not None else None
        if not stack:
            _propagate_as_before(record)
            return
        target, level = stack[-1]
        if record.levelno < level:
            return
        try:
            text = self.format(record)
        except (TypeError, ValueError):
            # A malformed format string must not bring down a run that would
            # otherwise have finished. The arguments are appended instead — and
            # redacted first: this is the one place that `repr()`s log arguments
            # nobody inspected, and it fires on a typo.
            from dbextractors.core import secrets

            text = secrets.redact(f"{record.msg} {record.args!r}")
        method = getattr(target, record.levelname.lower(), None) or target.info
        method(text)


_dispatcher = _Dispatcher()


def _propagate_as_before(record: logging.LogRecord) -> None:
    """Pass on a record from a thread that forwards nothing, as if nothing forwarded.

    Propagation is off while any run forwards, so this does its job: the
    record goes to the parent's handlers, provided it clears the level the
    package logger would have had.
    """
    saved = _saved
    package = logging.getLogger(PACKAGE_LOGGER)
    if saved is None or not saved[1] or package.parent is None:
        return
    level = saved[0] or package.parent.getEffectiveLevel()
    if record.levelno >= level:
        package.parent.handle(record)


def _is_own(target: Any) -> bool:
    """``True`` for a logger inside the package's own hierarchy.

    Forwarding the package to one of its own loggers would send every record
    round in a loop.
    """
    name = getattr(target, "name", None)
    return isinstance(target, logging.Logger) and (
        name == PACKAGE_LOGGER or str(name).startswith(PACKAGE_LOGGER + ".")
    )


def _apply_level(package: logging.Logger) -> None:
    """The lowest level any forwarding run needs. Called under the lock."""
    package.setLevel(min(level for stack in _targets.values() for _, level in stack))


@contextlib.contextmanager
def forward_to(target: Any | None) -> Iterator[None]:
    """Forward the package's records from this thread to ``target`` until the block ends.

    The package logger's level is lowered to the target's own effective level
    for the duration: otherwise it would inherit the root logger's ``WARNING``
    and every ``INFO`` record would be dropped before any handler saw it. What
    the target then shows is up to the target. A target that is not a
    `logging.Logger` gets ``INFO`` and above.

    ``None`` — and a logger of the package's own — forward nothing.
    """
    global _saved
    if target is None or _is_own(target):
        yield
        return

    package = logging.getLogger(PACKAGE_LOGGER)
    thread = threading.get_ident()
    level = target.getEffectiveLevel() if isinstance(target, logging.Logger) else logging.INFO
    with _lock:
        if _saved is None:
            _saved = (package.level, package.propagate)
            package.addHandler(_dispatcher)
            package.propagate = False
        _targets.setdefault(thread, []).append((target, level))
        _apply_level(package)
    try:
        yield
    finally:
        with _lock:
            stack = _targets[thread]
            stack.pop()
            if not stack:
                del _targets[thread]
            if _targets:
                _apply_level(package)
            elif _saved is not None:
                package.removeHandler(_dispatcher)
                package.setLevel(_saved[0])
                package.propagate = _saved[1]
                _saved = None
