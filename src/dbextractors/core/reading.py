"""Reading from the source in a way that survives a dropped connection.

This module came out of two failed long extractions in production::

    🛑 [<table>] extraction failed: (mysql.connector.errors.OperationalError)
    2013 (HY000): Lost connection to MySQL server during query

It was not a silent failure — it was a **missing retry**. The predecessor wrapped
every batch read in ``with_retry``, so a dropped connection was retried and usually
went through. dbextractors has the same function (`core/retry.py`) but nothing was
calling it.

On top of that, the shape of the reading changed. The predecessor paginated with
``LIMIT/OFFSET``, i.e. many short independent queries — each of which could be
retried. dbextractors holds a single cursor over a single query for the whole table
(faster, no quadratic work on the source), but then there is no such thing as
"batch 68" on the server side: a continuous stream is sent and `pandas` merely
slices it. There is nothing to retry, and one query stays open for tens of minutes,
which is exactly what ``net_write_timeout`` kills.

The fix is not to go back to pagination but to **finish the rest**: normally the
fast stream runs, and when the connection drops, reading continues by keyset from
the last key that got through. Nothing is downloaded twice.

**The precondition without which this does not work: the read must be ordered by
that key.** In an unordered stream, having seen ``max(id) = X`` in the batches
received says nothing about whether the server would still have sent rows with a
lower ``id`` — and continuing from ``id > X`` would drop them. The run would finish
green, which is the worst possible way to lose data. That is why ``ORDER BY pk`` is
added to the very first query, and why without a usable key nothing is **resumed at
all**.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Iterator, Optional

import pandas as pd

from dbextractors.core import secrets

_log = logging.getLogger(__name__)

#: How many times reading is resumed before giving up. Same as the predecessor.
DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_JITTER = 0.25


def is_temporary_error(err: BaseException) -> bool:
    """``True`` for connection errors that are worth retrying.

    The predecessor retried **anything** (``except Exception``). Here that is
    narrowed: retrying a SQL error three times only means failing six seconds later
    with three confusing lines in the log. The distinction comes from SQLAlchemy's
    taxonomy, not from the error text — ``OperationalError``/``InterfaceError``/
    ``InternalError`` are exactly the ones where the connection or the server is at
    fault, whereas ``ProgrammingError``, ``DataError`` and ``IntegrityError`` are not
    fixed by retrying.

    When SQLAlchemy is unavailable (it should not happen, but the module must not
    depend on it hard), ``False`` is returned — better not to retry than to retry
    something that must not be retried.
    """
    try:
        from sqlalchemy import exc as sa_exc
    except ImportError:  # pragma: no cover
        return False

    if isinstance(err, sa_exc.DBAPIError) and getattr(err, "connection_invalidated", False):
        return True
    return isinstance(err, (sa_exc.OperationalError, sa_exc.InterfaceError, sa_exc.InternalError))


def _backoff_delay(attempt: int, base_delay: float, max_delay: float, jitter: float) -> float:
    """Exponential backoff with jitter — the same computation as `core.retry`."""
    sleep = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return sleep * (1 + jitter * (random.random() * 2 - 1))


def read_with_resume(
    dialect: Any,
    engine: Any,
    batch_size: int,
    *,
    build_sql: Callable[[Optional[Any]], str],
    pk_in_batch: Optional[str],
    log: Optional[Callable[..., None]] = None,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
) -> Iterator[pd.DataFrame]:
    """Reads as a stream and, after a dropped connection, finishes from the last key.

    Args:
        build_sql: Builds the SELECT. It gets ``None`` for the first pass and the
            value of the last key seen when resuming; in the second case it must
            return a query restricted to ``pk > value``. Ordering by the key is the
            caller's responsibility — without it resuming is incorrect (see the
            module header).
        pk_in_batch: Name of the key column **in the batch**, i.e. the source name.
            ``None`` means "resuming is impossible"; a dropped connection then
            propagates to the caller.

    Yielded batches are **never repeated**: after a drop, reading continues past the
    last key the caller already received. The caller therefore need not deal with
    duplicates.
    """
    last_key: Optional[Any] = None
    saw_row = False

    for attempt in range(1, int(attempts) + 1):
        sql = build_sql(last_key if saw_row else None)
        try:
            for batch in dialect.iter_batches(engine, sql, batch_size):
                if pk_in_batch and len(batch) and pk_in_batch in batch.columns:
                    last_key = batch[pk_in_batch].max()
                    saw_row = True
                yield batch
            return
        except Exception as err:
            reason = _why_cannot_resume(err, pk_in_batch, saw_row, attempt, attempts)
            if reason:
                _log.error("🛑 Reading the source failed (%s): %s", reason, secrets.redact(err))
                # Re-raised unchanged: the caller decides on the **type**
                # (`is_temporary_error` reads SQLAlchemy's taxonomy) and rewrapping
                # would take that away. The unredacted text is caught one level up,
                # in `entrypoint.run`, which redacts the message and the traceback
                # before either reaches a log — see the funnel there.
                raise

            wait = _backoff_delay(attempt, base_delay, max_delay, jitter)
            message = (
                "⚠️ The connection to the source dropped (%s). Resuming from %s = %s "
                "in %.1f s (attempt %d/%d)."
            )
            args = (secrets.redact(err), pk_in_batch, last_key, wait, attempt + 1, attempts)
            if log:
                log("warning", message, *args)
            else:
                _log.warning(message, *args)
            time.sleep(wait)


def _why_cannot_resume(
    err: BaseException,
    pk_in_batch: Optional[str],
    saw_row: bool,
    attempt: int,
    attempts: int,
) -> Optional[str]:
    """The reason not to continue, or ``None`` when resuming is possible.

    Returns text for the log, not a bool — when a run fails, what matters is **why**
    it was not retried. "It failed" without a reason is exactly the message that
    costs half an hour of looking for the cause.
    """
    if not is_temporary_error(err):
        return "not a connection error, retrying makes no sense"
    if attempt >= attempts:
        return f"not even after {attempts} attempts"
    if not pk_in_batch:
        return "the source has no usable key to continue from"
    if not saw_row:
        return "the connection dropped before the first row arrived"
    return None


__all__ = ["read_with_resume", "is_temporary_error"]
