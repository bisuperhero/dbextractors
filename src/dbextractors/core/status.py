"""The DataFrame returned to Mage, plus the running batch log.

``load_data`` has to return a DataFrame; both the Mage block and its
``test_output`` rely on that. With multi-source loads (today only MSSQL) there are
several rows — which is why MSSQL was the only predecessor with the signature
``_build_status_df(statuses)`` rather than ``(status)``. The unified signature is
the plural one; the single-source case is a list with one element.

## Two functions from the predecessors were deliberately not carried over

- ``record_state`` — the old ``_record_state`` wrote into a per-client state table
  through a helper imported from the client repository, i.e. into a table the
  package knows nothing about. Its role — "what last succeeded for this source" —
  is filled here by `target_pg.write_fingerprint`, which additionally sits in the
  same transaction as the data. A deployment that still needs its own such table
  can fill it from the status frame `run` returns.
- ``log`` — logging that does not fall over without a logger is done by
  `LoadContext.log`. That is where it belongs: it has the logger at hand and does
  not need it passed in as a parameter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Columns of the status frame **and their order**. ``test_output`` in the client
#: repositories relies on these names (it reads ``success`` and ``source``), so
#: columns may only be added, never renamed.
STATUS_COLUMNS: tuple = (
    "table",
    "source",
    "rows_written",
    "load_method",
    "success",
    "is_incremental",
    "data_present",
    "fallback_reason",
    "connection_mode",
    "error",
)


def build_status_df(statuses: Sequence[dict]) -> pd.DataFrame:
    """Builds the returned DataFrame out of the per-source statuses.

    A missing key is filled in as ``None``, so that the frame always has the same
    columns in the same order — one run must not return a different shape from
    another just because there was nothing to fill in.
    """
    import pandas as pd

    if not statuses:
        raise ValueError("build_status_df was given an empty list of statuses.")

    return pd.DataFrame(
        {column: [state.get(column) for state in statuses] for column in STATUS_COLUMNS}
    )


def fmt_duration(seconds: float) -> str:
    """A duration in readable form for the log. Literally the old ``_fmt_duration``.

    Three bands (``s`` / ``m s`` / ``h m s``) — in a batch log the point is to see at
    a glance whether the run takes minutes or hours.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m {secs}s"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes}m {secs}s"


@dataclass
class BatchProgress:
    """Running batch log with throughput and an estimate of the time remaining.

    Replaces the old ``_log_batch_progress`` and changes its shape: the predecessor
    took seven positional parameters, five of which (`batch_num`, `total_batches`,
    `total_rows_written`, `start_time`, `phase`) are state the caller had to keep
    alongside the loop itself. Here this object keeps it and the caller is left with
    a single ``progress.tick(rows)``.

    ``total_rows`` may be ``None`` — with `hash_diff` and `id_watermark` the number
    of rows to come is not known in advance. Only counts and elapsed time are logged
    then, without percentages and ETA; lying with a made-up estimate would be worse
    than omitting it.
    """

    #: Usually ``ctx.log``. Takes ``(level, message, *args)``.
    log: Callable[..., None]
    total_rows: Optional[int] = None
    #: Optional phase label, for strategies that read in several passes.
    phase: str = ""
    #: Time source. Swappable for tests; ``monotonic``, not ``time``, so that a jump
    #: in the system clock cannot produce a negative elapsed time.
    clock: Callable[[], float] = time.monotonic
    start: float = field(default=0.0)
    batch_num: int = 0

    def __post_init__(self) -> None:
        self.start = self.clock()

    def tick(self, rows_done: int) -> None:
        """Records a finished batch. ``rows_done`` is the total **since the start**."""
        self.batch_num += 1
        elapsed = max(self.clock() - self.start, 0.0)

        parts = [f"batch {self.batch_num}"]
        if self.total_rows:
            share = min(rows_done / self.total_rows, 1.0)
            parts.append(f"{rows_done:,} / {self.total_rows:,} rows ({share:.1%})")
        else:
            parts.append(f"{rows_done:,} rows")
        parts.append(fmt_duration(elapsed))

        speed = rows_done / elapsed if elapsed > 0 else 0.0
        if speed > 0:
            parts.append(f"{speed:,.0f} rows/s")
            remaining = self._eta(rows_done, speed)
            if remaining is not None:
                parts.append(f"~{fmt_duration(remaining)} left")

        prefix = f"[{self.phase}] " if self.phase else ""
        self.log("info", "📦 %s%s", prefix, ", ".join(parts))

    def _eta(self, rows_done: int, rychlost: float) -> Optional[float]:
        """Time remaining, or ``None`` when there is nothing to compute it from.

        ``None`` also when more rows are done than the estimate promised — that
        happens with estimates taken from metadata (Firebird) and a negative ETA
        would be confusing.
        """
        if not self.total_rows or rows_done >= self.total_rows:
            return None
        return (self.total_rows - rows_done) / rychlost


def error_status(table: str, source: Optional[str], err: Any) -> dict:
    """Status row for a source that failed.

    ``success=False`` is the essential part: the client-side ``test_output`` fails
    the block on exactly that, so a run in which one of sixteen databases dropped out
    must not finish green.

    ``err`` goes through `secrets.redact`. The returned frame is not just read in
    Mage — it is a DataFrame a pipeline can print, write onward or attach to an
    alert, so this column travels further than a log line does. `entrypoint.run`
    redacts before it gets here, with the configured password in ``extra``;
    redacting again is a no-op on that text and covers every other caller.
    """
    from dbextractors.core import secrets

    return {
        "table": table,
        "source": source,
        "rows_written": 0,
        "load_method": "error",
        "success": False,
        "is_incremental": False,
        "data_present": False,
        "error": secrets.redact(err),
    }


__all__ = [
    "STATUS_COLUMNS",
    "BatchProgress",
    "build_status_df",
    "error_status",
    "fmt_duration",
]
