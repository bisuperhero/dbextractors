"""Tests for `core.status` — the status frame and the running batch log.

The status frame has two properties that look cosmetic and are not: **the same
columns on every run** and ``success=False`` for a failed source. The
``test_output`` block in the calling repositories rests on both, that is on
whether the Mage block finishes green or red.
"""

from __future__ import annotations

from typing import List

import pytest

from dbextractors.core import status

# --- Status frame -----------------------------------------------------------


def test_the_frame_always_has_the_same_columns_in_the_same_order() -> None:
    """Otherwise one run would return a different shape from the next and the
    caller could not guard against it."""
    frame = status.build_status_df([{"table": "s.t", "rows_written": 5}])
    assert list(frame.columns) == list(status.STATUS_COLUMNS)


def test_missing_keys_are_filled_in_as_none() -> None:
    frame = status.build_status_df([{"table": "s.t"}])
    assert frame["fallback_reason"].isna().all()
    assert frame["connection_mode"].isna().all()


def test_several_sources_give_several_rows() -> None:
    frame = status.build_status_df(
        [{"table": "s.t", "source": "a"}, {"table": "s.t", "source": "b"}]
    )
    assert frame["source"].tolist() == ["a", "b"]


def test_an_empty_list_is_an_error() -> None:
    """An empty frame would pass in Mage as "nothing happened" — that is not a
    state, it is a fault."""
    with pytest.raises(ValueError, match="empty list of statuses"):
        status.build_status_df([])


def test_an_error_status_is_not_a_success() -> None:
    """Whether the block finishes red rests on this."""
    state = status.error_status("s.t", "db_a", RuntimeError("unreachable"))
    assert state["success"] is False
    assert state["load_method"] == "error"
    assert state["error"] == "unreachable"
    assert state["source"] == "db_a"


def test_an_error_status_only_uses_known_columns() -> None:
    """The keys of an error status must be a subset of the frame's columns."""
    state = status.error_status("s.t", None, ValueError("x"))
    assert set(state) <= set(status.STATUS_COLUMNS)


# --- Duration ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0.0s"),
        (12.34, "12.3s"),
        (59.9, "59.9s"),
        (60.0, "1m 0s"),
        (61.0, "1m 1s"),
        (3599.0, "59m 59s"),
        (3600.0, "1h 0m 0s"),
        (7385.0, "2h 3m 5s"),
    ],
)
def test_the_duration_is_taken_verbatim_from_the_predecessors(seconds, expected) -> None:
    """``_fmt_duration`` — the 60 s and 3600 s band boundaries included."""
    assert status.fmt_duration(seconds) == expected


# --- Batch progress ---------------------------------------------------------


class _Clock:
    """Hand-advanced time, so the test does not depend on how fast the machine
    is."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _recorder() -> tuple:
    recorded: List[str] = []

    def log(_level, message, *args):
        recorded.append(message % args)

    return log, recorded


def test_progress_reports_a_percentage_and_the_time_left() -> None:
    log, recorded = _recorder()
    clock = _Clock()
    progress = status.BatchProgress(log, total_rows=1000, phase="full", clock=clock)

    clock.now = 10.0
    progress.tick(250)

    line = recorded[-1]
    assert "[full]" in line
    assert "batch 1" in line
    assert "250 / 1,000 rows (25.0%)" in line
    assert "25 rows/s" in line
    # 750 rows left at 25 rows/s = 30 s
    assert "~30.0s left" in line


def test_nothing_is_estimated_without_a_known_total() -> None:
    """`hash_diff` and `id_watermark` do not know in advance how many rows are
    coming.

    Lying with a made-up ETA would be worse than leaving it out.
    """
    log, recorded = _recorder()
    clock = _Clock()
    progress = status.BatchProgress(log, total_rows=None, clock=clock)

    clock.now = 5.0
    progress.tick(100)

    line = recorded[-1]
    assert "100 rows" in line
    assert "%" not in line
    assert "left" not in line


def test_an_overshot_estimate_gives_no_negative_eta() -> None:
    """An estimate from metadata (Firebird) tends to come in under the real
    figure — that must not break the log."""
    log, recorded = _recorder()
    clock = _Clock()
    progress = status.BatchProgress(log, total_rows=100, clock=clock)

    clock.now = 1.0
    progress.tick(500)

    line = recorded[-1]
    assert "left" not in line
    assert "(100.0%)" in line, "the share is clipped at 100 %, it does not read 500 %"


def test_the_batches_are_counted() -> None:
    log, recorded = _recorder()
    clock = _Clock()
    progress = status.BatchProgress(log, total_rows=30, clock=clock)

    for i in range(1, 4):
        clock.now = float(i)
        progress.tick(i * 10)

    # The emoji is an anchor for scanning the log in the Mage UI quickly, which
    # is why it is part of the message.
    assert [r.split(",")[0] for r in recorded] == ["📦 batch 1", "📦 batch 2", "📦 batch 3"]


def test_zero_elapsed_time_does_not_divide_by_zero() -> None:
    """The first batch may finish before the clock moves at all."""
    log, recorded = _recorder()
    progress = status.BatchProgress(log, total_rows=10, clock=_Clock())
    progress.tick(5)
    assert "batch 1" in recorded[-1]


def test_a_monotonic_clock_is_the_default() -> None:
    """``time.time`` would report a negative elapsed time if the system clock
    were adjusted."""
    import time

    assert status.BatchProgress(lambda *a: None).clock is time.monotonic
