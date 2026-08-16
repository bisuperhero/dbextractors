"""Live progress output for long comparisons.

A golden test over a large table stays silent for minutes — one table of 1.2 M
rows and 97 columns took 247 s, most of it in level 5, which does the
``FULL OUTER JOIN``. From a terminal that is indistinguishable from a hung
process, and a person kills it before it finishes.

Two decisions this rests on:

**It writes to stderr, not to stdout.** The report goes to stdout, where it is
piped, saved and read mechanically. Progress is information for the person at
the terminal, not part of the result.

**It turns itself on based on whether stderr is a terminal.** When the output
goes to a file or a CI log, the ticking numbers would only clutter it; worse,
a new line would appear every second, because overwriting through ``\\r`` does
not work in a file.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional, TextIO


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes} m {remainder} s"


class Progress:
    """The interface. The default implementation is silent — for non-interactive runs."""

    def message(self, text: str) -> None:
        """One-off information (row counts, which table out of how many)."""

    def start(self, label: str) -> None:
        """A phase that may take a long time is starting."""

    def finish(self, seconds: float) -> None:
        """The phase has finished."""

    def close(self) -> None:
        """Cleans up after itself — called on exceptions too."""


class TerminalProgress(Progress):
    """Live terminal output with a running timer.

    The running timer is the point: "level 5 is running" looks exactly the same
    after two minutes as it did at the start. Only a rising number says that
    something is happening.

    The thread is a daemon and event driven, so it has no way of getting stuck on
    ``Ctrl-C`` or on an exception. It writes strictly between `start` and
    `finish`, so it never fights the main thread over the output.
    """

    #: How often the running timer is redrawn.
    TICK_INTERVAL_S = 1.0

    def __init__(self, stream: Optional[TextIO] = None, line_width: int = 78) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._width = line_width
        self._label: Optional[str] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    # --- internals ---

    def _clear_line(self) -> None:
        self._stream.write("\r" + " " * self._width + "\r")
        self._stream.flush()

    def _tick(self, label: str, since: float, stop: threading.Event) -> None:
        while not stop.wait(self.TICK_INTERVAL_S):
            self._stream.write(f"\r  ⏳ {label} … {format_duration(time.monotonic() - since)}")
            self._stream.flush()

    # --- interface ---

    def message(self, text: str) -> None:
        self._stream.write(f"  {text}\n")
        self._stream.flush()

    def start(self, label: str) -> None:
        if self._thread is not None:
            # The previous phase never finished — close it, so that two threads
            # do not run at once and overwrite each other's line.
            self.finish(0.0)
        self._label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._tick, args=(label, time.monotonic(), self._stop_event), daemon=True
        )
        self._thread.start()

    def finish(self, seconds: float) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.TICK_INTERVAL_S * 2)
        self._clear_line()
        # Finished phases are not printed: only the one currently running is
        # interesting, and the finished report covering all five levels arrives
        # on stdout in a moment. The exception is a phase that took a long time —
        # that one is worth having in the transcript.
        if self._label and seconds >= 10:
            self._stream.write(f"  ✓ {self._label} · {format_duration(seconds)}\n")
            self._stream.flush()
        self._label = None
        self._stop_event = None
        self._thread = None

    def close(self) -> None:
        if self._thread is not None:
            self.finish(0.0)


def build(enabled: Optional[bool] = None, stream: Optional[TextIO] = None) -> Progress:
    """Progress reporter chosen from the environment.

    Args:
        enabled: ``None`` = decide by whether the stream is a terminal.
            ``True``/``False`` override it (the command-line switch).
    """
    target = stream if stream is not None else sys.stderr
    if enabled is None:
        enabled = bool(getattr(target, "isatty", lambda: False)())
    return TerminalProgress(target) if enabled else Progress()
