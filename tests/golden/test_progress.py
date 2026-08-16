"""Live output during a long comparison.

It came out of operations: one table took 247 s and another 206 s, and the
terminal stayed silent for all of that. There was no telling it apart from a hung
process.

The tests guard three things that are easy to get wrong and would in operation be
worse than no output at all:

1. **Nothing may reach stdout.** That is where the report goes, and it is piped
   and read mechanically.
2. **Outside a terminal it keeps quiet.** Overwriting a line through ``\\r`` does
   not work in a file, so the log would gain a line for every second of the run.
3. **The thread ends even on an exception.** Otherwise it would keep overwriting
   the line under the error message.
"""

from __future__ import annotations

import io
import time

import pytest

from dbextractors.golden import progress as progress_mod


class _Terminal(io.StringIO):
    """A StringIO that pretends to be a terminal."""

    def isatty(self) -> bool:
        return True


def test_outside_a_terminal_it_keeps_quiet() -> None:
    stream = io.StringIO()
    reporter = progress_mod.build(stream=stream)

    reporter.message("something")
    reporter.start("level 5")
    reporter.finish(120.0)

    assert isinstance(reporter, progress_mod.Progress)
    assert not isinstance(reporter, progress_mod.TerminalProgress)
    assert stream.getvalue() == "", "nothing should flow into a file or a CI log"


def test_in_a_terminal_it_prints() -> None:
    reporter = progress_mod.build(stream=_Terminal())
    assert isinstance(reporter, progress_mod.TerminalProgress)


def test_the_switch_overrides_the_terminal() -> None:
    """`--no-progress` has to silence a terminal as well."""
    assert not isinstance(
        progress_mod.build(False, stream=_Terminal()), progress_mod.TerminalProgress
    )
    # And the other way round: it can be forced on where it would keep quiet
    # (a run under `nohup`).
    assert isinstance(progress_mod.build(True, stream=io.StringIO()), progress_mod.TerminalProgress)


def test_the_clock_keeps_running_while_the_phase_does(monkeypatch) -> None:
    """Without a rising number a minute-long phase looks exactly like a hang."""
    monkeypatch.setattr(progress_mod.TerminalProgress, "TICK_INTERVAL_S", 0.02)
    stream = _Terminal()
    reporter = progress_mod.build(stream=stream)

    reporter.start("level 5")
    time.sleep(0.15)
    reporter.finish(0.15)

    assert stream.getvalue().count("⏳") >= 2, "it should redraw more than once"
    assert "level 5" in stream.getvalue()


def test_a_short_phase_leaves_nothing_behind() -> None:
    """Finished levels are not printed — the report on stdout covers all five."""
    stream = _Terminal()
    reporter = progress_mod.build(stream=stream)

    reporter.start("level 2")
    reporter.finish(0.4)

    assert "✓" not in stream.getvalue()


def test_a_long_phase_stays_in_the_transcript() -> None:
    """What took a long time should stay in the terminal — it is the raw material
    for measurements."""
    stream = _Terminal()
    reporter = progress_mod.build(stream=stream)

    reporter.start("level 5")
    reporter.finish(206.0)

    assert "✓ level 5 · 3 m 26 s" in stream.getvalue()


def test_an_exception_in_a_level_ends_the_thread(monkeypatch) -> None:
    """`_with_progress` has a ``finally`` — without it the thread would write over the
    error message."""
    from dbextractors.golden.compare import _with_progress

    monkeypatch.setattr(progress_mod.TerminalProgress, "TICK_INTERVAL_S", 0.02)
    reporter = progress_mod.build(stream=_Terminal())

    def _fails(_ctx):
        raise RuntimeError("the level failed")

    with pytest.raises(RuntimeError):
        _with_progress(reporter, "level 4", _fails, None)

    assert reporter._thread is None, "the thread has to be finished even after an exception"


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.4, "0.4 s"), (59.9, "59.9 s"), (60.0, "1 m 0 s"), (206.46, "3 m 26 s")],
)
def test_duration_formatting(seconds, expected) -> None:
    assert progress_mod.format_duration(seconds) == expected
