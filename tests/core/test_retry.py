"""Tests for `dbextractors.core.retry`.

Characterisation tests against the oracle (`with_retry`, `wait_for_port`) plus a
check that no exception is ever swallowed without a log entry.

`with_retry`/`wait_for_port` show up as several "variants" across the
predecessors, but the differences are only in the docstring or a comment
(`ast.unparse` counts those as a difference) — which is why the tests over
`ro.variants(...)` prove that every variant behaves the same rather than that we
are choosing between genuinely different behaviours.

Nothing here touches the network or production: `wait_for_port` /
`socket.create_connection` are tested against localhost (a genuinely closed
port) or through a monkeypatch.
"""

from __future__ import annotations

import logging
import random
import socket
import time
from typing import Any

import pytest

import reference_oracle as ro
from dbextractors.core import retry

# --- with_retry -------------------------------------------------------------


def test_with_retry_success_on_the_first_go_never_touches_the_delay() -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert retry.with_retry(fn, attempts=3, desc="test") == "ok"
    assert len(calls) == 1


@ro.live_only("with_retry")
def test_with_retry_characterisation_success_on_the_first_go() -> None:
    old = ro.get("A-mysql", "with_retry")

    def fn() -> int:
        return 42

    assert old(fn, attempts=3, desc="x") == retry.with_retry(fn, attempts=3, desc="x") == 42


@ro.live_only("with_retry")
def test_with_retry_characterisation_success_after_failures() -> None:
    """Deterministic: `jitter=0` zeroes the random component regardless of
    `random.random()`, so there is no need to patch `random`."""
    old = ro.get("A-mysql", "with_retry")

    def make_fn(fail_times: int):
        state = {"calls": 0}

        def fn() -> str:
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise ValueError(f"boom {state['calls']}")
            return "done"

        return fn, state

    for fail_times in (0, 1, 2):
        fn_old, state_old = make_fn(fail_times)
        fn_new, state_new = make_fn(fail_times)
        kwargs = {"attempts": 4, "base_delay": 0.001, "max_delay": 0.01, "jitter": 0, "desc": "x"}
        assert old(fn_old, **kwargs) == retry.with_retry(fn_new, **kwargs) == "done"
        assert state_old["calls"] == state_new["calls"] == fail_times + 1


@ro.live_only("with_retry")
def test_with_retry_characterisation_attempts_exhausted() -> None:
    old = ro.get("A-mysql", "with_retry")

    def make_fn():
        def fn() -> None:
            raise RuntimeError("always fails")

        return fn

    kwargs = {"attempts": 3, "base_delay": 0.001, "max_delay": 0.01, "jitter": 0, "desc": "x"}

    with pytest.raises(RuntimeError, match="always fails"):
        old(make_fn(), **kwargs)
    with pytest.raises(RuntimeError, match="always fails"):
        retry.with_retry(make_fn(), **kwargs)


@ro.live_only("with_retry")
def test_with_retry_characterisation_edge_case_zero_attempts() -> None:
    """`attempts=0` -> `range(1, 1)` is empty, `fn()` is never called and the
    function returns ``None`` unchanged — neither the old nor the new version
    has an explicit `return` here."""
    old = ro.get("A-mysql", "with_retry")
    calls = []

    def fn() -> str:
        calls.append(1)
        return "never"

    assert old(fn, attempts=0) is None
    assert retry.with_retry(fn, attempts=0) is None
    assert calls == []


@ro.live_only("with_retry")
def test_with_retry_characterisation_exact_delay_with_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full formula, jitter included: `random` is a module shared between
    the oracle (its `exec`-ed functions call the same `sys.modules['random']`)
    and the new code, so a monkeypatch on `random.random` applies equally to
    both sides."""
    old = ro.get("A-mysql", "with_retry")
    monkeypatch.setattr(random, "random", lambda: 0.9)

    old_sleeps: list[float] = []
    new_sleeps: list[float] = []

    def fn_old() -> None:
        raise ValueError("x")

    def fn_new() -> None:
        raise ValueError("x")

    monkeypatch.setattr(time, "sleep", lambda s: old_sleeps.append(s))
    with pytest.raises(ValueError):
        old(fn_old, attempts=3, base_delay=1.0, max_delay=30.0, jitter=0.25, desc="x")

    monkeypatch.setattr(time, "sleep", lambda s: new_sleeps.append(s))
    with pytest.raises(ValueError):
        retry.with_retry(fn_new, attempts=3, base_delay=1.0, max_delay=30.0, jitter=0.25, desc="x")

    assert old_sleeps == new_sleeps
    assert len(old_sleeps) == 2  # 3 attempts -> 2 waits between them


@ro.live_only("with_retry")
def test_with_retry_every_variant_behaves_the_same() -> None:
    """The comparison across the predecessors reports 3 "variants", but they
    differ only in the docstring — this test proves it on behaviour, not on
    text."""
    variants = ro.variants("with_retry")
    assert len(variants) >= 10

    for short, fn in variants.items():
        target, calls = _flaky_target(fail_times=1)
        result = fn(target, attempts=3, base_delay=0.001, max_delay=0.01, jitter=0, desc="x")
        assert result == "ok", short
        assert calls["n"] == 2, short


def _flaky_target(*, fail_times: int) -> tuple[Any, dict[str, int]]:
    """Return a function that fails ``fail_times`` times and then succeeds, plus
    a dict with the call count. Kept outside the loop body because of B023 (a
    closure over the loop variable would otherwise bind the last value, not the
    one from that iteration)."""
    calls = {"n": 0}

    def target() -> str:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ValueError("x")
        return "ok"

    return target, calls


def test_with_retry_logs_every_failed_attempt(caplog: pytest.LogCaptureFixture) -> None:
    """No exception may be lost without a trace."""
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError(f"boom {calls['n']}")
        return "ok"

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.retry"):
        result = retry.with_retry(fn, attempts=5, base_delay=0.001, max_delay=0.01, desc="thing")

    assert result == "ok"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("failed (attempt" in r.getMessage() for r in warnings)


def test_with_retry_logs_the_final_failure(caplog: pytest.LogCaptureFixture) -> None:
    def fn() -> None:
        raise RuntimeError("dead")

    with (
        caplog.at_level(logging.ERROR, logger="dbextractors.core.retry"),
        pytest.raises(RuntimeError),
    ):
        retry.with_retry(fn, attempts=2, base_delay=0.001, max_delay=0.01, desc="thing")

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("failed after 2 attempts" in r.getMessage() for r in errors)


# --- The backoff formula ------------------------------------------------------
#
# Every test of the formula above is a comparison with the oracle, and the
# oracle cannot be replayed for `with_retry` — a function passed as an argument
# is exactly what a recording cannot capture. So wherever the predecessor
# sources are absent, which is everywhere but one machine, none of them run. The
# two tests that do run pass ``base_delay=0``, which makes every formula produce
# zero. `delay = base_delay` — no doubling, no ceiling — survived the whole
# suite.
#
# These run unconditionally. They own the formula; the oracle tests above own
# the claim that it is the *same* formula the predecessors used.


def _sleeps(monkeypatch: pytest.MonkeyPatch, **kwargs) -> list[float]:
    """Run `with_retry` over a function that always fails and collect the waits."""
    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", waits.append)

    def always_fails() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError):
        retry.with_retry(always_fails, desc="x", **kwargs)
    return waits


def test_the_delay_doubles_with_every_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exponential, not constant.

    Retrying a source that is restarting at a fixed interval is what turns a
    recoverable blip into a run that hammers a database until it gives up.
    ``jitter=0`` takes the random component out; it has its own test below.
    """
    waits = _sleeps(monkeypatch, attempts=5, base_delay=1.0, max_delay=1000.0, jitter=0)

    assert waits == [1.0, 2.0, 4.0, 8.0]


def test_the_delay_stops_at_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the cap the doubling runs away: attempt 10 would wait 8.5 minutes."""
    waits = _sleeps(monkeypatch, attempts=6, base_delay=1.0, max_delay=5.0, jitter=0)

    assert waits == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_there_is_one_wait_fewer_than_there_are_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last failure is raised, not slept on. Waiting after it would add the
    delay to every failed run for nothing."""
    waits = _sleeps(monkeypatch, attempts=3, base_delay=1.0, max_delay=100.0, jitter=0)

    assert len(waits) == 2


@pytest.mark.parametrize(
    ("roll", "expected"),
    [
        # `random()` returns [0, 1); jitter spreads the delay by ±25 % of itself.
        (0.0, 0.75),
        (0.5, 1.0),
        (0.9, 1.2),
    ],
)
def test_jitter_spreads_the_delay_both_ways(
    monkeypatch: pytest.MonkeyPatch, roll: float, expected: float
) -> None:
    """Both ways, not just upwards.

    Three pipelines run at once and they fail together when the source does. A
    delay that is only ever lengthened keeps them in step; the point of the
    jitter is that they stop retrying in unison.
    """
    monkeypatch.setattr(random, "random", lambda: roll)
    waits = _sleeps(monkeypatch, attempts=2, base_delay=4.0, max_delay=100.0, jitter=0.25)

    assert waits == pytest.approx([4.0 * expected])


def test_the_default_delays_are_the_frozen_ones() -> None:
    """The signature was identical in all 15 predecessor files and is frozen.

    Defaults matter more than usual here: most callers pass none of them, so a
    changed default changes the behaviour of every retry in the package at once.
    """
    import inspect

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(retry.with_retry).parameters.items()
    }

    assert defaults["attempts"] == 3
    assert defaults["base_delay"] == 2.0
    assert defaults["max_delay"] == 30.0
    assert defaults["jitter"] == 0.25


# --- wait_for_port ------------------------------------------------------------
#
# `wait_for_port` is in `reference_oracle.LIVE_ONLY` — it touches the network,
# so a recording of it says nothing. The characterisation tests below were not
# marked as such, which meant that in replay mode the oracle half of each
# assertion answered from a fixture without ever reaching the monkeypatched
# socket: half the test was decoration. They are marked now, and the behaviour
# of the new implementation is covered by tests that do not need the oracle at
# all.


def test_wait_for_port_returns_true_as_soon_as_the_port_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And on the first try, without waiting: the tunnel is usually up by the
    time this is called, and 0.2 s per pipeline start is real."""

    class _FakeConn:
        def __enter__(self) -> _FakeConn:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    attempts: list[tuple] = []

    def answers(address, **kwargs):
        attempts.append(address)
        return _FakeConn()

    monkeypatch.setattr(socket, "create_connection", answers)
    monkeypatch.setattr(time, "sleep", lambda _s: pytest.fail("waited on an open port"))

    assert retry.wait_for_port(12345, host="127.0.0.1", timeout=1) is True
    assert attempts == [("127.0.0.1", 12345)]


def test_wait_for_port_keeps_trying_until_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """One attempt is not waiting for anything.

    The whole reason this exists is that the SSH tunnel takes a moment to bind:
    a check that gives up after the first refusal would report the tunnel dead
    every time.
    """
    attempts: list[int] = []

    def refuses(*args: object, **kwargs: object):
        attempts.append(1)
        raise OSError("connection refused")

    monkeypatch.setattr(socket, "create_connection", refuses)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert retry.wait_for_port(12345, host="127.0.0.1", timeout=0.3) is False
    assert len(attempts) > 1, "gave up after a single try"


def test_wait_for_port_lets_a_programming_error_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `except` is narrowed from `Exception` to `OSError` on purpose.

    An unreachable target raises `OSError` and its subclasses; a wrong argument
    type does not, and used to disappear into the retry loop until the timeout
    ran out and the tunnel was declared dead for the wrong reason.
    """

    def wrong_argument(*args: object, **kwargs: object):
        raise TypeError("port must be an integer")

    monkeypatch.setattr(socket, "create_connection", wrong_argument)

    with pytest.raises(TypeError):
        retry.wait_for_port(12345, host="127.0.0.1", timeout=0.3)


@ro.live_only("wait_for_port")
def test_wait_for_port_characterisation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    old = ro.get("A-mysql", "wait_for_port")

    class _FakeConn:
        def __enter__(self) -> _FakeConn:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_create_connection(*args: object, **kwargs: object) -> _FakeConn:
        return _FakeConn()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    assert old(12345, host="127.0.0.1", timeout=1) is True
    assert retry.wait_for_port(12345, host="127.0.0.1", timeout=1) is True


@ro.live_only("wait_for_port")
def test_wait_for_port_characterisation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    old = ro.get("A-mysql", "wait_for_port")

    def always_fails(*args: object, **kwargs: object):
        raise OSError("connection refused")

    monkeypatch.setattr(socket, "create_connection", always_fails)
    # So that the test does not needlessly spend 2x100 ms sleeping.
    monkeypatch.setattr(time, "sleep", lambda s: None)

    assert old(12345, host="127.0.0.1", timeout=0.05) is False
    assert retry.wait_for_port(12345, host="127.0.0.1", timeout=0.05) is False


def test_wait_for_port_returns_false_on_a_genuinely_closed_port() -> None:
    """No mock, but no traffic off the machine either: `127.0.0.1` on an unused
    port is a local TCP RST, not network traffic."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    # The port is free again now (and nobody is listening on it) -> the
    # connection has to fail.
    assert retry.wait_for_port(closed_port, host="127.0.0.1", timeout=0.3) is False


@ro.live_only("wait_for_port")
def test_wait_for_port_every_variant_behaves_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    variants = ro.variants("wait_for_port")
    assert len(variants) >= 10

    def always_fails(*args: object, **kwargs: object):
        raise OSError("nope")

    monkeypatch.setattr(socket, "create_connection", always_fails)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    for short, fn in variants.items():
        assert fn(1, host="127.0.0.1", timeout=0.02) is False, short
