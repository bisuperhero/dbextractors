"""Tests for the record-and-replay store itself — `tests/oracle_store.py`.

Every characterisation test in this repository stands on this file. The
predecessor sources are not published, so the "old" side of each comparison is
not a function at all: it is a recorded answer looked up by a fingerprint of the
call. That makes the store load-bearing in a way that is easy to miss, and it
fails in two directions that both look like something else:

- a **fingerprint** that is not specific enough silently hands one call the
  answer belonging to another. This happened: ``bytes`` and ``bytearray``
  encoded identically, so the two recordings overwrote each other and replaying
  ``b'abc'`` returned what ``bytearray(b'abc')`` had produced. Nothing failed —
  the comparison simply verified the wrong pair.
- a **decode** that loses fidelity blames the new implementation for a
  difference the old one never had. ``pd.testing.assert_frame_equal`` compares
  dtypes, so an ``Int64`` coming back as ``int64`` reads as a regression in code
  that is correct.

`tests/test_reference_oracle.py` covers the other half of the machinery — the
AST mining — and is skipped wherever the sources are absent, which is here. So
without this file the replay path has no test that ever runs.

The tests below go through the whole round trip (record -> JSON on disk ->
read back -> replay) rather than calling `encode`/`decode` back to back. The
JSON step is where a value that encodes fine still cannot be stored, and the
lookup step is where a fingerprint collision shows up; neither is visible from
a direct round trip.
"""

from __future__ import annotations

import datetime as dt
import decimal

import numpy as np
import pandas as pd
import pytest

import oracle_store
import reference_oracle as ro


@pytest.fixture()
def record(monkeypatch, tmp_path):
    """Record a series of calls into a scratch fixture directory, then replay them.

    Returns ``(results, replayer)``: what the real function returned while being
    recorded, and a stand-in that answers from the written fixture. The two are
    what every characterisation test compares in replay mode, one step apart.
    """

    def _record(fn, calls, *, short="A-mysql", name=None):
        monkeypatch.setattr(oracle_store, "FIXTURES", tmp_path)
        name = name or fn.__name__
        recording = ro._recording_wrapper(short, name, fn)

        results = []
        for args, kwargs in calls:
            try:
                results.append(recording(*args, **kwargs))
            except Exception as err:  # recorded on purpose — see `_recording_wrapper`
                results.append(err)

        oracle_store.save(short, name, ro._recorded.pop((short, name)))
        return results, ro._replay_function(short, name)

    return _record


# --- Telling calls apart -----------------------------------------------------


def test_bytes_and_bytearray_get_their_own_recordings(record) -> None:
    """The bug this whole section exists for.

    Both hold the same bytes, so a shared encoding gives them the same
    fingerprint and the second recording overwrites the first. The predecessor's
    ``to_jsonb`` tells them apart — ``json.dumps`` fails and the fallback does
    ``str(x)`` — so the two answers genuinely differ, and mixing them up is a
    wrong comparison that reports success.
    """

    def describe(value):
        return f"{type(value).__name__}:{bytes(value).hex()}"

    _, replay = record(describe, [((b"abc",), {}), ((bytearray(b"abc"),), {})])

    assert replay(b"abc") == "bytes:616263"
    assert replay(bytearray(b"abc")) == "bytearray:616263"


def test_the_fingerprint_follows_dict_insertion_order(record) -> None:
    """A known fragility, pinned deliberately — and pinned as *loud*.

    A dict is encoded as a list of pairs in insertion order, and
    ``json.dumps(sort_keys=True)`` sorts object keys, not list elements. So two
    dicts that compare equal fingerprint differently. Sorting them the way sets
    are sorted would be the obvious fix and is not one: it changes every
    fingerprint, which invalidates every recorded fixture in the repository at
    once.

    What makes it tolerable is that it cannot pass silently. A reordered
    argument is simply a call that was never recorded, and the message says so
    and says how to fix it. That — not the ordering itself — is what this test
    guards.
    """
    ordered = oracle_store.fingerprint(({"a": 1, "b": 2},), {})
    reordered = oracle_store.fingerprint(({"b": 2, "a": 1},), {})
    assert ordered != reordered

    _, replay = record(lambda d: sorted(d), [(({"a": 1, "b": 2},), {})], name="keys")

    assert replay({"a": 1, "b": 2}) == ["a", "b"]
    with pytest.raises(ro.OracleError, match="re-record"):
        replay({"b": 2, "a": 1})


def test_a_set_fingerprints_the_same_however_it_was_built(record) -> None:
    """A set has no order, so its encoded elements are sorted before hashing.

    Without that the fingerprint would follow Python's hash randomisation and a
    fixture recorded in one process would not be found in the next.
    """
    assert oracle_store.fingerprint(({1, 2, 3},), {}) == oracle_store.fingerprint(({3, 2, 1},), {})


def test_different_arguments_get_different_answers(record) -> None:
    """The lookup has to be by call, not by function."""
    _, replay = record(str.upper, [(("a",), {}), (("b",), {})], name="upper")

    assert replay("a") == "A"
    assert replay("b") == "B"


def test_a_call_that_was_never_recorded_says_so(record) -> None:
    """The alternative is a ``KeyError`` or a ``None``, both of which look like a
    difference in the new implementation rather than a missing fixture."""
    _, replay = record(str.upper, [(("a",), {})], name="upper")

    with pytest.raises(ro.OracleError, match="No recording"):
        replay("never asked")


def test_a_function_with_no_recordings_at_all_says_so() -> None:
    with pytest.raises(ro.OracleError, match="has not been recorded"):
        ro._replay_function("A-mysql", "a_function_nobody_ever_recorded")


# --- Output parameters -------------------------------------------------------
#
# Several of the old functions return nothing and deliver their result by
# mutating a collection they were handed — ``update_stats(name, series,
# all_columns, stats)`` is the one the conversion tests use. Recording only the
# return value would replay a `None` into a caller that then reads empty
# collections and asserts on them.


def test_a_mutated_set_argument_comes_back_mutated(record) -> None:
    def collect(names, seen):
        seen.update(names)

    seen: set = set()
    results, replay = record(collect, [((["a", "b"], seen), {})])

    assert results == [None] and seen == {"a", "b"}

    replayed: set = set()
    assert replay(["a", "b"], replayed) is None
    assert replayed == {"a", "b"}


def test_a_mutated_dict_argument_comes_back_mutated(record) -> None:
    def tally(values, stats):
        stats["n"] = len(values)
        stats["max"] = max(values)

    stats: dict = {}
    _, replay = record(tally, [(([3, 1, 2], stats), {})])

    replayed: dict = {}
    replay([3, 1, 2], replayed)

    assert replayed == {"n": 3, "max": 3}


def test_an_argument_that_was_not_mutated_is_left_alone(record) -> None:
    """`_apply_out_params` clears before it fills, so a collection the old
    function never touched has to come back exactly as it went in — not
    emptied."""

    def ignore(values, untouched):
        return len(values)

    _, replay = record(ignore, [(([1, 2], {"kept": True}), {})])

    untouched = {"kept": True}
    assert replay([1, 2], untouched) == 2
    assert untouched == {"kept": True}


# --- Type fidelity -----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -17,
        2**63,
        1.5,
        "text",
        "",
        b"\x00\xff",
        bytearray(b"\xde\xad"),
        decimal.Decimal("1234.5678"),
        decimal.Decimal("-1E-10"),
        dt.date(1858, 11, 17),
        dt.time(23, 59, 59, 999999),
        dt.datetime(1753, 1, 1),
        dt.timedelta(hours=25),
        dt.timedelta(hours=-1, minutes=-30),
        (1, "a", None),
        [1, [2, [3]]],
        {"a": 1},
        {1, 2},
        frozenset({"x"}),
        np.int64(-5),
        np.uint64(2**64 - 1),
        np.float32(1.5),
        np.bool_(True),
        pd.Timestamp("2026-01-31 12:34:56.123456"),
    ],
)
def test_a_value_survives_the_round_trip(value) -> None:
    restored = oracle_store.decode(oracle_store.encode(value))

    assert type(restored) is type(value), f"{value!r} came back as {type(restored).__name__}"
    assert restored == value


@pytest.mark.parametrize("value", [float("nan"), pd.NaT, pd.NA])
def test_the_missing_values_survive_by_identity(value) -> None:
    """They cannot be compared with ``==``, which is the reason they need tags of
    their own rather than going through JSON's ``null``: ``None``, ``NaT`` and
    ``NA`` are three different answers from the old functions."""
    restored = oracle_store.decode(oracle_store.encode(value))

    if value is pd.NaT or value is pd.NA:
        assert restored is value
    else:
        assert isinstance(restored, float) and restored != restored


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinities_survive(value) -> None:
    """JSON only knows them as an extension that other readers reject."""
    assert oracle_store.decode(oracle_store.encode(value)) == value


def test_a_frame_keeps_its_dtypes(record) -> None:
    """The reason the store is JSON with type tags rather than ``to_json``.

    ``assert_frame_equal`` compares dtypes, so ``Int64`` arriving as ``int64``
    fails on a difference the old function never produced.
    """
    frame = pd.DataFrame(
        {
            "nullable_int": pd.array([1, None], dtype="Int64"),
            "text": ["a", None],
            "stamp": pd.to_datetime(["2026-01-31", None]),
            "flag": [True, False],
        }
    )

    _, replay = record(lambda df: df, [((frame,), {})], name="identity")

    pd.testing.assert_frame_equal(replay(frame), frame)


def test_an_empty_frame_keeps_its_column_dtypes(record) -> None:
    """``pd.Series([], dtype=…)`` quietly falls back to ``object`` for some
    dtypes, which is why `_rebuild` goes through ``astype``.

    The columns are the claim, not the index: an empty frame comes back with an
    empty ``Index`` of dtype ``object`` where it went in with a ``RangeIndex``.
    No old function returns an empty frame today, so nothing depends on it;
    `_decode_frame` is the one place to change if one ever does.
    """
    frame = pd.DataFrame({"a": pd.array([], dtype="Int64"), "b": pd.Series([], dtype="float64")})

    _, replay = record(lambda df: df, [((frame,), {})], name="identity")
    restored = replay(frame)

    assert dict(restored.dtypes.astype(str)) == {"a": "Int64", "b": "float64"}
    assert len(restored) == 0


def test_a_series_keeps_its_name_index_and_dtype() -> None:
    series = pd.Series([1, None], index=["x", "y"], name="values", dtype="Int64")

    restored = oracle_store.decode(oracle_store.encode(series))

    pd.testing.assert_series_equal(restored, series)


# --- Timestamps and their timezone -------------------------------------------


def test_a_naive_timestamp_stays_naive() -> None:
    value = pd.Timestamp("2026-01-31 12:34:56.123456")
    restored = oracle_store.decode(oracle_store.encode(value))

    assert restored == value and restored.tz is None


def test_an_aware_timestamp_keeps_its_instant_and_its_offset() -> None:
    """The offset travels inside the ISO string, so it survives even though the
    separately stored ``tz`` is never read back."""
    value = pd.Timestamp("2026-01-31 12:34:56+02:00")

    restored = oracle_store.decode(oracle_store.encode(value))

    assert restored == value
    assert restored.utcoffset() == value.utcoffset()


def test_the_stored_timezone_is_recorded_but_not_restored() -> None:
    """A known limitation, pinned so that it is a decision rather than a surprise.

    `encode` writes ``tz`` next to the ISO string, and `decode` ignores it. For a
    fixed offset that changes nothing. For a **named** zone the instant and the
    offset still come back, but the name does not — a value that went in as
    ``Europe/Prague`` comes back as ``UTC+01:00``. No old function returns a
    zone name, so nothing depends on it today; if one ever does, the field is
    already in the fixtures and `decode` is the only place to change.
    """
    value = pd.Timestamp("2026-01-31 12:34:56", tz="Europe/Prague")

    encoded = oracle_store.encode(value)
    restored = oracle_store.decode(encoded)

    assert encoded["tz"] == "Europe/Prague"
    assert restored == value, "the instant is the same"
    assert str(restored.tz) != "Europe/Prague", "but the name is not restored"


# --- What must not be replayed silently --------------------------------------


def test_a_function_argument_refuses_to_replay() -> None:
    """`with_retry(operation, …)` is entirely about how often ``operation`` is
    called; a replayed "result" would verify nothing. The message has to point
    at `LIVE_ONLY`, because the fix is to mark the test, not to add a tag."""
    encoded = oracle_store.encode(lambda x: x)

    with pytest.raises(oracle_store.DecodeError, match="LIVE_ONLY"):
        oracle_store.decode(encoded)


def test_a_value_that_could_not_be_serialised_refuses_to_replay() -> None:
    """The fixture keeps the type and the ``repr`` so a reviewer can see what was
    there, but restoring it would be a guess."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    encoded = oracle_store.encode(Opaque())

    assert encoded["type"] == "Opaque" and encoded["repr"] == "<opaque>"
    with pytest.raises(oracle_store.DecodeError, match="oracle_store"):
        oracle_store.decode(encoded)


def test_an_object_without_a_tag_is_a_corrupt_fixture() -> None:
    """Every encoded object carries a tag. One that does not is a hand-edited or
    truncated fixture, and saying so beats a ``KeyError`` three frames deeper."""
    with pytest.raises(oracle_store.DecodeError, match="without a"):
        oracle_store.decode({"v": 1})


def test_an_unknown_tag_is_a_corrupt_fixture() -> None:
    with pytest.raises(oracle_store.DecodeError, match="Unknown tag"):
        oracle_store.decode({oracle_store.TAG: "invented"})


# --- Recorded failures -------------------------------------------------------


def test_an_exception_is_replayed_as_an_exception(record) -> None:
    """Some characterisation tests assert precisely that the old function fails
    on a given input. Recording only successes would make replay behave
    differently from the live oracle on exactly those tests."""

    def strict(value):
        return int(value)

    results, replay = record(strict, [(("nonsense",), {})], name="strict")

    assert isinstance(results[0], ValueError)
    with pytest.raises(ValueError, match="nonsense"):
        replay("nonsense")


# --- The fixtures stay reviewable --------------------------------------------


def test_the_recorded_arguments_can_be_read_back(record, tmp_path) -> None:
    """The stored ``args`` are not used for the lookup — the fingerprint is — so
    nothing would fail if they were nonsense. They are there to be read: the
    format is JSON precisely so that a fixture can be reviewed by eye and a diff
    shows what changed. This checks they still say what the call was.
    """
    _, _replay = record(lambda a, b=None: (a, b), [((7,), {"b": "x"})], name="pair")

    calls = oracle_store.load("A-mysql", "pair")
    (only_call,) = calls.values()

    assert oracle_store.decode(only_call["args"]) == [7]
    assert oracle_store.decode(only_call["kwargs"]) == {"b": "x"}
