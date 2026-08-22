"""A connection dropping mid-read must neither bring the run down nor lose rows.

This came out of two long extractions failing in production with::

    🛑 [one source] extraction failed: (mysql.connector.errors.OperationalError)
    2013 (HY000): Lost connection to MySQL server during query

It was not a silent failure — the old side wrapped every batch read in
``with_retry`` and this package called that function from nowhere at all.

The most important test in this file is
`test_without_a_key_there_is_no_resume`. Resuming from the last key is correct
**only** over an ordered read; started without a key it would drop rows and the
run would finish green. Silent data loss is worse than a failure, so failing
there is mandatory.

A second, narrower kind of resume was added later: restarting the identical
query from scratch when nothing has been yielded to the caller yet. That cannot
duplicate anything, because there is nothing yet to duplicate — see
`test_a_dropout_before_the_first_row_restarts_from_scratch`. It must not be
confused with the case above: once a single batch has gone out,
`test_without_a_key_there_is_no_resume` still has to fail, restart or not.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import pytest
from sqlalchemy import exc as sa_exc

from dbextractors.core import reading


def _connection_error(text: str = "2013 Lost connection") -> Exception:
    """The error SQLAlchemy raises when a connection drops."""
    return sa_exc.OperationalError("SELECT 1", {}, Exception(text))


def _sql_error() -> Exception:
    return sa_exc.ProgrammingError("SELECT nonsense", {}, Exception("syntax error"))


class FakeReader:
    """A source that fails after a given number of batches — and gets through on
    the next attempt.

    It holds the rows as a list and answers according to the ``id > X``
    condition, so it records what was really fetched rather than what the test
    assumes.
    """

    def __init__(
        self, rows: List[int], fail_after: List[int], error: Optional[Exception] = None
    ) -> None:
        self.rows = rows
        #: After how many yielded batches to fail, once per pass.
        self.fail_after = list(fail_after)
        self.error = error or _connection_error()
        self.queries: List[str] = []

    def iter_batches(self, engine, sql, batch_size):
        self.queries.append(sql)
        start = None
        if "> " in sql:
            start = int(sql.rsplit("> ", 1)[1])
        remaining = [r for r in self.rows if start is None or r > start]

        limit = self.fail_after.pop(0) if self.fail_after else None
        for yielded, i in enumerate(range(0, len(remaining), batch_size)):
            if limit is not None and yielded >= limit:
                raise self.error
            yield pd.DataFrame({"id": remaining[i : i + batch_size]})


class FakeReaderWithoutPkColumn(FakeReader):
    """Like `FakeReader`, but the batches never carry the pk column at all —
    the shape a projection that drops it, or a surrogate-key scan, produces.

    Exists to pin the case the ``saw_row``/``yielded_any`` split in
    `reading.read_with_resume` exists for: a batch without the key column still
    counts as *yielded*, even though no key was ever *captured*.
    """

    def iter_batches(self, engine, sql, batch_size):
        for batch in super().iter_batches(engine, sql, batch_size):
            yield batch.drop(columns=["id"])


def _read_all(source, batch_size=2, pk="id", **kw) -> List[int]:
    # `base_delay=0`: the exponential back-off is tested in `core/retry.py`,
    # here it would only make the suite longer. Without it this file takes 8 s
    # instead of 0.1 s.
    kw.setdefault("base_delay", 0)
    batches = reading.read_with_resume(
        source,
        engine=None,
        batch_size=batch_size,
        build_sql=lambda after: "SELECT id FROM t"
        + (f" WHERE id > {after}" if after is not None else ""),
        pk_in_batch=pk,
        **kw,
    )
    return [int(v) for d in batches for v in d["id"]]


def _drain(source, batch_size=2, pk="id", **kw) -> List[pd.DataFrame]:
    """Like `_read_all`, but does not assume the batches carry an ``id`` column —
    for tests where a batch may legitimately lack it.
    """
    kw.setdefault("base_delay", 0)
    batches = reading.read_with_resume(
        source,
        engine=None,
        batch_size=batch_size,
        build_sql=lambda after: "SELECT id FROM t"
        + (f" WHERE id > {after}" if after is not None else ""),
        pk_in_batch=pk,
        **kw,
    )
    return list(batches)


def test_without_a_dropout_everything_gets_through() -> None:
    source = FakeReader(list(range(1, 11)), fail_after=[])
    assert _read_all(source) == list(range(1, 11))
    assert len(source.queries) == 1, "with no dropout nothing should be repeated"


def test_a_dropout_fetches_the_rest_without_duplicates() -> None:
    """The heart of the matter: after a failure it carries on, it does not start
    over."""
    source = FakeReader(list(range(1, 11)), fail_after=[3])

    seen = _read_all(source, batch_size=2)

    assert seen == list(range(1, 11)), "rows were lost or duplicated"
    assert len(source.queries) == 2
    assert (
        "id > 6" in source.queries[1]
    ), f"the second query should pick up after the last key, not start over: {source.queries[1]}"


def test_a_repeated_dropout_carries_on() -> None:
    source = FakeReader(list(range(1, 13)), fail_after=[2, 2])

    assert _read_all(source, batch_size=2) == list(range(1, 13))
    assert len(source.queries) == 3


def test_without_a_key_there_is_no_resume() -> None:
    """Without a key there is nothing to resume from — and carrying on quietly
    is **forbidden**.

    Were it attempted, the rows the server had not sent yet would be dropped and
    the run would finish green. Silent data loss is worse than a failure.
    """
    source = FakeReader(list(range(1, 11)), fail_after=[3])

    with pytest.raises(sa_exc.OperationalError):
        _read_all(source, pk=None)

    assert len(source.queries) == 1, "without a key nothing should be repeated at all"


def test_an_error_in_the_sql_is_not_retried() -> None:
    """Retrying a typo in SQL only means failing six seconds later."""
    source = FakeReader(list(range(1, 11)), fail_after=[1], error=_sql_error())

    with pytest.raises(sa_exc.ProgrammingError):
        _read_all(source)

    assert len(source.queries) == 1


def test_a_dropout_before_the_first_row_restarts_from_scratch() -> None:
    """Nothing has reached the caller yet, so a restart cannot duplicate
    anything: the identical query is re-issued rather than the run failing.

    This is the case production actually hits: a FederatedX proxy that dies and
    restarts silently, often within a couple of seconds of a query starting — well
    before the first batch is out.
    """
    source = FakeReader(list(range(1, 11)), fail_after=[0])

    seen = _read_all(source)

    assert seen == list(range(1, 11)), "rows were lost"
    assert len(source.queries) == 2
    assert (
        source.queries[0] == source.queries[1]
    ), "restarting from scratch means the identical query, not a WHERE id > ..."


def test_a_dropout_before_the_first_row_restarts_even_without_a_key() -> None:
    """The same restart-from-scratch applies with ``pk_in_batch=None`` — this is
    the surrogate-key scan in `hash_diff`, which previously had no retry at all
    because it has no key to resume by."""
    source = FakeReader(list(range(1, 11)), fail_after=[0])

    seen = _read_all(source, pk=None)

    assert seen == list(range(1, 11)), "rows were lost"
    assert len(source.queries) == 2


def test_batches_without_the_pk_column_still_block_a_later_restart() -> None:
    """A batch can be handed to the caller without ever containing the pk column
    — that still counts as *yielded*. ``saw_row`` (a key was captured) and
    ``yielded_any`` (something went out) are different facts; conflating them was
    the original bug. Once yielded, a resume must refuse even though no key was
    ever captured.
    """
    source = FakeReaderWithoutPkColumn(list(range(1, 11)), fail_after=[1])

    with pytest.raises(sa_exc.OperationalError):
        _drain(source)

    assert len(source.queries) == 1, "rows already went out; this must not be retried"


def test_restart_from_scratch_is_bounded_by_attempts() -> None:
    """A source that never gets past its first batch must not be retried forever
    — the attempts budget applies to a restart exactly like it does to a keyset
    resume."""
    source = FakeReader(list(range(1, 11)), fail_after=[0, 0, 0])

    with pytest.raises(sa_exc.OperationalError):
        _read_all(source, attempts=3)

    assert len(source.queries) == 3, "three attempts mean three queries"
    assert source.queries[0] == source.queries[1] == source.queries[2]


def test_it_gives_up_after_the_configured_number_of_attempts() -> None:
    source = FakeReader(list(range(1, 100)), fail_after=[1, 1, 1, 1, 1])

    with pytest.raises(sa_exc.OperationalError):
        _read_all(source, attempts=3, base_delay=0)

    assert len(source.queries) == 3, "three attempts mean three queries"


class TestIsTemporaryError:
    """What is retried and what is not. The distinction follows SQLAlchemy's own
    taxonomy (`is_temporary_error`)."""

    @pytest.mark.parametrize(
        "err",
        [
            sa_exc.OperationalError("s", {}, Exception("2013")),
            sa_exc.InterfaceError("s", {}, Exception("gone away")),
            sa_exc.InternalError("s", {}, Exception("deadlock")),
        ],
    )
    def test_connection_errors_yes(self, err) -> None:
        assert reading.is_temporary_error(err) is True

    @pytest.mark.parametrize(
        "err",
        [
            sa_exc.ProgrammingError("s", {}, Exception("syntax")),
            sa_exc.DataError("s", {}, Exception("bad value")),
            sa_exc.IntegrityError("s", {}, Exception("duplicate")),
            ValueError("something else entirely"),
        ],
    )
    def test_everything_else_no(self, err) -> None:
        assert reading.is_temporary_error(err) is False
