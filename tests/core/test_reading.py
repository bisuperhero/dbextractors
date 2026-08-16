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


def test_a_dropout_before_the_first_row_fails() -> None:
    """There is nothing to pick up from — a resume would have to start over,
    which this module does not do."""
    source = FakeReader(list(range(1, 11)), fail_after=[0])

    with pytest.raises(sa_exc.OperationalError):
        _read_all(source)


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
