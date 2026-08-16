"""Tests for `MySQLDialect` — mainly the parts that can be compared verbatim
with the predecessor.

The hash expression is a special case: the **only** proof that ``row_hash`` will
not change is that the generated SQL is byte-for-byte identical to the
predecessor's. The same expression in the same engine over the same data returns
the same digest — and conversely, any difference in the expression silently
recomputes the digests of some 530 tables and the next run transfers everything.
"""

from __future__ import annotations

import pytest

from dbextractors.dialects.base import SurrogateKey
from dbextractors.dialects.mysql import MySQLDialect


@pytest.fixture()
def dialect() -> MySQLDialect:
    return MySQLDialect()


# --- Hash expression --------------------------------------------------------


def test_hash_expression_is_byte_identical_to_the_predecessor(dialect) -> None:
    """``build_hash_expr`` as the predecessor's MySQL extractor writes it."""
    expected = (
        "SHA2(CONCAT_WS('||', COALESCE(CAST(`id` AS CHAR), ''), "
        "COALESCE(CAST(`name` AS CHAR), '')), 256) AS `row_hash`"
    )
    assert dialect.render_hash_expr(["id", "name"], "row_hash") == expected


def test_hash_expression_ignores_column_types(dialect) -> None:
    """``column_types`` is in the signature only for MSSQL; MySQL must not use it."""
    without = dialect.render_hash_expr(["a"], "h")
    with_types = dialect.render_hash_expr(["a"], "h", {"a": "varbinary"})
    assert without == with_types


# --- Literals and the key filter --------------------------------------------


def test_literal_doubles_the_apostrophe(dialect) -> None:
    assert dialect.sql_literal("O'Brien") == "'O''Brien'"


def test_literal_doubles_the_backslash_as_well(dialect) -> None:
    """MySQL is the only one of the four dialects where ``\\`` is an escape character.

    Without doubling it, a key ending in a backslash would swallow the closing
    apostrophe and the query would fall apart — whereas in the other dialects
    doubling would change the value.
    """
    from dbextractors.dialects.base import SourceDialect

    assert dialect.sql_literal("path\\") == "'path\\\\'"
    assert SourceDialect.sql_literal(dialect, "path\\") == "'path\\'"


def test_literal_leaves_numbers_unquoted(dialect) -> None:
    assert dialect.sql_literal(42) == "42"


def test_literal_treats_bool_as_text(dialect) -> None:
    """``True`` is an ``int`` in Python, but a key never is — keep them apart."""
    assert dialect.sql_literal(True) == "'True'"


def test_key_filter(dialect) -> None:
    assert dialect.render_key_filter("id", ["1", "2"]) == "`id` IN ('1', '2')"


def test_key_filter_with_a_surrogate_key(dialect) -> None:
    """No column of that name exists in the source — the expression has to be used.

    And **without** ``AS``: an alias inside ``WHERE`` is a syntax error.
    """
    surrogate = SurrogateKey(enabled=True, alias="sk", expr="CONCAT(a, '-', b)")
    assert dialect.render_key_filter("sk", ["x"], surrogate) == "(CONCAT(a, '-', b)) IN ('x')"


def test_key_filter_without_keys_fails(dialect) -> None:
    """``IN ()`` is a syntax error — better caught here than at the source."""
    with pytest.raises(ValueError):
        dialect.render_key_filter("id", [])


# --- Connection parameters --------------------------------------------------


def test_streams_instead_of_buffering(dialect) -> None:
    """Without ``buffered=False`` the whole result is pulled into client RAM.

    SQLAlchemy 1.4 takes a buffered cursor for ``mysqlconnector``, so
    ``read_sql(chunksize=)`` does return batches, but only after everything has
    been downloaded. On 6 M rows that is 4 667 MB against 318 MB — and the
    orchestrator runs 3 pipelines at once.
    """
    assert dialect.connect_args.get("buffered") is False


def test_the_other_dialects_have_no_connect_args(dialect) -> None:
    """It is a property of this driver, not a general setting — keep it that way."""
    from dbextractors.dialects.base import SourceDialect

    assert SourceDialect.connect_args == {}


def test_the_session_raises_both_network_timeouts(dialect) -> None:
    """Two long extractions died in production as "2013 Lost connection to MySQL
    server during query", and this is the mitigation.

    The server waits ``net_write_timeout`` seconds (MySQL defaults to 60) for a
    slow client and then closes the connection. Unbuffered batch reading exposes
    that fully: converting a whole batch happens between two ``fetch`` calls, and
    on a wide table that pause is long. Ten minutes is comfortably above the
    worst measured batch — 3 644 rows/s over 48 columns, about 14 s for 50 000
    rows — and both directions are raised, because a read timeout on the same
    connection has the same effect.
    """
    (statement,) = dialect.session_sql

    assert "net_write_timeout = 600" in statement
    assert "net_read_timeout = 600" in statement


def test_the_other_dialects_set_nothing_on_the_session(dialect) -> None:
    """Like ``connect_args``, this is one driver's problem and must not become
    everybody's."""
    from dbextractors.dialects.base import SourceDialect

    assert SourceDialect.session_sql == ()


def test_a_missing_charset_falls_back_to_utf8mb4(dialect) -> None:
    """A configuration without ``charset`` must still produce a usable URL.

    Configuration parsing deliberately does not fill a charset in — which
    encoding a database expects is the dialect's knowledge, not the parser's —
    so ``SOURCE_DB`` without one reaches the dialect as an explicit ``None``.

    That used to become ``?charset=None`` in the URL, and the driver rejected it
    with "Character set 'None' unsupported". Every MySQL pipeline that did not
    spell the charset out was affected; it went unnoticed because no test had
    ever built a connection against a live server.
    """
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "d", "charset": None}, "h", 3306
    )

    assert url.endswith("?charset=utf8mb4")
    assert "None" not in url


def test_an_explicit_charset_is_kept(dialect) -> None:
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "d", "charset": "latin2"}, "h", 3306
    )

    assert url.endswith("?charset=latin2")


# --- Size estimate ----------------------------------------------------------
#
# The largest source of the four had no test of `estimate_size` at all. The
# mutation this section exists to kill is "catch the exception and return zero
# rows": a source that cannot be counted then looks exactly like an empty table,
# and with ``empty_rows_ok`` the run overwrites the target with nothing and
# reports success. That is the failure mode rule 5 of the specification is
# written against.
#
# The engine is a real SQLAlchemy engine over in-memory SQLite, not a stand-in.
# `estimate_size` reads its sample through `pd.read_sql` and sizes it with
# `memory_usage(deep=True)`; against a hand-written fake connection that
# arithmetic never runs, so the test would only prove that the method calls what
# we already assumed it calls. SQLite accepts MySQL's backtick quoting, so the
# SQL the dialect generates crosses unchanged.


@pytest.fixture()
def engine():
    """In-memory SQLite with three tables: five rows, none, and half a megabyte.

    ``bulky`` earns its place: ``size_mb`` is rounded to two decimals, so five
    short rows come out as a legitimate ``0.0`` and an assertion on them could
    not tell a working estimate from one that lost the sample.

    In-memory SQLite keeps a single connection under SQLAlchemy 1.4, so the
    tables survive between the dialect's own ``engine.connect()`` calls.
    """
    from sqlalchemy import create_engine, text

    eng = create_engine("sqlite://")
    with eng.connect() as con:
        con.execute(text("CREATE TABLE paged (id INTEGER, label TEXT)"))
        con.execute(text("CREATE TABLE empty (id INTEGER, label TEXT)"))
        con.execute(text("CREATE TABLE bulky (id INTEGER, label TEXT)"))
        for i in range(1, 6):
            con.execute(text("INSERT INTO paged VALUES (:id, :label)"), {"id": i, "label": f"r{i}"})
        con.execute(
            text("INSERT INTO bulky VALUES (:id, :label)"),
            [{"id": i, "label": "x" * 1000} for i in range(1, 501)],
        )
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture()
def statements(engine):
    """Every statement that reached the driver.

    Which queries are *not* sent matters as much as the numbers returned — a
    saved ``COUNT(*)`` on a table of 15 M rows is the whole point of the
    ``known_total_rows`` branch.
    """
    from sqlalchemy import event

    seen: list[str] = []
    event.listen(engine, "before_cursor_execute", lambda con, cur, sql, *rest: seen.append(sql))
    return seen


def test_the_estimate_counts_the_rows_and_measures_a_sample(dialect, engine) -> None:
    from dbextractors.dialects.base import TableRef

    estimate = dialect.estimate_size(engine, TableRef(name="paged"))

    assert estimate.rows == 5
    assert estimate.method == "count_and_sample"


def test_the_size_follows_the_data(dialect, engine) -> None:
    """500 rows of a kilobyte each are about half a megabyte.

    The exact number is pandas' business, so the assertion is a range — but a
    range that a zero, or a size measured over the wrong table, falls outside
    of. ``size_mb`` decides whether a load runs in one pass or in windows.
    """
    from dbextractors.dialects.base import TableRef

    estimate = dialect.estimate_size(engine, TableRef(name="bulky"))

    assert estimate.rows == 500
    assert 0.4 < estimate.size_mb < 1.0, estimate


def test_the_where_clause_narrows_the_count(dialect, engine) -> None:
    """The estimate has to describe the rows that will actually be read.

    Without the filter a windowed load reports the size of the whole table, and
    the progress logging then claims a finished run is at two percent.
    """
    from dbextractors.dialects.base import TableRef

    assert dialect.estimate_size(engine, TableRef(name="paged"), where="id > 3").rows == 2


def test_an_empty_table_is_never_sampled(dialect, engine, statements) -> None:
    """Zero rows is a legitimate answer — but only from a ``COUNT`` that ran.

    The sample is pointless then, and the predecessor skipped it too. The check
    on the statements is what tells this zero apart from a swallowed failure.
    """
    from dbextractors.dialects.base import TableRef

    estimate = dialect.estimate_size(engine, TableRef(name="empty"))

    assert estimate.rows == 0 and estimate.size_mb == 0.0
    assert len(statements) == 1, statements
    assert "COUNT(*)" in statements[0]


def test_a_known_row_count_saves_the_count(dialect, engine, statements) -> None:
    """The caller has already counted (the fingerprint does), so counting again
    over a table of millions of rows is pure cost."""
    from dbextractors.dialects.base import TableRef

    estimate = dialect.estimate_size(engine, TableRef(name="paged"), known_total_rows=7)

    assert estimate.rows == 7
    assert estimate.method == "known_total_rows"
    assert not any("COUNT" in sql.upper() for sql in statements), statements


def test_the_sample_is_limited(dialect, engine, statements) -> None:
    """Otherwise the "estimate" reads the whole table it is meant to size."""
    from dbextractors.dialects.base import TableRef

    dialect.estimate_size(engine, TableRef(name="paged"), sample_size=2)

    assert any("LIMIT 2" in sql for sql in statements), statements


def test_a_failed_count_fails_loudly(dialect, engine) -> None:
    """**Rule 5.** A source that cannot be counted must not pass for an empty one.

    With ``empty_rows_ok`` set — and many pipelines set it, because an empty
    window is normal for them — a returned zero truncates the target and the run
    comes out green. So the exception has to leave the method, not a
    `SizeEstimate`.
    """
    from dbextractors.dialects.base import TableRef

    with pytest.raises(Exception, match="no such table"):
        dialect.estimate_size(engine, TableRef(name="does_not_exist"))


def test_a_failed_sample_fails_loudly_too(dialect, engine, monkeypatch) -> None:
    """The other half of the same rule: the count succeeded, the sample did not.

    Patched at `pandas.read_sql` rather than by breaking the table, because this
    is the branch where the row count is already known and correct — the
    tempting swallow here returns *that* count with a zero size, and the size is
    what decides whether a load runs in one pass or in windows.
    """
    import pandas

    from dbextractors.dialects.base import TableRef

    def boom(*args, **kwargs):
        raise RuntimeError("the source went away mid-sample")

    monkeypatch.setattr(pandas, "read_sql", boom)

    with pytest.raises(RuntimeError, match="mid-sample"):
        dialect.estimate_size(engine, TableRef(name="paged"))
