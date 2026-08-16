"""Tests for the target-breaking helper the golden batch uses to exercise the
fetch path.

Without breaking something, a hash run over an unchanged source is correctly
**idle** — it transfers zero rows and the fetch path never runs at all. The
source may only be read from, so the change has to be made on the target side.

It is an instrument we measure with, so it needs tests of its own: if `perturb`
broke nothing and only claimed to, the whole gate would pass on an empty run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from golden_batch import PERTURB_MARKER, count_marker, perturb  # noqa: E402

# Every test here writes into a scratch schema through `rw_conn`, so the whole
# module needs a live PostgreSQL.
pytestmark = pytest.mark.needs_pg


@pytest.fixture()
def table(rw_conn, workspace: str) -> str:
    with rw_conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{workspace}".t (id integer, label text, number integer, row_hash text)'
        )
        cur.execute(
            f'INSERT INTO "{workspace}".t '
            f"SELECT i, 'row ' || i, i, md5(i::text) FROM generate_series(1, 100) i"
        )
    return "t"


def test_it_breaks_the_requested_number_of_rows(rw_conn, workspace, table) -> None:
    column, count = perturb(rw_conn, workspace, table, "id", 10)

    assert column == "label"
    assert count == 10
    assert count_marker(rw_conn, workspace, table, column) == 10


def test_it_throws_the_hash_away_too(rw_conn, workspace, table) -> None:
    """Without that, the diff would not find the broken row — it compares by hash."""
    perturb(rw_conn, workspace, table, "id", 10)

    with rw_conn.cursor() as cur:
        cur.execute(
            f'SELECT count(*) FROM "{workspace}".t WHERE row_hash LIKE %s', (f"{PERTURB_MARKER}%",)
        )
        assert cur.fetchone()[0] == 10


def test_it_breaks_at_most_half(rw_conn, workspace, table) -> None:
    """If nothing matched, the safety brake would fire and a full load would run.

    The fetch path would then again go untested — and that is precisely what the
    breaking is for.
    """
    _column, count = perturb(rw_conn, workspace, table, "id", 1000)

    assert count == 50


def test_a_table_without_a_text_column_is_not_broken(rw_conn, workspace) -> None:
    """It has to say so rather than quietly do nothing — otherwise the gate passes
    on an empty run."""
    with rw_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{workspace}".numbers_only (id integer, number integer)')
        cur.execute(f'INSERT INTO "{workspace}".numbers_only VALUES (1, 1), (2, 2)')

    column, count = perturb(rw_conn, workspace, "numbers_only", "id", 10)

    assert column is None
    assert count == 0


def test_the_primary_key_is_not_broken(rw_conn, workspace) -> None:
    """Breaking the key would turn the row into a new one rather than a changed
    one — that is a different scenario."""
    with rw_conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{workspace}".keyed (code text, description text, row_hash text)'
        )
        cur.execute(f"INSERT INTO \"{workspace}\".keyed VALUES ('a', 'x', 'h'), ('b', 'y', 'h')")

    column, count = perturb(rw_conn, workspace, "keyed", "code", 1)

    assert column == "description"
    assert count == 1
    assert count_marker(rw_conn, workspace, "keyed", "code") == 0


def test_managed_columns_are_not_broken(rw_conn, workspace) -> None:
    """`_timestamp` and the underscore-prefixed columns are managed by the package,
    not by the source.

    Breaking `_timestamp` would prove nothing — it is overwritten on every write.
    """
    with rw_conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{workspace}".managed '
            f'(id integer, "_timestamp" text, "_name" text, description text, row_hash text)'
        )
        cur.execute(
            f'INSERT INTO "{workspace}".managed '
            f"VALUES (1, 'a', 'b', 'c', 'h'), (2, 'a', 'b', 'c', 'h')"
        )

    column, count = perturb(rw_conn, workspace, "managed", "id", 1)

    assert column == "description"
    assert count == 1


def test_a_table_that_is_too_small_is_not_broken(rw_conn, workspace) -> None:
    """It returns **with** the column name — this is a different reason from
    "nothing to break".

    The golden batch tells the two apart in its message; were the two states to
    merge, a small table would look like a table without a text column.
    """
    with rw_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{workspace}".empty (id integer, label text)')

    column, count = perturb(rw_conn, workspace, "empty", "id", 10)

    assert column == "label"
    assert count == 0
