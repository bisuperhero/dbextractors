"""End-to-end tests of the ``dbx-golden`` command line.

The exit code is the CLI's real product: a migration script moves a group of
tables only when every comparison returns 0, so a wrong code either blocks a
clean migration or — far worse — waves a differing one through. The audit put
``cli.py`` at 18 % coverage with no test touching the exit codes at all.

``cli.main(argv)`` is called directly rather than through a subprocess: what is
under test is the argument wiring and the verdict-to-exit-code mapping, not the
console entry point that setuptools generates.
"""

from __future__ import annotations

import os

import pytest

from dbextractors.golden import cli, scratch

pytestmark = pytest.mark.needs_pg


def _make_table(conn, schema: str, table: str) -> None:
    """A small table with a primary key, so that level 5 can pair rows up."""
    qualified = f'"{schema}"."{table}"'
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {qualified} (id integer PRIMARY KEY, note text, row_hash text)")
        cur.execute(
            f"INSERT INTO {qualified} "
            "SELECT g, 'note ' || g, md5(g::text) FROM generate_series(1, 20) g"
        )


@pytest.fixture()
def cli_pair(rw_conn, workspace):
    """Two identical tables the CLI can be pointed at."""
    _make_table(rw_conn, workspace, "old")
    _make_table(rw_conn, workspace, "new")
    return f"{workspace}.old", f"{workspace}.new"


# --- compare: the verdict-to-exit-code mapping -------------------------------


def test_compare_exits_0_on_a_match(dsn, cli_pair):
    left, right = cli_pair
    assert cli.main(["--dsn", dsn, "compare", left, right]) == cli.EXIT_MATCH


def test_compare_exits_1_on_a_difference(dsn, rw_conn, workspace, cli_pair):
    left, right = cli_pair
    with rw_conn.cursor() as cur:
        cur.execute(f'UPDATE "{workspace}"."new" SET note = %s WHERE id = 7', ("other",))
    assert cli.main(["--dsn", dsn, "compare", left, right]) == cli.EXIT_DIFF


def test_compare_exits_1_when_a_table_is_missing(dsn, cli_pair):
    """A missing table is DIFF (1), **not** ERROR (2).

    This deliberately pins the actual semantics: a table that does not exist
    fails level 1 with a definite answer — "the outputs differ" — while ERROR is
    reserved for "we do not know whether they differ" (a broken DSN, a malformed
    argument). A migration script must treat a missing table as a red light, and
    conflating it with infrastructure trouble would hide which of the two it is.
    """
    left, _ = cli_pair
    code = cli.main(["--dsn", dsn, "compare", left, "dbx_golden_no_such_schema.nope"])
    assert code == cli.EXIT_DIFF


def test_compare_exits_2_on_a_malformed_table_spec(dsn):
    """A bare table name is refused — an implicit search_path is an ambiguity an
    oracle must not have — and refusal is an ERROR, not a DIFF."""
    assert cli.main(["--dsn", dsn, "compare", "no_dot_here", "also_none"]) == cli.EXIT_ERROR


def test_compare_exits_2_when_the_database_is_unreachable():
    """SessionError has to surface as exit 2, not as a traceback. No fixture on
    purpose: the connection fails before any table would be looked at."""
    dead_dsn = "host=127.0.0.1 port=1 dbname=nowhere user=nobody"
    code = cli.main(["--dsn", dead_dsn, "compare", "dbx_golden_x.a", "dbx_golden_x.b"])
    assert code == cli.EXIT_ERROR


# --- leftovers --clean: the only place the CLI deletes anything --------------


def test_leftovers_clean_drops_golden_schemas_and_nothing_else(dsn, rw_conn):
    """``leftovers --clean`` is the single destructive code path in the CLI, so
    it gets both directions asserted at once: a ``dbx_golden_`` schema disappears
    and a neighbouring schema without the prefix survives untouched. A mutant
    that widens the ``LIKE`` filter or bypasses ``assert_safe`` fails here on a
    live server, not just in unit-level reasoning.
    """
    leftover = scratch.scratch_schema_name("cliclean")
    scratch.create_schema(rw_conn, leftover)
    # Created directly on purpose: scratch.create_schema would (rightly) refuse
    # a name without the prefix.
    bystander = f"tmp_dbx_cli_bystander_{os.getpid()}"
    with rw_conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{bystander}"')

    def schema_exists(name: str) -> bool:
        with rw_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s", (name,))
            return cur.fetchone()[0] == 1

    try:
        assert cli.main(["--dsn", dsn, "leftovers", "--clean"]) == cli.EXIT_MATCH
        assert not schema_exists(leftover), "the golden leftover has to be dropped"
        assert schema_exists(bystander), "a schema without the prefix must survive --clean"
    finally:
        with rw_conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{bystander}" CASCADE')
            cur.execute(f'DROP SCHEMA IF EXISTS "{leftover}" CASCADE')


def test_leftovers_without_clean_only_lists(dsn, rw_conn, capsys):
    """Without --clean the command must not delete: it opens a read-only session,
    so even a mutant that tried to drop would hit the database-side guard."""
    leftover = scratch.scratch_schema_name("clilist")
    scratch.create_schema(rw_conn, leftover)
    try:
        assert cli.main(["--dsn", dsn, "leftovers"]) == cli.EXIT_MATCH
        out = capsys.readouterr().out
        assert leftover in out
        with rw_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s", (leftover,))
            assert cur.fetchone()[0] == 1, "listing must not drop the schema"
    finally:
        scratch.drop_schema(rw_conn, leftover)
