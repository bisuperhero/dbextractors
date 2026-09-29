"""The ABRA ID counter as each dialect renders it, on strings and against live servers.

`abra_watermark` compares the counter part of an ABRA record ID, reversed, in the
source. Two things have to hold in every dialect, and neither can be seen on a
string:

- the reversed counter orders the way the counter grows (``Z000000101`` is
  counter 35 and older than ``0100000101``, counter 36),
- the comparison is **byte-wise** even when the column carries a Czech
  collation, under which ``CH`` is a single letter sorting after ``H`` — so a
  counter reversed to ``CH00000`` would pass for newer than ``D000000``.

The live tests build the IDs as literals with a Czech collation, so they read
nothing from the seeded tables and write nothing anywhere. The condition under
test is the one the strategy builds (`abra_watermark._abra_where`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dbextractors.core.strategies.abra_watermark import _abra_where
from dbextractors.dialects.firebird import FirebirdDialect
from dbextractors.dialects.mssql import MSSQLDialect
from dbextractors.dialects.mysql import MySQLDialect
from dbextractors.dialects.postgres import PostgresDialect

#: ABRA-shaped IDs: the counter reversed, then the suffix. The last two lie
#: outside suffix 101 and must never be read.
IDS = [
    "1000000101",  # counter 1
    "Z000000101",  # counter 35 — the larger string, the smaller counter
    "0100000101",  # counter 36
    "00000HC101",  # reversed: CH00000
    "000000D101",  # reversed: D000000
    "ZZZZZZZ000",  # a record outside the counter
    "0200000102",  # another database
]

#: watermark -> what a byte comparison takes. A Czech collation would add
#: ``00000HC101`` to the first case.
CASES = {
    "D000000": set(),
    "000000Z": {"0100000101", "00000HC101", "000000D101"},
}


def _expected(watermark: str) -> set:
    """The expectation computed in Python, so it cannot drift from `CASES`."""
    return {i for i in IDS if i[7:] == "101" and i[:7][::-1] > watermark}


def test_the_cases_agree_with_a_byte_comparison() -> None:
    for watermark, expected in CASES.items():
        assert _expected(watermark) == expected


def _where(dialect, watermark: str) -> str:
    return _abra_where(SimpleNamespace(dialect=dialect, where=None), "ID", {"101": watermark})


# --- on strings ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        (
            FirebirdDialect(),
            """(SUBSTRING("ID" FROM 8) = '101' AND REVERSE(CAST(SUBSTRING("ID" FROM 1 FOR 7) """
            """AS VARCHAR(7) CHARACTER SET OCTETS)) > 'D000000')""",
        ),
        (
            MSSQLDialect(),
            "(STUFF([ID], 1, 7, '') = '101' AND REVERSE(SUBSTRING([ID], 1, 7)) "
            "COLLATE Latin1_General_BIN2 > 'D000000')",
        ),
        (
            MySQLDialect(),
            "(SUBSTRING(`ID`, 8) = '101' AND CAST(REVERSE(SUBSTRING(`ID`, 1, 7)) AS BINARY) "
            "> 'D000000')",
        ),
        (
            PostgresDialect(),
            """(substr("ID", 8) = '101' AND reverse(substr("ID", 1, 7)) COLLATE "C" > 'D000000')""",
        ),
    ],
    ids=lambda v: getattr(v, "name", "sql"),
)
def test_the_condition_as_rendered(dialect, expected) -> None:
    assert _where(dialect, "D000000") == expected


def test_several_suffixes_and_the_configured_where_are_combined() -> None:
    ctx = SimpleNamespace(dialect=PostgresDialect(), where="active = 1")

    where = _abra_where(ctx, "id", {"101": "0000001", "102": None})

    assert where == (
        """((substr("id", 8) = '101' AND reverse(substr("id", 1, 7)) COLLATE "C" > '0000001')"""
        """ OR substr("id", 8) = '102') AND (active = 1)"""
    )


def test_every_dialect_declares_the_feature() -> None:
    for dialect in (FirebirdDialect(), MSSQLDialect(), MySQLDialect(), PostgresDialect()):
        assert dialect.supports("abra_watermark"), dialect.name


# --- against live servers ---------------------------------------------------------


def _derived(name: str) -> str:
    """The IDs as a derived table whose ``ID`` column carries a Czech collation."""
    if name == "firebird":
        rows = [
            f"SELECT CAST('{i}' AS CHAR(10) CHARACTER SET WIN1250) COLLATE PXW_CSY AS ID "
            "FROM RDB$DATABASE"
            for i in IDS
        ]
    elif name == "mssql":
        rows = [f"SELECT CAST('{i}' AS CHAR(10)) COLLATE Czech_CI_AS AS ID" for i in IDS]
    else:
        rows = [f"SELECT CONVERT('{i}' USING utf8mb4) COLLATE utf8mb4_czech_ci AS ID" for i in IDS]
    return " UNION ALL ".join(rows)


_DIALECTS = {"firebird": FirebirdDialect(), "mssql": MSSQLDialect(), "mysql": MySQLDialect()}


@pytest.mark.parametrize("watermark", sorted(CASES))
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("firebird", marks=pytest.mark.needs_firebird),
        pytest.param("mssql", marks=pytest.mark.needs_mssql),
        pytest.param("mysql", marks=pytest.mark.needs_mysql),
    ],
)
def test_a_live_source_takes_exactly_the_newer_counters(name: str, watermark: str) -> None:
    from source_db import engine_for
    from sqlalchemy import text

    sql = f"SELECT ID FROM ({_derived(name)}) t WHERE {_where(_DIALECTS[name], watermark)}"
    with engine_for(name).connect() as con:
        found = {row[0].strip() for row in con.execute(text(sql))}

    assert found == CASES[watermark]


@pytest.mark.needs_pg
@pytest.mark.parametrize("watermark", sorted(CASES))
def test_a_live_postgres_takes_exactly_the_newer_counters(conn, watermark: str) -> None:
    """``cs-x-icu`` ships with every ICU-enabled PostgreSQL, the test image included."""
    derived = " UNION ALL ".join(f"""SELECT '{i}'::text COLLATE "cs-x-icu" AS "ID" """ for i in IDS)
    sql = f'SELECT "ID" FROM ({derived}) t WHERE {_where(PostgresDialect(), watermark)}'
    with conn.cursor() as cur:
        cur.execute(sql)
        found = {row[0] for row in cur.fetchall()}

    assert found == CASES[watermark]


@pytest.mark.needs_pg
def test_the_czech_collation_really_is_a_trap(conn) -> None:
    """Without this the live tests could pass for the wrong reason.

    Under the Czech collation the reversed counter ``CH00000`` sorts above
    ``D000000``; only the byte comparison the dialects pin puts it below.
    """
    with conn.cursor() as cur:
        cur.execute("""SELECT 'CH00000' COLLATE "cs-x-icu" > 'D000000'""")
        assert cur.fetchone()[0] is True
