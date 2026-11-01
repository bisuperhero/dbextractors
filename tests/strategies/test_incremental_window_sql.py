"""The incremental window expression, per dialect.

`NULLIF(col, '0000-00-00 00:00:00')` exists for MySQL's zero date. On PostgreSQL
the literal is parsed against the column type and rejected (``invalid input
syntax for type timestamp``) whether or not any row holds it, so it must never
be emitted for another dialect. These tests render the condition for each
dialect without a database; the two ``needs_pg`` ones run it on a real server.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from dbextractors.core.strategies.base import LoadContext
from dbextractors.core.strategies.incremental import (
    IncrementalStrategy,
    compute_incremental_cutoff,
)
from dbextractors.dialects.base import TableRef
from dbextractors.dialects.firebird import FirebirdDialect
from dbextractors.dialects.mssql import MSSQLDialect
from dbextractors.dialects.mysql import MySQLDialect
from dbextractors.dialects.postgres import PostgresDialect
from fakes import make_columns

CUTOFF = date(2026, 10, 25)
ZERO = "0000-00-00 00:00:00"

UPDATED_AT = {"primary_column": "id", "updated_at_column": "upd", "created_at_column": "crt"}
LEGACY = {
    "primary_column": "id",
    "incremental_date_column": "upd",
    "incremental_date_column_fallback": "crt",
}


def _where(dialect, settings, where=None) -> str:
    ctx = LoadContext(
        dialect=dialect,
        engine=None,
        source=TableRef(name="t"),
        target=None,  # type: ignore[arg-type]
        columns=make_columns([("id", "int", "INTEGER")]),
        settings=dict(settings),
        where=where,
    )
    return IncrementalStrategy()._build_where(ctx, CUTOFF)


def test_mysql_keeps_the_zero_date_nullif() -> None:
    assert _where(MySQLDialect(), UPDATED_AT) == (
        f"COALESCE(NULLIF(`upd`, '{ZERO}'), `crt`) >= '2026-10-25 00:00:00'"
    )


@pytest.mark.parametrize(
    ("dialect", "quote"),
    [
        (PostgresDialect(), '"'),
        (MSSQLDialect(), None),
        (FirebirdDialect(), '"'),
    ],
    ids=["postgres", "mssql", "firebird"],
)
def test_other_dialects_never_emit_the_zero_date(dialect, quote) -> None:
    where = _where(dialect, UPDATED_AT)

    assert "NULLIF" not in where
    assert ZERO not in where
    assert where.startswith("COALESCE(")
    assert where.endswith(" >= '2026-10-25 00:00:00'")


def test_postgres_window_text() -> None:
    assert _where(PostgresDialect(), UPDATED_AT) == (
        'COALESCE("upd", "crt") >= \'2026-10-25 00:00:00\''
    )


def test_without_a_created_column_the_window_is_the_updated_column_alone() -> None:
    settings = {**UPDATED_AT, "created_at_column": None}

    assert _where(MySQLDialect(), settings) == "`upd` >= '2026-10-25 00:00:00'"
    assert _where(PostgresDialect(), settings) == "\"upd\" >= '2026-10-25 00:00:00'"


def test_the_configured_where_is_appended_on_every_dialect() -> None:
    for dialect in (MySQLDialect(), PostgresDialect()):
        assert _where(dialect, UPDATED_AT, where="x = 1").endswith(" AND (x = 1)")


def test_the_legacy_window_keeps_coalesce_zero_where_it_works() -> None:
    assert _where(MySQLDialect(), LEGACY) == (
        "(COALESCE(`upd`, 0) >= '2026-10-25' OR COALESCE(`crt`, 0) >= '2026-10-25')"
    )
    assert "COALESCE(" in _where(MSSQLDialect(), LEGACY)


def test_the_legacy_window_on_postgres_has_no_integer_coalesce() -> None:
    """``COALESCE(timestamp_col, 0)`` does not type-check on PostgreSQL; a plain
    comparison is already false for NULL, which is all the ``0`` was for."""
    assert _where(PostgresDialect(), LEGACY) == (
        "(\"upd\" >= '2026-10-25' OR \"crt\" >= '2026-10-25')"
    )


# --- days_back and incremental_lookback_hours feed the same cutoff ---------


def test_the_cutoff_is_unchanged_by_the_dialect_split() -> None:
    """``days_back`` and ``incremental_lookback_hours`` are resolved before the
    dialect is asked anything; the rendered literal is just that date."""
    now = datetime.now()

    days, _, hours = compute_incremental_cutoff({"days_back": 7}, {})
    assert (days, hours) == ((now - timedelta(days=7)).date(), None)

    lookback, _, hours = compute_incremental_cutoff({"incremental_lookback_hours": 6}, {})
    assert (lookback, hours) == ((now - timedelta(hours=6)).date(), 6)

    runtime, _, hours = compute_incremental_cutoff(
        {"days_back": 7}, {"incremental_lookback_hours": 30}
    )
    assert (runtime, hours) == ((now - timedelta(hours=30)).date(), 30)

    for dialect in (MySQLDialect(), PostgresDialect()):
        ctx = LoadContext(
            dialect=dialect,
            engine=None,
            source=TableRef(name="t"),
            target=None,  # type: ignore[arg-type]
            columns=make_columns([("id", "int", "INTEGER")]),
            settings=dict(UPDATED_AT),
        )
        assert f">= '{days:%Y-%m-%d} 00:00:00'" in IncrementalStrategy()._build_where(ctx, days)


# --- On a real PostgreSQL --------------------------------------------------


@pytest.mark.needs_pg
@pytest.mark.parametrize("settings", [UPDATED_AT, LEGACY], ids=["updated_at", "legacy_date_column"])
def test_the_window_runs_on_postgres(conn, schema, settings) -> None:
    """The condition has to be accepted by the server and select the right rows."""
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{schema}".t (id int, upd timestamp, crt timestamp)')
        cur.execute(
            f'INSERT INTO "{schema}".t VALUES '
            "(1, '2026-10-30 10:00', '2026-01-01'),"  # updated recently
            "(2, NULL, '2026-10-30 10:00'),"  # never updated, created recently
            "(3, '2026-01-01', '2026-01-01'),"  # old
            "(4, NULL, NULL)"  # nothing to go on
        )
        where = _where(PostgresDialect(), settings)
        cur.execute(f'SELECT id FROM "{schema}".t WHERE {where} ORDER BY id')
        got = [r[0] for r in cur.fetchall()]

    # The legacy shape ORs the two columns; the updated_at shape falls back to
    # `crt` only when `upd` is NULL. Both select 1 and 2 here, and neither 3 or 4.
    assert got == [1, 2]
