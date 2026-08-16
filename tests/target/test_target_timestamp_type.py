"""The ``_timestamp`` column has **two different types** in production. Both have
to work.

This came out of a check on one deployment's order-items table:
`mark_deleted_in_source` wrote a hard-coded ``NOW()::text``, which on a table
with ``_timestamp TIMESTAMP`` ends in::

    column "_timestamp" is of type timestamp without time zone
    but expression is of type text

Where the two generations come from:

- **TEXT** — how this package creates the column (`MANAGED_COLUMNS`) and how it
  came out of `mage_ai.io.postgres`, because the predecessors put an ISO string
  into the frame;
- **TIMESTAMP** — how the predecessors' own add-the-column branch creates it
  (``ADD COLUMN _timestamp TIMESTAMP``, 8 of the 15 files).

A table that got the column the second way is common in production, and this
package has to be able to write into it. PostgreSQL does not convert between
``text`` and ``timestamp`` on assignment by itself, so the expression has to be
chosen from the column's actual type.
"""

from __future__ import annotations

import pytest

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import TargetRef

pytestmark = pytest.mark.needs_pg

TYPES = [
    ("text", "TEXT", "NOW()::text"),
    ("timestamp", "TIMESTAMP", "NOW()"),
]


def _target(conn, schema: str, timestamp_ddl: str) -> TargetRef:
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{schema}"."target" ('
            f"id INTEGER, _timestamp {timestamp_ddl}, "
            "_deleted_in_source BOOLEAN NOT NULL DEFAULT false)"
        )
        cur.execute(f'INSERT INTO "{schema}"."target" (id) VALUES (1), (2)')
        # The live-key table `SourceHashSnapshot` builds — id 2 is no longer in
        # the source, so it has to be marked as deleted.
        cur.execute("CREATE TEMP TABLE live_keys (id INTEGER)")
        cur.execute("INSERT INTO live_keys VALUES (1)")
    conn.commit()
    return TargetRef(schema=schema, table="target")


@pytest.mark.parametrize("name,ddl,expected", TYPES, ids=[t[0] for t in TYPES])
def test_the_timestamp_expression_follows_the_column_type(
    conn, schema, name, ddl, expected
) -> None:
    target = _target(conn, schema, ddl)
    assert target_pg.timestamp_expr(conn, target) == expected


@pytest.mark.parametrize("name,ddl,_", TYPES, ids=[t[0] for t in TYPES])
def test_mark_deleted_works_on_both_generations(conn, schema, name, ddl, _) -> None:
    """This is the failure: on `TIMESTAMP` it used to fail with a
    DatatypeMismatch."""
    target = _target(conn, schema, ddl)

    marked = target_pg.mark_deleted_in_source(conn, target, "id", "live_keys")

    assert marked == 1, f"{name}: exactly the row missing from the source should be marked"
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT id, _deleted_in_source, _timestamp FROM "{schema}"."target" ORDER BY id'
        )
        rows = cur.fetchall()

    state = {r[0]: r[1] for r in rows}
    assert state == {1: False, 2: True}, f"{name}: the wrong rows were marked"
    assert rows[1][2] is not None, f"{name}: `_timestamp` was not written on the marked row"


def test_a_missing_column_does_not_fail(conn, schema) -> None:
    """Without the column the text variant is returned — the caller will not
    write there anyway."""
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{schema}"."no_column" (id INTEGER)')
    conn.commit()

    target = TargetRef(schema=schema, table="no_column")
    assert target_pg.timestamp_expr(conn, target) == "NOW()::text"
