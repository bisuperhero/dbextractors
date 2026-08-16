"""`dbx_source_fingerprints` is the contract the multi-database selection rests on.

The caller's own helper that decides which source databases to extract was
rewritten to read this package's bookkeeping instead of the state table the old
extractor filled in. That turned this table into a **public interface** — while
nobody read it, its shape could be changed freely; now a change would silently
break the database selection for every pipeline of that deployment.

What the caller needs:

```sql
SELECT source_label, updated_at,
       updated_at < NOW() - INTERVAL '30 days' AS recheck_due
FROM <schema>.dbx_source_fingerprints
WHERE table_name = %s
```

These tests guard exactly that and nothing more — the table name, three columns
and their semantics. Not the contents of the fingerprint: that is opaque to the
caller.
"""

from __future__ import annotations

import time

import pytest

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import TargetRef

pytestmark = pytest.mark.needs_pg

#: Literally what the caller has in its own state-table constant.
TABLE_NAME = "dbx_source_fingerprints"


def _target(schema: str, table: str = "orders") -> TargetRef:
    return TargetRef(schema=schema, table=table)


def test_the_table_is_named_the_way_the_caller_looks_it_up(conn, schema) -> None:
    """A rename would mean an empty state at the caller — and an empty state
    reads as "nothing is loaded", so every database would be pulled again."""
    target_pg.ensure_fingerprint_store(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, TABLE_NAME),
        )
        assert cur.fetchone() is not None


def test_it_has_the_columns_the_caller_reads(conn, schema) -> None:
    target_pg.ensure_fingerprint_store(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, TABLE_NAME),
        )
        columns = {r[0] for r in cur.fetchall()}

    assert {"table_name", "source_label", "updated_at"} <= columns


def test_updated_at_moves_only_after_a_successful_write(conn, schema) -> None:
    """`updated_at` stands in for the earlier `last_full_load_at`.

    The caller decides by it whether to skip a closed database. Were it to move
    on a failure too, a database whose data never reached the target would be
    skipped.
    """
    target = _target(schema)
    target_pg.write_fingerprint(conn, target, "erp_2025", "fingerprint-1")
    conn.commit()
    first = _updated_at(conn, schema, target.table, "erp_2025")

    time.sleep(0.01)
    target_pg.write_fingerprint(conn, target, "erp_2025", "fingerprint-2")
    conn.commit()
    second = _updated_at(conn, schema, target.table, "erp_2025")

    assert second > first


def test_the_state_is_kept_per_table_and_database_pair(conn, schema) -> None:
    """The caller asks `WHERE table_name = %s` and expects a row per database.

    Were the key just `table_name`, the databases would overwrite each other's
    state and the skipping would follow whichever finished last.
    """
    target_pg.write_fingerprint(conn, _target(schema, "orders"), "erp_a", "a")
    target_pg.write_fingerprint(conn, _target(schema, "orders"), "erp_b", "b")
    target_pg.write_fingerprint(conn, _target(schema, "invoices"), "erp_a", "c")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            f'SELECT source_label FROM "{schema}"."{TABLE_NAME}" '
            "WHERE table_name = %s ORDER BY source_label",
            ("orders",),
        )
        assert [r[0] for r in cur.fetchall()] == ["erp_a", "erp_b"]


def _updated_at(conn, schema: str, table: str, label: str):
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT updated_at FROM "{schema}"."{TABLE_NAME}" '
            "WHERE table_name = %s AND source_label = %s",
            (table, label),
        )
        return cur.fetchone()[0]
