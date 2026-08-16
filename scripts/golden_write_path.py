#!/usr/bin/env python3
"""Golden test of the write path: a real table there and back through `core.target_pg`.

The rule it serves: *a module is not done until a golden test proves that the
legacy and the new component produce an identical target table — including column
names and their order, types and nullability.*

**What this script verifies and what it does not.** Comparing the legacy and the
new *extractor* end to end needs the source databases (MySQL, MSSQL, Firebird
behind a tunnel), which this machine has no access to — and they must not be run
against production without an explicit instruction. So the part that can be
verified is: the **write path**.

A real target table is read, pushed through the new `prepare_export_df` +
`copy_from_stdin` into a working schema, and the result is compared with the
original by the golden test on all five levels. If the new write path changed
names, order, types or values, that shows up on real data rather than on invented
data.

Production schemas are **read only**; writes go exclusively into a schema prefixed
with ``dbx_golden_``, which is dropped after the run (``--keep`` leaves it there).

Usage::

    python scripts/golden_write_path.py                       # the default trio
    python scripts/golden_write_path.py raw_sales.invoices
    python scripts/golden_write_path.py --limit 50000 --keep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from dbextractors.core import target_pg  # noqa: E402
from dbextractors.core.strategies.base import TargetRef  # noqa: E402
from dbextractors.golden import compare, scratch, session  # noqa: E402
from dbextractors.golden.model import Relation  # noqa: E402

#: 0 = no limit. The golden test only makes sense on the **whole** table — with a
#: cut-off, level 1 would report a difference in row counts and every checksum
#: would diverge.
DEFAULT_LIMIT = 0


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_ref(text: str) -> TargetRef:
    schema, _, table = text.partition(".")
    if not schema or not table:
        raise SystemExit(f"Expected 'schema.table', got {text!r}.")
    return TargetRef(schema=schema, table=table)


def read_source(conn, ref: TargetRef, limit: int) -> Tuple[pd.DataFrame, List[str], dict]:
    """Reads the table and its schema. **SELECT only.**"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (ref.schema, ref.table),
        )
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"Table {target_pg.qualify(ref)} does not exist.")

    columns = [r[0] for r in rows]
    types = {r[0]: str(r[1]).lower() for r in rows}
    cols_sql = ", ".join(target_pg.quote_ident(c) for c in columns)

    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {cols_sql} FROM {target_pg.qualify(ref)}{limit_sql}")
        data = cur.fetchall()

    return pd.DataFrame(data, columns=columns), columns, types


#: Types that `prepare_export_df` handles as date-like.
DATE_LIKE_TYPES = ("date", "timestamp")


def transfer(conn, source: TargetRef, target: TargetRef, limit: int) -> Tuple[int, float]:
    """Pushes the table through the whole write path. Returns (rows, seconds).

    It follows the **same order of steps as a real run**: first `load_pg_schema`
    derives the types from the target, then `prepare_export_df` converts the batch
    and only then does `copy_from_stdin` write. A shortcut straight to `COPY` would
    not work — a nullable integer column arrives from pandas as ``float64`` and
    ``COPY`` rejects it with "invalid input syntax for type integer: 1.0". That is
    exactly why the conversion layer exists.
    """
    df, columns, types = read_source(conn, source, limit)

    overwrite: dict = {}
    integer_mapping: dict = {}
    schema_info = target_pg.load_pg_schema(conn, source, overwrite, integer_mapping)
    date_like = {c for c, t in types.items() if any(d in t for d in DATE_LIKE_TYPES)}

    start = time.perf_counter()
    export_df = target_pg.prepare_export_df(
        batch_df=df,
        meta={},
        all_columns=columns,
        date_like_columns=date_like,
        integer_columns_mapping=integer_mapping,
        overwrite_types=overwrite,
        orig_type_map=types,
        db_integer_cols=sorted(schema_info.integer_columns),
    )
    export_df = target_pg.align_df_columns_to_db(export_df, columns)

    # The shadow is created with the **same types the original has** — otherwise
    # the comparison would suddenly cover both the write path and type inference,
    # and there would be no telling which of the two diverged.
    shadow = target_pg.create_shadow_table(conn, target, df=export_df, overwrite_types=dict(types))
    written = target_pg.copy_from_stdin(conn, shadow, export_df, columns)
    moved = target_pg.swap_shadow_table(conn, shadow, target)
    conn.commit()
    seconds = time.perf_counter() - start

    if moved != written:
        raise SystemExit(f"COPY wrote {written}, the swap moved over {moved}.")
    return written, seconds


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="*", default=None, help="'schema.table', may be repeated")
    parser.add_argument("--dsn")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="0 = the whole table (default)"
    )
    parser.add_argument("--keep", action="store_true", help="keep the working schema")
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    # No default table list on purpose. This used to carry a trio of tables from
    # one particular deployment, which is meaningless anywhere else — and a
    # default that points at schemas which do not exist fails in a far more
    # confusing way than being asked for an argument.
    tables = list(args.tables)
    if not tables:
        parser.error(
            "give at least one schema.table to compare, e.g. "
            "raw_source.orders raw_source.customers. Pick a small table with "
            "unusual types, a medium one with reserved-word column names, and a "
            "large one — that trio exercises the whole write path."
        )

    import psycopg2

    dsn = session.resolve_dsn(args.dsn)
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    workspace = scratch.scratch_schema_name("writepath")
    scratch.create_schema(conn, workspace)
    conn.commit()

    print("GOLDEN TEST OF THE WRITE PATH")
    print("=" * 70)
    print(f"working schema: {workspace}")
    print(f"row cap:        {args.limit:,}" if args.limit else "row cap:        no limit")
    print()

    verdicts = []
    try:
        for name in tables:
            source = parse_ref(name)
            target = TargetRef(schema=workspace, table=source.table)

            rows, seconds = transfer(conn, source, target, args.limit)
            speed = rows / seconds if seconds else float("inf")
            print(f"{name}")
            print(f"  transferred {rows:,} rows in {seconds:.2f}s ({speed:,.0f} rows/s)")

            with session.connect(dsn, read_only=True) as ro:
                report = compare.compare_tables(
                    ro,
                    Relation(source.schema, source.table),
                    Relation(target.schema, target.table),
                )
            for level in report.levels:
                if level.skipped:
                    mark, detail = "--  ", "skipped"
                elif level.passed:
                    mark = "ok  "
                    detail = "approximate" if level.approximate else ""
                else:
                    mark = "ERROR"
                    detail = f"{len(level.differences)} differences"
                print(f"  {mark} {int(level.level)}. {level.title:<34} {detail}")
                for note in level.notes[:3]:
                    print(f"        {note}")
                for difference in level.differences[:5]:
                    print(f"        {difference}")
            verdicts.append((name, report.verdict))
            print()
    finally:
        conn.rollback()
        if args.keep:
            print(f"Working schema {workspace} is kept.")
        else:
            scratch.drop_schema(conn, workspace)
            conn.commit()
        conn.close()

    print("=" * 70)
    for name, verdict in verdicts:
        print(f"  {verdict:<7} {name}")
    from dbextractors.golden.model import VERDICT_MATCH

    return 0 if all(v == VERDICT_MATCH for _, v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
