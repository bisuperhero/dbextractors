#!/usr/bin/env python3
"""Where the time goes on tables with large text columns.

The golden test showed the new path to be **3.2x slower** than the old one on a
table of 51 174 rows and 367 MB (~12 kB per row, dominated by two large text
columns), even though it is 4.3x faster on a narrow table. This script measures
the four suspects instead of leaving them to speculation.

No source database is needed - the batch is synthesised with the same profile.
Writes go exclusively into a schema prefixed with ``dbx_golden_``.

Usage::

    python scripts/profile_heavy.py [--rows 5000] [--kb 12]
"""

from __future__ import annotations

import argparse
import csv
import random
import string
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbextractors.core import target_pg  # noqa: E402
from dbextractors.core.strategies.base import TargetRef  # noqa: E402

SCHEMA_PREFIX = "dbx_golden_profile"


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


def build_batch(rows: int, kb: int, seed: int = 20260811) -> pd.DataFrame:
    """A batch with the heavy profile: a few keys and two large text columns."""
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + ' ,."\n'
    blob = "".join(rng.choice(alphabet) for _ in range(kb * 1024))

    return pd.DataFrame(
        {
            "id": range(1, rows + 1),
            "location_id": [rng.randint(1, 10**6) for _ in range(rows)],
            "_type": [rng.choice(["insert", "update", "delete"]) for _ in range(rows)],
            "old_data": [blob[i % 97 :] + blob[: i % 97] for i in range(rows)],
            "new_data": [blob[i % 89 :] + blob[: i % 89] for i in range(rows)],
            "row_hash": [f"{rng.getrandbits(256):064x}" for _ in range(rows)],
            "_timestamp": "2026-08-11T00:00:00+00:00",
            "_deleted_in_source": False,
        }
    )


def _timed(label: str, fn) -> float:
    start = time.perf_counter()
    fn()
    delta = time.perf_counter() - start
    print(f"   {label:<46}{delta:>8.2f}s")
    return delta


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--kb", type=int, default=12)
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")

    import psycopg2

    from dbextractors.golden import scratch, session

    df = build_batch(args.rows, args.kb)
    columns = list(df.columns)
    size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"HEAVY BATCH PROFILE - {args.rows:,} rows, {size_mb:,.0f} MB".replace(",", " "))
    print("=" * 62)

    conn = psycopg2.connect(session.resolve_dsn())
    conn.autocommit = False
    schema = scratch.scratch_schema_name("profile")
    scratch.create_schema(conn, schema)
    conn.commit()
    target = TargetRef(schema=schema, table="target")

    try:
        target_pg.create_table(conn, target, df, columns=columns)
        conn.commit()

        print("\n1) QUOTE_ALL vs QUOTE_MINIMAL - serialisation only, no database")
        for label, quoting in (
            ("QUOTE_MINIMAL (as mage does)", csv.QUOTE_MINIMAL),
            ("QUOTE_ALL", csv.QUOTE_ALL),
        ):
            _timed(
                label,
                lambda q=quoting: df.to_csv(header=False, index=False, na_rep="", quoting=q),
            )
        minimal = len(df.to_csv(header=False, index=False, na_rep="", quoting=csv.QUOTE_MINIMAL))
        quoted_all = len(df.to_csv(header=False, index=False, na_rep="", quoting=csv.QUOTE_ALL))
        print(
            f"   payload: {minimal / 1e6:.0f} MB -> {quoted_all / 1e6:.0f} MB "
            f"({quoted_all / minimal:.2f}x)"
        )

        print("\n2) COPY into the database")
        for chunk in (1000, 10_000):
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {target_pg.qualify(target)}")
            conn.commit()
            _timed(
                f"copy_from_stdin, chunk_rows={chunk:,}".replace(",", " "),
                lambda c=chunk: target_pg.copy_from_stdin(conn, target, df, columns, chunk_rows=c),
            )
            conn.commit()

        print("\n3) SeenKeys dedup - work the old path never did at all")
        with target_pg.SeenKeys(conn, "id") as seen:
            _timed("filter_new over the whole batch", lambda: seen.filter_new(df))
        conn.commit()

        print("\n4) Conversion layer")
        from dbextractors.core import coerce

        _timed("normalize_carriage_returns", lambda: coerce.normalize_carriage_returns(df.copy()))
        _timed(
            "fix_column_values (everything TEXT)",
            lambda: coerce.fix_column_values(df.copy(), {}),
        )
        _timed("_serialize_objects", lambda: target_pg._serialize_objects(df.copy()))
    finally:
        conn.rollback()
        scratch.drop_schema(conn, schema)
        conn.commit()
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
