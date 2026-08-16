#!/usr/bin/env python3
"""What a batch size costs — time against peak memory.

For a large table ``resolve_batch_size`` **grows** the batch beyond what the
configuration asks for: it aims at ``TARGET_BATCH_MB`` per batch, so 100 000
rows turn into roughly 1 000 000 on a table estimated at 2.4 GB. The
predecessor extractors did the same, but this is why a full load of 6 M rows
holds 2.9 GB RSS — and the orchestrator runs 3 pipelines side by side.

The question this script answers: **what does that memory buy?** The same full
load is measured at several values of ``TARGET_BATCH_MB``, recording both time
and peak RSS. Without those numbers, the choice between "configuration is a
ceiling" and "leave it alone" is guesswork.

Source and target are both scratch schemas prefixed ``dbx_golden_``; the source
table is generated and dropped once the measurement is done.

Usage::

    python scripts/bench_batch_size.py [--rows 1500000] [--mb 400 150 50]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_hash_memory import RssSampler  # noqa: E402

from dbextractors.golden import scratch, session  # noqa: E402

#: The table is deliberately **wide**: the coercion layer walks column by
#: column, so throughput is measured per row *times* column, not per row.
DDL = """
    id bigint PRIMARY KEY,
    code varchar(32) NOT NULL,
    label text NOT NULL,
    description text,
    amount numeric(12,2) NOT NULL,
    quantity integer NOT NULL,
    rate numeric(5,2),
    flag boolean NOT NULL,
    phase text NOT NULL,
    created_at timestamp NOT NULL,
    event_date date NOT NULL,
    branch text NOT NULL
"""

INSERT = """
    SELECT i,
           'K' || lpad(i::text, 10, '0'),
           'item ' || i,
           CASE WHEN i %% 11 = 0 THEN NULL
                ELSE 'description of item ' || i || ' - padding to a realistic width' END,
           ((i %% 99991) / 100.0)::numeric(12,2),
           i %% 997,
           ((i %% 21) / 100.0)::numeric(5,2),
           (i %% 2 = 0),
           CASE i %% 3 WHEN 0 THEN 'new' WHEN 1 THEN 'done' ELSE 'cancelled' END,
           timestamp '2024-01-01 00:00:00' + ((i %% 500000) || ' minutes')::interval,
           date '2024-01-01' + (i %% 700),
           'S' || (i %% 20)
    FROM generate_series(1, %s) i
"""


def _config(source_schema: str, target_schema: str, batch_size: int) -> dict:
    dsn = session.resolve_dsn()
    return {
        "TABLE": {
            "source_name": "wide",
            "source_schema": source_schema,
            "output_name": "wide",
            "output_schema": target_schema,
            "empty_rows_ok": True,
        },
        "SOURCE_DB": {
            "user": os.environ["POSTGRES_USER"],
            "password": os.environ["POSTGRES_PASSWORD"],
            "database": os.environ["POSTGRES_DB"],
            "host": dsn.split("host=")[1].split()[0],
            "port": int(dsn.split("port=")[1].split()[0]),
        },
        "LOAD_SETTINGS": {
            "load_method": "full",
            "primary_column": "id",
            "batch_size": batch_size,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_500_000)
    parser.add_argument(
        "--batch-size", type=int, default=100_000, help="the value taken from the configuration"
    )
    parser.add_argument(
        "--mb",
        type=int,
        nargs="+",
        default=[400, 150, 50],
        help="TARGET_BATCH_MB values to compare",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR)
    import psycopg2

    import dbextractors
    from dbextractors.core.strategies import base

    conn = psycopg2.connect(session.resolve_dsn())
    conn.autocommit = True
    source_schema = scratch.scratch_schema_name("bench_source")
    scratch.create_schema(conn, source_schema)

    print(f"BATCH SIZE — {args.rows:,} rows, 12 columns".replace(",", " "))
    print("=" * 74)
    print(f"batch_size from configuration: {args.batch_size:,}".replace(",", " "))

    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{source_schema}"."wide" ({DDL})')
        cur.execute(f'INSERT INTO "{source_schema}"."wide" {INSERT}', (args.rows,))
        cur.execute(f'ANALYZE "{source_schema}"."wide"')

    # The estimate is taken **once** and reused to derive the batch for every
    # variant -- otherwise the measurement would also absorb the fact that
    # `COUNT(*)` takes a different time on every call.
    from sqlalchemy import create_engine

    from dbextractors.dialects.base import TableRef
    from dbextractors.dialects.postgres import PostgresDialect

    cfg = _config(source_schema, "x", args.batch_size)["SOURCE_DB"]
    dialect = PostgresDialect()
    engine = create_engine(
        dialect.build_conn_str(cfg, cfg["host"], cfg["port"]),
    )
    estimate = dialect.estimate_size(engine, TableRef(name="wide", schema=source_schema))
    engine.dispose()
    print(f"source estimate: {estimate.rows:,} rows / {estimate.size_mb:,.0f} MB".replace(",", " "))

    original_target_mb = base.TARGET_BATCH_MB
    results: List[dict] = []
    try:
        header = (
            f"{'TARGET_BATCH_MB':>16}{'actual batch':>17}"
            f"{'time':>10}{'peak RSS':>13}{'rows/s':>10}"
        )
        print(f"\n{header}")
        for mb in args.mb:
            base.TARGET_BATCH_MB = mb
            target_schema = scratch.scratch_schema_name("bench_target")
            scratch.create_schema(conn, target_schema)
            sampler = RssSampler(0.2)
            sampler.start()
            start = time.perf_counter()
            try:
                status = dbextractors.run(
                    _config(source_schema, target_schema, args.batch_size), dialect="postgres"
                )
            finally:
                peak = sampler.stop()
                duration = time.perf_counter() - start
                scratch.drop_schema(conn, target_schema)

            written = int(status["rows_written"].sum())
            actual_batch = base.resolve_batch_size({"batch_size": args.batch_size}, {}, estimate)
            print(
                f"{mb:>16}{actual_batch:>17,}{duration:>9.1f}s{peak:>11.0f} MB"
                f"{written / duration:>10,.0f}".replace(",", " ")
            )
            results.append(
                {
                    "mb": mb,
                    "batch": actual_batch,
                    "s": duration,
                    "rss": peak,
                    "rps": written / duration,
                }
            )
    finally:
        base.TARGET_BATCH_MB = original_target_mb
        scratch.drop_schema(conn, source_schema)
        conn.close()

    if len(results) > 1:
        first, last = results[0], results[-1]
        print(
            f"\nFrom {first['mb']} MB to {last['mb']} MB: memory "
            f"{first['rss'] / max(last['rss'], 1):.1f}x lower, time "
            f"{last['s'] / max(first['s'], 0.001):.2f}x."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
