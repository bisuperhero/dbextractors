#!/usr/bin/env python3
"""Measures the write path: what today's ``loader.export()`` does and what the new one does.

Rows per second on the old path versus the new one. Two things are measured
separately, because each answers a different question:

**1. Overhead around COPY (no database).** ``mage_ai.io.postgres.Postgres.export()``
does not send INSERTs on a full load — it goes through ``COPY … FROM STDIN``,
exactly like the new path. The difference is in what it does around that
**on every batch**: ``infer_dtypes`` (pandas ``infer_dtype`` over every column),
``get_type`` (``min()``/``max()`` for integer columns), ``clean_df_for_export``
(a ``df.copy()`` plus a remap of every column), then another ``df.copy()``
inside ``upload_dataframe``, ``.apply(serialize_obj)`` over every object column
and ``replace({np.NaN: None})`` over the whole frame. Only then ``to_csv``.

This part can be measured without a database — a cursor that swallows
``copy_expert`` is enough. It only runs where ``mage_ai`` exists (that is, in
the production image, `make bench-write`).

**2. Real COPY throughput (with a database).** How many rows per second the new
path pushes into a live PostgreSQL, serialisation included. Needs ``POSTGRES_*``
or ``--dsn`` and writes **exclusively** into a schema prefixed ``dbx_golden_``.

Usage::

    python scripts/bench_write_path.py                # both, whatever is available
    python scripts/bench_write_path.py --rows 200000
    python scripts/bench_write_path.py --no-db        # part 1 only
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbextractors.core import target_pg  # noqa: E402
from dbextractors.core.strategies.base import TargetRef  # noqa: E402

#: The only schema that may be written to. The same safeguard as in the golden test.
BENCH_SCHEMA = "dbx_golden_bench_write"

REPEATS = 3


def load_dotenv(path: Path) -> None:
    """Loads ``.env`` if it exists. Same as ``tests/golden/conftest.py``."""
    if not path.exists():
        return
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_batch(rows: int, seed: int = 20260811) -> pd.DataFrame:
    """A batch that looks like a real one — not all numbers.

    The profile is read off real target tables: text dominates, plus integers,
    decimals, booleans, a date held as a string and a handful of values that
    traditionally break the write path (a quote, a tab, a CR, an empty string,
    NULL).
    """
    rng = np.random.default_rng(seed)
    words = ["".join(rng.choice(list(string.ascii_lowercase), 8)) for _ in range(512)]

    text = pd.Series(rng.choice(words, rows))
    # Every two-hundredth row gets a character that CSV has to escape.
    nasty = np.array(
        ['quote " in the middle', "tab\tinside", "CR\r\nbreak", "", "multibyte ÄÖÜ ß æøå"]
    )
    idx = rng.choice(rows, size=max(1, rows // 200), replace=False)
    text.iloc[idx] = rng.choice(nasty, size=len(idx))

    whole_numbers = pd.Series(rng.integers(-2_000_000, 2_000_000, rows))
    decimals = pd.Series(rng.normal(1000, 250, rows).round(4))
    booleans = pd.Series(rng.random(rows) > 0.5)
    dates = pd.Series(
        pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 2200, rows), unit="D")
    ).dt.strftime("%Y-%m-%d")

    df = pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "label": text,
            "description": text.str.repeat(3),
            "quantity": whole_numbers,
            "price": decimals,
            "is_active": booleans,
            "sold_on": dates,
            "row_hash": pd.Series(rng.integers(0, 16**8, rows)).map(lambda v: f"{v:032x}"),
            "_timestamp": "2026-08-11T00:00:00+00:00",
            "_deleted_in_source": False,
        }
    )
    # Sparse NULLs, so the missing-value branch gets measured too.
    null_idx = rng.choice(rows, size=rows // 50, replace=False)
    df.loc[null_idx, "description"] = None
    return df


class _SwallowingCursor:
    """A cursor that swallows COPY. What is measured is data preparation, not the network."""

    def __enter__(self) -> _SwallowingCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, *args: object, **kwargs: object) -> None:
        return None

    def copy_expert(self, sql: str, file: object, size: int = 8192) -> None:
        while file.read(1 << 20):  # type: ignore[attr-defined]
            pass

    def close(self) -> None:
        return None


def _timeit(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


# --- Part 1: overhead around COPY, without a database -----------------------


def bench_cpu(df: pd.DataFrame) -> Optional[List[Tuple[str, float]]]:
    try:
        from mage_ai.io.export_utils import clean_df_for_export, infer_dtypes
        from mage_ai.io.postgres import Postgres
    except ImportError:
        return None

    # No connection — only data preparation is measured. `ssh_tunnel` has to
    # exist, otherwise `BaseSQLConnection.__del__` makes noise on stderr when
    # the process shuts down.
    loader = Postgres.__new__(Postgres)
    loader.ssh_tunnel = None
    target = TargetRef(schema=BENCH_SCHEMA, table="bench")
    columns = list(df.columns)

    def old() -> None:
        frame = df.copy()
        dtypes = infer_dtypes(frame)
        frame = clean_df_for_export(frame, loader.clean, dtypes)
        dtypes = infer_dtypes(frame)
        db_dtypes = {col: loader.get_type(frame[col], dtypes[col]) for col in dtypes}
        loader.upload_dataframe(
            _SwallowingCursor(), frame, db_dtypes, dtypes, "bench.bench", buffer=_StringSink()
        )

    def new() -> None:
        target_pg.copy_from_stdin(_FakeConn(), target, df, columns)

    return [("old (mage export)", _timeit(old)), ("new (target_pg)", _timeit(new))]


class _StringSink:
    """A stand-in for ``StringIO`` that throws the data away. Time is measured, not memory."""

    def write(self, data: str) -> int:
        return len(data)

    def seek(self, *args: object) -> int:
        return 0

    def read(self, size: int = -1) -> str:
        return ""


class _FakeConn:
    def cursor(self) -> _SwallowingCursor:
        return _SwallowingCursor()


# --- Part 2: real COPY throughput -------------------------------------------


def bench_db(df: pd.DataFrame, dsn: Optional[str]) -> Optional[Tuple[float, int]]:
    try:
        import psycopg2

        from dbextractors.golden.session import resolve_dsn
    except ImportError:
        return None

    try:
        resolved = resolve_dsn(dsn)
    except Exception as err:
        print(f"  (part 2 skipped: {err})", file=sys.stderr)
        return None

    conn = psycopg2.connect(resolved)
    conn.autocommit = True
    target = TargetRef(schema=BENCH_SCHEMA, table="bench")
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {target_pg.quote_ident(BENCH_SCHEMA)} CASCADE")
        target_pg.create_table(conn, target, df)

        best = float("inf")
        for _ in range(REPEATS):
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {target_pg.qualify(target)}")
            start = time.perf_counter()
            written = target_pg.copy_from_stdin(conn, target, df, list(df.columns))
            best = min(best, time.perf_counter() - start)

        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {target_pg.qualify(target)}")
            in_db = cur.fetchone()[0]
        if in_db != written:
            raise SystemExit(
                f"COPY wrote {written}, the table holds {in_db} — the benchmark is void."
            )

        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {target_pg.quote_ident(BENCH_SCHEMA)} CASCADE")
        return best, in_db
    finally:
        conn.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument(
        "--dsn", help="DSN of the target PostgreSQL; otherwise POSTGRES_* from the environment"
    )
    parser.add_argument("--no-db", action="store_true", help="part 1 only")
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    df = build_batch(args.rows)
    print(f"WRITE PATH BENCHMARK — {args.rows:,} rows, {len(df.columns)} columns")
    print(f"best of {REPEATS} runs")
    print("=" * 66)

    print("\n1) OVERHEAD AROUND COPY (no database)")
    cpu = bench_cpu(df)
    if cpu is None:
        print("   mage_ai is not available — run `make bench-write` (production image).")
    else:
        (_, old), (_, new) = cpu
        print(f"   {'path':<26}{'time':>10}{'rows/s':>14}")
        for label, seconds in cpu:
            print(f"   {label:<26}{seconds:>9.3f}s{args.rows / seconds:>14,.0f}")
        print(f"   the new path is {old / new:.2f}x faster")

    if not args.no_db:
        print("\n2) REAL COPY THROUGHPUT (live PostgreSQL)")
        db = bench_db(df, args.dsn)
        if db is None:
            print("   database unavailable — skipped.")
        else:
            seconds, rows = db
            print(f"   COPY into the target:  {seconds:.3f}s   {rows / seconds:,.0f} rows/s")
            print("   (the measured ceiling for reading from a source is ~34 200 rows/s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
