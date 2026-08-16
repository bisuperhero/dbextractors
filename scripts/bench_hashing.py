#!/usr/bin/env python3
"""Benchmark of `core.hashing` — ``df.apply(_row_hash, axis=1)`` against vectorisation.

The premise being checked: ``df.apply(_row_hash, axis=1)`` is a row-wise apply,
the most expensive variant there is, and the main reason the hash path runs at
roughly 15 000 rows/s.

The script does not only measure, it also **verifies equality**: if the
vectorised path diverged from the old one by even a single hash, it fails.
Speed without that check would mean nothing — ``row_hash`` is part of the
contract.

Three widths are measured because, as in `core/coerce.py`, throughput is per
row *times* column.

Usage::

    python scripts/bench_hashing.py [--rows 100000]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbextractors.core import hashing  # noqa: E402

WIDTHS = (8, 24, 48)


def build_frame(rows: int, cols: int, seed: int = 20260811) -> pd.DataFrame:
    """A frame with the mix of types the extractors really meet.

    The mix matters: were every column of the same type, the type promotion
    that ``.apply(axis=1)`` performs would go unnoticed.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(cols):
        kind = i % 4
        if kind == 0:
            data[f"int_{i}"] = rng.integers(0, 10**6, rows)
        elif kind == 1:
            data[f"text_{i}"] = [f"text {v}" for v in rng.integers(0, 1000, rows)]
        elif kind == 2:
            data[f"dec_{i}"] = rng.random(rows).round(4)
        else:
            data[f"flag_{i}"] = rng.random(rows) > 0.5
    return pd.DataFrame(data)


def legacy_path(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Copied verbatim from the predecessor extractor."""

    def _row_hash(row):
        concat = "||".join("" if pd.isna(row[col]) else str(row[col]) for col in columns)
        return hashlib.sha256(concat.encode("utf-8")).hexdigest()

    return df.apply(_row_hash, axis=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    args = parser.parse_args(argv)

    print(f"row_hash BENCHMARK — {args.rows:,} rows".replace(",", " "))
    print("=" * 74)
    print(f"{'columns':>8}  {'old (apply)':>22}  {'new (vectorised)':>22}  {'speed-up':>9}")
    print("-" * 74)

    for cols in WIDTHS:
        df = build_frame(args.rows, cols)
        names = list(df.columns)

        start = time.perf_counter()
        old = legacy_path(df, names)
        old_seconds = time.perf_counter() - start

        start = time.perf_counter()
        new = hashing.compute_row_hashes(df, names)
        new_seconds = time.perf_counter() - start

        if old.tolist() != new.tolist():
            print("\nHASHES DIFFER — the benchmark is void, the results diverged.", file=sys.stderr)
            return 1

        print(
            f"{cols:>8}  {old_seconds:>8.2f}s {args.rows / old_seconds:>11,.0f} rows/s  "
            f"{new_seconds:>8.2f}s {args.rows / new_seconds:>11,.0f} rows/s  "
            f"{old_seconds / new_seconds:>8.1f}x".replace(",", " ")
        )

    print("-" * 74)
    print("Hashes match at all three widths — otherwise the script would have failed.")
    print("(the measured ceiling for reading from a source is ~34 200 rows/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
