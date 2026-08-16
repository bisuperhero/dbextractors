"""Benchmark of `core.coerce` — the legacy implementation against the new one.

The requirement behind this script: show a benchmark of the old versus the new
version over a DataFrame of 100 000 rows with a representative set of columns.
A number, not a claim.

The legacy implementation is driven through `tests/reference_oracle.py`, because
the predecessor code cannot simply be imported. Those sources are not part of
this repository, so outside it the comparison column is dropped and only the
current implementation is measured — see the note printed at the top of a run.

Usage:
    python scripts/bench_coerce.py [--rows 100000] [--repeat 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import reference_oracle as ro  # noqa: E402
from dbextractors.core import coerce  # noqa: E402

#: Oracle key of the legacy source the old implementation is taken from. It is
#: the variant that won the merge review for most functions of this module.
REFERENCE_FILE = "A-mysql"


def make_frame(rows: int, seed: int = 20260807) -> pd.DataFrame:
    """A representative batch — the mix of types the extractors really meet."""
    rng = np.random.default_rng(seed)
    idx = np.arange(rows)

    def sprinkle(values: list, nulls_every: int) -> list:
        return [None if i % nulls_every == 0 else v for i, v in enumerate(values)]

    return pd.DataFrame(
        {
            "id": idx,
            "cislo_text": sprinkle([str(v) for v in rng.integers(0, 10**6, rows)], 17),
            "castka": sprinkle([f"{v:.2f}" for v in rng.random(rows) * 1000], 23),
            "datum": sprinkle(
                [
                    "0000-00-00" if i % 50 == 0 else f"2026-{1 + i % 12:02d}-{1 + i % 28:02d}"
                    for i in range(rows)
                ],
                31,
            ),
            "cas": sprinkle([f"{i % 24:02d}:{i % 60:02d}:{i % 60:02d}" for i in range(rows)], 29),
            "vlajka": sprinkle(["true" if i % 2 else "false" for i in range(rows)], 19),
            "poznamka": sprinkle(
                [f"row {i} with\r\nbreak" if i % 7 == 0 else f"text {i}" for i in range(rows)], 13
            ),
            "bajty": sprinkle([f"bytes {i}".encode() for i in range(rows)], 11),
        }
    )


def _widen(frame: pd.DataFrame, integer_columns: list, mapping: dict, factor: int) -> tuple:
    """Repeat the column set so behaviour on a **wide** table can be measured.

    `coerce` walks the frame column by column, so throughput is expressed in
    rows *times* columns. A table with 43 columns is therefore an order of
    magnitude slower than an eight-column one at the same row count — and real
    target tables have 43 and 56 columns.
    """
    data = {}
    widened_integer_columns = []
    widened_mapping = {}
    for i in range(factor):
        for name in frame.columns:
            widened_name = name if i == 0 else f"{name}_{i}"
            data[widened_name] = frame[name]
            if name in integer_columns:
                widened_integer_columns.append(widened_name)
            if name in mapping:
                widened_mapping[widened_name] = mapping[name]
    return pd.DataFrame(data), widened_integer_columns, widened_mapping


def timed(label: str, fn, frame: pd.DataFrame, repeat: int) -> float:
    """Best of `repeat` runs. Best, not mean — less noise from the GC."""
    best = float("inf")
    for _ in range(repeat):
        working = frame.copy(deep=True)
        start = time.perf_counter()
        fn(working)
        best = min(best, time.perf_counter() - start)
    del label
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--width",
        type=int,
        default=1,
        help="how many times to repeat the column set (simulates a wide table)",
    )
    args = parser.parse_args()

    frame = make_frame(args.rows)
    integer_columns = ["cislo_text", "id"]
    mapping = {
        "cislo_text": "INTEGER",
        "datum": "DATE",
        "cas": "TIME",
        "vlajka": "BOOLEAN",
        "poznamka": "TEXT",
    }
    if args.width > 1:
        frame, integer_columns, mapping = _widen(frame, integer_columns, mapping, args.width)

    cases = [
        (
            "decode_bytes_columns",
            lambda df: ro.get(REFERENCE_FILE, "decode_bytes_columns")(df),
            lambda df: coerce.decode_bytes_columns(df, ("utf-8",)),
        ),
        (
            "_normalize_carriage_returns",
            lambda df: ro.get(REFERENCE_FILE, "_normalize_carriage_returns")(df),
            coerce.normalize_carriage_returns,
        ),
        (
            "fix_invalid_dates",
            lambda df: ro.get(REFERENCE_FILE, "fix_invalid_dates")(df),
            coerce.fix_invalid_dates,
        ),
        (
            "sanitize_integer_columns",
            lambda df: ro.get(REFERENCE_FILE, "sanitize_integer_columns")(df, integer_columns),
            lambda df: coerce.sanitize_integer_columns(df, integer_columns),
        ),
        (
            "replace_pdna_with_none",
            lambda df: ro.get(REFERENCE_FILE, "replace_pdna_with_none")(df),
            coerce.replace_pdna_with_none,
        ),
        (
            "fix_column_values",
            lambda df: ro.get(REFERENCE_FILE, "fix_column_values")(df, mapping),
            lambda df: coerce.fix_column_values(df, mapping),
        ),
    ]

    # The legacy side needs the predecessor's actual code, which lives outside
    # this repository. The recorded oracle cannot stand in for it here: a
    # recording answers one input, and a benchmark's whole point is to generate
    # its own — so in `replay` mode every call would miss. Rather than fail, the
    # comparison column is dropped and the new implementation is measured on its
    # own, which is the number the performance work is steered by anyway.
    compare = ro.MODE == "live"
    if not compare:
        print("Legacy side unavailable (the oracle is in replay mode) — measuring")
        print("the current implementation only. The comparison needs the")
        print("predecessor sources, which are not part of this repository.\n")

    print(f"DataFrame {args.rows:,} rows x {len(frame.columns)} columns".replace(",", " "))
    print(f"best of {args.repeat} runs, time in seconds\n")
    if compare:
        print(f"{'function':<30} {'old':>9} {'new':>9} {'speed-up':>10}")
    else:
        print(f"{'function':<30} {'new':>9}")
    print("-" * 62)

    total_old = total_new = 0.0
    for name, old_fn, new_fn in cases:
        new = timed(name, new_fn, frame, args.repeat)
        total_new += new
        if not compare:
            print(f"{name:<30} {new:>9.3f}")
            continue
        old = timed(name, old_fn, frame, args.repeat)
        total_old += old
        ratio = old / new if new else float("inf")
        print(f"{name:<30} {old:>9.3f} {new:>9.3f} {ratio:>9.2f}x")

    print("-" * 62)
    if compare:
        ratio = total_old / total_new if total_new else float("inf")
        print(f"{'total':<30} {total_old:>9.3f} {total_new:>9.3f} {ratio:>9.2f}x")
    else:
        print(f"{'total':<30} {total_new:>9.3f}")

    column_count = len(frame.columns)
    rows_per_second = args.rows / total_new if total_new else float("inf")
    print(
        f"\nnew version: {rows_per_second:,.0f} rows/s at {column_count} columns".replace(",", " ")
    )
    if compare:
        print(f"old version: {args.rows / total_old:,.0f} rows/s".replace(",", " "))
    print("(the measured ceiling for reading from a source is ~34 200 rows/s)")
    print()
    print("NOTE: throughput is per row TIMES column, not per row. Real target tables")
    print("have 43 and 56 columns, so the number above divides down for them.")
    print("Try `--width 5`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
