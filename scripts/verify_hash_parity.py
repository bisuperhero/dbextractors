#!/usr/bin/env python3
"""Does the ``row_hash`` computed in the source match the one computed in pandas?

Normally `row_hash` is computed **in the source database** — the expression is
embedded in the ``SELECT``. The pandas-side `_row_hash` only kicks in when the hash
column is missing from a batch, or under ``force_recompute``. Hence the question:
**do both paths produce the same value?** If they do not, a single fall-back onto
the pandas path marks the whole table as changed and forces a full reload.

The script answers that against a **real source**: for each table it runs the
reference hash expression, reads the same rows back and computes the hash in
pandas. It **only reads** from the source — nothing but ``SHOW`` and
``SELECT … LIMIT``.

Three variants of the pandas path are measured, because each of them corresponds
to a different place in the predecessor's code:

- **raw** — the way the predecessor does it: the hash is computed **before**
  `decode_bytes_columns`
- **after decode** — what would happen if the order were reversed
- **after decode + fixups** — additionally normalising the zero date and the
  scale of decimal numbers

Usage::

    python scripts/verify_hash_parity.py --url "mysql+mysqlconnector://…"
    python scripts/verify_hash_parity.py --url "…" --tables authorizations addresses
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_LIMIT = 50
MAX_COLUMNS = 40

#: The zero date MySQL is able to store and which the driver returns as ``None``.
MYSQL_ZERO_DATETIME = "0000-00-00 00:00:00"
MYSQL_ZERO_DATE = "0000-00-00"


def apply_row_hash(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Copied verbatim from the predecessor's MySQL streaming extractor."""

    def _row_hash(row):
        concat = "||".join("" if pd.isna(row[c]) else str(row[c]) for c in columns)
        return hashlib.sha256(concat.encode("utf-8")).hexdigest()

    return df.apply(_row_hash, axis=1).tolist()


def decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        if not series.map(lambda v: isinstance(v, (bytes, bytearray))).any():
            continue
        df[col] = series.map(
            lambda v: v.decode("utf-8", errors="replace")
            if isinstance(v, (bytes, bytearray))
            else v
        )
    return df


def fix_known_divergences(df: pd.DataFrame, types: Dict[str, str]) -> pd.DataFrame:
    """Normalise the two things pandas and MySQL render differently.

    1. **The zero date.** ``CAST('0000-00-00 00:00:00' AS CHAR)`` returns that
       string, but the driver hands it to pandas as ``None`` -> ``''``.
    2. **The scale of decimal numbers.** ``float(10,4)`` yields ``'27.0600'`` in
       MySQL, whereas pandas' ``str(27.06)`` yields ``'27.06'``.
    """
    for col, raw_type in types.items():
        if col not in df.columns:
            continue
        low = raw_type.lower()
        if low.startswith(("timestamp", "datetime")):
            df[col] = df[col].map(lambda v: MYSQL_ZERO_DATETIME if v is None else v)
        elif low.startswith("date"):
            df[col] = df[col].map(lambda v: MYSQL_ZERO_DATE if v is None else v)
        elif ("," in low) and low.startswith(("float", "double", "decimal", "numeric")):
            scale = int(low.split(",")[1].split(")")[0])
            df[col] = df[col].map(lambda v, s=scale: v if v is None else f"{float(v):.{s}f}")
    return df


def probe_table(
    conn, table: str, limit: int
) -> Optional[Tuple[str, int, int, int, int, int, List[str]]]:
    cols = pd.read_sql(f"SHOW COLUMNS FROM `{table}`", conn)
    names = cols["Field"].tolist()
    types = dict(zip(cols["Field"], cols["Type"].astype(str), strict=False))
    if not names or len(names) > MAX_COLUMNS:
        return None

    binary_cols = [c for c, t in types.items() if "binary" in t.lower() or "blob" in t.lower()]

    parts = ", ".join(f"COALESCE(CAST(`{c}` AS CHAR), '')" for c in names)
    select = ", ".join(f"`{c}`" for c in names)
    df = pd.read_sql(
        f"SELECT {select}, SHA2(CONCAT_WS('||', {parts}), 256) AS `row_hash` "
        f"FROM `{table}` LIMIT {int(limit)}",
        conn,
    )
    if df.empty:
        return None

    from_source = df["row_hash"].tolist()
    data = df[names]

    raw = sum(a == b for a, b in zip(from_source, apply_row_hash(data.copy(), names), strict=False))
    after_decode = sum(
        a == b
        for a, b in zip(from_source, apply_row_hash(decode_bytes(data.copy()), names), strict=False)
    )
    after_fixups = sum(
        a == b
        for a, b in zip(
            from_source,
            apply_row_hash(fix_known_divergences(decode_bytes(data.copy()), types), names),
            strict=False,
        )
    )
    return table, len(df), len(binary_cols), raw, after_decode, after_fixups, names


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="SQLAlchemy URL of the source (read-only)")
    parser.add_argument("--tables", nargs="*", help="only these tables; otherwise all of them")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-tables", type=int, default=60)
    args = parser.parse_args(argv)

    from sqlalchemy import create_engine

    engine = create_engine(args.url)
    with engine.connect() as conn:
        tables = args.tables or pd.read_sql("SHOW TABLES", conn).iloc[:, 0].tolist()

    print("row_hash PARITY: SOURCE vs PANDAS")
    print("=" * 78)
    print(f"sample of {args.limit} rows per table, at most {args.max_tables} tables")
    print()
    print(f"{'table':<38}{'rows':>4}{'bin':>5}{'raw':>9}{'+decode':>9}{'+fixups':>10}")
    print("-" * 78)

    results = []
    with engine.connect() as conn:
        for table in tables[: args.max_tables]:
            try:
                out = probe_table(conn, table, args.limit)
            except Exception as err:  # foreign source: one bad table must not kill the run
                print(f"{table:<38}  error: {type(err).__name__}")
                continue
            if out is None:
                continue
            name, rows, binary, raw, decode, fixed = out[:6]
            results.append(out)
            print(
                f"{name:<38}{rows:>4}{binary:>5}"
                f"{raw:>6}/{rows:<3}{decode:>6}/{rows:<3}{fixed:>7}/{rows}"
            )

    if not results:
        print("\nNot a single table could be read.")
        return 2

    without_binary = [v for v in results if v[2] == 0]
    full_match_raw = [v for v in results if v[3] == v[1]]
    full_match_fixed = [v for v in results if v[5] == v[1]]

    print("-" * 78)
    print(f"{'tables':<41}{len(results):>4}")
    print(f"{'without binary columns':<41}{len(without_binary):>4}")
    print(f"{'full match raw (what the code does today)':<41}{len(full_match_raw):>4}")
    print(f"{'full match after fixups':<41}{len(full_match_fixed):>4}")
    print()
    if full_match_raw and all(v[2] == 0 for v in full_match_raw):
        print("CONCLUSION: raw hashing matches exactly those tables WITHOUT binary columns.")
    print(
        f"So for {len(results) - len(full_match_raw)} of {len(results)} tables the pandas path\n"
        "would produce a different hash from the one stored in the target — a full reload."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
