#!/usr/bin/env python3
"""Verify `core.naming` against the **live target database**.

The only proof that really counts: read the actual column names out of the
production PostgreSQL and check that our function turns the original source
names into exactly those.

The source names themselves are not available (they live in MySQL/MSSQL/Firebird
behind a tunnel), so what gets checked are the **invariants** that follow from the
chain `create_column_mapping` -> `clean_column_name`. Any error in our replica of
Mage's behaviour would break at least one of them:

1. Every name is a **fixed point** of our function — ``clean_column_name(c) == c``.
   If our sanitisation stripped or substituted differently from Mage, an existing
   column would get renamed under our hands.
2. A column with an underscore prefix is either **metadata** or ``_`` + a reserved
   word. A prefix anywhere else would mean our word list holds a word too many.
3. A reserved word **without** a prefix does not exist. That would mean a word is
   missing from our list — the silent variant of the same fault.
4. Every name consists solely of ``[a-z0-9_]`` and does not start with ``letter_``.
   These are not invariants of the function (``\\w`` is unicode-aware and a
   ``letter_`` prefix is legitimate) but observations about the real data; they are
   reported as information, not as findings.

Schemas are **not guessed from their names**: every schema that has at least one
table with a ``_deleted_in_source`` or ``_timestamp`` column is taken. That is the
signature of our extractors, and it works even in repositories where the schemas
are not named ``raw_*``.

Reads ``information_schema`` **only**; the connection goes through
`dbextractors.golden.session`, i.e. with ``default_transaction_read_only = on``.

Usage::

    python scripts/verify_column_names.py                 # POSTGRES_* from the env
    python scripts/verify_column_names.py --dsn "..."
    python scripts/verify_column_names.py --schema raw_erp --schema raw_shop
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbextractors.core import naming  # noqa: E402
from dbextractors.golden.session import connect, resolve_dsn  # noqa: E402

#: Columns that identify a schema as one written by our extractors.
SIGNATURE = ("_deleted_in_source", "_timestamp")

Column = Tuple[str, str, str]  # schema, table, column


def load_dotenv(path: Path) -> None:
    """Load ``.env`` if it exists. Same behaviour as ``tests/golden/conftest.py``."""
    if not path.exists():
        return
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def detect_schemas(cur) -> List[str]:
    cur.execute(
        """
        SELECT DISTINCT table_schema
        FROM information_schema.columns
        WHERE column_name = ANY(%s)
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY 1
        """,
        (list(SIGNATURE),),
    )
    return [row[0] for row in cur.fetchall()]


def fetch_columns(cur, schemas: Sequence[str]) -> List[Column]:
    cur.execute(
        """
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = ANY(%s)
        ORDER BY table_schema, table_name, ordinal_position
        """,
        (list(schemas),),
    )
    return list(cur.fetchall())


def check(columns: Sequence[Column]) -> Tuple[int, List[str]]:
    """Return (number of findings, report lines)."""
    reserved = naming.load_reserved_words()
    metadata = set(naming.METADATA_COLUMNS)

    not_fixed_points = [c for c in columns if naming.clean_column_name(c[2]) != c[2]]
    prefixed = [c for c in columns if c[2].startswith("_")]
    unexplained = [c for c in prefixed if c[2] not in metadata and c[2][1:].upper() not in reserved]
    unprefixed_reserved = [c for c in columns if c[2].upper() in reserved]

    lines: List[str] = []
    lines.append(f"{'columns in total':<37}{len(columns):>7}")
    lines.append(f"{'distinct names':<37}{len({c[2] for c in columns}):>7}")
    lines.append(f"{'with an underscore prefix':<37}{len(prefixed):>7}")
    lines.append("")
    lines.append("INVARIANTS")
    lines.append(f"  {'1. not a fixed point of our function':<35}{len(not_fixed_points):>7}")
    lines.append(f"  {'2. prefix without an explanation':<35}{len(unexplained):>7}")
    lines.append(f"  {'3. reserved word without a prefix':<35}{len(unprefixed_reserved):>7}")

    findings = not_fixed_points + unexplained + unprefixed_reserved
    for heading, group in (
        ("1. NOT A FIXED POINT", not_fixed_points),
        ("2. PREFIX WITHOUT AN EXPLANATION", unexplained),
        ("3. RESERVED WORD WITHOUT A PREFIX", unprefixed_reserved),
    ):
        if not group:
            continue
        lines.append("")
        lines.append(heading)
        for schema, table, column in group[:50]:
            detail = ""
            if group is not_fixed_points:
                detail = f"  -> our function returns {naming.clean_column_name(column)!r}"
            lines.append(f"  {schema}.{table}.{column}{detail}")
        if len(group) > 50:
            lines.append(f"  … and {len(group) - 50} more")

    # Observations, not invariants.
    non_ascii = sorted({c[2] for c in columns if not c[2].isascii()})
    letter = sorted({c[2] for c in columns if c[2].startswith("letter_")})
    upper_case = sorted({c[2] for c in columns if c[2] != c[2].lower()})
    lines.append("")
    lines.append("OBSERVATIONS (not invariants)")
    lines.append(f"  {'non-ASCII':<35}{len(non_ascii):>7}  {non_ascii[:5]}")
    lines.append(f"  {'with a letter_ prefix':<35}{len(letter):>7}  {letter[:5]}")
    lines.append(f"  {'with an upper-case letter':<35}{len(upper_case):>7}  {upper_case[:5]}")

    most_common = Counter(c[2] for c in prefixed).most_common(8)
    lines.append("")
    lines.append("MOST COMMON PREFIXED NAMES")
    for name, count in most_common:
        lines.append(f"  {name:<24} {count:>5}")

    return len(findings), lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn", help="DSN of the target PostgreSQL; otherwise POSTGRES_* from the environment"
    )
    parser.add_argument(
        "--schema",
        action="append",
        dest="schemas",
        help="schema to verify; repeatable. Without it, schemas are detected by signature.",
    )
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")

    with connect(resolve_dsn(args.dsn)) as conn:
        cur = conn.cursor()
        schemas = args.schemas or detect_schemas(cur)
        if not schemas:
            print(
                "No schema found with a "
                f"{' or '.join(SIGNATURE)} column — is this really the target database?",
                file=sys.stderr,
            )
            return 2
        columns = fetch_columns(cur, schemas)

    print("COLUMN NAMES VERIFIED AGAINST THE LIVE TARGET")
    print("=" * 60)
    print(f"schemas ({len(schemas)}): {', '.join(schemas)}")
    print()
    finding_count, lines = check(columns)
    print("\n".join(lines))
    print()
    if finding_count:
        print(
            f"VERDICT: DIVERGENCE — {finding_count} findings. "
            "`core.naming` does not match the target."
        )
        return 1
    print("VERDICT: MATCH — every invariant holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
