#!/usr/bin/env python3
"""Generate ``src/dbextractors/core/_reserved_words.py`` from the data in ``docs/data/``.

The list of 825 reserved words is part of the **column-naming contract**, so it has
to ship inside the package rather than sit in the documentation — a client installs
the wheel, not the repository. It is generated rather than hand-written so that the
diff is visible in code review and so that nobody can quietly "fix" the module
without touching its source.

Source: ``docs/data/mage_sql_reserved_words.json``, extracted from the production
image ``mageai/mageai:0.9.79``. The words originate from Mage
(https://www.mage.ai/), licensed under the Apache License, Version 2.0; see NOTICE.

Usage::

    python scripts/gen_reserved_words.py          # write the module
    python scripts/gen_reserved_words.py --check  # only verify it is up to date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "data" / "mage_sql_reserved_words.json"
TARGET = ROOT / "src" / "dbextractors" / "core" / "_reserved_words.py"

HEADER = '''"""Reserved words taken from ``mage_ai.io.base.SQL_RESERVED_WORDS``.

**Generated — do not edit by hand.** The source is
``docs/data/mage_sql_reserved_words.json`` (mage_ai {version},
{source}); the generator is
``scripts/gen_reserved_words.py``.

Why the list is frozen inside the package instead of being read from ``mage_ai``:
target column names must not change, not even when Mage is upgraded. Were the list
read from the library, a Mage upgrade would silently rename columns and break the
dbt layer downstream. Frozen, any divergence shows up as a **failing test** rather
than as a silent schema change — see ``tests/naming/test_reserved_words.py``.
"""

from __future__ import annotations

#: {count} words, upper case, in the order they appear in the source.
SQL_RESERVED_WORDS: frozenset = frozenset(
    (
'''

FOOTER = """    )
)
"""


def render(payload: dict) -> str:
    words = payload["words"]
    body = "".join(f'        "{word}",\n' for word in words)
    return (
        HEADER.format(
            version=payload["mage_ai_version"],
            source=payload["source"],
            count=payload["count"],
        )
        + body
        + FOOTER
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify, do not write")
    args = parser.parse_args()

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(payload["words"]) != payload["count"]:
        print(
            f"ERROR: {SOURCE} declares count={payload['count']}, "
            f"but it holds {len(payload['words'])} words.",
            file=sys.stderr,
        )
        return 2
    if len(set(payload["words"])) != payload["count"]:
        print(f"ERROR: {SOURCE} contains duplicates.", file=sys.stderr)
        return 2

    rendered = render(payload)
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            print(
                f"ERROR: {TARGET.relative_to(ROOT)} does not match the source. "
                "Run `python scripts/gen_reserved_words.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"{TARGET.relative_to(ROOT)} is up to date ({payload['count']} words).")
        return 0

    TARGET.write_text(rendered, encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)} ({payload['count']} words).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
