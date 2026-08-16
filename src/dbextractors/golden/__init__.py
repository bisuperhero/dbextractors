"""Golden test — the oracle the new package is verified against.

Takes one table's configuration, has both the old and the new component process
it into two separate schemas, and decides mechanically whether the result is
identical. Without that, 670 tables cannot be migrated — the code would be
written with nobody able to tell whether it is right.

The comparison runs in five levels, ordered the way things break:

1. existence and row count
2. **column names and their order** — the most important one; see the column
   naming section of ``docs/legacy-compat.md``
3. data types and nullability
4. per-column checksums
5. row-by-row comparison through ``row_hash``

Two things make this a tool rather than a script:

- **It only reads production.** The comparing session runs with
  ``default_transaction_read_only = on``; a write is refused by the database,
  not by our own code.
- **An approximate comparison must not look exact.** Wherever the comparison
  goes through a textual representation that need not be canonical, the report
  says so explicitly.
"""

from dbextractors.golden.compare import CompareOptions, compare_tables
from dbextractors.golden.model import BatchReport, Level, Relation, TableReport

__all__ = [
    "BatchReport",
    "CompareOptions",
    "Level",
    "Relation",
    "TableReport",
    "compare_tables",
]
