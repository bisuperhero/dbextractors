## What this changes

<!-- One paragraph. What behaviour is different after this than before it. -->

## Why

<!-- The situation that made it necessary. If a production incident is behind
     it, say so — that reason belongs in the code comment too. -->

## How it was verified

<!-- Name the test that fails without this change. "Ran the suite" is not a
     verification; the suite was green before as well. For anything touching
     extraction that means a golden test — row count, column names and order,
     types, per-column checksums, per-row row_hash. -->

## Checklist

- [ ] A test fails without this change, and I checked that it does.
- [ ] No new configuration key was renamed or given a new meaning; anything new
      has a default that leaves existing pipelines behaving identically.
- [ ] No column name in the target changed.
- [ ] Runs on Python 3.10 / pandas 1.5.3 / SQLAlchemy 1.4.54. No new dependency,
      or a new one verified against those versions.
- [ ] `ruff check`, `ruff format --check` and `mypy` are clean.
- [ ] CHANGELOG.md updated, with an explicit note if anything breaks.

## Performance

<!-- Only if this touches the read, conversion or write path. Give a number
     from scripts/bench_*.py and the column count it was measured at —
     throughput here is per row *times column*, so a figure without the width
     does not mean anything. Delete this section otherwise. -->
