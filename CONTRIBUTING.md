# Contributing

Thanks for looking. Before you spend time on a change, read this page — this
project has a few constraints that are unusual enough that a perfectly
reasonable patch can be unmergeable for reasons that are not visible from the
code.

## The three constraints

**1. The runtime cannot be raised.** This package targets the Mage 0.9.79
image: Python 3.10, pandas 1.5.3, SQLAlchemy 1.4.54, PostgreSQL 17 as the
target. Not 3.11. Not pandas 2. `requires-python = ">=3.10,<3.11"` enforces it
and a CI job checks that the enforcement still holds. Before adding a
dependency, verify it installs and runs on those versions — a previous attempt
at this problem died exactly here.

**2. The configuration contract is frozen.** Existing keys are never renamed
and never change meaning, including the legacy top-level ones documented in
[docs/legacy-compat.md](docs/legacy-compat.md). Hundreds of pipeline
definitions in the wild pass those dicts. A new capability arrives as a **new
key with a safe default**, never as a change to an old one, and never as a
`_v2` module.

**3. Target column names cannot change.** The package reproduces Mage's habit
of prefixing an underscore onto any column name whose uppercase form is one of
825 reserved words. That is why `_type`, `_name`, `_date` and friends exist in
the target. Modelling layers are built on those names, so reproducing the
behaviour exactly matters more than the behaviour being sensible. `tests/naming`
pins it, and a CI job compares the local copy of the word list against the live
library inside the Mage image.

## Getting set up

```sh
make install     # uv venv on Python 3.10 + editable install with dev extras
make db-up       # PostgreSQL target + MySQL, MSSQL and Firebird sources, seeded
cp .env.example .env
make check       # lint, types, tests, runtime verification
```

`make db-up` needs Docker. Without it most of the suite still runs: tests that
need a database skip rather than fail. That is right locally and wrong in CI,
so CI additionally asserts that nothing was skipped.

Two suites do not run from the venv at all:

- `make test-mage` runs the reserved-word parity tests inside the Mage image,
  which is the only place `mage_ai` exists.
- `make bench` / `make bench-write` are the performance harness.

## What a change needs

- **A test that fails without it.** For anything touching extraction, that
  means a golden test: old and new components process the same table into two
  schemas and the row count, **column names and order**, types, per-column
  checksums and per-row `row_hash` are compared. See
  [docs/golden-test.md](docs/golden-test.md).
- **A reason in the comment, not a restatement of the code.** Most comments in
  this codebase exist because something surprising happened in production. Keep
  that ratio.
- **A CHANGELOG entry**, including an explicit note when something breaks.
- `ruff check` and `ruff format` clean, `mypy` clean.

Performance changes need a measurement, not an argument. `scripts/bench_*.py`
is where the existing numbers come from; the conversion layer's throughput is
per row *times column*, so quote the column count with any figure.

## Things that will be rejected

- Row-by-row `INSERT` on the write path. `COPY … FROM STDIN` is the only one.
- `DataFrame.apply(..., axis=1)` or `iterrows()` in a hot path.
- A full load that drops the target table. The swap goes through a shadow
  table, and never through `RENAME` — a view holds the table's OID, not its
  name, so renaming leaves dependent views silently serving last run's data.
- Swallowing an exception without logging it.
- Returning a zero row estimate when the estimate failed. Combined with
  `empty_rows_ok` that overwrites a target table with nothing and reports
  success.

## Commits and releases

Commit messages are in Czech, code and identifiers in English — the project has
a Czech-speaking maintainer and an English-speaking API. Versioning is semver
with a tag per release; clients pin the tag.

## Reporting a bug

Include the dialect, the load method, and the configuration dict with
credentials removed. If a value crossed the source/target boundary wrongly,
include the source column type — most of the interesting bugs in this package
live in that mapping.

Security issues do not go in the issue tracker; see [SECURITY.md](SECURITY.md).
