# Golden test

The oracle the package is verified against. It takes a table configuration, lets the old and the
new component each process it into a separate schema, and then decides mechanically whether the
two results are identical.

**Without it, 670 tables cannot be migrated.** You would be writing code that nobody can say is
correct.

## Quick use

```bash
# compare two existing tables
dbx-golden compare legacy_stg.orders dbx_golden_x.orders

# a batch driven by a manifest, with a machine-readable report
dbx-golden batch tables.yaml --json report.json

# scratch schemas left behind by interrupted runs
dbx-golden leftovers [--clean]
```

Exit code: `0` = MATCH, `1` = DIFF, `2` = ERROR. A migration script can be driven by it, group by
group.

Manifest:

```yaml
- label: orders            # how the row is named in the summary
  left:  legacy_stg.orders # reference side (old component)
  right: dbx_golden_x.orders
  key:   id                # optional; otherwise the primary key is looked up
```

## Five levels

The order is the order **in which things break**. The report always states the level at which it
broke first.

| # | what is compared | how |
|---|---|---|
| 1 | existence and row count | `to_regclass`, `count(*)` |
| 2 | **column names and their order** | `information_schema.columns` by `ordinal_position` |
| 3 | data types, nullability, collation | full type signature |
| 4 | per-column checksums | one aggregate query per table |
| 5 | row-level comparison | `FULL OUTER JOIN` on the key |

Level 2 matters most. Writing through `mage_ai.io.postgres` prefixes an underscore onto every name
whose upper-case form is among 825 reserved words. That is where `_type` in 113 tables, `_name` in
107 and `_date` in 92 come from. The dbt layer is built on those names. **A test that compares only
data and not names is worthless.**

The levels **do not stop at the first mismatch.** When column names diverge, levels 4 and 5 still
run over the intersection and are marked as partial — there is a difference between knowing "one
name differs, the data matches" and "a name differs, beyond that we know nothing".

### Why level 5 computes two hashes

The **stored** hash is the value of the `row_hash` column produced by the extractor. It is part of
the contract — 5 dbt models depend on it — so it has to be compared exactly as it is.

The **content** hash is computed from all compared columns. It exists because `row_hash` on its own
is not enough: the configuration contract has both `hash_include_columns` and
`hash_exclude_columns`, so in practice it is often computed from only part of the columns. A change
in an excluded column would not change `row_hash`.

Level 4 does not plug that hole either. Swap the values between two rows and the per-column
checksums stay identical — the content hash is exactly what catches that.

## What is guaranteed and what is not

### Determinism

Several textual representations in PostgreSQL depend on session settings, not on the data. Verified
on PG 17.9:

| what | depends on | evidence |
|---|---|---|
| `timestamptz::text` | `timezone` | `12:00:00+00` in UTC vs `13:00:00+01` in CET |
| `bytea::text` | `bytea_output` | `\x0102` (hex) vs `\001\002` (escape) |
| `float8::text` | `extra_float_digits` | number of digits |
| `date::text` | `DateStyle` | ISO vs German |

Both sides are compared within the same session, so this alone cannot produce a false mismatch. It
would however cause something worse: the stored JSON would not be reproducible. That is why the
settings are pinned on every connection (`session.DETERMINISM_SETTINGS`).

### Independence from row order

Text and other non-aggregatable types are compared by summing 32-bit slices of MD5. Addition is
commutative. Verified: `('a','b','c')` and `('c','a','b')` both yield 3929464487. Two independent
slices of the same MD5 are taken — a single 32-bit sum would collide far too readily across
millions of rows.

### Floating point

`sum(x)` over `double precision` depends on the order of addition, `sum(x::numeric)` does not.
Verified: `0.1+0.2+0.3` gives 0.6000000000000001 as `float8`, and exactly 0.6 through `::numeric`.

`NaN` and `Infinity` are kept out of the sum — a single `NaN` would swallow it and throw away the
information about the whole rest of the column. They are counted separately as `nonfinite`.

### NULL vs empty string

`count(*)` and `count(column)` differ by exactly the number of `NULL`s; an empty string still
counts towards `count(column)`. Verified over `('a', '', NULL)`: `count(v)=2`, `count(*)=3`.

### Where the comparison is approximate

The report **says so explicitly** and marks the level with `~` instead of `✓`. Approximate cases:

- type `json` (not `jsonb`) — it remembers the original text including whitespace and key order
- arrays, composite types, enums — compared through their textual representation

For these, **a match is not proof that the data matches**. The batch summary repeats the warning
for tables that ended with a MATCH verdict.

### A column with a different type on each side

By default such a column is **skipped** at levels 4 and 5. The reason is captured by a regression
test in `tests/golden/test_comparator.py`: `0.25` and `0.2500` are the same number but different
text, so the column would differ in every row and one genuinely changed row would drown in the
noise. The type mismatch itself is reported by level 3, and the report states how many columns were
skipped because of it.

With Firebird sources this is a column or two. **With PostgreSQL sources it is not:** the old side
misses the type map and falls back to `TEXT` — on a 92-column test table that happened for 74 of
the columns. Skipping three quarters of the columns means the MATCH verdict compared row counts,
names, order and a handful of values.

That is what the **`--compare-as-text`** option
(`CompareOptions.compare_type_mismatch_as_text`) is for: columns whose types disagree are cast to
text and their values are compared as well. What does not change:

- **level 3 keeps reporting the type mismatch** — the option adds a value comparison, it does not
  hide the finding,
- the comparison of those columns is marked **approximate**, and the report says at levels 4 and 5
  that a difference does not by itself prove different data.

It is deliberately not the default: for sources with only a handful of type mismatches the option
would merely add spurious differences caused by formatting.

### Memory

Nothing is pulled into Python. Levels 1–4 return a single row per table, level 5 computes
aggregates over a `FULL OUTER JOIN`. A table with 15 M rows therefore fits.

Level 5 can be switched off for genuinely large tables (`--skip-row-level`); the report then states
explicitly that it was not performed.

## Safety

**Production is read-only.** The comparison session runs with
`default_transaction_read_only = on` — a write attempt is refused by the database, not by our own
code, which could be worked around.

**Scratch schemas.** The harness creates and drops schemas with the `dbx_golden_` prefix and
nothing else. The safeguard (`scratch.assert_safe`) is deliberately dumb and cannot be talked out
of it: there is no switch that turns it off. It is called before every `CREATE` and every `DROP`,
not once at start-up.

The schema name carries a timestamp, a label and an eight-character fingerprint derived from the
PID and a UUID, so a collision between two concurrent runs is effectively impossible.

**Configuration redirection** (`runners.redirect_output`) rewrites *every* key in the configuration
that determines the target schema — `TABLE.output_schema` as well as the legacy `OUTPUT_SCHEMA`. If
only one of them were rewritten, the component could reach production through the other. After the
rewrite the result is validated, and a foreign schema anywhere in it aborts the run.

## Limitations worth knowing about

**The old component cannot be run from a plain pytest.** All 15 predecessor blocks import `mage_ai`
at module level, and two of them additionally import modules that live in a deployment-specific
repository and are not part of this package. `LegacyBlockRunner` reports that clearly instead of
failing with an `ImportError` from inside somebody else's file. Running the old component for real
therefore requires the corresponding Mage image.

**`PackageRunner` is fully functional** — `dbextractors.run()` is no longer a skeleton. It has been
verified by live runs against all four dialects; the largest documented comparison covers
10,745,664 rows and was a MATCH with no deviations on all five levels.

**Two tables in the same database are compared.** A cross-database comparison would need a
different route (dblink, postgres_fdw, or shipping hashes around) and is not needed so far.

## Accepted deviations

Without them the golden test reports DIFF even where the new component differs **deliberately** —
and in a listing over hundreds of tables a real defect would drown in the expected noise. The rules
live in `src/dbextractors/golden/deviations.py` and are deliberately narrow: each covers one specific,
justified difference.

A deviation is **printed**, not suppressed, and the verdict is then `MATCH*` — a different state
from a clean match, so that "it is the same" cannot be confused with "it differs, but we know about
it".

| rule | what it forgives |
|---|---|
| `_deleted_in_source` is NOT NULL | the new core creates the column as `NOT NULL DEFAULT false` |
| a reserved column lost its type | on a prefixed column the old side falls back to `text` |
| **a mandatory column the old side does not have** | `row_hash`, `_timestamp`, `_deleted_in_source` — variant B's Firebird extractor has none of them, while dbextractors **must** have them |
| different order of managed columns | only among the columns the package adds, and only for a table created from scratch |

The third rule was added on 2026-08-13: without it the golden test reported DIFF for every Firebird
table because the new side **did what it was asked to do**. It applies only to those three columns
by name and only in the direction "extra on the right" — an extra `_source`, a missing column, or a
column coming from the source do not fall under it. All three negative cases have tests.

## Tests of the test

*"A harness that always says match is worse than no harness at all."*

`tests/golden/test_comparator.py` verifies that the harness **finds** the difference, and for each
case also **at which level** — because what happens next is decided by the level:

| case | expected level |
|---|---|
| missing row | 1 |
| renamed column (`_type` → `type`) | 2 |
| reordered columns | 2 |
| different data type | 3 |
| different nullability | 3 |
| a changed value, in a number and in text | 4 |
| `NULL` replaced by an empty string | 4 |
| values swapped between two rows | 5 |
| a change in a column outside `row_hash` | 5 |

Plus a check that identical tables do produce a MATCH (without it the other tests could be passing
by accident), that two runs give the same result, and that the comparison connection really cannot
write.

The tests need a live PostgreSQL — the comparison is ninety per cent SQL and a mock would only
verify string concatenation. The database is taken from `DBX_GOLDEN_TEST_DSN` or from `.env`;
without one the tests are skipped, not failed.

`tests/golden/test_safety.py` and `test_sqlgen_and_report.py` do not need a database.

## Connecting from WSL

When the development PostgreSQL runs on Windows and the work happens in WSL, the LAN address of
that **same** machine is not reachable from WSL — the Windows Firewall blocks it. What is reachable
is the gateway of the virtual adapter, whose address changes on every WSL restart and therefore
must not be written into `.env`.

`session.resolve_dsn` handles it: when the configured host does not answer and we are running under
WSL, it tries the gateway discovered at run time from `ip route`. **The substitution is always
logged** — silently redirecting somewhere other than where the user aimed is unacceptable in an
oracle.
