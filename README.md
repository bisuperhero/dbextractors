# dbextractors

[![CI](https://github.com/bisuperhero/dbextractors/actions/workflows/ci.yml/badge.svg)](https://github.com/bisuperhero/dbextractors/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dbextractors)](https://pypi.org/project/dbextractors/)
[![Python](https://img.shields.io/pypi/pyversions/dbextractors)](https://pypi.org/project/dbextractors/)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)

Table extraction from **MySQL, MSSQL, PostgreSQL and Firebird** into
**PostgreSQL**, driven by a plain configuration dict.

It is not a general-purpose ETL framework. It does one direction of one job,
and it does it for tables large enough that the naive version stops working:
`COPY … FROM STDIN` as the only write path, a vectorised conversion layer,
incremental and hash-diff loading, keyset pagination, optional SSH tunnelling,
and a full load that swaps through a shadow table instead of dropping the
target.

```python
from dbextractors import run

df = run(config, dialect="mysql")
```

It grew out of a set of hand-copied extractor blocks that had accumulated for
years across several deployments, serving roughly 670 tables. That history
shows in two places, and both are deliberate:

- **The configuration contract is frozen.** Keys are never renamed and never
  change meaning; new capability arrives as a new key with a safe default.
- **Older configuration shapes still work.** See
  [Backward compatibility](docs/legacy-compat.md) — you do not need any of it
  for a new pipeline.

> **Runtime.** This package targets the Mage 0.9.79 image: **Python 3.10,
> pandas 1.5, SQLAlchemy 1.4**, with dependencies pinned to the exact versions
> that image ships. Installation on 3.11+ is refused on purpose rather than
> failing later in production. Widening that is a known and separate piece of
> work.

## Why this exists

- **So that a deployment can add an extractor without copying one.** A new table
  should be a few lines of configuration, and the complicated part should live in
  one place that is versioned and tested. That is why `run()` is the entire API and
  why the configuration contract is frozen: 15 hand-copied blocks became one call,
  and a single loader block now serves 101 pipelines.
- **So that awkward sources are reachable at all.** Firebird is the case in point.
  Its Python driver, `fdb`, is the oldest and most fragile of the four, which is why
  it sits behind its own extra (`pip install "dbextractors[firebird]"`) and gets its
  own CI job marked `continue-on-error` — a Firebird image that will not start is
  worth seeing, but should not block a release. General-purpose tooling tends not to
  carry a source that needs that much special handling.
- **So that the things a real deployment needs are in the box.** Reaching a source
  that is not directly routable is the clearest example, and it is often left out of
  open-source ETL tooling. Here it is `connection_mode: direct | ssh | auto`, host
  key verification through `ssh_host_key_checking`, and a tunnel started as an
  OpenSSH subprocess under `PR_SET_PDEATHSIG` — rather than a Python SSH library —
  specifically so a failed run cannot leave an orphaned process holding the
  forwarded port open.

## Mage integration

The package does not require Mage — there is a single lazy import of `mage_ai`,
used only to locate `io_config.yaml` when running inside it. Under Mage the
pipeline stays two blocks: a **config block** builds the dict and a
**data_loader block** hands it to `run()`. One such loader covers every database
extraction in a repository. The block itself, what has to be passed through it and
what it returns are in [The Mage loader block](docs/mage-loader-block.md).

## Running without Mage

Nothing in the extraction path needs Mage. The one `mage_ai` import is lazy and
returns `None` if the package is absent, so `run()` works from a plain script, a
cron job or a test:

```bash
pip install "dbextractors[target,mysql]"     # extras name the sources you read
export DBX_TARGET_DSN="postgresql://user:pass@localhost:5432/warehouse"
```

```python
import logging
from dbextractors import run

config = {
    'TABLE':         {'source_name': 'orders', 'output_schema': 'raw_shop',
                      'output_name': 'orders'},
    'SOURCE_DB':     {'user': 'readonly', 'password': '…', 'host': '10.0.0.1',
                      'database': 'shop'},
    'LOAD_SETTINGS': {'load_method': 'hash', 'primary_column': 'id'},
    'connection_mode': 'direct',
}

status = run(config, dialect='mysql', logger=logging.getLogger('dbx'))
```

`logger` is optional and takes any standard `logging.Logger`; the docstring calls it
the Mage logger because that is where it usually comes from, not because it has to
be one.

The return value is a status `DataFrame`, one row per source database, with the
columns `table`, `source`, `rows_written`, `load_method`, `success`,
`is_incremental`, `data_present`, `fallback_reason`, `connection_mode` and `error`.
`success` is what a caller should assert on; `fallback_reason` tells you why a run
took the expensive path, and `load_method` what it actually did, which is not always
what you asked for — see [Load methods](#load-methods).

**Where writing goes** is resolved in this order, and the order is not arbitrary —
it was set by a production failure in which the target was taken from `POSTGRES_*`
in a deployment where `POSTGRES_PORT` is the port published on the *host*, not the
one the database listens on inside the network:

1. `DBX_TARGET_DSN` — explicit, overrides everything. Use this outside Mage.
2. The `io_config.yaml` profile (`warehouse` by default; `TARGET_PROFILE` in the
   configuration, or `DBX_TARGET_PROFILE` for a whole container). This is what the
   replaced extractors used, so dbextractors lands where they landed.
3. `POSTGRES_HOST` / `POSTGRES_DB` / `POSTGRES_USER`, with `POSTGRES_PORT`
   defaulting to 5432 and an empty password allowed — for runs outside Mage where
   there is no `io_config.yaml`.
4. `DBX_GOLDEN_DSN`, so deployments that set it because of that bug keep working.

If none of them resolve, the run fails with `TargetConnectionError` naming
everything it tried.

One thing stays Mage-shaped even standalone: **target column names reproduce Mage's
reserved-word prefixing**, so a source column called `name` arrives as `_name`. That
is the compatibility contract rather than a leftover, and it applies to a brand-new
table too — see [Columns added to the target](#columns-added-to-the-target).

## Configuration

A plain dict. Only what the table actually needs has to be filled in —
everything outside `TABLE` and `SOURCE_DB` has a sensible default.

```python
config = {
    # Which source to read from: firebird | mysql | mssql | postgres.
    # Under Mage this is read by the loader block and passed to run(), because
    # Mage does not hand a block its own configuration from metadata.yaml.
    'DIALECT': 'firebird',

    # ── what moves, and where to ───────────────────────────────────────────
    'TABLE': {
        'source_name': 'source_table',      # table in the source
        'source_schema': '',                # MSSQL/PostgreSQL; leave empty for MySQL and Firebird
        'output_schema': 'raw_source',      # target schema; created if absent
        'output_name': 'target_table',      # target table (alias: 'output_table')

        'only_selected_columns': False,     # True = take only 'selected_columns'
        'selected_columns': [],
        'excluded_columns': ['password'],   # control columns (PK, hash, modification date) cannot
                                            # be excluded — they are put back
        'where_clause': None,               # appended to the source query's WHERE
        'empty_rows_ok': True,              # source is empty -> the target may be overwritten empty.
                                            # Without it the target is left alone, which is safer.
    },

    # ── where it is read from ──────────────────────────────────────────────
    'SOURCE_DB': {
        'user': 'readonly_user',
        'password': os.environ['SOURCE_DB_PASSWORD'],   # a secret store, not the source file
        'database': 'source_database',
        'charset': 'utf8mb4',

        'host': '10.0.0.1',                 # direct connection
        'port': 3306,                       # empty = the dialect's default port

        'ssh_address_or_host': 'ssh.example.com',   # tunnelled; used according to 'connection_mode'
        'ssh_username': 'tunnel',
        'ssh_pkey': '/home/src/.ssh/id_rsa',
        'ssh_port': 22,
        'ssh_wait_timeout': 20,
        'ssh_host_key_checking': 'off',     # off | accept-new | strict
                                            # 'off' is inherited behaviour: the host fingerprint
                                            # is not verified at all. Use 'accept-new' for a new
                                            # deployment. See "SSH host key verification".
        'remote_bind_address': ('127.0.0.1', 3306),
        'local_bind_address': ('127.0.0.1', 0),   # 0 = let the system pick the port
    },

    # ── how it is loaded ───────────────────────────────────────────────────
    'LOAD_SETTINGS': {
        'load_method': 'hash',              # full | full_by_source | incremental
                                            # hash (= hash_diff) | id_watermark
                                            # parent_incremental is inferred from 'incremental_parent_*'
                                            # An unknown value raises — no quiet fallback to full load.
        'primary_column': 'id',             # PK, case-sensitive (alias: 'primary_key_column')
        'batch_size': 500000,               # empty = derived from a size estimate
        'hash_column': None,                # row-hash column, when the source already has one

        # time-based increment (load_method: incremental)
        'created_at_column': 'created_at',
        'updated_at_column': 'updated_at',
        'days_back': 7,                     # window back, default 14

        # window taken from a parent table (parent_incremental)
        'incremental_parent_table': None,   # setting it selects the strategy
        'incremental_parent_date_column': None,
        'incremental_parent_key_column': None,   # the child's FK; None = 'parent_id'
        'incremental_parent_id_column': None,    # the parent's key; None = 'id'

        # hash_diff
        'hash_include_columns': None,       # None = every column
        'hash_exclude_columns': [],
        'ignore_hash': False,

        # other
        'compute_row_hash': None,           # None = the strategy decides
        'surrogate_key_enabled': False,     # a surrogate key instead of the source PK
        'surrogate_key_definition': None,   # the SQL expression that produces it
        'partition_by_source': False,       # multi-source: slices by '_source'
                                            # (shorthand for partition_by below)
        'partition_by': None,               # target partitioning; None = none. See "Target partitioning".
                                            # {'column': 'day', 'mode': 'range_month'}
                                            # mode: list | range_day | range_month | range_year
        'skip_size_estimate': False,
        'strict_integer_precision': False,  # fail the run instead of writing an integer that
                                            # float64 already rounded. Off = the inherited
                                            # behaviour. See "Large integers and NULL".
        'convert_nchar_to_varchar': False,  # MSSQL only: read NVARCHAR/NCHAR/NTEXT through a
                                            # server-side CONVERT so the text arrives whole.
                                            # Off = the inherited truncation. Turning it on
                                            # means reloading that table — see "MSSQL and
                                            # NVARCHAR text".
    },

    # ── top-level keys ─────────────────────────────────────────────────────
    'connection_mode': 'direct',            # direct | ssh | auto ('auto' is the inherited default)
    'DATABASES': None,                      # databases for a multi-source run (several of the same
                                            # shape on one server); key absent = the single database
                                            # from SOURCE_DB, empty list = the control layer selected
                                            # none, which is a successful run
    'FINGERPRINT': {},                      # cheap source fingerprint: {'mode': 'datsave'|'aggregate',
                                            # 'timestamp_column': …, 'aggregate_columns': [...]}
                                            # Skips databases in which nothing changed.
    'TARGET_PROFILE': None,                 # io_config.yaml profile, None = 'warehouse'
    'DEBUG': False,                         # verbose logging (type maps, WHERE, columns)
}
```

Older configurations put some of these keys at the top level instead of in a
section, and those are still read as a fallback. An unknown key is logged, not
rejected. See [Backward compatibility](docs/legacy-compat.md).

A `TARGET_DB` section, if present, is ignored: where the data is written is
decided by `io_config.yaml` (profile `warehouse`), optionally overridden by
`TARGET_PROFILE` or the `DBX_TARGET_DSN` environment variable.

## Load methods

`load_method` in `LOAD_SETTINGS` decides how much of the source is read and what
happens to the target. It is the single most consequential line in a table's
configuration: the difference between the cheapest and the most expensive method on
the same table is hours, and the difference between one that tracks deletions and
one that does not is a `_deleted_in_source` column that silently never changes.

An unknown value **raises**. The predecessor quietly fell back to a full load, which
is the most expensive possible outcome of a typo in YAML.

| method | for | needs from the source | what it does to the target | cost per run |
|---|---|---|---|---|
| `full` | small tables, or no usable key | nothing; a `primary_column` if you want dedup and an index | rebuilds it — shadow table, then `TRUNCATE` + `INSERT … SELECT` in one transaction | the whole source, every time |
| `full_by_source` | one table fed from several databases of the same shape | `multi_source` / `DATABASES`; MSSQL only today | replaces that source's `_source` slice, leaves the other slices alone | the whole source per database, minus the ones the fingerprint skipped |
| `incremental` | a trustworthy modification-time column | `updated_at_column`, optionally `created_at_column` | upserts the window | only the window |
| `hash_diff` (`hash`) | no CDC log, but a stable PK | `primary_column`; source-side hashing, so not Firebird | upserts changed rows and marks keys that stopped arriving | the whole source — a hash for every row — then fetches only what changed |
| `id_watermark` | append-only, increasing PK | an increasing `primary_column` | appends rows above `MAX(pk)` in the target | only the rows above the watermark |
| `parent_incremental` | child tables with no change date of their own | a parent table with a date column, and a key into it; Firebird only today | deletes the window and reloads it, in one transaction | only the window |

Two methods are not selected by name:

- **`parent_incremental` is inferred.** `load_method` stays `incremental`; the
  strategy is recognised **only by the presence** of `incremental_parent_table`,
  exactly as the predecessor does. Without that rule those child tables would go to
  `incremental`, look for a change timestamp of their own, fail to find one and
  raise. It affects 11 of one deployment's 102 configurations.
- **`full` + `resume_full_load` becomes `id_watermark`.** Underneath, resuming is
  literally the watermark strategy: read rows whose key is above `MAX(pk)` in the
  target, no state stored anywhere. Before this was wired up, the config layer read
  the key and the whole table was rebuilt anyway — on one production table the
  difference between fetching 823 thousand rows and reloading 9.9 million.
  `force_reload` wins over it, because forcing a reload means trusting nothing the
  target remembers, and a watermark is exactly what must not be trusted then.

### `full`

Reads the whole source and rebuilds the target. Roughly 90 tables run this way. It
is the only method that needs nothing from the source: no key, no timestamp, no
hash. Reach for it when the table is small enough that reading all of it is cheaper
than working out what changed, when there is no column you would trust to tell you
what changed, and as the first run of any of the incremental methods — they all fall
back to it.

The rebuild is a load into a shadow table followed by `TRUNCATE` and
`INSERT … SELECT` in a single transaction. It is deliberately **not** a `DROP` and
**not** a `RENAME`. `DROP … CASCADE` takes down up to 102 dependent views, and an
interruption leaves the target truncated — documented by a run that ended with 11.4
of 18.8 M rows. A `RENAME` is worse in a quieter way: a view holds an OID rather
than a name, so after renaming it stays bound to the old table and serves last run's
data without complaint. With the swap, a dependent view keeps working throughout and
a failure part-way through leaves the target exactly as it was.

Do not reach for it on a large table on a schedule. The whole source crosses the
wire every time, and on the widest tables that is hours. `resume_full_load` is not
an answer to that — it turns the run into `id_watermark` and inherits that method's
blind spots along with its speed.

Two things about an empty source are worth knowing before you meet them. If the
source returns no rows and `empty_rows_ok` is not set, the target is **left alone**;
with `empty_rows_ok` it is overwritten empty, which is what that key is for. And a
failed size estimate **fails the run loudly** rather than reporting zero rows —
zero rows plus `empty_rows_ok` would wipe a target and finish green.

### `full_by_source`

Several source databases of the same shape pour into one target table, partitioned
by `_source` so that one of them can be rewritten without touching the others. 16
tables run this way, and only MSSQL offers it today. Reach for it when you have that
exact shape: n identical databases, one warehouse table, and a need to reload one of
them on its own.

Rows carry `_source` regardless of how many databases a given run selected. That
looks like a detail and is a trap: deletion is **by `_source`**, so if the column
appeared only when the run happened to be processing several databases, a run over a
single database would wipe the whole table. `multi_source` in the configuration
decides it, not the length of the list.

`FINGERPRINT` makes it cheaper. One aggregate query against the source — `COUNT`
catches deletes, `MAX(pk)` catches inserts, a timestamp or a column sum catches
edits — and a database whose fingerprint has not moved is skipped without fetching a
row. It is written only **after** a successful transfer, and it is trusted only when
the target still holds that slice. That second condition is not extra caution: the
first database of a rebuilt table recreates it, so from the second one onwards the
table exists again and the fingerprint alone would skip every remaining database. In
the predecessor that left 3 rows of 220,972 in the target.

The one thing to know going in: **`row_hash` is not computed here.** The column
exists in the target and stays `NULL`. That is deliberate — the predecessor reads
these tables in one streamed pass with no hashing, because on the largest of them
hashing means `HASHBYTES` over 156 columns and 1.4 M rows every night on a
memory-starved SQL Server, and no dbt model touches `row_hash` on those tables
anyway. Set `compute_row_hash: true` if yours does.

### `incremental`

A window over a modification-time column: roughly, `WHERE COALESCE(updated_at,
created_at) >= cutoff`, where the cutoff is `days_back` days ago rounded down to
midnight (default 14). About 27 tables run this way. Reach for it when the source
maintains a modification timestamp you actually trust, and when you can live without
knowing about deletions.

The window is measured **from the time of the run, not from the state of the
target**, and three consequences follow that are all easy to overlook:

1. **An outage longer than `days_back` punches a permanent hole in the data.** If
   the pipeline does not run for 10 days and `days_back` is 7, rows changed in the
   first three days of the outage fall outside the window and are never picked up —
   the next run again looks back only 7 days. Nothing reveals it: the run is green
   and moves plenty of rows. The only repair is a full load. Widen the window with
   the `incremental_lookback_hours` runtime variable **before** the first run after
   a long outage, not after.
2. **A row with no change timestamp is never transferred.** `NULL` in both columns
   fails the condition. Falling back onto `created_at` softens that; it does not
   remove it.
3. **Deletions are invisible.** `_deleted_in_source` is declared on the target but
   stays `FALSE` under this method, exactly as it did in the predecessor. Marking
   deletions here would mean re-reading every source key on every incremental run,
   which is the cost the method exists to avoid. A table whose deletions matter
   belongs on `hash_diff`.

A missing target and an empty target both fall back to a full load, and the fallback
is always logged and reported in `fallback_reason`. The target must already carry a
`UNIQUE` or `PRIMARY KEY` index over `primary_column` — without one there is nothing
for the upsert to rest on, and the run fails saying so rather than appending
duplicates.

### `hash_diff` (alias `hash`)

The dominant method: about 530 of the ~670 tables. A `row_hash` is computed for
every source row — by the source itself where it can (`SHA2` on MySQL, `HASHBYTES`
on MSSQL) — compared against the hash stored in the target, and only rows whose hash
moved are fetched and written. Reach for it when there is no CDC log and no
timestamp you trust, but there is a stable primary key. It is the only method that
genuinely maintains `_deleted_in_source`: the diff already holds the full set of
live keys, so marking what stopped arriving costs nothing extra, and a key that
comes back is unmarked again.

Also the most expensive: it is **O(source rows, not changed rows)** — measured at
~15,000 rows/s — because every row has to be hashed to find out whether it changed.
On a table where a timestamp would do the same job, `incremental` is cheaper by the
ratio of changed rows to total rows. Firebird does not offer it at all;
`parent_incremental` is that dialect's answer to the same problem.

The diff itself is a single JOIN inside PostgreSQL: the source hashes are `COPY`ed
into a temporary table and joined against the target. It is not a dictionary of the
target snapshot in RAM and not a `seen_keys` set — at 11 M rows that was 2–3 GB, on
~530 tables, with three pipelines running concurrently.

The rule everything else follows from is that **seeding `row_hash` must use the same
computation path as the diff.** The source-side hash and the pandas hash do not
agree — measured against a real source, 4 of 51 tables matched, precisely the ones
with no binary columns — so a single fall onto the other path marks the whole table
as changed. This is handled for you, but it explains the fallbacks: the method falls
back to a full load when the target is missing, when it is empty, when `row_hash` is
entirely `NULL` (nothing to compare against), when the scan fails, when
`ignore_hash` is set, and when enough of the stored hashes disagree with freshly
computed ones that the column is treated as drifted and reseeded.

That last one is also the method's blind spot. The reseed only triggers on
wholesale drift — essentially every row mismatched and none matched. **Partial drift
is invisible:** the run stays green and simply transfers every row it thinks changed,
which looks like a busy night rather than a fault. Nothing distinguishes "the hashes
drifted" from "every row really did change", and both take the same full load.

`hash_include_columns` and `hash_exclude_columns` narrow what goes into the hash —
useful when a column changes on every read and would make every row look modified.
`hash_download_batch_size` sets how many changed keys go into one `IN (…)` when
fetching them. `hash_diff_buffer_size` is an inherited key whose original meaning
(when to flush the changed-key buffer to a file in `/tmp`) no longer exists, because
that list is now the result of a SQL query; it has been adopted onto the nearest
thing with the same meaning — how many source rows are read before being `COPY`ed
into the snapshot — and still governs the peak memory of the diff phase.

### `id_watermark`

Reads only rows whose primary key is above `MAX(pk)` in the target, by keyset
paging, which every dialect supports. Four tables run this way, all against
PostgreSQL sources, which is what the predecessor offered it for. The watermark is
not stored anywhere — there is no state file and no state table, so there is nothing
to lose and nothing to drift.
Reach for it when the source is genuinely append-only with a sequence-generated key,
and when the saving matters: it is the cheapest method here by a wide margin,
because it never looks at a row it has already seen.

That is also exactly what it cannot do. **Two blind spots are properties of the
method, not omissions**, and both are pinned by tests:

- **A row inserted below the watermark after the fact is never picked up.** With a
  sequence that cannot happen; with manually assigned keys it can, and nothing in
  the run will say so.
- **A deleted row is never marked.** `_deleted_in_source` stays permanently `FALSE`.
  Taking a live-key snapshot would mean reading the entire source table, and the
  whole saving would go with it.

An update below the watermark is likewise not transferred. If any of that matters
for your table, use `hash` or `incremental` instead. A missing target and an empty
target both fall back to a full load.

### `parent_incremental`

For child tables — document lines, order lines — that carry no trustworthy change
date of their own but do have a key into a parent that does. The window is computed
over the parent (`incremental_parent_table` plus `incremental_parent_date_column`,
with an optional `incremental_parent_date_column_fallback`), using the same cutoff
`incremental` computes, so a child's window matches the one its parent's own run
used. 24 tables run this way, all Firebird. You do not select it by name: set
`load_method: incremental` and give it `incremental_parent_table`.

The join between the two tables is `parent_id` in the child and `id` in the parent —
what the predecessor hard-codes. Where they are named otherwise, name them:
`incremental_parent_key_column` and `incremental_parent_id_column`. Leaving them
unset keeps those two defaults, so no existing pipeline moves.

The mechanics differ from every other method. Child rows whose parent falls inside
the window are **deleted from the target and loaded again**. That is the only way to
notice a child row that vanished from the source — it has no change date, so a
deletion cannot be learned about any other way. Deletion here is physical: the row
leaves the target rather than getting `_deleted_in_source = TRUE`. Outside the
window nothing is deleted, even when the source no longer has the row. The `DELETE`
and the write are in one transaction, loaded through a staging table first; the
predecessor committed the `DELETE` and only then began streaming, so a stream that
fell over — over Firebird via SSH, not a theoretical possibility — left the target
without the deleted rows and the run merely looking unsuccessful.

Two things will surprise you, and both are deliberate parity with the predecessor:

- **An empty target is not backfilled.** `incremental` and `id_watermark` fall back
  to a full load when the target exists but holds no rows. This one does not: it
  checks that the child and parent tables **exist** and nothing further, so over an
  empty target it transfers the window alone, leaves everything older missing, and
  the run comes out green with a partial table. Closing that gap would transfer a
  different set of rows than the old side does, so it is pinned by a test and can
  only change on purpose. The reasoning is recorded in
  [Backward compatibility](docs/legacy-compat.md).
- **`days_back` is ignored on this path.** A deep cutoff applies instead. The
  predecessor has no `days_back` branch here, but the config layer substitutes
  `days_back: 14` even where the configuration does not set it — and if that
  substituted default won, the child would get a "today minus 14 days" window
  instead. Documented in production: 0 rows transferred against 394 on the old side,
  with `status = completed`.

A missing target, and a parent that is not in the target, both fall back to a full
load.

### What every method does the same

**Fallback to a full load is reported, never silent.** Whenever one of the four
incremental methods cannot do its cheap thing — no target, no rows to compare
against, hashes drifted — it does the expensive thing and says so, in the log and in
the `fallback_reason` column of the returned frame. `full_by_source` is the
exception: it never falls back, because a full load of a multi-source table would
mean one database wiping the slices of all the others.

**Writes go through `COPY … FROM STDIN` and nothing else**, on every method.

**All three managed columns exist on every method**, but only some are maintained:

| | `_timestamp` | `row_hash` | `_deleted_in_source` |
|---|---|---|---|
| `full` | yes | yes | set `FALSE` |
| `full_by_source` | yes | `NULL` unless `compute_row_hash: true` | set `FALSE` |
| `incremental` | yes | yes | never flipped — stays `FALSE` |
| `hash_diff` | yes | yes | **maintained**, both directions |
| `id_watermark` | yes | yes | never flipped — stays `FALSE` |
| `parent_incremental` | yes | yes | not used — deletion is physical, inside the window |

The distinction matters because a column that exists but never changes does not fail
a dbt model; it just makes it wrong. 133 dbt models read `_deleted_in_source`.

**A column that appears in the source is handled differently by full loads and by
the rest.** Seen in production on 14 Aug 2026, as
`column "promo_code" ... does not exist`. A full load **adopts** the new column — the
table is rewritten anyway, so widening costs nothing, and the column goes at the end
so existing column order is untouched. The four methods that touch existing data —
`incremental`, `hash_diff`, `id_watermark` and `parent_incremental` — **drop** it
instead, because they must not change the target's shape mid-flight, but they drop it
with a warning naming the specific column. The predecessor dropped it silently, which
is how a hole in the warehouse stays hidden. Either way the next full load makes it
up.

## SSH host key verification

When the source is reached through a tunnel, the connection is opened with
`StrictHostKeyChecking=no` and `UserKnownHostsFile=/dev/null` — **the far end is
not verified**. This is inherited behaviour and defensible inside a closed
network, but it is the one place in the package where a connection can be
substituted.

`SOURCE_DB.ssh_host_key_checking` makes it a choice:

| value | behaviour |
|---|---|
| `off` | **default** — the fingerprint is not checked. Inherited behaviour. |
| `accept-new` | remember the fingerprint on first connect, refuse it if it changes |
| `strict` | the host must already be in `known_hosts` |

The default stays `off` deliberately: changing it would break runs against hosts
whose fingerprint was never recorded anywhere. For a new deployment
`accept-new` is the right choice — it protects against substitution and needs
nothing prepared in advance.

Both non-`off` values use the `known_hosts` of the user the process runs as.

## Target partitioning

The target table can be partitioned with `partition_by`. Without it there is
**no partitioning**, which is the state of every existing table and does not
change.

```python
'LOAD_SETTINGS': {
    'partition_by': {
        'column': 'day',            # column in the target (after renaming, see below)
        'mode': 'range_month',      # list | range_day | range_month | range_year
        'default_partition': True,  # catch-all partition, on by default
    },
}
```

The shorthand `'partition_by': 'country'` means
`{'column': 'country', 'mode': 'list'}`. The older `'partition_by_source': True`
keeps working and is exactly `{'column': '_source', 'mode': 'list'}`.

| mode | for | one partition per |
|---|---|---|
| `range_month` | fact tables with a document date, event streams | month |
| `range_day` | daily snapshots; note that 3 years is over a thousand partitions | day |
| `range_year` | long history where months would make too many partitions | year |
| `list` | a column with an enumerable set of values — `_source`, country, brand | value |

Partitions are **created automatically** from the values actually present in the
data: a new month or a new value needs no manual DDL. The catch-all `DEFAULT`
partition takes `NULL` partition keys and anything unclassifiable — without it a
single such row would fail the whole run with "no partition found for row".

Three things worth knowing in advance:

- **Name the column the way it is named in the target.** If the name is a reserved
  word it carries an underscore in the target (`date` → `_date`) — see
  [Columns added to the target](#columns-added-to-the-target).
- **An existing table is never rebuilt in place.** `relkind` cannot be changed in
  PostgreSQL, so converting a table means `DROP` + `CREATE`, and targets have views
  hanging off them. Partitioning is therefore applied only to a table that does not
  exist yet; on an existing plain table it is only logged and the run finishes
  without partitioning. Converting is a manual migration — the procedure is in
  [Enabling partitioning on an existing table](docs/partitioning-existing-table.md).
  A mismatch in the partition **key**, by contrast, fails the run instead of warning.
- **Upsert against a partitioned target behaves differently inside, not outside.**
  PostgreSQL requires a unique index over a partitioned table to contain the
  partition key, so `ON CONFLICT (id)` has nothing to rest on there. The incremental
  strategies therefore write through `DELETE` by key + `INSERT` in one transaction.
  A row whose partition value changes in the source is thereby **moved**, not
  duplicated — which is what a widened conflict key `(id, day)` would do.

## Large integers and NULL

An integer column that also holds a `NULL` cannot be `int64`, so pandas infers
`float64` — inside `pd.read_sql`, before any code of this package runs — and
`float64` carries 53 bits of mantissa. Every integer above 2**53 is rounded on the
way in. Measured end to end on all four dialects, read back out of the target:

| source value | value in the target |
|---|---:|
| `9007199254740993` | `9007199254740992` |
| `1234567890123456789` | `1234567890123456768` |

The trigger is the `NULL`, not the magnitude: the same values in a column without a
`NULL` arrive intact. Above 2**63 the run fails loudly instead.

**This is inherited and is not fixed.** Every predecessor extractor reads through a
bare `pd.read_sql` with no `dtype=` and loses the same bits in the same place, so
correcting the read would change what lands in the target relative to the
implementation this replaced. It reaches further than large numbers, too: because
the hash is rendered straight out of the frame, a plain nullable `INTEGER` holding
`10` is hashed today as `'1||10.0'`. Changing the read dtype would recompute
`row_hash` for every table with a nullable numeric column and force a full reload
of all of them. The mechanism is pinned in `tests/coerce/test_int64_precision.py`.

What you can do is refuse to write a number nobody can trust:

```yaml
LOAD_SETTINGS:
  strict_integer_precision: true
```

The run then fails, naming the column and an offending value, whenever a `float64`
column bound for an integer column in the target holds a value whose absolute value
exceeds 2**53. **Off by default**, because on by default it would turn an unknown
number of currently-green runs red.

The boundary is inclusive: exactly 2**53 passes, because every integer up to and
including it survives `float64` unchanged. The check is conservative in the other
direction — `2**53 + 2` is representable exactly and is still reported, since after
the read the rounded and the intact value are the same `float64`. And `2**53 + 1`
rounds down to exactly 2**53, so the smallest possible corruption is the one case
no check placed here can see.

## MSSQL and `NVARCHAR` text

MSSQL is read over a `cp1250` connection, inherited character for character from
both predecessor extractors, and pymssql and FreeTDS disagree about what that
means: the bytes are produced as latin-1 and decoded as cp1250. An `NVARCHAR`,
`NCHAR` or `NTEXT` value is therefore **cut** at the first character latin-1 cannot
express — `'příliš žluťoučký kůň'` arrives as `'p'` — and a character latin-1 *can*
express comes back as a different one (`ø` as `ř`). ASCII is unaffected, and so are
the single-byte `VARCHAR`, `CHAR` and `TEXT`, which are read correctly **because**
of that charset and are the majority of a CP1250-collated source.

No charset value gets both right; seven were measured. What does is converting on
the server instead:

```yaml
LOAD_SETTINGS:
  convert_nchar_to_varchar: true
```

The N-typed columns — and only those, chosen by their introspected type — are then
wrapped in `CONVERT(VARCHAR(MAX), …)` in the generated `SELECT`, so the server
converts them through the column's own collation and they arrive whole over the same
connection. What CP1250 cannot hold is degraded rather than truncating the value:
Cyrillic and emoji become `?`, and `ø` is folded onto `o`.

**Off by default, and per table.** Switching it on changes what those columns hold,
and `row_hash` does not move with it — the digests were always computed by the
server from the raw `NVARCHAR`, so under `load_method: hash` existing rows are not
seen as changed. Run the table once with `forced_full_load` to repair its history.
The measurements, and the one case it is not for (an N-typed primary key), are in
[Backward compatibility](docs/legacy-compat.md).

## Columns added to the target

Three columns are added to the target and the dbt layer depends on them:
`_deleted_in_source`, `_timestamp` and `row_hash`. Which of them a run actually
maintains depends on the load method — see [Load methods](#load-methods). The names
of the other columns do not change — including the underscore prefixes (`_type`,
`_name`, `_date`) that `mage_ai.io.postgres` produces today.

## Running the tests

Most of the suite needs nothing but a checkout:

```bash
make install
make test          # ~880 tests, no database required
```

The rest needs databases. One command brings up the target and all four
sources, seeded:

```bash
cp .env.example .env
make db-up
make test          # now ~1100 tests, nothing skipped
make db-down
```

Anything without a DSN skips rather than fails, so working on one dialect does
not require running the other three. CI requires them to actually run — a test
that quietly skips is a test that stopped protecting anything.

The seeds in `docker/seed/` are not sample data. Each one is built around the
types that behave neither like numbers nor like text, because that is the bug
class string assertions cannot reach: MySQL's zero date and its `TIME` that
legitimately exceeds 24 hours, MSSQL's three date/time families and `MONEY`,
Firebird's negative `NUMERIC` scale and space-padded `CHAR`.

## Where things are

| file | what for |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | module layout, `SourceDialect`, migration order |
| [`docs/mage-loader-block.md`](docs/mage-loader-block.md) | the `data_loader` block that calls `run()`, and the frame it returns |
| [`docs/partitioning-existing-table.md`](docs/partitioning-existing-table.md) | converting a table that already exists to a partitioned one |
| [`docs/legacy-compat.md`](docs/legacy-compat.md) | older configuration shapes, still read |
| [`docs/golden-test.md`](docs/golden-test.md) | how the core is verified against the predecessor |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed in which version and what it breaks |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | the three constraints that decide most patches, and how to set up |
| [`SECURITY.md`](SECURITY.md) | credential handling, SSH host keys, where to report |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before writing code. Three constraints
decide more patches here than the design does: the runtime cannot be raised,
the configuration contract is frozen, and target column names cannot change.
None of them are visible from the code alone.

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The reserved-word
list and the column-name cleaning behaviour reproduced in `src/dbextractors/core/naming.py`
originate in [Mage](https://github.com/mage-ai/mage-ai), also Apache-2.0; the
attribution is in `NOTICE`.
