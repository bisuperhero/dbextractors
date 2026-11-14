# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
the versioning follows [semver](https://semver.org/).

Every release is tagged, and the tag is what a deployment pins in `requirements.txt`.
**Every change carries a note on what it breaks** — roughly 670 tables depend on
this package.

## [2.4.1]

### Fixed

- **A COPY that ran past `DBX_TARGET_COPY_TIMEOUT` now ends its server backend,
  so the retry does not land on the same one.** Through pgbouncer in transaction
  mode the next client gets the most recently released server connection (LIFO):
  in production three attempts of one table, each on a new client connection,
  stalled on the same backend at the same depth of the batch while every other
  backend worked normally — the retry could not help. In a transaction the COPY now
  runs under a savepoint (`dbx_copy`); on a timeout `copy_in` rolls back to it and
  runs `pg_terminate_backend(pg_backend_pid())`. The transaction is still open, so
  the backend is provably this client's and nobody else's. Reproduced and verified
  against PostgreSQL 17 + PgBouncer 1.25 (transaction mode): before, both attempts
  ran on one backend; after, the second got a new one and the pool kept working.
- **The reported error after such a timeout stays `CopyTimeoutError`.** The target
  connection is now a `TargetConnection` (`target_pg.connection_class()`, passed as
  `connection_factory`), whose `rollback()` does nothing once the connection is
  closed. A plain psycopg2 connection would answer the strategies' `rollback()`
  with `InterfaceError: connection already closed`, which replaced the timeout in
  the status row and the log.

### Breaks

- After a COPY timeout the target connection is **closed**, not just rolled back:
  the shadow and temporary tables go with the backend (the cleanup that follows logs
  a warning instead of dropping them). A caller that hands `copy_in` its own plain
  psycopg2 connection and keeps using it after a `CopyTimeoutError` now gets
  `InterfaceError`; use `connection_class()` or reconnect. Without
  `DBX_TARGET_COPY_TIMEOUT`, and on autocommit, nothing changes.
- Two more statements per bounded COPY (`SAVEPOINT`, `RELEASE SAVEPOINT`). The
  role needs no extra privilege: ending its own backend is always allowed.

## [2.4.0]

One target table assembled from several MySQL sources on different hosts, one
`run()` per source, several runs at once. Motivation: one MySQL cluster per tenant,
each with a database of the same name, reported on as one table with a source
column — without a proxy database of union views.

Verified on a development copy of that setup before the release: 15 tenant
databases × 17 tables (255 runs, three tenants at a time) into a new schema —
all 255 succeeded in 1 098 s wall time, row counts per tenant and table equal the
source in all 255 cases, a second run skipped all 255 by fingerprint (96 s), no
staging table was left behind.

### Added

- **`LOAD_SETTINGS.source_label` — the `_source` value of a run.** One run is one
  slice of a shared `full_by_source` target, named by the configuration instead of
  by the database: every tenant's database may be called the same. Given, the run
  is multi-source whatever it selected, and the label goes verbatim into
  `_source`, the partition name (as a slug) and the fingerprint ledger. Absent
  (the default), everything is as before: `_source` is the database name, and only
  in a multi-source run. Looping over the sources, their hosts and running them in
  parallel is the caller's. `ConfigError` for: another strategy than
  `full_by_source` (all others ignore the label and would rewrite the whole table —
  `forced_full_load` keeps `full_by_source`, so it passes), an empty or blank or
  non-string label, `DATABASES` with more than one entry, `multi_source: false`.
  Read from `LOAD_SETTINGS` only. A failure is logged as `🛑 [<label>]`.
- **`LOAD_SETTINGS.source_column` — the name of the column that carries the
  source** of a multi-source target (partition key, second column of the unique
  index, what a slice is deleted by). Default `_source`, as before. Found on real
  data: a source column named `source` (a reserved word) lands in the target as
  `_source` too, and the label overwrote its values in every row; with
  `source_column: '_system'` the source keeps its own `_source`. A plain lowercase
  identifier, not a managed column, `full_by_source` only (`ConfigError`
  otherwise). An existing partitioned target whose partition key is another column
  is refused with a `TargetError` — renaming it is a manual migration.
- **`full_by_source` for MySQL/MariaDB.** `MySQLDialect` now offers
  `FEATURE_PARTITION_BY_SOURCE`; the fingerprint is the generic one
  (`COUNT(*)`, `MAX(pk)`, `MAX(ts)`, `SUM(CAST(… AS DECIMAL(38,6)))`), verified
  live against MySQL 8.
- **`SourceExtractionError.statuses`** — the status row of every source of the
  failed run (the keys and order of `status.STATUS_COLUMNS`, error text redacted),
  so a caller can report which source failed and why. The message and the fact
  that no frame is returned on a failure are unchanged. New helper:
  `status.normalize_statuses`.
- **`DBX_TARGET_LOCK_TIMEOUT`** — an upper bound on lock waits in the DDL phase of
  a `full_by_source` run (see Changed), a PostgreSQL duration, default `10min`. An
  invalid value raises `TargetError` naming the variable.
- New `target_pg` helpers: `ddl_lock`, `lock_timeout`, `find_index_on_columns`,
  `schema_exists`, `fingerprint_store_exists`, `partition_ref_for`,
  `partition_source_value`.

### Changed

- **The DDL of a `full_by_source` run happens in one short transaction under an
  advisory lock** (`pg_advisory_xact_lock` keyed by `schema.table`, with
  `SET LOCAL lock_timeout`): the schema, the staging table, the partitioned parent,
  this source's partition, new and managed columns, the fingerprint ledger. Inside,
  everything is re-checked from the catalogue, so a run that waited finds what the
  previous one created. The slice swap (`TRUNCATE` of the own partition +
  `INSERT … SELECT`) takes no advisory lock — sources swap side by side.
  Transaction-scoped on purpose: the target sits behind pgbouncer in transaction
  mode. Measured lock levels (PG 17.10): `CREATE TABLE … PARTITION OF` and
  `ADD COLUMN` take `AccessExclusiveLock` on the parent, so they wait for running
  swaps and readers — bounded by `DBX_TARGET_LOCK_TIMEOUT` — and no longer sit
  inside the swap transaction; `TRUNCATE` of a partition locks only that
  partition.
- **The staging table is committed (empty) before the stream**, and the read phase
  (ledger, "does the target still hold the slice") is committed before the source
  is read. Before, the staging table was created `LIKE` the target inside the data
  transaction, which held an `AccessShareLock` on the target for the whole
  download, and another source's `ADD COLUMN` — and everything queued behind it —
  waited for it.
- `full_by_source` creates a **missing plain target** in the DDL phase, before the
  stream, rather than after it; a stream that then fails leaves an empty target
  (the next run fills it — no fingerprint was written).
- **Reading a fingerprint no longer creates the schema and the ledger**; a missing
  ledger means no fingerprint. The ledger is created in the DDL phase (and, as
  before, when one is written), under an advisory lock of its own because every
  table of the schema shares it. **A failure to store the fingerprint is a
  warning**, not a failed source: the data is already in the target; the next run
  fetches again.
- **`ensure_schema` no longer issues `CREATE SCHEMA` for an existing schema**, and
  creates a missing one under an advisory lock keyed by the schema — for every
  strategy.
- **The unique index is looked up by its columns, not by its name**, before
  `CREATE UNIQUE INDEX` (all strategies, `create_unique_pk_index`) and before the
  plain index over `_source` (`ensure_source_index`). The predecessor's
  multi-source targets have `idx_<t>_id_source_unique`, while the package's name
  for `(id, _source)` is `idx_<t>_id__source_unique`; asking by name
  (`IF NOT EXISTS`) would have built a second, identical index over the whole
  table on the first run after the fix below, holding a `ShareLock`. Only a valid,
  non-partial, expression-free index over exactly those key columns counts. On a
  partitioned parent the lookup also saves the `ShareLock` that
  `CREATE INDEX IF NOT EXISTS` takes even when the index exists — every
  one-source run is the "last source" and would otherwise queue behind the other
  sources' inserts. The index of a partitioned `full_by_source` target is created
  under the DDL lock.
- **Two labels with the same partition name fail with both names.** `A-B` and
  `a_b` both map to `<t>__a_b`. It failed before too — `CheckViolation: no
  partition … found for row`, with the first label's slice intact after the
  rollback — now it is a `TargetError` naming both labels, raised in the DDL phase
  before the stream.
- **A value that the target column's type cannot hold fails a labelled source**
  instead of being written as NULL. Verified before the change: a source sending
  `'not a number'` into a column the target has as `integer` loaded green, with
  NULL in the slice (the conversion layer converts to the target's types with
  `errors='coerce'`). Decided by the values, only for columns whose type family
  (text, integer, number, date, timestamp, …) differs from the target's: a text
  column holding only numbers loads as it always has, and an empty string is not
  counted (it becomes NULL in every column type). The message names the source,
  the column, both types and the count; nothing is written. **Only with a
  configured `source_label`**; the `DATABASES` loop logs the same finding as a
  warning and writes NULL as before. A `COPY` that rejects a value
  (SQLSTATE class 22) is re-raised the same way.
- **A source column that collides with the source column fails a labelled run**
  in validation, before anything is touched, naming the column and pointing at
  `source_column`. The `DATABASES` loop logs it as a warning and overwrites the
  values as before.
- **An unchanged empty source is skipped by its fingerprint.** Its slice can never
  be "held" by the target, so the check "does the target still hold the slice"
  failed for it on every run and the empty source went through the DDL phase and a
  `TRUNCATE` each night. A matching fingerprint with a count of zero and an
  existing target is now enough. Applies to the `DATABASES` loop too.
- `full_by_source` reads the fingerprint stream to the end instead of taking
  `next()`: on MySQL a suspended unbuffered stream counts as abandoned and was
  cancelled with `KILL QUERY` from a second connection, the first one discarded.

### Fixed

- **The unique index of a partitioned target is committed.** The partitioned branch
  of `create_unique_pk_index` left it in the caller's transaction; `full_by_source`
  called it after its last commit and the connection was then closed — the index
  was rolled back and never existed (present in the session, gone after `close()`).
  Affects every `partition_by_source` target, MSSQL included. The test that should
  have caught it checked on the same connection; the new ones check from a fresh
  one.
- **Concurrent `full_by_source` runs into a new schema or a new target no longer
  fail** with `UniqueViolation` on `pg_namespace_nspname_index` (concurrent
  `CREATE SCHEMA IF NOT EXISTS`) or `pg_type_typname_nsp_index` (concurrent
  `CREATE TABLE` of the parent). Different tables into one new schema, and runs
  racing for the shared fingerprint ledger, are covered too.

### Breaks

- **The unique index `(pk, _source)` now really appears on partitioned
  `full_by_source` targets** — the first run after the upgrade builds it over the
  whole table (`ShareLock` on the parent while it builds; seconds for 0.25–2.2 M
  rows). An index the predecessor left under another name is recognised by its
  columns and no second one is built. From then on, **a source with a duplicate key
  in its own slice fails** (`UniqueViolation` in the swap; its slice and the others
  stay as they were), where it used to load. When the build itself fails on
  existing duplicates it is only logged, as before. Check existing targets before
  upgrading:

  ```sql
  SELECT <pk>, _source, count(*) FROM <schema>.<table>
  GROUP BY 1, 2 HAVING count(*) > 1;
  ```

  Checked before the release on one deployment's four multi-source targets
  (0.22–2.05 M rows, 5–8 sources each), in production and on a development
  copy: no duplicates, and each already has the predecessor's unique index
  `idx_<t>_id_source_unique`, which `find_index_on_columns` recognises (verified
  on the copy) — no index is built there.

- A process killed mid-stream leaves its empty-or-partial `dbx_shadow_*` table
  behind (it used to vanish with the rollback); `target_pg.find_stale_shadows`
  lists them. A failure the process survives still drops it.
- Concurrent runs that used to fail on the catalogue now wait for one another's
  DDL, at most `DBX_TARGET_LOCK_TIMEOUT` (default 10 minutes), then fail with a
  `TargetError` that says what they waited for.
- An unchanged empty source of the `DATABASES` loop is now skipped instead of
  having its (already empty) slice truncated again — same target content, no DDL.
- Nothing else changes for an existing configuration: without `source_label` and
  `source_column`, `_source`, partition names, the ledger and fingerprint skipping
  of non-empty sources are identical;
  other strategies are untouched apart from the index lookup by columns and
  `ensure_schema`. 2.3.x logs `source_label` as an unknown key and ignores it — on
  MySQL it then fails on `partition_by_source` anyway, so an older version cannot
  quietly write the wrong label; pin `dbextractors>=2.4.0`.

## [2.3.0]

### Added

- **`DBX_TARGET_COPY_TIMEOUT` — an upper bound for one `COPY … FROM STDIN` into
  the target**, as a PostgreSQL duration (`300000` = ms, `300s`, `5min`). Seen in
  production: a COPY on one connection slowed down to ~10 kB/s, every batch of a
  full load took 15–16 minutes and the run hit the orchestrator's limit after five
  hours, while a fresh connection loaded the same table at full speed. With the
  bound the COPY raises `target_pg.CopyTimeoutError`, the load rolls back and the
  orchestrator's retry starts on a new connection. Two halves: the client stops
  feeding `copy_expert` once the bound is up (psycopg2 reads the server's answer
  only after the last `read()`, so a server cancel alone surfaced only after all
  data had been sent), and `SET LOCAL statement_timeout` covers the server's own
  share; it is put back to the default right after the COPY, so the swap and the
  rest of the transaction stay unbounded. `SET LOCAL`, not `SET` or a DSN option,
  because the target sits behind pgbouncer in transaction mode. All five COPY
  paths in `target_pg` go through the new `target_pg.copy_in`. An invalid value
  raises `TargetError` before the first statement.

  **Breaks:** nothing while the variable is unset. With it set, the only new
  statements are the two `SET LOCAL` around each COPY; on an autocommit connection
  only the client half applies.

- **`load_method: key_sync` — deletion reconciliation that reads only the primary
  key.** For a table whose content is kept current by `incremental` or
  `id_watermark` and whose deletions are not tracked. The source gets one
  streamed `SELECT <pk> FROM <table> ORDER BY <pk>`; the keys are `COPY`ed into a
  temporary table and `mark_deleted_in_source` (the step `hash_diff` ends with)
  sets `_deleted_in_source = TRUE` and `_timestamp` on target rows whose key left
  the source, and flips it back to `FALSE` for keys that returned. No row is read
  from the target or fetched from the source; a source key the target lacks is
  only counted (`new_in_source` in the phase metrics, next to `source_keys`,
  `target_rows`, `deleted_marked`, `revived`). A missing or empty target returns a
  `fallback_reason` and does nothing, a surrogate key raises `StrategyError`, and
  a failed scan or an empty key set against a non-empty target raises instead of
  marking (`empty_rows_ok` allows the latter). New helpers:
  `target_pg.SourceKeySnapshot`, `target_pg.mark_deleted_counts`.
- `SOURCE_DB.time_zone` (MySQL/MariaDB), see Changed.

### Fixed

- **The incremental window no longer depends on a MySQL-only literal outside
  MySQL.** The window expression is now built by the dialect
  (`SourceDialect.incremental_window_expr`): only a dialect with a zero date
  (MySQL/MariaDB) wraps the modification column in `NULLIF(col, '0000-00-00
  00:00:00')`; PostgreSQL, MSSQL and Firebird get a plain
  `COALESCE(updated, created)`. On PostgreSQL that literal is rejected whatever
  the data holds (`date/time field value out of range` /
  `invalid input syntax for type timestamp`). The legacy
  `incremental_date_column` window had a second PostgreSQL-only failure,
  `COALESCE(col, 0)` (`COALESCE types timestamp … and integer cannot be matched`);
  PostgreSQL now renders the equivalent `col >= cutoff`, since `NULL >= x` is
  already not true. MySQL, MSSQL and Firebird windows are character-for-character
  what they were. The cutoff itself (`days_back`, `incremental_lookback_hours`,
  the deep cutoff) is computed before the dialect is involved and is untouched.

### Changed

- **MySQL/MariaDB sessions are pinned to UTC.** Every new source connection now
  runs `SET time_zone = '+00:00'` after the existing network timeouts, unless
  `SOURCE_DB.time_zone` gives another offset or zone name (validated, since it is
  interpolated into the statement; default `+00:00`). A MySQL `TIMESTAMP` is
  stored as UTC and rendered in the session zone, so two servers configured with
  different zones used to return different text for the same instant.
  `SourceDialect.session_statements(time_zone)` is the new hook; other dialects
  return their `session_sql` unchanged.

### Breaks

- **`TIMESTAMP` columns read from a MySQL/MariaDB server whose session zone is
  not UTC now arrive in UTC.** The values written to the target for those columns
  change by the server's offset (a `Europe/Prague` server: one or two hours
  earlier). `DATETIME` columns are stored as written and do not move, and a
  server already running in UTC is unaffected. `row_hash` computed by the source
  (`SHA2` over the cast text) changes with it for a table with such a column, so
  under `load_method: hash` every row of that table looks changed on the first
  run — run it with `forced_full_load`, or set `SOURCE_DB.time_zone` to the
  server's own zone to keep the old values. `incremental` windows are compared
  against date literals and shift by the same offset, which a window measured in
  days does not notice.
- `load_method: key_sync` is a new value; nothing existing changes. A deployment
  that sets it on 2.2.x fails with `Unknown load_method`, so pin 2.3.0 before
  configuring it.
- `forced_full_load` rewrites `key_sync` to `full`, as it does every other method.
- The PostgreSQL legacy window (`incremental_date_column`) is rendered
  differently (see Fixed); it returns the same rows, and previously could not run
  at all.

## [2.2.1]

### Fixed

- **An upsert into a partitioned target no longer scans every partition.**
  `replace_by_key` deleted with `t.pk = s.pk` alone against a staging table that
  had no statistics, so the planner picked a hash join over a seq scan of the
  whole target. On a 61-partition production table one 5 M-row `id_watermark`
  checkpoint ran over 40 minutes spilling to disk — and with checkpoints that
  repeats every 5 M rows. The staging table is now `ANALYZE`d and the delete is
  bounded by the staging key range (`BETWEEN MIN AND MAX`, taken from staging in
  the target's collation), which turns each partition into an index range scan.
  Locally on 12 partitions × 3 M rows: 1.17 s → 0.2 ms. An empty staging table
  deletes nothing.
- **A resume by a `VARBINARY` key no longer reads nothing and reports success.**
  mysql-connector returns such a key as `bytes`, and `sql_literal` rendered it as
  the text `'b''6ac9…'''`, which sorts above every key starting with a digit. After
  a dropped connection the resumed query `WHERE id > 'b''…'''` returned no rows and
  the run ended as a success with the rest of the delta unread (production,
  2026-10-12: 5 M of 42.7 M rows). MySQL now renders bytes as a hex literal
  (`X'…'`), which compares byte-wise like the column; the other dialects raise
  `TypeError` instead of rendering a wrong literal. The watermark itself comes from
  the target and was not affected, so the next run picked up the rest.

### Changed

- **MSSQL `NVARCHAR`, `NCHAR` and `NTEXT` arrive whole with pymssql 2.4.4 —
  the driver changed, not this package.** pymssql 2.4.4 (2026-10-09, *"Fixed
  unicode encoding problem #854 & #1000"*) builds its Linux wheels with
  `glibc-gconv-extra` (GNU libiconv on musl), so the bundled FreeTDS can
  convert to the `cp1250` connection charset instead of falling back to latin-1.
  Measured against the `dbo.unicode_edge` fixture with the key off:
  `'příliš žluťoučký kůň'` arrives whole instead of as `'p'`, and a character
  CP1250 cannot express (`ø`, Cyrillic, emoji) becomes `?` instead of cutting
  the string or arriving as a different letter (`ø` → `ř`). 2.4.2 and 2.4.3 still
  truncate. The legacy single-byte `VARCHAR`/`CHAR`/`TEXT` and
  `convert_nchar_to_varchar` read identically on both. The dependency range
  `pymssql>=2.3.13,<3` admits 2.4.4, so a fresh install gets it.

  The live tests in `tests/dialects/test_source_db.py` now assert whichever
  behaviour the installed pymssql has, instead of pinning the truncation.

### Breaks

- **What lands in N-typed MSSQL columns depends on the pymssql that gets
  installed.** A deployment that resolves pymssql 2.4.4 or newer writes the whole
  text where it used to write the truncated one, for every table — with or
  without `convert_nchar_to_varchar`. `row_hash` is computed by the server from
  the raw `NVARCHAR` and does not move, so under `load_method: hash` existing rows
  are not rewritten and keep their truncated text next to new rows that are
  whole; run a table with `forced_full_load` to make its history consistent. An
  N-typed primary key changes its value along with the text. Pin
  `pymssql<2.4.4` in the deployment's requirements to keep the old contents.

## [2.2.0]

### Fixed

- **MySQL reads no longer pull the whole result into RAM.** Under SQLAlchemy 2
  the `mysqlconnector` dialect opens every cursor buffered, overriding
  `connect_args={"buffered": False}`, so `iter_batches` downloaded the entire
  result before yielding the first batch: an `id_watermark` delta of 42.7 M
  rows was killed at 14 GiB without a single batch. `MySQLDialect.iter_batches`
  now reads through an unbuffered cursor on the DBAPI connection (the session
  timeouts still apply). Measured on 3 M rows: first batch after 0.05 s instead
  of 2.73 s, peak RSS 112 MB instead of 1 150 MB. The frames are identical to
  what `pd.read_sql` produced (checked live, dtypes and cell types). A dropped
  connection is still reported as SQLAlchemy's `OperationalError`, so read
  recovery retries it as before; a stream abandoned half-way is cancelled with
  `KILL QUERY` instead of being read to the end. PostgreSQL, MSSQL and Firebird
  were measured as already streaming and are unchanged.

### Added

- **`id_watermark` commits a large delta in checkpoints.** Every
  `LOAD_SETTINGS.checkpoint_rows` rows (default 5 000 000) are projected into
  the target and committed. A failed run keeps what it committed and the next
  run resumes above it, instead of reading the whole delta again. The read is
  ordered by the key, so the target always holds every new row up to some key.
  `checkpoint_rows: 0` keeps the old single transaction; a surrogate key never
  checkpoints, because its read is not ordered.

### Breaks

- **A failed `id_watermark` run of more than `checkpoint_rows` rows is no
  longer all-or-nothing**: the checkpoints before the failure stay in the
  target. That is the point of the change; set `checkpoint_rows: 0` for a table
  where a partial delta must never be visible.
- An abandoned MySQL read opens a second, short connection for `KILL QUERY`.

## [2.1.1]

### Fixed

- **Under Dagster, `INFO` records did not reach `context.log`.** 2.1.0 set the
  `dbextractors` logger to the level of the logger passed to `run(logger=...)`.
  Dagster's `context.log` is left at `NOTSET`, which on the package logger means
  "inherit" — so it inherited the root's `WARNING`, and progress, row counts and
  the rest of `INFO` were dropped before forwarding. Warnings and errors still
  came through. A host logger at `NOTSET` now receives everything and filters
  for itself.

### Breaks

- Nothing.

## [2.1.0]

The 1.0.x line (Mage) does not get this release.

### Added

- **`load_method: abra_watermark`**, meant primarily for ABRA ERP tables that
  have no column recording when a row was created or changed. It reads only the
  records whose ID counter is above the watermark in the target. An ABRA record
  ID is a base-36 counter written least significant digit first, followed by the
  identifier of the database the record was created in; the counter is reversed
  and compared byte by byte, per database identifier, on both sides. Like
  `id_watermark` it stores no state, sees neither changes nor deletions, and
  needs a periodic full load. Firebird, MSSQL, MySQL and PostgreSQL render it
  (`FEATURE_ABRA_WATERMARK`).
- **`LOAD_SETTINGS.abra_id_suffixes`** — the ABRA database identifiers the
  watermark is computed for, required by `abra_watermark`. A single string is
  accepted for one suffix; a number is refused, so that leading zeros survive.

### Changed — logging

- **Log levels say what the message is.** Under Mage nearly everything was
  logged as a warning, because that was what reached the pipeline log. Now the
  normal course of a run (strategy, window or watermark, row counts, batch
  progress, the swap, the tunnel) is `INFO`, debugging detail (SQL, hash
  expressions, type maps) is `DEBUG`, `WARNING` is kept for what a person should
  look at (a fallback to a full load, dropped columns, an ignored key, a retry,
  a failed cleanup) and `ERROR` for failures.
- **One logging path.** Every module logs to its own standard logger under
  `dbextractors`; the package adds only a `NullHandler`. `run(logger=...)`
  forwards the run's records to the given logger (Dagster's `context.log`, say)
  instead of being threaded through the code, so records from the tunnel, the
  target writes, retries and the dialects now reach it too — before, only the
  strategies' messages did. Forwarded records do not also propagate.
- The Mage `DictLogger` adapter (`core.logging.LoggerAdapter`, `adapt`) is
  removed; Mage stays on 1.0.x, which keeps it.

### Breaks

- **Anything that filters the log at `WARNING` sees far less**: progress and row
  counts are `INFO` now. Set the host's level to `INFO` to keep seeing them.
- `LoadContext.log` and `LoadContext.logger` are gone, and the internal helpers
  that took a `logger=` argument (`resolve_databases`, `build_context`,
  `compute_incremental_cutoff`, `partitioning.effective_spec`, `BatchProgress`,
  `reading.read_with_resume`) no longer do. `run()`'s signature is unchanged.
- No existing `load_method` changes behaviour, and a configuration without the
  new key parses as before. Under 1.0.x the key is only logged as
  unknown and `abra_watermark` is an unknown `load_method`, which raises — so a
  configuration shared with a Mage deployment has to keep `full` there.

## [2.0.0]

The runtime moves from the Mage 0.9.79 image to a current stack.
**The data does not change:** values and `row_hash` are what v1.0.2 produces
under pandas 1.5.3, and a `hash_diff` load over a target written by v1.0.2
rewrites nothing.

### Breaks

- **Python 3.10 is no longer supported**; the floor is 3.11. With it go the
  exact pins: SQLAlchemy 2 (`>=2.0,<3`), pandas `>=2.2,<3` and numpy
  `>=1.26,<3` replace SQLAlchemy 1.4.54, pandas 1.5.3 and numpy 1.26.4, and the
  driver extras are ranges up to the next major version.
- **Mage stays on 1.0.x.** The Mage 0.9.79 image is Python 3.10 with pandas 1.5
  and SQLAlchemy 1.4, on which this release refuses to install. Pin
  `dbextractors>=1.0,<1.1` there.
- **Firebird is read through `firebird-driver`, not `fdb`.** The SQLAlchemy URL
  scheme changes from `firebird+fdb` to `firebird+firebird`, the `firebird`
  extra installs `sqlalchemy-firebird` and `firebird-driver`, and the system
  needs the Firebird 3+ client library (`libfbclient.so.2`); it still reads a
  2.5 server. A refused or lost Firebird connection now counts as temporary and
  is retried, as it already was for the other three sources — `fdb` reported
  it in a way SQLAlchemy could not tell apart from a SQL error.
- `mysql-connector-python` 9.1 or newer (CVE-2024-21272).

### Fixed — behaviour pandas 2 would have changed, kept as it was

- `row_hash`: an `object` column holding `bytes` is rendered `"b'abc'"`, as
  pandas 1.5 did; pandas 2 renders `'abc'` (and raises on bytes that are not
  UTF-8).
- Date parsing: a column is parsed value by value, as pandas 1.5 did; pandas 2
  infers one format from the first value and turns every value in another
  format into `NaT`. A column of aware datetimes with two UTC offsets (a
  `timestamptz` across a daylight-saving change) keeps every value;
  `"now"` is the UTC wall clock; a time-zone abbreviation of the machine's own
  zone no longer raises.
- Durations: `Y`/`y` is 365.2425 days and `M` a minute, and `A`/`a` is
  rejected, as in pandas 1.5.

## [1.0.2]

### Fixed

- **`read_with_resume`: a dropped connection before the first batch now restarts
  the query instead of failing the run.** The source in production is a MariaDB
  10.4.11 FederatedX proxy that dies and restarts silently (`exit=0`, 433
  restarts since 2025-07-08), most nights, and often within a couple of seconds
  of a query starting. Every one of dbextractors's `hash_diff` scans of
  `vw_usr_contracts` — the table the proxy dies fastest on — was failing there:
  the module refused to resume because it only ever tracked one flag
  (``saw_row``) for two different facts, "a key was captured" and "a batch was
  already handed to the caller", and a drop before the first row set neither.

  ``saw_row`` now means only the first of those two facts and feeds keyset
  resume as before; a new ``yielded_any`` flag tracks the second and gates a
  new kind of resume — restarting the identical query from scratch — which is
  safe **only** while nothing has gone out yet, because there is nothing yet to
  duplicate. Once a single batch has been yielded, a source with no usable key
  still fails the run exactly as before; that includes a batch that arrives
  empty or without the primary-key column, which previously left ``saw_row``
  ``False`` and could have been mistaken for "nothing yielded yet". Also
  benefits: the surrogate-key scan path in `hash_diff` (`pk_in_batch=None`)
  gains a retry it never had, as long as it fails before its first batch.

  Pinned by `test_a_dropout_before_the_first_row_restarts_from_scratch`,
  `test_a_dropout_before_the_first_row_restarts_even_without_a_key`,
  `test_batches_without_the_pk_column_still_block_a_later_restart` and
  `test_restart_from_scratch_is_bounded_by_attempts` in `tests/core/test_reading.py`;
  `test_without_a_key_there_is_no_resume` still pins the case that must keep
  failing.

- **The retry defaults for reading raised from `attempts=3, base_delay=2.0` to
  `attempts=5, base_delay=5.0, max_delay=60.0`.** The old numbers (inherited
  from the predecessor) wait ~2 s and ~4 s between attempts, so all three
  attempts landed inside the FederatedX proxy's own restart window and the run
  failed regardless of the fix above. The new numbers wait 5/10/20/40/60 s
  (~135 s total before jitter), long enough for the last one or two attempts to
  land after the proxy has come back. This is a behaviour change on the failure
  path for every one of the ~670 dependent tables: a source that drops the
  connection now takes longer to give up before failing the run. Two new
  `LOAD_SETTINGS` keys, `read_retry_attempts` and `read_retry_base_delay`, let
  one table override the new defaults without moving them for the rest; absent,
  both default to `None` and change nothing.

## [1.0.1]

### Fixed

- **`parent_incremental`: the race between the parent's run and the child's.**
  The strategy builds two windows from one cutoff, but over two different
  tables: the condition against the source narrows the child by the **live**
  parent, while the `DELETE` narrows the target by the parent's **copy**, which
  its own, separate pipeline refreshes. Correct a row in the source after the
  parent's run has finished and before the child reads the source — in
  production a gap of roughly 35 minutes — and that parent is inside the first
  window and outside the second. The rows are read again, the old ones are not
  deleted, and `_insert_from_staging` (deliberately without `ON CONFLICT`) hits
  the unique index:

      duplicate key value violates unique constraint "idx_<table>_id_unique"

  Retrying cannot help, because the copy stays stale until the parent runs
  again — hence the observed pattern of three consecutive failures followed by a
  green cycle. Seen twice on `receivedorders2` against ABRA (16 and 19 August
  2026), each time on a correction landing in that gap; the `created_at`
  fallback does not catch it either, because the orders themselves are older
  than the window.

  The `DELETE` now has a second branch, `parent_id IN (SELECT DISTINCT
  parent_id FROM <staging>)`, which removes exactly what is about to be
  inserted. The original branch stays for the case only it covers: a parent
  inside the window whose children have all vanished from the source leaves
  nothing in staging, and its stale rows still have to go. Both remain in the
  one transaction, and the missing `ON CONFLICT` keeps its meaning — a conflict
  now really does mean the `DELETE` did not take effect. Pinned by
  `test_a_parent_copy_lagging_behind_the_source_does_not_collide`, which
  reproduces the production error without the fix.

  Two things deliberately not done: raising `incremental_lookback_hours` (it
  would have to be weeks and it inflates every run), and adding `ON CONFLICT DO
  UPDATE` on its own (it would also mask a `DELETE` that genuinely failed).

  **What breaks: nothing.** No configuration key changes and no data moves that
  did not move before — the extra branch deletes only rows the same transaction
  re-inserts. It does cost one more index scan of the staging table per run.
  There was no data loss from the bug either: the whole transaction rolls back,
  and the next successful cycle catches up. What it cost was failed runs, noise
  in error tracking, and a window in which the child table lagged behind its
  parent.

## [1.0.0]

### Added

- **`LOAD_SETTINGS.convert_nchar_to_varchar` — MSSQL `NVARCHAR` text that arrives
  whole.** MSSQL is read over a `cp1250` connection that pymssql and FreeTDS
  disagree about: the bytes are produced as latin-1 and decoded as cp1250, so an
  `NVARCHAR`/`NCHAR`/`NTEXT` value is cut at the first character latin-1 cannot
  express (`'příliš žluťoučký kůň'` → `'p'`) and one that latin-1 *can* express
  comes back as a different character (`ø` → `ř`). Documented under *Known
  limitations* below; the charset cannot simply be changed, because it is exactly
  what makes the legacy single-byte columns of a CP1250-collated source read
  correctly, and seven candidate values were measured with none right for both.

  With the key on, the N-typed columns — and only those, chosen by their
  introspected type — are wrapped in `CONVERT(VARCHAR(MAX), …)` in the generated
  `SELECT`. The server then converts them through the column's own collation and
  they arrive whole over the same connection; what CP1250 cannot hold is degraded
  rather than truncating the value (Cyrillic and emoji become `?`, `ø` is folded
  onto `o`). Legacy `VARCHAR`/`CHAR`/`TEXT` are never wrapped. Verified against a
  live MSSQL in `tests/dialects/test_source_db.py`, on the `dbo.unicode_edge`
  fixture, with every assertion having a counterpart for the key off.

  It lives in `LOAD_SETTINGS` rather than in `SOURCE_DB` beside `charset` because
  it is a property of one table's load, not of the connection — `SOURCE_DB` is
  shared by all 16 pipelines against this source, and `SourceDbConfig` never
  reaches the SELECT in the first place.

  **What breaks: nothing.** The default is `False` and the generated SQL is then
  byte-identical to before, pinned by a test. Enabling it for a table is a
  deliberate act with a cost: the contents of those columns change, while
  `row_hash` does not move with them (the digests were always computed by the
  server from the raw `NVARCHAR`), so under `load_method: hash` existing rows are
  not seen as changed — that table needs one `forced_full_load` to repair its
  history. Not for a table whose primary key is itself N-typed; see
  [Backward compatibility](docs/legacy-compat.md#mssql-nvarchar-text-arrives-truncated).

- **`LOAD_SETTINGS.incremental_parent_key_column` and
  `incremental_parent_id_column` — the `parent_incremental` join is configurable.**
  `ParentIncrementalStrategy` has always read both keys, but neither was part of
  the contract: `entrypoint._settings_dict` builds `ctx.settings` with
  `dataclasses.asdict`, so a key that is not a field of `LoadSettingsConfig` could
  never arrive, the hard-coded `parent_id` / `id` always won, and `config.parse`
  logged the key as unknown on top. The strategy's own error message said *"Set
  incremental_parent_key_column."* — advice that could not be followed.

  **What breaks: nothing.** The defaults are the two constants the strategy used
  anyway, which are what the predecessor hard-codes, so a configuration that does
  not name them loads exactly as it did. What changed is that naming them now
  works — for a child table whose foreign key is not called `parent_id`, or a
  parent not keyed by `id`.

- **`LOAD_SETTINGS.strict_integer_precision` — large integers fail loudly instead
  of being silently rounded.** An integer column that also holds a `NULL` is
  promoted to `float64` by `pd.read_sql` before this package sees the frame, and
  `float64` has a 53-bit mantissa: measured end to end on all four dialects, a
  source value of `9007199254740993` lands in the target as `9007199254740992`,
  and `1234567890123456789` as `1234567890123456768`. The row count is right and
  the run is green. The trigger is the `NULL`, not the magnitude.

  The rounding itself is **not** fixed and must not be — the predecessor loses the
  same bits in the same place, so repairing the read would change what lands in
  the target relative to it, and because the hash is rendered out of the frame it
  would recompute `row_hash` for every table with a nullable numeric column and
  force a full reload of all of them. Pinned in
  `tests/coerce/test_int64_precision.py`.

  With the new key on, a `float64` column bound for an integer column in the
  target that holds `|value| > 2**53` raises `coerce.IntegerPrecisionError`,
  naming the column, an offending value and the key that produced the failure.
  The check is vectorised (one numpy comparison per column) and sits in
  `target_pg.prepare_export_df`, at the last point where such a column is still a
  float. The boundary is inclusive — exactly 2**53 passes.

  **What breaks: nothing.** The default is `False` and every pipeline that does
  not name the key writes exactly what it wrote before, rounding included. Enabled
  by default it would fail an unknown number of tables, so switching it on is a
  per-pipeline decision.

- **A test stack for all four sources** (`docker/compose.yml`, `make db-up`).
  PostgreSQL as the target plus MySQL, MSSQL and Firebird as sources, each
  seeded with a table built around the types that behave neither like numbers
  nor like text: MySQL's zero date and its `TIME` that legitimately exceeds 24
  hours, MSSQL's three date/time families and `MONEY`, Firebird's negative
  `NUMERIC` scale and space-padded `CHAR`.

  Until now three of the four sources had **no live test at all** — only string
  assertions over generated SQL, which cannot catch a type that maps correctly
  and whose value still does not survive the trip. New markers `needs_mysql`,
  `needs_mssql` and `needs_firebird`; without a DSN those tests skip, and CI
  requires them to actually run.

  Also `.env.example`, which the `.gitignore` had whitelisted for a long time
  without the file ever existing.



- **`SOURCE_DB.ssh_host_key_checking` — verification of the SSH host key.**
  Until now the tunnel was opened with `StrictHostKeyChecking=no` and
  `UserKnownHostsFile=/dev/null`, that is **without verifying the remote end**.
  Inside a private network that is defensible, but it is the one place in the
  package where the connection can be spoofed — and the only one where the
  behaviour could not even be turned off::

      SOURCE_DB:
        ssh_host_key_checking: off | accept-new | strict

  `off` is the inherited behaviour, `accept-new` remembers the key on the first
  connection and refuses it once it changes, `strict` requires the host to be in
  `known_hosts` beforehand. An invalid value fails in `config.validate`, not later
  at the tunnel.

  **What breaks:** nothing. The default is `off`, i.e. today's behaviour — the
  configuration contract is frozen, and changing the default would take down runs
  against hosts whose key was never recorded anywhere. For new deployments
  `accept-new` is the recommended value.

- **`src/dbextractors/core/secrets.py` — one credential redaction for the whole package.**
  The password is interpolated into a string in seven places (four dialect
  `build_conn_str` implementations and three paths to the target), and in the
  implementations this package replaces only one of them was guarded — a single
  helper on the path to the target, protecting a single message. Everywhere else
  it was enough for the string to reach the text of an exception; typically
  through `create_engine`, which prints an unusable URL in full, password
  included, into the log.

  `secrets.redact()` handles a SQLAlchemy URL and a libpq DSN alike, and through
  `extra` also literal values that no pattern knows about. It **masks** the
  password (`password=***`) rather than discarding the whole token, so a redacted
  line still shows the shape of what failed.

  `entrypoint` wraps `create_engine` failures through the redaction and re-raises
  with `from None`. That last part matters more than it looks: chaining with
  `from err` would leave the original exception — carrying the full URL, password
  and all — attached as `__cause__`, and any handler that prints a whole traceback
  would put it in the log anyway. `target_conn.describe_dsn` delegates to the same
  redaction, so there is one implementation rather than two that can drift.

- **Every remaining path a credential could take out of the process now goes
  through that redaction.** The audit that closed the security pass triggered each
  candidate on purpose — a wrong password, an unreachable host, an invalid DSN, a
  session statement the server rejects, a tunnel that cannot bind — and read what
  actually came out. Three findings were real leaks and are fixed:

  - **A private key pasted into `SOURCE_DB.ssh_pkey` reached the log verbatim.**
    That key is contracted to be a *path*, but when it holds the key itself
    `os.stat` fails with `ENAMETOOLONG` and `core/tunnel.py` printed the whole
    value — at `ERROR` level, not gated on `DEBUG`. `ssh` then echoes its own `-i`
    argument back on stderr, which the package splices into the `RuntimeError`
    raised when the tunnel does not come up, and under `DEBUG` the argv is logged
    as well. `secrets.redact()` now masks any PEM private-key block, in every
    spelling `ssh` accepts and including one whose `-----END-----` never arrived.
    Of everything this package masks it is the only secret a changed database
    password does not retire.

  - **`psycopg2` quotes back the token it choked on in a malformed DSN.** A target
    password containing a space makes the DSN invalid *because of* the password,
    and the fragment quoted back is a piece of it: `invalid dsn: missing "=" after
    "…"`. `secrets.dsn_secrets()` extracts the password as written, whole and in
    pieces, for `redact(extra=…)`; `entrypoint._target_connection` and
    `golden.session.connect` both wrap the connect and re-raise redacted, with
    `from None`.

  - **`run()`'s per-database handler emitted an unredacted traceback** through
    `logging.exception`, and put `str(err)` into the Mage log, into the `error`
    column of the returned DataFrame and into `SourceExtractionError` — four exits
    for one string. The traceback is now rendered in the handler so that it *can*
    be redacted, and the redaction happens once, with the configured password in
    `extra`.

  Redaction was also added where nothing leaks today but the text is quoted
  wholesale and comes from a third party: `core/reading.py`, `core/retry.py`, the
  MSSQL and Firebird retry logs, `entrypoint._attach_session_sql`,
  `status.error_status`, `hash_diff`'s fallback reason, `golden/compare.py`'s
  `report.error` (which is written to the `--json` file on disk) and the
  malformed-format-string branch of the logger adapter, the one place in the
  package that `repr()`s log arguments it never inspected.

  All four drivers were checked live: none of psycopg2, mysql-connector, pymssql
  or fdb puts the connection string into a failed connection's message today.
  That is four third-party packages' current behaviour rather than a guarantee, so
  it is pinned by tests — a driver upgrade that changes it fails a test instead of
  filling a production log.

  **What breaks:** two exception *types* change on paths that previously raised a
  raw driver error. A failure to connect to the target raises
  `TargetConnectionError` instead of `psycopg2.OperationalError` /
  `ProgrammingError` (both are caught by `run`'s `except Exception`, so a pipeline
  sees no difference), and `golden.session.connect` raises `SessionError`, which
  `dbx-golden` already catches — the CLI now prints one redacted line and exits 2
  (`ERROR`, its documented code) instead of dumping a traceback and exiting 1. The
  `🛑 [db] extraction failed` record is emitted with `logging.error` and a rendered
  traceback rather than `logging.exception`, so the record no longer carries
  `exc_info`; the text is the same, chained `__cause__` frames included.

- **The characterisation oracle runs without the legacy extractor sources.**
  The characterisation tests used to need the predecessor's code; without it three
  test modules were not collected at all and another 154 tests failed.
  `tests/reference_oracle.py` now has two modes:

  - `live` — the legacy sources are available and answers are extracted from their
    AST as before. With `DBX_ORACLE_RECORD=1` every call is recorded along the way.
  - `replay` — the legacy sources are absent and answers come from frozen fixtures
    in `tests/fixtures/oracle/` (108 files, 1 359 calls, 904 kB).

  The mode is detected from whether that source tree is present; `DBX_ORACLE_MODE`
  forces it. Fixtures are recorded with `DBX_ORACLE_RECORD=1` where the
  predecessor sources exist, and CI runs the replay as a separate step so they cannot go
  stale.

  The format is JSON, not pickle: fixtures can be read by eye, a diff shows what
  changed, and loading one does not execute foreign code.

  **What is not replayed:** functions with a side effect outside their arguments
  (`ensure_private_key_permissions` changes file permissions) and functions that
  take a function as an argument (`with_retry`). They are listed in
  `reference_oracle.LIVE_ONLY` and their tests are skipped in `replay` mode —
  78 tests. It is an acknowledged gap, not a silent loss of coverage.

  **What breaks:** nothing. Where the legacy sources are available the behaviour is
  unchanged and the same 1171 tests pass.

- `scripts/golden_batch.py` carries its own `expand_env`, which it used to import
  from a migration helper built on top of the legacy extractor sources. That
  dependency pointed the wrong way: it made `golden_batch` — and with it
  `tests/golden/test_perturb.py` — impossible to import without the predecessor's
  code.

### Fixed

- **Three pieces of documentation that described something the code does not do.**
  None of them changes behaviour; all three were the kind of plausible, readable
  falsehood that costs an hour of looking in the wrong place.

  - `IncrementalStrategy` claimed to report a `stale_by_days` metric and to warn
    when it is exceeded. The name appears nowhere else in the package. The claim
    is removed rather than implemented, and the docstring now says why: detecting
    the hole an outage leaves needs the interval between two successful runs, and
    nothing records when a table last ran — every proxy the target can be asked
    for (`MAX(_timestamp)`, `MAX(updated_at)`) cannot tell a quiet table from one
    whose changes were missed, so it would warn on healthy tables for ever. The
    hazard itself is still documented; only the false promise is gone. The metric
    set is now pinned by a test, because this file had drifted twice.
  - `docs/mage-loader-block.md` listed six of the ten columns of the returned
    frame, omitting `load_method` and `error`. The table is corrected; "four extra
    columns" was right and stays.
  - `pagination_mode`, `keyset_pagination`, `conflict_columns` and `num_parallel`
    are parsed and read by no strategy. They **stay** in the contract — a pipeline
    that sets one has to keep starting — but each field in `core/config.py` now
    says it is accepted and inert, and `docs/legacy-compat.md` gained a section
    listing them with the reason. A key that silently does nothing is a trap for
    whoever sets it expecting an effect.

- **MySQL: a source without an explicit `charset` could not connect.**
  Configuration parsing deliberately does not fill a charset in — which
  encoding a database expects is the dialect's knowledge — so `SOURCE_DB`
  without one reached the dialect as an explicit `None`. `params.get("charset",
  "utf8mb4")` then returned that `None` rather than the default, the URL became
  `?charset=None`, and the driver refused it with *"Character set 'None'
  unsupported"*.

  Every MySQL pipeline that did not spell the charset out was affected. It went
  unnoticed because no test had ever built a connection against a live server;
  the very first one did. Firebird already used the `or` form and was never
  affected.

  **What breaks:** nothing — this only replaces a failure with a working
  default of `utf8mb4`, which is what the predecessors used.

First public release. The package itself is older than this version number —
everything before it happened in a private repository and is summarised under
`[0.1.0]` below. 1.0.0 is where the configuration contract becomes a public
promise rather than an internal one.


### Known limitations

- **MSSQL: `NVARCHAR`, `NCHAR` and `NTEXT` values are truncated at the first
  character latin-1 cannot express.** `'příliš žluťoučký kůň'` arrives as
  `'p'` — silently, with no error and no replacement character. Pure ASCII is
  unaffected, which is why most columns look healthy; a character that *is* in
  latin-1 arrives as a different character instead of being cut.

  The cause is the connection charset `cp1250`
  (`MSSQLDialect.connect_args`): with pymssql 2.3.13 the driver converts the
  server's UCS-2 as latin-1 while the client decodes the result as cp1250.

  **This is inherited, not new.** Both predecessor extractors open the
  connection with the same charset, so the affected target columns have held
  the truncated text since they were created. It remains the **default** on
  purpose: no charset value reads both the N-types and the legacy single-byte
  columns correctly, and `UTF-8` merely trades the truncation for mojibake in
  every legacy `VARCHAR` of a CP1250-collated source. Changing it silently would
  change the target's contents relative to the old side.

  A table can now opt out with `LOAD_SETTINGS.convert_nchar_to_varchar` (see
  *Added* above), which converts on the server instead of touching the charset.
  It stays off by default, so this remains the behaviour of every table that does
  not ask.

  **What breaks:** nothing — this is a description of behaviour that has not
  changed. The measurements behind the table, and what the opt-in costs, are in
  [Backward compatibility](docs/legacy-compat.md#mssql-nvarchar-text-arrives-truncated).

## [0.1.0] — pre-release history

Everything below this line happened before publication. The per-version notes from
that period were internal migration notes rather than a library changelog, so they
are summarised here instead of being released as public history.

The package replaces a set of hand-copied extractor blocks that had drifted apart
across several deployments while serving roughly 670 tables. Three constraints
drove the rewrite and still shape the code:

- **The configuration contract is frozen.** Hundreds of pipeline definitions keep
  working untouched, including older shapes that predate the sectioned
  configuration — see [Backward compatibility](docs/legacy-compat.md).
- **Target column names must not change.** The predecessor wrote through Mage's
  PostgreSQL exporter, which prefixes an underscore to any name whose uppercase
  form is one of 825 reserved words; a dbt layer is built on those names. The list
  is replicated in the package and compared against the live `mage_ai` inside the
  production image, so a Mage upgrade that touches it fails CI instead of quietly
  renaming columns.
- **`_deleted_in_source`, `_timestamp` and `row_hash` are maintained by every
  strategy**, not only by the ones that happened to support them before.

Capabilities arrived roughly in this order:

1. **Configuration, retry, SSH tunnel and the conversion layer.** Configuration is
   parsed into a typed dataclass and validated on input; the tunnel is a context
   manager with `PR_SET_PDEATHSIG` and an explicit `connection_mode`. The
   conversion layer is vectorised and 2.5x–3.1x faster than its predecessor;
   throughput is per row *times* column, so table width dominates — 67 000 rows/s
   at 8 columns, but only 8 600 at 48.
2. **Target column naming**, replicated from the Mage exporter and verified against
   a real target: 2 713 columns, 0 differences.
3. **The write path** (`src/dbextractors/core/target_pg.py`) on bare psycopg2. `COPY … FROM STDIN`
   is the only write path and reaches 260 000 rows/s; a full load goes into a
   shadow table swapped in a single transaction, so views survive and an
   interrupted run leaves the target untouched; deduplication runs through a
   temporary table instead of a Python `set` holding every key.
4. **Vectorised `row_hash`** — 8.2x to 9.0x faster than the per-row version and
   bit-for-bit identical to it.
5. **The six load strategies**: `full`, `incremental`, `hash_diff` (the dominant
   one, serving ~530 of the ~670 tables, computing its diff in SQL rather than in
   RAM), `id_watermark`, `parent_incremental` and `full_by_source`.
6. **The golden test** (`dbextractors.golden`, CLI `dbx-golden`), which compares
   two target tables on five levels — row counts, column names and order, types,
   per-column checksums and per-row `row_hash` — and returns a verdict. No function
   counted as finished until a golden test for it existed and passed.
7. **Dialects**: MySQL and PostgreSQL first, then MSSQL and Firebird, each verified
   against a live source. The largest verified comparison is 25 334 772 rows with
   no unexplained differences.
8. **Fixes that only running against live sources could find**: the target is
   resolved from `io_config.yaml` rather than from environment variables, a
   connection lost mid-read is retried and resumed from the last key that got
   through, a source that has gained a column no longer fails the load, and
   `_timestamp` is accepted as either `text` or `timestamp`.
9. **Partitioning of the target table** through `LOAD_SETTINGS.partition_by`
   (`list`, `range_day`, `range_month`, `range_year`), with partitions created from
   the values the data actually contains.

Two behaviours changed on purpose relative to the predecessor. An unreachable
source **fails** the run instead of finishing green with zero rows, because a green
run with zero rows is indistinguishable from "nothing changed in the source" and
lets a table freeze unnoticed. A missing `primary_column` **fails** as well,
instead of silently degrading to a full load — the most expensive possible answer
to a typo in YAML.
