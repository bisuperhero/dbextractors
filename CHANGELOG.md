# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
the versioning follows [semver](https://semver.org/).

Every release is tagged, and the tag is what a deployment pins in `requirements.txt`.
**Every change carries a note on what it breaks** — roughly 670 tables depend on
this package.

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
