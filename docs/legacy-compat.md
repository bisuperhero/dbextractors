# Backward compatibility

`dbextractors` grew out of a set of hand-copied extractor blocks that had been
running in production for years. Those blocks were never written to a shared
specification — each one accumulated its own keys, its own fallbacks and its own
quirks — and several hundred pipeline configurations still use them.

The package reads all of it. **The configuration contract is frozen:** an
existing key is never renamed and never changes meaning. New capability arrives
as a new key with a default that preserves today's behaviour.

For a new pipeline you do not need anything on this page — see the
[README](../README.md) instead. This document exists so that the older shapes
are documented somewhere other than the code, and so that nobody deletes a
branch that looks dead but is not.

Three predecessor variants are referred to throughout as **variant A**,
**variant B** and **variant C**. They correspond to three independent
deployments whose extractors diverged; where they disagreed, the winner and the
reason are recorded here.

---

## Top-level keys

The current shape puts everything into three sections — `TABLE`,
`LOAD_SETTINGS` and `SOURCE_DB`. Older configurations put some of it at the top
level instead. Those keys are still read, as a **fallback**: they apply only
when the section is missing, or when the key inside it is absent.

| top-level key | falls back for | note |
|---|---|---|
| `CONNECTION_PARAMS` | the whole `SOURCE_DB` section | used verbatim when `SOURCE_DB` is absent |
| `SSH_PARAMS` | the `ssh_*` keys of `SOURCE_DB` | supplies defaults, does not override |
| `BATCH_SIZE` | `LOAD_SETTINGS.batch_size` | |
| `EMPTY_ROWS_OK` | `TABLE.empty_rows_ok` | |
| `OUTPUT_SCHEMA` | `TABLE.output_schema` | |
| `OUTPUT_TABLE` | `TABLE.output_name` | `output_table` is also accepted inside the section |
| `TABLE_NAME` | `TABLE.source_name` | |
| `PRIVATE_HOST` | `SOURCE_DB.host` | |
| `DEBUG` | verbose logging | enables `show_debug` |

### Two different fallback rules, on purpose

Credentials (`user`, `password`, `database`, `port`) use
`source_db.get(key, legacy.get(key))` — the fallback applies only when the key
is **entirely absent**, not when its value is empty. An empty password is a
password.

`host` uses `or` instead: an empty string behaves like a missing value. A
configuration that carries `host: ''` means "not filled in", and the useful
thing to do is fall through to `PRIVATE_HOST` rather than try to connect to an
empty host.

`port` and `charset` are deliberately **not** given a dialect default here.
Which port MySQL listens on is knowledge that belongs to the dialect, not to
configuration parsing.

### Unknown keys warn, they do not fail

A key that no section recognises is logged and ignored. This matters because the
package serves several deployments at once: a key added for one of them must not
bring down the others. Early on this was strict, and it silently prevented 24 of
one deployment's pipelines from starting.

---

## Legacy shapes inside `LOAD_SETTINGS`

### `incremental_date_column` and friends

Variant B expresses an incremental window with its own key set:

    incremental_date_column
    incremental_date_column_fallback
    incremental_parent_table
    incremental_parent_date_column
    incremental_parent_date_column_fallback

Everything else uses `updated_at_column` / `created_at_column`. Both are read
and the shape decides which path runs.

One asymmetry is deliberate and easy to get wrong: with the
`incremental_date_column` shape the fallback column is **not** added to the set
of columns the strategy forces into the selection, because the predecessor did
not add it either. Adding it would give the target one more column than it has
today — and target column names are the one thing that must not change, because
a large dbt layer is built on them.

### `partition_by_source`

Superseded by `partition_by`, kept working. `partition_by_source: true` means
exactly `partition_by: {column: _source, mode: list}`.

### `primary_key_column`

An alias for `primary_column`. Parsing merges the two; `primary_column` wins
when both are present.

---

## Keys that are accepted and do nothing

Four keys of the frozen contract are parsed, validated and then read by nobody.
They are **not removed** — a pipeline that sets one has to keep starting, and
several do — but a key that silently has no effect is a trap for whoever sets it
expecting one, so it is written down here and at the field itself in
`core/config.py`.

| key | why nothing reads it |
|---|---|
| `pagination_mode` | there is no page-by-page mode left to select. Every dialect streams one cursor through `iter_batches`, and keyset only comes into it when a dropped connection has to be resumed (`core/reading.py`). |
| `keyset_pagination` | whether a read can be resumed by keyset is decided by `full._keyset_usable` — from the primary key and the dialect's `FEATURE_KEYSET` — not by this key. |
| `conflict_columns` | the upsert key is the primary key: `incremental._upsert_from_staging` renders `ON CONFLICT (primary_column)` and `target_pg.replace_by_key` deletes by the same column. Widening it changes what an upsert matches on, which needs a golden run rather than a quiet wiring-up. |
| `num_parallel` | nothing in the package runs two reads at once. Parallelism is the orchestrator's — three pipelines side by side, which is why the memory peak is already tripled (`resolve_batch_size`). |

Setting any of them is harmless: the value is parsed, no warning is logged, and
the run behaves as if the key were absent. If one of them ever grows an
implementation, its default has to keep meaning what the key means today —
nothing.

---

## `TARGET_DB` is not read

342 configuration blocks declare a `TARGET_DB` section, passwords included. Not
one of the predecessor extractors ever read it — the write target has always
come from Mage's `io_config.yaml`, profile `warehouse`.

The section is therefore ignored rather than honoured. Honouring it now would
redirect writes for every one of those pipelines at once, which is the opposite
of backward compatible. Where the target needs to move, use `TARGET_PROFILE` in
the configuration or the `DBX_TARGET_DSN` environment variable.

---

## `parent_incremental` does not treat an empty target as a full load

`incremental` and `id_watermark` switch to a full load when the target table
exists but holds no rows — there is no window and no watermark to build on.
`parent_incremental` does not: it checks that the child and the parent tables
**exist** and nothing further, so over an empty target it transfers the parent's
window alone and the run comes out green with a partial table.

The asymmetry is deliberate. The predecessor (`load_incremental_by_parent`,
variant B's Firebird extractor) gates on `to_regclass` and has no row count
anywhere, so adding a fallback would transfer a different set of rows than the
old side does. Parity outranks the symmetry between our own strategies. The
behaviour is pinned by
`tests/strategies/test_parent_incremental.py::test_an_empty_target_is_not_backfilled`;
closing the gap means changing that test on purpose.

---

## MSSQL: `NVARCHAR` text arrives truncated

**Symptom.** A Czech `NVARCHAR` value in an MSSQL source lands in the target as
its first character. `'příliš žluťoučký kůň'` becomes `'p'`. There is no error,
no warning and no replacement character; the row count matches and the column is
one letter long.

**It is not the seed and it is not the server.** The bytes in the column are
correct UTF-16 (`LEN()` returns 20, `CONVERT(…, VARBINARY(…), …)` shows the full
string). The loss happens on the way out.

**Cause.** The connection is opened with `charset='cp1250'`
(`MSSQLDialect.connect_args`). With pymssql 2.3.13 that combination is
internally inconsistent: the driver converts the server's UCS-2 to the
single-byte charset as **latin-1**, and pymssql then decodes those bytes with
the real **cp1250** codec. Two consequences follow from the one mismatch:

| the value contains | what arrives |
|---|---|
| ASCII only | intact |
| a character in latin-1 (`ø`, U+00F8) | a **different** character (`ř`) — same byte, other code page |
| a character outside latin-1 (`ř`, `š`, `ž`, `ť`, `ů`, `ň`, Cyrillic, emoji) | the string is **cut** at that character |

Czech text is the bad case: `ř š ž ť ď ň ů Ě Č Ř Š Ž` are all outside latin-1,
so a Czech value is usually cut within the first few characters. `NVARCHAR`,
`NCHAR` and `NTEXT` behave identically — the conversion is a property of the
connection, not of the column type. Single-byte `VARCHAR`, `CHAR` and `TEXT` are
unaffected.

**It is inherited.** Both predecessor MSSQL extractors open the connection the
same way — one with `connect_args={'charset': 'cp1250'}`, the other with
`connect_args={'charset': 'cp1250', 'login_timeout': 30}` — and neither carries
a comment explaining why.
The target tables have therefore been receiving the truncated text for as long
as they have existed. Restoring the full value is not a bugfix but a **change of
the target's contents relative to the old side** — it breaks golden parity, and
it is the maintainer's call rather than a silent improvement.

**Why the charset is there.** The source databases are CP1250-collated and most
of their text is legacy single-byte `VARCHAR`. Those columns are read correctly
*because* of this setting, and only because of it. Setting the connection to
`UTF-8` inverts the problem exactly:

| connection charset | `NVARCHAR` | legacy `VARCHAR` |
|---|---|---|
| `cp1250` (today) | truncated | **correct** |
| `UTF-8`, or omitted | **correct** | mojibake — `'pøíli\x9a \x9elu\x9douèký kùò'` |
| `ISO-8859-2` | truncated | mojibake |

No charset value gets both right; `cp1250`, `CP1250`, `WINDOWS-1250`,
`ISO-8859-2`, `UTF-8`, `utf8` and the parameter's absence were all measured
against the same stored bytes.

**The way out: `LOAD_SETTINGS.convert_nchar_to_varchar`.** Off by default, so
nothing changes for a table that does not ask for it. With it on, the generated
`SELECT` wraps the N-typed columns — and only those — in
`CONVERT(VARCHAR(MAX), …)`, so the server converts them through the column's own
collation and they arrive whole over the same `cp1250` connection:

| the value contains | key off | key on |
|---|---|---|
| ASCII only | intact | intact |
| CP1250 text (`ř`, `š`, `ž`) | **cut** at the first such character | intact |
| a character CP1250 folds (`ø`) | a *different* character (`ř`) | folded onto its base letter (`o`) |
| a character CP1250 cannot express (Cyrillic, emoji) | **cut** at that character | `?`, and the rest of the string kept |
| single-byte `VARCHAR`/`CHAR`/`TEXT` | intact | intact — never wrapped |

The wrap is driven by the **introspected column type**, never by a name: wrapping
a legacy single-byte column would corrupt exactly the columns the `cp1250`
setting gets right, which are the majority of this source.

Three things to know before switching it on:

- **The table has to be reloaded.** The key changes what those columns hold, but
  `row_hash` is computed by the server from the raw `NVARCHAR` and does not move
  with it — the digests in the target were always right, it was the text beside
  them that was truncated. Under `load_method: hash` an existing row is therefore
  *not* seen as changed and not rewritten; run the table once with
  `forced_full_load` (or `load_method: full`) to repair its history.
- **Not for an N-typed primary key.** Resuming a dropped read and `hash_diff`'s
  key filter both compare a value taken from a batch against the unconverted
  column in the source, so a converted key would stop matching itself. Keys in
  this source are numeric; the case is called out rather than guarded against.
- **It is per table, in `LOAD_SETTINGS`**, not in `SOURCE_DB` next to `charset`.
  `SOURCE_DB` describes the connection and is shared — all 16 pipelines against
  this source carry the same block — so a key there would invite flipping, and
  reloading, all sixteen tables at once. `SourceDbConfig` also never reaches the
  SELECT: it builds the engine URL and the tunnel and nothing else, while
  `LOAD_SETTINGS` arrives at every strategy as `ctx.settings`.

`SOURCE_DB.charset` could not have carried it in any case: all 16 pipelines have
`utf8mb4` in that key — the name of a *MySQL* encoding, which pymssql does not
know — so the dialect deliberately never reads it, and starting to would change
all 16 tables at once (`MSSQLDialect.build_conn_str`).

The route not taken was **a second connection charset for the N-types**, i.e.
reading them over a separate UTF-8 connection. That is correct for every
character rather than lossy, and costs two connections per table; it can be added
as another key if the `?` ever matters.

Pinned by `tests/dialects/test_source_db.py` under "the MSSQL client charset" and
"convert_nchar_to_varchar, the way out", against the `dbo.unicode_edge` fixture in
`docker/seed/mssql/001_types.sql`. Which columns get wrapped is settled in
`tests/dialects/test_mssql.py`, and the wiring from `LOAD_SETTINGS` to the SELECT
in `tests/strategies/test_nchar_conversion_wiring.py`.

---

## SSH host key checking

The predecessor opened its tunnel with `StrictHostKeyChecking=no` and
`UserKnownHostsFile=/dev/null`, so the far end was never verified. That remains
the default (`ssh_host_key_checking: off`) because changing it would break every
deployment whose host fingerprint was never recorded anywhere.

It is a genuine weakness and it is opt-out rather than opt-in only for
compatibility. New deployments should set `accept-new`. See
[README](../README.md#ssh-host-key-verification).

---

## Column naming

Target column names are produced by the same rules Mage's PostgreSQL exporter
used: names whose upper-case form appears in a list of 825 SQL reserved words
get an underscore prefix. That is where `_type`, `_name`, `_date`, `_hour` and
`_order` in the target come from — no source column starts with an underscore.

This is reimplemented rather than imported, so that an upstream change cannot
silently rename columns in hundreds of tables. The copy is checked against the
live library by a dedicated CI job; see [NOTICE](../NOTICE).
