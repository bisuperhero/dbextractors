# ARCHITECTURE.md — dbextractors

## Why it is shaped like this

Measured by a machine comparison of the 15 existing extractor files (25,723 lines):

| | identical lines |
|---|---|
| MySQL ↔ MSSQL | 87 % |
| MySQL ↔ PostgreSQL | 78 % |
| MySQL ↔ Firebird | 55 % |
| common core of MySQL+MSSQL+PG | ~69 % of significant lines |

And of 95 functions, **45 are identical everywhere** they appear. Splitting the code
into *a core plus thin adapters* is therefore not a design idea but a description of
what that code already looks like.

**The target side is always PostgreSQL** — it is the single largest block of code and
it is 100 % shared.

## Layout

```
src/dbextractors/
  entrypoint.py            run(config, dialect, logger) -> pd.DataFrame

  core/
    config.py              parsing and validation of TABLE / LOAD_SETTINGS / SOURCE_DB
    retry.py               with_retry, wait_for_port
    tunnel.py              SSH tunnel, keys, PDEATHSIG, connection_mode
    coerce.py              vectorised type conversion and sanitisation
    hashing.py             row_hash, choice of hashed columns
    naming.py              column name normalisation (reserved-word underscore prefixes)
    status.py              the returned DataFrame, logging, per-phase metrics
    target_pg.py           ← the largest module, 100 % shared

    strategies/
      base.py              LoadStrategy ABC
      full.py              shadow table + TRUNCATE and INSERT … SELECT in one transaction
      incremental.py       window by updated_at / days_back
      hash_diff.py         hash comparison without a CDC log
      id_watermark.py      advance by an increasing PK
      full_by_source.py    partitioning of the target by source (MSSQL only so far)
      parent_incremental.py window taken from a parent table (Firebird only so far)

  dialects/
    base.py                SourceDialect ABC
    mysql.py  mssql.py  postgres.py  firebird.py
```

The swap in `full.py` is deliberately **not** a `RENAME`. A view holds the OID of the
table it was built on, not its name, so after renaming the view stays bound to the old
relation and quietly serves data from the previous run. Loading into a shadow table and
then doing `TRUNCATE` + `INSERT … SELECT` in a single transaction keeps every dependent
view pointing at the same relation throughout, and an interrupted run leaves the target
untouched instead of truncated.

## Data flow

```
  config (from the calling block)
        │
        ▼
  core.config ── validation, filling in defaults
        │
        ▼
  core.tunnel ── direct | ssh | auto  →  SQLAlchemy engine
        │
        ▼
  dialects.<X> ── column introspection, type map, pagination SQL
        │
        ▼
  strategies.<Y> ── drives the loop: what to read, in what batches, what to write
        │            (calls the dialect to read, target_pg to write)
        ▼
  core.coerce ── vectorised sanitisation of the batch
        │
        ▼
  core.target_pg ── COPY into the shadow/target table, upsert, indexes,
                    _deleted_in_source, _timestamp
        │
        ▼
  core.status ── the returned DataFrame
```

The key division of responsibility:

- **The dialect knows nothing about the target.** It can say what columns a table has,
  how to read it in batches, and how its types map onto PG.
- **The strategy knows nothing about the SQL dialect.** It asks the dialect abstractly.
- **`target_pg` knows nothing about the source.** It is handed a DataFrame and column
  metadata.

Because of that, a new source is a new file in `dialects/` and nothing else.

## `SourceDialect` — the adapter interface

This is the only thing that has to be written for a new kind of database. A design, not
dogma — adjust it to what the comparison of the existing variants shows, but keep the
principle that the dialect does not know the target.

```python
class SourceDialect(ABC):
    name: str                      # 'mysql' | 'mssql' | 'postgres' | 'firebird'
    default_port: int
    type_map: dict[str, str]       # source type -> PG type

    def build_conn_str(self, params, host, port) -> str: ...
    def probe(self, host, port, timeout) -> bool: ...

    def introspect_columns(self, engine, database, schema, table) -> list[ColumnDef]: ...
    def estimate_size(self, engine, table, where) -> tuple[float, int]: ...

    def quote_ident(self, name: str) -> str: ...
    def render_select(self, columns, table, where, order_by) -> str: ...

    def source_ident(self, name: str) -> str: ...
        # a name that did not come from introspection (configuration, a default in the
        # code); Firebird upper-cases it, everywhere else this is quote_ident

    def supports(self, feature: str) -> bool: ...
        # 'keyset' | 'hash_diff' | 'partition_by_source' | 'parent_incremental'
```

**One deviation from this sketch:** `page_offset` and `page_keyset` were originally part
of the interface, but they were **deleted as dead code** — every path reads through a
single cursor via `pd.read_sql(chunksize=)`, so `LIMIT/OFFSET` is never built anywhere in
the package. The cursor is also faster and has none of the quadratic overhead of a
growing `OFFSET`. The `keyset` capability in `supports()` stays: `id_watermark` and `full`
use it, and it is an independent property. See the CHANGELOG.

`source_ident` was added instead: generated SQL mixes names that came from introspection
(which have the shape the source uses) with names that came from the configuration or
from a default in the code (which do not). The second group has to go through here, or
Firebird will not find them.

Notes from the existing code:

- **MSSQL** has `TOP (n)` and `OFFSET … ROWS`, not `LIMIT`. It is the only one that
  currently has multi-source and partitioning.
- **Firebird** is the most distant (55 % identical). It has no hash mode at all today,
  introspection goes through the `RDB$` tables, and incremental loading is resolved
  through a parent table. **Migrate it last.**
- **PostgreSQL as a source** additionally has `pagination_mode` and `id_watermark`.
- **MySQL** is 73 % of the volume — migrate it after the two smaller dialects, not first.

## Strategies

| strategy | when | tables today |
|---|---|---|
| `full` | small tables, or no usable key | ~90 |
| `hash_diff` | no CDC log, but a stable PK | ~530 |
| `incremental` | there is a reliable modification-time column | ~27 |
| `id_watermark` | append-only, increasing PK | 4 |
| `full_by_source` | one target table fed from several sources | 16 |
| `parent_incremental` | window taken from a parent table (Firebird) | 24 |

`hash_diff` is both the dominant and the most expensive one — today it is
**O(source rows, not changed rows)**, ~15,000 rows/s. Optimising this single strategy
touches ~530 tables.

## Distribution

A standalone package, installed with pip and pinned to a version:

```
# requirements.txt in the consuming repository
dbextractors==1.0.0
```

Extras pull in only the drivers a deployment actually needs, for example
`pip install "dbextractors[target,mysql]"`.

- The version is one line → upgrade and rollback happen per deployment, visibly in git.
- The package is tested outside Mage, in ordinary pytest.
- **The image build has to run in CI against a vendored wheel**, not `pip install` at
  build time from a network source — otherwise an unreachable index takes the deploy
  down.

## Migration order

By size of impact, smallest first:

| # | group | tables | why here |
|---|---|---|---|
| 1 | variant B / PostgreSQL | 21 | the smallest real dialect; fixes a known hash bug on the way |
| 2 | variant C / MSSQL | 16 | adds partitioning and multi-source |
| 3 | variant A / PostgreSQL | 58 | proves it scales |
| 4 | variant A / MySQL | 254 | |
| 5 | variant C / MySQL | 237 | including folding the `_v2` fork back into options |
| 6 | variant B / Firebird | 80 | the most distant dialect, last |
| 7 | retiring the older generation | 3 | `*_loader`, `*_universal_extractor` |

MySQL is 73 % of the volume but is deliberately only fourth. If the golden test turns out
to have holes, better that it shows on 21 tables.
