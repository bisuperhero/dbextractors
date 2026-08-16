"""``HashDiffStrategy`` — diff the source against the target via ``row_hash``, no CDC log.

The predecessor's ``load_hash_update`` (8 variants, 291 lines) is the
**dominant and the most expensive strategy**: it serves ~530 of ~670 tables and
is O(source rows, not changed rows) — measured at ~15,000 rows/s. Any
optimisation of this one strategy therefore shows up on most tables in the
package.

The principle: compute a ``row_hash`` for every source row — where the dialect
can do it on its own side (``SourceDialect.render_hash_expr``, requires
``FEATURE_HASH_DIFF``) — compare it with the hash stored in the target and write
only the changed rows. Keys that did not arrive in the current run are projected
into ``_deleted_in_source`` through the same temporary PostgreSQL table the diff
uses — **not** through a ``seen_keys = set()`` in RAM (2-3 GB at 11 M rows).

Firebird does not get this strategy (``FEATURE_HASH_DIFF`` is absent and
``render_hash_expr`` raises ``UnsupportedFeature`` there) —
``parent_incremental`` is its answer to the same class of problem.

Writes go through ``COPY ... FROM STDIN`` and nothing else.

## The one rule almost everything else follows from

    Seeding `row_hash` must use THE SAME computation path as the diff.

Why that is enough to know: the hash is computed either in the source (`SHA2` on
MySQL, `HASHBYTES` on MSSQL) or in pandas — and the two paths **do not agree**.
Measured against a real source: 4 of 51 tables match, and precisely those
without binary columns. A single fall onto the other path therefore marks the
whole table as changed.

Three concrete requirements follow:

1. **Seed during the full load.** When `load_method` is hash and a full load
   runs (first run or fallback), that full load must populate `row_hash`.
   Without it the target keeps NULL and the hash run falls back to a full load
   forever — in the predecessors that fix exists in exactly one file and it
   affected 55 of 58 PostgreSQL pipelines.
2. **The seed takes the same path as the diff.** On MySQL and MSSQL that means
   the source expression in the `SELECT`, not `compute_row_hashes`. Variant A
   seeds in pandas only because its diff computes in pandas too
   (`force_recompute=True`) — that is consistency, not a general rule.
3. **Seed before `decode_bytes_columns`.** The diff hashes raw values, so the
   seed has to as well; otherwise the hashes of binary columns drift apart. The
   predecessor has a comment about this in the same place.

`FullLoadStrategy` satisfies all three: it takes the hash from the source
expression (`_build_select`) and decodes bytes only afterwards (`_prepare`).
This strategy can therefore fall back to a full load with no further
arrangements — including when the diff shows that the hashes have drifted (see
`_reseed_reason`).

## The diff belongs in SQL, not in RAM

The predecessor holds **the entire target snapshot in memory** — a
`target_hashes = {}` filled by `fetchmany(50000)` over every row of the target
table — and then compares with `chunk.iterrows()`. Both are forbidden here, and
it is O(target rows) of RAM across ~530 tables, three of which the orchestrator
runs concurrently.

The target is in PostgreSQL anyway, and the source snapshot can be `COPY`ed into
PostgreSQL. The diff is then **a single JOIN** — `target_pg.SourceHashSnapshot`.
Four things from the predecessor fall away at once: the target snapshot in RAM,
`iterrows()`, the file of changed keys in ``/tmp`` and the CSV of live PKs. The
same temporary table doubles as the input for
`target_pg.mark_deleted_in_source`, so the source does not have to be read
twice.

## What `hash_diff_buffer_size` and `hash_download_batch_size` mean

Both keys are identical across the predecessors (13 copies, the same
expression) and both fall back to ``batch_size``. Their meaning there:

- ``hash_diff_buffer_size`` — after how many changed keys the RAM buffer was
  flushed to a file in ``/tmp``.
- ``hash_download_batch_size`` — how many keys go into one ``IN (…)`` when
  fetching the changed rows.

The second is adopted unchanged. The first has nothing left to govern, because
the list of changed keys never exists in RAM or in ``/tmp`` — it is the result
of a SQL query. The key is therefore adopted onto the **nearest thing with the
same meaning**: how many source rows are read before being `COPY`ed into the
snapshot. It still governs the peak memory of the diff phase, just one layer
lower down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import pandas as pd

from dbextractors.core import coerce, hashing, reading, secrets, target_pg
from dbextractors.core.strategies.base import (
    DEFAULT_BATCH_SIZE,
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
    apply_text_dtypes,
    fallback_full,
    resolve_batch_size,
    surrogate_provides,
)
from dbextractors.core.strategies.full import (
    FullLoadStrategy,
    _hash_column,
    _integer_columns,
    _is_truthy,
    _resolved_pk,
    _text_dtype_map,
    append_hash_expr,
    column_types,
)
from dbextractors.core.strategies.incremental import (
    _all_columns,
    _drop_columns_missing_in_target,
    _upsert_from_staging,
    _with_where,
)
from dbextractors.dialects.base import FEATURE_HASH_DIFF

#: Mismatch ratio above which the diff is taken as drifted hashes, not as changed data.
#: Taken from variant A's extractor.
RESEED_MISMATCH_RATIO = 0.95


@dataclass(frozen=True)
class TargetHashState:
    """What has to be known about the target **before** the source is touched."""

    rows: int
    null_hashes: int


class HashDiffStrategy(LoadStrategy):
    """Compare the source's ``row_hash`` with the target's and move only changed rows."""

    name = "hash_diff"
    #: What gets reported into the status frame. The predecessor returns
    #: ``'hash'`` and downstream code reads it — the class name is an internal
    #: matter, this is the contract.
    reported_method = "hash"
    required_features: frozenset = frozenset({FEATURE_HASH_DIFF})

    def validate(self, ctx: LoadContext) -> None:
        """Check the primary key and the dialect's ability to hash on the source side.

        **Deviation from the predecessor:** there, a missing ``primary_column``
        leads to a silent full load. A full load is the most expensive possible
        answer to a typo in YAML and costs hours on a 15 M row table — so here,
        as in `IncrementalStrategy`, it raises. The fallback stays for states
        that can only be recognised at run time (empty target, drifted hashes),
        not for a broken configuration.
        """
        super().validate(ctx)

        pk = _resolved_pk(ctx)
        if not pk:
            raise StrategyError(
                "hash_diff: primary_column (or primary_key_column) is missing from "
                "LOAD_SETTINGS. Without a key there is nothing to match source to target on."
            )
        if pk not in ctx.target_names and not surrogate_provides(ctx, pk):
            raise StrategyError(f"hash_diff: primary key {pk!r} is not among the table's columns.")
        if _hash_column(ctx) is None:
            raise StrategyError(
                "hash_diff: compute_row_hash=False disables the hash entirely, so there "
                "is nothing to compare. Use load_method 'full'."
            )

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)

        pk = _resolved_pk(ctx)
        hash_column = _hash_column(ctx)
        assert pk is not None and hash_column is not None  # guaranteed by validate()

        reason, state = self._precheck(ctx, pk, hash_column)
        if reason:
            return fallback_full(ctx, reason)
        assert state is not None

        hashed = hashing.compute_hash_columns(
            ctx.source_names, ctx.settings.get("hash_column"), ctx.table_cfg, ctx.settings
        )
        if not hashed:
            return fallback_full(ctx, "no column was left to hash")

        # A missing index must fail before the first source row is read —
        # otherwise it is only noticed after a scan of the whole table.
        target_pg.assert_unique_pk_index(ctx.target_conn, ctx.target, pk)

        ctx.log(
            "warning",
            "🚀 Hash diff: %s -> %s, the target has %s rows (%s without a hash).",
            ctx.source.name,
            ctx.target.qualified(),
            f"{state.rows:,}",
            f"{state.null_hashes:,}",
        )

        with target_pg.SourceHashSnapshot(ctx.target_conn, pk, hash_column) as snapshot:
            diff: Optional[dict] = None
            reason = self._scan(ctx, snapshot, pk, hash_column, hashed)
            if reason is None:
                snapshot.index()
                diff = snapshot.diff(ctx.target)
                self._log_diff(ctx, diff)
                reason = self._reseed_reason(ctx, diff, state)
            if reason is None and diff is not None:
                return self._apply(ctx, snapshot, diff, pk, hash_column)

        return fallback_full(ctx, reason or "the diff did not run")

    # -- steps -------------------------------------------------------------

    def _precheck(
        self, ctx: LoadContext, pk: str, hash_column: str
    ) -> Tuple[Optional[str], Optional[TargetHashState]]:
        """Target states in which a diff makes no sense and a full load runs instead.

        All of them come from the predecessor and all of them are recognisable
        **before** the source is read. Returns a ``(reason, state)`` pair; the
        reason is ``None`` when the diff should continue.
        """
        if _is_truthy(ctx.settings.get("ignore_hash")):
            # The predecessor handles this combination by rewriting load_method
            # to 'full' — a hash diff with the hash switched off has no way to
            # work.
            return "ignore_hash=True — no hash is computed, so there is nothing to compare", None

        conn = ctx.target_conn
        if not target_pg.table_exists(conn, ctx.target):
            return "the target table does not exist", None

        columns = _target_columns(conn, ctx.target)
        if pk not in columns:
            return f"the target has no primary key column {pk!r}", None
        if hash_column not in columns:
            return f"the target has no hash column {hash_column!r}", None

        state = _target_hash_state(conn, ctx.target, hash_column)
        if state.rows == 0:
            return "the target table is empty", None
        if state.null_hashes == state.rows:
            # Nothing would match and the diff would mark absolutely everything
            # as changed.
            return "the target's hash column is all NULL — the hashes have to be seeded", None

        return None, state

    def _scan(
        self,
        ctx: LoadContext,
        snapshot: target_pg.SourceHashSnapshot,
        pk: str,
        hash_column: str,
        hashed: List[str],
    ) -> Optional[str]:
        """Read ``(pk, hash)`` pairs from the source and pour them into the snapshot.

        Returns a reason for a full load, or ``None`` when the scan succeeded.

        A failed scan **is not a run error**: a long stream can fall over on a
        transient disconnect (MySQL 2013) and a full load makes it up. The
        predecessor logs it as a warning on purpose, so that it does not raise a
        false error — that is adopted, with the difference that the reason also
        ends up in the result's `fallback_reason`, so the status frame knows
        about it too.
        """
        pk_source = _source_name(ctx, pk)
        hash_in_pandas = ctx.dialect.hash_in_pandas
        if hash_in_pandas:
            # The source cannot compute a hash that matches what is in the target
            # (PostgreSQL), so the **values** have to be read and the hash
            # computed here. It is more expensive — a whole row is transferred
            # instead of a single fingerprint — but it is the only way to hit the
            # fingerprints that are actually in the target.
            #
            # **All** columns are read, not just the hashed ones.
            # `compute_row_hashes` looks for a common type over the **whole
            # frame** (to stay identical to `.apply(axis=1)`, see
            # `core/hashing.py`), so a narrower scan would produce a different
            # fingerprint for the same row than the full load's seed did. The
            # predecessor reads whole rows during the scan as well.
            columns = list(dict.fromkeys([pk_source, *ctx.source_names]))
            expr = None
        else:
            columns = [pk_source]
            expr = ctx.dialect.render_hash_expr(
                hashed, hash_column, {c.name: c.source_type for c in ctx.columns}
            )

        # The scan reads the **whole** table in a single query, so it stays open
        # longer than any other read and is the most exposed to a dropped
        # connection. It orders by the key so it can be resumed from the last one
        # that got through.
        def _build_scan_sql(after):
            condition = ctx.where
            if after is not None:
                key_expr = ctx.dialect.render_key_expr(pk_source, ctx.surrogate)
                bound = f"{key_expr} > {ctx.dialect.sql_literal(after)}"
                condition = f"({bound}) AND ({condition})" if condition else bound
            sql = ctx.dialect.render_select(
                columns,
                ctx.source,
                where=condition,
                order_by=[pk_source] if not ctx.surrogate else None,
                surrogate=ctx.surrogate,
            )
            return sql if expr is None else append_hash_expr(sql, expr)

        rows_scanned = 0
        try:
            for batch in reading.read_with_resume(
                ctx.dialect,
                ctx.engine,
                _scan_batch_size(ctx),
                build_sql=_build_scan_sql,
                pk_in_batch=pk_source if not ctx.surrogate else None,
                log=ctx.log,
            ):
                batch = batch.rename(columns=ctx.name_map)
                if hash_in_pandas:
                    # **The same steps in the same order as
                    # `FullLoadStrategy._prepare`.** `apply_text_dtypes` changes
                    # values (uuid and varchar to text), so a scan that skips it
                    # computes a different fingerprint from the seed — and the
                    # table gets transferred forever. A regression test covers
                    # this.
                    batch = apply_text_dtypes(batch, _text_dtype_map(ctx))
                    batch = coerce.normalize_memoryviews(batch)
                    batch = hashing.add_hash_and_timestamp(
                        batch,
                        hash_column,
                        force_recompute=True,
                        hashed_columns_override=[ctx.name_map.get(c, c) for c in hashed],
                    )
                rows_scanned += snapshot.add(batch)
        except Exception as err:
            # The reason is not only logged — it becomes `fallback_reason` in the
            # status frame `run` returns, so it leaves the process.
            safe = secrets.redact(err)
            ctx.log("warning", "⚠️ The source hash scan failed (%s).", safe)
            return f"the source hash scan failed: {type(err).__name__}: {safe}"

        ctx.log("warning", "🔢 Source scan: %s rows into the snapshot.", f"{rows_scanned:,}")
        return None

    def _log_diff(self, ctx: LoadContext, diff: dict) -> None:
        ctx.log(
            "warning",
            "📊 Diff: source %s rows, added %s, changed %s, unchanged %s.",
            f"{diff['source_rows']:,}",
            f"{diff['added']:,}",
            f"{diff['changed']:,}",
            f"{diff['matched']:,}",
        )

    def _reseed_reason(self, ctx: LoadContext, diff: dict, state: TargetHashState) -> Optional[str]:
        """Emergency brake against drifted hashes.

        When **not one** row matches and yet almost all of them differ, that is
        not a data change — it is a sign that the hash in the target was computed
        along a different path from the current one (a different dialect, a
        different set of columns, a different version of the expression).
        Transferring the whole table in batches through ``IN (…)`` would in that
        case be many times more expensive than simply pouring it in with a full
        load, and the hashes would stay drifted next time round too.

        A table in which every row really did change falls under the same
        condition. That is fine: a full load is the cheaper route to the same
        result there. Telling the two situations apart would require downloading
        the data anyway — which is precisely the work being avoided.

        Only variant A has this today; it is adopted as the standard.
        """
        if not state.rows:
            return None

        ratio = diff["changed"] / state.rows
        if ratio >= RESEED_MISMATCH_RATIO and diff["matched"] == 0:
            ctx.log("warning", "⚠️ %.1f %% of rows mismatch and not one matches.", ratio * 100)
            return (
                f"the hashes have drifted (mismatch {ratio:.2%}, 0 rows matched) — "
                "a full load will seed them again"
            )

        changed_total = diff["added"] + diff["changed"]
        if changed_total >= state.rows and state.null_hashes:
            # Fetching more rows in batches than the target holds in total is
            # always more expensive than a full load.
            return (
                f"{changed_total:,} rows are to be fetched, the target holds {state.rows:,} "
                f"and {state.null_hashes:,} of those have no hash"
            )

        return None

    def _apply(
        self,
        ctx: LoadContext,
        snapshot: target_pg.SourceHashSnapshot,
        diff: dict,
        pk: str,
        hash_column: str,
    ) -> LoadResult:
        """Fetch the changed rows, project them into the target and mark the deleted ones."""
        conn = ctx.target_conn
        to_fetch = diff["added"] + diff["changed"]

        target_pg.ensure_generated_columns(
            conn, ctx.target, include_row_hash=True, hash_column=hash_column
        )

        rows_read = rows_written = 0
        if to_fetch:
            rows_read, rows_written = self._download(ctx, snapshot, pk, hash_column)
        else:
            ctx.log("warning", "ℹ️ No new or changed rows.")

        # Deleted rows have to be marked even when nothing changed — a vanishing
        # row leaves no trace whatsoever in the hashes.
        marked_deleted = target_pg.mark_deleted_in_source(
            conn, ctx.target, pk, snapshot.table, source_label=None, pk_as_text=True
        )
        conn.commit()

        return LoadResult(
            rows_read=rows_read,
            rows_written=rows_written,
            load_method=self.reported_method,
            is_incremental=True,
            data_present=rows_written > 0,
            phase_metrics={
                "source_rows": diff["source_rows"],
                "added": diff["added"],
                "changed": diff["changed"],
                "matched": diff["matched"],
                "deleted_marked": marked_deleted,
            },
        )

    def _download(
        self,
        ctx: LoadContext,
        snapshot: target_pg.SourceHashSnapshot,
        pk: str,
        hash_column: str,
    ) -> Tuple[int, int]:
        """Fetch the rows whose key changed and project them with an upsert.

        The write goes through a staging table and a single
        ``INSERT … ON CONFLICT``, not through
        ``loader.export(unique_conflict_method='UPDATE')``, which internally
        issues row-by-row ``executemany`` calls.
        """
        conn = ctx.target_conn
        full = FullLoadStrategy()
        columns = _all_columns(ctx, hash_column)
        batch_size = _download_batch_size(ctx)

        # Types from the **existing** target override the guess from the source
        # catalogue: a column may already have been promoted to BIGINT, and
        # pulling it back down to INTEGER would break the write on the first
        # large value.
        types = column_types(ctx, hash_column)
        target_pg.load_pg_schema(conn, ctx.target, types, _integer_columns(types))

        columns = _drop_columns_missing_in_target(ctx, columns)
        staging = target_pg.create_shadow_table(conn, ctx.target, columns=columns)
        rows_read = rows_written = 0
        try:
            for keys in snapshot.changed_keys(ctx.target, batch_size):
                for batch in self._read_keys(ctx, keys, pk, hash_column, batch_size):
                    rows_read += len(batch)
                    export_df = full._prepare(ctx, batch, hash_column, types)
                    if export_df.empty:
                        continue
                    target_pg.copy_from_stdin(conn, staging, export_df, columns)
                ctx.log("info", "📥 %s rows fetched.", f"{rows_read:,}")

            rows_written = _upsert_from_staging(
                conn, staging, ctx.target, columns, pk, ctx.partition_spec(conn)
            )
            target_pg.drop_shadow_table(conn, staging)
        except Exception:
            conn.rollback()
            try:
                target_pg.drop_shadow_table(conn, staging)
                conn.commit()
            except Exception as cleanup_err:
                ctx.log("error", "⚠️ The staging table could not be cleaned up: %s", cleanup_err)
            raise

        return rows_read, rows_written

    def _read_keys(
        self,
        ctx: LoadContext,
        keys: List[str],
        pk: str,
        hash_column: str,
        batch_size: int,
    ) -> Iterator[pd.DataFrame]:
        """One batch of keys -> batches of rows from the source.

        The source computes the hash with **the same expression** as during the
        scan: `_build_select` assembles it from the same input as `_scan`. If the
        hash reached this point along a different route, a hash the next diff
        cannot find would be stored in the target — and the same rows would be
        transferred forever.
        """
        condition = ctx.dialect.render_key_filter(_source_name(ctx, pk), keys, ctx.surrogate)
        if ctx.where:
            condition = f"({condition}) AND ({ctx.where})"

        # The fetch goes through `full.read_batches`, so a dropped connection
        # resumes the remainder here too instead of failing the run. A batch of
        # keys is short, but there are many of them in a row and one falling over
        # is enough.
        yield from FullLoadStrategy().read_batches(
            _with_where(ctx, condition), hash_column, batch_size
        )


# --- Helper functions -------------------------------------------------------


def _source_name(ctx: LoadContext, target_name: str) -> str:
    """Target column name -> source name. A ``WHERE`` against the source needs the latter.

    The configuration gives the primary key under its source name, but the
    strategy works with it under the target name (a reserved word gains an
    underscore). For a query against the source it has to be translated back.
    """
    for source, target in ctx.name_map.items():
        if target == target_name:
            return source
    return target_name


def _target_columns(conn: Any, target: Any) -> set:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (target.schema, target.table),
        )
        return {r[0] for r in cur.fetchall()}


def _target_hash_state(conn: Any, target: Any, hash_column: str) -> TargetHashState:
    """Row count of the target and how many of those have no hash. In one pass."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*), count(*) FILTER (WHERE "
            f"{target_pg.quote_ident(hash_column)} IS NULL) "
            f"FROM {target_pg.qualify(target)}"
        )
        rows, nulls = cur.fetchone()
    return TargetHashState(rows=int(rows), null_hashes=int(nulls))


def _positive_int(value: Any, fallback: int) -> int:
    """A non-positive or non-numeric value is ignored, as in the predecessor."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _scan_batch_size(ctx: LoadContext) -> int:
    """How many source rows are read before being `COPY`ed into the snapshot."""
    return _positive_int(ctx.settings.get("hash_diff_buffer_size"), _base_batch_size(ctx))


def _download_batch_size(ctx: LoadContext) -> int:
    """How many keys go into one ``IN (…)`` when fetching the changed rows."""
    return _positive_int(ctx.settings.get("hash_download_batch_size"), _base_batch_size(ctx))


def _base_batch_size(ctx: LoadContext) -> int:
    """``batch_size`` including the legacy ``BATCH_SIZE`` fallback (docs/legacy-compat.md)."""
    return _positive_int(resolve_batch_size(ctx.settings, ctx.table_cfg, None), DEFAULT_BATCH_SIZE)


__all__ = ["HashDiffStrategy", "TargetHashState", "RESEED_MISMATCH_RATIO"]
