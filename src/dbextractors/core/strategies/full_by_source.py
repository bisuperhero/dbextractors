"""``FullBySourceStrategy`` — one target table assembled from several sources.

The predecessor's ``load_full_by_source`` and its seven helpers (8 functions,
335 lines) exist **only in the MSSQL extractor** — several source databases of
the same structure pour rows into a single target table, partitioned by source
in PostgreSQL so that one source can be rewritten or reloaded without touching
the others.

**All seven helpers are closures** over the local state of the call — which is
why they have parameterless signatures there. Here six of them are methods of
`core.target_pg` (it is pure PostgreSQL DDL: partitions, an index over
``_source``, swapping a slice, the fingerprint ledger) and the strategy says
**what**, not how.

Requires ``FEATURE_PARTITION_BY_SOURCE`` — dialects that cannot do multi-source
today (MySQL, PostgreSQL, Firebird) do not offer the strategy.

## Rows carry `_source` regardless of how many databases the run selected

It looks like a detail, but it is a trap the predecessor has a comment about:
`full_by_source` deletes **by `_source`**, so if the column were added only when
the run happens to be processing several databases, a run over a single database
would wipe the whole table. The configuration (``multi_source``) therefore
decides, not the length of the list — and where the configuration is silent, the
length of the list decides **once**, in the entrypoint, not inside the strategy.

## What the fingerprint does

The fingerprint is one aggregate query against the source (``COUNT`` catches
deletes, ``MAX(pk)`` catches inserts, a change timestamp or a column sum catches
edits). When it has not changed since last time **and the target still holds
that slice**, the database is skipped without fetching a single row.

That second condition is not extra caution: the first database of a rebuilt
table recreates it, so from the second one onwards the table is back and the
fingerprint alone would skip every remaining database. In the predecessor this
left 3 rows of 220,972 in the target.

It is written **only after** a successful transfer — otherwise an interrupted
run would leave a fingerprint in the ledger for a source whose data is not in the
target.

The general rules hold here too: writes only through ``COPY ... FROM STDIN``,
``_deleted_in_source``/``_timestamp``/``row_hash`` on every strategy, and a
threefold memory peak because the orchestrator runs 3 pipelines concurrently.

## ``row_hash`` is **not computed** here

The only strategy where the column exists in the target but stays NULL — and
that is deliberate, not unfinished. The predecessor omits it on this one path
and says why:

    "Reads the source table in ONE streamed pass — **no row hashing**, no OFFSET
    paging, no 100k-element IN lists — which is the cheapest shape a
    memory-constrained SQL Server can serve"

What it would cost is not theoretical: on the largest of these tables it means
``HASHBYTES`` over 156 columns and 1.4 M rows every night, on a server the
predecessor itself describes as memory-starved. And it has been verified that
nobody misses it — not one dbt model touches `row_hash` on those tables, it is
merely declared in `sources.yml`.

It can be switched on with ``compute_row_hash: true`` in ``LOAD_SETTINGS`` — an
existing key of the frozen contract, just with the opposite default from the
other strategies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd

from dbextractors.core import status, target_pg
from dbextractors.core.strategies.base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
    TargetRef,
    resolve_batch_size,
)
from dbextractors.core.strategies.full import (
    FullLoadStrategy,
    _integer_columns,
    _is_truthy,
    _resolved_pk,
    column_types,
)
from dbextractors.dialects.base import FEATURE_PARTITION_BY_SOURCE

_log = logging.getLogger(__name__)


class FullBySourceStrategy(LoadStrategy):
    """Full load of one source into its partition of the target table."""

    name = "full_by_source"
    required_features: frozenset = frozenset({FEATURE_PARTITION_BY_SOURCE})

    def validate(self, ctx: LoadContext) -> None:
        """Check whatever can be checked before the source is first touched.

        The predecessor's ``load_full_by_source`` finds this out at run time;
        partitioning by ``_source`` with no source label is nonsense that ought
        to be recognised straight away.
        """
        super().validate(ctx)

        if _is_truthy(ctx.settings.get("partition_by_source")) and not ctx.source_label:
            raise StrategyError(
                "full_by_source: partition_by_source is on, but the run has no "
                "source_label — there is nothing to partition by."
            )

        # This strategy partitions the target by source and can do nothing else.
        # A general `partition_by` over another column would quietly do nothing,
        # so it raises.
        spec = ctx.partition_spec()
        if spec is not None and spec.column != target_pg.SOURCE_COLUMN:
            raise StrategyError(
                f"full_by_source partitions the target by {target_pg.SOURCE_COLUMN!r}, but "
                f"partition_by says {spec.column!r}. This strategy knows no other layout."
            )

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)
        conn = ctx.target_conn
        source_label = ctx.source_label

        _log.info("🚀 Full load by source, source=%s", source_label or "-")

        fingerprint, parts = self._fingerprint(ctx)
        skip = self._can_skip(ctx, fingerprint)
        # The read phase ends here. Reading the ledger and asking whether the
        # target still holds the slice opened a transaction with an
        # `AccessShareLock` on the target; left open, it would last through the
        # whole stream, and another source's `ALTER TABLE … ADD COLUMN` — and
        # everything queued behind it — would wait for this one's download.
        conn.commit()
        if skip:
            _log.info(
                "🔕 [%s] unchanged (fingerprint %s) — skipping the fetch.",
                source_label,
                fingerprint,
            )
            return LoadResult(
                0,
                0,
                "unchanged",
                data_present=True,
                phase_metrics={"fingerprint": fingerprint, "skipped": True},
            )

        estimate = ctx.dialect.estimate_size(
            ctx.engine, ctx.source, ctx.where, known_total_rows=parts.get("count")
        )
        _log.info("🔢 Rows in the source: %s", f"{estimate.rows:,}")

        if estimate.rows == 0:
            return self._empty_source(ctx, fingerprint, parts)

        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, estimate)
        return self._replace_slice(ctx, batch_size, fingerprint, parts)

    # -- steps -------------------------------------------------------------

    def _fingerprint(self, ctx: LoadContext) -> Tuple[str | None, dict]:
        """A cheap fingerprint of the source, or ``(None, {})`` when off or failed.

        **A failed fingerprint never brings down the extraction.** It is an
        optimisation: when it does not work out, the data is loaded normally,
        only more expensively. It is the one place in the package where an
        exception may be swallowed — and even then it is logged.
        """
        cfg = dict(ctx.fingerprint_cfg or {})
        mode = str(cfg.get("mode") or "off").lower()
        if mode in ("off", "none", ""):
            return None, {}

        by_lower_name = {c.lower(): c for c in ctx.source_names}

        def find_column(name: str | None) -> str | None:
            if not name:
                return None
            actual = by_lower_name.get(str(name).lower())
            if actual is None:
                _log.warning(
                    "⚠️ Column %r for the fingerprint is not in the source — ignoring it.",
                    name,
                )
            return actual

        pk = find_column(cfg.get("primary_column") or ctx.settings.get("primary_column"))
        timestamp_col = find_column(cfg.get("timestamp_column")) if mode == "datsave" else None
        aggregate_cols = (
            [c for c in (find_column(c) for c in (cfg.get("aggregate_columns") or [])) if c]
            if mode == "aggregate"
            else []
        )

        sql = ctx.dialect.render_fingerprint(
            ctx.source,
            pk=pk,
            timestamp_column=timestamp_col,
            aggregate_columns=aggregate_cols,
            where=ctx.where,
        )
        try:
            # **Read to the end**, not `next()`: the aggregate is one row, but on
            # MySQL the stream is an unbuffered cursor, and a generator left
            # suspended after its first frame counts as abandoned — `_release`
            # then cancels the query with `KILL QUERY` from a second connection
            # and discards the first one. With one run per source and table that
            # is a thrown-away connection per run. Draining one row costs
            # nothing and closes the cursor normally.
            frames = list(ctx.dialect.iter_batches(ctx.engine, sql, 1))
            frame = frames[0]
        except Exception as err:
            _log.warning("⚠️ The fingerprint query failed (%s) — fetching normally.", err)
            return None, {}
        if frame.empty:
            return None, {}

        first_row = frame.iloc[0]
        parts = {
            "count": int(first_row["fp_count"]) if first_row["fp_count"] is not None else 0,
            "max_id": first_row["fp_max_id"],
            "max_ts": first_row["fp_max_ts"],
            "agg": first_row["fp_agg"],
        }
        return _fingerprint_key(parts), parts

    def _can_skip(self, ctx: LoadContext, fingerprint: str | None) -> bool:
        """May this source be skipped?

        Three conditions, and all three must hold: a fingerprint exists, it
        matches the stored one, and **the target still holds that slice**.
        """
        if not fingerprint or not ctx.source_label:
            return False
        if _is_truthy(ctx.settings.get("force_reload")):
            _log.warning("❗ force_reload — the fingerprint is ignored.")
            return False

        try:
            stored = target_pg.read_fingerprint(ctx.target_conn, ctx.target, ctx.source_label)
        except Exception as err:
            # The `rollback()` matters here: the target connection has autocommit
            # off, so a failed query leaves the transaction aborted and the next
            # statement on the same connection (`table_exists` in `run()`) fails
            # with `InFailedSqlTransaction` — an error unrelated to the original
            # one.
            ctx.target_conn.rollback()
            _log.warning("⚠️ The stored fingerprint could not be read (%s).", err)
            return False

        if stored is None or stored != fingerprint:
            if stored is not None:
                _log.info(
                    "🔄 [%s] the fingerprint changed: %s -> %s",
                    ctx.source_label,
                    stored,
                    fingerprint,
                )
            return False

        if not target_pg.target_holds_source(ctx.target_conn, ctx.target, ctx.source_label):
            _log.warning(
                "🔄 [%s] the fingerprint matches, but the slice is missing from the target — "
                "fetching again.",
                ctx.source_label,
            )
            return False
        return True

    def _resolve_partitioning(self, ctx: LoadContext, target_exists: bool) -> bool:
        """For an existing target reality decides, for a new one the configuration does.

        An existing plain table is **never quietly rebuilt** as a partitioned
        one: that would mean rebuilding a target that views hang off.
        """
        wants_partitioning = _is_truthy(ctx.settings.get("partition_by_source")) and bool(
            ctx.source_label
        )
        if not target_exists:
            return wants_partitioning

        already_partitioned = target_pg.is_partitioned(ctx.target_conn, ctx.target)
        if wants_partitioning and not already_partitioned:
            _log.warning(
                "⚠️ partition_by_source is on, but %s is a plain table — deleting with DELETE. "
                "To convert: drop the table and run once with force_reload.",
                ctx.target.qualified(),
            )
        return already_partitioned

    def _empty_source(self, ctx: LoadContext, fingerprint: str | None, parts: dict) -> LoadResult:
        """The source reports zero rows. What happens to the target depends on whether it exists.

        An empty source with an **existing** target is legitimate: the database
        was emptied and its slice should disappear. With a target that does not
        exist, this **raises** unless ``empty_rows_ok`` is set — an empty read
        must never quietly produce an empty target. The first source **may** be
        empty and the target must still come out in the right shape
        (partitioned, when asked for).
        """
        conn = ctx.target_conn
        _log.info("🔢 Source [%s] reports 0 rows.", ctx.source_label or ctx.source.name)

        target_exists = target_pg.table_exists(conn, ctx.target)
        conn.commit()
        if not target_exists and not _is_truthy(
            ctx.table_cfg.get("empty_rows_ok", ctx.table_cfg.get("EMPTY_ROWS_OK", False))
        ):
            raise StrategyError(
                f"Source {ctx.source.name} in {ctx.source_label or '-'} is empty and the "
                f"target {ctx.target.qualified()} does not exist (empty_rows_ok=False)."
            )

        partitioned = False
        if not target_exists or ctx.source_label:
            plan = self._prepare_ddl(ctx)
            partitioned = plan.partitioned
            try:
                if ctx.source_label:
                    if partitioned:
                        part = target_pg.partition_ref_for(ctx.target, ctx.source_label)
                        with conn.cursor() as cur:
                            cur.execute(f"TRUNCATE TABLE {target_pg.qualify(part)}")
                        _log.info("🗑️ The source is empty — partition emptied.")
                    else:
                        with conn.cursor() as cur:
                            cur.execute(
                                f"DELETE FROM {target_pg.qualify(ctx.target)} "
                                f"WHERE {target_pg.quote_ident(target_pg.SOURCE_COLUMN)} = %s",
                                (ctx.source_label,),
                            )
                            _log.info("🗑️ The source is empty — %s rows deleted.", cur.rowcount)
                target_pg.drop_shadow_table(conn, plan.staging)
                conn.commit()
            except Exception:
                self._discard_staging(conn, plan.staging)
                raise
            if plan.created:
                _log.info("ℹ️ empty_rows_ok — an empty target was created.")

        self._finish(ctx, fingerprint, parts)
        return LoadResult(
            0,
            0,
            self.name,
            data_present=False,
            phase_metrics={"fingerprint": fingerprint, "partitioned": partitioned},
        )

    def _prepare_ddl(self, ctx: LoadContext) -> _DdlPlan:
        """The DDL phase: everything the slice swap needs to exist, in one short transaction.

        Runs under `target_pg.ddl_lock`, so concurrent runs over the same target
        (one source each) take turns here instead of failing on one another's
        ``CREATE``. Inside, everything is decided **again** from the catalogue —
        a run that waited for the lock finds whatever the previous holder
        created:

        1. the schema;
        2. on an existing target: the types it really has, the managed columns
           and columns the source has newly started sending;
        3. the staging table — ``LIKE`` the target when it exists, so it has
           every column just added, otherwise from the source's types;
        4. a missing target, shaped like the staging table: partitioned by
           ``_source`` when asked for, plain otherwise;
        5. this source's partition (or, on a plain multi-source target, the
           index over ``_source``);
        6. the fingerprint ledger, when a fingerprint is configured.

        The commit at the end publishes all of it, the **empty** staging table
        included. That is what lets the stream run without any lock on the
        target: before, the staging table was created ``LIKE`` the target inside
        the data transaction, which kept an ``AccessShareLock`` on the target for
        the whole download. The cost: a process killed mid-stream leaves its
        ``dbx_shadow_*`` behind instead of having it vanish with a rollback
        (`target_pg.find_stale_shadows` lists them). A failure the process
        survives drops it as before.
        """
        conn = ctx.target_conn
        full = FullLoadStrategy()
        hash_column, _ = _hash_plan(ctx)
        with target_pg.ddl_lock(conn, ctx.target):
            target_pg.ensure_schema(conn, ctx.target.schema)
            types = column_types(ctx, hash_column)
            types.setdefault(target_pg.SOURCE_COLUMN, target_pg.SOURCE_COLUMN_TYPE)
            # Under the lock: the column order is read off the target as it is
            # now, not as it was before another run created or widened it.
            columns = _all_columns(ctx, hash_column)
            target_types: dict = {}
            exists = target_pg.table_exists(conn, ctx.target)
            if exists:
                schema = target_pg.load_pg_schema(conn, ctx.target, types, _integer_columns(types))
                target_types = dict(schema.types)
                # **Before** the staging table: it is created `LIKE <target>`,
                # so it inherits the target's gaps too, and the first batch's
                # `COPY` then fails with `column "row_hash" … does not exist`.
                # Not `full._ensure_managed_columns`: that one commits, which
                # would end the lock half-way through.
                target_pg.ensure_generated_columns(
                    conn, ctx.target, include_row_hash=True, hash_column=hash_column
                )
                # The source may gain a column. `full_by_source` is still a full
                # load — that source's whole slice is rewritten — so the column
                # is adopted rather than dropped; in the other slices it is NULL.
                full._adopt_new_columns(ctx, columns, types)
            partitioned = self._resolve_partitioning(ctx, exists)

            staging = target_pg.create_shadow_table(
                conn, ctx.target, columns=columns, overwrite_types=types
            )
            if not exists:
                self._create_target(ctx, staging, partitioned)
            if ctx.source_label:
                if partitioned:
                    target_pg.ensure_partition(conn, ctx.target, ctx.source_label)
                else:
                    target_pg.ensure_source_index(conn, ctx.target)
                if _fingerprint_on(ctx):
                    target_pg.ensure_fingerprint_store(conn, ctx.target.schema)
        return _DdlPlan(
            staging=staging,
            columns=columns,
            types=types,
            target_types=target_types,
            partitioned=partitioned,
            created=not exists,
        )

    def _create_target(self, ctx: LoadContext, staging: TargetRef, partitioned: bool) -> None:
        """A missing target, shaped like the staging table. Inside the DDL phase.

        The shape comes from the staging table, not from a guess: it was built
        from the very columns about to be streamed. A partitioned parent cannot
        be produced by renaming a plain table, so it is created through ``LIKE``;
        the plain target is too, so the slice swap fills both the same way.
        """
        conn = ctx.target_conn
        if partitioned:
            target_pg.create_partitioned_table(conn, ctx.target, staging)
        else:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE TABLE {target_pg.qualify(ctx.target)} "
                    f"(LIKE {target_pg.qualify(staging)} INCLUDING DEFAULTS)"
                )
        _log.info(
            "📝 Target %s did not exist — created from the staging table.",
            ctx.target.qualified(),
        )

    def _replace_slice(
        self,
        ctx: LoadContext,
        batch_size: int,
        fingerprint: str | None,
        parts: dict,
    ) -> LoadResult:
        """Stream the source into a staging table and swap the target's slice for it.

        A failure mid-stream therefore leaves the target's data **untouched** —
        unlike `full`, where the whole table is swapped, only a slice is swapped
        here, so the other databases do not lose their rows even on a failure.

        Three transactions: the DDL phase (`_prepare_ddl`, under the target's
        lock), the stream into the staging table (no lock on the target), and
        the swap — ``TRUNCATE`` of this source's partition plus ``INSERT …
        SELECT`` — which takes no advisory lock, so sources swap side by side.
        """
        conn = ctx.target_conn
        full = FullLoadStrategy()
        hash_column, compute_hash = _hash_plan(ctx)

        plan = self._prepare_ddl(ctx)
        staging, columns, types = plan.staging, plan.columns, plan.types
        rows_read = rows_written = 0
        deleted = 0
        # `parts["count"]` is the row count from the fingerprint — when the
        # fingerprint is off it is `None`, and progress is then logged without
        # percentages and ETA rather than lying about them.
        progress = status.BatchProgress(
            _log, total_rows=parts.get("count"), phase=ctx.source_label or self.name
        )
        lossy = _columns_of_another_type(ctx, plan)
        try:
            try:
                for batch in full.read_batches(
                    ctx, hash_column, batch_size, compute_hash=compute_hash
                ):
                    rows_read += len(batch)
                    export_df = full._prepare(
                        ctx, batch, hash_column, types, compute_hash=compute_hash
                    )
                    _check_nothing_was_lost(ctx, plan, lossy, batch, export_df)
                    if ctx.source_label:
                        export_df[target_pg.SOURCE_COLUMN] = ctx.source_label
                    if export_df.empty:
                        continue
                    target_pg.copy_from_stdin(conn, staging, export_df, columns)
                    progress.tick(rows_read)
            except Exception as err:
                explained = _explain_type_mismatch(ctx, err, plan)
                if explained is None:
                    raise
                raise explained from err

            if rows_read == 0:
                # The estimate reported rows, but the source returned not one.
                # Swapping the slice for nothing would be silent data loss.
                raise StrategyError(
                    f"Source {ctx.source.name} in {ctx.source_label or '-'} reported rows "
                    "but streamed none — refusing to swap the slice in the target."
                )

            deleted, rows_written = target_pg.replace_source_slice(
                conn, ctx.target, staging, columns, ctx.source_label
            )
            target_pg.drop_shadow_table(conn, staging)
            conn.commit()
        except Exception:
            self._discard_staging(conn, staging)
            raise

        self._finish(ctx, fingerprint, parts)
        return LoadResult(
            rows_read=rows_read,
            rows_written=rows_written,
            load_method=self.name,
            data_present=rows_written > 0,
            phase_metrics={
                "fingerprint": fingerprint,
                "deleted": deleted,
                "partitioned": plan.partitioned,
                "source": ctx.source_label,
            },
        )

    @staticmethod
    def _discard_staging(conn: Any, staging: TargetRef) -> None:
        """Roll back and drop the staging table. Never replaces the error that got here.

        The staging table was committed in the DDL phase, so the rollback alone
        no longer removes it.
        """
        conn.rollback()
        try:
            target_pg.drop_shadow_table(conn, staging)
            conn.commit()
        except Exception as cleanup_err:
            conn.rollback()
            _log.warning("⚠️ The staging table could not be cleaned up: %s", cleanup_err)

    def _finish(self, ctx: LoadContext, fingerprint: str | None, parts: dict) -> None:
        """The fingerprint and the unique index. Both only **after** a successful transfer.

        Neither may turn a source whose data is already in the target into a
        failed one: the fingerprint is an optimisation (failing to store it
        means the next run fetches again), and the index is created with
        ``fatal=False``.
        """
        conn = ctx.target_conn
        if fingerprint and ctx.source_label:
            try:
                target_pg.write_fingerprint(conn, ctx.target, ctx.source_label, fingerprint, parts)
                conn.commit()
            except Exception as err:
                conn.rollback()
                _log.warning(
                    "⚠️ [%s] the fingerprint could not be stored (%s) — the data is in "
                    "the target; the next run will fetch this source again.",
                    ctx.source_label,
                    err,
                )

        if not ctx.is_last_source:
            return
        pk = _resolved_pk(ctx)
        if not pk or pk not in ctx.target_names:
            return
        # The index spans the pair (pk, _source): the same key recurs across
        # different databases, so on its own it is not unique.
        index_columns = [pk, target_pg.SOURCE_COLUMN] if ctx.source_label else [pk]
        if target_pg.is_partitioned(conn, ctx.target):
            # Under the DDL lock: concurrent sources must not race to create the
            # same index. `create_unique_pk_index` asks the catalogue first (by
            # columns), so once the index exists this costs a lookup and no
            # `ShareLock`; it commits, which also releases the lock.
            conn.commit()
            try:
                with target_pg.ddl_lock(conn, ctx.target):
                    target_pg.create_unique_pk_index(
                        conn, ctx.target, pk, ctx.target_names, fatal=False, columns=index_columns
                    )
            except Exception as err:  # the data is committed; the index must not fail it
                _log.error(
                    "🛑 The unique index over %s could not be created: %s", index_columns, err
                )
            return
        target_pg.create_unique_pk_index(
            conn, ctx.target, pk, ctx.target_names, fatal=False, columns=index_columns
        )


@dataclass(frozen=True)
class _DdlPlan:
    """What the DDL phase created and decided, for the stream and the swap."""

    staging: TargetRef
    columns: List[str]
    types: dict
    #: The target's own column types as they were before this run — empty for a
    #: target this run created. For explaining a type mismatch.
    target_types: dict
    partitioned: bool
    created: bool


# --- Helper functions -------------------------------------------------------


def _fingerprint_on(ctx: LoadContext) -> bool:
    """Is a fingerprint configured? The same test `_fingerprint` applies."""
    mode = str((ctx.fingerprint_cfg or {}).get("mode") or "off").lower()
    return mode not in ("off", "none", "")


#: SQLSTATE class 22, "data exception": a value the target column type cannot
#: hold (invalid text for a number, out of range, a bad date, …).
_DATA_EXCEPTION_CLASS = "22"

#: The column a ``COPY`` failed on, from the error context PostgreSQL attaches:
#: ``COPY dbx_shadow_…, line 1, column label: "abc"``.
_COPY_COLUMN_RE = re.compile(r"COPY [^,]+, line \d+, column ([^:]+):")


def _explain_type_mismatch(ctx: LoadContext, err: Exception, plan: _DdlPlan) -> Exception | None:
    """A readable error for a source column whose values the target's type rejects.

    The schema of the sources may differ for a while (a rollout reaches one
    tenant before another). A column **missing** in this source is NULL in its
    slice, and a column **only** this source has is adopted; but a column the
    target already has with a type this source's values do not fit fails the
    ``COPY`` into the staging table — before the swap, so the target and the
    other slices are unchanged. PostgreSQL's message names the column at best
    and neither type nor source; this names all of them.

    Returns ``None`` when ``err`` is something else — the caller re-raises it
    unchanged. Only explains, never decides: the run fails either way.
    """
    if not str(getattr(err, "pgcode", None) or "").startswith(_DATA_EXCEPTION_CLASS):
        return None
    diag = getattr(err, "diag", None)
    context = str(getattr(diag, "context", None) or "")
    match = _COPY_COLUMN_RE.search(context)
    label = ctx.source_label or ctx.source.name
    if not match:
        return StrategyError(f"[{label}] a source value does not fit the target column: {err}")

    column = match.group(1).strip()
    source_type = next(
        (
            f"{c.source_type} (→ {c.pg_type})"
            for c in ctx.columns
            if (c.target_name or c.name) == column
        ),
        "unknown",
    )
    target_type = plan.target_types.get(column) or plan.types.get(column) or "unknown"
    detail = str(getattr(diag, "message_primary", None) or err).strip()
    return StrategyError(
        f"[{label}] column {column!r}: the source type {source_type} does not fit the "
        f"target column type {target_type} of {ctx.target.qualified()} — {detail}. "
        "Nothing was written: the target and the other sources' slices are unchanged. "
        "Align the column type across the sources, or change the target column by hand."
    )


#: Type families for telling "the same kind of column" from "another kind".
#: Checked in order, by substring of the lowercased PostgreSQL type name.
_TYPE_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("timestamp", ("timestamp",)),
    ("date", ("date",)),
    ("time", ("time", "interval")),
    ("boolean", ("bool",)),
    ("integer", ("smallint", "integer", "bigint", "int2", "int4", "int8", "serial")),
    ("number", ("numeric", "decimal", "real", "double", "float", "money")),
    ("binary", ("bytea",)),
    ("json", ("json",)),
    ("uuid", ("uuid",)),
    ("text", ("char", "text", "citext", "name")),
)


def _type_family(pg_type: str | None) -> str:
    text = str(pg_type or "").lower()
    for family, needles in _TYPE_FAMILIES:
        if any(needle in text for needle in needles):
            return family
    return text or "unknown"


def _columns_of_another_type(ctx: LoadContext, plan: _DdlPlan) -> List[Tuple[str, str, str, str]]:
    """Columns the target already has with another kind of type than the source's.

    ``[(batch column, target column, source type, target type)]``. Only these can
    lose values on the way: the conversion layer converts a batch to the
    **target's** types (`target_pg.load_pg_schema`), and a value that does not
    convert — text into an integer column — becomes NULL there without a word.
    Columns of the same family (``integer``/``bigint``, ``varchar``/``text``,
    ``timestamp``/``timestamptz``) are left alone.
    """
    if not plan.target_types:
        return []
    out = []
    for column in ctx.columns:
        target_name = column.target_name or column.name
        target_type = plan.target_types.get(target_name)
        if target_type is None:
            continue
        if _type_family(column.pg_type) != _type_family(target_type):
            # The batch is compared **before** `_prepare` renames it, so it still
            # carries the source name.
            batch_name = column.name
            source_type = f"{column.source_type} (→ {column.pg_type})"
            out.append((batch_name, target_name, source_type, target_type))
    return out


def _present(values: pd.Series) -> Any:
    """Which values carry something: not NULL and not an empty string.

    An empty string becomes NULL in the target whatever the column type (see
    `target_pg.EMPTY_STRING_AS_NULL`), so its turning into NULL is not a loss.
    """
    blank = values.map(lambda v: isinstance(v, str) and not v.strip())
    return (values.notna() & ~blank.astype(bool)).to_numpy()


def _check_nothing_was_lost(
    ctx: LoadContext,
    plan: _DdlPlan,
    lossy: List[Tuple[str, str, str, str]],
    batch: pd.DataFrame,
    export_df: pd.DataFrame,
) -> None:
    """Fail the source when a value would be written as NULL because of the target's type.

    Sources whose schemas differ for a while (a rollout reaches one tenant before
    another) must not quietly blank a column in their slice: the run fails, before
    the swap, so the target and the other slices stay as they were, and the
    message names the column, both types and the source. Decided by the values,
    not by the types alone — a text column holding only numbers still loads into
    an integer column, as it always has.

    **Only for a run with a configured** ``source_label``. Without one — the
    multi-source loop over ``DATABASES`` from before 2.4.0 — the same finding is a
    warning and the values are written as before: failing those runs now would
    change the behaviour of configurations that exist, which a new version must
    not do.
    """
    if not lossy or len(batch) != len(export_df):
        return
    for batch_name, target_name, source_type, target_type in lossy:
        if batch_name not in batch.columns or target_name not in export_df.columns:
            continue
        lost = int((_present(batch[batch_name]) & ~_present(export_df[target_name])).sum())
        if not lost:
            continue
        message = (
            f"[{ctx.source_label or ctx.source.name}] column {target_name!r}: {lost} value(s) of "
            f"the source type {source_type} do not fit the target column type {target_type} of "
            f"{ctx.target.qualified()} and would be written as NULL."
        )
        if not ctx.settings.get("source_label"):
            _log.warning("⚠️ %s They are written as NULL, as before 2.4.0.", message)
            continue
        raise StrategyError(
            f"{message} Nothing was written: the target and the other sources' slices are "
            "unchanged. Align the column type across the sources, or change the target "
            "column by hand."
        )


def _hash_plan(ctx: LoadContext) -> Tuple[str, bool]:
    """``(name of the hash column in the target, compute it?)``.

    Two independent questions that the other strategies merge into one. Here they
    come apart: the column in the target **always exists** (the predecessor
    creates it, and a large dbt layer reads tables that have it), but the value
    is not computed — see the module docstring.

    ``compute_row_hash`` is an existing key of the frozen contract, just with the
    opposite default here. ``hash_column`` from the configuration determines the
    name, not whether it gets computed; the 16 pipelines on this path have it set
    to ``None``.

    That is why ``compute_row_hash`` is **three-state** in the configuration.
    With an ordinary default of ``True`` it would always be present in
    ``ctx.settings`` and this opposite default would never apply — which is
    exactly what the golden test caught.
    """
    from dbextractors.core.naming import DEFAULT_HASH_COLUMN

    configured = ctx.settings.get("hash_column")
    name = ctx.name_map.get(configured, configured) if configured else DEFAULT_HASH_COLUMN
    flag = ctx.settings.get("compute_row_hash")
    return str(name), flag is not None and _is_truthy(flag)


def _all_columns(ctx: LoadContext, hash_column: str | None) -> List[str]:
    """Target columns. Multi-source adds ``_source`` at the end.

    It is added **only** for multi-source tables — giving it to all of them would
    change the DDL of all ~670 tables, and target column names are the one thing
    that must not change, because a large dbt layer is built on them.
    """
    from dbextractors.core import naming

    metadata: tuple[str, ...] = naming.METADATA_COLUMNS
    if ctx.source_label:
        metadata = (*metadata, target_pg.SOURCE_COLUMN)
    wanted = naming.target_column_names(ctx.source_names, hash_column, metadata=metadata)
    # The existing targets on this path were created by the MSSQL
    # `full_by_source` with the order
    # `_deleted_in_source, _timestamp, _source, row_hash` — the only variant this
    # one predecessor file has. It is taken from the target rather than
    # replicated in code.
    return naming.reconcile_with_existing(
        wanted, target_pg.existing_column_order(ctx.target_conn, ctx.target)
    )


def _fingerprint_key(parts: dict) -> str:
    """The fingerprint parts as one comparable string. Literally as the predecessor does it."""

    def text(value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "isoformat"):
            return str(value.isoformat())
        return str(value)

    return "|".join(
        [
            str(parts.get("count", 0)),
            text(parts.get("max_id")),
            text(parts.get("max_ts")),
            "" if parts.get("agg") is None else f"{float(parts['agg']):.6f}",
        ]
    )


__all__ = ["FullBySourceStrategy"]
