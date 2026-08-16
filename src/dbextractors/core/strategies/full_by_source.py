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

from typing import Any, List, Optional, Tuple

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

        ctx.log("warning", "🚀 Full load by source, source=%s", source_label or "-")

        fingerprint, parts = self._fingerprint(ctx)
        if self._can_skip(ctx, fingerprint):
            ctx.log(
                "warning",
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
        ctx.log("warning", "🔢 Rows in the source: %s", f"{estimate.rows:,}")

        target_exists = target_pg.table_exists(conn, ctx.target)
        partitioned = self._resolve_partitioning(ctx, target_exists)

        if estimate.rows == 0:
            return self._empty_source(ctx, target_exists, partitioned, fingerprint, parts)

        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, estimate)
        return self._replace_slice(ctx, batch_size, partitioned, fingerprint, parts)

    # -- steps -------------------------------------------------------------

    def _fingerprint(self, ctx: LoadContext) -> Tuple[Optional[str], dict]:
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

        def find_column(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            actual = by_lower_name.get(str(name).lower())
            if actual is None:
                ctx.log(
                    "warning",
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
            frame = next(iter(ctx.dialect.iter_batches(ctx.engine, sql, 1)))
        except Exception as err:
            ctx.log("warning", "⚠️ The fingerprint query failed (%s) — fetching normally.", err)
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

    def _can_skip(self, ctx: LoadContext, fingerprint: Optional[str]) -> bool:
        """May this source be skipped?

        Three conditions, and all three must hold: a fingerprint exists, it
        matches the stored one, and **the target still holds that slice**.
        """
        if not fingerprint or not ctx.source_label:
            return False
        if _is_truthy(ctx.settings.get("force_reload")):
            ctx.log("warning", "❗ force_reload — the fingerprint is ignored.")
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
            ctx.log("warning", "⚠️ The stored fingerprint could not be read (%s).", err)
            return False

        if stored is None or stored != fingerprint:
            if stored is not None:
                ctx.log(
                    "warning",
                    "🔄 [%s] the fingerprint changed: %s -> %s",
                    ctx.source_label,
                    stored,
                    fingerprint,
                )
            return False

        if not target_pg.target_holds_source(ctx.target_conn, ctx.target, ctx.source_label):
            ctx.log(
                "warning",
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
            ctx.log(
                "warning",
                "⚠️ partition_by_source is on, but %s is a plain table — deleting with DELETE. "
                "To convert: drop the table and run once with force_reload.",
                ctx.target.qualified(),
            )
        return already_partitioned

    def _empty_source(
        self,
        ctx: LoadContext,
        target_exists: bool,
        partitioned: bool,
        fingerprint: Optional[str],
        parts: dict,
    ) -> LoadResult:
        """The source reports zero rows. What happens to the target depends on whether it exists.

        An empty source with an **existing** target is legitimate: the database
        was emptied and its slice should disappear. With a target that does not
        exist, this **raises** unless ``empty_rows_ok`` is set — an empty read
        must never quietly produce an empty target.
        """
        conn = ctx.target_conn
        ctx.log("warning", "🔢 Source [%s] reports 0 rows.", ctx.source_label or ctx.source.name)

        if not target_exists:
            if not _is_truthy(
                ctx.table_cfg.get("empty_rows_ok", ctx.table_cfg.get("EMPTY_ROWS_OK", False))
            ):
                raise StrategyError(
                    f"Source {ctx.source.name} in {ctx.source_label or '-'} is empty and the "
                    f"target {ctx.target.qualified()} does not exist (empty_rows_ok=False)."
                )
            self._create_empty_target(ctx, partitioned)
        elif ctx.source_label:
            if partitioned:
                part = target_pg.ensure_partition(conn, ctx.target, ctx.source_label)
                with conn.cursor() as cur:
                    cur.execute(f"TRUNCATE TABLE {target_pg.qualify(part)}")
                ctx.log("warning", "🗑️ The source went empty — partition emptied.")
            else:
                target_pg.ensure_source_index(conn, ctx.target)
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {target_pg.qualify(ctx.target)} "
                        f"WHERE {target_pg.quote_ident(target_pg.SOURCE_COLUMN)} = %s",
                        (ctx.source_label,),
                    )
                    ctx.log("warning", "🗑️ The source went empty — %s rows deleted.", cur.rowcount)

        self._finish(ctx, fingerprint, parts)
        return LoadResult(
            0, 0, self.name, data_present=False, phase_metrics={"fingerprint": fingerprint}
        )

    def _create_empty_target(self, ctx: LoadContext, partitioned: bool) -> None:
        """An empty target of the right shape — via a staging table when partitioned.

        A partitioned parent cannot be produced by renaming a plain table, so the
        shape is taken through ``LIKE``. The first source **may** be empty and
        the target must still come out partitioned.
        """
        conn = ctx.target_conn
        hash_column, _ = _hash_plan(ctx)
        columns = _all_columns(ctx, hash_column)
        types = column_types(ctx, hash_column)
        types.setdefault(target_pg.SOURCE_COLUMN, target_pg.SOURCE_COLUMN_TYPE)

        staging = target_pg.create_shadow_table(
            conn, ctx.target, columns=columns, overwrite_types=types
        )
        try:
            if partitioned:
                target_pg.create_partitioned_table(conn, ctx.target, staging)
                target_pg.ensure_partition(conn, ctx.target, str(ctx.source_label))
            else:
                target_pg.swap_shadow_table(conn, staging, ctx.target)
        finally:
            target_pg.drop_shadow_table(conn, staging)
        conn.commit()
        ctx.log("warning", "ℹ️ empty_rows_ok — an empty target was created.")

    def _replace_slice(
        self,
        ctx: LoadContext,
        batch_size: int,
        partitioned: bool,
        fingerprint: Optional[str],
        parts: dict,
    ) -> LoadResult:
        """Stream the source into a staging table and swap the target's slice for it.

        A failure mid-stream therefore leaves the target **untouched** — unlike
        `full`, where the whole table is swapped, only a slice is swapped here,
        so the other databases do not lose their rows even on a failure.
        """
        conn = ctx.target_conn
        full = FullLoadStrategy()
        hash_column, compute_hash = _hash_plan(ctx)
        columns = _all_columns(ctx, hash_column)

        types = column_types(ctx, hash_column)
        types.setdefault(target_pg.SOURCE_COLUMN, target_pg.SOURCE_COLUMN_TYPE)
        if target_pg.table_exists(conn, ctx.target):
            target_pg.load_pg_schema(conn, ctx.target, types, _integer_columns(types))
            # **Before** the staging table: it is created with `LIKE <target>`,
            # so it inherits the target's gaps too, and the first batch's `COPY`
            # then fails with `column "row_hash" … does not exist`. When the
            # target does not exist yet, the shape comes from `overwrite_types`
            # and there is nothing to add.
            full._ensure_managed_columns(ctx, hash_column)
            # The source may gain a column. `full_by_source` is still a full load
            # — that source's whole slice is rewritten — so the column is adopted
            # rather than dropped. Same ordering and same reason as the line
            # above.
            full._adopt_new_columns(ctx, columns, types)

        staging = target_pg.create_shadow_table(
            conn, ctx.target, columns=columns, overwrite_types=types
        )
        rows_read = rows_written = 0
        deleted = 0
        # `parts["count"]` is the row count from the fingerprint — when the
        # fingerprint is off it is `None`, and progress is then logged without
        # percentages and ETA rather than lying about them.
        progress = status.BatchProgress(
            ctx.log, total_rows=parts.get("count"), phase=ctx.source_label or self.name
        )
        try:
            for batch in full.read_batches(ctx, hash_column, batch_size, compute_hash=compute_hash):
                rows_read += len(batch)
                export_df = full._prepare(ctx, batch, hash_column, types, compute_hash=compute_hash)
                if ctx.source_label:
                    export_df[target_pg.SOURCE_COLUMN] = ctx.source_label
                if export_df.empty:
                    continue
                target_pg.copy_from_stdin(conn, staging, export_df, columns)
                progress.tick(rows_read)

            if rows_read == 0:
                # The estimate reported rows, but the source returned not one.
                # Swapping the slice for nothing would be silent data loss.
                raise StrategyError(
                    f"Source {ctx.source.name} in {ctx.source_label or '-'} reported rows "
                    "but streamed none — refusing to swap the slice in the target."
                )

            self._prepare_target(ctx, staging, partitioned)
            deleted, rows_written = target_pg.replace_source_slice(
                conn, ctx.target, staging, columns, ctx.source_label
            )
            target_pg.drop_shadow_table(conn, staging)
            conn.commit()
        except Exception:
            conn.rollback()
            try:
                target_pg.drop_shadow_table(conn, staging)
                conn.commit()
            except Exception as cleanup_err:
                ctx.log("error", "⚠️ The staging table could not be cleaned up: %s", cleanup_err)
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
                "partitioned": partitioned,
                "source": ctx.source_label,
            },
        )

    def _prepare_target(self, ctx: LoadContext, staging: TargetRef, partitioned: bool) -> None:
        """Make sure the target exists and has every column the staging table streamed."""
        conn = ctx.target_conn
        if not target_pg.table_exists(conn, ctx.target):
            if partitioned:
                target_pg.create_partitioned_table(conn, ctx.target, staging)
            else:
                # The shape comes from the staging table, not from a guess: the
                # staging table was built from the very data about to be
                # inserted. `replace_source_slice` fills it with the same
                # `INSERT … SELECT` as for an existing target, so that path does
                # not fork.
                with conn.cursor() as cur:
                    cur.execute(
                        f"CREATE TABLE {target_pg.qualify(ctx.target)} "
                        f"(LIKE {target_pg.qualify(staging)} INCLUDING DEFAULTS)"
                    )
            ctx.log(
                "warning",
                "📝 Target %s did not exist — created from the staging table.",
                ctx.target.qualified(),
            )

        if ctx.source_label:
            target_pg.ensure_source_index(conn, ctx.target)

    def _finish(self, ctx: LoadContext, fingerprint: Optional[str], parts: dict) -> None:
        """The fingerprint and the unique index. Both only **after** a successful transfer."""
        conn = ctx.target_conn
        if fingerprint and ctx.source_label:
            target_pg.write_fingerprint(conn, ctx.target, ctx.source_label, fingerprint, parts)
            conn.commit()

        if not ctx.is_last_source:
            return
        pk = _resolved_pk(ctx)
        if not pk or pk not in ctx.target_names:
            return
        # The index spans the pair (pk, _source): the same key recurs across
        # different databases, so on its own it is not unique.
        index_columns = [pk, target_pg.SOURCE_COLUMN] if ctx.source_label else [pk]
        target_pg.create_unique_pk_index(
            conn, ctx.target, pk, ctx.target_names, fatal=False, columns=index_columns
        )


# --- Helper functions -------------------------------------------------------


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


def _all_columns(ctx: LoadContext, hash_column: Optional[str]) -> List[str]:
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
