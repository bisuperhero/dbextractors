"""``KeySyncStrategy`` — reconcile deletions by comparing primary keys, nothing else.

``hash_diff`` finds deleted rows as a by-product of hashing every row of the
source. When a different strategy already keeps the *content* current
(`incremental` or `id_watermark`) and only the deletions are missing, that is
far too much work: a source whose rows are wide pays for hashing every column of
4.5 M rows to learn which handful of keys disappeared.

This strategy reads **only the primary key** from the source — ``SELECT <pk>
FROM <table> ORDER BY <pk>``, streamed in batches — and pours it into a
temporary PostgreSQL table. Then the same `target_pg.mark_deleted_in_source` that
`hash_diff` finishes with does the rest in two ``UPDATE`` statements on the keys
alone, so no target row travels anywhere:

- a target row whose key is **not** in the source gets ``_deleted_in_source =
  TRUE`` (and a fresh ``_timestamp``);
- a row already marked that **is** in the source again is flipped back to
  ``FALSE``;
- every other row is left untouched.

**It never fetches a row.** A key the source has and the target lacks is only
*counted* (``new_in_source`` in the phase metrics): bringing it over is the job
of whatever strategy loads the content, and doing it here would turn a cheap
reconciliation into a second loader.

Not supported with a surrogate key. The surrogate is an expression over several
columns that the loader joins into each batch; reading only "the key" would mean
evaluating that expression for every row, which is the cost this strategy exists
to avoid — and a wrong expression would mark live rows as deleted.

## Safety

Marking is destructive-looking (a mistake hides live rows from every model that
filters on the flag), so the failure modes lean towards refusing:

- a scan that fails raises. It does **not** fall back to anything: marking from
  a partial key set would flag every key that had not been read yet;
- a source that returns **no keys at all** while the target holds rows raises
  unless ``empty_rows_ok`` is set, because an empty read is far more often a
  wrong ``where_clause`` or a permissions problem than a table that was emptied;
- nothing is committed until the whole scan has finished.
"""

from __future__ import annotations

import logging

from dbextractors.core import reading, status, target_pg
from dbextractors.core.strategies.base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
)
from dbextractors.core.strategies.full import _hash_column, _is_truthy, _resolved_pk
from dbextractors.core.strategies.hash_diff import _base_batch_size, _source_name

_log = logging.getLogger(__name__)


class KeySyncStrategy(LoadStrategy):
    """Compare the source's primary keys with the target's and flag what vanished."""

    name = "key_sync"
    required_features: frozenset = frozenset()

    def validate(self, ctx: LoadContext) -> None:
        super().validate(ctx)

        if ctx.surrogate is not None and ctx.surrogate.enabled:
            raise StrategyError(
                "key_sync: a surrogate key is not supported. The surrogate is an expression "
                "over several columns, so the keys cannot be read on their own. Use "
                "load_method 'hash' for this table."
            )
        pk = _resolved_pk(ctx)
        if not pk:
            raise StrategyError(
                "key_sync: primary_column (or primary_key_column) is missing from "
                "LOAD_SETTINGS. Without a key there is nothing to compare."
            )
        if pk not in ctx.target_names:
            raise StrategyError(f"key_sync: primary key {pk!r} is not among the table's columns.")

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)

        pk = _resolved_pk(ctx)
        assert pk is not None  # guaranteed by validate()
        conn = ctx.target_conn

        # No fallback to a full load here, unlike `incremental`: key_sync adds
        # nothing to the target by design, and a full load of a multi-million row
        # table is a surprising thing to get from a deletion check. The reason is
        # reported and the next load run creates the table.
        if not target_pg.table_exists(conn, ctx.target):
            return self._skipped("the target table does not exist")
        columns = _target_columns(conn, ctx.target)
        if pk not in columns:
            return self._skipped(f"the target has no primary key column {pk!r}")

        target_rows = _count_rows(conn, ctx.target)
        if target_rows == 0:
            return self._skipped("the target table is empty")

        # Every strategy leaves the target with the managed columns, so one that
        # starts from an older extractor's table is brought up to the same shape
        # here too; `ADD COLUMN` without a default is a catalogue change only.
        hash_column = _hash_column(ctx)
        target_pg.ensure_generated_columns(
            conn,
            ctx.target,
            include_row_hash=hash_column is not None,
            hash_column=hash_column or "row_hash",
        )

        _log.info(
            "🚀 Key sync: %s -> %s, the target has %s rows.",
            ctx.source.name,
            ctx.target.qualified(),
            f"{target_rows:,}",
        )

        try:
            with target_pg.SourceKeySnapshot(conn, pk) as snapshot:
                source_keys = self._scan(ctx, snapshot, pk)
                if source_keys == 0 and not _is_truthy(ctx.table_cfg.get("empty_rows_ok")):
                    raise StrategyError(
                        "key_sync: the source returned no keys while the target holds "
                        f"{target_rows:,} rows. Marking would flag every one of them as deleted; "
                        "check where_clause and the source's permissions, or set "
                        "empty_rows_ok if the source really is empty."
                    )
                snapshot.index()
                new_in_source = snapshot.count_new(ctx.target)
                deleted, revived = target_pg.mark_deleted_counts(
                    conn, ctx.target, pk, snapshot.table, source_label=None, pk_as_text=True
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        _log.info(
            "ℹ️ Key sync: %s source keys, %s of them not in the target yet.",
            f"{source_keys:,}",
            f"{new_in_source:,}",
        )
        return LoadResult(
            rows_read=source_keys,
            # No row is written, only flags flip; `deleted_marked` / `revived`
            # carry that. `data_present` still reports the change, because a
            # downstream model filtering on the flag has new input.
            rows_written=0,
            load_method=self.name,
            is_incremental=True,
            data_present=(deleted + revived) > 0,
            phase_metrics={
                "source_keys": source_keys,
                "target_rows": target_rows,
                "new_in_source": new_in_source,
                "deleted_marked": deleted,
                "revived": revived,
            },
        )

    def _skipped(self, reason: str) -> LoadResult:
        _log.warning("⚠️ Key sync did nothing: %s", reason)
        return LoadResult(
            0, 0, self.name, is_incremental=True, data_present=False, fallback_reason=reason
        )

    def _scan(self, ctx: LoadContext, snapshot: target_pg.SourceKeySnapshot, pk: str) -> int:
        """Stream ``SELECT <pk>`` from the source into the snapshot; return the key count."""
        pk_source = _source_name(ctx, pk)

        # Ordered by the key so a dropped connection resumes from the last key
        # that got through (`core.reading`) instead of starting a multi-million
        # row stream again.
        def _build_sql(after):
            condition = ctx.where
            if after is not None:
                key_expr = ctx.dialect.render_key_expr(pk_source)
                bound = f"{key_expr} > {ctx.dialect.sql_literal(after)}"
                condition = f"({bound}) AND ({condition})" if condition else bound
            return ctx.dialect.render_select(
                [pk_source], ctx.source, where=condition, order_by=[pk_source]
            )

        scanned = 0
        progress = status.BatchProgress(_log, phase="keys")
        for batch in reading.read_with_resume(
            ctx.dialect,
            ctx.engine,
            _base_batch_size(ctx),
            build_sql=_build_sql,
            pk_in_batch=pk_source,
            **reading.retry_kwargs_from_settings(ctx.settings),
        ):
            batch = batch.rename(columns=ctx.name_map)
            scanned += snapshot.add(batch[[pk]])
            progress.tick(scanned)
        _log.info("🔢 Source scan: %s keys into the snapshot.", f"{scanned:,}")
        return scanned


def _target_columns(conn, target) -> set:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (target.schema, target.table),
        )
        return {r[0] for r in cur.fetchall()}


def _count_rows(conn, target) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {target_pg.qualify(target)}")
        return int(cur.fetchone()[0])


__all__ = ["KeySyncStrategy"]
