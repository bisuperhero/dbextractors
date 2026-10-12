"""``IdWatermarkStrategy`` — reads only rows whose PK is above the watermark.

The predecessor's ``load_id_watermark`` (1 variant, 105 lines) exists only for a
PostgreSQL source — the one function among the predecessors with a single
variant, so nothing about the logic changes here, only the shape of the
interface. Serves 4 tables in production today: append-only, with an increasing
primary key and no UPDATE/DELETE in the source.

The mechanics are one-directional keyset paging (``WHERE pk > watermark ORDER BY
pk``), which is why ``required_features`` asks for ``FEATURE_KEYSET``.

The watermark is not stored anywhere: it is ``MAX(pk)`` in the **target**. There
is no state file and no state table, so there is nothing to lose and nothing to
drift.

## What to watch out for

By definition this strategy **sees neither changes nor deletions** — it sees
only what was added past the last key. That is fine for an append-only source,
but it means two things:

1. ``_deleted_in_source`` stays permanently ``FALSE`` for rows that vanished
   from the source. The column has to exist all the same (a large dbt layer
   depends on it), but no live-key snapshot is taken here: it would have to read
   the entire source table, and the whole saving would disappear with it.
2. A row inserted into the source **late** — with a key lower than one already
   in the target — is never transferred. With a sequence that cannot happen;
   with manually assigned keys it can.

Both are properties of the strategy, not omissions; anyone who needs to capture
changes should use ``hash`` or ``incremental``.

Writes go through ``COPY ... FROM STDIN`` and nothing else.
"""

from __future__ import annotations

import logging
from typing import Any

from dbextractors.core import status, target_pg
from dbextractors.core.strategies.base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
    fallback_full,
    resolve_batch_size,
    surrogate_provides,
)
from dbextractors.core.strategies.full import (
    FullLoadStrategy,
    _hash_column,
    _integer_columns,
    _pk_in_source,
    _resolved_pk,
    column_types,
)
from dbextractors.core.strategies.incremental import (
    _all_columns,
    _drop_columns_missing_in_target,
    _upsert_from_staging,
    _with_where,
)
from dbextractors.dialects.base import FEATURE_KEYSET

_log = logging.getLogger(__name__)

#: How many rows are staged before they are projected into the target and
#: committed, when ``LOAD_SETTINGS.checkpoint_rows`` is not set.
#:
#: Without a checkpoint the whole delta is one transaction, so a run that fails
#: three quarters of the way through rolls everything back and the retry reads
#: the same delta from the same ``MAX(pk)`` again — after an OOM on a 42.7 M-row
#: delta, every retry started from zero. Five million rows is minutes of reading
#: and one upsert of tens of seconds, so the extra commits cost next to nothing
#: against what a failure no longer throws away.
DEFAULT_CHECKPOINT_ROWS = 5_000_000


class IdWatermarkStrategy(LoadStrategy):
    """Transfer only rows whose PK is above the last watermark."""

    name = "id_watermark"
    required_features: frozenset = frozenset({FEATURE_KEYSET})

    def validate(self, ctx: LoadContext) -> None:
        """Check that an increasing primary key is configured for the watermark.

        The predecessor's ``load_id_watermark`` has no validation phase of its
        own and responds to a missing key with a silent full load. As in
        `hash_diff`, this raises instead — a full load is the most expensive
        possible answer to a typo in YAML.
        """
        super().validate(ctx)

        pk = _resolved_pk(ctx)
        if not pk:
            raise StrategyError(
                "id_watermark: primary_column (or primary_key_column) is missing from "
                "LOAD_SETTINGS. Without an increasing key there is no watermark to compute."
            )
        if pk not in ctx.target_names and not surrogate_provides(ctx, pk):
            raise StrategyError(
                f"id_watermark: primary key {pk!r} is not among the table's columns."
            )
        _checkpoint_rows(ctx)

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)

        pk = _resolved_pk(ctx)
        assert pk is not None  # guaranteed by validate()
        conn = ctx.target_conn

        if not target_pg.table_exists(conn, ctx.target):
            return fallback_full(ctx, "the target table does not exist")

        watermark = _watermark(conn, ctx.target, pk)
        if watermark is None:
            # `MAX(pk) IS NULL` means an empty target (or a target whose key is
            # all NULL — which amounts to the same: nothing to start from).
            return fallback_full(ctx, "the target table is empty, there is no watermark to compute")

        _log.info("🚀 ID watermark: taking rows with %s > %r", pk, watermark)
        target_pg.assert_unique_pk_index(conn, ctx.target, pk)

        pk_source = _source_name(ctx, pk)
        where = _watermark_where(ctx, pk_source, watermark)
        estimate = ctx.dialect.estimate_size(ctx.engine, ctx.source, where)
        if estimate.rows == 0:
            _log.info("ℹ️ No new rows above the watermark.")
            return LoadResult(0, 0, self.name, is_incremental=True, data_present=False)

        _log.info("🔢 New rows: %s", f"{estimate.rows:,}")
        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, estimate)
        result = self._load_new_rows(ctx, where, batch_size, pk, total_rows=estimate.rows)
        result.phase_metrics.update({"watermark": str(watermark), "estimated_rows": estimate.rows})
        return result

    def _load_new_rows(
        self,
        ctx: LoadContext,
        where: str,
        batch_size: int,
        pk: str,
        *,
        total_rows: int | None = None,
    ) -> LoadResult:
        """Read the rows above the watermark and project them into the target.

        **Deviation:** the predecessor pages with a keyset batch by batch,
        re-querying after each one (``WHERE pk > last``). Here the read is a
        single query through ``iter_batches``, because the ``pk > watermark``
        condition is fixed — a keyset would add nothing but N extra queries. The
        source is read once and consistently.

        The upsert is deliberate (rather than a plain ``INSERT``): the
        predecessor uses ``ON CONFLICT UPDATE`` and it is needed, because between
        the ``MAX(pk)`` and the read a row with that same key may already have
        reached the target by another route.

        **Checkpoints.** Every ``checkpoint_rows`` staged rows (default
        `DEFAULT_CHECKPOINT_ROWS`) the staging table is upserted into the target
        and committed. Because the read is ordered by the key, what is committed
        is always every delta row up to some key, so a failed run leaves the
        target consistent and the next run's ``MAX(pk)`` resumes exactly after
        it; whatever a resume reads twice, the upsert absorbs. Off (one
        transaction, as before 2.2.0) with ``checkpoint_rows: 0``, and always off
        for a surrogate key, whose read is not ordered.
        """
        full = FullLoadStrategy()
        hash_column = _hash_column(ctx)
        window_ctx = _with_where(ctx, where)
        columns = _all_columns(ctx, hash_column)
        conn = ctx.target_conn

        types = column_types(ctx, hash_column)
        target_pg.load_pg_schema(conn, ctx.target, types, _integer_columns(types))

        # **Before** the staging table, not just before the upsert: the staging
        # table is created with `LIKE <target>`, so it inherits the target's gaps
        # too, and the first batch's `COPY` fails before the columns could be
        # added. Same ordering as in `full`, `incremental` and
        # `parent_incremental`.
        full._ensure_managed_columns(ctx, hash_column)
        columns = _drop_columns_missing_in_target(ctx, columns)

        checkpoint_rows = _checkpoint_rows(ctx)
        if checkpoint_rows and _pk_in_source(ctx) is None:
            # Committing part of an **unordered** stream would leave the target
            # holding an arbitrary subset, and resuming above its `MAX(pk)` would
            # then skip every row not yet read below that key. Only a read
            # ordered by the watermark key (see `full.read_batches`) makes what
            # has been committed a prefix of the delta.
            _log.info(
                "ℹ️ Checkpoints are off: the read is not ordered by %s (surrogate key), "
                "so the delta is committed in one transaction.",
                pk,
            )
            checkpoint_rows = None

        staging = target_pg.create_shadow_table(conn, ctx.target, columns=columns)
        progress = status.BatchProgress(_log, total_rows=total_rows, phase=self.name)
        rows_read = rows_written = staged = checkpoints = 0
        last_key: Any = None
        try:
            for batch in full.read_batches(window_ctx, hash_column, batch_size):
                rows_read += len(batch)
                export_df = full._prepare(window_ctx, batch, hash_column, types)
                if export_df.empty:
                    continue
                target_pg.copy_from_stdin(conn, staging, export_df, columns)
                staged += len(export_df)
                if pk in export_df.columns:
                    last_key = export_df[pk].max()
                progress.tick(rows_read)

                if checkpoint_rows and staged >= checkpoint_rows:
                    rows_written += _upsert_from_staging(
                        conn, staging, ctx.target, columns, pk, ctx.partition_spec(conn)
                    )
                    # Dropped and created anew rather than truncated, in the same
                    # transaction as the upsert: the staging table is then never
                    # committed, so a run killed outright (an OOM kill runs no
                    # `except`) leaves no `dbx_shadow_*` table behind.
                    target_pg.drop_shadow_table(conn, staging)
                    conn.commit()
                    checkpoints += 1
                    staged = 0
                    _log.info(
                        "💾 Checkpoint %d: %s rows committed to the target so far "
                        "(%s up to %r); a retry resumes above it.",
                        checkpoints,
                        f"{rows_written:,}",
                        pk,
                        last_key,
                    )
                    staging = target_pg.create_shadow_table(conn, ctx.target, columns=columns)

            rows_written += _upsert_from_staging(
                conn, staging, ctx.target, columns, pk, ctx.partition_spec(conn)
            )
            target_pg.drop_shadow_table(conn, staging)
            conn.commit()
        except Exception:
            conn.rollback()
            try:
                target_pg.drop_shadow_table(conn, staging)
                conn.commit()
            except Exception as cleanup_err:
                _log.warning("⚠️ The staging table could not be cleaned up: %s", cleanup_err)
            if checkpoints:
                _log.warning(
                    "⚠️ The run failed after %d checkpoint(s); %s rows stay committed in "
                    "the target and the next run resumes above them.",
                    checkpoints,
                    f"{rows_written:,}",
                )
            raise

        return LoadResult(
            rows_read=rows_read,
            rows_written=rows_written,
            load_method=self.name,
            is_incremental=True,
            data_present=rows_written > 0,
            phase_metrics={"checkpoints": checkpoints},
        )


# --- Helper functions -------------------------------------------------------


def _watermark(conn: Any, target: Any, pk: str) -> Any | None:
    """``MAX(pk)`` in the target. ``None`` for an empty table.

    **Deviation:** the predecessor swallows an error from this query and goes to
    a full load. An unreachable database then looks like an empty target and the
    whole table is transferred; that is the "fail loudly" rule turned inside out.
    Here the error is allowed to propagate — and the target is the very database
    that is about to be written to, so if it is not working, a full load has
    nowhere to write either.
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX({target_pg.quote_ident(pk)}) FROM {target_pg.qualify(target)}")
        row = cur.fetchone()
    return row[0] if row else None


def _checkpoint_rows(ctx: LoadContext) -> int | None:
    """``LOAD_SETTINGS.checkpoint_rows``: ``None`` (not set) is the default, ``0`` is off.

    Raises on a value that is not a whole number, before the source is touched —
    the same reasoning as everywhere else in this package: a typo must not quietly
    choose a behaviour.
    """
    raw = ctx.settings.get("checkpoint_rows")
    if raw is None or raw == "":
        return DEFAULT_CHECKPOINT_ROWS
    value: int | None = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif (isinstance(raw, float) and raw.is_integer()) or (
        isinstance(raw, str) and raw.strip().lstrip("-").isdigit()
    ):
        value = int(raw)
    if value is None:
        raise StrategyError(f"id_watermark: checkpoint_rows must be a whole number, got {raw!r}.")
    return value if value > 0 else None


def _watermark_where(ctx: LoadContext, pk_source: str, watermark: Any) -> str:
    """``pk > watermark``, optionally glued to the condition from the configuration.

    The value goes into the SQL as a literal via `dialect.sql_literal` — for the
    same reason as in `hash_diff`: ``iter_batches`` knows nothing about bound
    parameters.
    """
    key_expr = ctx.dialect.render_key_expr(pk_source, ctx.surrogate)
    clause = f"{key_expr} > {ctx.dialect.sql_literal(watermark)}"
    if ctx.where:
        clause = f"({clause}) AND ({ctx.where})"
    return clause


def _source_name(ctx: LoadContext, target_name: str) -> str:
    """Target column name -> source name. A query against the source needs the latter."""
    for source, target in ctx.name_map.items():
        if target == target_name:
            return source
    return target_name


__all__ = ["IdWatermarkStrategy"]
