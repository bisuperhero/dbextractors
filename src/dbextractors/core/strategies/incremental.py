"""``IncrementalStrategy`` — a window over ``updated_at_column``/``created_at_column``.

Replaces the predecessor's ``load_incremental_data`` (10 variants, 151 lines).
Serves ~27 tables in production today.

## What the window is measured from — and what a longer outage does

This is written out **precisely** because it was documented nowhere and is a
source of silent holes in the data.

The window is measured **from the time of the run, not from the state of the
target**::

    cutoff = now() - days_back days, rounded down to midnight
    WHERE COALESCE(NULLIF(updated_at, '<zero date>'), created_at) >= cutoff

Three consequences follow, and all three are easy to overlook:

1. **An outage longer than ``days_back`` punches a hole in the data.** If the
   pipeline does not run for 10 days and ``days_back`` is 7, rows changed during
   the first three days of the outage fall outside the window and are **never
   picked up** — the next run again looks back only 7 days. Nothing reveals it:
   the run is green and transfers plenty of rows. The only repair is a full load.

   **This strategy does not detect it, and the docstring used to claim it did**
   (a `stale_by_days` metric and a warning; neither ever existed anywhere in the
   package — this is the file's second such drift, see point 3). Detecting it
   needs the interval between two successful runs, and nothing records when this
   table last ran: `target_pg.write_fingerprint` stores a time, but only for the
   fingerprint path, which `incremental` does not take. Every proxy the target
   *can* be asked for — ``MAX(_timestamp)``, ``MAX(updated_at)`` — cannot tell a
   table where nothing changed for a month from one whose changes were missed,
   so it would warn on healthy quiet tables for ever. A metric that fires when
   nothing is wrong is the class of bug this file already had to remove once.
   Recording each successful run's time would settle it, but that is new state
   written to the target and belongs to a decision of its own.
2. **A row with no change timestamp is never transferred.** ``NULL`` in both
   columns fails the condition. The `COALESCE` onto `created_at` softens that,
   it does not remove it.
3. **The window cannot see deleted rows.** ``_deleted_in_source`` is therefore
   declared on the target but stays ``FALSE`` under this strategy — same as the
   predecessor, whose incremental path never flipped it either. The strategy
   that genuinely maintains the column is `hash_diff`: its diff already holds
   the full set of live keys, so marking deletions costs nothing extra there.
   An earlier version of this docstring claimed the flip happened here through
   `target_pg.apply_live_pk_snapshot`; an audit showed nothing calls it. The
   claim was the bug — doing the sweep here would both diverge from the
   predecessor (breaking golden parity) and re-read every source key on every
   incremental run. A table that needs deletions tracked should use
   ``load_method: hash``.

When the target does not exist or is empty, control falls to
`core.strategies.base.fallback_full` — and that is always logged.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional, Tuple

from dbextractors.core import partitioning, status, target_pg
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
    _resolved_pk,
    column_types,
)

#: Default window when the configuration does not give one.
#: Taken from variant A's extractor.
DEFAULT_DAYS_BACK = 14

#: The month in which the "deep" cutoff flips over to the current accounting year.
#: Taken from variant B's extractor.
DEEP_CUTOFF_SWITCH_MONTH = 7


class IncrementalStrategy(LoadStrategy):
    """Transfer only the rows newer than the computed cutoff."""

    name = "incremental"
    required_features: frozenset = frozenset()

    def validate(self, ctx: LoadContext) -> None:
        """Missing columns are caught **before** the read, not during it.

        On a missing ``primary_column`` or ``updated_at_column`` the predecessor
        only logs and returns an empty result — the run then comes out green and
        transfers nothing. That is worse than a failure: it looks like "nothing
        changed".
        """
        super().validate(ctx)

        pk = _resolved_pk(ctx)
        if not pk:
            raise StrategyError(
                "incremental: primary_column (or primary_key_column) is missing "
                "from LOAD_SETTINGS. Without a key there is nothing to update."
            )
        if pk not in ctx.target_names and not surrogate_provides(ctx, pk):
            raise StrategyError(
                f"incremental: primary key {pk!r} is not among the table's columns."
            )

        updated = _updated_column(ctx)
        if not updated:
            raise StrategyError(
                "incremental: updated_at_column is missing from LOAD_SETTINGS. Without "
                "a change-timestamp column the window cannot be computed."
            )
        if updated not in ctx.source_names:
            raise StrategyError(
                f"incremental: updated_at_column {updated!r} is not in the source table."
            )

        created = _created_column(ctx)
        if created and created not in ctx.source_names:
            raise StrategyError(
                f"incremental: created_at_column {created!r} is not in the source table."
            )

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)

        if not target_pg.table_exists(ctx.target_conn, ctx.target):
            return fallback_full(ctx, "the target table does not exist")
        if _target_is_empty(ctx):
            return fallback_full(ctx, "the target table is empty")

        cutoff, _numeric, lookback_hours = compute_incremental_cutoff(
            ctx.settings, ctx.runtime, logger=ctx.logger
        )
        mode = window_mode(ctx.settings, lookback_hours)
        where = self._build_where(ctx, cutoff)

        ctx.log(
            "warning",
            "🚀 Incremental window from %s (%s), condition: %s",
            cutoff,
            window_label(ctx.settings, lookback_hours),
            where,
        )

        # The index is checked NOW, not after a COUNT(*) over the window.
        pk = _resolved_pk(ctx)
        assert pk is not None  # guaranteed by validate()
        target_pg.assert_unique_pk_index(ctx.target_conn, ctx.target, pk)

        estimate = ctx.dialect.estimate_size(ctx.engine, ctx.source, where)
        if estimate.rows == 0:
            ctx.log("warning", "ℹ️ There are no rows in the window.")
            return LoadResult(0, 0, self.name, is_incremental=True, data_present=False)

        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, estimate)
        ctx.log("warning", "🔢 Rows in the window: %s", f"{estimate.rows:,}")

        result = self._load_window(ctx, where, batch_size, total_rows=estimate.rows)
        result.phase_metrics.update(
            {
                "cutoff": str(cutoff),
                # `days_back` only when the window really came from it. It used to
                # be written unconditionally (and `config.parse` supplies 14), so
                # under a deep cutoff the telemetry repeated the same lie as the
                # log line did.
                "mode": mode,
                "days_back": (
                    int(ctx.settings.get("days_back") or DEFAULT_DAYS_BACK)
                    if mode == "days_back"
                    else None
                ),
                "lookback_hours": lookback_hours,
                "estimated_rows": estimate.rows,
            }
        )
        return result

    def _build_where(self, ctx: LoadContext, cutoff: date) -> str:
        """The window condition. The ``COALESCE`` onto ``created_at`` comes from the predecessor.

        A zero date in the source (`0000-00-00 00:00:00` on MySQL) must first be
        normalised to ``NULL``, otherwise ``COALESCE`` would never reach for
        ``created_at`` and a row with a zeroed `updated_at` would drop out of the
        window. How a zero date is written is the dialect's knowledge.

        **There are two shapes of window and they are not interchangeable.**
        Which key the configuration uses decides which condition it gets::

            updated_at_column        COALESCE(NULLIF(updated, '<zero>'), created) >= cutoff
            incremental_date_column  (COALESCE(a, 0) >= cutoff OR COALESCE(b, 0) >= cutoff)

        The first looks at the fallback column **only when the main one is
        NULL**; the second takes the row when either of the two is above the
        threshold. A row with a stale correction timestamp and a fresh creation
        timestamp is therefore taken by the second shape and dropped by the
        first. Mapping them onto each other would transfer something different
        from what is transferred today, so each configuration keeps its own
        shape. See ``docs/legacy-compat.md``.
        """
        updated_raw = _updated_column(ctx)
        assert updated_raw is not None  # guaranteed by validate()
        updated = ctx.dialect.quote_ident(updated_raw)
        created = _created_column(ctx)
        sentinel = getattr(ctx.dialect, "zero_datetime_literal", None)

        if _uses_legacy_date_column_window(ctx):
            # Literally the predecessor's parent condition: `OR` across both
            # columns, `COALESCE` onto zero so that an empty value fails the
            # condition.
            value = ctx.dialect.sql_literal(ctx.dialect.cutoff_value(cutoff))
            columns = [updated] + ([ctx.dialect.quote_ident(created)] if created else [])
            parts = [f"COALESCE({c}, 0) >= {value}" for c in columns]
            clause = "(" + " OR ".join(parts) + ")"
            if ctx.where:
                clause = f"({clause}) AND ({ctx.where})"
            return clause

        if created:
            created_q = ctx.dialect.quote_ident(created)
            expr = (
                f"COALESCE(NULLIF({updated}, {sentinel!r}), {created_q})"
                if sentinel
                else f"COALESCE({updated}, {created_q})"
            )
        else:
            expr = updated

        clause = f"{expr} >= '{cutoff.strftime('%Y-%m-%d')} 00:00:00'"
        if ctx.where:
            clause = f"({clause}) AND ({ctx.where})"
        return clause

    def _load_window(
        self,
        ctx: LoadContext,
        where: str,
        batch_size: int,
        *,
        total_rows: Optional[int] = None,
    ) -> LoadResult:
        """Read the window and project it into the target with an upsert.

        The write goes through a staging table and ``INSERT … ON CONFLICT``, not
        through a row-by-row ``executemany``: `COPY` is the only write path in
        this package, and `COPY` cannot upsert, hence the staging table.

        ``total_rows`` only serves progress reporting. Without it this path was
        silent from start to finish: 13.7 minutes of silence measured on a
        4.87 M row window, whereas the predecessor announced every batch with an
        `n/12` counter. Over a run of tens of minutes that is the difference
        between "I can see where I am" and "I am waiting and hoping".
        """
        full = FullLoadStrategy()
        hash_column = _hash_column(ctx)
        window_ctx = _with_where(ctx, where)
        columns = _all_columns(ctx, hash_column)
        pk = _resolved_pk(ctx)

        conn = ctx.target_conn
        # **Before** the staging table: it is created with `LIKE <target>`, so it
        # inherits the target's gaps too, and the first batch's `COPY` then fails
        # with `column "row_hash" … does not exist`. Observed in production. Same
        # ordering as in `full` and `parent_incremental`.
        full._ensure_managed_columns(ctx, hash_column)
        columns = _drop_columns_missing_in_target(ctx, columns)
        staging = target_pg.create_shadow_table(
            conn,
            ctx.target,
            df=None,
            overwrite_types=column_types(ctx, hash_column),
            columns=columns,
        )
        rows_read = rows_written = 0
        progress = status.BatchProgress(ctx.log, total_rows=total_rows, phase="staging")
        try:
            for batch in full.read_batches(window_ctx, hash_column, batch_size):
                rows_read += len(batch)
                export_df = full._prepare(window_ctx, batch, hash_column)
                if export_df.empty:
                    continue
                target_pg.copy_from_stdin(conn, staging, export_df, columns)
                progress.tick(rows_read)
            # An upsert over a million-row staging table takes minutes and is the
            # one step during which the target can stay unchanged for a long
            # time — so its start is announced, not just its result.
            ctx.log(
                "warning",
                "✅ Staging done (%s rows), projecting into the target…",
                f"{rows_read:,}",
            )
            rows_written = _upsert_from_staging(
                conn, staging, ctx.target, columns, pk, ctx.partition_spec(conn)
            )
            ctx.log("warning", "✅ %s rows projected into the target.", f"{rows_written:,}")
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

        return LoadResult(
            rows_read=rows_read,
            rows_written=rows_written,
            load_method=self.name,
            is_incremental=True,
            data_present=rows_written > 0,
        )


def compute_incremental_cutoff(
    load_settings: dict, kwargs: dict, logger: Optional[Any] = None
) -> Tuple[date, Optional[int], Optional[int]]:
    """Compute the cutoff shared by the incremental and the parent-driven path.

    The signature follows the predecessor's ``_compute_incremental_cutoff(
    load_settings, kwargs, logger)`` — a single variant, found only in variant
    B's Firebird extractor. `strategies.parent_incremental` uses it as well, so
    that a child's window matches its parent's.

    Two modes:

    - **delta** — ``incremental_lookback_hours`` is set (the runtime variable
      overrides LOAD_SETTINGS): cutoff = now minus N hours.
    - **deep** — otherwise: 1 January of the previous or the current year,
      depending on whether it is before or after July. Used by the warm-up run.

    **Deviation:** in the predecessor the returned ``cutoff_numeric`` is a count
    of days since the Firebird epoch. That is the dialect's knowledge, not the
    strategy's, so ``None`` is returned here and Firebird does the conversion
    itself. The third member (the lookback actually used) stays.
    """
    log = logger.warning if logger else (lambda *a: None)
    err = logger.error if logger else (lambda *a: None)

    lookback = kwargs.get("incremental_lookback_hours")
    if lookback is None:
        lookback = load_settings.get("incremental_lookback_hours")

    lookback_hours: Optional[int]
    if lookback is None or lookback in ("", 0, "0"):
        lookback_hours = None
    else:
        try:
            lookback_hours = int(lookback)
        except (TypeError, ValueError):
            err("⚠️ Invalid incremental_lookback_hours=%r, ignoring it (deep cutoff).", lookback)
            lookback_hours = None

    today = date.today()
    mode = window_mode(load_settings, lookback_hours)
    if mode == "lookback":
        cutoff = (datetime.now() - timedelta(hours=lookback_hours or 0)).date()
        log("📅 Lookback %s h — delta mode.", lookback_hours)
    elif mode == "days_back":
        days_back = int(load_settings.get("days_back") or DEFAULT_DAYS_BACK)
        cutoff = (datetime.now() - timedelta(days=days_back)).date()
    else:
        year = today.year - 1 if today.month < DEEP_CUTOFF_SWITCH_MONTH else today.year
        cutoff = date(year, 1, 1)

    return cutoff, None, lookback_hours


def window_mode(load_settings: dict, lookback_hours: Optional[int]) -> str:
    """Which of the three window modes applies: ``lookback`` / ``days_back`` / ``deep``.

    The single place where that is decided. The returned token also reaches
    ``phase_metrics['mode']``, so it is machine-readable output, not prose.

    Previously `compute_incremental_cutoff` held the branch itself and each
    strategy assembled its own log label — each of them knew only two of the
    three branches, so a deep cutoff was reported as ``days_back 14``. The window
    was right (1 January), the label next to it lied, and while investigating
    "why did it transfer that many rows" it pointed the wrong way.
    """
    if lookback_hours:
        return "lookback"
    if not _deep_cutoff_applies(load_settings) and load_settings.get("days_back") is not None:
        return "days_back"
    return "deep"


def window_label(load_settings: dict, lookback_hours: Optional[int]) -> str:
    """The window label for the log. Derived from `window_mode`, not assembled again."""
    mode = window_mode(load_settings, lookback_hours)
    if mode == "lookback":
        return f"lookback {lookback_hours} h"
    if mode == "days_back":
        return f"days_back {int(load_settings.get('days_back') or DEFAULT_DAYS_BACK)}"
    return "deep cutoff"


# --- Helper functions -------------------------------------------------------


def _deep_cutoff_applies(load_settings: dict) -> bool:
    """``True`` when the configuration describes its window with variant B's keys.

    Variant B's predecessor has **no ``days_back`` branch at all**: it knows only
    two modes — lookback, or a deep cutoff (1 January of last/this year). The
    other deployments compute the window as ``now − days_back`` and have no
    concept of a deep cutoff.

    The decision therefore has to be made on the **presence of the key**, not on
    which deployment is running — same as in
    :func:`_uses_legacy_date_column_window`. Without that, ``days_back`` always
    wins: ``config.parse`` supplies 14 even when the configuration does not
    mention it, which would make the deep branch unreachable for variant B.

    Observed in production: this package computed a cutoff of 2026-07-30
    (= today − 14 days) and transferred 0 rows, while the predecessor computed
    2026-01-01 and transferred 394.

    ``incremental_parent_table`` is listed here because a child table has no date
    column of its own — for a child the shape is only recognisable via its
    parent.
    """
    return bool(
        load_settings.get("incremental_date_column")
        or load_settings.get("incremental_parent_table")
    )


def _uses_legacy_date_column_window(ctx: LoadContext) -> bool:
    """``True`` when the configuration carries ``incremental_date_column``.

    That key selects the older of the two window shapes documented in
    ``docs/legacy-compat.md``: an ``OR`` across the primary and fallback date
    columns, each compared with ``COALESCE(column, 0) >= cutoff``, rather than a
    ``COALESCE`` of one onto the other.

    What decides is the **presence of the key**, not which deployment is
    running. A configuration carrying `incremental_date_column` gets the
    condition this function selects; one carrying `updated_at_column` gets its
    own. No existing configuration changes behaviour because of this.
    """
    return bool(ctx.settings.get("incremental_date_column"))


def _updated_column(ctx: LoadContext) -> Optional[str]:
    return ctx.settings.get("incremental_date_column") or ctx.settings.get("updated_at_column")


def _created_column(ctx: LoadContext) -> Optional[str]:
    value = ctx.settings.get("incremental_date_column_fallback") or ctx.settings.get(
        "created_at_column"
    )
    return value or None


def _all_columns(ctx: LoadContext, hash_column: Optional[str]) -> list:
    """The existing target's order wins — see `naming.reconcile_with_existing`."""
    from dbextractors.core import naming

    wanted = naming.target_column_names(ctx.source_names, hash_column)
    return naming.reconcile_with_existing(
        wanted, target_pg.existing_column_order(ctx.target_conn, ctx.target)
    )


def _drop_columns_missing_in_target(ctx: LoadContext, columns: list) -> list:
    """Drop columns the target does not have and log it loudly. Returns the rest.

    The source gains columns. Strategies that touch **existing** data
    (incremental, hash, id_watermark, parent_incremental) do not change the shape
    of the target — the staging table is created with ``LIKE <target>``, so an
    extra column would bring down the first batch's ``COPY``::

        column "promo_code_code" of relation "dbx_shadow_<table>_…"
        does not exist

    The predecessor did the same thing, only silently (a ``reindex`` onto the
    target's columns). The dropping stays, the silence does not: without a
    message nobody finds out that not everything is reaching the warehouse. The
    next full load makes it up, adopting the column
    (`full.FullLoadStrategy._adopt_new_columns`).
    """
    missing = target_pg.columns_missing_in_target(ctx.target_conn, ctx.target, columns)
    if not missing:
        return columns

    ctx.log(
        "warning",
        "⚠️ The source has %d columns the target lacks: %s — dropping them. "
        "The next full load will make them up, or add them by hand.",
        len(missing),
        ", ".join(missing),
    )
    discard = set(missing)
    return [c for c in columns if c not in discard]


def _target_is_empty(ctx: LoadContext) -> bool:
    with ctx.target_conn.cursor() as cur:
        cur.execute(f"SELECT EXISTS (SELECT 1 FROM {target_pg.qualify(ctx.target)} LIMIT 1)")
        return not cur.fetchone()[0]


def _with_where(ctx: LoadContext, where: str) -> LoadContext:
    """A copy of the context with a different condition. The context is never mutated."""
    import dataclasses

    return dataclasses.replace(ctx, where=where)


def _upsert_from_staging(
    conn, staging, target, columns: list, pk: Optional[str], spec: Any = None
) -> int:
    """``INSERT … SELECT … ON CONFLICT DO UPDATE`` from the staging table.

    Replaces the predecessor's ``loader.export(…, unique_conflict_method=
    'UPDATE')``, which internally issues an ``executemany`` over
    ``df.iterrows()`` — row-by-row INSERTs. That is the write path this package
    forbids; here the database does it in a single statement.

    **Over a partitioned target ``ON CONFLICT`` is not used.** It would have
    nothing to lean on (the unique index there has to contain the partition key),
    and a widened conflict key would duplicate a row whose partition value
    changes instead of updating it. The write goes through
    `target_pg.replace_by_key` instead. What decides is **the reality in the
    target**, not the configuration — another strategy may have created the
    partitioned table.

    All three upserting strategies flow through here: `incremental`,
    `id_watermark` and `hash_diff`.
    """
    if spec is not None and not target_pg.table_exists(conn, target):
        partitioning.create_parent(conn, target, staging, spec)

    if target_pg.is_partitioned(conn, target):
        if spec is not None:
            partitioning.ensure_partitions_from_table(conn, target, spec, staging)
        if pk:
            return target_pg.replace_by_key(conn, staging, target, columns, pk)

    quoted = ", ".join(target_pg.quote_ident(c) for c in columns)
    sql = (
        f"INSERT INTO {target_pg.qualify(target)} ({quoted}) "
        f"SELECT {quoted} FROM {target_pg.qualify(staging)}"
    )
    if pk:
        updates = ", ".join(
            f"{target_pg.quote_ident(c)} = EXCLUDED.{target_pg.quote_ident(c)}"
            for c in columns
            if c != pk
        )
        sql += f" ON CONFLICT ({target_pg.quote_ident(pk)}) DO UPDATE SET {updates}"

    with conn.cursor() as cur:
        cur.execute(sql)
        return int(cur.rowcount)


__all__ = ["IncrementalStrategy", "compute_incremental_cutoff", "DEFAULT_DAYS_BACK"]
