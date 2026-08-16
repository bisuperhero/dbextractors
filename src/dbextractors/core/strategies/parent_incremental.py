"""``ParentIncrementalStrategy`` — a window over the parent table's date.

The predecessor's ``load_incremental_by_parent`` and ``_fb_parent_condition``
(2 functions, 162 lines) exist **only in the Firebird extractor**. It is used
for child tables that have no trustworthy change-date column of their own but do
have a foreign key to a parent table that has one — so the window is computed
over the parent, not over the child (``incremental_parent_table`` +
``incremental_parent_date_column``, with an optional
``incremental_parent_date_column_fallback``).

The cutoff is shared with ``incremental``
(:func:`dbextractors.core.strategies.incremental.compute_incremental_cutoff`),
so that a child's window matches the window its parent's own ``incremental`` run
computed. Serves 24 tables in production today.

## Delete and reload — but in one transaction

The mechanics differ from the other strategies: child rows whose parent falls
inside the window are **deleted** from the target and loaded again. It is the
only way to also catch a child row that vanished from the source — the child has
no change date of its own, so a deletion cannot be learned about any other way.

The predecessor **commits** that `DELETE` and only then starts streaming data.
When the stream falls over — and over Firebird via SSH that is not a theoretical
possibility — the target is left without the deleted rows: the run looks
unsuccessful while the data is gone. Here the `DELETE` and the write are **in
one transaction**: everything is loaded into a staging table first, and only
then do the `DELETE` + `INSERT … SELECT` and the commit happen. Same reasoning
as for the shadow table in `full`.

## `COALESCE(column, 0) >= cutoff` simplified to `column >= cutoff`

The predecessor compares `COALESCE("DATE_COL", 0) >= cutoff_numeric`, because
Firebird holds a date as a count of days since the epoch. That zero does
nothing, though: for any realistic cutoff (> 0) it means "NULL does not count",
which is exactly what a plain `"DATE_COL" >= cutoff` does too — `NULL >= x` is
`NULL`, hence unsatisfied. Dropping the zero unties the condition from the date
being a number, and makes it work for dialects where it is a `DATE`.

How the cutoff is written into the source SQL is the dialect's knowledge
(`cutoff_value`) — on Firebird it is a day count, elsewhere an ISO date. That
removes the last piece of Firebird-specific knowledge from here.

## An empty target is **not** a full load here

`incremental` and `id_watermark` fall back to a full load when the target exists
but holds no rows. This strategy does not: it checks that the child and the
parent tables **exist** and nothing more, so over an empty target it transfers
the window alone and leaves everything older missing — with a green run.

That is deliberate parity, not an omission. The predecessor's
``load_incremental_by_parent`` gates on ``to_regclass`` and has no row count
anywhere either, so a fallback here would transfer a different set of rows than
the old side does. Pinned by
``tests/strategies/test_parent_incremental.py::test_an_empty_target_is_not_backfilled``;
closing the gap means changing that test on purpose.
"""

from __future__ import annotations

from datetime import date
from typing import Any, List, Optional

from dbextractors.core import status, target_pg
from dbextractors.core.strategies.base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
    TargetRef,
    fallback_full,
    resolve_batch_size,
)
from dbextractors.core.strategies.full import (
    FullLoadStrategy,
    _hash_column,
    _integer_columns,
    _resolved_pk,
    column_types,
)
from dbextractors.core.strategies.incremental import (
    _all_columns,
    _drop_columns_missing_in_target,
    _with_where,
    compute_incremental_cutoff,
    window_label,
)
from dbextractors.dialects.base import FEATURE_PARENT_INCREMENTAL

#: The child's column referencing the parent. The predecessor hard-codes it
#: (`parent_id` in the target, `"PARENT_ID"` in the source); here it is the
#: default of ``LOAD_SETTINGS.incremental_parent_key_column``, so the strategy
#: also works where the foreign key is named differently.
DEFAULT_PARENT_KEY = "parent_id"

#: The parent key the child points at, i.e. the default of
#: ``LOAD_SETTINGS.incremental_parent_id_column``. The predecessor hard-codes
#: `id` / `"ID"` and has no configuration for it at all.
#:
#: Both keys were read here before they were part of the contract, and
#: `entrypoint._settings_dict` builds ``ctx.settings`` with
#: ``dataclasses.asdict`` — so neither could ever arrive and these two constants
#: always won. Today's behaviour was therefore the predecessor's, which is why
#: adding the keys changed nothing for anyone; what it changed is that
#: `validate`'s "Set incremental_parent_key_column." is now advice that can be
#: followed.
DEFAULT_PARENT_ID = "id"


class ParentIncrementalStrategy(LoadStrategy):
    """A child table's window taken from the change date in its parent table."""

    name = "parent_incremental"
    required_features: frozenset = frozenset({FEATURE_PARENT_INCREMENTAL})

    def validate(self, ctx: LoadContext) -> None:
        """Check that both the parent table and its date column are configured.

        The predecessor's ``load_incremental_by_parent`` deals with a missing
        parent configuration at run time (falling back to a full load); here it
        is caught earlier — as in the other strategies, a missing configuration
        is an error, not a reason for the most expensive possible run.
        """
        super().validate(ctx)

        if not ctx.settings.get("incremental_parent_table"):
            raise StrategyError(
                "parent_incremental: incremental_parent_table is missing from LOAD_SETTINGS. "
                "Without a parent table there is nothing to compute the window from."
            )
        if not ctx.settings.get("incremental_parent_date_column"):
            raise StrategyError(
                "parent_incremental: incremental_parent_date_column is missing from "
                "LOAD_SETTINGS."
            )

        fk_column = _parent_key(ctx)
        if fk_column not in ctx.target_names:
            raise StrategyError(
                f"parent_incremental: the column referencing the parent {fk_column!r} is not "
                f"among the table's columns. Set incremental_parent_key_column."
            )

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)

        conn = ctx.target_conn
        if not target_pg.table_exists(conn, ctx.target):
            return fallback_full(ctx, "the target child table does not exist")

        parent_ref = _parent_target(ctx)
        if not target_pg.table_exists(conn, parent_ref):
            # Without the parent in the target there is nothing to select the
            # rows to delete from.
            return fallback_full(
                ctx, f"the parent table {parent_ref.qualified()} is missing from the target"
            )

        # `ctx.runtime`, not `{}`: parent and child must get **the same** cutoff,
        # so when a run is given `incremental_lookback_hours` it has to show up
        # in both. The predecessor does the same — the shared
        # `_compute_incremental_cutoff` takes kwargs on both paths.
        cutoff, _numeric, lookback = compute_incremental_cutoff(
            ctx.settings, ctx.runtime, logger=ctx.logger
        )
        ctx.log(
            "warning",
            "🚀 Window from parent %s starting %s (%s), columns %s.",
            ctx.settings.get("incremental_parent_table"),
            cutoff,
            window_label(ctx.settings, lookback),
            ", ".join(_parent_date_columns(ctx)),
        )

        pk = _resolved_pk(ctx)
        if pk:
            # The predecessor creates the index **before** the loop on purpose:
            # without it a repeated run after an unfinished delete would produce
            # duplicates.
            target_pg.ensure_unique_index(conn, ctx.target, pk, ctx.target_names)

        where = self._source_where(ctx, cutoff)
        estimate = ctx.dialect.estimate_size(ctx.engine, ctx.source, where)
        if estimate.rows == 0:
            ctx.log("warning", "ℹ️ There are no child rows in the parent's window.")
            return LoadResult(0, 0, self.name, is_incremental=True, data_present=False)

        ctx.log("warning", "🔢 Rows in the window: %s", f"{estimate.rows:,}")
        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, estimate)
        result = self._replace_window(
            ctx, where, batch_size, cutoff, parent_ref, total_rows=estimate.rows
        )
        result.phase_metrics.update(
            {
                "cutoff": str(cutoff),
                "lookback_hours": lookback,
                "estimated_rows": estimate.rows,
                "parent_table": parent_ref.table,
            }
        )
        return result

    # -- steps -------------------------------------------------------------

    def _source_where(self, ctx: LoadContext, cutoff: date) -> str:
        """The condition against the source: "the parent falls inside the window"."""
        # `source_ident`, not `quote_ident`: these three names do not come from
        # source introspection but from the configuration or from a default in
        # the code (`parent_id`, `id`) — on a dialect that folds identifiers they
        # would not be found when quoted.
        parent_ident = ctx.dialect.source_ident(str(ctx.settings["incremental_parent_table"]))
        fk_ident = ctx.dialect.source_ident(_parent_key_source(ctx))
        parent_id_ident = ctx.dialect.source_ident(_parent_id_column(ctx))
        condition = self._parent_condition(ctx, cutoff, _parent_date_columns(ctx))

        clause = f"{fk_ident} IN (SELECT {parent_id_ident} FROM {parent_ident} WHERE {condition})"
        if ctx.where:
            clause = f"({clause}) AND ({ctx.where})"
        return clause

    def _parent_condition(self, ctx: LoadContext, cutoff: date, columns: List[str]) -> str:
        """Build the SQL condition "the parent falls inside the window" over the parent's columns.

        The predecessor's ``_fb_parent_condition()`` — a parameterless closure
        over ``cutoff_numeric`` and the parent's column names, computed in the
        body of ``load_incremental_by_parent``. Here the parameters are explicit.

        There can be more than one column (primary and fallback) and they are
        joined with ``OR``: on some tables the change date is filled in only one
        of them.
        """
        value = ctx.dialect.sql_literal(ctx.dialect.cutoff_value(cutoff))
        parts = [f"{ctx.dialect.quote_ident(c)} >= {value}" for c in columns]
        return "(" + " OR ".join(parts) + ")"

    def _replace_window(
        self,
        ctx: LoadContext,
        where: str,
        batch_size: int,
        cutoff: date,
        parent_ref: TargetRef,
        *,
        total_rows: Optional[int] = None,
    ) -> LoadResult:
        """Load the window into a staging table and only then delete and insert.

        The ordering is the entire point of this method. The predecessor deletes
        first and commits, so a failed stream leaves a hole in the target. Here
        the delete happens only once the new data is ready, and both are in one
        transaction.
        """
        full = FullLoadStrategy()
        hash_column = _hash_column(ctx)
        window_ctx = _with_where(ctx, where)
        columns = _all_columns(ctx, hash_column)
        conn = ctx.target_conn

        types = column_types(ctx, hash_column)
        target_pg.load_pg_schema(conn, ctx.target, types, _integer_columns(types))

        # **Before** the staging table, not just before the `INSERT … SELECT`:
        # the staging table is created with `LIKE <target>`, so it inherits the
        # target's gaps too. When the target was created by an older extractor
        # without `row_hash`, the first batch's `COPY` already fails on it — that
        # is, before the columns could be added:
        #
        #     column "row_hash" of relation "dbx_shadow_<table>_…"
        #     does not exist
        #
        # Observed in production. Same reasoning and same ordering as in
        # `full._ensure_managed_columns`.
        full._ensure_managed_columns(ctx, hash_column)
        columns = _drop_columns_missing_in_target(ctx, columns)

        staging = target_pg.create_shadow_table(conn, ctx.target, columns=columns)
        progress = status.BatchProgress(ctx.log, total_rows=total_rows, phase=self.name)
        rows_read = rows_written = deleted = 0
        try:
            for batch in full.read_batches(window_ctx, hash_column, batch_size):
                rows_read += len(batch)
                export_df = full._prepare(window_ctx, batch, hash_column, types)
                if export_df.empty:
                    continue
                target_pg.copy_from_stdin(conn, staging, export_df, columns)
                progress.tick(rows_read)

            deleted = self._delete_window(ctx, cutoff, parent_ref)
            rows_written = _insert_from_staging(conn, staging, ctx.target, columns)
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

        ctx.log("warning", "✅ %s rows deleted, %s inserted.", f"{deleted:,}", f"{rows_written:,}")
        return LoadResult(
            rows_read=rows_read,
            rows_written=rows_written,
            load_method=self.name,
            is_incremental=True,
            data_present=rows_written > 0,
            phase_metrics={"deleted": deleted},
        )

    def _delete_window(self, ctx: LoadContext, cutoff: date, parent_ref: TargetRef) -> int:
        """Delete from the target the child rows whose parent falls inside the window.

        The parent's column names are taken **in target naming** (`core.naming`),
        not in source naming: in the target the parent is stored already
        normalised, so `CHANGED_AT` is `changed_at` there — and if it were a
        reserved word, with an underscore.

        The cutoff goes through ``dialect.cutoff_value`` here as well, where the
        target is being queried rather than the source. It looks like a layering
        mix-up but it is right: the target holds a copy of the source column, so
        on Firebird it is a count of days since the epoch just as in the source.
        Using a date here would compare against nothing.
        """
        from dbextractors.core import naming

        columns = [naming.clean_column_name(c) for c in _parent_date_columns(ctx, target=True)]
        value = ctx.dialect.cutoff_value(cutoff)
        conditions = " OR ".join(f"{target_pg.quote_ident(c)} >= %s" for c in columns)
        fk_col = target_pg.quote_ident(_parent_key(ctx))
        parent_id_col = target_pg.quote_ident(naming.clean_column_name(_parent_id_column(ctx)))

        sql = (
            f"DELETE FROM {target_pg.qualify(ctx.target)} WHERE {fk_col} IN "
            f"(SELECT {parent_id_col} FROM {target_pg.qualify(parent_ref)} WHERE {conditions})"
        )
        with ctx.target_conn.cursor() as cur:
            cur.execute(sql, tuple([value] * len(columns)))
            return int(cur.rowcount)


# --- Helper functions -------------------------------------------------------


def _parent_target(ctx: LoadContext) -> TargetRef:
    """Where the parent table lives in the target.

    The predecessor looks for it in the same schema as the child and merely
    lower-cases the name. That is adopted; ``incremental_parent_output_name`` can
    override it should the parent be named differently in the target.
    """
    table_name = (
        ctx.settings.get("incremental_parent_output_name")
        or str(ctx.settings["incremental_parent_table"]).lower()
    )
    return TargetRef(schema=ctx.target.schema, table=table_name)


def _parent_date_columns(ctx: LoadContext, target: bool = False) -> List[str]:
    """The parent's date columns — the primary one and an optional fallback."""
    del target  # the names are the same; the caller translates to target naming
    columns = [str(ctx.settings["incremental_parent_date_column"])]
    fallback = ctx.settings.get("incremental_parent_date_column_fallback")
    if fallback:
        columns.append(str(fallback))
    return columns


def _parent_key(ctx: LoadContext) -> str:
    """The child's column referencing the parent, in **target** naming."""
    from dbextractors.core import naming

    raw = ctx.settings.get("incremental_parent_key_column") or DEFAULT_PARENT_KEY
    return ctx.name_map.get(raw, naming.clean_column_name(raw))


def _parent_key_source(ctx: LoadContext) -> str:
    """The same column, but under its name in the source."""
    raw = ctx.settings.get("incremental_parent_key_column")
    if raw:
        return str(raw)
    target_name = _parent_key(ctx)
    for source, target in ctx.name_map.items():
        if target == target_name:
            return source
    return DEFAULT_PARENT_KEY


def _parent_id_column(ctx: LoadContext) -> str:
    return str(ctx.settings.get("incremental_parent_id_column") or DEFAULT_PARENT_ID)


def _insert_from_staging(conn: Any, staging: Any, target: Any, columns: List[str]) -> int:
    """``INSERT … SELECT`` without ``ON CONFLICT``.

    Unlike the incremental path, no upsert is needed here: the window was deleted
    from the target first, so nothing conflicting was left in it. Should a
    conflict happen anyway, that is a sign the `DELETE` did not take effect — and
    that has to fail, not quietly overwrite.
    """
    quoted = ", ".join(target_pg.quote_ident(c) for c in columns)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {target_pg.qualify(target)} ({quoted}) "
            f"SELECT {quoted} FROM {target_pg.qualify(staging)}"
        )
        return int(cur.rowcount)


__all__ = ["ParentIncrementalStrategy", "DEFAULT_PARENT_KEY", "DEFAULT_PARENT_ID"]
