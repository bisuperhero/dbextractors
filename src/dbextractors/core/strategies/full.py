"""``FullLoadStrategy`` — full load into a shadow table with an atomic swap.

The predecessor's ``load_full_data`` (12 variants, 194 lines) does
``DROP TABLE … CASCADE`` and only then starts streaming. Both halves of that are
forbidden here:

- ``DROP CASCADE`` takes down up to 102 views, and an interruption leaves the
  target truncated — documented by a run that ended with 11.4 of 18.8 M rows.
- The replacement is a load into a **shadow table** and a swap in a single
  transaction. The mechanics live in `core.target_pg`; **it is not a `RENAME`**
  — a view holds an OID, so after a rename it would stay bound to the old table
  and quietly serve last run's data. Verified on PostgreSQL 17.9.

Serves ~90 tables in production today.

## Options that are a switch, not a file of their own

`skip_size_estimate`, `resume_full_load`, `keyset_pagination`,
`compute_row_hash` and `conflict_columns` live in a forked copy of one
predecessor extractor that serves 3 tables. A forked copy is exactly what this
package exists to remove, so they belong here as keys in `LOAD_SETTINGS` whose
default preserves the behaviour of all the other files.

Two of those five are **accepted and inert**: nothing reads `keyset_pagination`
(whether a read can be resumed by keyset is decided by `_keyset_usable`, from the
primary key and the dialect's `FEATURE_KEYSET`) or `conflict_columns` (the upsert
matches on the primary key alone). They stay in the contract so the pipelines
that set them keep starting; see the fields in `core/config.py` and
docs/legacy-compat.md.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from dbextractors.core import coerce, hashing, partitioning, status, target_pg
from dbextractors.core.strategies.base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
    apply_text_dtypes,
    resolve_batch_size,
    surrogate_provides,
)


class FullLoadStrategy(LoadStrategy):
    """Move the whole table into a shadow table and swap it in atomically."""

    name = "full"
    required_features: frozenset = frozenset()

    def validate(self, ctx: LoadContext) -> None:
        """Check whatever can be checked **before** the source is first touched.

        The predecessor's ``load_full_data`` has no validation phase of its own —
        the checks are scattered through the body and some of them only surface
        after a ``COUNT(*)`` over the whole table.
        """
        super().validate(ctx)

        if not ctx.target.schema or not ctx.target.table:
            raise StrategyError("full: neither the target schema nor the table may be empty.")

        pk = _resolved_pk(ctx)
        if pk and pk not in ctx.target_names and not surrogate_provides(ctx, pk):
            raise StrategyError(
                f"full: primary key {pk!r} is not among the target columns "
                f"({', '.join(ctx.target_names[:8])}…)."
            )
        if _is_truthy(ctx.settings.get("resume_full_load")) and not _keyset_usable(ctx):
            ctx.log(
                "warning",
                "ℹ️ resume_full_load is on, but keyset paging cannot be used "
                "(primary_column=%r) — ignoring it.",
                pk,
            )

    def run(self, ctx: LoadContext) -> LoadResult:
        ctx.log("warning", "🚀 Full load: %s -> %s", ctx.source, ctx.target.qualified())
        self.validate(ctx)

        estimate = self._estimate(ctx)
        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, estimate)
        total_rows = None if estimate is None else estimate.rows

        if total_rows == 0:
            return self._empty_target(ctx)

        hash_column = _hash_column(ctx)
        self._ensure_managed_columns(ctx, hash_column)

        promoted: dict = {}
        shadow = None
        rows_read = 0
        rows_written = 0
        conn = ctx.target_conn
        pk = _resolved_pk(ctx)
        spec = ctx.partition_spec(conn)
        progress = status.BatchProgress(ctx.log, total_rows=total_rows, phase=self.name)
        try:
            with target_pg.SeenKeys(conn, pk or "") as seen:
                for batch in self.read_batches(ctx, hash_column, batch_size):
                    rows_read += len(batch)
                    if shadow is None:
                        # The shadow is created **from the first batch**, not
                        # earlier: type promotion (enum(0,1) -> BOOLEAN,
                        # INTEGER -> BIGINT) can only be recognised from data,
                        # and the DDL has to know about it. The predecessor does
                        # the same — the table is created by the first batch's
                        # `export()`.
                        promoted = _promote_types(ctx, batch, hash_column)
                        self._adopt_new_columns(ctx, _all_columns(ctx, hash_column), promoted)
                        shadow = target_pg.create_shadow_table(
                            conn,
                            ctx.target,
                            df=_empty_frame(ctx.target_names),
                            overwrite_types=promoted,
                            columns=_all_columns(ctx, hash_column),
                        )
                    export_df = self._prepare(ctx, batch, hash_column, promoted)
                    if pk:
                        export_df = seen.filter_new(export_df)
                    if export_df.empty:
                        continue
                    rows_written += target_pg.copy_from_stdin(
                        conn, shadow, export_df, _all_columns(ctx, hash_column)
                    )
                    progress.tick(rows_written)

            if shadow is None:
                # The estimate reported rows, but the source returned not one
                # batch. In that case the target is left alone — same as for an
                # empty source without `empty_rows_ok`.
                ctx.log("warning", "⚠️ The source returned no batch, the target is left as is.")
                conn.rollback()
                return LoadResult(rows_read, 0, self.name, data_present=False)

            target_pg.swap_shadow_table(conn, shadow, ctx.target, spec)
            target_pg.drop_shadow_table(conn, shadow)
            conn.commit()
            shadow = None
        except Exception:
            conn.rollback()
            if shadow is not None:
                # Cleanup has to happen after a failure too, or shadows pile up.
                try:
                    target_pg.drop_shadow_table(conn, shadow)
                    conn.commit()
                except Exception as cleanup_err:
                    ctx.log(
                        "error",
                        "⚠️ Shadow %s could not be cleaned up: %s",
                        shadow.table,
                        cleanup_err,
                    )
            raise

        self._finish(ctx, hash_column)
        return LoadResult(
            rows_read=rows_read,
            rows_written=rows_written,
            load_method=self.name,
            is_incremental=False,
            data_present=rows_written > 0,
            phase_metrics={
                "batch_size": batch_size,
                "estimated_rows": total_rows,
                "estimate_method": None if estimate is None else estimate.method,
            },
        )

    # -- steps -------------------------------------------------------------

    def _estimate(self, ctx: LoadContext):
        """Estimate the size of the source. **A failure raises loudly.**

        ``skip_size_estimate`` skips the estimate altogether (on huge tables the
        ``COUNT(*)`` takes a long time and it only serves batch auto-tuning).
        That is a different thing from an estimate that failed: it returns
        ``None``, never ``0``.

        The predecessors come in two variants and this is **exactly** where they
        differ: variant A lets the exception through, the forked copy swallows it
        and sets ``total_rows = 0``. A zero combined with ``empty_rows_ok``
        overwrites the target with an empty table and the run comes out green.
        Variant A wins.
        """
        if _is_truthy(ctx.settings.get("skip_size_estimate")):
            ctx.log("warning", "🔕 Size estimate skipped (skip_size_estimate).")
            return None

        estimate = ctx.dialect.estimate_size(ctx.engine, ctx.source, ctx.where)
        ctx.log(
            "warning",
            "📏 Rows: %s, estimated %.1f MB (%s).",
            f"{estimate.rows:,}",
            estimate.size_mb,
            estimate.method,
        )
        return estimate

    def _empty_target(self, ctx: LoadContext) -> LoadResult:
        """The source is empty. What happens to the target is up to ``empty_rows_ok``.

        Without it the target is **left alone** — overwriting it with an empty
        table because of an empty read is precisely the silent data loss this
        package refuses to perform.
        """
        empty_ok = _is_truthy(
            ctx.table_cfg.get("empty_rows_ok", ctx.table_cfg.get("EMPTY_ROWS_OK", False))
        )
        if not empty_ok:
            ctx.log(
                "warning",
                "⚠️ The source is empty and empty_rows_ok is not set — the target is left as is.",
            )
            return LoadResult(0, 0, self.name, data_present=False)

        ctx.log("warning", "ℹ️ The source is empty, empty_rows_ok — creating an empty target.")
        hash_column = _hash_column(ctx)
        self._ensure_managed_columns(ctx, hash_column)
        columns = _all_columns(ctx, hash_column)
        conn = ctx.target_conn
        types = column_types(ctx, hash_column)
        self._adopt_new_columns(ctx, columns, types)
        shadow = target_pg.create_shadow_table(
            conn,
            ctx.target,
            df=_empty_frame(columns),
            overwrite_types=types,
            columns=columns,
        )
        target_pg.swap_shadow_table(conn, shadow, ctx.target)
        target_pg.drop_shadow_table(conn, shadow)
        conn.commit()
        self._finish(ctx, hash_column)
        return LoadResult(0, 0, self.name, data_present=False)

    def _build_select(
        self,
        ctx: LoadContext,
        hash_column: Optional[str],
        *,
        compute_hash: bool = True,
        order_by: Optional[Sequence[str]] = None,
        resume_from: Optional[tuple] = None,
    ) -> str:
        """SELECT against the source, optionally carrying the ``row_hash`` expression.

        ``compute_hash=False`` means "the column exists in the target but stays
        NULL" — used by `full_by_source`, where the predecessor deliberately does
        not compute a hash. That is not the same as ``hash_column=None``, where
        the column is **absent** from the target.

        ``order_by`` and ``resume_from`` serve resuming after a dropped
        connection (`core.reading`). ``resume_from`` is a pair
        ``(source key, value)`` that adds ``pk > value`` to the condition; it
        only makes sense **together** with ``order_by`` over the same key,
        otherwise rows the server had not sent yet would be discarded.
        """
        where = ctx.where
        if resume_from is not None:
            pk_source, value = resume_from
            key_expr = ctx.dialect.render_key_expr(pk_source, ctx.surrogate)
            bound = f"{key_expr} > {ctx.dialect.sql_literal(value)}"
            where = f"({bound}) AND ({where})" if where else bound

        select_sql = ctx.dialect.render_select(
            ctx.source_names,
            ctx.source,
            where=where,
            order_by=order_by,
            surrogate=ctx.surrogate,
            # The single plumbing point for `convert_nchar_to_varchar`: every
            # strategy reads its data through `read_batches`, which comes back
            # here — `incremental`, `id_watermark`, `parent_incremental`,
            # `full_by_source` and the row fetch of `hash_diff` alike. The types
            # handed over are the introspected source types; **which** of them
            # get converted is the dialect's decision, and three of the four
            # dialects have no such types at all.
            column_types={c.name: c.source_type for c in ctx.columns},
            convert_nchar=_is_truthy(ctx.settings.get("convert_nchar_to_varchar")),
        )
        if hash_column is None or not compute_hash:
            return select_sql
        if ctx.dialect.hash_in_pandas:
            # The source cannot compute a hash that matches what is in the target
            # (PostgreSQL). The column is therefore missing from the batch and
            # `_prepare` fills it in with pandas — the same path the diff takes.
            return select_sql

        hashed = hashing.compute_hash_columns(
            ctx.source_names,
            ctx.settings.get("hash_column"),
            ctx.table_cfg,
            ctx.settings,
        )
        expr = ctx.dialect.render_hash_expr(
            hashed, hash_column, {c.name: c.source_type for c in ctx.columns}
        )
        ctx.log("warning", "🧮 The source computes the hash: %s", expr)
        return append_hash_expr(select_sql, expr)

    def read_batches(
        self,
        ctx: LoadContext,
        hash_column: Optional[str],
        batch_size: int,
        *,
        compute_hash: bool = True,
    ):
        """Batches from the source, resuming the remainder after a dropped connection.

        Shared by `full`, `incremental`, `id_watermark` and `parent_incremental`
        — each of them reads the source the same way, and the missing retry was
        common to all of them (`core.reading`). When no usable key exists the
        read proceeds as before and a dropped connection propagates; resuming
        without an ordering by key would silently discard rows.
        """
        from dbextractors.core import reading

        pk_source = _pk_in_source(ctx)
        order_by = [pk_source] if pk_source else None

        def build_select(after):
            return self._build_select(
                ctx,
                hash_column,
                compute_hash=compute_hash,
                order_by=order_by,
                resume_from=None if after is None else (pk_source, after),
            )

        return reading.read_with_resume(
            ctx.dialect,
            ctx.engine,
            batch_size,
            build_sql=build_select,
            pk_in_batch=pk_source,
            log=ctx.log,
        )

    def _prepare(
        self,
        ctx: LoadContext,
        batch: pd.DataFrame,
        hash_column: Optional[str],
        promoted: Optional[dict] = None,
        *,
        compute_hash: bool = True,
    ) -> pd.DataFrame:
        """Batch from the source -> batch ready for `COPY`.

        The order of the steps comes from the predecessor and is **not
        arbitrary**: rename, then hash and timestamp, and only then decode bytes.
        Decoding earlier would change the `row_hash` of every table with a binary
        column — measured, not assumed.
        """
        batch = batch.rename(columns=ctx.name_map)
        batch = apply_text_dtypes(batch, _text_dtype_map(ctx))
        # **Before** the hash, not after: a psycopg2 `memoryview` would otherwise
        # hash as a memory address. Decoding is not allowed here — the target is
        # BYTEA and binary data has to arrive there as binary data.
        batch = coerce.normalize_memoryviews(batch)
        batch = hashing.add_hash_and_timestamp(
            batch,
            hash_column or "row_hash",
            compute_hash=hash_column is not None and compute_hash,
            # `force_recompute` for dialects that do not compute the hash
            # themselves: a column of that name can also arrive from the source,
            # and leaving it be would store a foreign fingerprint in the target.
            # The PostgreSQL predecessor does the same — it passes
            # `force_recompute=True` on all three paths.
            force_recompute=ctx.dialect.hash_in_pandas,
            hashed_columns_override=_pandas_hash_columns(ctx, hash_column),
        )
        types = promoted or column_types(ctx, hash_column)
        binary_cols = _binary_columns(types)
        batch = coerce.decode_bytes_columns(batch, ctx.dialect.decode_encodings, skip=binary_cols)
        batch = coerce.bytes_to_pg_hex(batch, binary_cols)
        return target_pg.prepare_export_df(
            batch_df=batch,
            meta={"_deleted_in_source": False},
            all_columns=_all_columns(ctx, hash_column),
            date_like_columns=coerce.extract_date_columns(ctx.orig_type_map),
            integer_columns_mapping=_integer_columns(types),
            overwrite_types=types,
            orig_type_map=ctx.orig_type_map,
            # The single plumbing point for `strict_integer_precision`: every
            # strategy prepares its batches through this method — `incremental`,
            # `hash_diff`, `id_watermark`, `parent_incremental` and
            # `full_by_source` all call `full._prepare` — so the key needs no
            # wiring of its own anywhere else.
            strict_integer_precision=_is_truthy(ctx.settings.get("strict_integer_precision")),
        )

    def _ensure_managed_columns(self, ctx: LoadContext, hash_column: Optional[str]) -> None:
        """Add `row_hash`, `_timestamp` and `_deleted_in_source` **before** the shadow.

        The shadow is created with ``LIKE <target>``, so it inherits the shape of
        the target. When the target was created by an older extractor that lacks
        those columns, the shadow lacks them too and ``COPY`` fails on them::

            column "row_hash" of relation "dbx_shadow_<table>_…" does not exist

        Observed in production on two tables at once. This is not an edge case
        but the **default state of the migration**: one predecessor extractor has
        no `_deleted_in_source` at all, and five more only declare it without
        ever flipping it. The first run of this package over such a table would
        therefore always fail.

        It has to happen **before** the shadow, not in `_finish`: the swap only
        carries columns the target already has, so a column added afterwards
        would stay empty and the computed values would be thrown away.
        """
        target_pg.ensure_generated_columns(
            ctx.target_conn,
            ctx.target,
            include_row_hash=hash_column is not None,
            hash_column=hash_column or "row_hash",
        )
        ctx.target_conn.commit()

    def _adopt_new_columns(
        self, ctx: LoadContext, columns: Sequence[str], overwrite_types: dict
    ) -> None:
        """Adopt columns the source has newly started sending. Full load only.

        The source gains a column — and the target does not have it, because an
        earlier run created the target. Observed in production on an order-lines
        table::

            column "promo_code_code" of relation "dbx_shadow_<table>_…"
            does not exist

        The predecessor handled this by reindexing the batch onto the target's
        columns, i.e. it **silently dropped** the new column. A full load
        rewrites the whole table anyway, so here it is adopted instead;
        strategies that touch existing data still drop it, but say so loudly.

        It has to happen **before the shadow**: the shadow is created with
        ``LIKE <target>``, so a column added to the target beforehand is
        inherited and carried over by the swap. Added afterwards, it would stay
        empty in the target.
        """
        added = target_pg.adopt_new_source_columns(
            ctx.target_conn, ctx.target, columns, overwrite_types=overwrite_types
        )
        if added:
            ctx.log(
                "warning",
                "📝 The target was missing %d columns the source sends — adopting them: %s",
                len(added),
                ", ".join(added),
            )
            ctx.target_conn.commit()

    def _finish(self, ctx: LoadContext, hash_column: Optional[str]) -> None:
        """Generated columns and the unique index. Every strategy owes the target both."""
        conn = ctx.target_conn
        target_pg.ensure_generated_columns(
            conn,
            ctx.target,
            include_row_hash=hash_column is not None,
            hash_column=hash_column or "row_hash",
        )
        conn.commit()

        pk = _resolved_pk(ctx)
        if pk:
            # Over a partitioned target the unique index must contain the
            # partition key (PostgreSQL requires it), and a plain index over the
            # key itself belongs with it — `replace_by_key` leans on that one.
            spec = ctx.partition_spec(conn)
            index_columns = partitioning.unique_index_columns(spec, pk)
            target_pg.create_unique_pk_index(
                conn, ctx.target, pk, ctx.target_names, fatal=False, columns=index_columns
            )
            if spec is not None:
                target_pg.ensure_key_index(conn, ctx.target, pk)
                conn.commit()


# --- Helper functions -------------------------------------------------------


def append_hash_expr(select_sql: str, expr: str) -> str:
    """Splice the hash expression after the column list: ``SELECT <cols>, <hash> FROM …``.

    Exactly as the predecessor does it. Shared with `strategies.hash_diff`, which
    builds a SELECT over the primary key alone but splices the hash in the same
    way.
    """
    head, sep, tail = select_sql.partition(" FROM ")
    if not sep:
        raise StrategyError(
            "The dialect returned a SELECT without FROM; there is nowhere to splice the "
            "hash expression."
        )
    return f"{head}, {expr}{sep}{tail}"


def _is_truthy(value: object) -> bool:
    return coerce.to_bool(value) is True


def _pk_in_source(ctx: LoadContext) -> Optional[str]:
    """The primary key under its **source** name, or ``None`` when it cannot be used.

    A query against the source needs the source name, not the target one: in the
    target, reserved words carry an underscore prefix, and that prefix does not
    exist in the source.

    A surrogate key is deliberately excluded here — it only comes into being as
    an expression in the SELECT, so it cannot be used to order by or to resume
    from reliably.
    """
    if ctx.surrogate:
        return None
    raw = ctx.settings.get("primary_column") or ctx.settings.get("primary_key_column")
    if not raw:
        return None
    return str(raw) if str(raw) in ctx.source_names else None


def _resolved_pk(ctx: LoadContext) -> Optional[str]:
    """The primary key in **target** naming.

    The configuration gives it under its source name, and that name may be a
    reserved word (`id` is not, `order` is) — without the translation the index
    would be created over a column that does not exist.
    """
    raw = ctx.settings.get("primary_column") or ctx.settings.get("primary_key_column")
    if not raw:
        return None
    return ctx.name_map.get(raw, raw)


def _hash_column(ctx: LoadContext) -> Optional[str]:
    """Name of the hash column in the target, or ``None`` when no hash is computed.

    ``compute_row_hash=False`` (used by a single forked predecessor) disables the
    hash entirely. ``None`` means the configuration is silent — here that means
    ``True``, i.e. the behaviour of the other twelve predecessor files.
    `full_by_source` defaults the other way round.
    """
    flag = ctx.settings.get("compute_row_hash")
    if flag is not None and not _is_truthy(flag):
        return None
    configured = ctx.settings.get("hash_column")
    if not configured:
        from dbextractors.core.naming import DEFAULT_HASH_COLUMN

        return DEFAULT_HASH_COLUMN
    return ctx.name_map.get(configured, configured)


def column_types(ctx: LoadContext, hash_column: Optional[str]) -> dict:
    """Types of every target column — those from the source and those the package adds.

    ``ctx.overwrite_types`` only knows the source columns. Without the additions
    `target_pg.build_create_table` would have nothing to derive the types of
    `row_hash` and `_timestamp` from (the batch is empty at that point) and would
    fail. The types of the managed columns are not a choice — they are read off
    the target, see `target_pg.MANAGED_COLUMNS`.
    """
    types = dict(ctx.overwrite_types)
    if hash_column:
        types.setdefault(hash_column, target_pg.ROW_HASH_TYPE)
    for name, ddl in target_pg.MANAGED_COLUMNS.items():
        types.setdefault(name, ddl)
    return types


def _all_columns(ctx: LoadContext, hash_column: Optional[str]) -> List[str]:
    """Target table columns in the right order. The order is part of the contract.

    When the target already exists, **its** order is adopted — this package never
    reorders an existing table. See `naming.reconcile_with_existing`.
    """
    from dbextractors.core import naming

    wanted = naming.target_column_names(ctx.source_names, hash_column)
    return naming.reconcile_with_existing(
        wanted, target_pg.existing_column_order(ctx.target_conn, ctx.target)
    )


def _empty_frame(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _integer_columns(types: dict) -> dict:
    integer_types = {"INTEGER", "BIGINT", "SMALLINT"}
    return {
        name: pg_type for name, pg_type in types.items() if (pg_type or "").upper() in integer_types
    }


#: Above this value ``INTEGER`` is promoted to ``BIGINT``.
#: Taken from variant A's extractor (90 % of INT_MAX).
INT_PROMOTION_THRESHOLD = int(2_147_483_647 * 0.9)


def _promote_types(ctx: LoadContext, batch: pd.DataFrame, hash_column: Optional[str]) -> dict:
    """Promote types from the **first batch**, the way the predecessor does.

    Two promotions, both driven by actual data rather than by the catalogue:

    - ``INTEGER`` -> ``BIGINT`` when the batch contains a value above the
      threshold. Without it the write fails on the millionth row.
    - an ``enum`` holding only 0/1 -> ``BOOLEAN``. The catalogue reports it as
      ``enum`` and the type map yields ``TEXT``; that these are booleans can only
      be told from the data.

    Inferring from a single batch is a weakness the predecessor shares — a column
    whose large numbers only arrive later stays ``INTEGER``. It is not adopted
    blindly: `target_pg.load_pg_schema` overrides the type back upwards from the
    existing target, so after the first run it no longer shrinks.
    """
    types = column_types(ctx, hash_column)
    renamed = batch.rename(columns=ctx.name_map)

    seen: set = set()
    stats: dict = {}
    for column in renamed.columns:
        coerce.update_stats(column, renamed[column], seen, stats)

    orig_types = ctx.orig_type_map
    for column, column_stats in stats.items():
        current = (types.get(column) or "").upper()

        if current == "INTEGER":
            largest = column_stats.get("int_max")
            if largest is not None and largest > INT_PROMOTION_THRESHOLD:
                types[column] = "BIGINT"
                ctx.log("warning", "📋 Column %r promoted to BIGINT (max %s).", column, largest)
            continue

        if (
            current == "TEXT"
            and (orig_types.get(column) or "").lower() == "enum"
            and column_stats.get("seen_integer")
            and column_stats.get("int_min") == 0
            and column_stats.get("int_max") == 1
            and not column_stats.get("seen_float")
            and not column_stats.get("has_non_numeric")
        ):
            types[column] = "BOOLEAN"
            ctx.log("warning", "📋 Column %r is enum(0,1), promoted to BOOLEAN.", column)

    return types


def _binary_columns(types: dict) -> List[str]:
    """Columns whose target type is ``BYTEA``. These must not be decoded."""
    return [name for name, pg_type in types.items() if (pg_type or "").upper().startswith("BYTEA")]


def _pandas_hash_columns(ctx: LoadContext, hash_column: Optional[str]) -> Optional[List[str]]:
    """Which columns feed the hash when pandas computes it.

    Without this, a dialect with `hash_in_pandas` would ignore
    ``hash_include_columns``/``hash_exclude_columns`` and hash the whole batch.
    The predecessor solves it with the same ``hashed_columns_override``.

    Returns ``None`` when the source computes the hash: nothing is computed in
    pandas there and an override would only be confusing.
    """
    if hash_column is None or not ctx.dialect.hash_in_pandas:
        return None
    hashed = hashing.compute_hash_columns(
        ctx.source_names, ctx.settings.get("hash_column"), ctx.table_cfg, ctx.settings
    )
    return [ctx.name_map.get(c, c) for c in hashed]


def _text_dtype_map(ctx: LoadContext) -> dict:
    """Columns that must be read as text.

    Stops an eighteen-digit EAN in a ``varchar`` column from being read as
    ``float64`` and losing precision. Which types those are is the dialect's
    knowledge.
    """
    text_types: frozenset = getattr(ctx.dialect, "text_like_types", frozenset())
    return {c.name: str for c in ctx.columns if (c.source_type or "").lower() in text_types}


def _keyset_usable(ctx: LoadContext) -> bool:
    from dbextractors.dialects.base import FEATURE_KEYSET

    return bool(_resolved_pk(ctx)) and ctx.dialect.supports(FEATURE_KEYSET) and not ctx.surrogate


__all__ = ["FullLoadStrategy", "append_hash_expr", "column_types"]
