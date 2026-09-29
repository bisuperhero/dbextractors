"""``AbraWatermarkStrategy`` — new rows of an ABRA ERP table, by the counter in its ID.

Meant primarily for **ABRA ERP** sources. ABRA tables often have no column that
records when a row was created or changed, so `incremental` has nothing to build
its window from, and a large table then has only `full` left: every run reads
the whole table, although only a small part of it is new.

What ABRA does have is its record ID. It is a fixed-width string: a base-36
counter written **least significant digit first**, followed by the identifier
of the database the record was created in (``101`` in a typical installation,
``000`` for records ABRA ships itself)::

    GMXCO00 101
    ^^^^^^^ ^^^
    counter database

The counter grows with every new record, but the ID as a string does not —
``Z000000101`` is older than ``0100000101`` — so `id_watermark`'s
``pk > MAX(pk)`` cannot be used. Reversing the counter part gives a string whose
byte order is the counter's order. That is what this strategy compares, per
database identifier, on both sides:

- **the watermark** is ``MAX(<reversed counter>)`` in the target, one per
  configured suffix, with no state stored anywhere (as in `id_watermark`),
- **the read** takes the rows whose reversed counter is above that watermark.

The expressions are rendered by the dialect (`SourceDialect.render_abra_counter`
and `render_abra_suffix`), and each dialect pins a byte comparison: under a
linguistic collation the counter would not sort the way it grows.

## Configuration

::

    'LOAD_SETTINGS': {
        'load_method': 'abra_watermark',
        'primary_column': 'ID',
        'abra_id_suffixes': ['101'],
    }

``abra_id_suffixes`` is required: which database identifiers the installation
creates records under is a property of that installation, and a guessed default
would quietly skip the records of every other one. Rows with any other suffix
(``000`` among them) are never read by this strategy — only a full load brings
them over.

## What it does not see

The same as `id_watermark`, and for the same reason: it reads only what was
added past the watermark.

1. **Changes to existing rows** and **deleted rows**. ``_deleted_in_source``
   stays ``FALSE``.
2. **A record whose counter is below the watermark** — should the counter ever
   not grow with time, the row is never picked up.

The monotonic counter is an observation about how ABRA assigns IDs, not
something ABRA documents. A table loaded this way therefore **needs a periodic
full load** (``forced_full_load``) to catch up on changes, deletions and
anything the counter did not order.

## Cost on the source

The counter expression cannot use an index, so the source scans the table.
What it saves is the transfer and the write: only the new rows travel and only
they are upserted.

**There is no size estimate**, unlike every other incremental strategy. Here a
``COUNT(*)`` over the condition would be a second scan of the whole table, and
it would buy nothing: when nothing is new, the read finds that out for the same
single scan; the batch cap it feeds only ever shrinks a batch, and the new rows
are few; and the row count it would log in advance is ``rows_read`` afterwards.
The batch size is the configured one.

The watermark itself is a scan too, on the target: ``MAX`` over the reversed
counter cannot use the primary-key index the way `id_watermark`'s ``MAX(pk)``
does. It reads one narrow column of a local PostgreSQL table, which is small
next to reading the source; no expression index is created for it, so that the
strategy adds no DDL of its own to the target.

Writes go through a staging table and an upsert on the primary key, so a row at
the watermark read twice (a rerun, a restart after a failure) is neither
duplicated nor a unique-index violation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from dbextractors.core import target_pg
from dbextractors.core.strategies.base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    StrategyError,
    fallback_full,
    resolve_batch_size,
)
from dbextractors.core.strategies.full import _resolved_pk
from dbextractors.core.strategies.id_watermark import IdWatermarkStrategy, _source_name
from dbextractors.core.strategies.incremental import _target_is_empty
from dbextractors.dialects.base import FEATURE_ABRA_WATERMARK
from dbextractors.dialects.postgres import abra_counter_sql, abra_suffix_sql

_log = logging.getLogger(__name__)

#: The LOAD_SETTINGS key naming the database identifiers to watermark.
SUFFIXES_KEY = "abra_id_suffixes"


class AbraWatermarkStrategy(IdWatermarkStrategy):
    """Transfer only the ABRA records whose ID counter is above the watermark.

    The read and the write are `IdWatermarkStrategy`'s; what differs is the
    watermark and the condition built from it.
    """

    name = "abra_watermark"
    required_features: frozenset = frozenset({FEATURE_ABRA_WATERMARK})

    def validate(self, ctx: LoadContext) -> None:
        """The key, the suffixes and the dialect, all before the source is touched."""
        LoadStrategy.validate(self, ctx)

        pk = _resolved_pk(ctx)
        if not pk:
            raise StrategyError(
                "abra_watermark: primary_column (or primary_key_column) is missing from "
                "LOAD_SETTINGS. The watermark is read from the ABRA record ID."
            )
        if ctx.surrogate is not None and ctx.surrogate.enabled and ctx.surrogate.alias == pk:
            raise StrategyError(
                "abra_watermark: the primary key is a surrogate key. The watermark needs "
                "the ABRA record ID itself."
            )
        if pk not in ctx.target_names:
            raise StrategyError(
                f"abra_watermark: primary key {pk!r} is not among the table's columns."
            )
        abra_id_suffixes(ctx.settings)

    def run(self, ctx: LoadContext) -> LoadResult:
        self.validate(ctx)

        pk = _resolved_pk(ctx)
        assert pk is not None  # guaranteed by validate()
        suffixes = abra_id_suffixes(ctx.settings)
        conn = ctx.target_conn

        if not target_pg.table_exists(conn, ctx.target):
            return fallback_full(ctx, "the target table does not exist")
        if _target_is_empty(ctx):
            return fallback_full(ctx, "the target table is empty, there is no watermark to compute")

        watermarks = _watermarks(conn, ctx.target, pk, suffixes)
        for suffix in suffixes:
            if watermarks[suffix] is None:
                _log.warning(
                    "⚠️ ABRA watermark: the target has no ID with suffix %r — "
                    "reading all of that suffix's rows.",
                    suffix,
                )
            else:
                _log.info(
                    "🚀 ABRA watermark: suffix %r, taking counters above %r",
                    suffix,
                    watermarks[suffix],
                )
        target_pg.assert_unique_pk_index(conn, ctx.target, pk)

        where = _abra_where(ctx, _source_name(ctx, pk), watermarks)
        # No estimate: it would be a second full scan of the source, see the
        # module docstring. `None` leaves the configured batch size as it is.
        batch_size = resolve_batch_size(ctx.settings, ctx.table_cfg, None)
        result = self._load_new_rows(ctx, where, batch_size, pk)
        if result.rows_read == 0:
            _log.info("ℹ️ No new rows above the watermark.")
        else:
            _log.info("🔢 New rows: %s", f"{result.rows_read:,}")
        result.phase_metrics.update(
            {"watermark": _describe(watermarks), "abra_id_suffixes": ",".join(suffixes)}
        )
        return result


# --- Helper functions -------------------------------------------------------


def abra_id_suffixes(settings: dict) -> List[str]:
    """``abra_id_suffixes`` from LOAD_SETTINGS as a list of distinct strings.

    A single string is accepted for one suffix. A number is **refused** rather
    than converted: a suffix such as ``001`` written unquoted arrives as ``1``,
    and a watermark over suffix ``'1'`` matches nothing — a run that is green
    and transfers nothing for ever.
    """
    raw = settings.get(SUFFIXES_KEY)
    values = [raw] if isinstance(raw, str) else list(raw or ())
    if not values:
        raise StrategyError(
            f"abra_watermark: {SUFFIXES_KEY} is missing from LOAD_SETTINGS. Name the ABRA "
            "database identifiers whose records are loaded, e.g. ['101']."
        )
    out: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise StrategyError(
                f"abra_watermark: {SUFFIXES_KEY} must hold non-empty strings, got {value!r}. "
                "Write the suffix in quotes, so that leading zeros survive."
            )
        if value not in out:
            out.append(value)
    return out


def _watermarks(conn: Any, target: Any, pk: str, suffixes: List[str]) -> Dict[str, str | None]:
    """The highest reversed counter in the target, per suffix. ``None`` where there is none.

    An error is allowed to propagate, as in `id_watermark._watermark`: a target
    that cannot answer must not look like an empty one.
    """
    key = target_pg.quote_ident(pk)
    suffix = abra_suffix_sql(key)
    sql = (
        f"SELECT {suffix}, MAX({abra_counter_sql(key)}) "
        f"FROM {target_pg.qualify(target)} WHERE {suffix} = ANY(%s) GROUP BY 1"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (list(suffixes),))
        found = {row[0]: row[1] for row in cur.fetchall()}
    return {s: found.get(s) for s in suffixes}


def _abra_where(ctx: LoadContext, pk_source: str, watermarks: Dict[str, str | None]) -> str:
    """One branch per suffix, ``OR``-ed, then glued to the condition from the configuration.

    Values go in as literals via `dialect.sql_literal`, for the same reason as
    in `id_watermark`: ``iter_batches`` knows nothing about bound parameters.
    """
    dialect = ctx.dialect
    key = dialect.quote_ident(pk_source)
    counter = dialect.render_abra_counter(key)
    suffix = dialect.render_abra_suffix(key)

    branches = []
    for value, watermark in watermarks.items():
        branch = f"{suffix} = {dialect.sql_literal(value)}"
        if watermark is not None:
            branch = f"({branch} AND {counter} > {dialect.sql_literal(watermark)})"
        branches.append(branch)
    clause = " OR ".join(branches)
    if ctx.where:
        clause = f"({clause}) AND ({ctx.where})"
    return clause


def _describe(watermarks: Dict[str, str | None]) -> str:
    """``101:00OCXMG, 102:-`` — the watermarks for the status, one string like `id_watermark`'s."""
    return ", ".join(f"{s}:{w if w is not None else '-'}" for s, w in watermarks.items())


__all__ = ["AbraWatermarkStrategy", "abra_id_suffixes"]
