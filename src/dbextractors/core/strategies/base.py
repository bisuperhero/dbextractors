"""``LoadStrategy`` — how a table gets from the source to the target.

Why a context object rather than parameters: in the predecessor extractors
``load_full_data`` exists in **7 signatures with 16 to 20 positional
parameters** (``engine, table_name, column_names, …, overwrite_types,
orig_type_map, date_like_columns, integer_columns_mapping,
all_columns_ordered, surrogate``). That is no longer a parameter list, it is a
context threaded through by hand — and it is exactly why the signature drifted
into seven variants.

Division of responsibility:

- **a strategy knows nothing about the SQL dialect** — it asks the dialect
  abstractly
- a strategy drives the loop: what to read, in what batches, what to write
- it reads through ``ctx.dialect`` and writes through ``core.target_pg``

In the predecessors this whole layer is a single ``load_table_from_engine``
function repeated inside each of the 15 files — 10 variants, 116 lines.
Splitting it into `LoadStrategy` is what turns it into shared code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence

import pandas as pd

from dbextractors.dialects.base import ColumnDef, SourceDialect, SurrogateKey, TableRef

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine

_log = logging.getLogger(__name__)

#: Default batch size when the configuration does not give one.
#: Taken from variant A's extractor.
DEFAULT_BATCH_SIZE = 10_000

#: Target volume of a single batch. When the estimate says more, the batch shrinks.
#: Taken from variant A's extractor.
TARGET_BATCH_MB = 400

#: Smallest batch the auto-tuning may shrink to.
MIN_BATCH_SIZE = 1_000


@dataclass(frozen=True)
class TargetRef:
    """Address of the target table in PostgreSQL.

    Corresponds to the ``schema_name`` + ``output_table_name`` pair that all 15
    predecessor files carry around separately.
    """

    schema: str
    table: str

    def qualified(self) -> str:
        from dbextractors.core import target_pg

        return target_pg.qualify(self)


@dataclass
class LoadContext:
    """Everything a strategy needs in order to move one table.

    ``dbextractors.entrypoint.run`` assembles it from the configuration. It is a
    value object with no behaviour, so a test can build one without a live
    database.

    ``target_conn`` is part of it because without a connection to the target a
    strategy could neither create a shadow table nor run `COPY`.
    """

    dialect: SourceDialect
    engine: Engine
    source: TableRef
    target: TargetRef
    #: Connection to the target PostgreSQL. Autocommit must be **off** —
    #: `target_pg.swap_shadow_table` insists on it.
    target_conn: Any = None
    #: Source columns after introspection, including `pg_type` and `target_name`.
    columns: List[ColumnDef] = field(default_factory=list)
    #: The LOAD_SETTINGS section of the configuration. The contract is frozen.
    settings: dict = field(default_factory=dict)
    #: The TABLE section of the configuration.
    table_cfg: dict = field(default_factory=dict)
    where: Optional[str] = None
    surrogate: Optional[SurrogateKey] = None
    logger: Optional[logging.Logger] = None
    #: Which source database this run reads from. ``None`` for single-source
    #: tables — and it is **deliberately independent** of how many databases the
    #: run happens to have selected: `full_by_source` deletes by `_source`, so a
    #: run that happened to pull a single database would otherwise wipe the whole
    #: table.
    source_label: Optional[str] = None
    #: Verbose logging — the top-level ``DEBUG`` key of the configuration. It
    #: governs only what gets printed, never what gets done.
    debug: bool = False
    #: Governs what happens once per table rather than once per source.
    is_first_source: bool = True
    is_last_source: bool = True
    #: The FINGERPRINT section of the configuration. Empty = fingerprinting off.
    fingerprint_cfg: dict = field(default_factory=dict)
    #: Pipeline runtime variables out of Mage kwargs — only those the package
    #: knows (`entrypoint.RUNTIME_KEYS`). This is not configuration: it varies
    #: between runs of the same pipeline, whereas `settings` is the same every
    #: time. A key that is **absent** means "the variable was not passed"; a key
    #: holding ``None`` means "it was passed empty", and to
    #: `compute_incremental_cutoff` those two are different.
    runtime: dict = field(default_factory=dict)

    # -- partitioning -------------------------------------------------------

    def partition_spec(self, conn: Any = None) -> Any:
        """The partitioning that will actually apply to the target, or ``None``.

        Without ``conn`` it reports only what the configuration says. With one it
        decides on **what is really in the target**: an existing plain table is
        never quietly rebuilt as a partitioned one, it is only logged
        (`partitioning.effective_spec`).
        """
        from dbextractors.core import partitioning

        spec = partitioning.from_settings(self.settings)
        if conn is None or spec is None:
            return spec
        return partitioning.effective_spec(conn, self.target, spec, self.log)

    # -- convenient views over the columns ----------------------------------

    @property
    def source_names(self) -> List[str]:
        """Column names as the source has them."""
        return [c.name for c in self.columns]

    @property
    def target_names(self) -> List[str]:
        """Column names in the target. `core.naming` fills these in, not the dialect."""
        return [c.target_name or c.name for c in self.columns]

    @property
    def name_map(self) -> Dict[str, str]:
        """Source name -> target name. For ``df.rename``."""
        return {c.name: (c.target_name or c.name) for c in self.columns}

    @property
    def overwrite_types(self) -> Dict[str, str]:
        """Target name -> PostgreSQL type as the dialect determined it."""
        return {(c.target_name or c.name): c.pg_type for c in self.columns}

    @property
    def orig_type_map(self) -> Dict[str, str]:
        """Target name -> raw source type. `coerce.convert_time_columns` needs it."""
        return {(c.target_name or c.name): c.source_type for c in self.columns}

    def log(self, level: str, message: str, *args: Any) -> None:
        """Logging that does not blow up when there is no logger.

        Mage passes its own logger in ``kwargs``; outside Mage (tests, the golden
        harness) there is none. The predecessors call ``logger.warning``
        unconditionally and fail with ``AttributeError`` outside Mage.
        """
        target = self.logger or _log
        getattr(target, level, target.info)(message, *args)


@dataclass
class LoadResult:
    """What a strategy reports back about its run.

    The status DataFrame returned to Mage is assembled from this.
    """

    rows_read: int
    rows_written: int
    load_method: str = ""
    is_incremental: bool = False
    data_present: bool = True
    #: Timings and counts per phase — input for `core.status`.
    phase_metrics: dict = field(default_factory=dict)
    #: Non-empty when the strategy took the emergency path (`fallback_full`).
    fallback_reason: Optional[str] = None


class LoadStrategy(ABC):
    """One way of getting a table from the source into the target."""

    #: 'full' | 'hash_diff' | 'incremental' | 'id_watermark' |
    #: 'full_by_source' | 'parent_incremental'
    name: str
    #: Dialect features without which the strategy makes no sense.
    #: Checked in `validate()`, i.e. before the first read from the source.
    required_features: frozenset = frozenset()

    def validate(self, ctx: LoadContext) -> None:
        """Check that the configuration and the dialect suffice for this strategy.

        It runs **before** the source is first touched, so that a mismatch shows
        up straight away rather than half-way through a run. The default
        implementation checks the dialect features and that the context is usable
        at all; subclasses add their own checks.
        """
        for feature in self.required_features:
            ctx.dialect.require(feature)

        if not ctx.columns:
            raise StrategyError(
                f"{self.name}: the context has no columns. Source introspection must "
                "happen before the strategy starts."
            )
        if ctx.target_conn is None:
            raise StrategyError(f"{self.name}: the context has no target connection (target_conn).")
        if getattr(ctx.target_conn, "autocommit", False):
            raise StrategyError(
                f"{self.name}: the target connection has autocommit on. The atomic "
                "swap of a full load would then not be atomic."
            )

        self._warn_about_inapplicable_nchar_conversion(ctx)

    @staticmethod
    def _warn_about_inapplicable_nchar_conversion(ctx: LoadContext) -> None:
        """Say out loud that ``convert_nchar_to_varchar`` has nothing to do here.

        The key repairs MSSQL's national-character columns, and MSSQL is the only
        dialect that has any (`SourceDialect.NCHAR_TYPES`). Set anywhere else it
        would change nothing — and a key that silently does nothing is exactly
        the trap this package documents rather than tolerates. It warns rather
        than raises: a configuration shared between a MySQL and an MSSQL source
        is a real shape, and refusing to start would be worse than saying so.
        """
        from dbextractors.core.coerce import to_bool

        if to_bool(ctx.settings.get("convert_nchar_to_varchar")) is not True:
            return
        if ctx.dialect.NCHAR_TYPES:
            return
        ctx.log(
            "warning",
            "ℹ️ convert_nchar_to_varchar is set, but %s has no national-character types "
            "— it changes nothing here.",
            ctx.dialect.name,
        )

    @abstractmethod
    def run(self, ctx: LoadContext) -> LoadResult:
        """Move the table. Drives both the read and the write loop."""


class StrategyError(RuntimeError):
    """The configuration or the state of the target does not suffice for this strategy."""


# --- Shared helpers -------------------------------------------------------


def chunked_list(items: Sequence, size: int) -> Iterator[list]:
    """Cut a list into chunks of the given size. Only one variant in the predecessors."""
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def apply_text_dtypes(df: pd.DataFrame, dtype_map: Optional[dict]) -> pd.DataFrame:
    """Apply text dtypes to a batch.

    It exists because pandas 1.5 has no ``dtype`` parameter on ``read_sql``. It
    stops an eighteen-digit EAN in a ``varchar`` column from being read as
    ``float64`` and losing precision. Only one variant across the predecessors.
    """
    if not dtype_map:
        return df
    for col, dtype in dtype_map.items():
        if col not in df.columns or not (dtype is str or dtype == "str"):
            continue
        df[col] = df[col].map(_to_text_or_keep)
    return df


def _to_text_or_keep(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return value
    return str(value)


def resolve_batch_size(settings: dict, config: dict, estimate: Optional[Any]) -> int:
    """Batch size from the configuration, capped by the size estimate.

    The predecessor **replaces** the configured value with whatever fits into
    `TARGET_BATCH_MB` — which means it also enlarges it when circumstances allow.
    On a table estimated at 418 MB that turns 100,000 into 764,635 rows.

    **Here it is a cap, not a replacement** (``min``), so a batch may only
    shrink. This is not trading time for memory: measured over 800,000 rows and
    12 columns, a large batch buys **nothing**:

    | TARGET_BATCH_MB | batch | time | peak RSS |
    |---:|---:|---:|---:|
    | 400 (predecessor) | 764,635 | 72.0 s | 2,151 MB |
    | 150 | 286,738 | 68.0 s | 1,319 MB |
    | 50 | 95,579 | **65.9 s** | **759 MB** |

    The smaller batch is 2.8x leaner and 9 % faster on top. Shrinking stays
    (tables with 12 kB per row exist, where 100,000 rows is 1.2 GB), enlarging
    goes. The orchestrator runs 3 pipelines concurrently, so the difference is
    three times larger at the peak.

    ``BATCH_SIZE`` at the top level of the configuration is a legacy fallback —
    see ``docs/legacy-compat.md``.
    """
    batch_size = settings.get("batch_size") or config.get("BATCH_SIZE") or DEFAULT_BATCH_SIZE
    batch_size = int(batch_size)

    if estimate is None or not getattr(estimate, "rows", 0):
        return batch_size
    if estimate.size_mb <= TARGET_BATCH_MB:
        return batch_size

    scaled = int(estimate.rows / (estimate.size_mb / TARGET_BATCH_MB))
    return max(MIN_BATCH_SIZE, min(batch_size, scaled))


def fallback_full(ctx: LoadContext, reason: str) -> LoadResult:
    """Emergency switch to a full load when the incremental path cannot run.

    The predecessor's `_fallback_full` — used when the target table does not
    exist or is empty. There it is a closure that calls ``load_full_data`` with
    twenty parameters; here the context suffices.

    **It is always logged as a warning.** A silent switch to a full load is
    expensive, and the log has to show why it happened.
    """
    from dbextractors.core.strategies.full import FullLoadStrategy

    ctx.log("warning", "⚠️ Falling back to a full load: %s", reason)
    result = FullLoadStrategy().run(ctx)
    result.fallback_reason = reason
    return result


_STRATEGY_PACKAGE = "dbextractors.core.strategies"

#: Map of ``load_method`` -> strategy. The keys are part of the frozen
#: configuration contract and must not be renamed.
#:
#: The value is ``module:Class`` inside `_STRATEGY_PACKAGE`. The path is
#: assembled so the package name is written in exactly one place — spelled out
#: in full, the paths no longer fit on a line.
STRATEGIES: Dict[str, str] = {
    key: f"{_STRATEGY_PACKAGE}.{path}"
    for key, path in {
        "full": "full:FullLoadStrategy",
        "incremental": "incremental:IncrementalStrategy",
        "hash": "hash_diff:HashDiffStrategy",
        "hash_diff": "hash_diff:HashDiffStrategy",
        "id_watermark": "id_watermark:IdWatermarkStrategy",
        "full_by_source": "full_by_source:FullBySourceStrategy",
        "parent_incremental": "parent_incremental:ParentIncrementalStrategy",
    }.items()
}


def resume_is_watermark(load_method: str, settings: Optional[dict]) -> bool:
    """``True`` when ``full`` + ``resume_full_load`` should be served by a watermark.

    ``resume_full_load`` is not a strategy of its own, even though the
    configuration makes it look like one. Underneath it, the predecessor does
    this::

        SELECT MAX("id") FROM schema.table;
        ▶️ RESUMING full load: appending rows with `id` > {resume_last_id}
           (no DROP, no replace).

    Which is **literally** `IdWatermarkStrategy`: read rows whose key is above
    ``MAX(pk)`` in the target, keyset paging, no stored state. Even the edge
    cases agree — a missing target and an empty target both lead to a full load.

    Without this rule ``resume_full_load`` got lost on the way: the config layer
    read it, `FullLoadStrategy.validate` checked that keyset paging was usable,
    and then the whole table was rebuilt anyway. On one production table that is
    the difference between fetching 823 thousand rows and reloading 9.9 million.

    The absence of a ``primary_column`` condition here is **deliberate**. Whether
    the key is usable is checked by `IdWatermarkStrategy.validate`, and if it is
    not, that fails loudly. Deciding it twice in two places means the two places
    will drift apart sooner or later.
    """
    if load_method != "full" or not settings:
        return False
    # `force_reload` means "trust nothing the target remembers". A watermark is
    # exactly what must not be trusted — it takes `MAX(pk)` from the target — so
    # forcing a reload would otherwise quietly turn back into fetching a handful
    # of rows. The forced reload wins.
    if settings.get("force_reload"):
        return False
    value = settings.get("resume_full_load")
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "ano"}
    return bool(value)


def parent_incremental_applies(load_method: str, settings: Optional[dict]) -> bool:
    """``True`` when ``incremental`` should be served through a parent table.

    Variant B has a family of child tables — document lines, order lines and the
    like — that carry no change timestamp of their own, so the window is computed
    over the parent instead. The configuration does not show this in
    `load_method`, which is still `incremental`; it is recognised **only by the
    presence** of ``incremental_parent_table``, exactly as the predecessor does.

    Without this rule the child tables would go to `IncrementalStrategy`, which
    would look for a change-timestamp column of their own, fail to find one and
    raise. It affects 11 of that deployment's 102 configurations.
    """
    if load_method != "incremental" or not settings:
        return False
    return bool(settings.get("incremental_parent_table"))


def surrogate_provides(ctx: LoadContext, pk: Optional[str]) -> bool:
    """``True`` when that primary key is produced by a surrogate expression.

    A surrogate key **is not and cannot be** in the source — it is assembled by
    the expression in ``surrogate_key_definition`` and only joins the batch in
    the SELECT. Every strategy checks that the key is among the columns, and
    without this exception a pipeline with a surrogate key would always be
    rejected.

    One production pipeline is set up that way: a junction table with no ``id``
    of its own and a ``_uni_surrogate_key`` composed of three columns. The check
    still applies to every other key — a typo in ``primary_column`` must fail.
    """
    surrogate = ctx.surrogate
    return bool(pk and surrogate and surrogate.enabled and surrogate.alias == pk)


def resolve_strategy(load_method: str, settings: Optional[dict] = None) -> LoadStrategy:
    """Pick the strategy from ``load_method`` in LOAD_SETTINGS.

    In the predecessors this decision is scattered through
    ``load_table_from_engine`` as an ``if/elif`` chain in each of the 15 files.
    ``'hash'`` and ``'hash_diff'`` mean the same thing — both spellings occur in
    live configurations.

    An unknown value **raises**. The predecessor quietly falls back to a full
    load, which is the most expensive possible outcome of a typo in YAML.

    ``settings`` decide where ``load_method`` alone is not enough — see
    `resume_is_watermark`.
    """
    key = (load_method or "").strip().lower()
    if resume_is_watermark(key, settings):
        key = "id_watermark"
    elif parent_incremental_applies(key, settings):
        key = "parent_incremental"
    if key not in STRATEGIES:
        raise StrategyError(
            f"Unknown load_method {load_method!r}. Known: " f"{', '.join(sorted(set(STRATEGIES)))}."
        )

    module_name, _, class_name = STRATEGIES[key].partition(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "MIN_BATCH_SIZE",
    "STRATEGIES",
    "TARGET_BATCH_MB",
    "LoadContext",
    "LoadResult",
    "LoadStrategy",
    "StrategyError",
    "TargetRef",
    "apply_text_dtypes",
    "chunked_list",
    "fallback_full",
    "parent_incremental_applies",
    "resolve_batch_size",
    "resolve_strategy",
    "resume_is_watermark",
    "surrogate_provides",
    # Re-exported so strategies do not have to reach into dialects.
    "ColumnDef",
    "SurrogateKey",
    "TableRef",
]
