"""Parsing and validation of the pipeline configuration.

A configuration is a plain dict with three sections — ``TABLE``,
``LOAD_SETTINGS`` and ``SOURCE_DB``. See the README for what goes in them.

**The contract is frozen.** Several hundred pipeline configurations depend on
these key names, so an existing key is never renamed and never changes meaning.
New capability arrives as a new key whose default preserves today's behaviour.

Older configurations also put some keys at the top level instead of in a
section, and those are still read as a fallback. They are not something you
need for a new pipeline — see ``docs/legacy-compat.md``.

Validation happens **during parsing**, not when the source is first touched. A
missing required key raises here rather than half-way through an extraction, on
the same principle that a failed row-count estimate must fail loudly: a quiet
run over wrong data is worse than a noisy failure at startup.

This module knows nothing about dialects, strategies or tunnels. That is
deliberate; where a field brushes against a decision that belongs elsewhere, the
field says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

_log = logging.getLogger(__name__)

#: Keys of the TABLE section. Part of the frozen contract.
TABLE_KEYS: frozenset[str] = frozenset(
    {
        "source_name",
        "source_schema",
        "output_schema",
        "output_name",
        "output_table",
        "selected_columns",
        "only_selected_columns",
        "excluded_columns",
        "where_clause",
        "empty_rows_ok",
    }
)

#: Keys of the LOAD_SETTINGS section.
LOAD_SETTINGS_KEYS: frozenset[str] = frozenset(
    {
        "load_method",
        "primary_column",
        "primary_key_column",
        "batch_size",
        "hash_column",
        "hash_include_columns",
        "hash_exclude_columns",
        "hash_diff_buffer_size",
        "hash_download_batch_size",
        "surrogate_key_enabled",
        "surrogate_key_definition",
        "created_at_column",
        "updated_at_column",
        "days_back",
        "pagination_mode",
        "partition_by_source",
        "partition_by",
        "multi_source",
        "keyset_pagination",
        "resume_full_load",
        "skip_size_estimate",
        "compute_row_hash",
        "conflict_columns",
        "ignore_hash",
        "num_parallel",
        "strict_integer_precision",
        "convert_nchar_to_varchar",
        "read_retry_attempts",
        "read_retry_base_delay",
        # An older spelling of the incremental window. Different names for the
        # same idea, but **not** the same window semantics — see
        # `IncrementalStrategy._build_where`. Without them the keys were
        # silently dropped and 24 of one deployment's 102 pipelines would not
        # start. See docs/legacy-compat.md.
        "incremental_date_column",
        "incremental_date_column_fallback",
        "incremental_lookback_hours",
        # Parent path: a child table has no modification date of its own, so
        # the window is computed over its parent. Read by
        # `ParentIncrementalStrategy`.
        "incremental_parent_table",
        "incremental_parent_output_name",
        "incremental_parent_date_column",
        "incremental_parent_date_column_fallback",
        "incremental_parent_key_column",
        "incremental_parent_id_column",
    }
)

#: Keys of the SOURCE_DB section.
SOURCE_DB_KEYS: frozenset[str] = frozenset(
    {
        "user",
        "password",
        "host",
        "port",
        "database",
        "charset",
        "ssh_address_or_host",
        "ssh_port",
        "ssh_username",
        "ssh_pkey",
        "ssh_wait_timeout",
        "ssh_host_key_checking",
        "remote_bind_address",
        "local_bind_address",
    }
)

#: Top-level keys read as a fallback. See docs/legacy-compat.md.
LEGACY_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "CONNECTION_PARAMS",
        "SSH_PARAMS",
        "BATCH_SIZE",
        "EMPTY_ROWS_OK",
        "OUTPUT_SCHEMA",
        "OUTPUT_TABLE",
        "TABLE_NAME",
        "DEBUG",
        "PRIVATE_HOST",
    }
)

#: Top-level keys this package added; the predecessors had no equivalent.
_NEW_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"connection_mode"})

#: Keys read as a section dict; everything else at the top level is a
#: fallback or an addition.
_SECTION_KEYS: frozenset[str] = frozenset({"TABLE", "LOAD_SETTINGS", "SOURCE_DB"})

#: Top-level keys `parse` reads that are not sections: ``DATABASES`` is a list
#: (see ``ParsedConfig.databases``), ``FINGERPRINT`` a dict of fingerprint
#: settings, ``TARGET_PROFILE`` the io_config.yaml profile to write to. Without
#: them `parse` would report the key a run depends on as unknown.
#:
#: ``DIALECT`` is not read by `parse` — the caller reads it and passes it to
#: ``run()`` — but it is a documented top-level key (see the README example),
#: so warning about it would be a lie. Found by an audit: every pipeline was
#: logging "Unknown top-level key 'DIALECT' — ignoring" while the key was doing
#: exactly what it should, and `parse` itself read ``TARGET_PROFILE`` two lines
#: after warning that it would ignore it.
_READ_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"DATABASES", "FINGERPRINT", "TARGET_PROFILE", "DIALECT"}
)

_KNOWN_TOP_LEVEL_KEYS: frozenset[str] = (
    _SECTION_KEYS | _READ_TOP_LEVEL_KEYS | LEGACY_TOP_LEVEL_KEYS | _NEW_TOP_LEVEL_KEYS
)

#: Accepted values of ``connection_mode``. ``auto`` is the inherited
#: behaviour — probe the direct route, fall back to the tunnel; see core/tunnel.py.
VALID_CONNECTION_MODES: frozenset[str] = frozenset({"direct", "ssh", "auto"})

#: Accepted values of ``SOURCE_DB.ssh_host_key_checking``.
#:
#: The predecessor opened its tunnel with ``StrictHostKeyChecking=no`` and
#: ``UserKnownHostsFile=/dev/null``, so the far end was **never verified**.
#: Inside a closed network that is defensible, but it is the one place in the
#: package where a connection can be substituted. This key makes it a choice
#: without changing anybody's behaviour underneath them:
#:
#: ``off``         — the inherited behaviour, and the default. Changing the
#:                   default would break deployments whose host fingerprint was
#:                   never recorded anywhere.
#: ``accept-new``  — remember the fingerprint on first connect, refuse it if it
#:                   changes. The sensible choice for a new deployment.
#: ``strict``      — the host must already be in ``known_hosts``.
VALID_SSH_HOST_KEY_CHECKING: frozenset[str] = frozenset({"off", "accept-new", "strict"})

#: Spellings accepted as true, taken unchanged from the predecessors.
_TRUTHY_STRINGS: frozenset[str] = frozenset({"true", "t", "1", "yes", "y", "ano", "a"})


class ConfigError(ValueError):
    """The configuration is not valid.

    Raised by ``parse()``/``validate()``, before anything touches the source
    database: a missing required key must surface immediately, not half-way
    through a nightly run.
    """


def is_truthy(value: Any) -> bool:
    """Lenient conversion to bool — configuration values arrive as strings too.

    A bool passes through, ``None`` is always ``False``, anything else is
    compared as a trimmed lower-case string against a set of truthy spellings.

    Applied to every boolean key in ``TABLE`` and ``LOAD_SETTINGS``. The
    predecessors were inconsistent about this — ``bool(...)`` in one place, a
    bare ``.get()`` in another — and unifying on this function is safe because
    it accepts a strict superset of what they accepted.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY_STRINGS


def resolve(name: Optional[str], actual: Optional[dict[str, str]] = None) -> Optional[str]:
    """Look ``name`` up case-insensitively among the source's actual columns.

    Despite the name, this resolves **column names**, not environment variables
    or secrets — configuration spells a column one way and the source spells it
    another, and the two have to be matched before a query is built.

    It is not called during configuration parsing: at that point there is no
    connection to the source, so there is no ``actual`` map to match against.
    It lives here because it operates on configured names; the callers are the
    dialect and hashing layers, which know the source's live columns.

    Args:
        name: The name to look up, typically from configuration. Empty or
            ``None`` yields ``None``.
        actual: Map of lower-case name -> real name, taken from the source.
            When it is missing or empty, ``name`` is returned unchanged —
            there is nothing to match against.
    """
    if not name:
        return None
    if not actual:
        return name
    found = actual.get(str(name).lower())
    if found is None:
        _log.warning("Column '%s' was not found among the actual columns — ignoring.", name)
    return found


@dataclass(frozen=True)
class TableConfig:
    """The ``TABLE`` section, parsed.

    ``output_table`` is resolved from ``output_name``, then ``output_table``,
    then the top-level fallback; see ``docs/legacy-compat.md``.
    """

    source_name: str
    output_schema: str
    output_table: str
    source_schema: Optional[str] = None
    selected_columns: tuple[str, ...] = ()
    only_selected_columns: bool = False
    excluded_columns: tuple[str, ...] = ()
    where_clause: Optional[str] = None
    empty_rows_ok: bool = False


@dataclass(frozen=True)
class LoadSettingsConfig:
    """The ``LOAD_SETTINGS`` section, parsed.

    ``primary_column`` also accepts the older spelling ``primary_key_column``;
    see ``docs/legacy-compat.md``.

    Defaults that are a property of the configuration are filled in here
    (``days_back=14``, ``pagination_mode='auto'``). Defaults that depend on
    something this layer cannot know are left as ``None`` — ``batch_size``, for
    instance, is derived from a size estimate, which is the dialect's business,
    not the parser's.
    """

    load_method: str = "full"
    primary_column: Optional[str] = None
    batch_size: Optional[int] = None
    hash_column: Optional[str] = None
    hash_include_columns: Optional[tuple[str, ...]] = None
    hash_exclude_columns: tuple[str, ...] = ()
    hash_diff_buffer_size: Optional[int] = None
    hash_download_batch_size: Optional[int] = None
    surrogate_key_enabled: bool = False
    surrogate_key_definition: Optional[str] = None
    created_at_column: Optional[str] = None
    updated_at_column: Optional[str] = None
    days_back: int = 14
    #: **Accepted and currently inert.** Part of the frozen contract, so a
    #: pipeline that sets it keeps starting, but no strategy reads it: every
    #: dialect streams one cursor through ``iter_batches`` and resumes by keyset
    #: only after a dropped connection (`core.reading`), so there is no
    #: page-by-page mode left to select. Do not remove it; see
    #: docs/legacy-compat.md, "Keys that are accepted and do nothing".
    pagination_mode: str = "auto"
    partition_by_source: bool = False
    #: General target partitioning. ``None`` means **no partitioning**, which is
    #: the state of every existing table. Held as a plain mapping rather than a
    #: `PartitionSpec` because `entrypoint._settings_dict` hands strategies a
    #: ``dataclasses.asdict``, which would flatten a nested dataclass anyway, and
    #: strategies are meant to stay driven by a dict.
    partition_by: Optional[dict] = None
    multi_source: Optional[bool] = None
    #: **Accepted and currently inert**, for the same reason as
    #: `pagination_mode`. Whether a read can be resumed by keyset is decided by
    #: `full._keyset_usable` from the primary key and the dialect's
    #: ``FEATURE_KEYSET``, not by this key. Do not remove it; see
    #: docs/legacy-compat.md.
    keyset_pagination: bool = False
    resume_full_load: bool = False
    skip_size_estimate: bool = False
    #: **Three states, not two.** ``None`` means the configuration is silent and
    #: the strategy supplies its own default: `full` and friends compute the
    #: hash, `full_by_source` does not. A plain ``bool = True`` would put the key
    #: in `ctx.settings` **always**, and the strategy would never get to apply
    #: its own default.
    compute_row_hash: Optional[bool] = None
    #: **Accepted and currently inert.** The upsert key is the primary key:
    #: `incremental._upsert_from_staging` renders ``ON CONFLICT (primary_column)``
    #: and `target_pg.replace_by_key` deletes by the same column, neither of them
    #: looking here. Widening the conflict key is a change to what an upsert
    #: matches on, which needs a golden run rather than a quiet wiring-up. Do not
    #: remove it; see docs/legacy-compat.md.
    conflict_columns: tuple[str, ...] = ()
    ignore_hash: bool = False
    #: **Accepted and currently inert.** Nothing in the package runs two reads at
    #: once; parallelism is the orchestrator's (three pipelines side by side, so
    #: the memory peak is already tripled — see `resolve_batch_size`). Do not
    #: remove it; see docs/legacy-compat.md.
    num_parallel: Optional[int] = None
    #: Turn the silent rounding of large integers into a failed run.
    #:
    #: An integer column holding a NULL is read as ``float64`` — by
    #: ``pd.read_sql``, before this package sees the frame — and every integer
    #: above 2**53 is then rounded on the way into the target. With this key on,
    #: such a batch raises `coerce.IntegerPrecisionError` instead of being
    #: written; see `coerce.check_integer_precision`.
    #:
    #: **Off by default, and it has to stay that way.** The rounding is inherited
    #: from the predecessor and is the current behaviour of all ~670 tables. On by
    #: default it would turn an unknown number of green runs red overnight, which
    #: the frozen configuration contract does not allow a new key to do.
    strict_integer_precision: bool = False
    #: Read MSSQL's national-character columns through a server-side
    #: ``CONVERT(VARCHAR(MAX), …)``.
    #:
    #: MSSQL is read over a ``cp1250`` connection, which pymssql and FreeTDS
    #: disagree about: the bytes are produced as latin-1 and decoded as cp1250,
    #: so an ``NVARCHAR``/``NCHAR``/``NTEXT`` value is **cut** at the first
    #: character latin-1 cannot express and a character latin-1 *can* express
    #: comes back as a different one. With this key on, the server converts those
    #: columns before they reach the wire, so they arrive whole; see
    #: `dialects.mssql.MSSQLDialect.render_nchar_convert` and
    #: docs/legacy-compat.md.
    #:
    #: **Off by default, and per table.** Turning it on changes the contents of
    #: those columns, so the table has to be reloaded in full for the change to
    #: reach rows that already exist — the frozen contract does not let a new key
    #: do that to ~670 tables by itself. It lives in ``LOAD_SETTINGS`` rather than
    #: in ``SOURCE_DB`` next to ``charset`` because it is a property of one
    #: table's load, not of the connection: `SourceDbConfig` reaches only the
    #: engine URL and the tunnel and never arrives at the SELECT, while
    #: ``LOAD_SETTINGS`` is handed to every strategy as ``ctx.settings``.
    #:
    #: The other three dialects have no national-character types and ignore it —
    #: `LoadStrategy.validate` says so out loud rather than staying silent.
    convert_nchar_to_varchar: bool = False
    #: Override `core.reading.DEFAULT_ATTEMPTS` / `DEFAULT_BASE_DELAY` per table.
    #: ``None`` means the module default applies. Added alongside raising that
    #: default (see `core.reading`, the FederatedX proxy that dies and restarts
    #: silently most nights) so one unusually slow-to-recover source can be
    #: tuned without moving the default for the other ~670 tables.
    read_retry_attempts: Optional[int] = None
    read_retry_base_delay: Optional[float] = None
    #: An older spelling of the incremental window. **Not** aliases of
    #: `updated_at_column` / `created_at_column` — the window is built
    #: differently, see `IncrementalStrategy._build_where` and
    #: docs/legacy-compat.md.
    incremental_date_column: Optional[str] = None
    incremental_date_column_fallback: Optional[str] = None
    incremental_lookback_hours: Optional[int] = None
    #: Parent path, for child tables with no modification date of their own.
    incremental_parent_table: Optional[str] = None
    incremental_parent_output_name: Optional[str] = None
    incremental_parent_date_column: Optional[str] = None
    incremental_parent_date_column_fallback: Optional[str] = None
    #: The child's column referencing the parent, and the parent's own key.
    #: ``None`` means the defaults `ParentIncrementalStrategy` has always used
    #: (``parent_id`` / ``id``), which is what the predecessor hard-codes — so a
    #: configuration that does not name them loads exactly as before.
    #:
    #: Both were read by the strategy long before they were part of the contract,
    #: which meant `entrypoint._settings_dict` (a ``dataclasses.asdict``) could
    #: never carry them and the strategy's own error message pointed at a key
    #: nothing could set.
    incremental_parent_key_column: Optional[str] = None
    incremental_parent_id_column: Optional[str] = None


@dataclass(frozen=True)
class SourceDbConfig:
    """The ``SOURCE_DB`` section, parsed.

    Credentials fall back to older top-level keys only when absent, while
    ``host`` also falls back when empty — an empty password is a password, an
    empty host is not a host. The full rule is in ``docs/legacy-compat.md``.

    ``port`` and ``charset`` deliberately get no default here. Which port a
    given database listens on is the dialect's knowledge, not configuration's.
    """

    user: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    charset: Optional[str] = None
    ssh_address_or_host: Optional[str] = None
    ssh_port: int = 22
    ssh_username: Optional[str] = None
    ssh_pkey: Optional[str] = None
    ssh_wait_timeout: int = 20
    #: SSH host key verification. Defaults to ``off``, the inherited
    #: behaviour; see `VALID_SSH_HOST_KEY_CHECKING`.
    ssh_host_key_checking: str = "off"
    remote_bind_address: Optional[tuple[str, int]] = None
    local_bind_address: Optional[tuple[str, int]] = None


@dataclass(frozen=True)
class ParsedConfig:
    """The configuration, parsed into sections with defaults applied.

    A typed dataclass rather than a dict on purpose: a typo in a field name
    should surface at import time, not at three in the morning.
    """

    table: TableConfig
    load_settings: LoadSettingsConfig
    source_db: SourceDbConfig
    connection_mode: str = "auto"
    debug: bool = False
    #: Source databases to iterate over (top-level ``DATABASES``), filled in by
    #: the caller's own control layer. ``None`` means the key is absent, so
    #: there is a single database from ``SOURCE_DB.database``. An **empty list**
    #: means the control layer selected none, which is a legitimate state, not
    #: an error.
    databases: Optional[tuple[str, ...]] = None
    #: Source fingerprint, the ``FINGERPRINT`` section. Empty disables it.
    fingerprint: dict = field(default_factory=dict)
    #: Which `io_config.yaml` profile to write to (top-level ``TARGET_PROFILE``).
    #: ``None`` means the default, ``warehouse`` — the one every predecessor had
    #: hard-coded.
    #:
    #: This supersedes the ``TARGET_DB`` section, which 342 configuration blocks
    #: declare, passwords included, and which **nothing has ever read**. It
    #: looked authoritative and was not; see docs/legacy-compat.md.
    target_profile: Optional[str] = None


def _required_str(value: Any) -> str:
    """A required text field; missing or empty becomes ``""``.

    The annotation says ``str`` because that is what dialects and strategies
    want to read, but the actual "it is missing" check happens in
    ``validate()`` — deliberately, so that every missing required key is
    reported at once instead of ``parse()`` failing on the first one.
    """
    return str(value) if value else ""


def _as_tuple(value: Any) -> tuple[str, ...]:
    """A configured sequence to a tuple. ``None`` or empty yields ``()``."""
    if not value:
        return ()
    return tuple(value)


def _as_optional_tuple(value: Any) -> Optional[tuple[str, ...]]:
    """Like ``_as_tuple``, but keeps "key absent" (``None``) distinct from "empty"."""
    if value is None:
        return None
    return tuple(value)


def _partition_by(load_cfg: dict, table_cfg: dict) -> Optional[dict]:
    """``partition_by``, parsed and validated, or ``None``.

    Validated **here** rather than at the target: a nonsensical mode should fail
    `parse()`, before anything touches the source. It is looked for in
    ``LOAD_SETTINGS`` first and ``TABLE`` second, same as its predecessor.

    ``partition_by_source: true`` keeps working and means exactly
    ``partition_by: {column: _source, mode: list}``. Someone who writes both
    meant the more specific one, so ``partition_by`` wins.
    """
    from dbextractors.core import partitioning

    raw = load_cfg.get("partition_by", table_cfg.get("partition_by"))
    legacy = is_truthy(load_cfg.get("partition_by_source", table_cfg.get("partition_by_source")))
    try:
        spec = partitioning.parse_spec(raw, legacy_by_source=legacy)
    except ValueError as err:
        raise ConfigError(str(err)) from err
    return spec.as_dict() if spec else None


def _as_optional_bool(value: Any) -> Optional[bool]:
    """``is_truthy`` that keeps ``None`` instead of quietly returning ``False``.

    Used for ``multi_source``, where an absent value means "derive it at run
    time from the number of source databases" — a different thing from an
    explicit
    ``False``.

    And for ``compute_row_hash``, where ``None`` means the strategy decides:
    `full` computes it, `full_by_source` does not.
    """
    if value is None:
        return None
    return is_truthy(value)


def _as_bind_address(value: Any, *, label: str) -> Optional[tuple[str, int]]:
    """Validate a configured ``(host, port)`` pair. A bad shape warns and yields ``None``.

    Filling in a runtime default — deriving it from ``private_host`` or picking
    a free port — belongs to core/tunnel.py, not here. This only takes what was
    given.
    """
    if value is None:
        return None
    if isinstance(value, list | tuple) and len(value) == 2:
        host, port = value
        try:
            return (str(host), int(port))
        except (TypeError, ValueError):
            pass
    _log.warning(
        "SOURCE_DB.%s has an invalid shape (%r), expected [host, port] — ignoring.",
        label,
        value,
    )
    return None


def _warn_unknown_keys(config: dict, table_cfg: dict, load_cfg: dict, source_db_cfg: dict) -> None:
    """Log unknown keys as warnings. Never raises: a key added for one
    deployment must not bring down the others."""
    for key in config:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            _log.warning("Unknown top-level key '%s' — ignoring.", key)
    for key in table_cfg:
        if key not in TABLE_KEYS:
            _log.warning("Unknown key '%s' in the TABLE section — ignoring.", key)
    for key in load_cfg:
        if key not in LOAD_SETTINGS_KEYS:
            _log.warning("Unknown key '%s' in the LOAD_SETTINGS section — ignoring.", key)
    for key in source_db_cfg:
        if key not in SOURCE_DB_KEYS and key not in _NEW_TOP_LEVEL_KEYS:
            _log.warning("Unknown key '%s' in the SOURCE_DB section — ignoring.", key)


def parse(config: dict) -> ParsedConfig:
    """Parse the configuration dict into sections, applying fallbacks and defaults.

    Must also accept configurations that carry only the older top-level keys;
    the order of fallbacks is documented on each dataclass and in
    ``docs/legacy-compat.md``.

    Validates immediately: a missing required key or an invalid
    ``connection_mode`` raises ``ConfigError`` here, not later in a strategy or
    dialect.

    Unknown keys — in any section or at the top level — are only logged. A key
    added for one deployment must not bring down the others.
    """
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a dict, got {type(config).__name__}.")

    table_cfg = config.get("TABLE") or {}
    load_cfg = config.get("LOAD_SETTINGS") or {}
    legacy_conn = config.get("CONNECTION_PARAMS") or {}
    source_db_cfg = config.get("SOURCE_DB") or legacy_conn
    ssh_defaults = config.get("SSH_PARAMS") or {}

    _warn_unknown_keys(config, table_cfg, load_cfg, source_db_cfg)

    table = TableConfig(
        source_name=_required_str(table_cfg.get("source_name") or config.get("TABLE_NAME")),
        source_schema=table_cfg.get("source_schema"),
        output_schema=_required_str(table_cfg.get("output_schema") or config.get("OUTPUT_SCHEMA")),
        output_table=_required_str(
            table_cfg.get("output_name")
            or table_cfg.get("output_table")
            or config.get("OUTPUT_TABLE")
        ),
        selected_columns=_as_tuple(table_cfg.get("selected_columns")),
        only_selected_columns=is_truthy(table_cfg.get("only_selected_columns", False)),
        excluded_columns=_as_tuple(table_cfg.get("excluded_columns")),
        where_clause=table_cfg.get("where_clause"),
        empty_rows_ok=is_truthy(table_cfg.get("empty_rows_ok", config.get("EMPTY_ROWS_OK", False))),
    )

    load_settings = LoadSettingsConfig(
        load_method=load_cfg.get("load_method", "full"),
        primary_column=load_cfg.get("primary_column") or load_cfg.get("primary_key_column"),
        batch_size=load_cfg.get("batch_size", config.get("BATCH_SIZE")),
        hash_column=load_cfg.get("hash_column"),
        hash_include_columns=_as_optional_tuple(load_cfg.get("hash_include_columns")),
        hash_exclude_columns=_as_tuple(load_cfg.get("hash_exclude_columns")),
        hash_diff_buffer_size=load_cfg.get("hash_diff_buffer_size"),
        hash_download_batch_size=load_cfg.get("hash_download_batch_size"),
        surrogate_key_enabled=is_truthy(load_cfg.get("surrogate_key_enabled", False)),
        surrogate_key_definition=load_cfg.get("surrogate_key_definition"),
        created_at_column=load_cfg.get("created_at_column"),
        updated_at_column=load_cfg.get("updated_at_column"),
        days_back=load_cfg.get("days_back", 14),
        pagination_mode=load_cfg.get("pagination_mode", "auto"),
        # partition_by_source / multi_source belong to LOAD_SETTINGS by the
        # contract, but the one predecessor that used them read them from TABLE.
        # LOAD_SETTINGS is tried first, TABLE second, so both shapes work.
        # Viz docs/merge-decisions.md.
        partition_by_source=is_truthy(
            load_cfg.get("partition_by_source", table_cfg.get("partition_by_source", False))
        ),
        partition_by=_partition_by(load_cfg, table_cfg),
        multi_source=_as_optional_bool(load_cfg.get("multi_source", table_cfg.get("multi_source"))),
        keyset_pagination=is_truthy(load_cfg.get("keyset_pagination", False)),
        resume_full_load=is_truthy(load_cfg.get("resume_full_load", False)),
        skip_size_estimate=is_truthy(load_cfg.get("skip_size_estimate", False)),
        compute_row_hash=_as_optional_bool(load_cfg.get("compute_row_hash")),
        conflict_columns=_as_tuple(load_cfg.get("conflict_columns")),
        ignore_hash=is_truthy(load_cfg.get("ignore_hash", False)),
        num_parallel=load_cfg.get("num_parallel"),
        strict_integer_precision=is_truthy(load_cfg.get("strict_integer_precision", False)),
        convert_nchar_to_varchar=is_truthy(load_cfg.get("convert_nchar_to_varchar", False)),
        read_retry_attempts=load_cfg.get("read_retry_attempts"),
        read_retry_base_delay=load_cfg.get("read_retry_base_delay"),
        incremental_date_column=load_cfg.get("incremental_date_column") or None,
        incremental_date_column_fallback=load_cfg.get("incremental_date_column_fallback") or None,
        incremental_lookback_hours=load_cfg.get("incremental_lookback_hours") or None,
        incremental_parent_table=load_cfg.get("incremental_parent_table") or None,
        incremental_parent_output_name=load_cfg.get("incremental_parent_output_name") or None,
        incremental_parent_date_column=load_cfg.get("incremental_parent_date_column") or None,
        incremental_parent_date_column_fallback=(
            load_cfg.get("incremental_parent_date_column_fallback") or None
        ),
        incremental_parent_key_column=load_cfg.get("incremental_parent_key_column") or None,
        incremental_parent_id_column=load_cfg.get("incremental_parent_id_column") or None,
    )

    source_db = SourceDbConfig(
        user=source_db_cfg.get("user", legacy_conn.get("user")),
        password=source_db_cfg.get("password", legacy_conn.get("password")),
        host=source_db_cfg.get("host") or config.get("PRIVATE_HOST"),
        port=source_db_cfg.get("port", legacy_conn.get("port")),
        database=source_db_cfg.get("database", legacy_conn.get("database")),
        charset=source_db_cfg.get("charset"),
        ssh_address_or_host=source_db_cfg.get(
            "ssh_address_or_host", ssh_defaults.get("ssh_address_or_host")
        ),
        ssh_port=source_db_cfg.get("ssh_port", ssh_defaults.get("ssh_port", 22)),
        ssh_username=source_db_cfg.get("ssh_username", ssh_defaults.get("ssh_username")),
        ssh_pkey=source_db_cfg.get("ssh_pkey", ssh_defaults.get("ssh_pkey")),
        ssh_wait_timeout=source_db_cfg.get(
            "ssh_wait_timeout", ssh_defaults.get("ssh_wait_timeout", 20)
        ),
        # A new key with a safe default. `or` rather than `get(…, default)`:
        # an empty value means "not filled in", not "the empty mode" — the same
        # reasoning as `host` above.
        ssh_host_key_checking=str(
            source_db_cfg.get("ssh_host_key_checking")
            or ssh_defaults.get("ssh_host_key_checking")
            or "off"
        )
        .strip()
        .lower(),
        remote_bind_address=_as_bind_address(
            source_db_cfg.get("remote_bind_address"), label="remote_bind_address"
        ),
        local_bind_address=_as_bind_address(
            source_db_cfg.get("local_bind_address"), label="local_bind_address"
        ),
    )

    # A key this package added; no predecessor had it. It lives in SOURCE_DB
    # because it is about how the source is reached, alongside the ssh_* keys;
    # the top level is accepted as a convenience.
    raw_mode = source_db_cfg.get("connection_mode") or config.get("connection_mode") or "auto"
    connection_mode = str(raw_mode).strip().lower()

    # `DATABASES` must **not** collapse to None when empty: an empty list means
    # "the control layer selected no database", an absent key means "one, from
    # SOURCE_DB.database". Those are different states.
    databases_cfg = config.get("DATABASES")
    databases = (
        None
        if databases_cfg is None
        else tuple(str(db).strip() for db in databases_cfg if str(db).strip())
    )

    parsed = ParsedConfig(
        table=table,
        load_settings=load_settings,
        source_db=source_db,
        connection_mode=connection_mode,
        debug=is_truthy(config.get("DEBUG", False)),
        databases=databases,
        fingerprint=dict(config.get("FINGERPRINT") or {}),
        target_profile=(config.get("TARGET_PROFILE") or None),
    )
    validate(parsed)
    return parsed


def validate(parsed: ParsedConfig) -> None:
    """Validate the configuration before anything touches the source.

    An unknown key is not an error — the contract may grow, and ``parse()``
    has already logged it. This checks required values and internal consistency
    (``connection_mode`` vs. co je k dispozici v ``SOURCE_DB``).

    Every problem is collected before raising, so that a broken configuration
    shows all of its faults on the first run rather than one per attempt.
    """
    errors: list[str] = []

    if not parsed.table.source_name:
        errors.append("TABLE.source_name (or the top-level TABLE_NAME) is missing.")
    if not parsed.table.output_schema:
        errors.append("TABLE.output_schema (or the top-level OUTPUT_SCHEMA) is missing.")
    if not parsed.table.output_table:
        errors.append("TABLE.output_name/output_table (or the top-level OUTPUT_TABLE) is missing.")

    if not parsed.source_db.user:
        errors.append("SOURCE_DB.user (or CONNECTION_PARAMS.user) is missing.")
    if not parsed.source_db.password:
        errors.append("SOURCE_DB.password (or CONNECTION_PARAMS.password) is missing.")
    # A multi-source configuration **cannot** carry a database name in
    # ``SOURCE_DB``: each entry of ``DATABASES`` is a different database on the
    # same server and the extractor switches between them, so ``SOURCE_DB``
    # holds only host, port and credentials. Requiring ``database``
    # unconditionally would stop every such pipeline from running.
    if not parsed.source_db.database and not parsed.databases:
        errors.append(
            "SOURCE_DB.database (or CONNECTION_PARAMS.database) is missing "
            "and there is no DATABASES list either."
        )

    if parsed.source_db.ssh_host_key_checking not in VALID_SSH_HOST_KEY_CHECKING:
        errors.append(
            "SOURCE_DB.ssh_host_key_checking must be one of "
            f"{sorted(VALID_SSH_HOST_KEY_CHECKING)}, "
            f"got {parsed.source_db.ssh_host_key_checking!r}."
        )

    if parsed.connection_mode not in VALID_CONNECTION_MODES:
        errors.append(
            "connection_mode must be one of "
            f"{sorted(VALID_CONNECTION_MODES)}, got {parsed.connection_mode!r}."
        )
    elif parsed.connection_mode == "direct" and not parsed.source_db.host:
        errors.append(
            "connection_mode='direct' requires SOURCE_DB.host (or the top-level PRIVATE_HOST)."
        )
    elif parsed.connection_mode == "ssh":
        missing_ssh = [
            field_name
            for field_name, value in (
                ("ssh_address_or_host", parsed.source_db.ssh_address_or_host),
                ("ssh_username", parsed.source_db.ssh_username),
                ("ssh_pkey", parsed.source_db.ssh_pkey),
            )
            if not value
        ]
        if missing_ssh:
            errors.append(
                "connection_mode='ssh' requires SOURCE_DB." + ", SOURCE_DB.".join(missing_ssh)
            )

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
