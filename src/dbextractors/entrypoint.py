"""The single public entry point of the package.

It replaces the body of 15 hand-copied ``load_data`` blocks. The interface must
not change: it takes the same ``config`` dict the Mage config blocks produce
today and returns a ``pd.DataFrame`` with the state of the run (the predecessors'
``_build_status_df``).

It wires together `core.config` -> `core.tunnel` -> the dialect -> `core.naming`
-> the strategy -> `core.target_pg`.

## The tunnel is opened once per run, not once per database

``run`` iterates over the ``DATABASES`` list in a multi-source run, but the SSH
tunnel is the same for all of them: it leads to one server and only the database
name in the URL changes. Opening it inside the loop would mean N ``ssh``
processes one after another and N waits on ``wait_for_port`` — with sixteen
databases that is sixteen handshakes instead of one. Hence
``with open_tunnel(...)`` **around** the loop.

## Runtime variables

Mage passes the pipeline's runtime variables in ``kwargs``. The package reads two
of them (``forced_full_load``, ``incremental_lookback_hours``); the rest of the
Mage kwargs (``execution_date``, ``block_uuid``, …) is ignored. See
`_runtime_vars`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, cast

import pandas as pd

from dbextractors.core import secrets

if TYPE_CHECKING:  # pragma: no cover
    from dbextractors.core.strategies.base import LoadContext, LoadResult

_log = logging.getLogger(__name__)

#: All four sources. The module is imported only in `resolve_dialect`, because
#: the driver (`fdb`, `pymssql`) need not be installed in every environment.
DIALECTS = {
    "mysql": "dbextractors.dialects.mysql:MySQLDialect",
    "mssql": "dbextractors.dialects.mssql:MSSQLDialect",
    "postgres": "dbextractors.dialects.postgres:PostgresDialect",
    "firebird": "dbextractors.dialects.firebird:FirebirdDialect",
}

#: Names accepted in addition. `postgresql` is the more common spelling than
#: `postgres` — it is what the SQLAlchemy dialect is called and what the
#: predecessor's file name says, so nearly everyone reaches for it. The alias is
#: cheaper than a runtime error across 80 pipelines: `run()` is called inside the
#: block, so a typo shows up when the pipeline runs, not when it is deployed.
#: Deliberately this one alias only. Aliases "just in case" (`pg`, `sqlserver`,
#: `mariadb`) are not here — with `mariadb` they would also claim a behavioural
#: equivalence nobody has verified. One gets added once someone actually reaches
#: for that name.
DIALECT_ALIASES = {"postgresql": "postgres"}


#: The runtime variables the package really reads out of the Mage kwargs. Both of
#: them are documented in the predecessors.
RUNTIME_KEYS: tuple = ("forced_full_load", "incremental_lookback_hours")

#: The strategy that ``forced_full_load`` does **not** rewrite. If
#: `full_by_source` were rewritten to `full`, every database in a multi-source run
#: would `TRUNCATE` the whole target table and leave only its own rows in it — the
#: last database would win and the rest would vanish. The predecessors handle it
#: by not changing the path at all for MSSQL and merely silencing the fingerprint;
#: only the Firebird one rewrites it (``resolve_load_path``), and there
#: `full_by_source` does not exist.
_FORCE_KEEPS_STRATEGY: frozenset = frozenset({"full_by_source"})


class EntrypointError(RuntimeError):
    """The run could not even start."""


class SourceExtractionError(EntrypointError):
    """One or more sources failed during extraction.

    A type of its own, because it is a different situation from "the run could
    not start": the other databases finished and their data is in the target. The
    predecessors arrived at it the same way — failures used to be collected into
    the status frame and the block ended green even when whole databases were
    missing from the target.
    """


def resolve_dialect(name: str):
    """Build the dialect named in ``run(dialect=…)``."""
    key = (name or "").strip().lower()
    key = DIALECT_ALIASES.get(key, key)
    if key not in DIALECTS:
        raise EntrypointError(f"Unknown dialect {name!r}. Known: {', '.join(sorted(DIALECTS))}.")
    import importlib

    module_name, _, class_name = DIALECTS[key].partition(":")
    return getattr(importlib.import_module(module_name), class_name)()


def run(
    config: dict,
    dialect: str,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Perform one extraction according to the configuration.

    Args:
        config: The configuration dict from the Mage config block. The contract
            is frozen.
        dialect: ``'mysql' | 'mssql' | 'postgres' | 'firebird'``.
        logger: The Mage logger from ``kwargs.get('logger')``. May be ``None``.
        **kwargs: The rest of the Mage kwargs. The pipeline's runtime variables
            are read from it — see `RUNTIME_KEYS`.

    Returns:
        A DataFrame with the state of the run.
    """
    from dbextractors.core import config as config_module
    from dbextractors.core import status as status_module
    from dbextractors.core import tunnel as tunnel_module
    from dbextractors.core.logging import adapt
    from dbextractors.core.strategies.base import resolve_strategy

    # Mage's `DictLogger` takes no positional arguments, so the first `%s` message
    # would kill the whole run. It is wrapped right here so that everything below
    # — strategies, dialects, the tunnel, writing to the target — gets a logger
    # that behaves like a logger. See `dbextractors/core/logging.py`.
    logger = adapt(logger)

    parsed = config_module.parse(config)
    source = resolve_dialect(dialect)
    runtime = _runtime_vars(kwargs)

    strategy_name = _strategy_name(parsed.load_settings.load_method, runtime, logger=logger)
    # `settings` decide where `load_method` alone is not enough — today with
    # `resume_full_load`, which is effectively a watermark, not a full load.
    strategy = resolve_strategy(strategy_name, _settings_dict(parsed.load_settings, runtime))

    databases = resolve_databases(parsed, logger=logger)
    if databases is None:
        return _nothing_selected_frame(parsed)

    #: Whether the rows carry `_source` **must not** depend on how many databases
    #: this particular run selected: `full_by_source` deletes by `_source`, so a
    #: run over a single database would otherwise wipe the whole table. The
    #: configuration decides, and when it says nothing this is decided here
    #: **once** for the whole run — not inside the strategy.
    multi = parsed.load_settings.multi_source
    if multi is None:
        multi = len(databases) > 1

    target_label = f"{parsed.table.output_schema}.{parsed.table.output_table}"
    statuses: List[dict] = []
    # The tunnel wraps the **whole** loop, not the individual databases — see the
    # module docstring. `probe` is supplied by the dialect: MSSQL has one with
    # three attempts, the others with one, and that is the right knowledge in the
    # right place.
    # The `cast` is safe: `config.parse` has validated the value against
    # `VALID_CONNECTION_MODES` and raises `ConfigError` on an unknown one.
    with tunnel_module.open_tunnel(
        _tunnel_params(parsed, source),
        cast("tunnel_module.ConnectionMode", parsed.connection_mode),
        probe=source.probe,
        show_debug=parsed.debug,
    ) as address:
        for index, database in enumerate(databases):
            source_label = database if multi else None
            try:
                statuses.append(
                    _run_one(
                        parsed,
                        source,
                        strategy,
                        logger=logger,
                        database=database,
                        source_label=source_label,
                        is_first_source=index == 0,
                        is_last_source=index == len(databases) - 1,
                        address=address,
                        runtime=runtime,
                    )
                )
            except Exception as err:
                # The databases are independent of each other, so one failing must
                # not take away the chance of the remaining fifteen. The summary
                # exception comes after the loop — literally the predecessors'
                # behaviour.
                #
                # This is the funnel every failure below `run` passes through, and it
                # has three exits: the package log, the Mage log and the `error`
                # column of the returned frame (which `SourceExtractionError` then
                # quotes as well). Redaction therefore happens **here**, once, rather
                # than at each of the three.
                secret = parsed.source_db.password
                _log.error(
                    # Not `_log.exception`: that emits the traceback through
                    # `exc_info`, i.e. formatted by the handler, where there is
                    # nothing left to redact. The traceback is rendered here so that
                    # it can be — the text is the same, the chained `__cause__`
                    # frames included.
                    "🛑 [%s] extraction failed.\n%s",
                    database,
                    secrets.redact(_format_traceback(err), extra=[secret]),
                )
                safe = secrets.redact(err, extra=[secret])
                if logger:
                    logger.error("🛑 [%s] extraction failed: %s", database, safe)
                statuses.append(status_module.error_status(target_label, source_label, safe))

    failed = [s for s in statuses if not s.get("success")]
    if failed:
        detail = "; ".join(f"{s.get('source') or '-'}: {s.get('error')}" for s in failed)
        raise SourceExtractionError(
            f"{len(failed)} of {len(statuses)} sources failed extracting {target_label} → {detail}"
        )

    return status_module.build_status_df(statuses)


def _format_traceback(err: BaseException) -> str:
    """The traceback as text, so that it can go through `secrets.redact`.

    ``logging.exception`` hands the exception to the handler and the handler
    formats it — by which point the string is out of our reach. Rendering it here
    costs one join and buys the only place a redaction can be applied.
    """
    import traceback

    return "".join(traceback.format_exception(type(err), err, err.__traceback__))


def _log_columns(ctx) -> None:
    """Print the columns that will be worked with, and under ``DEBUG`` the type maps.

    The predecessors report this for every table (`Columns after selection`, plus
    `[DEBUG] integer_columns_mapping / overwrite_types / orig_type_map /
    all_columns_ordered`) and it is the first thing anyone looks at while
    debugging: that listing shows an excluded column, a badly guessed type and the
    ordering. This package did not print it at all.

    It is deliberately **one place for all strategies** — if each logged it for
    itself, three of the six would sooner or later stop doing it.
    """
    ctx.log(
        "warning", "📋 Columns after selection (%d): %s", len(ctx.target_names), ctx.target_names
    )
    if not ctx.debug:
        return
    ctx.log("warning", "🔍 [DEBUG] overwrite_types: %s", ctx.overwrite_types)
    ctx.log("warning", "🔍 [DEBUG] orig_type_map: %s", ctx.orig_type_map)
    ctx.log("warning", "🔍 [DEBUG] surrogate: %s", ctx.surrogate)
    ctx.log("warning", "🔍 [DEBUG] where: %s", ctx.where)


def _run_one(parsed, source, strategy, **kwargs) -> dict:
    """One source: build the context, run the strategy, return the status row."""
    ctx = build_context(parsed, source, **kwargs)
    _log_columns(ctx)
    try:
        result = strategy.run(ctx)
    finally:
        ctx.target_conn.close()
    return status_row(ctx, result, address=kwargs.get("address"))


def _runtime_vars(kwargs: dict) -> dict:
    """Pick the runtime variables the package knows out of the Mage kwargs.

    The rest (``execution_date``, ``block_uuid``, ``pipeline_run``, …) is dropped
    **silently**: Mage passes dozens of them and warning about each would turn the
    log into noise. Drop them we must, though — ``ctx.runtime`` goes into the
    strategies and they ask it for keys by name.

    A key that is not in the kwargs at all does not reach the result. That is not
    the same as ``None``: `compute_incremental_cutoff` reads ``None`` as "the
    runtime variable is empty, take the value from LOAD_SETTINGS", which is a
    different thing from "the variable does not exist".
    """
    return {key: kwargs[key] for key in RUNTIME_KEYS if key in kwargs}


def _strategy_name(load_method: str, runtime: dict, *, logger=None) -> str:
    """The strategy name once ``forced_full_load`` has been taken into account.

    The runtime variable ``forced_full_load`` means "trust nothing the target
    remembers and pull everything". It has two independent effects and it is easy
    to remember only one of them:

    1. **the path** — the incremental strategies are rewritten to `full` (the
       predecessors' ``resolve_load_path``),
    2. **the fingerprint** — it has to be ignored, otherwise "force" would still
       skip unchanged databases. That is done by `_settings_dict` through
       ``force_reload``.

    The exception is `full_by_source`, see `_FORCE_KEEPS_STRATEGY`.
    """
    from dbextractors.core.config import is_truthy

    name = (load_method or "").strip().lower()
    if not is_truthy(runtime.get("forced_full_load")):
        return name

    if name in _FORCE_KEEPS_STRATEGY:
        if logger:
            logger.warning(
                "❗ forced_full_load: strategy %s stays as it is (rewriting it to full "
                "would delete the other databases' slices in a multi-source run) — only "
                "the fingerprint is ignored.",
                name,
            )
        return name

    if logger and name != "full":
        logger.warning("❗ forced_full_load: %s is rewritten to full.", name)
    return "full"


def _tunnel_params(parsed, source_dialect) -> dict:
    """``SOURCE_DB`` in the shape `core.tunnel.open_tunnel` understands.

    The port is filled in here from the dialect's default. `core.config`
    deliberately leaves it empty (the default port is dialect knowledge, not
    configuration), but ``connection_mode='direct'`` without a port is an error —
    and it would be an error about something the user does not have to fill in.
    """
    import dataclasses

    params = dataclasses.asdict(parsed.source_db)
    params["port"] = parsed.source_db.port or source_dialect.default_port
    return params


def resolve_databases(parsed, *, logger=None) -> Optional[List[str]]:
    """Which source databases the run should go over.

    Three different states that must not be confused:

    - ``DATABASES`` is **not** in the configuration -> one database from
      ``SOURCE_DB``,
    - ``DATABASES`` is there and holds something -> the run goes over them,
    - ``DATABASES`` is there and is **empty** -> the control layer selected none.
      ``None`` is returned and it is a legitimate state, not an error: every
      database is closed and up to date, there is nothing to fetch.
    """
    if parsed.databases is None:
        if not parsed.source_db.database:
            raise EntrypointError(
                "The configuration does not name a source database — both "
                "SOURCE_DB.database and the DATABASES list are missing."
            )
        return [parsed.source_db.database]

    if not parsed.databases:
        if logger:
            logger.warning(
                "ℹ️ The control layer selected no database — all of them are closed and "
                "up to date, there is nothing to fetch."
            )
        return None
    return list(parsed.databases)


def _nothing_selected_frame(parsed) -> pd.DataFrame:
    """The status frame for a run that had nothing to do. It must be **green**.

    ``output_table``, not ``output_name``: `TableConfig` merges both contract
    names into one field. The other one used to stand here and this branch failed
    with ``AttributeError`` — so exactly that legitimate state ("the control layer
    selected no database") ended red.
    """
    from dbextractors.core import status as status_module

    return status_module.build_status_df(
        [
            {
                "table": f"{parsed.table.output_schema}.{parsed.table.output_table}",
                "rows_written": 0,
                "load_method": "nothing_selected",
                "success": True,
                "is_incremental": False,
                "data_present": True,
                # `connection_mode` stays None — the tunnel was never opened for
                # this run, there was nothing to fetch.
            }
        ]
    )


def build_context(
    parsed,
    source_dialect,
    *,
    logger=None,
    database: Optional[str] = None,
    source_label: Optional[str] = None,
    is_first_source: bool = True,
    is_last_source: bool = True,
    address: Optional[Any] = None,
    runtime: Optional[dict] = None,
) -> LoadContext:
    """Assemble a `LoadContext` from the parsed configuration.

    This is where all three layers meet: the dialect says what columns and types
    the source has, `core.naming` adds the target names to them and
    `core.target_pg` supplies the connection to the target.

    ``address`` is a `core.tunnel.TunnelAddress` — where the engine should
    connect. With a tunnel that is ``127.0.0.1`` and a local port, with a direct
    connection the host from ``SOURCE_DB``. When it is not passed, the run goes
    directly by ``SOURCE_DB``; that branch is used by the golden harness, which
    manages the connection itself.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    from dbextractors.core.strategies.base import LoadContext, TargetRef
    from dbextractors.dialects.base import SurrogateKey, TableRef

    source_db = parsed.source_db
    # Which database is read from is the caller's decision (`run` iterates the
    # `DATABASES` list); `SOURCE_DB.database` is the fallback for single-source
    # tables.
    database = database or source_db.database
    if address is not None:
        host, port = address.host, int(address.port)
    else:
        host = source_db.host
        port = source_db.port or source_dialect.default_port
        if not host:
            raise EntrypointError(
                "SOURCE_DB has no host and no tunnel address was given either. Fill in "
                "SOURCE_DB.host, or set connection_mode: ssh and the ssh_* details."
            )

    conn_str = source_dialect.build_conn_str(
        {
            "user": source_db.user,
            "password": source_db.password,
            "database": database,
            "charset": source_db.charset,
        },
        host,
        port,
    )
    # NullPool: the extraction is read-only and the connection is thrown away
    # right after use. The predecessors got here through "Lost connection to MySQL
    # server" on dead idle connections — a pooled connection has time to fall
    # apart between batches.
    #
    # `connect_args` is supplied by the dialect. With MySQL it decides whether the
    # result is streamed or pulled into RAM in one piece — on 6 M rows that is the
    # difference between 318 MB and 4 667 MB.
    #
    # `conn_str` carries the password. When the URL is unusable for the dialect,
    # SQLAlchemy prints it in full in the exception text — and that ends up in the
    # Mage log, which more people see than the pipeline configuration. The message
    # is therefore rewrapped through `secrets.redact`; the original exception stays
    # in `__cause__`, so no debugging context is lost.
    try:
        engine = create_engine(
            conn_str, poolclass=NullPool, connect_args=dict(source_dialect.connect_args)
        )
    except Exception as err:
        # `from None`, not `from err`: chaining would keep the original
        # exception — full URL, password included — attached as __cause__, and
        # anything that logs the whole traceback ("The above exception was the
        # direct cause…") would print it. Redacting the message while chaining
        # the unredacted original is security theatre; the redacted text below
        # carries everything the original said.
        raise EntrypointError(
            f"The connection to the source could not be built: "
            f"{secrets.redact(err, extra=[source_db.password])}"
        ) from None

    _attach_session_sql(engine, source_dialect, logger)

    ref = TableRef(
        name=parsed.table.source_name,
        schema=parsed.table.source_schema,
        database=database,
    )
    columns = source_dialect.introspect_columns(engine, ref)
    columns = _select_columns(columns, parsed.table, parsed.load_settings)
    columns = _with_target_names(columns, parsed.load_settings.hash_column)

    surrogate = None
    if parsed.load_settings.surrogate_key_enabled:
        surrogate = SurrogateKey(
            enabled=True,
            alias=parsed.load_settings.primary_column,
            expr=parsed.load_settings.surrogate_key_definition,
        )

    return LoadContext(
        dialect=source_dialect,
        engine=engine,
        source=ref,
        target=TargetRef(schema=parsed.table.output_schema, table=parsed.table.output_table),
        target_conn=_target_connection(logger, parsed.target_profile),
        columns=columns,
        settings=_settings_dict(parsed.load_settings, runtime),
        table_cfg={"empty_rows_ok": parsed.table.empty_rows_ok},
        where=parsed.table.where_clause,
        surrogate=surrogate,
        logger=logger,
        debug=parsed.debug,
        source_label=source_label,
        is_first_source=is_first_source,
        is_last_source=is_last_source,
        fingerprint_cfg=dict(parsed.fingerprint or {}),
        runtime=dict(runtime or {}),
    )


def _required_columns(columns: List[Any], load_settings) -> set:
    """Columns a strategy needs even when the selection does not list them.

    The predecessors do not keep only the selected columns but the **union** of
    ``selected_columns`` with the required ones — and symmetrically subtract the
    same set from ``excluded_columns``, so the key cannot be excluded by accident.

    Without this a real pipeline breaks: it has
    ``incremental_date_column: 'CORRECTEDAT$DATE'``, but that column is not among
    its 29 ``selected_columns`` (only the fallback ``CREATEDAT$DATE`` is). The
    Firebird extractor supplies it today; without this safeguard it gets filtered
    out and validation ends with "updated_at_column … is not in the source table".
    That pipeline runs 15 times a week.

    **The set has to match the predecessors, not be larger.** If this package held
    on to one extra column, the target would have one column more than it has
    today, and the target column names are frozen. That is why the inherited shape
    (``incremental_date_column``, whose fallback is **not** added, because the
    predecessor does not have it there) is distinguished from the rest
    (``updated_at_column`` and friends). See ``docs/legacy-compat.md``.

    Only columns the source **really has** are added — the same as in the
    predecessors (``if col and col in column_names``).
    """
    names = {c.name for c in columns}
    method = (load_settings.load_method or "").strip().lower()
    # `primary_key_column` is not read here: `config.parse` has already merged it
    # into `primary_column` (see LoadSettingsConfig, "primary_column wins over
    # primary_key_column").
    pk = load_settings.primary_column
    hash_col = load_settings.hash_column

    candidates: tuple[Optional[str], ...]
    if method == "incremental":
        if load_settings.incremental_date_column:
            candidates = (pk, load_settings.incremental_date_column)
        else:
            candidates = (
                pk,
                load_settings.updated_at_column,
                load_settings.created_at_column,
                hash_col,
            )
    elif method == "hash":
        candidates = (pk, hash_col)
    else:
        candidates = (hash_col,)

    return {c for c in candidates if c and c in names}


def _select_columns(columns: List[Any], table_cfg, load_settings) -> List[Any]:
    """Narrow the columns down by ``selected_columns``/``excluded_columns``.

    The order stays **that of the source**, not of the configuration — it is part
    of the contract. Columns a strategy needs cannot be dropped from the selection
    — see :func:`_required_columns`.
    """
    selected = set(table_cfg.selected_columns or ())
    excluded = set(table_cfg.excluded_columns or ())
    required = _required_columns(columns, load_settings)

    out = list(columns)
    if table_cfg.only_selected_columns and selected:
        keep = selected | required
        out = [c for c in out if c.name in keep]
    if excluded:
        exclude = excluded - required
        out = [c for c in out if c.name not in exclude]

    if not out:
        raise EntrypointError(
            "No column was left after applying selected_columns/excluded_columns."
        )
    return out


def _with_target_names(columns: List[Any], hash_column: Optional[str]) -> List[Any]:
    """Fill in the target names from `core.naming`. The target names are frozen."""
    import dataclasses

    from dbextractors.core import naming

    targets = naming.target_column_names([c.name for c in columns], hash_column)
    return [
        dataclasses.replace(column, target_name=target)
        for column, target in zip(columns, targets, strict=False)
    ]


def _settings_dict(load_settings, runtime: Optional[dict] = None) -> dict:
    """`LoadSettingsConfig` -> the dict the strategies expect.

    The strategies take a dict on purpose: the keys are a frozen contract and a
    new key has to be addable without touching their signatures.

    ``forced_full_load`` is translated to ``force_reload`` here. It is the second
    half of that runtime variable (the first is the choice of strategy in
    `_strategy_name`) and without it "force" would still skip databases whose
    fingerprint has not changed — exactly what a predecessor comments as "it must
    also defeat the fingerprint".

    The translation is here rather than in the strategy on purpose: the strategies
    are to know nothing about Mage kwargs, and ``force_reload`` already exists as a
    settings key.
    """
    import dataclasses

    from dbextractors.core.config import is_truthy

    out = dict(dataclasses.asdict(load_settings))
    if is_truthy((runtime or {}).get("forced_full_load")):
        out["force_reload"] = True
    return out


def _attach_session_sql(engine, source_dialect, logger: Optional[logging.Logger] = None) -> None:
    """Run ``dialect.session_sql`` on every new connection to the source.

    It cannot be done through ``connect_args`` — ``mysql-connector`` has no
    ``init_command`` and session variables are set only once the connection is
    established. It has to happen on **every** connection, not just the first: the
    engine runs on ``NullPool``, so the connection is discarded after each use.

    A failure is **not swallowed silently**: if the setting does not take, it is
    logged and the run continues — it mitigates a cause, it is not a precondition,
    and the resilience rests on read recovery in `core.reading`.
    """
    commands = tuple(getattr(source_dialect, "session_sql", ()) or ())
    if not commands:
        return

    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _apply(dbapi_conn, _record):  # pragma: no cover - needs a live source
        cur = dbapi_conn.cursor()
        try:
            for command in commands:
                cur.execute(command)
        except Exception as err:
            # `err` comes straight from the driver. None of the four puts the
            # connection string into a statement error today (checked live against
            # mysql-connector 9, psycopg2 2.9, pymssql 2.3 and fdb 2.0), but this
            # listener sits on the connection and a driver that started quoting its
            # own DSN back would leak it once per batch. The commands themselves
            # are dialect constants, so nothing is lost by redacting them too.
            if logger:
                logger.warning(
                    "⚠️ The source session setting failed (%s): %s",
                    secrets.redact(command),
                    secrets.redact(err),
                )
        finally:
            cur.close()


def _target_connection(logger: Optional[logging.Logger] = None, profile: Optional[str] = None):
    """A connection to the target PostgreSQL with autocommit **off**.

    The target is resolved by `core.target_conn`, not by the golden test's
    resolver. `dbextractors.golden.session.resolve_dsn` used to be imported here,
    which meant the production write path rested on a variable named after a
    testing tool — and, worse, knew nothing about the `io_config.yaml` it had
    replaced. In production that showed up as a failure against
    `postgres:54333`, because there `POSTGRES_PORT` means the port published on
    the host, not the port to connect to.
    """
    import psycopg2

    from dbextractors.core.target_conn import (
        TargetConnectionError,
        describe_dsn,
        resolve_target_dsn,
    )

    dsn = resolve_target_dsn(profile=profile)
    if logger:
        logger.warning("🎯 Target: %s", describe_dsn(dsn))
    try:
        conn = psycopg2.connect(dsn)
    except Exception as err:
        # psycopg2 answers a DSN it cannot parse by quoting the offending token
        # back: `invalid dsn: missing "=" after "CANARY"`. When the target password
        # holds a space, the DSN is invalid *because of the password* and the token
        # quoted back is a piece of it — `secrets.dsn_secrets` is what covers that.
        # `from None` for the same reason as at `create_engine` above: a chained
        # original would put the unredacted text back into every traceback dump.
        raise TargetConnectionError(
            f"The connection to the target could not be established: "
            f"{secrets.redact(err, extra=secrets.dsn_secrets(dsn))}"
        ) from None
    conn.autocommit = False
    return conn


def status_row(ctx: LoadContext, result: LoadResult, *, address: Optional[Any] = None) -> dict:
    """The status row of one source. The columns are the predecessors' ``_build_status_df``.

    Three of them are new, and each answers a question that cannot be answered
    from the Mage UI today:

    - ``data_present`` — it ran, but did anything arrive?
    - ``fallback_reason`` — a fall back to a full load leaves no trace in the
      return value today, so there is no way to tell the run took the expensive
      path.
    - ``connection_mode`` — direct or through a tunnel? With ``connection_mode:
      auto`` that is decided at run time, and it is the first thing anyone asks
      when a run is unexpectedly slow.

    Adding a column is safe: the client-side ``test_output`` reads the existing
    columns by name, so a new one does not bother it. Renaming or removing one
    would not be allowed.
    """
    return {
        "table": f"{ctx.target.schema}.{ctx.target.table}",
        "source": ctx.source_label,
        "rows_written": result.rows_written,
        "load_method": result.load_method,
        "success": True,
        "is_incremental": result.is_incremental,
        "data_present": result.data_present,
        "fallback_reason": result.fallback_reason,
        "connection_mode": getattr(address, "mode", None),
    }


def status_frame(
    ctx: LoadContext, result: LoadResult, *, address: Optional[Any] = None
) -> pd.DataFrame:
    """A one-row status DataFrame. A shortcut for a caller with a single source."""
    from dbextractors.core import status as status_module

    return status_module.build_status_df([status_row(ctx, result, address=address)])


__all__ = [
    "run",
    "build_context",
    "resolve_dialect",
    "status_frame",
    "status_row",
    "EntrypointError",
    "SourceExtractionError",
]
