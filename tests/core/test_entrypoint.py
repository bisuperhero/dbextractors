"""Tests for `dbextractors.entrypoint`.

Three things that are hard to spot anywhere but here:

- **the tunnel is opened once per run**, not once per database. With sixteen
  source databases that is the difference between one and sixteen ``ssh``
  handshakes.
- **``forced_full_load`` has two effects** (the route and the fingerprint) and
  it is easy to implement only one of them. "Force" then quietly skips the
  databases that did not change.
- **``forced_full_load`` must not override `full_by_source`.** If it did, every
  database in a multi-source run would `TRUNCATE` the whole target table.

None of these tests touches a database: the strategy and the target connection
are both fakes.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import pytest

from dbextractors import entrypoint
from dbextractors.core.tunnel import TunnelAddress

# --- Scaffolding ------------------------------------------------------------


def _config(**overrides: Any) -> dict:
    cfg: dict = {
        "TABLE": {"source_name": "t", "output_name": "target", "output_schema": "s"},
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d", "host": "h"},
        "LOAD_SETTINGS": {"load_method": "full", "primary_column": "id"},
    }
    for key, value in overrides.items():
        if key in cfg and isinstance(value, dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


class _FakeStrategy:
    """A strategy that only records the context it was called with."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def run(self, ctx):
        from dbextractors.core.strategies.base import LoadResult

        self.calls.append(ctx)
        return LoadResult(rows_read=1, rows_written=1, load_method="full")


def _fake_ctx(schema: str = "s", table: str = "target", source_label=None):
    return SimpleNamespace(
        target=SimpleNamespace(schema=schema, table=table),
        target_conn=SimpleNamespace(close=lambda: None),
        source_label=source_label,
        # Right after building the context `_run_one` prints the columns, and
        # under DEBUG the type maps too, so the fake context needs them as well.
        log=lambda *a, **k: None,
        target_names=["id"],
        overwrite_types={"id": "TEXT"},
        orig_type_map={"id": "varchar"},
        surrogate=None,
        where=None,
        debug=False,
    )


@pytest.fixture()
def no_database(monkeypatch):
    """Swap the strategy, the context and the tunnel for fakes, and return the
    record of the calls made."""
    record: dict = {"tunnels": [], "contexts": []}
    strategy = _FakeStrategy()

    monkeypatch.setattr(
        "dbextractors.core.strategies.base.resolve_strategy",
        lambda _name, _settings=None: strategy,
        raising=True,
    )

    def _fake_build_context(parsed, dialect, **kwargs):
        record["contexts"].append(kwargs)
        return _fake_ctx(
            parsed.table.output_schema,
            parsed.table.output_table,
            source_label=kwargs.get("source_label"),
        )

    monkeypatch.setattr(entrypoint, "build_context", _fake_build_context)

    from contextlib import contextmanager

    @contextmanager
    def _fake_open_tunnel(source_db, mode=None, *, probe=None, show_debug=False):
        record["tunnels"].append({"source_db": source_db, "mode": mode, "probe": probe})
        yield TunnelAddress(host="127.0.0.1", port=54321, mode="ssh")

    monkeypatch.setattr("dbextractors.core.tunnel.open_tunnel", _fake_open_tunnel)

    record["strategy"] = strategy
    return record


# --- Runtime variables ------------------------------------------------------


def test_only_known_runtime_variables_are_taken() -> None:
    """Mage passes dozens of them; only the ones the package knows may reach a
    strategy."""
    out = entrypoint._runtime_vars(
        {
            "forced_full_load": True,
            "incremental_lookback_hours": 6,
            "execution_date": "2026-08-13",
            "block_uuid": "abc",
            "logger": object(),
        }
    )
    assert out == {"forced_full_load": True, "incremental_lookback_hours": 6}


def test_a_missing_key_is_not_filled_in_as_none() -> None:
    """The distinction `compute_incremental_cutoff` rests on.

    ``None`` means "the variable was passed empty, take LOAD_SETTINGS"; a
    missing key means "it was not passed at all". Filling it in as ``None``
    would merge the two and the configured lookback would stop working.
    """
    assert entrypoint._runtime_vars({}) == {}
    assert entrypoint._runtime_vars({"forced_full_load": None}) == {"forced_full_load": None}


# --- forced_full_load: choosing the route -----------------------------------


@pytest.mark.parametrize(
    "original", ["incremental", "parent_incremental", "id_watermark", "hash_diff"]
)
def test_force_rewrites_incremental_routes_to_full(original) -> None:
    assert entrypoint._strategy_name(original, {"forced_full_load": True}) == "full"


def test_force_leaves_full_by_source_alone(caplog) -> None:
    """**The most important test in this file.**

    `full_by_source` replaces only one database's slice of the target. Were
    force to rewrite it to `full`, every database in a multi-source run would
    `TRUNCATE` the whole table and only the last one would survive in the
    target. The predecessor's MSSQL extractor behaves the same way:
    `forced_full_load` does not change the route there at all.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        result = entrypoint._strategy_name(
            "full_by_source", {"forced_full_load": True}, logger=logging.getLogger("t")
        )
    assert result == "full_by_source"
    assert "stays as it is" in caplog.text


def test_without_force_nothing_changes() -> None:
    assert entrypoint._strategy_name("incremental", {}) == "incremental"
    assert entrypoint._strategy_name("incremental", {"forced_full_load": False}) == "incremental"


@pytest.mark.parametrize("value", ["True", "true", "1", 1, "ano", "yes"])
def test_force_accepts_textual_values(value) -> None:
    """Mage passes runtime variables as strings, not as bools."""
    assert entrypoint._strategy_name("incremental", {"forced_full_load": value}) == "full"


@pytest.mark.parametrize("value", ["False", "false", "0", 0, "", None])
def test_indeterminate_values_do_not_switch_force_on(value) -> None:
    assert entrypoint._strategy_name("incremental", {"forced_full_load": value}) == "incremental"


# --- forced_full_load: silencing the fingerprint ----------------------------


def test_force_silences_the_fingerprint() -> None:
    """The other half of that variable. Without it "force" would skip the
    databases that did not change."""
    from dbextractors.core import config as config_module

    parsed = config_module.parse(_config(LOAD_SETTINGS={"load_method": "full_by_source"}))
    settings = entrypoint._settings_dict(parsed.load_settings, {"forced_full_load": "true"})
    assert settings["force_reload"] is True


def test_without_force_no_force_reload_is_added() -> None:
    from dbextractors.core import config as config_module

    parsed = config_module.parse(_config())
    assert "force_reload" not in entrypoint._settings_dict(parsed.load_settings, {})
    assert "force_reload" not in entrypoint._settings_dict(parsed.load_settings)


def test_both_halves_of_force_hold_together(no_database) -> None:
    """A pass through the whole of `run`: the route stays, the fingerprint is
    silenced."""
    entrypoint.run(
        _config(
            LOAD_SETTINGS={"load_method": "full_by_source"},
            DATABASES=["a", "b"],
        ),
        "mysql",
        forced_full_load="true",
    )
    contexts = no_database["contexts"]
    assert len(contexts) == 2
    assert all(c["runtime"] == {"forced_full_load": "true"} for c in contexts)


# --- Tunnel -----------------------------------------------------------------


def test_the_tunnel_is_opened_once_for_every_database(no_database) -> None:
    """Otherwise sixteen source databases would mean sixteen ssh handshakes."""
    entrypoint.run(_config(DATABASES=["a", "b", "c"]), "mysql")

    assert len(no_database["tunnels"]) == 1
    assert len(no_database["contexts"]) == 3


def test_the_tunnel_address_reaches_the_context(no_database) -> None:
    entrypoint.run(_config(), "mysql")
    address = no_database["contexts"][0]["address"]
    assert (address.host, address.port, address.mode) == ("127.0.0.1", 54321, "ssh")


def test_the_probe_comes_from_the_dialect(no_database) -> None:
    """MSSQL probes three times, the others once — that is dialect knowledge."""
    from dbextractors.dialects.mssql import MSSQLDialect

    entrypoint.run(_config(), "mssql")
    probe = no_database["tunnels"][0]["probe"]
    assert probe.__self__.__class__ is MSSQLDialect


def test_connection_mode_from_the_configuration_is_passed_on(no_database) -> None:
    entrypoint.run(_config(SOURCE_DB={"connection_mode": "direct"}), "mysql")
    assert no_database["tunnels"][0]["mode"] == "direct"


def test_no_tunnel_is_opened_when_no_database_is_selected(no_database) -> None:
    """An empty ``DATABASES`` is a legitimate state — and no reason to build a
    tunnel."""
    frame = entrypoint.run(_config(DATABASES=[]), "mysql")

    assert no_database["tunnels"] == []
    assert frame["load_method"].tolist() == ["nothing_selected"]
    assert bool(frame["success"].iloc[0]) is True


# --- Tunnel parameters ------------------------------------------------------


def test_the_port_is_filled_in_from_the_dialect_default() -> None:
    """`core.config` leaves it empty, but ``direct`` without a port is an
    error."""
    from dbextractors.core import config as config_module
    from dbextractors.dialects.mysql import MySQLDialect

    parsed = config_module.parse(_config())
    assert entrypoint._tunnel_params(parsed, MySQLDialect())["port"] == 3306


def test_a_configured_port_takes_precedence() -> None:
    from dbextractors.core import config as config_module
    from dbextractors.dialects.mysql import MySQLDialect

    parsed = config_module.parse(_config(SOURCE_DB={"port": 3307}))
    assert entrypoint._tunnel_params(parsed, MySQLDialect())["port"] == 3307


def test_ssh_details_reach_the_tunnel() -> None:
    from dbextractors.core import config as config_module
    from dbextractors.dialects.mysql import MySQLDialect

    parsed = config_module.parse(
        _config(
            SOURCE_DB={
                "ssh_address_or_host": "gateway",
                "ssh_username": "robot",
                "ssh_pkey": "/tmp/k",
            }
        )
    )
    params = entrypoint._tunnel_params(parsed, MySQLDialect())
    assert params["ssh_address_or_host"] == "gateway"
    assert params["ssh_username"] == "robot"
    assert params["ssh_pkey"] == "/tmp/k"


# --- Status frame -----------------------------------------------------------


def test_the_status_carries_the_mode_actually_used() -> None:
    """Under ``auto`` this is decided at run time — and it is the first question
    asked when a run is slow."""
    from dbextractors.core.strategies.base import LoadResult

    frame = entrypoint.status_frame(
        _fake_ctx(),
        LoadResult(rows_read=5, rows_written=5, load_method="full"),
        address=TunnelAddress(host="127.0.0.1", port=1, mode="ssh"),
    )
    assert frame["connection_mode"].tolist() == ["ssh"]


def test_the_status_survives_a_missing_address() -> None:
    from dbextractors.core.strategies.base import LoadResult

    frame = entrypoint.status_frame(_fake_ctx(), LoadResult(1, 1, "full"))
    assert frame["connection_mode"].isna().all()


def test_an_empty_run_has_the_same_columns_as_a_normal_one() -> None:
    """The caller's ``test_output`` reads the columns by name — none of them may
    be missing."""
    from dbextractors.core import config as config_module
    from dbextractors.core.strategies.base import LoadResult

    parsed = config_module.parse(_config())
    empty = entrypoint._nothing_selected_frame(parsed)
    normal = entrypoint.status_frame(_fake_ctx(), LoadResult(1, 1, "full"))
    assert list(empty.columns) == list(normal.columns)


# --- Context ----------------------------------------------------------------


def test_neither_a_host_nor_a_tunnel_is_an_error() -> None:
    """This message used to say the SSH tunnel was not wired up yet; now it has
    to give advice that actually helps."""
    from dbextractors.core import config as config_module
    from dbextractors.dialects.mysql import MySQLDialect

    parsed = config_module.parse(
        _config(SOURCE_DB={"host": "", "ssh_address_or_host": "b", "ssh_username": "u"})
    )
    with pytest.raises(entrypoint.EntrypointError, match="connection_mode"):
        entrypoint.build_context(parsed, MySQLDialect())


def test_an_unusable_connection_string_keeps_the_password_out_of_the_traceback(
    monkeypatch,
) -> None:
    """Redacting the message and chaining the original exception is theatre.

    When a URL is unusable, SQLAlchemy raises with the whole of it — password
    included. The message here is rewrapped through `core.secrets`, but for a
    while it was raised ``from err``, so the unredacted original stayed on
    ``__cause__`` and anything printing the full traceback ("The above exception
    was the direct cause…") put the password in the Mage log anyway. More people
    read that log than read the pipeline configuration.

    So the assertion is on the **rendered traceback**, not on the message: the
    message alone was already clean while the leak was open.
    """
    import traceback
    import urllib.parse

    import sqlalchemy

    from dbextractors.core import config as config_module
    from dbextractors.dialects.mysql import MySQLDialect

    password = "s3cr3t@p/ss"

    def refuse(url, **kwargs):
        raise ValueError(f"Could not parse SQLAlchemy URL from string {url!r}")

    monkeypatch.setattr(sqlalchemy, "create_engine", refuse)
    parsed = config_module.parse(_config(SOURCE_DB={"password": password}))

    with pytest.raises(entrypoint.EntrypointError) as err:
        entrypoint.build_context(parsed, MySQLDialect())

    rendered = "".join(
        traceback.format_exception(type(err.value), err.value, err.value.__traceback__)
    )
    # Both spellings: the dialect URL-encodes the password, and the encoded form
    # is still the password.
    assert password not in rendered
    assert urllib.parse.quote_plus(password) not in rendered
    assert err.value.__cause__ is None
    assert err.value.__suppress_context__, "the original would be printed after 'During handling…'"
    # Without the password, but not without the facts — an error still has to be
    # reportable in full.
    assert "h" in str(err.value)


# --- Session settings on the source -----------------------------------------
#
# `session_sql` exists because of two long extractions that died in production
# with "2013 Lost connection to MySQL server during query": the server drops a
# client that has not read for `net_write_timeout` seconds, and unbuffered batch
# reading spends exactly that gap converting a batch. It had no test.
#
# The engine below is real SQLite on disk with `NullPool` — the pool production
# uses — because the claim is about connections, not about calls: the setting
# has to be applied to *every* new connection, and a pooled or in-memory engine
# would hide a listener that only ever fires once. `PRAGMA user_version` is the
# SQLite equivalent of a session variable that can be read back afterwards.


class _DialectWithSession:
    session_sql = ("PRAGMA user_version = 7",)


def _sqlite_engine(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    return create_engine(f"sqlite:///{tmp_path}/source.db", poolclass=NullPool)


def _user_version(engine) -> int:
    from sqlalchemy import text

    with engine.connect() as con:
        return int(con.execute(text("PRAGMA user_version")).scalar())


def test_the_session_settings_are_applied_when_the_source_is_connected(tmp_path) -> None:
    engine = _sqlite_engine(tmp_path)
    entrypoint._attach_session_sql(engine, _DialectWithSession())

    assert _user_version(engine) == 7


def test_the_session_settings_are_applied_to_every_connection(tmp_path) -> None:
    """Not just the first one.

    The engine runs on `NullPool`, so the connection is thrown away after each
    use and a fresh one starts from the server's defaults again. A listener that
    fired once would leave every batch after the first exposed to the timeout
    that killed those two extractions.
    """
    from sqlalchemy import text

    engine = _sqlite_engine(tmp_path)
    entrypoint._attach_session_sql(engine, _DialectWithSession())

    assert _user_version(engine) == 7
    with engine.connect() as con:
        con.execute(text("PRAGMA user_version = 0"))
    assert _user_version(engine) == 7, "the second connection started from the default again"


def test_a_dialect_without_session_settings_changes_nothing(tmp_path) -> None:
    """Three of the four dialects have none, and must not inherit MySQL's."""

    class _NoSession:
        session_sql = ()

    engine = _sqlite_engine(tmp_path)
    entrypoint._attach_session_sql(engine, _NoSession())

    assert _user_version(engine) == 0


def test_a_session_setting_that_fails_is_logged_and_the_run_goes_on(tmp_path, caplog) -> None:
    """It mitigates a cause; it is not a precondition.

    A server that refuses the setting (an older version, a restricted account)
    must not stop the extraction — the resilience rests on read recovery in
    `core.reading`. But it must not vanish either: this is precisely the
    "swallowed exception with nothing in the log" the project rules forbid.
    """

    class _BadSession:
        session_sql = ("THIS IS NOT SQL",)

    logger = logging.getLogger("test_session_sql")
    engine = _sqlite_engine(tmp_path)
    entrypoint._attach_session_sql(engine, _BadSession(), logger)

    with caplog.at_level(logging.WARNING, logger="test_session_sql"):
        assert _user_version(engine) == 0, "the connection still works"

    assert any("THIS IS NOT SQL" in record.getMessage() for record in caplog.records), caplog.text


# --- Isolating failures between databases -----------------------------------


class _FailsOnOne:
    """A strategy that fails on one particular database."""

    def __init__(self, fails_on: str) -> None:
        self.fails_on = fails_on
        self.seen: List[Any] = []

    def run(self, ctx):
        from dbextractors.core.strategies.base import LoadResult

        self.seen.append(ctx.source_label)
        if ctx.source_label == self.fails_on:
            raise RuntimeError("source unavailable")
        return LoadResult(rows_read=1, rows_written=1, load_method="full_by_source")


def test_one_database_failing_does_not_stop_the_others(no_database, monkeypatch) -> None:
    """The databases are independent — the second one failing must not rob the
    third of its chance.

    The predecessor did the same: it caught per database, carried on, and raised
    a summary exception at the end.
    """
    strategy = _FailsOnOne("b")
    monkeypatch.setattr(
        "dbextractors.core.strategies.base.resolve_strategy",
        lambda _n, _s=None: strategy,
        raising=True,
    )

    with pytest.raises(entrypoint.SourceExtractionError) as err:
        entrypoint.run(_config(DATABASES=["a", "b", "c"]), "mysql")

    assert strategy.seen == ["a", "b", "c"], "after failing on 'b' it should have gone on to 'c'"
    assert "b: source unavailable" in str(err.value)
    assert "1 of 3" in str(err.value)


def test_a_failure_is_not_a_green_run(no_database, monkeypatch) -> None:
    """This must not turn into a returned frame — the block would finish green.

    That is exactly why `SourceExtractionError` came about: per-database
    failures used to be collected into the status DataFrame and returned, which
    made the block finish green.
    """

    class _AlwaysFails:
        def run(self, ctx):
            raise RuntimeError("source unavailable")

    monkeypatch.setattr(
        "dbextractors.core.strategies.base.resolve_strategy",
        lambda _n, _s=None: _AlwaysFails(),
        raising=True,
    )
    with pytest.raises(entrypoint.SourceExtractionError):
        entrypoint.run(_config(DATABASES=["a"]), "mysql")


def test_a_successful_run_has_a_source_on_every_row(no_database) -> None:
    """The caller's ``test_output`` lists the failed sources by the ``source``
    column."""
    frame = entrypoint.run(_config(DATABASES=["a", "b"]), "mysql")
    assert frame["source"].tolist() == ["a", "b"]
    assert frame["success"].tolist() == [True, True]


# --- Dialect name -----------------------------------------------------------
#
# `run(dialect=…)` is called inside the Mage block, so a typo only shows up at
# run time — for one deployment's 21 pipelines that would mean 21 failed runs.
# `postgresql` is the more common spelling than `postgres`: it is what the
# SQLAlchemy dialect is called, and what the predecessor's extractor file was
# named after.


def test_postgresql_is_an_alias_for_postgres() -> None:
    from dbextractors.dialects.postgres import PostgresDialect

    assert isinstance(entrypoint.resolve_dialect("postgresql"), PostgresDialect)
    assert isinstance(entrypoint.resolve_dialect("PostgreSQL"), PostgresDialect)


def test_an_unknown_dialect_fails_and_lists_the_known_names() -> None:
    """An alias must not turn a typo into a silent choice of another dialect."""
    with pytest.raises(entrypoint.EntrypointError) as err:
        entrypoint.resolve_dialect("postgress")
    assert "firebird" in str(err.value), "the message should list what is accepted"
