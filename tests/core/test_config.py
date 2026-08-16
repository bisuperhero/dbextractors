"""Tests for `dbextractors.core.config`.

Cover: the modern config (sections), a purely legacy config (top-level keys),
a mixed config (checks the exact order of fallbacks), missing required keys,
an unknown key (a warning only), ``connection_mode``, a characterisation of
`is_truthy` against the oracle recorded from the predecessors, a full roundtrip
of every contract key, and the fallbacks/aliases exercised in **both**
directions — the direction the earlier suite never took.

Nothing here touches the network or production — ``parse()``/``validate()`` are
pure functions over a dict.
"""

from __future__ import annotations

import logging

import pytest

import reference_oracle as ro
from dbextractors.core import config

# --- Modern config (sections TABLE / LOAD_SETTINGS / SOURCE_DB) -----------


def _modern_config() -> dict:
    return {
        "TABLE": {
            "source_name": "orders",
            "source_schema": "sales",
            "output_schema": "raw",
            "output_name": "orders",
            "selected_columns": ["id", "amount"],
            "only_selected_columns": True,
            "excluded_columns": ["secret"],
            "where_clause": "id > 0",
            "empty_rows_ok": True,
        },
        "LOAD_SETTINGS": {
            "load_method": "hash",
            "primary_column": "id",
            "batch_size": 5000,
            "hash_column": "row_hash",
            "hash_include_columns": ["amount"],
            "days_back": 7,
            "pagination_mode": "keyset",
            "compute_row_hash": False,
        },
        "SOURCE_DB": {
            "user": "u",
            "password": "p",
            "host": "db.internal",
            "port": 5432,
            "database": "erp",
            "charset": "utf8mb4",
        },
        "connection_mode": "direct",
    }


def test_modern_config_is_parsed_correctly() -> None:
    parsed = config.parse(_modern_config())

    assert parsed.table.source_name == "orders"
    assert parsed.table.source_schema == "sales"
    assert parsed.table.output_schema == "raw"
    assert parsed.table.output_table == "orders"
    assert parsed.table.selected_columns == ("id", "amount")
    assert parsed.table.only_selected_columns is True
    assert parsed.table.excluded_columns == ("secret",)
    assert parsed.table.where_clause == "id > 0"
    assert parsed.table.empty_rows_ok is True

    assert parsed.load_settings.load_method == "hash"
    assert parsed.load_settings.primary_column == "id"
    assert parsed.load_settings.batch_size == 5000
    assert parsed.load_settings.hash_column == "row_hash"
    assert parsed.load_settings.hash_include_columns == ("amount",)
    assert parsed.load_settings.days_back == 7
    assert parsed.load_settings.pagination_mode == "keyset"
    assert parsed.load_settings.compute_row_hash is False

    assert parsed.source_db.user == "u"
    assert parsed.source_db.password == "p"
    assert parsed.source_db.host == "db.internal"
    assert parsed.source_db.port == 5432
    assert parsed.source_db.database == "erp"
    assert parsed.source_db.charset == "utf8mb4"

    assert parsed.connection_mode == "direct"


def test_modern_config_has_safe_load_settings_defaults() -> None:
    """Defaults consistent across every predecessor — days_back=14,
    pagination_mode='auto', load_method='full'.

    ``compute_row_hash`` is **not** among them: it has three states and a silent
    configuration must stay ``None``. The default is supplied by the strategy,
    because `full` and `full_by_source` differ — were the parser to fill it in,
    the strategy would never get its say.
    """
    minimal = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
    }
    parsed = config.parse(minimal)

    assert parsed.load_settings.load_method == "full"
    assert parsed.load_settings.days_back == 14
    assert parsed.load_settings.pagination_mode == "auto"
    assert parsed.load_settings.compute_row_hash is None
    assert parsed.load_settings.ignore_hash is False
    # Off unless a pipeline asks for it: turning it on by default would fail runs
    # that are green today, across ~670 tables. See the field's own comment.
    assert parsed.load_settings.strict_integer_precision is False
    # Off for the same reason: on, it changes what an MSSQL N-typed column holds
    # in the target, which is a reload of that table.
    assert parsed.load_settings.convert_nchar_to_varchar is False
    # `None`, not the strategy's constants: the parser must not know
    # `parent_id`/`id`, and `None` is what makes the strategy's own defaults win
    # for every configuration that stays silent about them.
    assert parsed.load_settings.incremental_parent_key_column is None
    assert parsed.load_settings.incremental_parent_id_column is None
    assert parsed.load_settings.multi_source is None
    assert parsed.connection_mode == "auto"
    assert parsed.debug is False


@pytest.mark.parametrize(
    "value,expected", [(True, True), ("ano", True), (False, False), ("ne", False)]
)
def test_explicit_compute_row_hash_is_carried_through(value, expected) -> None:
    """The three-state handling must swallow silence only, never an explicit value."""
    parsed = config.parse(
        {
            "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
            "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
            "LOAD_SETTINGS": {"compute_row_hash": value},
        }
    )
    assert parsed.load_settings.compute_row_hash is expected


# --- Purely legacy config (top-level keys only) ---------------------------


def _legacy_config() -> dict:
    return {
        "TABLE_NAME": "orders",
        "OUTPUT_SCHEMA": "raw",
        "OUTPUT_TABLE": "orders",
        "CONNECTION_PARAMS": {
            "user": "u",
            "password": "p",
            "database": "erp",
            "port": 5432,
        },
        "SSH_PARAMS": {
            "ssh_address_or_host": "bastion.example.com",
            "ssh_username": "svc",
            "ssh_pkey": "/keys/id_rsa",
        },
        "PRIVATE_HOST": "db.internal",
        "BATCH_SIZE": 2000,
        "EMPTY_ROWS_OK": True,
        "DEBUG": True,
    }


def test_purely_legacy_config_is_parsed_correctly() -> None:
    parsed = config.parse(_legacy_config())

    assert parsed.table.source_name == "orders"
    assert parsed.table.output_schema == "raw"
    assert parsed.table.output_table == "orders"
    assert parsed.table.empty_rows_ok is True

    assert parsed.source_db.user == "u"
    assert parsed.source_db.password == "p"
    assert parsed.source_db.database == "erp"
    assert parsed.source_db.port == 5432
    assert parsed.source_db.host == "db.internal"
    assert parsed.source_db.ssh_address_or_host == "bastion.example.com"
    assert parsed.source_db.ssh_username == "svc"
    assert parsed.source_db.ssh_pkey == "/keys/id_rsa"

    assert parsed.load_settings.batch_size == 2000
    assert parsed.debug is True
    # No predecessor knew connection_mode — the legacy default is 'auto'.
    assert parsed.connection_mode == "auto"


# --- Mixed config: a legacy and a modern key mean the same thing ----------


def test_mixed_config_modern_wins_where_or_is_used() -> None:
    """`source_name`/`output_schema`/`output_table` are picked with `or`, the
    way the predecessor did it — the TABLE section takes precedence over the
    top-level legacy key."""
    mixed = {
        "TABLE": {"source_name": "modern_name", "output_schema": "modern_schema"},
        "TABLE_NAME": "legacy_name",
        "OUTPUT_SCHEMA": "legacy_schema",
        "OUTPUT_TABLE": "legacy_table",
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
    }
    parsed = config.parse(mixed)

    assert parsed.table.source_name == "modern_name"
    assert parsed.table.output_schema == "modern_schema"
    # TABLE carries neither output_name nor output_table, so even a modern
    # config falls back to the legacy OUTPUT_TABLE - part of the fallback
    # chain, not a bug.
    assert parsed.table.output_table == "legacy_table"


def test_mixed_config_falls_back_to_connection_params_per_key() -> None:
    """`SOURCE_DB.get(key, CONNECTION_PARAMS.get(key))` is per key, not per
    section: when SOURCE_DB exists but a particular key is absent from it, the
    fallback applies to that key alone."""
    mixed = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {"user": "modern_user"},
        "CONNECTION_PARAMS": {
            "user": "legacy_user",
            "password": "legacy_pw",
            "database": "legacy_db",
        },
    }
    parsed = config.parse(mixed)

    assert parsed.source_db.user == "modern_user"
    assert parsed.source_db.password == "legacy_pw"
    assert parsed.source_db.database == "legacy_db"


def test_mixed_config_load_settings_beats_top_level_batch_size() -> None:
    mixed = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
        "LOAD_SETTINGS": {"batch_size": 999},
        "BATCH_SIZE": 111,
    }
    parsed = config.parse(mixed)

    assert parsed.load_settings.batch_size == 999


def test_mixed_config_empty_source_db_section_falls_back_to_legacy() -> None:
    """`config.get('SOURCE_DB') or config.get('CONNECTION_PARAMS', {})` — an
    empty dict is falsy, so the fallback takes CONNECTION_PARAMS as a whole."""
    mixed = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {},
        "CONNECTION_PARAMS": {"user": "u", "password": "p", "database": "d"},
    }
    parsed = config.parse(mixed)

    assert parsed.source_db.user == "u"
    assert parsed.source_db.password == "p"
    assert parsed.source_db.database == "d"


# --- Missing required keys -------------------------------------------------


@pytest.mark.parametrize(
    "remove",
    ["source_name", "output_schema", "output_name"],
)
def test_missing_required_table_key_fails(remove: str) -> None:
    cfg = _modern_config()
    cfg["TABLE"].pop(remove, None)
    if remove == "output_name":
        # output_table is a legitimate key too, so both have to go; otherwise
        # the fallback chain quietly wins.
        cfg["TABLE"].pop("output_table", None)

    with pytest.raises(config.ConfigError, match="is missing"):
        config.parse(cfg)


@pytest.mark.parametrize("remove", ["user", "password", "database"])
def test_missing_credential_fails(remove: str) -> None:
    cfg = _modern_config()
    cfg["SOURCE_DB"].pop(remove, None)

    with pytest.raises(config.ConfigError, match="is missing"):
        config.parse(cfg)


def test_databases_stands_in_for_a_missing_source_db_database() -> None:
    """A multi-source configuration carries no database name in ``SOURCE_DB``.

    Such deployments all look alike: ``SOURCE_DB`` holds only host, port and
    credentials, and the database names arrive as a ``DATABASES`` list the
    extractor switches between. Requiring ``database`` unconditionally would
    have blocked every one of them.
    """
    cfg = _modern_config()
    cfg["SOURCE_DB"].pop("database", None)
    cfg["DATABASES"] = ["erp_2025", "erp_2024"]

    parsed = config.parse(cfg)

    assert parsed.databases == ("erp_2025", "erp_2024")
    assert not parsed.source_db.database


def test_neither_database_nor_databases_fails() -> None:
    """An empty list must not pass as a substitute — there would be nothing to
    connect to."""
    cfg = _modern_config()
    cfg["SOURCE_DB"].pop("database", None)
    cfg["DATABASES"] = []

    with pytest.raises(config.ConfigError, match="DATABASES"):
        config.parse(cfg)


def test_missing_key_message_lists_every_fault_at_once() -> None:
    cfg: dict = {"TABLE": {}, "SOURCE_DB": {}}

    with pytest.raises(config.ConfigError) as excinfo:
        config.parse(cfg)

    message = str(excinfo.value)
    assert "source_name" in message
    assert "output_schema" in message
    assert "user" in message
    assert "password" in message
    assert "database" in message


def test_config_must_be_a_dict() -> None:
    with pytest.raises(config.ConfigError, match="dict"):
        config.parse(["not", "a", "dict"])  # type: ignore[arg-type]


# --- Unknown key: a warning, not a failure ---------------------------------


def test_unknown_top_level_key_is_only_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _modern_config()
    cfg["SOME_NEW_KEY"] = "value"

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        parsed = config.parse(cfg)

    assert parsed.table.source_name == "orders"
    assert any("SOME_NEW_KEY" in record.message for record in caplog.records)


def test_unknown_key_in_a_section_is_only_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _modern_config()
    cfg["LOAD_SETTINGS"]["some_new_load_setting"] = True
    cfg["TABLE"]["some_new_table_key"] = "x"
    cfg["SOURCE_DB"]["some_new_source_db_key"] = "y"

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        parsed = config.parse(cfg)

    assert parsed.table.source_name == "orders"
    messages = [record.message for record in caplog.records]
    assert any("some_new_load_setting" in m for m in messages)
    assert any("some_new_table_key" in m for m in messages)
    assert any("some_new_source_db_key" in m for m in messages)


def test_connection_mode_is_not_reported_as_unknown_in_source_db(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _modern_config()
    del cfg["connection_mode"]
    cfg["SOURCE_DB"]["connection_mode"] = "ssh"
    cfg["SOURCE_DB"]["ssh_address_or_host"] = "bastion"
    cfg["SOURCE_DB"]["ssh_username"] = "svc"
    cfg["SOURCE_DB"]["ssh_pkey"] = "/keys/id_rsa"

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        parsed = config.parse(cfg)

    assert parsed.connection_mode == "ssh"
    assert not any("connection_mode" in record.message for record in caplog.records)


# --- ssh_host_key_checking ---------------------------------------------------


def test_ssh_host_key_checking_defaults_to_off() -> None:
    """The default must stay the inherited behaviour — otherwise deployments
    that today run against a host whose fingerprint nobody ever recorded would
    break."""
    parsed = config.parse(_modern_config())
    assert parsed.source_db.ssh_host_key_checking == "off"


@pytest.mark.parametrize("value", ["off", "accept-new", "strict"])
def test_ssh_host_key_checking_accepts_valid_values(value: str) -> None:
    cfg = _modern_config()
    cfg["SOURCE_DB"]["ssh_host_key_checking"] = value

    parsed = config.parse(cfg)
    assert parsed.source_db.ssh_host_key_checking == value


@pytest.mark.parametrize("value", ["STRICT", " Accept-New "])
def test_ssh_host_key_checking_normalises_spelling(value: str) -> None:
    cfg = _modern_config()
    cfg["SOURCE_DB"]["ssh_host_key_checking"] = value

    parsed = config.parse(cfg)
    assert parsed.source_db.ssh_host_key_checking == value.strip().lower()


def test_ssh_host_key_checking_invalid_value_fails() -> None:
    cfg = _modern_config()
    cfg["SOURCE_DB"]["ssh_host_key_checking"] = "yes-please"

    with pytest.raises(config.ConfigError, match="ssh_host_key_checking"):
        config.parse(cfg)


def test_ssh_host_key_checking_empty_value_means_not_filled_in() -> None:
    """An empty string in YAML means "not filled in", not "the empty mode"."""
    cfg = _modern_config()
    cfg["SOURCE_DB"]["ssh_host_key_checking"] = ""

    parsed = config.parse(cfg)
    assert parsed.source_db.ssh_host_key_checking == "off"


def test_ssh_host_key_checking_from_legacy_ssh_params() -> None:
    cfg = _modern_config()
    cfg["SSH_PARAMS"] = {"ssh_host_key_checking": "strict"}

    parsed = config.parse(cfg)
    assert parsed.source_db.ssh_host_key_checking == "strict"


def test_ssh_host_key_checking_is_not_reported_as_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _modern_config()
    cfg["SOURCE_DB"]["ssh_host_key_checking"] = "strict"

    with caplog.at_level(logging.WARNING):
        config.parse(cfg)

    assert not any("ssh_host_key_checking" in record.message for record in caplog.records)


# --- connection_mode ---------------------------------------------------------


def test_connection_mode_defaults_to_auto() -> None:
    cfg = _modern_config()
    del cfg["connection_mode"]
    parsed = config.parse(cfg)
    assert parsed.connection_mode == "auto"


@pytest.mark.parametrize("mode", ["direct", "ssh", "auto"])
def test_connection_mode_accepts_valid_values(mode: str) -> None:
    cfg = _modern_config()
    cfg["connection_mode"] = mode
    if mode == "ssh":
        cfg["SOURCE_DB"]["ssh_address_or_host"] = "bastion"
        cfg["SOURCE_DB"]["ssh_username"] = "svc"
        cfg["SOURCE_DB"]["ssh_pkey"] = "/keys/id_rsa"

    parsed = config.parse(cfg)
    assert parsed.connection_mode == mode


def test_connection_mode_invalid_value_fails() -> None:
    cfg = _modern_config()
    cfg["connection_mode"] = "vpn"

    with pytest.raises(config.ConfigError, match="connection_mode"):
        config.parse(cfg)


def test_connection_mode_direct_requires_host() -> None:
    cfg = _modern_config()
    cfg["connection_mode"] = "direct"
    del cfg["SOURCE_DB"]["host"]

    with pytest.raises(config.ConfigError, match="host"):
        config.parse(cfg)


def test_connection_mode_ssh_requires_ssh_credentials() -> None:
    cfg = _modern_config()
    cfg["connection_mode"] = "ssh"
    # Without ssh_address_or_host/ssh_username/ssh_pkey there is nothing for
    # SSH to work with.

    with pytest.raises(config.ConfigError, match="ssh_"):
        config.parse(cfg)


def test_connection_mode_auto_requires_neither_direct_nor_ssh_credentials() -> None:
    """`auto` is the inherited probe-and-fall-back behaviour — validation has
    nothing to enforce here, because the route is picked at run time."""
    cfg = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
        "connection_mode": "auto",
    }
    parsed = config.parse(cfg)
    assert parsed.connection_mode == "auto"


# --- ParsedConfig is a typed dataclass, not a dict --------------------------


def test_parsed_config_is_a_frozen_dataclass() -> None:
    parsed = config.parse(_modern_config())

    with pytest.raises(AttributeError):
        parsed.connection_mode = "ssh"  # type: ignore[misc]


# --- Characterisation of is_truthy against the oracle -----------------------

_IS_TRUTHY_INPUTS = [
    True,
    False,
    "true",
    "false",
    "yes",
    "no",
    "1",
    "0",
    "",
    None,
    0,
    1,
    2,
    -1,
    [],
    {},
    "TRUE",
    " Yes ",
    "t",
    "ano",
    "a",
    "Ano",
    "NE",
    "nope",
    3.14,
    "  ",
]


@pytest.mark.parametrize("value", _IS_TRUTHY_INPUTS, ids=repr)
def test_is_truthy_against_the_oracle(value: object) -> None:
    old_is_truthy = ro.get("C-mssql", "is_truthy")
    assert config.is_truthy(value) == old_is_truthy(value)


# --- resolve() ---------------------------------------------------------------


def test_resolve_empty_name_is_none() -> None:
    assert config.resolve(None) is None
    assert config.resolve("") is None


def test_resolve_without_a_map_returns_the_name_unchanged() -> None:
    assert config.resolve("Price", None) == "Price"
    assert config.resolve("Price", {}) == "Price"


def test_resolve_case_insensitive_lookup() -> None:
    actual = {"price": "Price_Amount"}
    assert config.resolve("PRICE", actual) == "Price_Amount"


def test_resolve_logs_a_warning_for_a_column_it_cannot_find(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        result = config.resolve("does_not_exist", {"price": "Price_Amount"})

    assert result is None
    assert any("does_not_exist" in record.message for record in caplog.records)


# --- Roundtrip of the whole frozen contract ---------------------------------
#
# Every key of every section, each with a value distinguishable from every other
# key and from the field's default. A typo in any `get()` inside `parse()` makes
# the value fall back to a default and one of the asserts below goes red — a
# mutation of exactly that shape (`incremental_date_colum`) survived the earlier
# suite, because no test ever set the key.


def _full_config() -> dict:
    """A config that uses **all** keys of TABLE, LOAD_SETTINGS and SOURCE_DB.

    Aliased keys (``output_name``/``output_table``, ``primary_column``/
    ``primary_key_column``, ``partition_by``/``partition_by_source``) carry
    *different* values on purpose, so a swapped precedence shows up as a wrong
    value, not as a coincidence.
    """
    return {
        "TABLE": {
            "source_name": "src_orders",
            "source_schema": "src_schema",
            "output_schema": "out_schema",
            "output_name": "out_name_wins",
            "output_table": "out_table_loses",
            "selected_columns": ["sel_a", "sel_b"],
            "only_selected_columns": True,
            "excluded_columns": ["exc_col"],
            "where_clause": "id > 42",
            "empty_rows_ok": True,
        },
        "LOAD_SETTINGS": {
            "load_method": "incremental",
            "primary_column": "pk_wins",
            "primary_key_column": "pk_loses",
            "batch_size": 1234,
            "hash_column": "my_hash",
            "hash_include_columns": ["hash_inc"],
            "hash_exclude_columns": ["hash_exc"],
            "hash_diff_buffer_size": 2345,
            "hash_download_batch_size": 3456,
            "surrogate_key_enabled": True,
            "surrogate_key_definition": "concat(a, b)",
            "created_at_column": "created_col",
            "updated_at_column": "updated_col",
            "days_back": 9,
            "pagination_mode": "keyset",
            "partition_by_source": True,
            "partition_by": {"column": "region", "mode": "list", "default_partition": False},
            "multi_source": True,
            "keyset_pagination": True,
            "resume_full_load": True,
            "skip_size_estimate": True,
            "compute_row_hash": True,
            "conflict_columns": ["conf_a", "conf_b"],
            "ignore_hash": True,
            "num_parallel": 4,
            "strict_integer_precision": True,
            "convert_nchar_to_varchar": True,
            "incremental_date_column": "inc_date",
            "incremental_date_column_fallback": "inc_date_fb",
            "incremental_lookback_hours": 48,
            "incremental_parent_table": "parent_tbl",
            "incremental_parent_output_name": "parent_out",
            "incremental_parent_date_column": "parent_date",
            "incremental_parent_date_column_fallback": "parent_date_fb",
            "incremental_parent_key_column": "parent_fk",
            "incremental_parent_id_column": "parent_pk",
        },
        "SOURCE_DB": {
            "user": "db_user",
            "password": "db_pass",
            "host": "db-host.internal",
            "port": 3306,
            "database": "db_name",
            "charset": "cp1250",
            "ssh_address_or_host": "bastion.internal",
            "ssh_port": 2222,
            "ssh_username": "ssh_user",
            "ssh_pkey": "/keys/full.pem",
            "ssh_wait_timeout": 33,
            "ssh_host_key_checking": "accept-new",
            "remote_bind_address": ["remote-db", 1521],
            "local_bind_address": ["127.0.0.1", 15432],
        },
        "connection_mode": "direct",
    }


def test_the_full_config_covers_the_whole_frozen_contract() -> None:
    """The roundtrip below is only as strong as its input.

    When a key is added to the contract, this fails first and forces the
    roundtrip to grow with it — otherwise the new key would join the suite with
    the same blind spot the incremental keys had.
    """
    cfg = _full_config()
    assert set(cfg["TABLE"]) == set(config.TABLE_KEYS)
    assert set(cfg["LOAD_SETTINGS"]) == set(config.LOAD_SETTINGS_KEYS)
    assert set(cfg["SOURCE_DB"]) == set(config.SOURCE_DB_KEYS)


def test_full_roundtrip_of_the_table_section() -> None:
    parsed = config.parse(_full_config())

    assert parsed.table.source_name == "src_orders"
    assert parsed.table.source_schema == "src_schema"
    assert parsed.table.output_schema == "out_schema"
    assert parsed.table.output_table == "out_name_wins"
    assert parsed.table.selected_columns == ("sel_a", "sel_b")
    assert parsed.table.only_selected_columns is True
    assert parsed.table.excluded_columns == ("exc_col",)
    assert parsed.table.where_clause == "id > 42"
    assert parsed.table.empty_rows_ok is True


def test_full_roundtrip_of_the_load_settings_section() -> None:
    parsed = config.parse(_full_config())
    ls = parsed.load_settings

    assert ls.load_method == "incremental"
    assert ls.primary_column == "pk_wins"
    assert ls.batch_size == 1234
    assert ls.hash_column == "my_hash"
    assert ls.hash_include_columns == ("hash_inc",)
    assert ls.hash_exclude_columns == ("hash_exc",)
    assert ls.hash_diff_buffer_size == 2345
    assert ls.hash_download_batch_size == 3456
    assert ls.surrogate_key_enabled is True
    assert ls.surrogate_key_definition == "concat(a, b)"
    assert ls.created_at_column == "created_col"
    assert ls.updated_at_column == "updated_col"
    assert ls.days_back == 9
    assert ls.pagination_mode == "keyset"
    assert ls.partition_by_source is True
    assert ls.partition_by == {"column": "region", "mode": "list", "default_partition": False}
    assert ls.multi_source is True
    assert ls.keyset_pagination is True
    assert ls.resume_full_load is True
    assert ls.skip_size_estimate is True
    assert ls.compute_row_hash is True
    assert ls.conflict_columns == ("conf_a", "conf_b")
    assert ls.ignore_hash is True
    assert ls.num_parallel == 4
    assert ls.strict_integer_precision is True
    assert ls.convert_nchar_to_varchar is True
    assert ls.incremental_date_column == "inc_date"
    assert ls.incremental_date_column_fallback == "inc_date_fb"
    assert ls.incremental_lookback_hours == 48
    assert ls.incremental_parent_table == "parent_tbl"
    assert ls.incremental_parent_output_name == "parent_out"
    assert ls.incremental_parent_date_column == "parent_date"
    assert ls.incremental_parent_date_column_fallback == "parent_date_fb"
    assert ls.incremental_parent_key_column == "parent_fk"
    assert ls.incremental_parent_id_column == "parent_pk"


def test_full_roundtrip_of_the_source_db_section() -> None:
    parsed = config.parse(_full_config())
    db = parsed.source_db

    assert db.user == "db_user"
    assert db.password == "db_pass"
    assert db.host == "db-host.internal"
    assert db.port == 3306
    assert db.database == "db_name"
    assert db.charset == "cp1250"
    assert db.ssh_address_or_host == "bastion.internal"
    assert db.ssh_port == 2222
    assert db.ssh_username == "ssh_user"
    assert db.ssh_pkey == "/keys/full.pem"
    assert db.ssh_wait_timeout == 33
    assert db.ssh_host_key_checking == "accept-new"
    assert db.remote_bind_address == ("remote-db", 1521)
    assert db.local_bind_address == ("127.0.0.1", 15432)


# --- Fallbacks in both directions -------------------------------------------
#
# The earlier suite exercised the section-beats-legacy direction only for
# ``user``; for every other credential the fallback never decided anything, so a
# swapped or dropped fallback would have passed unnoticed.


@pytest.mark.parametrize(
    "key,section_value,legacy_value",
    [
        ("password", "section_pw", "legacy_pw"),
        ("database", "section_db", "legacy_db"),
        ("port", 1111, 2222),
    ],
)
def test_source_db_beats_connection_params_per_key(key, section_value, legacy_value) -> None:
    cfg = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d", key: section_value},
        "CONNECTION_PARAMS": {key: legacy_value},
    }
    parsed = config.parse(cfg)

    assert getattr(parsed.source_db, key) == section_value


@pytest.mark.parametrize(
    "key,legacy_value",
    [
        ("password", "legacy_pw"),
        ("database", "legacy_db"),
        ("port", 2222),
    ],
)
def test_connection_params_fills_a_key_absent_from_source_db(key, legacy_value) -> None:
    """Not just precedence — the fallback itself has to fire when the section
    exists but the key is absent from it."""
    section = {"user": "u", "password": "p", "database": "d"}
    section.pop(key, None)
    cfg = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": section,
        "CONNECTION_PARAMS": {key: legacy_value},
    }
    parsed = config.parse(cfg)

    assert getattr(parsed.source_db, key) == legacy_value


def test_source_db_ssh_pkey_beats_ssh_params() -> None:
    cfg = _modern_config()
    cfg["SOURCE_DB"]["ssh_pkey"] = "/keys/section.pem"
    cfg["SSH_PARAMS"] = {"ssh_pkey": "/keys/legacy.pem"}

    parsed = config.parse(cfg)

    assert parsed.source_db.ssh_pkey == "/keys/section.pem"


def test_ssh_params_fills_ssh_keys_absent_from_source_db() -> None:
    cfg = _modern_config()
    cfg["SSH_PARAMS"] = {
        "ssh_pkey": "/keys/legacy.pem",
        "ssh_username": "legacy_svc",
        "ssh_port": 2200,
    }

    parsed = config.parse(cfg)

    assert parsed.source_db.ssh_pkey == "/keys/legacy.pem"
    assert parsed.source_db.ssh_username == "legacy_svc"
    assert parsed.source_db.ssh_port == 2200


def test_explicit_empty_rows_ok_false_beats_the_top_level_true() -> None:
    """`table_cfg.get('empty_rows_ok', config.get('EMPTY_ROWS_OK'))` — the
    top-level key is a *default*, not an override. An explicit ``False`` in the
    section must win, otherwise a pipeline that opted back into the safety check
    would silently lose it again."""
    cfg = _modern_config()
    cfg["TABLE"]["empty_rows_ok"] = False
    cfg["EMPTY_ROWS_OK"] = True

    parsed = config.parse(cfg)

    assert parsed.table.empty_rows_ok is False


# --- Aliases decided by the fallback alone ----------------------------------


def test_primary_key_column_alone_lands_in_primary_column() -> None:
    """The earlier tests always sent both spellings, so the alias never decided
    anything — dropping the fallback would have passed."""
    cfg = _modern_config()
    del cfg["LOAD_SETTINGS"]["primary_column"]
    cfg["LOAD_SETTINGS"]["primary_key_column"] = "legacy_pk"

    parsed = config.parse(cfg)

    assert parsed.load_settings.primary_column == "legacy_pk"


def test_primary_column_beats_primary_key_column() -> None:
    cfg = _modern_config()
    cfg["LOAD_SETTINGS"]["primary_column"] = "modern_pk"
    cfg["LOAD_SETTINGS"]["primary_key_column"] = "legacy_pk"

    parsed = config.parse(cfg)

    assert parsed.load_settings.primary_column == "modern_pk"


def test_output_name_beats_output_table_in_the_same_section() -> None:
    """Until now this precedence was covered only against the golden suite's
    own reimplementation of the rule — a shared misreading would have agreed
    with itself. This pins it against `parse()` directly."""
    cfg = _modern_config()
    cfg["TABLE"]["output_name"] = "name_wins"
    cfg["TABLE"]["output_table"] = "table_loses"

    parsed = config.parse(cfg)

    assert parsed.table.output_table == "name_wins"


def test_an_empty_host_falls_back_to_private_host() -> None:
    """`source_db_cfg.get('host') or config.get('PRIVATE_HOST')` — an empty
    host is not a host (unlike an empty password, which is a password). The
    rule sits in the `SourceDbConfig` docstring; this is its first test."""
    cfg = _modern_config()
    cfg["SOURCE_DB"]["host"] = ""
    cfg["PRIVATE_HOST"] = "private.internal"

    parsed = config.parse(cfg)

    assert parsed.source_db.host == "private.internal"


# --- TARGET_DB: declared everywhere, read by nothing -------------------------


def test_target_db_section_is_ignored_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """342 configuration blocks declare ``TARGET_DB``, passwords included, and
    nothing has ever read it — the target comes from ``TARGET_PROFILE``. It must
    stay that way: parse succeeds, nothing from the section lands anywhere, and
    the key is flagged as unknown so the dead weight is at least visible."""
    cfg = _modern_config()
    cfg["TARGET_DB"] = {"host": "warehouse.internal", "password": "target_pw"}

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        parsed = config.parse(cfg)

    assert any("TARGET_DB" in record.message for record in caplog.records)
    assert parsed.source_db.host == "db.internal"
    assert parsed.source_db.password == "p"
    assert parsed.target_profile is None


def test_dialect_and_target_profile_are_not_reported_as_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: every pipeline used to log "Unknown top-level key 'DIALECT'"
    while the key was doing exactly what it should, and `parse` warned about
    ``TARGET_PROFILE`` two lines before reading it. Both are documented keys and
    must pass silently."""
    cfg = _modern_config()
    cfg["DIALECT"] = "mysql"
    cfg["TARGET_PROFILE"] = "warehouse_two"

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        parsed = config.parse(cfg)

    assert parsed.target_profile == "warehouse_two"
    assert not any("DIALECT" in record.message for record in caplog.records)
    assert not any("TARGET_PROFILE" in record.message for record in caplog.records)


# --- Bind addresses through parse() ------------------------------------------


@pytest.mark.parametrize("key", ["remote_bind_address", "local_bind_address"])
def test_a_valid_bind_address_becomes_a_tuple(key: str) -> None:
    """YAML delivers a list; the tunnel layer wants ``(host, port)`` with a
    numeric port. Until now `_as_bind_address` was only reachable in tests
    through the helper itself, never through `parse()`."""
    cfg = _modern_config()
    cfg["SOURCE_DB"][key] = ["bind-host", 5432]

    parsed = config.parse(cfg)

    assert getattr(parsed.source_db, key) == ("bind-host", 5432)


@pytest.mark.parametrize("key", ["remote_bind_address", "local_bind_address"])
@pytest.mark.parametrize(
    "bad", ["host:5432", ["host"], ["host", "not-a-port"]], ids=["string", "one-item", "bad-port"]
)
def test_an_invalid_bind_address_warns_and_yields_none(
    key: str, bad: object, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad shape must not fail the run (the tunnel can derive its own
    defaults), but it must not pass silently either — the operator typed it for
    a reason."""
    cfg = _modern_config()
    cfg["SOURCE_DB"][key] = bad

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.config"):
        parsed = config.parse(cfg)

    assert getattr(parsed.source_db, key) is None
    assert any(key in record.message for record in caplog.records)


# --- only_selected_columns without a selection -------------------------------


def test_only_selected_columns_with_an_empty_selection_keeps_every_column() -> None:
    """Pin of an inherited edge, not a design choice: the flag with no
    ``selected_columns`` is a **no-op** — every source column is kept.

    The guard in `entrypoint._select_columns` is
    ``if table_cfg.only_selected_columns and selected``, same as the
    predecessors. The other two readings — "select nothing" (an empty target
    table) or a config error — would both change behaviour under existing
    pipelines, so the no-op is frozen here."""
    from dbextractors.dialects.base import ColumnDef
    from dbextractors.entrypoint import _select_columns

    cfg = _modern_config()
    cfg["TABLE"]["only_selected_columns"] = True
    cfg["TABLE"]["selected_columns"] = []
    parsed = config.parse(cfg)

    assert parsed.table.only_selected_columns is True
    assert parsed.table.selected_columns == ()

    columns = [ColumnDef(name=n, source_type="varchar", pg_type="TEXT") for n in ("id", "note")]
    kept = _select_columns(columns, parsed.table, parsed.load_settings)

    assert [c.name for c in kept] == ["id", "note"]
