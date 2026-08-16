"""Tests for `core.partitioning` without a database — configuration parsing and
naming.

The DDL is tested against a live PostgreSQL (`tests/target/test_partitioning_db.py`);
what is here is the pure arithmetic over a configuration and a date.
"""

from __future__ import annotations

import datetime as dt

import pytest

from dbextractors.core import partitioning as p
from dbextractors.core.config import ConfigError, parse
from dbextractors.core.strategies.base import TargetRef

# --- Parsing the configuration ----------------------------------------------


def test_no_key_means_no_partitioning() -> None:
    """The state of all ~670 existing tables — the frozen contract demands it."""
    assert p.parse_spec(None) is None
    assert p.parse_spec(False) is None
    assert p.parse_spec("") is None


def test_the_shorthand_is_list_by_column() -> None:
    spec = p.parse_spec("_source")
    assert spec == p.PartitionSpec(column="_source", mode=p.MODE_LIST, default_partition=True)


def test_a_mapping_with_every_key() -> None:
    spec = p.parse_spec({"column": "event_date", "mode": "range_month", "default_partition": False})
    assert spec is not None
    assert (spec.column, spec.mode, spec.default_partition) == ("event_date", "range_month", False)
    assert spec.is_range is True


def test_legacy_partition_by_source() -> None:
    """Today's spelling has to keep meaning the same thing — otherwise an
    existing deployment breaks."""
    spec = p.parse_spec(None, legacy_by_source=True)
    assert spec == p.PartitionSpec(column="_source", mode=p.MODE_LIST)


def test_partition_by_overrides_the_legacy_key() -> None:
    """Whoever writes both meant the more specific one."""
    spec = p.parse_spec({"column": "event_date", "mode": "range_month"}, legacy_by_source=True)
    assert spec is not None and spec.column == "event_date"


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "range_month"},  # column missing
        {"column": "event_date", "mode": "range_hour"},  # unknown mode
        {"column": "  ", "mode": "list"},  # empty name
        123,  # nonsensical type
    ],
)
def test_nonsense_fails(value) -> None:
    with pytest.raises(ValueError):
        p.parse_spec(value)


def test_a_bad_mode_already_brings_config_parse_down() -> None:
    """It has to fail in `parse()`, that is before anything touches the source."""
    cfg = {
        "TABLE": {"source_name": "t", "output_schema": "s", "output_name": "t"},
        "SOURCE_DB": {"host": "h", "database": "d", "user": "u", "password": "p"},
        "LOAD_SETTINGS": {"partition_by": {"column": "event_date", "mode": "range_hour"}},
    }
    with pytest.raises(ConfigError, match="range_hour"):
        parse(cfg)


def test_config_passes_the_spec_into_the_settings() -> None:
    cfg = {
        "TABLE": {"source_name": "t", "output_schema": "s", "output_name": "t"},
        "SOURCE_DB": {"host": "h", "database": "d", "user": "u", "password": "p"},
        "LOAD_SETTINGS": {"partition_by": {"column": "event_date", "mode": "range_month"}},
    }
    parsed = parse(cfg)
    assert parsed.load_settings.partition_by == {
        "column": "event_date",
        "mode": "range_month",
        "default_partition": True,
    }


def test_from_settings_rebuilds_the_spec() -> None:
    """Strategies are handed a flat dict, and the spec has to be recoverable
    from it."""
    settings = {"partition_by": {"column": "event_date", "mode": "range_month"}}
    assert p.from_settings(settings) == p.PartitionSpec("event_date", "range_month", True)


def test_from_settings_knows_the_legacy_key_too() -> None:
    assert p.from_settings({"partition_by_source": True}) == p.PartitionSpec("_source", "list")


# --- Period bounds ----------------------------------------------------------


@pytest.mark.parametrize(
    "mode,value,start,end",
    [
        ("range_day", dt.date(2026, 8, 14), dt.date(2026, 8, 14), dt.date(2026, 8, 15)),
        ("range_month", dt.date(2026, 8, 14), dt.date(2026, 8, 1), dt.date(2026, 9, 1)),
        ("range_year", dt.date(2026, 8, 14), dt.date(2026, 1, 1), dt.date(2027, 1, 1)),
        # The turn of the year is the boundary mistakes get made on.
        ("range_month", dt.date(2026, 12, 31), dt.date(2026, 12, 1), dt.date(2027, 1, 1)),
        ("range_day", dt.date(2026, 12, 31), dt.date(2026, 12, 31), dt.date(2027, 1, 1)),
        # February in a leap year.
        ("range_month", dt.date(2028, 2, 29), dt.date(2028, 2, 1), dt.date(2028, 3, 1)),
    ],
)
def test_period_bounds(mode, value, start, end) -> None:
    spec = p.PartitionSpec(column="event_date", mode=mode)
    assert p.period_bounds(spec, value) == (start, end)


def test_the_bounds_accept_a_timestamp_and_text_too() -> None:
    spec = p.PartitionSpec(column="event_date", mode=p.MODE_RANGE_MONTH)
    expected = (dt.date(2026, 8, 1), dt.date(2026, 9, 1))
    assert p.period_bounds(spec, dt.datetime(2026, 8, 14, 23, 59)) == expected
    assert p.period_bounds(spec, "2026-08-14T23:59:00") == expected


def test_a_period_makes_no_sense_for_list() -> None:
    with pytest.raises(ValueError):
        p.period_start(p.PartitionSpec(column="x", mode=p.MODE_LIST), dt.date(2026, 8, 14))


# --- Names ------------------------------------------------------------------


def _target(table: str = "inv") -> TargetRef:
    return TargetRef(schema="raw", table=table)


@pytest.mark.parametrize(
    "mode,value,suffix",
    [
        (p.MODE_RANGE_DAY, dt.date(2026, 8, 4), "2026_08_04"),
        (p.MODE_RANGE_MONTH, dt.date(2026, 8, 4), "2026_08"),
        (p.MODE_RANGE_YEAR, dt.date(2026, 8, 4), "2026"),
        (p.MODE_LIST, "Company A/B", "company_a_b"),
    ],
)
def test_partition_suffix(mode, value, suffix) -> None:
    spec = p.PartitionSpec(column="x", mode=mode)
    assert p.partition_suffix(spec, value) == suffix
    assert p.partition_ref(_target(), spec, value).table == f"inv__{suffix}"


def test_a_long_name_is_shortened_and_stays_unambiguous() -> None:
    """Cutting at 63 characters would give two values the same partition."""
    spec = p.PartitionSpec(column="x", mode=p.MODE_LIST)
    target = _target("t" * 40)
    a = p.partition_ref(target, spec, "shared_prefix_" * 3 + "a")
    b = p.partition_ref(target, spec, "shared_prefix_" * 3 + "b")

    assert len(a.table) <= 63 and len(b.table) <= 63
    assert a.table != b.table


def test_the_catch_all_partition_has_a_name_of_its_own() -> None:
    assert p.default_ref(_target()).table == "inv__default"


# --- Unique index -----------------------------------------------------------


def test_without_partitioning_the_unique_index_is_just_the_key() -> None:
    assert p.unique_index_columns(None, "id") == ["id"]


def test_the_unique_index_must_contain_the_partition_key() -> None:
    """PostgreSQL requires this; it is not our choice."""
    spec = p.PartitionSpec(column="event_date", mode=p.MODE_RANGE_MONTH)
    assert p.unique_index_columns(spec, "id") == ["id", "event_date"]


def test_a_partition_key_identical_to_the_pk_is_not_repeated() -> None:
    spec = p.PartitionSpec(column="id", mode=p.MODE_LIST)
    assert p.unique_index_columns(spec, "id") == ["id"]
