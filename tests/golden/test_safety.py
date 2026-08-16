"""Tests for the safeguards that keep production out of harm's way.

They need no database — it is pure logic. At the same time it is the part of the
harness where a mistake would be most expensive: if redirecting the configuration
failed to take effect, the old component would write into a production schema.
"""

from __future__ import annotations

import re

import pytest

from dbextractors.golden import scratch
from dbextractors.golden.runners import (
    UnsafeConfigError,
    redirect_output,
    target_table_name,
)

# --- The schema safeguard ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "public",
        "stg_legacy",
        "reporting",
        "",
        "dbx_golden",  # no trailing underscore after the prefix
        "DBX_GOLDEN_x",  # upper case
        "x_dbx_golden_y",  # the prefix is not at the start
        "dbx_golden_x; DROP SCHEMA public",
        'dbx_golden_"; --',
    ],
)
def test_it_rejects_a_foreign_schema(name):
    with pytest.raises(scratch.UnsafeSchemaError):
        scratch.assert_safe(name)


@pytest.mark.parametrize("name", ["dbx_golden_a", "dbx_golden_20260807151226_test_e8ee3c7c"])
def test_it_accepts_its_own_schema(name):
    scratch.assert_safe(name)


#: What a generated name must look like. Deliberately spelled out here rather
#: than imported from ``scratch``: were the module's own pattern to loosen, a
#: test that reuses it would loosen along with it and prove nothing.
_GENERATED_SHAPE = re.compile(r"^dbx_golden_[a-z0-9_]+$")


def test_a_generated_name_has_the_required_shape():
    """The predecessor of this test called ``assert_safe`` on the generated name
    — which the generator already does itself, so the test was a tautology and a
    broken generator would have sailed through. The shape is therefore asserted
    independently here.

    The 63-character bound is the PostgreSQL identifier limit (NAMEDATALEN-1): a
    longer name would be **silently truncated** by the server, the truncation
    could merge two distinct scratch schemas, and ``DROP`` would then take down
    another run's data.
    """
    labels = ("", "invoices", "Invoices - 2026 / items", "x" * 100, "Řízení / ÚČTY (2026)")
    for label in labels:
        name = scratch.scratch_schema_name(label)
        assert name.startswith(scratch.SCRATCH_PREFIX), name
        assert _GENERATED_SHAPE.fullmatch(name), name
        assert len(name) <= 63, f"{name!r} exceeds the PostgreSQL identifier limit"


class _UntouchableConnection:
    """A connection that fails loudly the moment anything uses it.

    It exists to prove that the safeguard fires **before** the database is
    touched. The audit found that removing ``assert_safe`` from ``create_schema``
    and ``drop_schema`` survived the whole suite — the guard was only ever tested
    directly, never at its points of use. With this fake it does not matter *how*
    those functions invoke the guard; any mutant that reaches the database raises
    ``AssertionError`` instead of the expected ``UnsafeSchemaError``.
    """

    def cursor(self, *args, **kwargs):
        raise AssertionError("the database was touched before the safeguard fired")

    def __getattr__(self, name):
        raise AssertionError(f"the connection was used ({name}) before the safeguard fired")


#: Names that must never reach the database: a production schema, an injection
#: attempt, and a case variant that PostgreSQL would fold onto the real prefix.
_DANGEROUS_NAMES = ["public", 'dbx_golden_x"; DROP TABLE y; --', "DBX_GOLDEN_x"]


@pytest.mark.parametrize("name", _DANGEROUS_NAMES)
def test_create_schema_refuses_before_touching_the_database(name):
    with pytest.raises(scratch.UnsafeSchemaError):
        scratch.create_schema(_UntouchableConnection(), name)


@pytest.mark.parametrize("name", _DANGEROUS_NAMES)
def test_drop_schema_refuses_before_touching_the_database(name):
    with pytest.raises(scratch.UnsafeSchemaError):
        scratch.drop_schema(_UntouchableConnection(), name)


def test_the_names_do_not_repeat():
    """A collision would mean two runs overwriting each other's data."""
    names = {scratch.scratch_schema_name("invoices") for _ in range(200)}
    assert len(names) == 200


def test_the_name_carries_a_label_so_it_can_be_placed():
    assert "invoices" in scratch.scratch_schema_name("invoices")


# --- Redirecting the configuration ------------------------------------------

SCRATCH = "dbx_golden_test_x"


def test_it_overwrites_output_schema():
    config = {
        "TABLE": {
            "source_name": "invoices",
            "output_schema": "stg_legacy",
            "output_name": "invoices",
        }
    }
    patched = redirect_output(config, SCRATCH)
    assert patched["TABLE"]["output_schema"] == SCRATCH


def test_it_does_not_change_the_input_config():
    """The caller has to keep its own configuration untouched."""
    config = {"TABLE": {"output_schema": "stg_legacy", "output_name": "invoices"}}
    redirect_output(config, SCRATCH)
    assert config["TABLE"]["output_schema"] == "stg_legacy"


def test_it_overwrites_the_legacy_key_too():
    """A legacy top-level key — see ``docs/legacy-compat.md``."""
    config = {"OUTPUT_SCHEMA": "stg_legacy", "OUTPUT_TABLE": "invoices"}
    patched = redirect_output(config, SCRATCH)
    assert patched["OUTPUT_SCHEMA"] == SCRATCH


def test_it_overwrites_both_keys_at_once():
    """The most dangerous case.

    Were the configuration to have both variants and only one of them
    overwritten, the component could reach production through the other.
    """
    config = {
        "TABLE": {"output_schema": "stg_legacy", "output_name": "invoices"},
        "OUTPUT_SCHEMA": "reporting",
    }
    patched = redirect_output(config, SCRATCH)
    assert patched["TABLE"]["output_schema"] == SCRATCH
    assert patched["OUTPUT_SCHEMA"] == SCRATCH


def test_it_fills_in_the_schema_when_the_config_has_none():
    config = {"TABLE": {"source_name": "invoices", "output_name": "invoices"}}
    patched = redirect_output(config, SCRATCH)
    assert patched["TABLE"]["output_schema"] == SCRATCH


def test_it_refuses_to_redirect_into_production():
    """The redirect has a safeguard of its own, not only scratch.py."""
    config = {"TABLE": {"output_schema": "stg_legacy", "output_name": "invoices"}}
    with pytest.raises(scratch.UnsafeSchemaError):
        redirect_output(config, "public")


def test_it_leaves_the_source_schema_alone():
    """It is the target that gets rewritten, not the source. The source is only read."""
    config = {
        "TABLE": {
            "source_schema": "dbo",
            "source_name": "invoices",
            "output_schema": "stg_legacy",
            "output_name": "invoices",
        }
    }
    patched = redirect_output(config, SCRATCH)
    assert patched["TABLE"]["source_schema"] == "dbo"


# --- The target table name --------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"TABLE": {"output_name": "a"}}, "a"),
        ({"TABLE": {"output_table": "b"}}, "b"),
        ({"OUTPUT_TABLE": "c"}, "c"),
        # output_name takes precedence over output_table — that is how the
        # predecessor reads it.
        ({"TABLE": {"output_name": "a", "output_table": "b"}}, "a"),
    ],
)
def test_it_finds_the_target_table(config, expected):
    assert target_table_name(config) == expected


def test_it_rejects_a_config_without_a_target_table():
    with pytest.raises(UnsafeConfigError):
        target_table_name({"TABLE": {"source_name": "invoices"}})
