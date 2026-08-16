"""Tests for the rules covering agreed deviations.

The negative ones matter most: a rule that is too broad hides a genuine
difference — and that is worse than the noise it was created to remove.
"""

from __future__ import annotations

import pytest

from dbextractors.golden import deviations
from dbextractors.golden.model import Difference, Level


def _type_diff(where: str, left: str, right: str) -> Difference:
    return Difference(what="different data type", left=left, right=right, where=where)


# --- What the rules SHOULD cover --------------------------------------------


def test_deleted_in_source_nullability_is_a_deviation() -> None:
    diff = _type_diff("_deleted_in_source", "boolean NULL", "boolean NOT NULL")
    assert deviations.match(Level.COLUMN_TYPES, diff) is not None


@pytest.mark.parametrize(
    "right",
    ["character varying NULL", "integer NULL", "numeric NULL", "boolean NULL"],
)
def test_a_reserved_column_that_lost_its_type_is_a_deviation(right: str) -> None:
    """On a prefixed column the old side slides down to ``text``."""
    assert deviations.match(Level.COLUMN_TYPES, _type_diff("_name", "text NULL", right)) is not None


def _order_diff(left: str, right: str, position: int = 17) -> Difference:
    return Difference(
        what="different column order", left=left, right=right, where=f"position {position}"
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ("_deleted_in_source", "row_hash"),
        ("_source", "_deleted_in_source"),
        ("row_hash", "_source"),
        ("_timestamp", "row_hash"),
    ],
)
def test_reordered_managed_columns_are_a_deviation(left: str, right: str) -> None:
    """Exactly the four columns the old side produces in a different order on MSSQL."""
    assert deviations.match(Level.COLUMN_NAMES, _order_diff(left, right)) is not None


# --- What the rules MUST NOT cover ------------------------------------------


def test_a_nullability_change_on_a_prefixed_column_is_not_a_deviation() -> None:
    """**This is precisely what the rule used to swallow.**

    ``text NULL`` -> ``text NOT NULL`` is a genuine difference, not a consequence
    of a type map that was not found. Caught by
    `test_it_finds_a_nullability_change`.
    """
    diff = _type_diff("_name", "text NULL", "text NOT NULL")
    assert deviations.match(Level.COLUMN_TYPES, diff) is None


def test_a_nullability_change_together_with_a_type_change_is_not_a_deviation() -> None:
    diff = _type_diff("_name", "text NULL", "integer NOT NULL")
    assert deviations.match(Level.COLUMN_TYPES, diff) is None


def test_a_difference_between_two_concrete_types_is_not_a_deviation() -> None:
    """``integer`` vs ``bigint`` is a real difference — it overflows on write."""
    diff = _type_diff("_order", "integer NULL", "bigint NULL")
    assert deviations.match(Level.COLUMN_TYPES, diff) is None


def test_a_non_prefixed_column_is_not_a_deviation() -> None:
    diff = _type_diff("price", "text NULL", "numeric NULL")
    assert deviations.match(Level.COLUMN_TYPES, diff) is None


def test_deleted_in_source_with_a_different_type_is_not_a_deviation() -> None:
    """The rule is tied to a concrete pair of types, not just to a column name."""
    diff = _type_diff("_deleted_in_source", "boolean NULL", "text NULL")
    assert deviations.match(Level.COLUMN_TYPES, diff) is None


def test_the_rules_only_apply_to_their_own_level() -> None:
    diff = _type_diff("_deleted_in_source", "boolean NULL", "boolean NOT NULL")
    assert deviations.match(Level.COLUMN_CHECKSUMS, diff) is None
    assert deviations.match(Level.ROW_HASHES, diff) is None


def test_a_reordered_source_column_is_not_a_deviation() -> None:
    """**This is the boundary the rule is kept narrow for.**

    The order of source columns is part of the contract. Were the rule to forgive
    order in general, a swapped `price` and `quantity` would pass as a deviation
    and the dbt layer would break in silence.
    """
    assert deviations.match(Level.COLUMN_NAMES, _order_diff("price", "quantity")) is None


def test_a_source_column_against_a_managed_one_is_not_a_deviation() -> None:
    """One side outside the four is enough for the rule not to fire."""
    assert deviations.match(Level.COLUMN_NAMES, _order_diff("price", "row_hash")) is None
    assert deviations.match(Level.COLUMN_NAMES, _order_diff("row_hash", "price")) is None


def test_a_missing_column_is_not_a_deviation() -> None:
    """The rule binds to the description "different column order", not to a pair
    of names."""
    missing = Difference(
        what="column missing on the right", left="row_hash", right="_timestamp", where="row_hash"
    )
    assert deviations.match(Level.COLUMN_NAMES, missing) is None


# --- Classification ---------------------------------------------------------


def test_classify_splits_the_list_and_keeps_the_order() -> None:
    diffs = [
        _type_diff("price", "text NULL", "numeric NULL"),
        _type_diff("_name", "text NULL", "character varying NULL"),
        _type_diff("_deleted_in_source", "boolean NULL", "boolean NOT NULL"),
    ]
    real, accepted = deviations.classify(Level.COLUMN_TYPES, diffs)
    assert [d.where for d in real] == ["price"]
    assert [d.where for d in accepted] == ["_name", "_deleted_in_source"]


def test_every_rule_has_a_reason() -> None:
    """Without a recorded reason a rule cannot be judged six months later."""
    for rule in deviations.DEVIATIONS:
        assert rule.name and len(rule.reason) > 40


# --- Required columns -------------------------------------------------------


def _surplus(name: str) -> Difference:
    return Difference(what="column only on the right", left=None, right=name, where=name)


@pytest.mark.parametrize("column", ["row_hash", "_timestamp", "_deleted_in_source"])
def test_a_surplus_required_column_is_a_deviation(column: str) -> None:
    """Variant B's Firebird extractor does not have them at all; dbextractors has
    to have them.

    Without this rule the golden test reported DIFF for every Firebird table
    because the new side did what it was asked to.
    """
    assert deviations.match(Level.COLUMN_NAMES, _surplus(column)) is not None


def test_a_surplus_source_column_is_not_a_deviation() -> None:
    """`_source` is only added for multi-source tables, so a surplus one is a
    genuine finding — the target would be split into partitions that do not
    belong there."""
    assert deviations.match(Level.COLUMN_NAMES, _surplus("_source")) is None


def test_a_surplus_column_from_the_source_is_not_a_deviation() -> None:
    """The rule may only forgive those three by name. A surplus column from the
    source means the new side has parted ways with the old one over which columns
    are selected."""
    assert deviations.match(Level.COLUMN_NAMES, _surplus("amount")) is None


def test_a_missing_required_column_is_not_a_deviation() -> None:
    """The opposite direction is always a finding: were `row_hash` missing **on
    the right**, that breaks the requirement that every strategy provides it and
    the golden test has to catch it."""
    missing = Difference(
        what="column missing on the right", left="row_hash", right=None, where="row_hash"
    )

    assert deviations.match(Level.COLUMN_NAMES, missing) is None
