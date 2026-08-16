"""Characterisation tests for `core.naming` — the new implementation against the old.

Step 1 (``create_column_mapping``) is compared against **all 14 predecessor copies**
through ``reference_oracle``. Step 2 (the underscore prefix) does not exist as a
function in the predecessors — ``mage_ai`` does it inside ``export()`` — so it is
verified elsewhere: ``test_reserved_words.py`` against the live library and
``scripts/verify_column_names.py`` against a real target database.
"""

from __future__ import annotations

import pytest

import reference_oracle as ro
from dbextractors.core import naming

# --- The battery of names --------------------------------------------------
# Every line covers one step of `create_column_mapping`, not a random string.
#
# The values are **frozen**: they are hashed into the call fingerprints in
# tests/fixtures/oracle/, so changing one turns into an unrecorded call rather than
# a new comparison. Diacritics are written as escapes so the file stays ASCII while
# the strings themselves stay byte for byte the same.

NAMES = [
    # plain
    "id",
    "cena",
    "cena_bez_dph",
    # letter case
    "Cena",
    "CENA",
    "CenaBezDPH",
    # whitespace (\s+ -> _)
    "cena bez dph",
    "cena  bez   dph",
    " cena ",
    "cena\tbez\ndph",
    # non-alphanumeric characters (they are deleted, not replaced!)
    "cena-bez-dph",
    "cena.bez.dph",
    "cena/bez/dph",
    "cena(bez dph)",
    "cena%",
    "#cislo",
    "e-mail@domena.cz",
    "50%_sleva",
    # repeated and leading/trailing underscores
    "a__b",
    "a___b",
    "_cena",
    "cena_",
    "__cena__",
    "_",
    "___",
    # unicode: \w is unicode-aware in Python, so accented letters survive
    "\u010de\u0161tina",
    "P\u0159\xedjmen\xed",
    "cena_rozes\xedlky",
    "已经",
    "Ω",
    # leading digits
    "2023",
    "2023_trzby",
    "1. \u010dtvrtlet\xed",
    # edge cases
    "",
    "a",
    "row_hash",
    "_timestamp",
    "_deleted_in_source",
]


@pytest.fixture(scope="module")
def predecessor():
    """The reference ``create_column_mapping``. There is only one — 14 files, one variant.

    It is still taken through the oracle rather than copied in: if the predecessor
    sources ever changed, that has to fail here.
    """
    return ro.get("A-mysql", "create_column_mapping")


def test_the_predecessors_have_a_single_variant() -> None:
    """14 files, 1 real variant.

    The test proves it by running them, not by comparing text: all 14 copies must
    return the same thing over the whole battery.
    """
    variants = ro.variants("create_column_mapping")
    assert len(variants) == 14
    results = {short: fn(NAMES) for short, fn in variants.items()}
    pattern = results["A-mysql"]
    differing = [short for short, out in results.items() if out != pattern]
    assert differing == []


@pytest.mark.parametrize("name", NAMES)
def test_create_column_mapping_matches_the_predecessor(predecessor, name: str) -> None:
    assert naming.create_column_mapping([name]) == predecessor([name])


def test_create_column_mapping_over_the_whole_battery(predecessor) -> None:
    """All at once — this also catches a mistake in the order or in the dict keys."""
    assert naming.create_column_mapping(NAMES) == predecessor(NAMES)


def test_the_mapping_is_keyed_by_the_source_name() -> None:
    mapping = naming.create_column_mapping(["Cena Bez DPH"])
    assert mapping == {"Cena Bez DPH": "cena_bez_dph"}


def test_non_alphanumeric_characters_are_deleted_not_replaced() -> None:
    """The difference against step 2, where ``\\W`` **replaces** with an underscore.

    Were the two unified, ``cena-bez-dph`` would land in the target as
    ``cena_bez_dph`` instead of ``cenabezdph`` and dbt would stop seeing the column.
    """
    assert naming.create_column_mapping(["cena-bez-dph"])["cena-bez-dph"] == "cenabezdph"
    assert naming.mage_clean_name("cena-bez-dph") == "cena_bez_dph"


def test_leading_and_trailing_underscores_are_stripped() -> None:
    """Which is exactly why the metadata columns must not go through this step."""
    assert naming.create_column_mapping(["_timestamp"])["_timestamp"] == "timestamp"


# --- Step 2 on its own -----------------------------------------------------


def test_a_leading_digit_gets_the_letter_prefix() -> None:
    """Step 1 does not do this, step 2 does — it only shows on the composed chain."""
    assert naming.create_column_mapping(["2023_trzby"])["2023_trzby"] == "2023_trzby"
    assert naming.mage_clean_name("2023_trzby") == "letter_2023_trzby"


def test_step_two_deletes_quotes_dollars_and_the_bom() -> None:
    assert naming.mage_clean_name('a"b$c') == "abc"
    assert naming.mage_clean_name("\ufeffcena") == "cena"


def test_an_empty_name_passes_through_unchanged() -> None:
    assert naming.mage_clean_name("") == ""
    assert naming.clean_column_name("") == ""


# --- The whole chain -------------------------------------------------------


def test_the_order_matches_the_predecessor() -> None:
    """Source columns, then the hash, then ``_timestamp``, then ``_deleted_in_source``."""
    assert naming.target_column_names(["Id", "Type", "Cena Bez DPH"], "row_hash") == [
        "id",
        "_type",
        "cena_bez_dph",
        "row_hash",
        "_timestamp",
        "_deleted_in_source",
    ]


def test_column_order_follows_variants_a_and_c_for_postgres() -> None:
    """Variant B appends ``row_hash`` **last**; variants A and C put it second to last.

    The predecessors disagree here, and the order is part of the frozen target
    contract, so this is a decision rather than a detail. Variants A and C win: they
    match 2 deployments out of 3 and the 45 golden tables that have already passed.

    In practice it only shows on a **newly created** table. When the target already
    exists, the shadow table is created with ``LIKE target``, so it inherits the order,
    and `COPY` sends an explicit column list — guarded by
    `tests/target/test_target_pg_db.py`.
    """
    columns = naming.target_column_names(["id", "label"], "row_hash")

    assert columns[-3:] == ["row_hash", "_timestamp", "_deleted_in_source"]
    assert columns[-1] != "row_hash", "variant B's order is deliberately not adopted"


def test_no_hash_column() -> None:
    """A departure from the predecessors: ``None`` means "do not add", not "row_hash".

    The predecessors hard-code a fallback to ``'row_hash'``, so even a table with no
    hashing gets an empty column. Whether to add it is a decision for the strategy.
    """
    assert naming.target_column_names(["id"]) == ["id", "_timestamp", "_deleted_in_source"]


def test_the_hash_column_is_normalised_too() -> None:
    assert naming.target_column_names(["id"], "Row Hash")[1] == "row_hash"


def test_a_source_column_of_the_same_name_is_not_added_twice(caplog) -> None:
    """The predecessors add a generated column only when it is not there already."""
    result = naming.target_column_names(["id", "row_hash"], "row_hash")
    assert result == ["id", "row_hash", "_timestamp", "_deleted_in_source"]
    assert "collides with a generated column" in caplog.text


def test_a_collision_of_two_source_columns_fails() -> None:
    """Today this passes silently and breaks later, in ``CREATE TABLE``."""
    with pytest.raises(naming.ColumnNameCollision) as excinfo:
        naming.target_column_names(["cena-bez-dph", "cena.bez.dph"])
    message = str(excinfo.value)
    assert "cena-bez-dph" in message
    assert "cena.bez.dph" in message
    assert "cenabezdph" in message


def test_the_metadata_columns_survive_step_two() -> None:
    """Neither ``_timestamp`` nor ``_deleted_in_source`` may lose its underscore.

    133 dbt models hang off ``_deleted_in_source``.
    """
    for column in naming.METADATA_COLUMNS:
        assert naming.clean_column_name(column) == column
    assert naming.target_column_names([])[-2:] == list(naming.METADATA_COLUMNS)


# --- Against a table the old extractor actually produced --------------------


def test_matches_a_table_produced_by_the_old_extractor() -> None:
    """The strongest evidence there is for the frozen column-name contract.

    This is not an invariant over some pre-existing table: it is a target table
    **freshly built by a predecessor extractor** out of a real MySQL source table, with
    the run captured in a log.

    It guards three things at once: the prefix on reserved words (``name``, ``admin``),
    that non-reserved words do not get one (``edit``, ``del``, ``modified``), and the
    order of the generated columns at the end.
    """
    source = [
        "id",
        "name",
        "admin",
        "edit",
        "del",
        "authority",
        "modified",
        "created",
        "date_modified",
        "date_created",
    ]
    actually_in_the_target = [
        "id",
        "_name",
        "_admin",
        "edit",
        "del",
        "authority",
        "modified",
        "created",
        "date_modified",
        "date_created",
        "row_hash",
        "_timestamp",
        "_deleted_in_source",
    ]
    assert naming.target_column_names(source, "row_hash") == actually_in_the_target


# --- reconcile_with_existing ------------------------------------------------


def test_the_existing_order_wins() -> None:
    """A table created by variant B's PostgreSQL streaming extractor has `row_hash` last.

    The predecessors build the order of the managed columns in six different ways, so
    the configuration does not say which of them a given table has. Reordering it would
    break `SELECT *` and everything that relies on `ordinal_position`.
    """
    desired = ["id", "row_hash", "_timestamp", "_deleted_in_source"]
    existing = ["id", "_timestamp", "_deleted_in_source", "row_hash"]

    assert naming.reconcile_with_existing(desired, existing) == existing


def test_a_new_column_goes_to_the_end() -> None:
    """Inserting a column in the middle would be exactly the reordering we avoid — even
    if `desired` wants it there."""
    result = naming.reconcile_with_existing(["id", "added", "row_hash"], ["id", "row_hash"])

    assert result == ["id", "row_hash", "added"]


def test_an_extra_column_in_the_target_is_not_written() -> None:
    """A column the target has and `desired` does not (typically one dropped via
    `excluded_columns`) does not belong in the list for `COPY`."""
    result = naming.reconcile_with_existing(["id", "row_hash"], ["id", "dropped", "row_hash"])

    assert result == ["id", "row_hash"]


def test_without_an_existing_target_the_desired_order_applies() -> None:
    """A new table gets the dominant variant V1 — there is nothing to preserve."""
    desired = ["id", "row_hash", "_timestamp", "_deleted_in_source"]

    assert naming.reconcile_with_existing(desired, []) == desired


def test_an_unusual_existing_order_is_preserved_including_source() -> None:
    """MSSQL `full_by_source` is the only predecessor with the order
    `_deleted_in_source, _timestamp, _source, row_hash`. It is not replicated in code —
    it is taken from the target."""
    existing = ["id", "amount", "_deleted_in_source", "_timestamp", "_source", "row_hash"]
    desired = ["id", "amount", "row_hash", "_timestamp", "_deleted_in_source", "_source"]

    assert naming.reconcile_with_existing(desired, existing) == existing
