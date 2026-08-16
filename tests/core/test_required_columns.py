"""The columns a strategy needs must survive the column selection.

The predecessors keep not just the selected columns but the **union** of
``selected_columns`` with ``required_columns``, and symmetrically subtract that
same set from ``excluded_columns``.

Without that safeguard an incremental pipeline that runs 15 times a week fails.
Its configuration is copied here almost literally, because the general case
would miss this particular shape: the column named by
``incremental_date_column`` is absent from the selection while its **fallback**
is present, so at first glance the selection looks complete.

The second half of the tests matters just as much: the set must not be
**larger** than the predecessors'. Were an extra column held on to, the target
would gain a column it does not have today — and the target's column names are
frozen.
"""

from __future__ import annotations

import pytest

from dbextractors.core.config import LoadSettingsConfig, TableConfig
from dbextractors.dialects.base import ColumnDef
from dbextractors.entrypoint import EntrypointError, _required_columns, _select_columns

#: The source's orders table, cut down to what matters.
SOURCE = ("ID", "FIRM_ID", "CORRECTED_AT", "CREATED_AT", "ORDER_NUMBER", "NOTE")


def _columns(names=SOURCE) -> list:
    return [ColumnDef(name=n, source_type="varchar", pg_type="TEXT") for n in names]


def _table(**kwargs) -> TableConfig:
    base = {"source_name": "ORDERS", "output_schema": "raw_erp", "output_table": "x"}
    base.update(kwargs)
    return TableConfig(**base)


def _names(columns) -> list:
    return [c.name for c in columns]


# --- The incremental orders pipeline ----------------------------------------


def _orders() -> LoadSettingsConfig:
    return LoadSettingsConfig(
        load_method="incremental",
        primary_column="ID",
        incremental_date_column="CORRECTED_AT",
        incremental_date_column_fallback="CREATED_AT",
    )


def test_the_incremental_date_column_survives_a_selection_without_it() -> None:
    """The literal shape of that pipeline: the window column is not in the
    selection."""
    table = _table(
        only_selected_columns=True,
        selected_columns=("ID", "FIRM_ID", "CREATED_AT", "ORDER_NUMBER"),
    )

    result = _names(_select_columns(_columns(), table, _orders()))

    assert "CORRECTED_AT" in result, "without it IncrementalStrategy.validate fails"
    assert "NOTE" not in result, "an unselected column must not be smuggled in"


def test_the_fallback_is_not_added_when_it_is_not_in_the_selection() -> None:
    """**Counter-intuitive, but it matches the predecessors.**

    Firebird puts only `incremental_date_column` into `required_columns`, not
    its fallback. Were this package to add it, the target would gain a column it
    does not have today — and the target's column names are frozen. For the
    orders pipeline it makes no difference, because the fallback is in the
    selection; this test guards the case where it is not.
    """
    table = _table(only_selected_columns=True, selected_columns=("ID", "ORDER_NUMBER"))

    result = _names(_select_columns(_columns(), table, _orders()))

    assert "CORRECTED_AT" in result
    assert "CREATED_AT" not in result


def test_the_window_column_cannot_be_excluded() -> None:
    """Symmetry: `excluded_columns` is subtracted from the required set."""
    table = _table(excluded_columns=("CORRECTED_AT", "NOTE"))

    result = _names(_select_columns(_columns(), table, _orders()))

    assert "CORRECTED_AT" in result, "the window key cannot be excluded by accident"
    assert "NOTE" not in result, "an ordinary exclusion still applies"


# --- Other shapes of configuration ------------------------------------------


def test_a_standard_increment_holds_on_to_updated_and_created() -> None:
    """The other shape has a different set in the predecessors: `updated`,
    `created`, hash."""
    settings = LoadSettingsConfig(
        load_method="incremental",
        primary_column="ID",
        updated_at_column="CORRECTED_AT",
        created_at_column="CREATED_AT",
    )
    table = _table(only_selected_columns=True, selected_columns=("ORDER_NUMBER",))

    result = _names(_select_columns(_columns(), table, settings))

    assert {"ID", "CORRECTED_AT", "CREATED_AT"} <= set(result)


def test_hash_holds_on_to_the_key_and_the_hash_column() -> None:
    settings = LoadSettingsConfig(
        load_method="hash", primary_column="ID", hash_column="ORDER_NUMBER"
    )
    table = _table(only_selected_columns=True, selected_columns=("FIRM_ID",))

    result = _names(_select_columns(_columns(), table, settings))

    assert {"ID", "ORDER_NUMBER", "FIRM_ID"} == set(result)


def test_full_holds_on_to_the_hash_column_only() -> None:
    """Under `full` the predecessors protect only the hash column — not the key.
    Holding on to more would mean a column in the target that is not there
    today."""
    settings = LoadSettingsConfig(
        load_method="full", primary_column="ID", hash_column="ORDER_NUMBER"
    )
    table = _table(only_selected_columns=True, selected_columns=("FIRM_ID",))

    result = _names(_select_columns(_columns(), table, settings))

    assert set(result) == {"FIRM_ID", "ORDER_NUMBER"}
    assert "ID" not in result


def test_a_column_the_source_does_not_have_is_not_added() -> None:
    """The predecessors add only what is `in column_names`. Otherwise a name
    that does not exist would reach the selection and the `SELECT` would fail."""
    settings = LoadSettingsConfig(
        load_method="incremental", primary_column="ID", incremental_date_column="NOT_THERE"
    )

    assert _required_columns(_columns(), settings) == {"ID"}


def test_an_empty_result_fails() -> None:
    """A loud failure beats a silently empty target."""
    table = _table(only_selected_columns=True, selected_columns=("DOES_NOT_EXIST",))
    settings = LoadSettingsConfig(load_method="full")

    with pytest.raises(EntrypointError):
        _select_columns(_columns(), table, settings)
