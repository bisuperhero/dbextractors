"""Tests for ``LOAD_SETTINGS.strict_integer_precision``.

The corruption these tests guard is **not** fixed here and must not be — the
mechanism, the measured values and the reason parity outranks correctness are all
pinned in ``test_int64_precision.py`` next door. This file covers only the new
key, which turns that silent rounding into a failed run for pipelines that ask
for it.

Two properties matter more than the rest:

* **the default is off**, so every pipeline that does not name the key writes
  exactly what it wrote yesterday, rounding included;
* **the boundary is inclusive at 2**53**, because every integer up to and
  including 2**53 survives ``float64`` exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dbextractors.core import coerce, target_pg

#: The largest integer ``float64`` still represents exactly.
LIMIT = 9007199254740992  # 2**53

#: The realistic 19-digit key from the end-to-end measurement.
WIDE_KEY = 1234567890123456789
ROUNDED_WIDE_KEY = 1234567890123456768


def _frame(values: list, dtype: str = "float64") -> pd.DataFrame:
    return pd.DataFrame({"id": range(len(values)), "big_value": pd.array(values, dtype=dtype)})


# --- The check on its own ----------------------------------------------------


def test_it_fires_on_a_value_above_the_mantissa() -> None:
    frame = _frame([float(WIDE_KEY), np.nan])

    with pytest.raises(coerce.IntegerPrecisionError) as excinfo:
        coerce.check_integer_precision(frame, ["big_value"])

    message = str(excinfo.value)
    # Actionable means all three: which column, which value, which key.
    assert "big_value" in message
    assert str(ROUNDED_WIDE_KEY) in message
    assert "strict_integer_precision" in message


def test_exactly_two_to_the_fifty_third_does_not_fire() -> None:
    """The boundary is inclusive. ``2**53`` itself round-trips through
    ``float64`` without losing a bit, so flagging it would be a false alarm."""
    coerce.check_integer_precision(_frame([float(LIMIT), np.nan]), ["big_value"])


def test_one_step_past_the_boundary_fires() -> None:
    """``2**53 + 2`` is the first value above the limit that ``float64`` can
    still hold, so it is the tightest test of the comparison itself."""
    with pytest.raises(coerce.IntegerPrecisionError):
        coerce.check_integer_precision(_frame([float(LIMIT + 2), np.nan]), ["big_value"])


def test_the_smallest_possible_corruption_is_the_one_case_it_cannot_see() -> None:
    """An honest limit of the guard, stated rather than hidden.

    ``2**53 + 1`` is the *smallest* integer ``float64`` cannot represent — and it
    rounds down to exactly ``2**53``, which is inside the safe range. By the time
    the frame reaches any code of this package the two are the same number, so no
    check placed here can tell them apart. Everything from ``2**53 + 2`` upwards
    is caught.
    """
    assert float(LIMIT + 1) == float(LIMIT)
    coerce.check_integer_precision(_frame([float(LIMIT + 1), np.nan]), ["big_value"])


def test_it_fires_on_negative_values_too() -> None:
    """The comparison is on the absolute value — a key can be negative."""
    with pytest.raises(coerce.IntegerPrecisionError) as excinfo:
        coerce.check_integer_precision(_frame([-float(WIDE_KEY), np.nan]), ["big_value"])

    assert f"-{ROUNDED_WIDE_KEY}" in str(excinfo.value)


def test_negative_two_to_the_fifty_third_does_not_fire() -> None:
    """The safe range is symmetric: ``[-2**53, 2**53]``."""
    coerce.check_integer_precision(_frame([-float(LIMIT), np.nan]), ["big_value"])


def test_a_column_of_nulls_alone_does_not_fire() -> None:
    """``NaN`` compares false, which is why the check needs no null mask."""
    coerce.check_integer_precision(_frame([np.nan, np.nan]), ["big_value"])


def test_an_intact_integer_column_is_never_flagged() -> None:
    """Without a NULL the column stays ``int64`` and no bit was ever lost, so a
    value far above 2**53 is legitimate and must pass."""
    frame = pd.DataFrame({"big_value": pd.array([WIDE_KEY, 1], dtype="int64")})

    coerce.check_integer_precision(frame, ["big_value"])
    assert frame["big_value"][0] == WIDE_KEY


def test_a_nullable_int64_column_is_never_flagged() -> None:
    """``Int64`` keeps the value exactly. It is only demoted to ``float64`` by
    `coerce.replace_pdna_with_none`, and the check sits after that."""
    coerce.check_integer_precision(_frame([WIDE_KEY, None], dtype="Int64"), ["big_value"])


def test_an_unknown_column_is_skipped_rather_than_raising() -> None:
    """The three sources of integer column names do not have to agree — see
    `target_pg._integer_bound_columns`."""
    coerce.check_integer_precision(_frame([1.0]), ["not_in_the_frame"])


# --- Through prepare_export_df, where it is actually wired -------------------


def _prepare_args(**overrides) -> dict:
    args = {
        "meta": {"_deleted_in_source": False},
        "all_columns": ["id", "big_value"],
        "date_like_columns": set(),
        "integer_columns_mapping": {"id": "INTEGER", "big_value": "BIGINT"},
        "overwrite_types": {"id": "INTEGER", "big_value": "BIGINT"},
        "orig_type_map": {},
    }
    args.update(overrides)
    return args


def _batch() -> pd.DataFrame:
    """The shape ``pd.read_sql`` produces for a nullable ``BIGINT``: the NULL
    forced the whole column to ``float64`` and the key is already rounded."""
    return pd.DataFrame.from_records([(1, WIDE_KEY), (2, None)], columns=["id", "big_value"])


def test_prepare_export_df_is_silent_by_default() -> None:
    """The inherited behaviour, unchanged: the rounded value is written."""
    out = target_pg.prepare_export_df(batch_df=_batch(), **_prepare_args())

    assert out["big_value"][0] == str(ROUNDED_WIDE_KEY)


def test_prepare_export_df_fails_when_the_key_is_on() -> None:
    with pytest.raises(coerce.IntegerPrecisionError, match="big_value"):
        target_pg.prepare_export_df(
            batch_df=_batch(), strict_integer_precision=True, **_prepare_args()
        )


def test_prepare_export_df_lets_an_ordinary_batch_through_with_the_key_on() -> None:
    """Switching the key on must not fail a batch that holds nothing suspicious —
    otherwise nobody could ever switch it on."""
    batch = pd.DataFrame.from_records([(1, 10), (2, None)], columns=["id", "big_value"])

    out = target_pg.prepare_export_df(
        batch_df=batch, strict_integer_precision=True, **_prepare_args()
    )

    assert out["big_value"][0] == "10"


def test_a_float_column_bound_for_a_float_target_is_not_touched() -> None:
    """The check applies to integer-bound columns only. A ``NUMERIC`` column may
    legitimately hold a number larger than 2**53 and loses nothing by being a
    float — that is the type it was asked for."""
    batch = pd.DataFrame.from_records([(1, float(WIDE_KEY)), (2, None)], columns=["id", "castka"])
    args = _prepare_args(
        all_columns=["id", "castka"],
        integer_columns_mapping={"id": "INTEGER"},
        overwrite_types={"id": "INTEGER", "castka": "NUMERIC"},
    )

    out = target_pg.prepare_export_df(batch_df=batch, strict_integer_precision=True, **args)

    assert "castka" in out.columns


def test_it_catches_the_second_loss_point_as_well() -> None:
    """A column that arrived **intact** as ``Int64`` and was demoted to
    ``float64`` by the third `coerce.replace_pdna_with_none`.

    This is the reason the check sits where it sits: the value was still exact
    when `prepare_export_df` was called and is not exact any more when it writes.
    """
    batch = pd.DataFrame({"id": [1, 2], "big_value": pd.array([WIDE_KEY, None], dtype="Int64")})

    with pytest.raises(coerce.IntegerPrecisionError, match="big_value"):
        target_pg.prepare_export_df(
            batch_df=batch, strict_integer_precision=True, **_prepare_args()
        )


def test_a_column_known_only_from_the_target_is_checked_too() -> None:
    """``db_integer_cols`` is the third source of integer column names, and a
    column that appears only there still ends up in an integer target column."""
    args = _prepare_args(
        integer_columns_mapping={"id": "INTEGER"},
        overwrite_types={"id": "INTEGER"},
        db_integer_cols=["big_value"],
    )

    with pytest.raises(coerce.IntegerPrecisionError, match="big_value"):
        target_pg.prepare_export_df(batch_df=_batch(), strict_integer_precision=True, **args)
