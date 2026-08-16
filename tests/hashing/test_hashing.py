"""Characterisation tests for `core.hashing` — the new implementation against the old.

The requirement is blunt: ``row_hash`` has to come out **bit for bit** the same as it
does today. Not equivalent — the same. Were it to change, hash-diff would judge every
table to have changed and roughly 530 tables would demand a full reload. Silently,
with a green run.

The most important test in the file is `test_vectorisation_matches_apply`. It compares
the vectorised path against today's ``df.apply(_row_hash, axis=1)`` over a battery of
frames whose type composition forces pandas to promote.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import reference_oracle as ro
from dbextractors.core import hashing

# --- The battery of frames --------------------------------------------------
# Every entry covers one situation in which `.apply(axis=1)` behaves differently from
# what one would expect — hence whole frames rather than single values.
#
# The frames themselves are **frozen**: they are hashed into the call fingerprints in
# tests/fixtures/oracle/, so neither the column names nor the values may change. Only
# the keys of this dict are free, because they only serve as test ids. Non-ASCII values
# are written as escapes so the file stays ASCII while the data stays identical.

FRAMES = {
    "all integers": pd.DataFrame({"id": [1, 2, 3], "pocet": [10, 20, 30]}),
    "integers + decimals (promoted to float)": pd.DataFrame(
        {"id": [1, 2, 3], "cena": [1.5, 2.5, 3.5]}
    ),
    "integers + text (no promotion)": pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]}),
    "with missing values": pd.DataFrame(
        {"id": [1, None, 3], "text": ["a", None, "c"]},
    ),
    "boolean": pd.DataFrame({"vlajka": [True, False, True], "id": [1, 2, 3]}),
    "date and time": pd.DataFrame(
        {
            "kdy": pd.to_datetime(["2026-08-11", "2020-01-01", "1999-12-31"]),
            "text": ["a", "b", "c"],
        }
    ),
    "Decimal": pd.DataFrame(
        {"castka": [Decimal("1.50"), Decimal("2"), Decimal("-3.25")], "id": [1, 2, 3]}
    ),
    "date as an object": pd.DataFrame(
        {"d": [dt.date(2026, 8, 11), dt.date(2020, 1, 1), None], "id": [1, 2, 3]}
    ),
    "unicode and whitespace": pd.DataFrame(
        {
            "t": ["p\u0159\xedli\u0161 \u017elu\u0165ou\u010dk\xfd", "已经", "  mezery  "],
            "id": [1, 2, 3],
        }
    ),
    "separator inside a value": pd.DataFrame({"a": ["x||y", "x", "||"], "b": ["y", "|y", ""]}),
    "empty string vs None": pd.DataFrame({"a": ["", None, "x"], "b": [1, 2, 3]}),
    "numpy types": pd.DataFrame(
        {"i": np.array([1, 2, 3], dtype=np.int32), "f": np.array([1.5, 2.5, 3.5], dtype=np.float32)}
    ),
    "nullable Int64": pd.DataFrame(
        {"i": pd.array([1, None, 3], dtype="Int64"), "t": ["a", "b", "c"]}
    ),
    "a single column": pd.DataFrame({"a": ["x", "y", "z"]}),
    "all missing": pd.DataFrame({"a": [None, None, None], "b": [None, None, None]}),
    "bytes": pd.DataFrame({"b": [b"abc", b"", None], "id": [1, 2, 3]}),
}


def _apply_row_by_row(df: pd.DataFrame, columns: list) -> pd.Series:
    """Today's ``df.apply(_row_hash, axis=1)``, taken verbatim from the predecessor.

    The oracle is not used here, because in the predecessor this expression is nested
    inside `add_hash_and_timestamp` and cannot be extracted on its own. It is copied
    character for character from the predecessor's MySQL streaming extractor.
    """

    def _row_hash(row):
        concat = "||".join("" if pd.isna(row[col]) else str(row[col]) for col in columns)
        return hashlib.sha256(concat.encode("utf-8")).hexdigest()

    return df.apply(_row_hash, axis=1)


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_vectorisation_matches_apply(name: str) -> None:
    """**The most important test here.**

    Vectorisation must not change a single hash. The battery deliberately contains
    frames where `.apply(axis=1)` promotes the row type (``int64`` + ``float64`` ->
    float), because a naive column-by-column pass would build a different string there.
    """
    df = FRAMES[name]
    columns = list(df.columns)
    expected = _apply_row_by_row(df.copy(), columns)
    got = hashing.compute_row_hashes(df.copy(), columns)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_type_promotion_really_happens() -> None:
    """Evidence that the previous test is not toothless.

    If `.apply(axis=1)` did not promote the type, there would be no difference between
    a row-wise and a column-wise pass, and the test would be guarding nothing. Here the
    difference is visible — and `compute_row_hashes` reproduces it.
    """
    df = pd.DataFrame({"id": [1], "cena": [1.5]})
    naive = hashing.row_hash([df["id"].iloc[0], df["cena"].iloc[0]])
    actual = hashing.compute_row_hashes(df, ["id", "cena"]).iloc[0]
    assert naive != actual
    assert actual == hashlib.sha256(b"1.0||1.5").hexdigest()
    assert naive == hashlib.sha256(b"1||1.5").hexdigest()


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_add_hash_and_timestamp_matches_the_predecessor(name: str) -> None:
    predecessor = ro.get("A-mysql", "add_hash_and_timestamp")
    df = FRAMES[name]

    new = hashing.add_hash_and_timestamp(df.copy(), "row_hash")
    original = predecessor(df.copy(), "row_hash")

    assert new["row_hash"].tolist() == original["row_hash"].tolist()


def test_the_hash_is_not_recomputed_when_it_is_already_there() -> None:
    """The usual path: the source computed the hash in the SELECT, we leave it alone."""
    df = pd.DataFrame({"id": [1, 2], "row_hash": ["from-source-1", "from-source-2"]})
    out = hashing.add_hash_and_timestamp(df, "row_hash")
    assert out["row_hash"].tolist() == ["from-source-1", "from-source-2"]


def test_force_recompute_overwrites_an_existing_hash() -> None:
    df = pd.DataFrame({"id": [1, 2], "row_hash": ["old", "old"]})
    out = hashing.add_hash_and_timestamp(df, "row_hash", force_recompute=True)
    assert out["row_hash"].tolist() != ["old", "old"]
    assert all(len(h) == 64 for h in out["row_hash"])


def test_compute_hash_false_leaves_the_hash_alone() -> None:
    """``compute_row_hash`` from LOAD_SETTINGS. The default is ``True``, i.e. today."""
    df = pd.DataFrame({"id": [1, 2]})
    out = hashing.add_hash_and_timestamp(df, "row_hash", compute_hash=False)
    assert "row_hash" not in out.columns
    assert "_timestamp" in out.columns


def test_the_timestamp_is_added_but_never_overwritten() -> None:
    df = pd.DataFrame({"id": [1], "_timestamp": ["original"]})
    assert hashing.add_hash_and_timestamp(df, "row_hash")["_timestamp"].tolist() == ["original"]


def test_the_timestamp_does_not_enter_the_hash() -> None:
    """Otherwise the hash would change on every run — which defeats hash-diff entirely."""
    a = pd.DataFrame({"id": [1], "_timestamp": ["2026-01-01"]})
    b = pd.DataFrame({"id": [1], "_timestamp": ["2026-12-31"]})
    assert (
        hashing.add_hash_and_timestamp(a, "row_hash")["row_hash"].iloc[0]
        == hashing.add_hash_and_timestamp(b, "row_hash")["row_hash"].iloc[0]
    )


def test_none_becomes_an_empty_string_not_the_word_none() -> None:
    assert hashing.row_hash([None, "a"]) == hashlib.sha256(b"||a").hexdigest()


def test_an_empty_string_and_none_give_the_same_hash() -> None:
    """Today's behaviour. It is not pretty, but changing it moves ~530 tables."""
    assert hashing.row_hash([None]) == hashing.row_hash([""])


def test_add_hash_tolerates_none_instead_of_a_frame() -> None:
    assert hashing.add_hash_and_timestamp(None, "row_hash") is None


# --- compute_hash_columns ---------------------------------------------------

#: Frozen for the same reason as FRAMES — these names are hashed into the fingerprint of
#: every recorded `compute_hash_columns` call, so here they are data, not language.
COLUMNS = ["id", "nazev", "cena", "row_hash", "interni"]


@pytest.mark.parametrize("short", sorted(ro.variants("compute_hash_columns")))
def test_compute_hash_columns_matches_the_predecessor(short: str) -> None:
    """Against all 13 copies. The two "variants" differ only in a debug log line."""
    predecessor = ro.variants("compute_hash_columns")[short]
    cases = [
        ({}, {}),
        ({}, {"hash_exclude_columns": ["interni"]}),
        ({}, {"hash_include_columns": ["id", "nazev"]}),
        ({}, {"hash_include_columns": ["id", "nazev"], "hash_exclude_columns": ["nazev"]}),
        ({"only_selected_columns": True, "selected_columns": ["id", "cena"]}, {}),
        (
            {"only_selected_columns": True, "selected_columns": ["id", "cena"]},
            {"hash_include_columns": ["nazev"]},
        ),
        ({"only_selected_columns": False, "selected_columns": ["id"]}, {}),
        ({}, {"hash_exclude_columns": None}),
    ]
    for table_settings, load_settings in cases:
        assert hashing.compute_hash_columns(
            COLUMNS, "row_hash", table_settings, load_settings
        ) == predecessor(COLUMNS, "row_hash", table_settings, load_settings)


def test_the_hash_column_does_not_enter_the_hash() -> None:
    assert "row_hash" not in hashing.compute_hash_columns(COLUMNS, "row_hash", {}, {})


def test_the_order_comes_from_the_source_not_from_include() -> None:
    """Order matters for a hash — ``a||b`` is not the same as ``b||a``."""
    assert hashing.compute_hash_columns(
        ["a", "b", "c"], None, {}, {"hash_include_columns": ["c", "a"]}
    ) == ["a", "c"]


# --- pk_to_str --------------------------------------------------------------


@pytest.mark.parametrize("short", sorted(ro.variants("_pk_to_str")))
def test_pk_to_str_matches_the_predecessor(short: str) -> None:
    predecessor = ro.variants("_pk_to_str")[short]
    for value in [1, "a", None, b"abc", bytearray(b"xy"), 1.5, True, b"\xff\xfe", ""]:
        assert hashing.pk_to_str(value) == predecessor(value)


def test_pk_to_str_decodes_bytes() -> None:
    """``str(b'x')`` would give ``"b'x'"`` — and the key would never pair up."""
    assert hashing.pk_to_str(b"abc") == "abc"


def test_pk_series_to_str_gives_the_same_as_element_by_element() -> None:
    for values in ([1, 2, None], ["a", None, "c"], [b"x", b"y", None], [1.5, None, 2.5]):
        series = pd.Series(values, dtype=object)
        assert hashing.pk_series_to_str(series).tolist() == [
            hashing.pk_to_str(v) if v is not None else "" for v in values
        ]


# --- The trap: the common type is found over the whole frame ----------------


def test_a_non_hashed_column_still_affects_the_hash() -> None:
    """It looks like nonsense, but it is today's behaviour — and it has to stay.

    ``.apply(axis=1)`` builds the row out of **all** columns, so a column that is not
    hashed at all can promote the row type and change how the hashed values are
    rendered. Here the price column never enters the hash, and still changes it.
    """
    without = pd.DataFrame({"id": [1]})
    with_price = pd.DataFrame({"id": [1], "cena": [1.5]})

    assert hashing.compute_row_hashes(without, ["id"]).iloc[0] == hashlib.sha256(b"1").hexdigest()
    assert (
        hashing.compute_row_hashes(with_price, ["id"]).iloc[0] == hashlib.sha256(b"1.0").hexdigest()
    )


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_vectorisation_matches_apply_on_a_subset(name: str) -> None:
    """The same as the main test, but only some of the columns are hashed.

    That is exactly what `add_hash_and_timestamp` does: the hash column does not enter
    the hash, yet it is present in the frame.
    """
    df = FRAMES[name].copy()
    df["row_hash"] = None
    columns = [c for c in df.columns if c != "row_hash"]
    expected = _apply_row_by_row(df.copy(), columns)
    got = hashing.compute_row_hashes(df.copy(), columns)
    pd.testing.assert_series_equal(got, expected, check_names=False)
