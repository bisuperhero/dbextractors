"""Characterisation tests for `core.coerce` — the new implementation against the old.

The requirement: the output has to be identical character for character with today's.
Not "equivalent" — identical. The reason is concrete: ``row_hash`` is computed from the
string representation of the values. Were ``to_date_str`` to start returning
``2026-08-07`` instead of ``2026-08-07 00:00:00``, every hash would change, hash-diff
would judge the whole table changed, and roughly 530 tables would demand a full reload.
Silently, with a green run.

The old implementation is run through ``reference_oracle`` — see its docstring for why
a plain import will not do.

**Both the value and the type are compared**, not just equality. ``0 == False`` and
``1 == True``, so ``==`` on its own would not reveal a bool substituted for an int —
and that difference does reach ``row_hash`` (``'0'`` vs ``'False'``).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import reference_oracle as ro
from dbextractors.core import coerce

# --- The battery of inputs -------------------------------------------------
# The brief was: valid values, None, pd.NA, NaN, an empty string, 0000-00-00, a date
# out of range, a number above INT_MAX, text with \r\n, bytes in latin-2, JSON,
# tinyint(1), negative numbers, numbers with a dot and with a comma.
#
# These vectors are **frozen**: they are hashed into the call fingerprints in
# tests/fixtures/oracle/, so a changed value becomes an unrecorded call rather than a
# new comparison. Non-ASCII values are written as escapes so that the file itself stays
# ASCII while the strings under test stay byte for byte the same.

INT_MAX = 2_147_483_647

MISSING = [None, pd.NA, float("nan"), np.nan]

NUMBERS = [
    0,
    1,
    -1,
    42,
    -42,
    INT_MAX,
    INT_MAX + 1,
    -INT_MAX - 1,
    2**63,
    -(2**63),
    0.0,
    1.0,
    -1.0,
    3.5,
    -3.5,
    2.000000001,
    np.int64(7),
    np.int32(-7),
    np.float64(8.0),
    np.float32(1.5),
    Decimal("3"),
    Decimal("3.5"),
    Decimal("-0"),
]

BOOLEANS = [True, False, np.bool_(True), np.bool_(False)]

NUMERIC_TEXT = [
    "0",
    "1",
    "-1",
    "42",
    " 42 ",
    "3.5",
    "-3.5",
    "3,5",
    "1e3",
    "1_000",
    str(INT_MAX),
    str(INT_MAX + 1),
    "007",
    "+5",
    "  ",
    "",
]

NULL_TEXT = ["nan", "NaN", "none", "None", "null", "NULL", "  none  "]

BOOLEAN_TEXT = [
    "true",
    "TRUE",
    "True",
    "false",
    "False",
    "1",
    "0",
    "yes",
    "no",
    "ano",
    "ne",
    "t",
    "f",
    "T",
    "F",
    " ano ",
    "mozna",
    "y",
    "n",
]

DATES = [
    "2026-08-07",
    "2026-08-07 12:34:56",
    "2026-08-07T12:34:56",
    "0000-00-00",
    "0000-00-00 00:00:00",
    "0000-00-00T00:00:00",
    "9999-12-31",
    "9999-12-31 00:00:00",
    "9999-12-31 23:59:59",
    "9999-12-31T00:00:00",
    "9999-12-31T23:59:59",
    "1900-01-01",
    "07/08/2026",
    "not a date",
    "2026-13-45",
    "2026-02-30",
    "  2026-08-07  ",
    "",
]

TIMES = [
    "12:34:56",
    "1:02:03",
    "25:00:00",
    "3 days 01:02:03",
    "00:00:00",
    "-01:00:00",
    "abc",
    "",
    dt.time(12, 34, 56),
    dt.time(0, 0, 0),
    pd.Timedelta(hours=25, minutes=1, seconds=2),
    pd.Timedelta(seconds=0),
    3600,
    0,
    -60,
    3661.7,
]

OBJECTS = [
    dt.date(2026, 8, 7),
    dt.datetime(2026, 8, 7, 12, 34, 56),
    pd.Timestamp("2026-08-07 12:34:56"),
    {"a": 1},
    {"a": "\u010d"},
    {"\u017e": "\u0159"},
    [1, 2],
    [],
    {},
    b"abc",
    bytearray(b"abc"),
    "text s \r\n koncem",
    "text s \r koncem",
    "\u010d" * 3,
]

EVERYTHING = (
    MISSING + NUMBERS + BOOLEANS + NUMERIC_TEXT + NULL_TEXT + BOOLEAN_TEXT + DATES + OBJECTS
)


def _same(left: object, right: object) -> bool:
    """Equality of the value **and** of the type.

    ``0 == False`` and ``1 == True``, so ``==`` alone would not reveal the swap. It does
    show up in ``row_hash`` though (``'0'`` vs ``'False'``).
    """
    if left is None or right is None:
        return left is None and right is None
    # `pd.NA == pd.NA` returns `pd.NA` and `bool(pd.NA)` raises, so identity has to be
    # checked before it ever gets to `==`.
    if left is pd.NA or right is pd.NA:
        return left is pd.NA and right is pd.NA
    if isinstance(left, float) and isinstance(right, float):
        return (np.isnan(left) and np.isnan(right)) or left == right
    if type(left) is not type(right):
        return False
    result = left == right
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _describe(value: object) -> str:
    return f"{value!r} ({type(value).__name__})"


# --- Scalar functions ------------------------------------------------------
# Which predecessor source is used follows the merge decisions: where the union or the
# richer variant was adopted, the oracle has to reach for the variant that was chosen.

SCALARS = [
    # (function name, new function, source with the winning variant, inputs)
    ("to_int_or_none", coerce.to_int_or_none, "A-mysql", EVERYTHING),
    ("to_int_or_na", coerce.to_int_or_na, "A-mysql", EVERYTHING),
    ("to_str_or_none", coerce.to_str_or_none, "A-mysql", EVERYTHING),
    ("_fmt_int_for_csv", coerce.fmt_int_for_csv, "A-mysql", EVERYTHING),
    # `to_bool`, `to_date_str`, `to_datetime_str`, `to_time_str` — the winner is the
    # richer variant, the one the PostgreSQL streaming extractor has (`S-pg`).
    ("to_bool", coerce.to_bool, "B-pg", EVERYTHING),
    ("to_date_str", coerce.to_date_str, "B-pg", [*EVERYTHING, dt.date(2026, 8, 7)]),
    ("to_datetime_str", coerce.to_datetime_str, "B-pg", EVERYTHING),
    ("to_time_str", coerce.to_time_str, "B-pg", TIMES),
    ("_is_pd_na", coerce.is_pd_na, "A-mysql", EVERYTHING),
    ("_clean", coerce.clean_invalid_date, "B-mysql", DATES + MISSING + NUMBERS),
]


@pytest.mark.parametrize(
    ("name", "new", "source_key", "inputs"), SCALARS, ids=lambda v: str(v)[:24]
)
def test_scalar_function_behaves_like_the_predecessor(name, new, source_key, inputs) -> None:
    predecessor = ro.get(source_key, name)
    differences = []
    for value in inputs:
        try:
            expected = predecessor(value)
        except Exception as exc:
            expected = f"<{type(exc).__name__}>"
        try:
            got = new(value)
        except Exception as exc:
            got = f"<{type(exc).__name__}>"
        if not _same(expected, got):
            differences.append(f"    {_describe(value)}: old={expected!r} new={got!r}")

    assert not differences, f"{name} diverges from the old implementation:\n" + "\n".join(
        differences
    )


def test_to_jsonb_both_variants() -> None:
    """`ensure_ascii` is a parameter — both of today's variants must be reproducible."""
    ascii_true = ro.get("A-mysql", "to_jsonb")  # json.dumps(x)
    ascii_false = ro.get("B-pg", "to_jsonb")  # json.dumps(x, ensure_ascii=False)

    for value in OBJECTS + MISSING + NUMBERS:
        assert _same(coerce.to_jsonb(value, ensure_ascii=True), ascii_true(value))
        assert _same(coerce.to_jsonb(value, ensure_ascii=False), ascii_false(value))


def test_to_jsonb_default_matches_the_majority() -> None:
    """The default `ensure_ascii=True` is what 7 of the 9 predecessor files do."""
    assert coerce.to_jsonb({"a": "\u010d"}) == '{"a": "\\u010d"}'
    assert coerce.to_jsonb({"a": "\u010d"}, ensure_ascii=False) == '{"a": "\u010d"}'


# --- Decoding bytes --------------------------------------------------------


def test_decode_value_both_encoding_orders() -> None:
    """Both of today's strategies must be reachable through a parameter."""
    utf8_first = ro.get("A-mysql", "decode_bytes_columns")  # only to check availability
    assert utf8_first is not None

    cascade = ro.get("B-mssql", "_decode_value")  # cp1250 -> utf-8 -> latin-1
    for text in [
        "d\xe1ma",
        "P\u0159\xedli\u0161 \u017elu\u0165ou\u010dk\xfd k\u016f\u0148",
        "abc",
        "",
    ]:
        raw = text.encode("cp1250")
        assert coerce.decode_value(raw, ("cp1250", "utf-8", "latin-1")) == cascade(raw)


def test_utf8_first_does_not_produce_mojibake() -> None:
    """Evidence for the decision about the order of encodings.

    `cp1250` is a single-byte encoding that rejects only five bytes (0x81 0x83 0x88 0x90
    0x98). UTF-8 text therefore usually passes through it — as mojibake. Which is why
    the order is supplied by the dialect, according to what its source really stores,
    and is not unified.
    """
    words = ["d\xe1ma", "n\xe1lada", "v\xedce", "\u017elu\u0165ou\u010dk\xfd"]
    for word in words:
        raw = word.encode("utf-8")
        assert coerce.decode_value(raw, ("utf-8", "cp1250")) == word
        # cp1250-first mangles the same input — which is why it is not a safe default.
        assert coerce.decode_value(raw, ("cp1250", "utf-8")) != word


def test_a_non_bytes_value_passes_through_unchanged() -> None:
    for value in ["text", 42, None, {"a": 1}]:
        assert coerce.decode_value(value) is value or coerce.decode_value(value) == value


# --- Functions over a DataFrame --------------------------------------------


def _frame() -> pd.DataFrame:
    """A representative batch: text, numbers, dates, bytes, missing values.

    Frozen like the vectors above — this frame is passed to the oracle, so neither its
    column names nor its values may change.
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "cislo": ["1", "3.5", "", None],
            "datum": ["2026-08-07", "0000-00-00", "9999-12-31", None],
            "cas": ["12:34:56", "25:00:00", None, "abc"],
            "text": ["a\r\nb", "c\rd", None, "\u017e"],
            "bajty": [b"abc", None, b"xyz", b""],
            "vlajka": ["true", "f", None, "mozna"],
        }
    )


def test_normalize_carriage_returns_like_the_predecessor() -> None:
    predecessor = ro.get("A-mysql", "_normalize_carriage_returns")
    expected = predecessor(_frame())
    got = coerce.normalize_carriage_returns(_frame())
    pd.testing.assert_frame_equal(got, expected)


def test_fix_invalid_dates_like_the_predecessor() -> None:
    """Characterisation against the oracle.

    This test was missing at first, and a regression slipped through because of it: the
    reference's vectorised `pd.to_datetime` over the whole series was replaced by
    element-by-element calls. The result stayed almost the same, but the run got 27x
    slower (0.32 s -> 8.9 s on 100 000 rows). Only a benchmark caught it.
    """
    predecessor = ro.get("A-mysql", "fix_invalid_dates")
    samples = pd.DataFrame(
        {
            "datum": [*DATES, "0", None, "  2026-01-01  "],
            "cas_text": [str(i) for i in range(len(DATES) + 3)],
        }
    )
    pd.testing.assert_frame_equal(
        coerce.fix_invalid_dates(samples.copy()), predecessor(samples.copy())
    )


def test_fix_invalid_dates_handles_mixed_types() -> None:
    """A column that cannot be parsed as a whole falls back to the per-element path."""
    predecessor = ro.get("A-mysql", "fix_invalid_dates")
    samples = pd.DataFrame({"datum": ["2026-08-07", 42, dt.date(2026, 1, 1), None, {"a": 1}]})
    pd.testing.assert_frame_equal(
        coerce.fix_invalid_dates(samples.copy()), predecessor(samples.copy())
    )


def test_fix_invalid_dates_removes_both_sentinels() -> None:
    """The union of the variants: the MySQL sentinel `0000-00-00` and `9999-12-31`."""
    out = coerce.fix_invalid_dates(_frame())
    assert out["datum"].tolist() == ["2026-08-07", None, None, None]


def test_fix_invalid_dates_respects_the_given_columns() -> None:
    out = coerce.fix_invalid_dates(_frame(), date_like_columns={"cas"})
    # the date column must not be cleaned when the caller did not name it
    assert out["datum"].tolist()[1] == "0000-00-00"


def test_sanitize_integer_columns_like_the_predecessor() -> None:
    predecessor = ro.get("A-mysql", "sanitize_integer_columns")
    expected = predecessor(_frame(), ["cislo", "id"])
    got = coerce.sanitize_integer_columns(_frame(), ["cislo", "id"])
    pd.testing.assert_frame_equal(got, expected)


def test_decode_bytes_columns_like_the_predecessor() -> None:
    """Against the utf-8 variant (MySQL/PG), which 8 of the 11 files use today."""
    predecessor = ro.get("A-mysql", "decode_bytes_columns")
    expected = predecessor(_frame())
    got = coerce.decode_bytes_columns(_frame(), ("utf-8",))
    pd.testing.assert_frame_equal(got, expected)


def test_replace_pdna_keeps_todays_trap() -> None:
    """In numeric columns `None` does not survive — today's behaviour, kept deliberately.

    Fixing it would change `row_hash`, so it does not belong here. It is a finding to be
    written up, not a bug to be quietly corrected.
    """
    predecessor = ro.get("A-mysql", "replace_pdna_with_none")
    for column in [
        pd.Series([1, None]),
        pd.Series([1, None], dtype="object"),
        pd.Series([1, None], dtype="Int64"),
        pd.Series(["x", None]),
    ]:
        expected = predecessor(pd.DataFrame({"a": column.copy()}))
        got = coerce.replace_pdna_with_none(pd.DataFrame({"a": column.copy()}))
        pd.testing.assert_frame_equal(got, expected)


def test_extract_date_columns_unifies_the_dialects() -> None:
    """Each dialect uses its own type names; unifying them must not lose any."""
    definitions = {
        "a": "date",
        "b": "datetime",  # MySQL
        "c": "timestamp without time zone",  # PostgreSQL
        "d": "datetime2",  # MSSQL
        "e": "smalldatetime",  # MSSQL
        "f": "varchar",
        "g": "int",
    }
    assert coerce.extract_date_columns(definitions) == {"a", "b", "c", "d", "e"}


def test_create_safe_dtype_mapping_takes_the_map_as_a_parameter() -> None:
    """The variants differed only in their type map — and that is dialect knowledge."""
    definitions = {"a": "int", "b": "varchar", "c": "unknown"}
    mysql_map = {"int": "INTEGER", "varchar": "VARCHAR"}

    assert coerce.create_safe_dtype_mapping_from_definitions(
        ["a", "b", "c"], definitions, mysql_map
    ) == {"a": "INTEGER", "b": "VARCHAR", "c": "TEXT"}

    # Without a map the type is used as it arrived — the behaviour of the fourth variant.
    assert coerce.create_safe_dtype_mapping_from_definitions(["a"], definitions) == {"a": "int"}


def test_create_safe_dtype_mapping_skips_generated() -> None:
    definitions = {"a": "generated", "b": "nan", "c": "", "d": None, "e": "int"}
    assert coerce.create_safe_dtype_mapping_from_definitions(
        ["a", "b", "c", "d", "e"], definitions
    ) == {"e": "int"}


# --- The guarded fast path for dates ---------------------------------------


TRICKY_COLUMNS = [
    ("ISO only", ["2026-08-07", "1999-12-31", "2026-01-01"]),
    ("the whole date battery", DATES),
    ("mixed formats", ["2026-08-07", "07/08/2026", "2026-08-07 12:00:00"]),
    ("ambiguous DD/MM", ["13/01/2026", "01/13/2026", "05/06/2026"]),
    ("with NULL and empty", ["2026-08-07", None, "", "0000-00-00", float("nan")]),
    ("outside the Timestamp range", ["9999-12-31", "1600-01-01", "2262-04-12"]),
    # These two broke the fast path — hence the guard.
    ("time zones", ["2026-08-07T12:00:00+02:00", "2026-08-07"]),
    ("numbers instead of text", [1234567890, "2026-08-07", 0]),
    ("objects", [dt.date(2026, 8, 7), dt.datetime(2026, 8, 7, 12, 0), None]),
    ("an empty column", []),
    ("all NULL", [None, None]),
]


@pytest.mark.parametrize(("name", "values"), TRICKY_COLUMNS, ids=lambda v: str(v)[:22])
def test_the_fast_date_path_gives_the_same_as_element_by_element(name, values) -> None:
    """Vectorisation over dates is **not** universally interchangeable.

    ``pd.to_datetime`` over a series differs from per-element calls in two documented
    cases: with mixed time zones it returns an ``object`` series with no ``.dt``, and it
    reads numbers as an epoch (``1234567890`` -> ``1970-01-01``), whereas ``to_date_str``
    returns ``None`` for a number.

    The fast path is therefore taken only where it is safe. This test guards that the
    safeguard does not break.
    """
    del name
    series = pd.Series(values, dtype="object")
    assert coerce.to_date_str_series(series) == [coerce.to_date_str(v) for v in series]
    assert coerce.to_datetime_str_series(series) == [coerce.to_datetime_str(v) for v in series]


def test_the_fast_path_really_is_used_when_it_can_be() -> None:
    """Without this the safeguard could be so strict that it never fires at all."""
    clean = pd.Series(["2026-08-07", "1999-12-31", None], dtype="object")
    assert coerce._fast_temporal_series(clean, coerce.INVALID_DATE_STRINGS, "%Y-%m-%d")

    with_numbers = pd.Series([1234567890, "2026-08-07"], dtype="object")
    assert (
        coerce._fast_temporal_series(with_numbers, coerce.INVALID_DATE_STRINGS, "%Y-%m-%d") is None
    )


#: The type map `fix_column_values` is compared against the reference on.
#: The time column is in it on purpose: the reference **does not** handle it (that is
#: `convert_time_columns`' job), and the text column is deliberately **absent** — it has
#: to fall into the last branch and be converted to a string even though the map has no
#: entry for it.
FIX_TYPE_MAP = {"cislo": "INTEGER", "datum": "DATE", "cas": "TIME", "vlajka": "BOOLEAN"}


def _frame_without_unified_values() -> pd.DataFrame:
    """`_frame`, minus the values on which the variants **deliberately** differ.

    ``'f'`` as ``False`` is understood only by the PostgreSQL variants — that is how
    ``psql`` prints booleans. Unifying that was an agreed decision, so what is tested
    here is the rest: the order and the structure of the pass.
    """
    df = _frame()
    df["vlajka"] = ["true", "false", None, "mozna"]
    return df


@pytest.mark.parametrize("short", sorted(ro.variants("fix_column_values")))
def test_fix_column_values_matches_the_predecessor(short: str) -> None:
    """Against **all 9 copies**, not against what one imagines the function does.

    There used to be hand-written expectations here and they were wrong: they codified
    that the iteration goes over the type map (not over the frame's columns), that
    ``TIME`` is converted here, and that an integer column ends up as ``object``. The
    reference does all three differently, and the difference would only show up in the
    data in the target.
    """
    predecessor = ro.variants("fix_column_values")[short]
    new = coerce.fix_column_values(_frame_without_unified_values(), dict(FIX_TYPE_MAP))
    original = predecessor(_frame_without_unified_values(), dict(FIX_TYPE_MAP))
    pd.testing.assert_frame_equal(new, original)


@pytest.mark.parametrize("short", ["A-pg", "B-pg"])
def test_fix_column_values_matches_on_postgres_booleans(short: str) -> None:
    """With ``'t'``/``'f'`` we match the PostgreSQL variants — they know them too.

    The other seven files return ``None`` for those; the superset is an agreed decision
    (a source never produces a value belonging to a foreign dialect).
    """
    predecessor = ro.variants("fix_column_values")[short]
    new = coerce.fix_column_values(_frame(), dict(FIX_TYPE_MAP))
    original = predecessor(_frame(), dict(FIX_TYPE_MAP))
    pd.testing.assert_frame_equal(new, original)


def test_fix_column_values_converts_a_column_outside_the_map() -> None:
    """A column with no entry in the map gets ``TEXT``, not "leave it alone".

    ``2.0`` -> ``'2'``: were the column left alone, a different value would end up in the
    target. This is exactly the difference the hand-written test used to miss.
    """
    df = pd.DataFrame({"outside_the_map": [2.0, 3.5, None]})
    out = coerce.fix_column_values(df, {"other": "INTEGER"})
    assert out["outside_the_map"].tolist() == ["2", "3.5", None]


def test_fix_column_values_leaves_time_alone() -> None:
    """``TIME`` is handled by `convert_time_columns`; a second pass would break it."""
    df = pd.DataFrame({"cas": ["12:34:56", "25:00:00"]})
    out = coerce.fix_column_values(df, {"cas": "TIME"})
    assert out["cas"].tolist() == ["12:34:56", "25:00:00"]


def test_fix_column_values_returns_an_integer_column_as_int64() -> None:
    """Both `sanitize_integer_columns` and `fmt_int_for_csv` build on this."""
    out = coerce.fix_column_values(_frame(), dict(FIX_TYPE_MAP))
    assert str(out["cislo"].dtype) == "Int64"


def test_fix_column_values_propagates_ensure_ascii() -> None:
    df = pd.DataFrame({"j": [{"a": "\u010d"}]})
    assert coerce.fix_column_values(df.copy(), {"j": "JSONB"})["j"][0] == '{"a": "\\u010d"}'
    out = coerce.fix_column_values(df.copy(), {"j": "JSONB"}, ensure_ascii=False)
    assert out["j"][0] == '{"a": "\u010d"}'


def test_convert_integer_columns_keeps_none() -> None:
    out = coerce.convert_integer_columns(_frame(), {"cislo": "INTEGER"})
    assert out["cislo"].isna().tolist() == [False, True, True, True]
    assert str(out["cislo"].dtype) == "Int64"


def test_convert_time_columns_knows_the_pg_type_names() -> None:
    """The superset of both variants: `time` and `time without time zone`."""
    df = _frame()
    out = coerce.convert_time_columns(df, {"cas": "X"}, {"cas": "time without time zone"})
    assert out["cas"].tolist() == ["12:34:56", "25:00:00", None, None]


@pytest.mark.parametrize("short", sorted(ro.variants("update_stats")))
def test_update_stats_matches_the_predecessor(short: str) -> None:
    """Against all 7 copies (a single variant).

    The original test here checked ``rows``/``nulls``, which the reference does not have
    at all — it codified an implementation that had been imagined rather than read. The
    real function collects ``int_min``/``int_max``/``seen_integer``/``has_non_numeric``,
    because the type promotion after the first batch (`strategies.full`) rests on them.
    And ``all_columns`` is **filled**, not read: the original condition
    ``if col_name not in all_columns: return`` meant that with an empty set nothing was
    computed and the promotion silently never happened.
    """
    predecessor = ro.variants("update_stats")[short]
    # ``"sloupec_id"`` is frozen: the column name goes into the oracle fingerprint, so
    # renaming it would turn every call below into an unrecorded one.
    cases = {
        "enum 0/1": pd.Series(["0", "1", "0"]),
        "integers": pd.Series([1, 2, 300000]),
        "above the INT threshold": pd.Series([1, 3_000_000_000]),
        "decimals": pd.Series([1.5, 2.0]),
        "text": pd.Series(["a", "b"]),
        "all missing": pd.Series([None, None], dtype=object),
        "numbers and text mixed": pd.Series([1, "x"]),
        "numeric text": pd.Series(["10", "20"]),
    }
    for name, series in cases.items():
        old_cols: set = set()
        old_stats: dict = {}
        new_cols: set = set()
        new_stats: dict = {}
        predecessor("sloupec_id", series, old_cols, old_stats)
        coerce.update_stats("sloupec_id", series, new_cols, new_stats)
        assert new_stats == old_stats, name
        assert new_cols == old_cols == {"sloupec_id"}, name


def test_update_stats_fills_all_columns() -> None:
    """``all_columns`` is an output parameter, not a filter."""
    seen: set = set()
    coerce.update_stats("a", pd.Series([1]), seen, {})
    coerce.update_stats("b", pd.Series([2]), seen, {})
    assert seen == {"a", "b"}


def test_update_stats_stops_collecting_after_a_non_number() -> None:
    """``has_non_numeric`` wins — the type is then not promoted."""
    stats: dict = {}
    coerce.update_stats("x", pd.Series([1, "abc"]), set(), stats)
    assert stats["x"]["has_non_numeric"] is True
    assert stats["x"]["int_max"] is None
