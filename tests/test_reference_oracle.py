"""Tests for the oracle — the tool that mines the old functions out of the predecessor sources.

Every characterisation test in the suite rests on this. If the oracle returns
something other than what the real old function returns, we freeze the wrong
behaviour and the golden test will not catch it — it would be comparing two
equally wrong implementations.

**The whole file is live-only.** It does not test the package's behaviour but the
AST mining itself — something that does not exist without the predecessor sources.
In a repository without them it is skipped, and rightly so: replaying a test of a
tool that reads files which are not there would verify nothing.
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

import reference_oracle as ro

pytestmark = pytest.mark.skipif(
    ro.MODE == "replay",
    reason="tests the AST mining of the predecessor sources, which this repository lacks",
)

# The functions the conversion, retry and tunnel characterisation tests need.
EXPECTED_FUNCTIONS = [
    "to_int_or_none",
    "to_int_or_na",
    "to_date_str",
    "to_datetime_str",
    "to_bool",
    "to_jsonb",
    "to_str_or_none",
    "to_time_str",
    "_clean",
    "_fix",
    "_is_pd_na",
    "fix_invalid_dates",
    "fix_column_values",
    "convert_time_columns",
    "convert_integer_columns",
    "sanitize_integer_columns",
    "replace_pdna_with_none",
    "_fmt_int_for_csv",
    "decode_bytes_columns",
    "_decode_value",
    "extract_date_columns",
    "create_safe_dtype_mapping_from_definitions",
    "update_stats",
    "_normalize_carriage_returns",
    "with_retry",
    "wait_for_port",
    "find_free_port",
    "ensure_private_key_permissions",
    "normalize_private_key_contents",
    "_ssh_tunnel_preexec",
    "_resolve",
    "is_truthy",
]


@pytest.mark.parametrize("short", sorted(ro.FILES))
def test_source_can_be_loaded(short: str) -> None:
    assert ro._namespace(short)


@pytest.mark.parametrize("function", EXPECTED_FUNCTIONS)
def test_expected_function_is_available(function: str) -> None:
    """If one of them were missing, no characterisation test could be written for it."""
    assert ro.variants(function), f"{function} could not be extracted from any source"


def test_unknown_name_fails_clearly() -> None:
    with pytest.raises(ro.OracleError, match="could not be extracted"):
        ro.get("A-mysql", "function_that_does_not_exist")


def test_unknown_source_key_fails() -> None:
    with pytest.raises(ro.OracleError, match="Unknown source"):
        ro.get("neexistuje", "to_bool")


# --- Safety ---------------------------------------------------------------


def test_loading_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The most important test in this module.**

    The predecessor sources have lines such as ``engine = create_engine(...)`` and
    ``conn = psycopg2.connect(...)`` in the body of ``load_data``. If the oracle
    evaluated constants blindly ("try it and catch the exception"), such lines
    would succeed — and merely running the tests would reach a production database.

    ``_is_pure_value`` is the safeguard. This test checks that it holds.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the oracle tried to open a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    ro._namespace.cache_clear()
    try:
        for short in ro.FILES:
            ro._namespace(short)
    finally:
        ro._namespace.cache_clear()


@pytest.mark.parametrize(
    "source",
    [
        "engine = create_engine('postgresql://x')",
        "conn = psycopg2.connect(dsn)",
        "loader = Postgres.with_config(cfg)",
        "df = pd.read_sql(query, engine)",
        "x = subprocess.run(['ssh'])",
    ],
)
def test_impure_assignments_are_not_evaluated(source: str) -> None:
    import ast

    node = ast.parse(source).body[0]
    assert not ro._is_pure_value(node.value)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "source",
    [
        "INT_MAX = 2_147_483_647",
        "TYPES = {'int': 'INTEGER', 'text': 'TEXT'}",
        "PATTERNS = [re.compile(r'^0{4}-0{2}-0{2}$')]",
        "NAMES = frozenset({'a', 'b'})",
    ],
)
def test_pure_constants_are_evaluated(source: str) -> None:
    import ast

    node = ast.parse(source).body[0]
    assert ro._is_pure_value(node.value)  # type: ignore[attr-defined]


# --- Mined functions really do behave like functions -----------------------


def test_nested_functions_are_found_at_any_depth() -> None:
    """``to_int_or_none`` is nested inside ``sanitize_integer_columns`` and
    ``to_time_str`` inside ``convert_time_columns``. If only definitions directly
    inside ``load_data`` were taken, half the functions would be missing."""
    assert ro.get("A-mysql", "to_int_or_none")("42") == 42
    assert ro.get("A-mysql", "to_time_str") is not None


def test_functions_can_see_each_other() -> None:
    """``replace_pdna_with_none`` calls ``_is_pd_na``. Flattening them into a single
    namespace has to keep that working.

    The text column is deliberate: for numeric columns that function does not
    return ``None`` (see ``tests/coerce/test_characterization_coerce.py``), so this
    test would say nothing about the functions being wired together."""
    result = ro.get("A-mysql", "replace_pdna_with_none")(pd.DataFrame({"a": ["x", None]}))
    assert result["a"].tolist() == ["x", None]


def test_depends_on_a_constant_from_the_load_data_body() -> None:
    """``_clean`` inside ``fix_invalid_dates`` uses ``invalid_date_patterns``, a list
    of ``re.compile(...)`` defined only in the body of the enclosing function."""
    clean = ro.get("A-mysql", "_clean")
    assert clean("0000-00-00") is None
    assert clean("2026-08-07") == "2026-08-07"


def test_logging_survives_and_can_be_read() -> None:
    ro.get("A-mysql", "sanitize_integer_columns")(pd.DataFrame({"a": ["1", "x"]}), ["a"], True)
    assert isinstance(ro.logger_of("A-mysql").messages, list)
