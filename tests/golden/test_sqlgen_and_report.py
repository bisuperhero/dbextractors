"""Tests for the SQL generation and the rendering of the report. No database."""

from __future__ import annotations

import json

import pytest

from dbextractors.golden import report as report_mod
from dbextractors.golden import sqlgen
from dbextractors.golden.model import (
    VERDICT_DIFF,
    VERDICT_MATCH,
    BatchReport,
    Difference,
    Level,
    LevelResult,
    Relation,
    TableReport,
)

# --- Classifying types ------------------------------------------------------


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [
        ("integer", sqlgen.Family.NUMERIC_EXACT),
        ("numeric", sqlgen.Family.NUMERIC_EXACT),
        ("double precision", sqlgen.Family.NUMERIC_FLOAT),
        ("boolean", sqlgen.Family.BOOLEAN),
        ("timestamp with time zone", sqlgen.Family.TEMPORAL),
        ("date", sqlgen.Family.TEMPORAL),
        ("text", sqlgen.Family.TEXTUAL),
        ("character varying", sqlgen.Family.TEXTUAL),
        ("bytea", sqlgen.Family.BINARY),
        ("jsonb", sqlgen.Family.JSON_CANONICAL),
        ("json", sqlgen.Family.JSON_LOOSE),
        ("ARRAY", sqlgen.Family.OTHER),
        ("unknown type", sqlgen.Family.OTHER),
    ],
)
def test_it_classifies_the_type(data_type, expected):
    assert sqlgen.classify(data_type) is expected


def test_json_and_arrays_are_declared_approximate():
    """Where the comparison is necessarily approximate, the report has to say so."""
    assert sqlgen.Family.JSON_LOOSE in sqlgen.APPROXIMATE_FAMILIES
    assert sqlgen.Family.OTHER in sqlgen.APPROXIMATE_FAMILIES
    # jsonb has a canonical representation, so it is not approximate
    assert sqlgen.Family.JSON_CANONICAL not in sqlgen.APPROXIMATE_FAMILIES


# --- Quoting ----------------------------------------------------------------


def test_it_quotes_reserved_words():
    """Target tables have columns such as `_type` and `_order` — see the column
    naming section of ``docs/legacy-compat.md``."""
    assert sqlgen.quote_ident("_type") == '"_type"'
    assert sqlgen.quote_ident("_order") == '"_order"'


def test_it_doubles_the_quotes_in_a_name():
    assert sqlgen.quote_ident('odd"name') == '"odd""name"'


# --- Aggregates -------------------------------------------------------------


def test_float_excludes_nan_and_infinity():
    """A single NaN would swallow the whole sum through ::numeric."""
    agg = sqlgen.build_column_aggregates("v", sqlgen.Family.NUMERIC_FLOAT)
    assert "nonfinite" in agg.expressions
    assert "FILTER" in agg.expressions["sum"]


def test_numbers_are_summed_through_numeric():
    """sum() over float8 depends on the order, over numeric it does not."""
    agg = sqlgen.build_column_aggregates("v", sqlgen.Family.NUMERIC_EXACT)
    assert "::numeric" in agg.expressions["sum"]


def test_every_column_counts_its_nulls():
    """NULL against an empty string is a trap the comparison has to catch."""
    for family in sqlgen.Family:
        agg = sqlgen.build_column_aggregates("v", family)
        assert "nulls" in agg.expressions


def test_text_uses_two_independent_hashes():
    agg = sqlgen.build_column_aggregates("v", sqlgen.Family.TEXTUAL)
    assert agg.expressions["h1"] != agg.expressions["h2"]
    assert "sum(" in agg.expressions["h1"], "a sum is commutative, hence order independent"


def test_one_query_per_table_not_per_column():
    """On a table of 15 M rows every pass is expensive."""
    sql, aggregates = sqlgen.build_checksum_query(
        '"s"."t"',
        [("a", sqlgen.Family.NUMERIC_EXACT), ("b", sqlgen.Family.TEXTUAL)],
    )
    assert sql.count("FROM") == 1
    assert len(aggregates) == 2


def test_the_result_is_split_up_in_the_order_of_the_aggregates():
    _, aggregates = sqlgen.build_checksum_query(
        '"s"."t"', [("a", sqlgen.Family.BOOLEAN), ("b", sqlgen.Family.BOOLEAN)]
    )
    width = 1 + sum(len(a.expressions) for a in aggregates)
    row = tuple(range(width))
    parsed = sqlgen.parse_checksum_row(row, aggregates)
    assert parsed["row_count"] == 0
    assert set(parsed) == {"row_count", "a", "b"}
    assert set(parsed["a"]) == set(aggregates[0].expressions)


# --- Report -----------------------------------------------------------------


def _report(passed: bool = True) -> TableReport:
    report = TableReport(
        left=Relation("stg_legacy", "invoices"),
        right=Relation("dbx_golden_x", "invoices"),
        duration_s=1.25,
    )
    report.levels.append(
        LevelResult(
            level=Level.EXISTENCE_AND_ROWS,
            passed=True,
            stats={
                "exists": {"left": True, "right": True},
                "rows": {"left": 1234567, "right": 1234567, "delta": 0},
            },
        )
    )
    report.levels.append(
        LevelResult(
            level=Level.COLUMN_NAMES,
            passed=passed,
            stats={"counts": {"left": 42, "right": 42}},
            differences=[]
            if passed
            else [Difference(what="column missing on the right", left="_type", where="_type")],
        )
    )
    for level in (Level.COLUMN_TYPES, Level.COLUMN_CHECKSUMS, Level.ROW_HASHES):
        report.levels.append(LevelResult(level=level, passed=True, stats={"key": ["id"]}))
    return report


def test_the_verdict_is_a_match_when_everything_fits():
    assert _report(passed=True).verdict == VERDICT_MATCH


def test_the_verdict_is_a_diff_and_names_the_first_failed_level():
    report = _report(passed=False)
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.COLUMN_NAMES


def test_an_error_is_not_confused_with_a_difference():
    """ "It is different" and "we do not know whether it is different" are not the
    same thing."""
    report = _report(passed=True)
    report.error = "OperationalError: the connection dropped"
    assert report.verdict == "ERROR"


def test_a_skipped_level_is_not_a_failure():
    report = _report(passed=True)
    report.levels.append(LevelResult(level=Level.ROW_HASHES, passed=False, skipped=True))
    assert report.first_failed_level is None


def test_the_text_report_is_readable():
    text = report_mod.render_table(_report(passed=False))
    assert "DIFF" in text
    assert "broke at level 2" in text
    assert "column names" in text
    assert "1 234 567" in text, "numbers grouped in thousands, so they can be read at a glance"
    assert "_type" in text


def test_json_and_text_come_from_the_same_source():
    report = _report(passed=False)
    payload = json.loads(report_mod.to_json(report))
    assert payload["verdict"] == VERDICT_DIFF
    assert payload["first_failed_level"] == 2
    assert payload["levels"][1]["differences"][0]["left"] == "_type"


def test_an_approximate_match_is_flagged_in_the_summary_too():
    report = _report(passed=True)
    report.level(Level.COLUMN_CHECKSUMS).approximate = True

    batch = BatchReport(reports=[report])
    text = report_mod.render_batch(batch)
    assert "approximate" in text
    assert "not proof that the data matches" in text


def test_the_summary_tells_apart_rows_with_the_same_left_table():
    """In the manifest the same source table routinely appears more than once.

    Were only the left side printed, two problem rows would look identical in a
    summary of 250 tables and there would be no telling which is which.
    """
    first, second = _report(passed=False), _report(passed=False)
    first.label = "invoices — MySQL"
    second.label = "invoices — MSSQL"

    text = report_mod.render_batch(BatchReport(reports=[first, second]))
    assert "invoices — MySQL" in text
    assert "invoices — MSSQL" in text


def test_the_summary_without_a_label_prints_both_sides():
    report = _report(passed=False)
    assert report.label is None
    text = report_mod.render_batch(BatchReport(reports=[report]))
    assert "stg_legacy.invoices → dbx_golden_x.invoices" in text


def test_the_batch_summary_prints_only_the_problems():
    ok, bad = _report(passed=True), _report(passed=False)
    text = report_mod.render_batch(BatchReport(reports=[ok, bad, ok]))
    assert "3 tables: 2 match, 1 diff, 0 error" in text
    assert text.count("DIFF") >= 1


# --- A wide table: the limit of 100 arguments -------------------------------


def test_concat_ws_nests_above_a_hundred_columns() -> None:
    """PostgreSQL does not allow more than 100 arguments to a function.

    One real table has 156 columns and level 5 ended on it with the error
    "cannot pass more than 100 arguments to a function" — the table then came out
    as an ERROR even though all the other levels matched.
    """
    from dbextractors.golden.compare import _concat_ws

    assert _concat_ws([f"c{i}" for i in range(156)]).count("concat_ws") > 1


def test_concat_ws_stays_flat_below_the_limit() -> None:
    """So that the expression is not folded where there is no need."""
    from dbextractors.golden.compare import _concat_ws

    assert _concat_ws(["a", "b", "c"]).count("concat_ws") == 1
    assert _concat_ws(["a"]) == "a"


@pytest.mark.needs_pg
def test_a_wide_table_only_goes_through_once_nested(ro_conn) -> None:
    """Evidence that this is not mere caution — the flat variant fails on 156 columns."""
    import psycopg2

    from dbextractors.golden.compare import _concat_ws

    parts = [f"quote_nullable({i}::text)" for i in range(156)]

    def flat_fails() -> None:
        with ro_conn.cursor() as cur:
            cur.execute("SELECT concat_ws(E'\\x1f', " + ", ".join(parts) + ")")

    with pytest.raises(psycopg2.errors.TooManyArguments):
        flat_fails()
    ro_conn.rollback()

    with ro_conn.cursor() as cur:
        cur.execute(f"SELECT {_concat_ws(parts)}")
        assert cur.fetchone()[0].count("\x1f") == 155


@pytest.mark.needs_pg
def test_nesting_gives_the_same_string(ro_conn) -> None:
    """**The essential part**: nesting must not change the result.

    ``concat_ws`` skips only ``NULL`` arguments and none arrive here — they have
    all been through ``quote_nullable``. Verified against a live database, not by
    reasoning.
    """
    from dbextractors.golden.compare import _concat_ws

    parts = [f"quote_nullable({i}::text)" for i in range(1, 96)]
    flat = "concat_ws(E'\\x1f', " + ", ".join(parts) + ")"
    with ro_conn.cursor() as cur:
        cur.execute(f"SELECT {flat}, {_concat_ws(parts)}")
        left, right = cur.fetchone()
    assert left == right
