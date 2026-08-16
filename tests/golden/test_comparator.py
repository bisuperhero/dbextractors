"""A test of the test — it verifies that the harness **can find a difference**.

Two schemas are built deliberately differing in (a) one missing row, (b) a
renamed column, (c) a changed value in one field, (d) a different column type,
and each of those has to be reported. A harness that always says "match" is worse
than no harness at all.

Each of those four cases has its own test, and each checks not only **that** the
difference was found but also **at which level** — because the level is what the
decision about the table then rests on.
"""

from __future__ import annotations

import pytest

from dbextractors.golden.compare import CompareOptions, compare_tables
from dbextractors.golden.model import (
    VERDICT_DIFF,
    VERDICT_MATCH,
    Level,
    Relation,
)

pytestmark = pytest.mark.needs_pg


# --- Test data --------------------------------------------------------------

BASE_DDL = """
CREATE TABLE {rel} (
    id            integer      NOT NULL,
    "_type"       text,
    "_name"       text,
    price         numeric(12,2),
    quantity      double precision,
    valid_from    date,
    changed_at    timestamptz,
    active        boolean,
    note          text,
    payload       jsonb,
    row_hash      text,
    _timestamp    timestamptz,
    _deleted_in_source boolean NOT NULL DEFAULT false
);
"""

BASE_ROWS = """
INSERT INTO {rel} (id, "_type", "_name", price, quantity, valid_from, changed_at,
                   active, note, payload, row_hash, _timestamp)
SELECT
    g,
    'type_' || mod(g, 3),
    'name ' || g,
    (g * 1.5)::numeric(12,2),
    g * 0.1,
    date '2026-01-01' + g,
    timestamptz '2026-01-01 00:00:00+00' + (g || ' hours')::interval,
    (mod(g, 2) = 0),
    CASE WHEN mod(g, 5) = 0 THEN NULL WHEN mod(g, 7) = 0 THEN '' ELSE 'note ' || g END,
    jsonb_build_object('k', g),
    md5('row' || g),
    now()
FROM generate_series(1, {rows}) g;
"""


def _make_table(conn, schema: str, table: str, rows: int = 200) -> Relation:
    relation = Relation(schema=schema, table=table)
    with conn.cursor() as cur:
        cur.execute(BASE_DDL.format(rel=relation.qualified()))
        cur.execute(BASE_ROWS.format(rel=relation.qualified(), rows=rows))
        cur.execute(f"CREATE UNIQUE INDEX ON {relation.qualified()} (id)")
    return relation


@pytest.fixture()
def pair(rw_conn, workspace):
    """Two identical tables in the same scratch schema."""
    left = _make_table(rw_conn, workspace, "old")
    right = _make_table(rw_conn, workspace, "new")
    return left, right


def _compare(conn, pair, **kwargs):
    left, right = pair
    return compare_tables(conn, left, right, CompareOptions(**kwargs))


def _failed_levels(report) -> set:
    return {r.level for r in report.levels if not r.passed and not r.skipped}


# --- The baseline: a match --------------------------------------------------


def test_identical_tables_give_a_match(ro_conn, pair):
    """Without this test all the others could be passing by accident.

    Watch out for `_timestamp`: both tables have it filled in with `now()`, that
    is, differently. Were the harness to compare it, it would report a difference
    on every table forever. That is why it is in `VOLATILE_COLUMNS` — but only for
    comparing *data*; its existence is checked at levels 2 and 3.
    """
    report = _compare(ro_conn, pair)
    assert report.error is None, report.error
    assert report.verdict == VERDICT_MATCH
    assert _failed_levels(report) == set()
    assert len(report.levels) == 5


def test_a_report_from_a_real_run_serialises_to_json(ro_conn, rw_conn, pair):
    """A regression test.

    `sum()` over `bigint` returns `numeric` in PostgreSQL, that is a `Decimal`,
    which `json.dumps` cannot handle. It would only show up when writing the
    report — at the end of a batch of over 250 tables, after half an hour of
    computing.
    """
    import json

    from dbextractors.golden import report as report_mod

    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {right.qualified()} SET note = 'other' WHERE id = 5")

    payload = json.loads(report_mod.to_json(_compare(ro_conn, pair)))
    assert payload["verdict"] == VERDICT_DIFF
    assert payload["levels"], "the report has to keep its levels after serialisation"


def test_a_match_does_not_change_on_a_repeated_run(ro_conn, pair):
    """Idempotence. Two runs over the same data have to give the same result."""
    first = _compare(ro_conn, pair)
    second = _compare(ro_conn, pair)
    assert first.to_dict()["levels"] == second.to_dict()["levels"]


# --- (a) a missing row ------------------------------------------------------


def test_it_finds_a_missing_row(ro_conn, rw_conn, pair):
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"DELETE FROM {right.qualified()} WHERE id = 42")

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.EXISTENCE_AND_ROWS

    level1 = report.level(Level.EXISTENCE_AND_ROWS)
    assert level1.stats["rows"] == {"left": 200, "right": 199, "delta": -1}

    # It does not stop at the first mismatch: level 5 has to name that row.
    level5 = report.level(Level.ROW_HASHES)
    assert level5.stats["only_in_left"] == 1
    assert level5.stats["only_in_right"] == 0
    # The key is serialised through `quote_nullable` so that NULL does not merge
    # with the string "NULL" — hence the value in quotes.
    assert any(s["key"] == "'42'" for s in level5.stats["samples"])


# --- (b) a renamed column ---------------------------------------------------


def test_it_finds_a_renamed_column(ro_conn, rw_conn, pair):
    """The most important case of the whole test.

    The legacy write path prefixes 825 reserved words with an underscore, which is
    where `_type` in 113 tables comes from. Were the new component not to repeat
    that prefix, exactly this difference would arise — and it would silently break
    the dbt layer.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f'ALTER TABLE {right.qualified()} RENAME COLUMN "_type" TO "type"')

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.COLUMN_NAMES

    level2 = report.level(Level.COLUMN_NAMES)
    described = {(d.what, d.left or d.right) for d in level2.differences}
    assert ("column missing on the right", "_type") in described
    assert ("column only on the right", "type") in described


def test_it_finds_a_changed_column_order(ro_conn, rw_conn, workspace):
    """The order is part of the contract, even when the names and the data match."""
    left = _make_table(rw_conn, workspace, "order_a", rows=20)
    right = Relation(schema=workspace, table="order_b")
    with rw_conn.cursor() as cur:
        # The same columns, with the first two swapped.
        cur.execute(
            f"CREATE TABLE {right.qualified()} AS "
            f'SELECT "_type", id, "_name", price, quantity, valid_from, changed_at, '
            f"active, note, payload, row_hash, _timestamp, _deleted_in_source "
            f"FROM {left.qualified()}"
        )

    report = compare_tables(ro_conn, left, right, CompareOptions(key_columns=["id"]))
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.COLUMN_NAMES

    level2 = report.level(Level.COLUMN_NAMES)
    assert any(d.what == "different column order" for d in level2.differences)
    assert any("only the order differs" in note for note in level2.notes)


# --- (c) a changed value ----------------------------------------------------


def test_it_finds_a_changed_value(ro_conn, rw_conn, pair):
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {right.qualified()} SET price = price + 1 WHERE id = 7")

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    # The row count and the column names match — it has to break on the checksums.
    assert report.first_failed_level == Level.COLUMN_CHECKSUMS

    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert {d.where for d in level4.differences} == {"price"}


def test_it_finds_a_changed_value_in_text(ro_conn, rw_conn, pair):
    """A text column is compared by hash, not by sum — a separate code path."""
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {right.qualified()} SET note = 'other' WHERE id = 11")

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert "note" in {d.where for d in level4.differences}


def test_it_tells_null_from_an_empty_string(ro_conn, rw_conn, pair):
    """A trap worth calling out explicitly.

    The `note` column has both in the test data. Swapping NULL for an empty string
    changes neither the row count nor the type — it has to show up on the
    checksums.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {right.qualified()} SET note = '' WHERE note IS NULL")

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert "note" in {d.where for d in level4.differences}
    assert any("nulls" in d.what for d in level4.differences)


def test_it_finds_values_swapped_between_rows(ro_conn, rw_conn, pair):
    """The sum and the count stay the same, only which row holds what changes.

    Level 4 cannot catch this in principle — which is why level 5 exists.

    Watch out: `row_hash` is computed from `id` alone in the test data, so a swap
    leaves it **unchanged**. What has to catch it is therefore the content hash,
    not the stored one — precisely the case level 5 computes both for.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(
            f"UPDATE {right.qualified()} SET price = "
            f"CASE id WHEN 3 THEN (SELECT price FROM {right.qualified()} WHERE id = 4) "
            f"        ELSE (SELECT price FROM {right.qualified()} WHERE id = 3) END "
            f"WHERE id IN (3, 4)"
        )

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF

    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert level4.passed, "a swap must not change the checksums — that is what level 5 is for"

    level5 = report.level(Level.ROW_HASHES)
    assert not level5.passed
    assert level5.stats["hash_differs"] == 0, "the stored row_hash does not depend on `price`"
    assert level5.stats["content_differs"] == 2
    assert {s["key"] for s in level5.stats["samples"]} == {"'3'", "'4'"}


def test_an_empty_row_hash_on_the_left_does_not_drown_the_data_figure(ro_conn, rw_conn, pair):
    """A difference in `row_hash` must not be counted into "different data" as well.

    Found on a real 16-row table: level 5 reported "different row_hash 16,
    different data 16" for a table whose data differed in not a single column. The
    cause was that `row_hash` also entered the **content** hash — and when the old
    side never filled it in, it differed on every row.

    It is a real state, not a made-up one: in one predecessor's PostgreSQL
    extractor the full-load path is the only one that does not call the
    hash-and-timestamp step, so on a `full` load `row_hash` stays empty.

    The "different data" figure is meant to answer the question *does the data
    differ?* — and here the answer is no.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {left.qualified()} SET row_hash = NULL")

    report = _compare(ro_conn, pair)
    level5 = report.level(Level.ROW_HASHES)

    assert level5.stats["hash_differs"] > 0, "an empty hash on the left is a finding in itself"
    assert level5.stats["content_differs"] == 0, "the data does not differ, and it has to say so"
    assert any("counted twice" in note for note in level5.notes)

    # And level 4 has to say why it is empty — so that it does not look like a
    # regression in dbextractors.
    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert any("is empty on the left side" in note for note in level4.notes)


def test_an_empty_row_hash_does_not_hide_a_real_change(ro_conn, rw_conn, pair):
    """Keeping `row_hash` out of the content hash must not blind level 5."""
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {left.qualified()} SET row_hash = NULL")
        cur.execute(f"UPDATE {right.qualified()} SET note = 'tampered' WHERE id = 99")

    level5 = _compare(ro_conn, pair).level(Level.ROW_HASHES)

    assert level5.stats["content_differs"] == 1, "a change in the data still has to be found"


def test_it_finds_a_change_in_a_column_outside_row_hash(ro_conn, rw_conn, pair):
    """The trickiest case of the whole test.

    The configuration contract has both `hash_include_columns` and
    `hash_exclude_columns`, so `row_hash` is routinely computed from only some of
    the columns. Were level 5 to compare only that, a change in an excluded column
    would pass as a match — and that is exactly the kind of silent error the
    golden test exists for.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        # `row_hash` depends only on `id` in the test data, so this UPDATE leaves
        # it unchanged.
        cur.execute(f"UPDATE {right.qualified()} SET note = 'tampered' WHERE id = 99")

    report = _compare(ro_conn, pair)
    level5 = report.level(Level.ROW_HASHES)
    assert level5.stats["hash_differs"] == 0
    assert level5.stats["content_differs"] == 1
    assert not level5.passed
    assert any("hash_exclude_columns" in note for note in level5.notes)


# --- (d) a different column type --------------------------------------------


def test_a_type_mismatch_does_not_drown_the_finding_at_level_5(ro_conn, rw_conn, pair):
    """A regression test for a bug that only a sample run brought to light.

    `double precision` and `numeric` have different textual forms (`0.25` vs
    `0.2500`), so a column with a mismatched type differs in **every** row. Were it
    to enter the content hash, level 5 would report a difference on all 50 000 rows
    and the one genuinely changed row would be lost in it.

    Level 5 therefore leaves out columns with a type mismatch just as level 4 does
    — level 3 has already reported them.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {right.qualified()} ALTER COLUMN quantity TYPE numeric(12,4)")
        cur.execute(f"UPDATE {right.qualified()} SET note = 'tampered' WHERE id = 123")

    report = _compare(ro_conn, pair)
    level5 = report.level(Level.ROW_HASHES)
    assert level5.stats["content_differs"] == 1, "one row is to be found, not all 200"
    assert {s["key"] for s in level5.stats["samples"]} == {"'123'"}
    assert any("with a type mismatch" in note for note in level5.notes)


# --- (d2) a different column type we nevertheless want compared -------------
#
# Leaving them out by default is right for Firebird sources, where it is one or
# two columns. With **PostgreSQL sources**, however, the old side misses the type
# map and falls back to `TEXT` — on one wide test table for 74 of 92 columns.
# Leaving out three quarters of the columns means "MATCH" no longer says
# anything, and that is exactly what `compare_type_mismatch_as_text` is there to
# prevent.
#
# The option **must not** mask the type difference: level 3 has to keep reporting
# it. Were it to swallow it, the option would be a tool for turning reports green.


def test_as_text_finds_a_change_in_a_column_with_a_different_type(ro_conn, rw_conn, pair):
    """With the option on, a changed value is found even in a column with a
    different type."""
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {right.qualified()} ALTER COLUMN note TYPE varchar(500)")
        cur.execute(f"UPDATE {right.qualified()} SET note = 'tampered' WHERE id = 123")

    without = _compare(ro_conn, pair)
    as_text = _compare(ro_conn, pair, compare_type_mismatch_as_text=True)

    # Without the option `note` is left out and the changed value **slips through**.
    level4_without = without.level(Level.COLUMN_CHECKSUMS)
    assert "note" not in [d.where for d in level4_without.differences]
    assert any("Skipped" in note for note in level4_without.notes)

    # With the option the difference is found at levels 4 and 5.
    level4 = as_text.level(Level.COLUMN_CHECKSUMS)
    assert "note" in [d.where for d in level4.differences], "the change is to be found"
    assert any("through text" in note for note in level4.notes)
    assert as_text.level(Level.ROW_HASHES).stats["content_differs"] == 1


def test_as_text_keeps_reporting_the_type_difference(ro_conn, rw_conn, pair):
    """The option adds a comparison of values, it **does not mask** the type mismatch."""
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {right.qualified()} ALTER COLUMN note TYPE varchar(500)")

    report = _compare(ro_conn, pair, compare_type_mismatch_as_text=True)

    level3 = report.level(Level.COLUMN_TYPES)
    assert not level3.passed, "level 3 has to keep reporting the type mismatch"
    assert any(d.where == "note" for d in level3.differences)
    assert report.first_failed_level == Level.COLUMN_TYPES


def test_as_text_marks_the_comparison_as_approximate(ro_conn, rw_conn, pair):
    """A textual comparison is approximate and the report has to say so.

    `0.25` and `0.2500` are the same number and different text. Were it to present
    itself as exact, the reader would conclude from a difference that the data
    differs — and would be right only some of the time.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {right.qualified()} ALTER COLUMN quantity TYPE numeric(12,4)")

    report = _compare(ro_conn, pair, compare_type_mismatch_as_text=True)

    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert level4.approximate, "a comparison through text has to be marked approximate"
    assert any("quantity" in note for note in level4.notes)
    assert any(
        "does not by itself prove that the data differs" in note
        for note in report.level(Level.ROW_HASHES).notes
    )


def test_as_text_copes_with_a_numeric_column_too(ro_conn, rw_conn, pair):
    """`sum(length(col))` over `integer` would fail — hence the cast on the column.

    This test guards the generated SQL, not the result: without a cast right on
    the column the query would end in an error and the whole comparison would fail
    with an exception.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {right.qualified()} ALTER COLUMN id TYPE bigint")

    report = _compare(ro_conn, pair, compare_type_mismatch_as_text=True)

    assert report.error is None, f"the comparison failed: {report.error}"
    assert report.level(Level.COLUMN_CHECKSUMS).stats["compared_columns"] > 0


def test_it_finds_a_different_column_type(ro_conn, rw_conn, pair):
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {right.qualified()} ALTER COLUMN price TYPE text")

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.COLUMN_TYPES

    level3 = report.level(Level.COLUMN_TYPES)
    difference = next(d for d in level3.differences if d.where == "price")
    assert "numeric" in difference.left
    assert "text" in difference.right

    # A column with a different type is left out of the checksums, but that has to
    # be in the notes — otherwise it would look as though it had been compared and
    # come out fine.
    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert any("because of a type mismatch" in note for note in level4.notes)


def test_it_finds_a_nullability_change(ro_conn, rw_conn, pair):
    """Nullability is part of the type — on its own it changes neither the data nor
    the counts, so level 3 has to catch it or it slips through."""
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f'ALTER TABLE {right.qualified()} ALTER COLUMN "_name" SET NOT NULL')

    report = _compare(ro_conn, pair)
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.COLUMN_TYPES
    difference = next(d for d in report.level(Level.COLUMN_TYPES).differences if d.where == "_name")
    assert "NULL" in difference.left and "NOT NULL" in difference.right


# --- A missing table --------------------------------------------------------


def test_a_non_existent_table_is_not_a_match(ro_conn, workspace, rw_conn):
    left = _make_table(rw_conn, workspace, "present", rows=5)
    right = Relation(schema=workspace, table="absent")

    report = compare_tables(ro_conn, left, right, CompareOptions())
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.EXISTENCE_AND_ROWS
    # The remaining levels have to be marked skipped, not passed.
    for level in (Level.COLUMN_NAMES, Level.COLUMN_TYPES, Level.COLUMN_CHECKSUMS):
        assert report.level(level).skipped


# --- Approximation must not present itself as exactness ---------------------


def test_a_json_column_is_marked_approximate(ro_conn, rw_conn, workspace):
    """The `json` type (not `jsonb`) remembers the original spelling, so a hash over
    it is not reliable. The report has to say so."""
    left = Relation(schema=workspace, table="json_a")
    right = Relation(schema=workspace, table="json_b")
    with rw_conn.cursor() as cur:
        for relation in (left, right):
            cur.execute(f"CREATE TABLE {relation.qualified()} (id int primary key, data json)")
            cur.execute(f"INSERT INTO {relation.qualified()} VALUES (1, '{{\"a\": 1}}')")

    report = compare_tables(ro_conn, left, right, CompareOptions())
    assert report.verdict == VERDICT_MATCH
    assert report.is_approximate, "a match over `json` must not present itself as exact"

    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert level4.approximate
    assert any("Approximate comparison" in note for note in level4.notes)


def test_a_missing_row_hash_is_reported_and_the_comparison_goes_on(ro_conn, rw_conn, workspace):
    """A missing `row_hash` is a finding in itself — every target table is required
    to have it — but the comparison does not stop over it: the content hash keeps
    working."""
    left = Relation(schema=workspace, table="nohash_a")
    right = Relation(schema=workspace, table="nohash_b")
    with rw_conn.cursor() as cur:
        for relation in (left, right):
            cur.execute(f"CREATE TABLE {relation.qualified()} (id int primary key, v text)")
            cur.execute(f"INSERT INTO {relation.qualified()} VALUES (1, 'a'), (2, 'b')")
    with rw_conn.cursor() as cur:
        cur.execute(f"UPDATE {right.qualified()} SET v = 'other' WHERE id = 2")

    report = compare_tables(ro_conn, left, right, CompareOptions())
    level5 = report.level(Level.ROW_HASHES)
    assert not level5.passed
    assert any("a finding in its own right" in note for note in level5.notes)
    # The stored hash could not be compared, the content one could — and it found
    # the difference exactly.
    assert level5.stats["hash_differs"] == 0
    assert level5.stats["content_differs"] == 1
    assert not level5.approximate, "text compares exactly, no reason to report approximation"


# --- Safety -----------------------------------------------------------------


def test_the_comparing_connection_cannot_write(ro_conn, workspace):
    """Read-only is not a declaration but a setting of the database.

    The golden test may only read from a production database.
    """
    import psycopg2

    with pytest.raises(psycopg2.Error), ro_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{workspace}".attempt (id int)')


# --- Partial results after a level-2 failure ---------------------------------


def test_a_failed_level_2_still_yields_partial_results(ro_conn, rw_conn, pair):
    """When the column names diverge, levels 4 and 5 must still run.

    The module docstring of ``compare.py`` promises exactly this: over a batch of
    250 tables there is a world of difference between "one column name differs,
    the data matches" and "a column name differs, beyond that we do not know".
    The audit found that a mutant which skips levels 4–5 after a level-2 failure
    survived the suite — every existing test either had matching names or only
    looked at the failing level. This one pins the promise down: the intersection
    of columns carries identical data, so levels 4 and 5 have to *run* and *pass*.
    """
    left, right = pair
    with rw_conn.cursor() as cur:
        cur.execute(f'ALTER TABLE {right.qualified()} RENAME COLUMN "_type" TO "type"')

    report = _compare(ro_conn, pair)
    assert report.error is None, report.error
    assert report.verdict == VERDICT_DIFF
    assert report.first_failed_level == Level.COLUMN_NAMES

    # Level 4 ran over the intersection: 13 columns minus the renamed pair
    # (present on one side each) minus the volatile `_timestamp`.
    level4 = report.level(Level.COLUMN_CHECKSUMS)
    assert not level4.skipped, "level 4 must not be skipped after a level-2 failure"
    assert level4.passed, "the shared columns hold identical data — level 4 has to say so"
    assert level4.stats["compared_columns"] == 11

    # Level 5 paired the rows through the key and found the data identical.
    level5 = report.level(Level.ROW_HASHES)
    assert not level5.skipped, "level 5 must not be skipped after a level-2 failure"
    assert level5.passed
    assert level5.stats["only_in_left"] == 0
    assert level5.stats["only_in_right"] == 0
    assert level5.stats["content_differs"] == 0
