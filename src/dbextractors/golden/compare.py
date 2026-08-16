"""The five levels of comparing two target tables.

The order of the levels is the order **in which things break**. If the table does
not exist there is no point comparing columns; if the column names disagree, the
checksums over them say nothing useful. The report therefore always states the
level at which it broke first — that is information one can act on.

The levels do **not stop at the first mismatch**, though. When the column names
diverge, levels 4 and 5 still run over the intersection and are marked partial.
Migration happens in groups of up to 250 tables, and there is a difference
between knowing "one column name differs, the data matches" and "a column name
differs, beyond that we do not know".

Nothing here writes anything. The session runs with
``default_transaction_read_only = on``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Collection, List, Optional

from dbextractors.golden import introspect, sqlgen
from dbextractors.golden import progress as progress_mod
from dbextractors.golden.model import (
    Difference,
    Level,
    LevelResult,
    Relation,
    TableReport,
)
from dbextractors.golden.sqlgen import quote_ident

if TYPE_CHECKING:  # pragma: no cover
    import psycopg2.extensions

#: Columns produced by the extractor itself, which are left out of the data
#: comparison. `_timestamp` is the run time — two runs always differ in it and
#: that means nothing. Its *existence* is still checked at levels 2 and 3,
#: because the dbt layer depends on it.
VOLATILE_COLUMNS: frozenset[str] = frozenset({"_timestamp"})

DEFAULT_HASH_COLUMN = "row_hash"


@dataclass
class CompareOptions:
    """Settings of a single comparison."""

    #: Key used for the row-by-row comparison. When ``None``, the primary key or
    #: the narrowest unique index is looked up.
    key_columns: Optional[list[str]] = None
    #: Column holding the row hash. When the tables do not have it, a substitute
    #: is computed and level 5 is marked approximate.
    hash_column: str = DEFAULT_HASH_COLUMN
    #: How many differing keys to print into the report.
    sample_limit: int = 10
    #: Columns left out of the data comparison (not out of the name and type
    #: comparison).
    ignore_columns: frozenset[str] = VOLATILE_COLUMNS
    #: Level 5 does a FULL OUTER JOIN of both tables. On genuinely large tables
    #: it can be turned off, but the report then says explicitly that it did not
    #: run.
    skip_row_level: bool = False
    #: How many differing columns to list before the output is truncated.
    max_reported_columns: int = 25
    #: Also compare columns whose type differs between the two sides, through
    #: their textual form.
    #:
    #: By default such a column is left out of levels 4 and 5, because `0.25` vs
    #: `0.2500` is different text in every single row and a genuine finding would
    #: drown in it. For Firebird sources that is one or two columns and does no
    #: harm; for **PostgreSQL sources**, however, the old side misses the type map
    #: and falls back to `TEXT` (74 out of 92 columns on one test table), so
    #: almost everything would be skipped and a "MATCH" would mean nothing.
    #:
    #: The option does **not hide** the type difference — level 3 still reports
    #: it. It only makes the values of those columns take part in the comparison,
    #: and the report says the comparison is approximate for them.
    compare_type_mismatch_as_text: bool = False


@dataclass
class _Context:
    conn: psycopg2.extensions.connection
    left: Relation
    right: Relation
    options: CompareOptions
    left_columns: list[introspect.ColumnInfo] = field(default_factory=list)
    right_columns: list[introspect.ColumnInfo] = field(default_factory=list)
    #: Columns present on both sides.
    common: list[str] = field(default_factory=list)
    #: Columns present on both sides, with the same type, and not ignored. Levels
    #: 4 and 5 must work over the **same** list — see how this is filled in
    #: `compare_tables`.
    comparable: list[str] = field(default_factory=list)
    #: Common columns whose types disagree (reported by level 3). They only enter
    #: `comparable` when `compare_type_mismatch_as_text` is on; otherwise they are
    #: left out of levels 4 and 5.
    type_mismatch: list[str] = field(default_factory=list)
    #: The subset of `type_mismatch` that is compared anyway — through text.
    text_coerced: list[str] = field(default_factory=list)


# --- Level 1 ---------------------------------------------------------------


def _level_existence_and_rows(ctx: _Context) -> LevelResult:
    result = LevelResult(level=Level.EXISTENCE_AND_ROWS, passed=True)

    left_exists = introspect.table_exists(ctx.conn, ctx.left)
    right_exists = introspect.table_exists(ctx.conn, ctx.right)
    result.stats["exists"] = {"left": left_exists, "right": right_exists}

    if not left_exists or not right_exists:
        result.passed = False
        result.differences.append(
            Difference(
                what="table does not exist",
                left="present" if left_exists else "MISSING",
                right="present" if right_exists else "MISSING",
            )
        )
        return result

    left_rows = introspect.row_count(ctx.conn, ctx.left)
    right_rows = introspect.row_count(ctx.conn, ctx.right)
    result.stats["rows"] = {"left": left_rows, "right": right_rows, "delta": right_rows - left_rows}

    if left_rows != right_rows:
        result.passed = False
        result.differences.append(Difference(what="row count", left=left_rows, right=right_rows))
    return result


# --- Level 2 ---------------------------------------------------------------


def _level_column_names(ctx: _Context) -> LevelResult:
    """The most important level of the whole test.

    The legacy write path through ``mage_ai.io.postgres`` prefixes with an
    underscore every name whose upper-case form is among 825 reserved words.
    That is where ``_type`` in 113 tables, ``_name`` in 107 and ``_date`` in 92
    come from. The dbt layer rests on those names. See the column naming section
    of ``docs/legacy-compat.md``.

    What is compared is the **sequence**, not the set — the order is part of the
    contract.
    """
    result = LevelResult(level=Level.COLUMN_NAMES, passed=True)

    left_names = [c.name for c in ctx.left_columns]
    right_names = [c.name for c in ctx.right_columns]
    result.stats["counts"] = {"left": len(left_names), "right": len(right_names)}

    if left_names == right_names:
        return result

    result.passed = False
    left_set, right_set = set(left_names), set(right_names)

    for name in left_names:
        if name not in right_set:
            result.differences.append(
                Difference(what="column missing on the right", left=name, right=None, where=name)
            )
    for name in right_names:
        if name not in left_set:
            result.differences.append(
                Difference(what="column only on the right", left=None, right=name, where=name)
            )

    # A renamed column shows up as a missing/surplus pair. When both sets are
    # equal and only the order differs, that is a different fault with a
    # different fix — hence the distinction.
    if left_set == right_set:
        pairs = zip(left_names, right_names, strict=False)
        for position, (left_name, right_name) in enumerate(pairs, start=1):
            if left_name != right_name:
                result.differences.append(
                    Difference(
                        what="different column order",
                        left=left_name,
                        right=right_name,
                        where=f"position {position}",
                    )
                )
        result.notes.append(
            "The set of names matches, only the order differs. The data may well be "
            "fine, but column order is part of the contract."
        )

    # Only here, not earlier: neither a missing nor a surplus column falls under
    # any rule, so `_accept_deviations` is called on the complete list.
    _accept_deviations(result)
    return result


# --- Level 3 ---------------------------------------------------------------


def _accept_deviations(result: LevelResult) -> None:
    """Moves agreed deviations aside so that the level does not fail on them.

    It does not suppress them — they move into `LevelResult.deviations` and the
    report prints them. See `dbextractors.golden.deviations`.
    """
    from dbextractors.golden import deviations as rules

    real, accepted = rules.classify(result.level, result.differences)
    result.differences = real
    result.deviations = accepted
    if accepted and not real:
        result.passed = True
        for diff in accepted:
            rule = rules.match(result.level, diff)
            if rule is not None:
                result.notes.append(f"Deviation {rule.name!r}: {rule.reason}")


def _level_column_types(ctx: _Context) -> LevelResult:
    result = LevelResult(level=Level.COLUMN_TYPES, passed=True)

    right_by_name = {c.name: c for c in ctx.right_columns}
    compared = 0

    for left_col in ctx.left_columns:
        right_col = right_by_name.get(left_col.name)
        if right_col is None:
            continue  # A missing column was already reported by level 2.
        compared += 1
        if left_col.type_signature() != right_col.type_signature():
            result.passed = False
            result.differences.append(
                Difference(
                    what="different data type",
                    left=left_col.describe_type(),
                    right=right_col.describe_type(),
                    where=left_col.name,
                )
            )

    _accept_deviations(result)

    result.stats["compared_columns"] = compared
    if compared < len(ctx.left_columns):
        result.notes.append(
            f"Compared {compared} of {len(ctx.left_columns)} columns — "
            "the rest are missing on one side (see level 2)."
        )
    return result


# --- Level 4 ---------------------------------------------------------------


def _level_checksums(ctx: _Context) -> LevelResult:
    result = LevelResult(level=Level.COLUMN_CHECKSUMS, passed=True)

    left_by_name = {c.name: c for c in ctx.left_columns}
    coerced = set(ctx.text_coerced)
    skipped_type_mismatch = [c for c in ctx.type_mismatch if c not in coerced]
    comparable: list[tuple[str, sqlgen.Family]] = [
        (name, sqlgen.classify(left_by_name[name].data_type, left_by_name[name].udt_name))
        for name in ctx.comparable
    ]

    if not comparable:
        result.skipped = True
        result.notes.append("Nothing to compare — no common column with a matching type.")
        return result

    left_values = _run_checksums(ctx.conn, ctx.left, comparable, coerced)
    right_values = _run_checksums(ctx.conn, ctx.right, comparable, coerced)

    approximate_columns = []
    for name, family in comparable:
        # A cast column is always approximate: what gets compared is the
        # formatting, not the value. `0.25` and `0.2500` are the same number and
        # different text.
        if family in sqlgen.APPROXIMATE_FAMILIES or name in coerced:
            approximate_columns.append(name)
        left_metrics = left_values[name]
        right_metrics = right_values[name]
        for metric, left_value in left_metrics.items():
            right_value = right_metrics.get(metric)
            if left_value != right_value:
                result.passed = False
                result.differences.append(
                    Difference(
                        what=f"different checksum ({metric})",
                        left=left_value,
                        right=right_value,
                        where=name,
                    )
                )

    result.stats["compared_columns"] = len(comparable)
    result.stats["row_count"] = {
        "left": left_values["row_count"],
        "right": right_values["row_count"],
    }

    if approximate_columns:
        result.approximate = True
        listed = ", ".join(sorted(approximate_columns)[:10])
        result.notes.append(
            f"Approximate comparison on {len(approximate_columns)} columns ({listed}"
            f"{', …' if len(approximate_columns) > 10 else ''}): "
            "they are compared through a textual representation that need not be canonical."
        )
    if skipped_type_mismatch:
        result.notes.append(
            f"Skipped {len(skipped_type_mismatch)} columns because of a type mismatch "
            f"({', '.join(sorted(skipped_type_mismatch)[:10])}) — see level 3."
        )
    # An empty hash on the old side is not a difference in the data but a gap in
    # the legacy extractor — better to say so outright than to let it look like a
    # regression. It is deliberately **not** forgiven as a deviation: once
    # dbextractors replaces the old path, those empty `row_hash` values will be
    # filled in, and that is a genuine change to production data which ought to
    # be visible.
    hash_col = ctx.options.hash_column
    if hash_col in ctx.comparable:
        row_count = left_values.get("row_count")
        left_nulls = left_values.get(hash_col, {}).get("nulls")
        right_nulls = right_values.get(hash_col, {}).get("nulls")
        if row_count and left_nulls == row_count and right_nulls == 0:
            result.notes.append(
                f"Column `{hash_col}` is empty on the left side in all {row_count} rows "
                "and filled in on the right. This is not a difference in the data: "
                "the legacy extractor does not compute it on this path (five of the "
                "predecessor files merely declare the required columns). Expect those "
                "values to be filled in once the switch happens."
            )
    if coerced:
        result.notes.append(
            f"{len(coerced)} columns with a type mismatch were compared through text "
            f"({', '.join(sorted(coerced)[:10])}"
            f"{', …' if len(coerced) > 10 else ''}) — "
            "`compare_type_mismatch_as_text` is on. A difference in how the same "
            "value is written (`0.25` vs `0.2500`) shows up here as a difference in "
            "the data; the type mismatch itself is reported by level 3."
        )
    if ctx.options.ignore_columns:
        result.notes.append(
            "Left out of the data comparison: "
            f"{', '.join(sorted(ctx.options.ignore_columns))} "
            "(they change with every run; their existence is checked at levels 2 and 3)."
        )
    _truncate(result, ctx.options.max_reported_columns)
    return result


def _run_checksums(
    conn: psycopg2.extensions.connection,
    relation: Relation,
    columns: list[tuple[str, sqlgen.Family]],
    text_coerced: Collection[str] = (),
) -> dict:
    sql, aggregates = sqlgen.build_checksum_query(
        relation.qualified(), columns, text_coerced=text_coerced
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    return sqlgen.parse_checksum_row(row, aggregates)


# --- Level 5 ---------------------------------------------------------------


def _level_row_hashes(ctx: _Context) -> LevelResult:
    result = LevelResult(level=Level.ROW_HASHES, passed=True)
    options = ctx.options

    if options.skip_row_level:
        result.skipped = True
        result.notes.append("Turned off by --skip-row-level. Rows were not compared.")
        return result

    key = options.key_columns or introspect.unique_key(ctx.conn, ctx.left)
    if not key:
        result.skipped = True
        result.notes.append(
            "The table has neither a primary key nor a complete unique index, so rows "
            "cannot be paired up. Supply a key with --key."
        )
        return result

    missing_key = [c for c in key if c not in ctx.common]
    if missing_key:
        result.skipped = True
        result.notes.append(
            f"The key {', '.join(key)} is not present on both sides "
            f"(missing {', '.join(missing_key)})."
        )
        return result

    stored_expr, content_expr, notes, approximate = _hash_expressions(ctx, key)
    result.notes.extend(notes)
    result.approximate = approximate

    key_expr = _key_expression(key)
    counts_sql = f"""
        WITH l AS (SELECT {key_expr} AS k, {stored_expr} AS hs, {content_expr} AS hc
                   FROM {ctx.left.qualified()}),
             r AS (SELECT {key_expr} AS k, {stored_expr} AS hs, {content_expr} AS hc
                   FROM {ctx.right.qualified()})
        SELECT
          count(*) FILTER (WHERE r.k IS NULL) AS only_left,
          count(*) FILTER (WHERE l.k IS NULL) AS only_right,
          count(*) FILTER (WHERE l.k IS NOT NULL AND r.k IS NOT NULL
                             AND l.hs IS DISTINCT FROM r.hs) AS stored_differs,
          count(*) FILTER (WHERE l.k IS NOT NULL AND r.k IS NOT NULL
                             AND l.hc IS DISTINCT FROM r.hc) AS content_differs
        FROM l FULL OUTER JOIN r ON l.k = r.k
    """
    with ctx.conn.cursor() as cur:
        cur.execute(counts_sql)
        only_left, only_right, stored_differs, content_differs = cur.fetchone()

    result.stats.update(
        {
            "key": key,
            "only_in_left": only_left,
            "only_in_right": only_right,
            "hash_differs": stored_differs,
            "content_differs": content_differs,
        }
    )

    if only_left or only_right or stored_differs or content_differs:
        result.passed = False
        for what, count in (
            ("row only on the left", only_left),
            ("row only on the right", only_right),
            ("same key, different row_hash", stored_differs),
            ("same key and row_hash, different data", content_differs - stored_differs),
        ):
            if count and count > 0:
                result.differences.append(Difference(what=what, left=count, right=None))
        result.stats["samples"] = _sample_differences(
            ctx, key_expr, stored_expr, content_expr, options.sample_limit
        )
    return result


def _key_expression(key: list[str]) -> str:
    """The key as a single text expression.

    The obvious thing would be to join the tables through
    ``l.id IS NOT DISTINCT FROM r.id``, to handle ``NULL`` in the key. PostgreSQL
    refuses that in a ``FULL OUTER JOIN`` condition: *"FULL JOIN is only supported
    with merge-joinable or hash-joinable join conditions."*

    Serialising the key into text sidesteps that problem and handles ``NULL`` as
    well: for ``NULL``, ``quote_nullable`` returns the string ``NULL`` without
    quotes, whereas for the string ``'NULL'`` it returns ``'NULL'`` **with**
    quotes. The two cases therefore do not merge. The result is also never
    ``NULL``, so ``l.k IS NULL`` can safely be read as "nothing on the right
    matches".
    """
    return _concat_ws([f"quote_nullable({quote_ident(c)}::text)" for c in key])


#: PostgreSQL does not allow more than 100 arguments to a function. The margin is
#: deliberately generous — `concat_ws` takes the separator as its first argument
#: and the limit must not be approached closely.
_MAX_ARGS = 90


def _concat_ws(parts: List[str]) -> str:
    """``concat_ws`` over an arbitrarily wide table.

    Above 100 arguments PostgreSQL rejects the function call (*"cannot pass more
    than 100 arguments to a function"*) and level 5 ends in an error — hit in
    practice on a table with 156 columns. The expression is therefore built in
    groups and joined into a tree.

    **The result is bit for bit the same as the flat version.** ``concat_ws``
    only skips ``NULL`` arguments and none arrive here — every one of them went
    through ``quote_nullable``, which returns the string ``NULL`` for ``NULL``.
    Groups therefore never vanish and the separators line up.
    """
    if not parts:
        raise ValueError("_concat_ws was given an empty list.")
    while len(parts) > _MAX_ARGS:
        parts = [
            f"concat_ws(E'\\x1f', {', '.join(parts[i : i + _MAX_ARGS])})"
            for i in range(0, len(parts), _MAX_ARGS)
        ]
    if len(parts) == 1:
        return parts[0]
    return f"concat_ws(E'\\x1f', {', '.join(parts)})"


def _hash_expressions(ctx: _Context, key: list[str]) -> tuple[str, str, list[str], bool]:
    """Two row hashes, not one. Returns (stored, content, notes, approximate).

    The **stored** one is the value of the ``row_hash`` column produced by the
    extractor. It is part of the contract — 5 dbt models depend on it — so it has
    to be compared exactly as it is.

    The **content** one is computed from every compared column. It is here because
    ``row_hash`` alone is not enough: the configuration contract has both
    ``hash_include_columns`` and ``hash_exclude_columns``, so it is routinely
    computed from a subset of the columns. A change in an excluded column would
    then leave ``row_hash`` unchanged and level 5 would miss it.

    Level 4 does not plug that gap — swap the values between two rows and the
    per-column checksums stay identical. The content hash is exactly what catches
    that.
    """
    notes: list[str] = []
    approximate = False

    wanted = ctx.options.hash_column
    if wanted in ctx.common:
        stored = quote_ident(wanted)
    else:
        stored = "NULL::text"
        notes.append(
            f"Column `{wanted}` is not present in the tables, so it was not compared. "
            "Every target table is required to have it — this is a finding in its "
            "own right."
        )

    # The same list level 4 uses. A column with a type mismatch must not get in
    # here: its textual form differs in **every** row (`0.25` vs `0.2500`), so the
    # content hash would mark every row as different and drown the real finding.
    # The type mismatch itself is reported by level 3.
    # The hash column does **not** belong in the content hash. It is compared
    # separately as `stored` and it is metadata, not data. When the two sides
    # differ in `row_hash` — say because the old side never filled it in on a
    # full load (in one predecessor's PostgreSQL extractor the full-load path is
    # the only one that does not call the hash-and-timestamp step) — the
    # difference would show up on **every** row and the "different data" column
    # would report the total row count. That is exactly what happened on one
    # 16-row table: "different row_hash 16, different data 16", while the data
    # matched. The difference would be counted twice and the second number would
    # lose its meaning.
    payload = [c for c in ctx.comparable if c not in key and c != wanted]
    if not payload:
        notes.append("Nothing to compare besides the key — the content hash was not computed.")
        return stored, "NULL::text", notes, True

    concat = _concat_ws([f"quote_nullable({quote_ident(c)}::text)" for c in payload])
    content = f"md5({concat})"

    approximate_columns = _approximate_columns(ctx, payload)
    if approximate_columns:
        approximate = True
        notes.append(
            f"The content hash covers {len(approximate_columns)} columns whose textual "
            "representation need not be canonical "
            f"({', '.join(sorted(approximate_columns)[:5])}) — "
            "the comparison is approximate for them."
        )
    notes.append(
        f"Besides `{wanted}`, a content hash over {len(payload)} columns is compared "
        f"as well (without the key and without `{wanted}` itself — that one is "
        "compared separately, otherwise its difference would be counted twice). "
        "`row_hash` is routinely computed from a subset of the columns "
        "(hash_include_columns / hash_exclude_columns), so on its own it would not "
        "catch a change in an excluded column."
    )
    vynechane = [c for c in ctx.type_mismatch if c not in set(ctx.text_coerced)]
    if vynechane:
        notes.append(
            f"{len(vynechane)} columns with a type mismatch are kept out of the content "
            f"hash ({', '.join(sorted(vynechane)[:5])}) — otherwise every row would "
            "differ and the real finding would drown in it. The type mismatch itself "
            "is reported by level 3."
        )
    if ctx.text_coerced:
        approximate = True
        notes.append(
            f"{len(ctx.text_coerced)} columns with a type mismatch **do enter** the "
            f"content hash ({', '.join(sorted(ctx.text_coerced)[:5])}) — "
            "`compare_type_mismatch_as_text` is on. What gets compared is the textual "
            "form, so a different spelling of the same value comes out as a differing "
            "row. A difference at level 5 therefore does not by itself prove that the "
            "data differs."
        )
    return stored, content, notes, approximate


def _approximate_columns(ctx: _Context, names: list[str]) -> list[str]:
    by_name = {c.name: c for c in ctx.left_columns}
    return [
        name
        for name in names
        if name in by_name
        and sqlgen.classify(by_name[name].data_type, by_name[name].udt_name)
        in sqlgen.APPROXIMATE_FAMILIES
    ]


def _sample_differences(
    ctx: _Context, key_expr: str, stored_expr: str, content_expr: str, limit: int
) -> list[dict]:
    """The first N differing keys. Sorted, so that runs print the same list."""
    sql = f"""
        WITH l AS (SELECT {key_expr} AS k, {stored_expr} AS hs, {content_expr} AS hc
                   FROM {ctx.left.qualified()}),
             r AS (SELECT {key_expr} AS k, {stored_expr} AS hs, {content_expr} AS hc
                   FROM {ctx.right.qualified()})
        SELECT
          CASE
            WHEN r.k IS NULL THEN 'left only'
            WHEN l.k IS NULL THEN 'right only'
            WHEN l.hs IS DISTINCT FROM r.hs THEN 'row_hash differs'
            ELSE 'data differs'
          END AS kind,
          coalesce(l.k, r.k) AS key_value,
          l.hs::text, r.hs::text
        FROM l FULL OUTER JOIN r ON l.k = r.k
        WHERE l.k IS NULL OR r.k IS NULL
           OR l.hs IS DISTINCT FROM r.hs
           OR l.hc IS DISTINCT FROM r.hc
        ORDER BY 1, 2
        LIMIT %s
    """
    with ctx.conn.cursor() as cur:
        cur.execute(sql, (limit,))
        return [
            {"kind": kind, "key": key_value, "left_hash": left_hash, "right_hash": right_hash}
            for kind, key_value, left_hash, right_hash in cur.fetchall()
        ]


# --- Putting it together ---------------------------------------------------


def _truncate(result: LevelResult, limit: int) -> None:
    if len(result.differences) > limit:
        hidden = len(result.differences) - limit
        result.stats["differences_total"] = len(result.differences)
        del result.differences[limit:]
        result.notes.append(f"Printed the first {limit} differences, {hidden} more hidden.")


def _with_progress(progress, label: str, func, ctx: _Context) -> LevelResult:
    """Runs one level and reports it.

    The ``finally`` matters: if the level raised, without it the thread rewriting
    the line would stay alive and the error message would get mixed into it.
    """
    progress.start(label)
    started = time.monotonic()
    try:
        return func(ctx)
    finally:
        progress.finish(time.monotonic() - started)


def compare_tables(
    conn: psycopg2.extensions.connection,
    left: Relation,
    right: Relation,
    options: Optional[CompareOptions] = None,
    label: Optional[str] = None,
    progress: Optional[progress_mod.Progress] = None,
) -> TableReport:
    """Compares two tables across all five levels.

    ``left`` is the reference side (the old component), ``right`` the one under
    test (the new one). The order matters only for how differences are worded.

    ``progress`` receives messages about which level is currently running.
    Without it the tool stays silent — but comparing a large table takes minutes
    (247 s for one of them) and without feedback nobody can tell that apart from
    a hung process.
    """
    options = options or CompareOptions()
    reporter = progress or progress_mod.Progress()
    ctx = _Context(conn=conn, left=left, right=right, options=options)
    report = TableReport(left=left, right=right, label=label)
    started = time.monotonic()

    try:
        level1 = _with_progress(reporter, "level 1 · row count", _level_existence_and_rows, ctx)
        report.levels.append(level1)

        # Row counts say roughly how long this is going to take — and they are
        # known right after the first level, that is, before the expensive part
        # starts.
        row_counts = level1.stats.get("rows")
        if row_counts:
            reporter.message(
                f"{left.qualified()} {row_counts['left']:,} rows · "
                f"{right.qualified()} {row_counts['right']:,} rows".replace(",", " ")
            )

        if not level1.stats.get("exists", {}).get("left") or not level1.stats.get("exists", {}).get(
            "right"
        ):
            for level in (
                Level.COLUMN_NAMES,
                Level.COLUMN_TYPES,
                Level.COLUMN_CHECKSUMS,
                Level.ROW_HASHES,
            ):
                report.levels.append(
                    LevelResult(
                        level=level,
                        passed=False,
                        skipped=True,
                        notes=["Skipped — one of the tables does not exist."],
                    )
                )
            return report

        ctx.left_columns = introspect.columns(conn, left)
        ctx.right_columns = introspect.columns(conn, right)
        right_by_name = {c.name: c for c in ctx.right_columns}
        ctx.common = [c.name for c in ctx.left_columns if c.name in right_by_name]
        # Comparable = present on both sides, same type, not ignored. Levels 4
        # and 5 have to work over the same list, otherwise they contradict each
        # other.
        for column in ctx.left_columns:
            if column.name not in right_by_name or column.name in options.ignore_columns:
                continue
            if column.type_signature() == right_by_name[column.name].type_signature():
                ctx.comparable.append(column.name)
            else:
                ctx.type_mismatch.append(column.name)
                # Level 3 reports the type difference either way — this only
                # decides whether the column's **values** are compared too.
                if options.compare_type_mismatch_as_text:
                    ctx.comparable.append(column.name)
                    ctx.text_coerced.append(column.name)

        for label, func in (
            ("level 2 · column names", _level_column_names),
            ("level 3 · types", _level_column_types),
            ("level 4 · checksums", _level_checksums),
            ("level 5 · row comparison", _level_row_hashes),
        ):
            report.levels.append(_with_progress(reporter, label, func, ctx))
    except Exception as exc:
        # `report.error` is printed by `report.render_table` and written verbatim
        # into the `--json` file, so it outlives the run. The connection is already
        # open by the time anything here can fail, which is why nothing observed so
        # far carries a DSN — but a report on disk is the wrong place to find out
        # that something does.
        from dbextractors.core import secrets

        report.error = f"{type(exc).__name__}: {secrets.redact(exc)}"
    finally:
        reporter.close()
        report.duration_s = time.monotonic() - started

    return report
