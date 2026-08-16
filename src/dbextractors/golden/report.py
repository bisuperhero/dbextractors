"""Rendering the report — machine readable (JSON) and human readable.

Both come out of the same dataclasses in ``model.py``, not out of two
independent code paths. If the JSON and the text were computed separately, sooner
or later they would say different things — and an oracle that contradicts itself
cannot be trusted.

A requirement the shape of the text output comes from: the report has to be
readable by a person, not just by a machine. Someone approving a migration of 254
tables needs to be able to skim it. The text output is therefore built for
scanning top to bottom: the verdict first, the level it broke at right behind it,
the details below that.
"""

from __future__ import annotations

import json
from typing import Optional

from dbextractors.golden.model import (
    VERDICT_DIFF,
    VERDICT_ERROR,
    VERDICT_MATCH,
    BatchReport,
    Level,
    LevelResult,
    TableReport,
)

MARK_OK = "✓"
MARK_FAIL = "✗"
MARK_SKIP = "–"
MARK_ERROR = "!"
MARK_APPROX = "~"

_WIDTH = 78


#: Thousands separator. Deliberately an **ordinary** space (U+0020) rather than a
#: non-breaking one (U+00A0), even though the latter would be typographically
#: nicer: the report is read in a terminal, where it gets grepped and copied, and
#: a non-breaking space silently breaks a search for a number that is plainly
#: visible on screen. Written as an escape so that it is not an invisible
#: character in the source.
THOUSANDS_SEPARATOR = "\u0020"


def _num(value: object) -> str:
    """A number grouped by thousands. 1234567 -> ``1 234 567``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return str(value)
    return f"{value:,}".replace(",", THOUSANDS_SEPARATOR)


def _short(value: object, limit: int = 60) -> str:
    if value is None:
        return "—"
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _identify(report: TableReport) -> str:
    """How a row of the batch summary is named.

    The left-hand table alone is not enough: the same source table routinely
    appears in a manifest several times (against different candidates), and two
    rows with the same name cannot be told apart in a summary of 250 tables. The
    `label` from the manifest therefore wins, and without one both sides are
    printed.
    """
    if report.label:
        return _short(report.label, 44)
    return _short(f"{report.left} → {report.right}", 44)


def _level_mark(result: LevelResult) -> str:
    if result.skipped:
        return MARK_SKIP
    if not result.passed:
        return MARK_FAIL
    return MARK_APPROX if result.approximate else MARK_OK


def _level_summary(result: LevelResult) -> str:
    """One-line summary of a level — what gets read while skimming."""
    stats = result.stats

    if result.level is Level.EXISTENCE_AND_ROWS:
        rows = stats.get("rows")
        if rows:
            if rows["left"] == rows["right"]:
                return f"{_num(rows['left'])} rows"
            return (
                f"{_num(rows['left'])} vs {_num(rows['right'])} rows "
                f"({rows['delta']:+,})".replace(",", THOUSANDS_SEPARATOR)
            )
        exists = stats.get("exists", {})
        left = "yes" if exists.get("left") else "NO"
        right = "yes" if exists.get("right") else "NO"
        return f"left {left}, right {right}"

    if result.level is Level.COLUMN_NAMES:
        counts = stats.get("counts", {})
        if result.passed:
            return f"{counts.get('left', '?')} columns, same order"
        return f"{counts.get('left', '?')} vs {counts.get('right', '?')} columns"

    if result.level is Level.COLUMN_TYPES:
        return f"{stats.get('compared_columns', 0)} columns compared"

    if result.level is Level.COLUMN_CHECKSUMS:
        if result.skipped:
            return "not run"
        return f"{stats.get('compared_columns', 0)} columns compared"

    if result.level is Level.ROW_HASHES:
        if result.skipped:
            return "not run"
        key = ", ".join(stats.get("key", []))
        if result.passed:
            return f"key [{key}], every row matches"
        return (
            f"key [{key}]: left only {_num(stats.get('only_in_left', 0))}, "
            f"right only {_num(stats.get('only_in_right', 0))}, "
            f"row_hash differs {_num(stats.get('hash_differs', 0))}, "
            f"data differs {_num(stats.get('content_differs', 0))}"
        )

    return ""


def render_table(report: TableReport, *, verbose: bool = True) -> str:
    """Readable report for one pair of tables."""
    lines: list[str] = []
    verdict = report.verdict

    lines.append("═" * _WIDTH)
    head = f"  {verdict}   {report.left}  →  {report.right}"
    lines.append(head)

    if report.error:
        lines.append(f"         {report.error}")
    else:
        failed = report.first_failed_level
        if failed is None:
            detail = "all five levels match"
            if report.is_approximate:
                detail += " (part of the comparison is approximate, see below)"
        else:
            level_result = report.level(failed)
            title = level_result.title if level_result else ""
            detail = f"broke at level {int(failed)} — {title}"
        lines.append(f"         {detail} · {report.duration_s:.2f} s")
    lines.append("═" * _WIDTH)

    for result in sorted(report.levels, key=lambda r: r.level):
        lines.append(
            f"  {_level_mark(result)} {int(result.level)}  "
            f"{result.title:<34} {_level_summary(result)}"
        )

        if verbose or not result.passed:
            for diff in result.differences:
                where = f"[{diff.where}] " if diff.where else ""
                lines.append(f"        {where}{diff.what}")
                if diff.left is not None or diff.right is not None:
                    lines.append(
                        f"            left:  {_short(diff.left)}\n"
                        f"            right: {_short(diff.right)}"
                    )
            total = result.stats.get("differences_total")
            if total:
                lines.append(f"        ({_num(total)} differences in total)")

        # Deviations are printed **always**, even when the level passed.
        # Suppressing them would amount to claiming that the two sides do not
        # differ — and they do, we just know about it.
        for diff in result.deviations:
            where = f"[{diff.where}] " if diff.where else ""
            lines.append(f"        deviation: {where}{_short(diff.left)} -> {_short(diff.right)}")

        for note in result.notes:
            lines.append(f"        note: {note}")

        samples = result.stats.get("samples")
        if samples:
            lines.append("        sample of differing keys:")
            for sample in samples:
                lines.append(f"            {sample['kind']:<16} {_short(sample['key'], 40)}")

    if report.is_approximate:
        lines.append("")
        lines.append(
            f"  {MARK_APPROX} Part of the comparison is approximate. A match in those "
            "places is not proof that the data matches."
        )

    return "\n".join(lines)


def render_batch(batch: BatchReport, *, verbose: bool = False) -> str:
    """Summary of a batch run. Migration happens in groups of 20–250 tables."""
    lines: list[str] = []
    counts = (
        f"{len(batch.reports)} tables: "
        f"{len(batch.matched)} match, "
        f"{len(batch.differing)} diff, "
        f"{len(batch.errored)} error"
    )
    lines.append("═" * _WIDTH)
    lines.append(f"  SUMMARY   {batch.verdict}")
    lines.append(f"            {counts}")
    lines.append("═" * _WIDTH)

    problems = batch.differing + batch.errored
    if not problems:
        lines.append(f"  {MARK_OK} Every table matches.")
    else:
        for report in problems:
            if report.verdict == VERDICT_ERROR:
                lines.append(
                    f"  {MARK_ERROR} {VERDICT_ERROR:<7} {_identify(report):<44} "
                    f"{_short(report.error, 26)}"
                )
            else:
                failed = report.first_failed_level
                level_result = report.level(failed) if failed else None
                title = level_result.title if level_result else ""
                lines.append(
                    f"  {MARK_FAIL} {VERDICT_DIFF:<7} {_identify(report):<44} "
                    f"level {int(failed) if failed else '?'} — {title}"
                )

    approximate = [r for r in batch.reports if r.is_approximate]
    if approximate:
        lines.append("")
        lines.append(
            f"  {MARK_APPROX} In {len(approximate)} tables part of the comparison "
            "is approximate."
        )

    matched_approx = [r for r in batch.matched if r.is_approximate]
    if matched_approx:
        lines.append(
            f"      Of those, {len(matched_approx)} ended with the verdict {VERDICT_MATCH} — "
            "for them a match is not proof that the data matches."
        )

    if verbose:
        lines.append("")
        for report in batch.reports:
            lines.append(render_table(report, verbose=False))
            lines.append("")

    return "\n".join(lines)


def _fallback(value: object) -> str:
    """Last resort for types ``json`` cannot handle.

    The aggregates are converted to text back in SQL, but PostgreSQL's type
    palette is wide and a new version may return something unexpected. Without
    this, that would only surface when the report is written — that is, at the end
    of a batch of over 250 tables, after half an hour of computation. Losing the
    whole run over it would be a shame.
    """
    return str(value)


def to_json(payload: TableReport | BatchReport, path: Optional[str] = None) -> str:
    text = json.dumps(payload.to_dict(), ensure_ascii=False, indent=2, default=_fallback)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return text
