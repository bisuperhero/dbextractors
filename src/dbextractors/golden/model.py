"""Data model of the golden report.

The report is machine readable (JSON) and human readable. Both come out of the
same dataclasses — not two independent code paths — so that they cannot drift
apart.

The order of the levels is not arbitrary. They are ordered the way things
break: if the table does not exist there is no point comparing columns, and if
the column names do not match there is no point comparing checksums over them.
The report therefore always states the level at which it broke first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class Level(IntEnum):
    """The five comparison levels, in the order they are performed."""

    EXISTENCE_AND_ROWS = 1
    COLUMN_NAMES = 2
    COLUMN_TYPES = 3
    COLUMN_CHECKSUMS = 4
    ROW_HASHES = 5


LEVEL_TITLES: dict[Level, str] = {
    Level.EXISTENCE_AND_ROWS: "existence and row count",
    Level.COLUMN_NAMES: "column names and their order",
    Level.COLUMN_TYPES: "data types and nullability",
    Level.COLUMN_CHECKSUMS: "per-column checksums",
    Level.ROW_HASHES: "row comparison through row_hash",
}

#: Level 2 is the most important one: the legacy write path through
#: `mage_ai.io.postgres` prefixes 825 reserved words with an underscore, and the
#: dbt layer rests on those names. A test that compares only data and not names
#: is worthless. See the column naming section of ``docs/legacy-compat.md``.
CRITICAL_LEVEL = Level.COLUMN_NAMES

VERDICT_MATCH = "MATCH"
#: The data matches, but a difference remains that has been decided to be fine.
#: Deliberately a **different** state from a clean match: "it is the same" must
#: not be confusable with "it differs, but we know about it". See
#: `dbextractors.golden.deviations`.
VERDICT_MATCH_WITH_DEVIATIONS = "MATCH*"
VERDICT_DIFF = "DIFF"
VERDICT_ERROR = "ERROR"


@dataclass(frozen=True)
class Relation:
    """One table taking part in a comparison."""

    schema: str
    table: str

    @classmethod
    def parse(cls, text: str) -> Relation:
        """Accepts ``schema.table``. Refuses a bare name — an implicit
        ``search_path`` is exactly the kind of ambiguity an oracle must not
        have."""
        schema, sep, table = text.partition(".")
        if not sep or not schema or not table:
            raise ValueError(f"expected 'schema.table', got {text!r}")
        return cls(schema=schema, table=table)

    def qualified(self) -> str:
        return f'"{self.schema}"."{self.table}"'

    def __str__(self) -> str:
        return f"{self.schema}.{self.table}"


@dataclass
class Difference:
    """One concrete difference. This is what gets shown to a person."""

    what: str
    left: Any = None
    right: Any = None
    #: Where it is — a column name, a primary key value and so on.
    where: Optional[str] = None

    def to_dict(self) -> dict:
        return {"what": self.what, "left": self.left, "right": self.right, "where": self.where}


@dataclass
class LevelResult:
    """The outcome of one of the five levels."""

    level: Level
    passed: bool
    #: The level did not run at all (because level 1 failed, for instance).
    skipped: bool = False
    #: The comparison is necessarily approximate. This must not pass silently —
    #: the report has to **say so explicitly**.
    approximate: bool = False
    differences: list[Difference] = field(default_factory=list)
    #: Differences covered by a rule in `dbextractors.golden.deviations`. They
    #: are printed, not suppressed — they merely do not fail the level.
    deviations: list[Difference] = field(default_factory=list)
    #: Notes for a person: what had to be worked around, what was estimated.
    notes: list[str] = field(default_factory=list)
    #: Numbers something else can key off mechanically.
    stats: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return LEVEL_TITLES[self.level]

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "title": self.title,
            "passed": self.passed,
            "skipped": self.skipped,
            "approximate": self.approximate,
            "differences": [d.to_dict() for d in self.differences],
            "deviations": [d.to_dict() for d in self.deviations],
            "notes": self.notes,
            "stats": self.stats,
        }


@dataclass
class TableReport:
    """The outcome of comparing one pair of tables."""

    left: Relation
    right: Relation
    levels: list[LevelResult] = field(default_factory=list)
    duration_s: float = 0.0
    #: Set when the comparison itself raised. The verdict is then ERROR —
    #: deliberately not folded into DIFF, so that "it differs" cannot be
    #: confused with "we do not know whether it differs".
    error: Optional[str] = None
    label: Optional[str] = None

    @property
    def verdict(self) -> str:
        if self.error is not None:
            return VERDICT_ERROR
        if self.first_failed_level is not None:
            return VERDICT_DIFF
        if any(r.deviations for r in self.levels):
            return VERDICT_MATCH_WITH_DEVIATIONS
        return VERDICT_MATCH

    @property
    def first_failed_level(self) -> Optional[Level]:
        for result in sorted(self.levels, key=lambda r: r.level):
            if not result.passed and not result.skipped:
                return result.level
        return None

    @property
    def is_approximate(self) -> bool:
        return any(r.approximate for r in self.levels)

    def level(self, level: Level) -> Optional[LevelResult]:
        return next((r for r in self.levels if r.level == level), None)

    def to_dict(self) -> dict:
        failed = self.first_failed_level
        return {
            "label": self.label,
            "left": str(self.left),
            "right": str(self.right),
            "verdict": self.verdict,
            "first_failed_level": int(failed) if failed is not None else None,
            "approximate": self.is_approximate,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "levels": [r.to_dict() for r in sorted(self.levels, key=lambda r: r.level)],
        }


@dataclass
class BatchReport:
    """Summary of a batch run. Migration happens in groups of 20–250 tables."""

    reports: list[TableReport] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def matched(self) -> list[TableReport]:
        return [r for r in self.reports if r.verdict == VERDICT_MATCH]

    @property
    def differing(self) -> list[TableReport]:
        return [r for r in self.reports if r.verdict == VERDICT_DIFF]

    @property
    def errored(self) -> list[TableReport]:
        return [r for r in self.reports if r.verdict == VERDICT_ERROR]

    @property
    def verdict(self) -> str:
        if self.errored:
            return VERDICT_ERROR
        return VERDICT_DIFF if self.differing else VERDICT_MATCH

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "counts": {
                "total": len(self.reports),
                "match": len(self.matched),
                "diff": len(self.differing),
                "error": len(self.errored),
            },
            "tables": [r.to_dict() for r in self.reports],
        }
