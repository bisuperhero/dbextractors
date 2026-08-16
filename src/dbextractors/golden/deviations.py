"""Deviations that have been decided to be acceptable.

Without this, the golden test reports DIFF even where the new component differs
**on purpose** — and in a report covering some 530 hash-loaded tables a real bug
would drown in the expected noise.

## The rules are narrow on purpose

Each rule covers one concrete, justified difference and nothing else. There must
be no general "do not compare types": that would suppress exactly the real
difference. A rule therefore consists of:

- **a condition on the left and the right side at once** — "column `_x`" is not
  enough, the pair of types has to match too,
- **a recorded reason**, so that the why can be looked up later.

A deviation is **printed** in the report, not suppressed. The table's verdict is
then ``MATCH*`` — a different state from a clean match, so that "it is the same"
cannot be mistaken for "it differs, but we know about it".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from dbextractors.golden.model import Difference, Level

#: Target columns that came from a reserved word are recognisable by the leading
#: underscore. No source column starts with an underscore, so this is a reliable
#: marker. See the column naming section of ``docs/legacy-compat.md``.
_PREFIXED = re.compile(r"^_[a-z0-9_]+$")

#: The type the old side falls back to when it fails to find a type mapping.
_INFERRED = "text"


def _base_type(type_spec: object) -> str:
    """Type without nullability. ``'text NULL'`` -> ``'text'``."""
    text = str(type_spec)
    for suffix in (" NOT NULL", " NULL"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _nullability(type_spec: object) -> str:
    return "NOT NULL" if str(type_spec).endswith("NOT NULL") else "NULL"


#: Columns the package adds to the target. Their order **relative to each other**
#: is the only thing the rule below may forgive — and only when both sides fill
#: those positions with the same set.
_MANAGED = frozenset({"row_hash", "_timestamp", "_deleted_in_source", "_source"})

#: How `compare` describes a difference in column order.
_ORDER = "different column order"


@dataclass(frozen=True)
class Deviation:
    """One named rule."""

    name: str
    #: Why this is acceptable. Printed in the report next to the difference.
    reason: str
    level: Level
    matches: Callable[[Difference], bool]


def _deleted_in_source_nullability(diff: Difference) -> bool:
    return (
        diff.where == "_deleted_in_source"
        and str(diff.left) == "boolean NULL"
        and str(diff.right) == "boolean NOT NULL"
    )


def _reserved_column_lost_its_type(diff: Difference) -> bool:
    """The old side fails to find a type mapping for a prefixed column.

    The predecessors fill ``overwrite_types`` with keys that have **no**
    underscore prefix, while the dataframe column already carries one — so the
    lookup misses and the type is inferred from the data, which almost always
    means ``text``. The rule is therefore narrow in three ways at once:

    - the column has to start with an underscore (a reserved word),
    - the **base type** on the left has to be exactly ``text`` and something else
      on the right — a difference between two concrete types (``integer`` vs
      ``bigint``) does not fall under the rule,
    - **nullability has to match.** Without that condition the rule would also
      swallow a genuine ``text NULL`` -> ``text NOT NULL`` change, which is a
      real difference. Caught by `test_it_finds_a_nullability_change`.
    """
    return (
        diff.where is not None
        and bool(_PREFIXED.match(diff.where))
        and diff.where != "_deleted_in_source"
        and _base_type(diff.left) == _INFERRED
        and _base_type(diff.right) != _INFERRED
        and _nullability(diff.left) == _nullability(diff.right)
    )


#: How `compare` describes a column that only the new side has.
_SURPLUS = "column only on the right"

#: Columns that **have to** appear where the old side does not have them at all.
#: `_source` does not belong here: it is only added for multi-source tables, and
#: a surplus one would be a genuine finding.
_MANDATORY = frozenset({"row_hash", "_timestamp", "_deleted_in_source"})


def _mandatory_column_added(diff: Difference) -> bool:
    """The new side added a column the old one lacks — and it is a required one.

    This is not an edge case but the **default state of the migration**: variant
    B's Firebird extractor has neither ``row_hash`` nor ``_deleted_in_source`` at
    all, and five further predecessor files only declare ``_deleted_in_source``
    and never flip it. Meanwhile 133 dbt models depend on ``_deleted_in_source``
    and another 5 on ``row_hash``/``_timestamp``.

    So the golden test used to report DIFF for every Firebird table because the
    new side **did what it was asked to**. The rule is narrow: it holds only for
    those three columns by name and only in the "only on the right" direction. A
    missing column, a rename, or a surplus ``_source`` do not fall under it.
    """
    return str(diff.what) == _SURPLUS and str(diff.right) in _MANDATORY and diff.left is None


def _managed_columns_reordered(diff: Difference) -> bool:
    """Reordering **among the managed columns**, nowhere else.

    For MSSQL the old side produces ``_deleted_in_source, _timestamp, _source,
    row_hash``, whereas the new one produces ``row_hash, _timestamp,
    _deleted_in_source, _source``. The latter is the order the predecessor
    **declares** for itself in ``all_columns_ordered``; the former arises by
    accident from the key order of the ``meta`` dict, which its export
    preparation pours into the dataframe earlier.

    The rule is narrow enough that it cannot be abused for anything else:

    - **both** sides have to be one of the four columns the package adds. A
      reordered source column, a missing column and a rename do not fall under
      it,
    - it only shows up on a table created from scratch. When the target already
      exists, the shadow table is created through ``LIKE target``, so it inherits
      the order and no difference arises.
    """
    return str(diff.what) == _ORDER and str(diff.left) in _MANAGED and str(diff.right) in _MANAGED


#: Every rule. Adding one requires a recorded decision.
DEVIATIONS: Tuple[Deviation, ...] = (
    Deviation(
        name="_deleted_in_source is NOT NULL",
        reason=(
            "The new core creates the column as NOT NULL DEFAULT false, today's tables "
            "have it nullable. The stricter definition breaks nothing and saves 133 dbt "
            "models from having to handle NULL. Agreed."
        ),
        level=Level.COLUMN_TYPES,
        matches=_deleted_in_source_nullability,
    ),
    Deviation(
        name="type map applied to reserved columns as well",
        reason=(
            "For a column with an underscore prefix, the old side fails to find the type "
            "map (its keys have no prefix) and infers the type from the data. The new side "
            "applies it correctly. Agreed — the bug is not to be repeated."
        ),
        level=Level.COLUMN_TYPES,
        matches=_reserved_column_lost_its_type,
    ),
    Deviation(
        name="required column the old side does not have",
        reason=(
            "`_deleted_in_source` (133 dbt models), `row_hash` and `_timestamp` (another 5) "
            "have to be present in every strategy. Variant B's Firebird extractor does not "
            "have them at all, and five further predecessor files only declare "
            "`_deleted_in_source` and never flip it. The new side supplies them — that is "
            "the requirement being met, not a difference. Agreed."
        ),
        level=Level.COLUMN_NAMES,
        matches=_mandatory_column_added,
    ),
    Deviation(
        name="different order of the columns the package adds",
        reason=(
            "The new side keeps the order the predecessor declares for itself "
            "(row_hash, _timestamp, _deleted_in_source, _source). The old side has a "
            "different one, but only by accident — it comes from the key order of the "
            "`meta` dict. This only affects tables created from scratch; an existing one "
            "keeps its order through LIKE target. Agreed."
        ),
        level=Level.COLUMN_NAMES,
        matches=_managed_columns_reordered,
    ),
)


def classify(
    level: Level, differences: List[Difference]
) -> Tuple[List[Difference], List[Difference]]:
    """Splits differences into genuine ones and agreed deviations.

    Returns ``(genuine, deviations)``. The order inside both lists is preserved.
    """
    real: List[Difference] = []
    accepted: List[Difference] = []
    for diff in differences:
        rule = match(level, diff)
        if rule is None:
            real.append(diff)
        else:
            accepted.append(diff)
    return real, accepted


def match(level: Level, diff: Difference) -> Optional[Deviation]:
    """Which rule covers the difference, or ``None``."""
    for rule in DEVIATIONS:
        if rule.level == level and rule.matches(diff):
            return rule
    return None


__all__ = ["DEVIATIONS", "Deviation", "classify", "match"]
