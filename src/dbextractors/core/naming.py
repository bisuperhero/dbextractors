"""Normalisation of column names for the target.

**This is the hardest constraint in the whole project:** target column names must
not change.

The predecessor extractors normalise in two steps, and only the first one is
theirs:

1. ``create_column_mapping`` — lower case, spaces to underscores, dropping
   non-alphanumeric characters, collapsing repeated underscores, trimming the ones
   at the edges. Across all 14 files that have the function it is **identical
   character for character**.

2. ``mage_ai.io.postgres.Postgres.export()`` internally calls
   ``BaseSQLDatabase._clean_column_name``, which runs ``clean_name`` and, on top of
   that, **prefixes with an underscore every name whose upper-case form is in the
   list of 825 reserved words** (``mage_ai/io/base.py``, ``SQL_RESERVED_WORDS``).

Hence ``_type`` in 113 tables, ``_name`` in 107, ``_date`` in 92, ``_hour`` in 72,
``_order`` in 42 — even though no column in any source starts with an underscore.
The dbt layer rests on those names; changing them silently breaks hundreds of
models.

That second step is implicit today. The decision taken: replicate it here rather
than depend on ``mage_ai``. This module is therefore derived from Mage — see
``NOTICE`` for the attribution and the licence.

The evidence that replicating is safe, measured on the target database in the
schemas written by the existing extractors:

===========================================  =======
columns in total                               2 713
with an underscore prefix                        226
of those explained by a reserved word/metadata   226
reserved word **without** a prefix                 0
outside ``[a-z0-9_]``                              0
===========================================  =======

The only two exceptions across all ``raw_*`` schemas and the single unexplained
prefix all live in schemas written by other loaders, not by these 15 extractors.

The safety net against a Mage upgrade is in ``tests/naming/`` — the list is compared
both against the frozen snapshot and against the live ``mage_ai`` whenever it is
importable.

Supporting material:

- ``docs/data/mage_sql_reserved_words.json`` — 825 words from ``mageai/mageai:0.9.79``
- ``docs/data/mage_clean_name.py.txt`` — the source of ``clean_name`` and
  ``_clean_column_name``
- ``src/dbextractors/core/_reserved_words.py`` — generated from that JSON
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence

from dbextractors.core._reserved_words import SQL_RESERVED_WORDS

_log = logging.getLogger(__name__)

__all__ = [
    "SQL_RESERVED_WORDS",
    "DEFAULT_HASH_COLUMN",
    "METADATA_COLUMNS",
    "ColumnNameCollision",
    "create_column_mapping",
    "mage_clean_name",
    "apply_reserved_word_prefix",
    "clean_column_name",
    "target_column_names",
    "load_reserved_words",
]

#: Name of the hash column when the configuration does not give one. In the
#: predecessor this fallback sits inline in the body of ``load_data``:
#: ``… if hash_column_cfg else 'row_hash'``.
DEFAULT_HASH_COLUMN = "row_hash"

#: Columns created by the package, not by the source; every strategy has to
#: maintain them. The order is part of the contract — the predecessor appends them
#: in exactly this order, after the hash column.
METADATA_COLUMNS = ("_timestamp", "_deleted_in_source")

#: Characters that ``mage_ai.io.base.clean_name`` deletes **before** the ``\W``
#: substitution, in the order they appear in the source. The original lists the BOM
#: twice (``'\ufeff'`` and ``'\ufeff'`` are the same character), which has no
#: effect — it is kept once here. Written as an escape sequence deliberately: a
#: literal BOM is invisible in an editor.
_CLEAN_NAME_DROPPED = ("\ufeff", '"', "$", "\n", "\r", "\t")


class ColumnNameCollision(ValueError):
    """Two different source columns normalise to the same name.

    Nobody checks for this today: the mapping is a dict keyed by the *source* name,
    so a collision passes through and only breaks later, in ``CREATE TABLE``, with
    "column specified more than once" — or worse, with ``append`` only one of the
    columns is silently written. Better to fail at once, with both names in the
    message.
    """


# --- Step 1: the normalisation the predecessor performs ---------------------


def create_column_mapping(src_columns: list) -> dict:
    """Source name -> normalised name, **without** the underscore prefix.

    A character-for-character transcription of the predecessor (14 files, a single
    variant). Nothing is improved here: ``re.sub(r'[^\\w\\d_]', '')`` *deletes* the
    character (it does not replace it with an underscore, the way step 2 does), and
    ``\\w`` is unicode-aware in Python, so accented letters pass through. Both are
    visible in the target data, so they are not typos waiting to be fixed.

    The trailing ``strip('_')`` is the reason metadata columns **must not** go
    through here — ``_timestamp`` would become ``timestamp``. The predecessor adds
    them after the mapping, exactly as `target_column_names` does here.
    """
    mapping = {}
    for col in src_columns:
        new_name = col.lower()
        new_name = re.sub(r"\s+", "_", new_name)
        new_name = re.sub(r"[^\w\d_]", "", new_name)
        new_name = re.sub(r"_+", "_", new_name)
        new_name = new_name.strip("_")
        mapping[col] = new_name
    return mapping


# --- Step 2: what mage_ai does implicitly today -----------------------------


def mage_clean_name(name: str, *, case_sensitive: bool = False) -> str:
    """Replica of ``mage_ai.io.base.clean_name`` with the defaults it is called with.

    No ``loader.export()`` call in any of the predecessor extractors passes
    ``allow_reserved_words``, ``case_sensitive`` or ``auto_clean_name``, so the
    defaults apply: ``clean_name(name, case_sensitive=False)`` with
    ``allow_characters=None`` and ``allow_number=False``.

    The ``allow_characters`` parameter is deliberately **absent** here. If anyone
    started using it in Mage, that would be a change of contract, and we want to see
    it as a failing test in ``tests/naming/``, not as behaviour quietly carried over.
    """
    for char in _CLEAN_NAME_DROPPED:
        name = name.replace(char, "")

    name = re.sub(r"\W", "_", name)

    if name and re.match(r"\d", name[0]):
        name = f"letter_{name}"

    return name if case_sensitive else name.lower()


def apply_reserved_word_prefix(name: str) -> str:
    """Step 2b — the underscore prefix for reserved words.

    The test is ``col_new.upper() in SQL_RESERVED_WORDS``, i.e. on the **already
    cleaned** name. That is why it runs after `mage_clean_name`, not before it.
    """
    return f"_{name}" if name.upper() in SQL_RESERVED_WORDS else name


def clean_column_name(name: str, *, case_sensitive: bool = False) -> str:
    """Replica of ``BaseSQLDatabase._clean_column_name`` — the whole of step 2.

    It has to produce a **byte-identical result** to mage_ai. Not a "similar" one.
    ``tests/naming/test_reserved_words.py`` verifies it against the live library.
    """
    return apply_reserved_word_prefix(mage_clean_name(name, case_sensitive=case_sensitive))


# --- The whole chain --------------------------------------------------------


def target_column_names(
    src_columns: Sequence[str],
    hash_column: Optional[str] = None,
    *,
    metadata: Iterable[str] = METADATA_COLUMNS,
) -> List[str]:
    """Source names -> target table names, in the right order.

    The order is as much part of the contract as the names — the golden test compares
    it. The predecessor builds it like this: source columns in catalogue order, then
    the hash column, then ``_timestamp``, then ``_deleted_in_source``; a generated
    column is appended only when it is not in the list already.

    **Departure from the predecessor:** ``hash_column=None`` here means "do not add a
    hash column", not "use ``row_hash``". The predecessor has the fallback to
    ``'row_hash'`` hard-coded in the body, so even a table with no hashing gets an
    extra empty column. Whether to add it is a decision for the strategy, not for
    naming — the strategy passes `DEFAULT_HASH_COLUMN` explicitly.
    """
    mapping = create_column_mapping(list(src_columns))

    ordered: List[str] = []
    origin: Dict[str, str] = {}
    for src in src_columns:
        target = clean_column_name(mapping[src])
        previous = origin.get(target)
        if previous is not None:
            raise ColumnNameCollision(
                f"Source columns {previous!r} and {src!r} both normalise to "
                f"{target!r}. The target table would have the column twice; "
                f"exclude one of them via TABLE.excluded_columns."
            )
        origin[target] = src
        ordered.append(target)

    generated: List[str] = []
    if hash_column:
        generated.append(create_column_mapping([hash_column])[hash_column])
    generated.extend(metadata)

    for name in generated:
        target = clean_column_name(name)
        if target in origin:
            # The source has a column named like a generated one. In that case the
            # predecessor does not append it again (the condition `if gen_col_norm
            # not in column_types_df[...]`) — the column stays in its original
            # position and the value we compute overwrites it.
            _log.warning(
                "The source has a column %r that collides with a generated column. "
                "It keeps its position from the source and its values will be "
                "overwritten.",
                target,
            )
            continue
        origin[target] = name
        ordered.append(target)

    return ordered


def load_reserved_words() -> frozenset:
    """The reserved words the underscore prefix is decided against.

    Returns the frozen snapshot from ``dbextractors.core._reserved_words``, not the
    contents of the installed ``mage_ai``. Deliberately so: column names must not
    change even when Mage is upgraded. Divergence from the library is caught by a
    test, not at runtime.
    """
    return SQL_RESERVED_WORDS


def reconcile_with_existing(desired: Sequence[str], existing: Sequence[str]) -> List[str]:
    """Reconciles the desired column order with the one the target has today.

    **Decided:** an existing target table is never reordered. The predecessors build
    the order of managed columns in six different ways, and three files even differ
    from themselves depending on whether the source was empty on the first run — so
    the order a given table has cannot be derived from the configuration. The only
    reliable thing is to take what is actually in the target.

    Columns the target has but ``desired`` does not (typically ones excluded through
    ``excluded_columns``) are left out of the result: we do not write them, so they
    do not belong in the column list for `COPY` either.

    New columns not yet in the target are appended **at the end**, in ``desired``
    order — inserting a column in the middle would be exactly the reordering we are
    avoiding.

    When the target does not exist (empty ``existing``), ``desired`` applies
    unchanged.
    """
    if not existing:
        return list(desired)

    wanted = set(desired)
    out = [c for c in existing if c in wanted]
    known = set(out)
    out.extend(c for c in desired if c not in known)
    return out
