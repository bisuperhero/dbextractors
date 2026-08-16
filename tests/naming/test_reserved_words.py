"""The safety net under the reserved-word list — part of the column-naming contract.

The decision was to **replicate** the list rather than depend on ``mage_ai`` at
runtime. The price of that decision is this file: the replica has to be provably
identical, not "close enough".

The tests come in three tiers, because each catches a different mistake:

1. **The generated module against the source JSON** — catches a hand edit of
   ``_reserved_words.py``. Always runs.
2. **Our ``clean_column_name`` against the source in** ``docs/data/`` — catches the
   replica having been written from memory rather than from the source. Always runs.
3. **Everything against the live ``mage_ai``** — catches a Mage upgrade. Skipped when
   the library is absent (marker ``needs_mage``); in CI it runs in a job on the
   production image.

Tier 3 is the one that makes this worth having: if Mage added a word to the list, a
column would be silently renamed across hundreds of tables. This way it fails in CI
instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dbextractors.core import naming
from dbextractors.core._reserved_words import SQL_RESERVED_WORDS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_JSON = REPO_ROOT / "docs" / "data" / "mage_sql_reserved_words.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(SOURCE_JSON.read_text(encoding="utf-8"))


# --- Tier 1: the generated module matches its source ----------------------


def test_the_word_count_matches(payload: dict) -> None:
    assert payload["count"] == 825
    assert len(SQL_RESERVED_WORDS) == payload["count"]


def test_every_word_from_the_source_is_in_the_module(payload: dict) -> None:
    """Item by item, not just the count — two swapped words would hide in a count."""
    missing = [w for w in payload["words"] if w not in SQL_RESERVED_WORDS]
    assert missing == []


def test_the_module_has_nothing_extra(payload: dict) -> None:
    extra = sorted(SQL_RESERVED_WORDS - set(payload["words"]))
    assert extra == []


def test_the_generator_is_up_to_date() -> None:
    """``_reserved_words.py`` has to be exactly what the generator produces.

    Without this the module could be hand-"fixed" and drift away from its source.
    """
    result = subprocess.run(
        [sys.executable, "scripts/gen_reserved_words.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_words_are_upper_case() -> None:
    """The test inside mage is ``col_new.upper() in SQL_RESERVED_WORDS``.

    A lower-case word in the list would never match, and the prefix would stop being
    added for it.
    """
    assert [w for w in SQL_RESERVED_WORDS if w != w.upper()] == []


# --- Tier 2: the replica matches the source in docs/data/ -----------------


def test_impact_on_real_columns() -> None:
    """The five most frequent cases across the existing target tables."""
    assert naming.clean_column_name("type") == "_type"
    assert naming.clean_column_name("name") == "_name"
    assert naming.clean_column_name("date") == "_date"
    assert naming.clean_column_name("hour") == "_hour"
    assert naming.clean_column_name("order") == "_order"


def test_the_prefix_is_added_only_to_reserved_words() -> None:
    assert naming.clean_column_name("cena_bez_dph") == "cena_bez_dph"
    assert naming.clean_column_name("zakaznik_id") == "zakaznik_id"
    # `ID` is not in the list — were it there, nearly every primary key would be
    # prefixed.
    assert naming.clean_column_name("id") == "id"
    # `A` on the other hand is. Single-letter columns are not just theory; several
    # MSSQL sources have them.
    assert naming.clean_column_name("a") == "_a"


def test_the_prefix_is_not_added_twice() -> None:
    """``_type`` already has the prefix; ``_TYPE`` is not in the list, so no second one.

    This is why the metadata columns (``_timestamp``, ``_deleted_in_source``) come
    through step 2 unchanged.
    """
    assert naming.clean_column_name("_type") == "_type"
    assert naming.clean_column_name("_timestamp") == "_timestamp"
    assert naming.clean_column_name("_deleted_in_source") == "_deleted_in_source"


def test_load_reserved_words_returns_the_snapshot() -> None:
    assert naming.load_reserved_words() is SQL_RESERVED_WORDS


# --- The prefix rule over the whole list, not a sample ----------------------
#
# Seven pinned words proved too small a net: an implementation that carved out a
# per-word exception (it happened to be "MONTH") passed the suite and CI. The
# rule is one line, so the only faithful test is the exhaustive one — 825 words
# cost milliseconds.


def test_every_reserved_word_gets_the_prefix() -> None:
    """Any per-word exception, typo'd comparison or dropped word fails here."""
    wrong = [
        word
        for word in sorted(naming.load_reserved_words())
        if naming.apply_reserved_word_prefix(word.lower()) != "_" + word.lower()
    ]
    assert wrong == []


def test_a_name_outside_the_list_passes_through_unchanged() -> None:
    """The complement matters as much: a rule that prefixed everything would
    also pass the loop above."""
    assert naming.apply_reserved_word_prefix("xyz") == "xyz"
    assert naming.apply_reserved_word_prefix("cena_bez_dph") == "cena_bez_dph"


def test_the_membership_test_ignores_case_but_the_prefix_preserves_it() -> None:
    """The rule is ``name.upper() in SQL_RESERVED_WORDS`` — membership is
    case-insensitive, the name itself is left alone. Lower-casing is step 2a's
    job (`mage_clean_name`), and doing it here too would hide a missing call."""
    assert naming.apply_reserved_word_prefix("Type") == "_Type"
    assert naming.apply_reserved_word_prefix("TYPE") == "_TYPE"
    assert naming.apply_reserved_word_prefix("type") == "_type"


def test_no_reserved_word_starts_with_an_underscore() -> None:
    """Precondition of the idempotence loop below: were ``_SOMETHING`` ever on
    the list, an already-prefixed name would gain a second underscore and the
    target contract would break."""
    assert [w for w in SQL_RESERVED_WORDS if w.startswith("_")] == []


def test_an_already_prefixed_name_is_never_prefixed_again() -> None:
    """``_type`` upper-cases to ``_TYPE``, which is not on the list, so the
    function is a no-op on its own output. This is what lets the metadata
    columns (``_timestamp``, ``_deleted_in_source``) pass through step 2 — and
    what makes re-running the chain over an already-normalised name safe."""
    wrong = [
        word
        for word in sorted(naming.load_reserved_words())
        if naming.apply_reserved_word_prefix("_" + word.lower()) != "_" + word.lower()
    ]
    assert wrong == []


# --- Tier 3: against the live mage_ai -------------------------------------


def _mage_base():
    return pytest.importorskip(
        "mage_ai.io.base",
        reason="mage_ai is not installed; in CI a job on the production image covers this",
    )


#: The inputs the replica is compared against the library on. They are not random —
#: every line is one step of ``clean_name`` or a case seen in real data. Several of
#: them are deliberately non-ASCII and are written as escapes so that the file itself
#: stays ASCII while the strings under test do not change.
SAMPLE_NAMES = [
    "type",
    "name",
    "ORDER",
    "Cena Bez DPH",
    "cena-bez-dph",
    "cena_bez_dph",
    "  mezery  ",
    "v\xedce\tb\xedl\xfdch\nznak\u016f",
    'uvozovky"a$dolar',
    "\ufeffbom_na_zacatku",
    "2023_trzby",
    "23",
    "\u010de\u0161tina_s_diakritikou",
    "sm\xed\u0161en\xe9_\u010cESK\xc9_P\xedsmo",
    "已经",
    "a__b___c",
    "_type",
    "_timestamp",
    "_deleted_in_source",
    "row_hash",
    "",
    "_",
    "__",
    "%",
    "col%pct",
    "tabul\xe1tor\tuprost\u0159ed",
]


@pytest.mark.needs_mage
def test_the_list_matches_the_live_mage() -> None:
    """A Mage upgrade that touches ``SQL_RESERVED_WORDS`` fails right here."""
    base = _mage_base()
    live = set(base.SQL_RESERVED_WORDS)
    assert live - SQL_RESERVED_WORDS == set(), "mage added words, see docs/data/"
    assert SQL_RESERVED_WORDS - live == set(), "mage removed words, see docs/data/"


@pytest.mark.needs_mage
@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_clean_name_matches_the_live_mage(name: str) -> None:
    base = _mage_base()
    assert naming.mage_clean_name(name) == base.clean_name(name)


@pytest.mark.needs_mage
@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_clean_column_name_matches_the_live_mage(name: str) -> None:
    """The whole of step 2 including the prefix, against an unbound ``BaseSQLDatabase``.

    ``_clean_column_name`` never touches ``self``, so it can be called as an unbound
    function and needs no live database connection.
    """
    base = _mage_base()
    expected = base.BaseSQLDatabase._clean_column_name(None, name)
    assert naming.clean_column_name(name) == expected
