"""``convert_nchar_to_varchar`` from ``LOAD_SETTINGS`` to the generated SELECT.

What the conversion does to a value is a property of the server and is pinned
live in ``tests/dialects/test_source_db.py``; which columns it wraps is settled
in ``tests/dialects/test_mssql.py``. What is left — and what these tests cover —
is the wiring in between: that the key reaches the SELECT at all, that it reaches
it through the **one** plumbing point every strategy funnels through
(`full.FullLoadStrategy._build_select`, reached from `read_batches`), and that a
dialect with no national-character types is left alone.

None of this needs a database: a SELECT is a string, and the fake dialect is the
only way to ask a dialect that is *not* MSSQL the same question.
"""

from __future__ import annotations

import pytest

from dbextractors.core.strategies.full import FullLoadStrategy
from fakes import FakeDialect, make_columns, make_context

#: An N-typed column, a single-byte one and a number — the same three cases the
#: MSSQL fixture has, in the shape `make_columns` wants.
COLUMNS = [
    ("id", "int", "INTEGER"),
    ("note", "nvarchar", "TEXT"),
    ("legacy_note", "varchar", "VARCHAR"),
]


class NcharFakeDialect(FakeDialect):
    """A fake source that has national-character types, as MSSQL does.

    Deliberately not `MSSQLDialect` itself: what is being checked here is that a
    *strategy* hands the flag and the introspected types over, and using the real
    dialect would let an assertion pass on MSSQL's own SQL rather than on the
    strategy's behaviour.
    """

    NCHAR_TYPES = frozenset({"nchar", "nvarchar", "ntext"})

    def render_nchar_convert(self, column_name: str) -> str:
        return f"CONVERTED({self.quote_ident(column_name)}) AS {self.quote_ident(column_name)}"


def _select(settings: dict, dialect=None) -> str:
    ctx = make_context(
        None,
        "irrelevant",
        dialect=dialect or NcharFakeDialect(),
        columns=make_columns(COLUMNS),
        settings=settings,
    )
    return FullLoadStrategy()._build_select(ctx, None)


def test_the_select_is_unchanged_when_the_key_is_absent() -> None:
    """The default. Every one of ~670 tables reads exactly what it read before."""
    assert _select({}) == "SELECT `id`, `note`, `legacy_note` FROM `zdroj`"


def test_the_key_reaches_the_select() -> None:
    """The strategy has to pass **both** halves: the flag and the source types.

    Passing only the flag would leave the dialect with nothing to decide on and
    convert nothing; passing only the types would convert nothing either. The
    single-byte column staying bare is the half that matters most — it is read
    correctly today.
    """
    assert _select({"convert_nchar_to_varchar": True}) == (
        "SELECT `id`, CONVERTED(`note`) AS `note`, `legacy_note` FROM `zdroj`"
    )


@pytest.mark.parametrize("value", [True, "true", "ano", 1])
def test_the_configuration_spellings_of_true_all_work(value) -> None:
    """A YAML value arrives as a string as often as as a bool, and the whole
    contract is read through `config.is_truthy`."""
    assert "CONVERTED(`note`)" in _select({"convert_nchar_to_varchar": value})


@pytest.mark.parametrize("value", [False, "false", "ne", 0, None, ""])
def test_the_spellings_of_false_leave_the_select_alone(value) -> None:
    assert "CONVERTED" not in _select({"convert_nchar_to_varchar": value})


@pytest.mark.parametrize("name", ["mysql", "postgres", "firebird"])
def test_a_dialect_without_national_character_types_ignores_the_key(name) -> None:
    """The other three sources are read over an encoding that carries their text.

    They accept the two arguments for the sake of the shared signature — the same
    way they accept ``column_types`` in `render_hash_expr` — and must render the
    query they always rendered. A dialect that started wrapping columns here
    would corrupt them, since it has no such conversion to offer.
    """
    from dbextractors.dialects.base import TableRef
    from dbextractors.entrypoint import resolve_dialect

    dialect = resolve_dialect(name)
    types = {"note": "varchar", "id": "int"}
    ref = TableRef(name="t")

    plain = dialect.render_select(["id", "note"], ref, column_types=types)
    assert dialect.render_select(["id", "note"], ref, column_types=types, convert_nchar=True) == (
        plain
    )


def test_setting_the_key_on_such_a_dialect_is_said_out_loud(caplog) -> None:
    """A key that quietly does nothing is the trap this package documents.

    `LoadStrategy.validate` runs before the source is touched, so the message
    arrives at the top of the run rather than after a full read.
    """
    ctx = make_context(
        None,
        "irrelevant",
        dialect=FakeDialect(),
        columns=make_columns(COLUMNS),
        settings={"convert_nchar_to_varchar": True},
    )

    with caplog.at_level("WARNING"):
        FullLoadStrategy()._warn_about_inapplicable_nchar_conversion(ctx)

    assert "convert_nchar_to_varchar" in caplog.text
    assert "fake" in caplog.text, "the message has to name the dialect that ignores it"


def test_nothing_is_warned_about_when_the_dialect_can_convert(caplog) -> None:
    ctx = make_context(
        None,
        "irrelevant",
        dialect=NcharFakeDialect(),
        columns=make_columns(COLUMNS),
        settings={"convert_nchar_to_varchar": True},
    )

    with caplog.at_level("WARNING"):
        FullLoadStrategy()._warn_about_inapplicable_nchar_conversion(ctx)

    assert caplog.text == ""
