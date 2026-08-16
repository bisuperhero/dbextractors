"""MySQL, MSSQL and Firebird against a **live** source.

Every other test for these three works on strings: it checks the SQL that comes
out of the dialect, never what comes back. That leaves the bug class this file
exists for — a type that maps correctly and whose value still does not arrive
intact.

There is a precedent for taking it seriously. PostgreSQL's ``money`` maps to
``NUMERIC`` perfectly well on paper, and the driver hands it over as a
*localised string* (``'3,50 EUR'``), so `COPY` into a NUMERIC column fails. That
was found only once a live test existed. These three sources had none at all,
and each of them has its own version of the same trap:

- **MySQL** has a zero date that is not a date, and a ``TIME`` that is really a
  signed interval and legitimately exceeds 24 hours.
- **MSSQL** has three date/time families with different precision, and ``MONEY``,
  which is fixed point but not ``DECIMAL``.
- **Firebird** reports ``NUMERIC`` scale as a **negative** number, pads ``CHAR``
  with spaces, and counts days from 1858 rather than from the Unix epoch.

The seeds in ``docker/seed/`` put all of that into one table per source. Nothing
here writes to a source; they are read-only fixtures.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from source_db import engine_for, parse_dsn, require

from dbextractors.dialects.base import TableRef
from dbextractors.entrypoint import resolve_dialect

#: Source name -> (marker, table reference as that source spells it).
#:
#: The spelling is the point: Firebird upper-cases unquoted identifiers, MSSQL
#: puts the table in a schema, and MySQL has no schema at all. Introspection has
#: to cope with all three from the same configuration shape.
SOURCES = [
    pytest.param("mysql", TableRef(name="types_wide"), marks=pytest.mark.needs_mysql, id="mysql"),
    pytest.param(
        "mssql",
        TableRef(name="types_wide", schema="dbo"),
        marks=pytest.mark.needs_mssql,
        id="mssql",
    ),
    pytest.param(
        "firebird",
        TableRef(name="TYPES_WIDE"),
        marks=pytest.mark.needs_firebird,
        id="firebird",
    ),
]


def _read_all(name: str, ref: TableRef) -> tuple[list[str], list[tuple]]:
    """Every row of the seeded table, read the way the package reads.

    Deliberately through the dialect's own ``render_select`` rather than a
    hand-written query: pagination and quoting are where the four dialects
    differ most, and SQL that a test wrote itself would verify the database
    instead of the code that has to talk to it.

    This is the **driver** path — one step below what a load actually does. It
    shows what the driver hands over before pandas gets its hands on it, which
    is where a type that maps on paper first goes wrong. `_read_frame` below
    takes the other step; both matter, and they do not agree on everything.
    """
    require(name)
    dialect = resolve_dialect(name)
    eng = engine_for(name)
    try:
        columns = dialect.introspect_columns(eng, ref)
        names = [c.name for c in columns]
        sql = dialect.render_select(names, ref)
        with eng.connect() as conn:
            rows = [tuple(row) for row in conn.exec_driver_sql(sql)]
    finally:
        eng.dispose()
    return names, rows


def _read_batches(
    name: str, ref: TableRef, batch_size: int = 100, *, convert_nchar: bool = False
) -> list:
    """The same rows through the **production** path — ``dialect.iter_batches``.

    Not the same call as `_read_all`, and that is the point: a load never sees
    driver objects, it sees whatever pandas made of them, and it sees them a
    batch at a time. Neither of those was exercised anywhere against a live
    source — the fake dialect the strategy tests use ignores ``batch_size``
    entirely, so batching had no live check at all.

    ``convert_nchar`` is ``LOAD_SETTINGS.convert_nchar_to_varchar`` as
    `full._build_select` passes it on: the flag together with the introspected
    source types, never one without the other.
    """
    require(name)
    dialect = resolve_dialect(name)
    eng = engine_for(name)
    try:
        columns = dialect.introspect_columns(eng, ref)
        sql = dialect.render_select(
            [c.name for c in columns],
            ref,
            column_types={c.name: c.source_type for c in columns},
            convert_nchar=convert_nchar,
        )
        return list(dialect.iter_batches(eng, sql, batch_size))
    finally:
        eng.dispose()


def _row(names: list[str], rows: list[tuple], row_id: int) -> dict:
    """One seeded row as ``{column: value}``, found by its ``id``."""
    id_at = next(i for i, n in enumerate(names) if n.lower() == "id")
    match = next(row for row in rows if row[id_at] == row_id)
    return dict(zip(names, match, strict=True))


# --- introspection -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ref"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_introspection_finds_every_column(name: str, ref: TableRef) -> None:
    """Introspection has to see the whole table, not most of it.

    A column that introspection misses is silently dropped from the target, and
    nothing downstream notices until a model asks for it.
    """
    require(name)
    dialect = resolve_dialect(name)
    eng = engine_for(name)
    try:
        columns = dialect.introspect_columns(eng, ref)
    finally:
        eng.dispose()

    names = {c.name.lower() for c in columns}
    assert "id" in names
    assert {
        "type",
        "name",
        "order",
    } <= names, f"{name}: reserved-word columns are missing from introspection: {sorted(names)}"


@pytest.mark.parametrize(
    ("name", "ref"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_every_column_gets_a_target_type(name: str, ref: TableRef) -> None:
    """No column may arrive without a mapped PostgreSQL type.

    An empty ``pg_type`` means the target column would be created as whatever
    the writer guessed, which is how a numeric column ends up as text.
    """
    require(name)
    dialect = resolve_dialect(name)
    eng = engine_for(name)
    try:
        columns = dialect.introspect_columns(eng, ref)
    finally:
        eng.dispose()

    unmapped = [c.name for c in columns if not c.pg_type]
    assert not unmapped, f"{name}: no target type for {unmapped}"


#: Source column -> the PostgreSQL type it has to map to. Only the traps: the
#: columns where the mapping is a decision rather than a transcription.
#:
#: This is the assertion that was missing. Checking that *some* type came back
#: passes just as happily when ``datetime`` maps to ``TEXT`` — and a timestamp
#: column silently created as text is not a failure, it is a modelling layer
#: quietly reading strings for a year.
EXPECTED_PG_TYPES: dict[str, dict[str, str]] = {
    "mysql": {
        # `tinyint` is INTEGER, not BOOLEAN: the promotion to boolean is decided
        # from the data (an enum of 0/1), never from the declared type.
        "flag": "INTEGER",
        "bit_value": "BOOLEAN",
        "zero_date": "DATE",
        "zero_datetime": "TIMESTAMP",
        "long_time": "TIME",
        "big_unsigned": "BIGINT",
        "money_amount": "NUMERIC",
        "binary_blob": "BYTEA",
        "payload": "JSONB",
        "enum_value": "TEXT",
        "set_value": "TEXT",
        "year_value": "INTEGER",
    },
    "mssql": {
        "money_amount": "NUMERIC",
        "small_money": "NUMERIC",
        "precise_value": "NUMERIC",
        # All three date/time families land on the same target type; they differ
        # in precision, not in kind.
        "legacy_datetime": "TIMESTAMP",
        "modern_datetime": "TIMESTAMP",
        "zoned_datetime": "TIMESTAMP",
        "day_date": "DATE",
        "time_of_day": "TIME",
        "flag": "BOOLEAN",
        "key_uuid": "TEXT",
        "xml_payload": "TEXT",
        # A computed column is read like any other.
        "doubled": "INTEGER",
    },
    "firebird": {
        # Scale arrives negated from the driver, and it is the **sign** that
        # decides between an integer and a NUMERIC. Getting that wrong does not
        # fail; it moves the decimal point.
        "MONEY_AMOUNT": "NUMERIC",
        "SMALL_SCALE": "NUMERIC",
        "NO_SCALE": "NUMERIC",
        "DECIMAL_VALUE": "NUMERIC",
        "BIG_VALUE": "BIGINT",
        "DAY_DATE": "DATE",
        "STAMP": "TIMESTAMP",
        "TIME_OF_DAY": "TIME",
        "FLAG_CHAR": "TEXT",
        "FLAG_SMALLINT": "INTEGER",
        # Subtype decides: a text blob is TEXT, a binary one is BYTEA.
        "BLOB_TEXT": "TEXT",
        "BLOB_BINARY": "BYTEA",
        "FLOAT_VALUE": "DOUBLE PRECISION",
    },
}


@pytest.mark.parametrize(
    ("name", "ref"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_the_traps_map_to_the_type_they_have_to(name: str, ref: TableRef) -> None:
    """Each source's awkward types land where the target expects them."""
    require(name)
    dialect = resolve_dialect(name)
    eng = engine_for(name)
    try:
        columns = dialect.introspect_columns(eng, ref)
    finally:
        eng.dispose()

    mapped = {c.name: c.pg_type for c in columns}
    for column, expected in EXPECTED_PG_TYPES[name].items():
        assert mapped[column] == expected, f"{name}.{column}: {mapped[column]} != {expected}"


@pytest.mark.parametrize(
    ("name", "ref"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_reserved_words_are_prefixed_in_the_target(name: str, ref: TableRef) -> None:
    """`type`, `name` and `order` must gain an underscore on the way in.

    This is the constraint the whole package is built around: the target column
    names are the one thing that must not change, because a large modelling
    layer is built on them. See the column naming section of
    ``docs/legacy-compat.md``.

    Naming is applied to what introspection returned, not by introspection
    itself: the dialect leaves ``target_name`` as ``None`` on purpose, because
    the rule is identical for all four sources and belongs in one place. What
    this adds over the unit tests of that rule is the real input — Firebird
    hands the name over upper-cased, and the rule has to cope with that too.
    """
    from dbextractors.core.naming import clean_column_name

    require(name)
    dialect = resolve_dialect(name)
    eng = engine_for(name)
    try:
        columns = dialect.introspect_columns(eng, ref)
    finally:
        eng.dispose()

    by_source = {c.name.lower(): clean_column_name(c.name) for c in columns}
    for reserved in ("type", "name", "order"):
        assert (
            by_source[reserved] == f"_{reserved}"
        ), f"{name}: {reserved!r} became {by_source[reserved]!r}, expected '_{reserved}'"


# --- reading -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ref"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_reading_returns_every_seeded_row(name: str, ref: TableRef) -> None:
    """Three rows go in, three rows come out.

    Deliberately the dialect's own batch query rather than a hand-written
    SELECT: pagination is where the dialects differ most (``LIMIT`` versus
    ``FIRST``/``SKIP`` versus ``OFFSET … ROWS``), and a query that a test wrote
    itself would verify the database rather than the package.
    """
    _, rows = _read_all(name, ref)

    assert len(rows) == 3, f"{name}: expected 3 seeded rows, got {len(rows)}"


@pytest.mark.parametrize(
    ("name", "ref"), [(p.values[0], p.values[1]) for p in SOURCES], ids=[p.id for p in SOURCES]
)
def test_a_row_of_nulls_survives_reading(name: str, ref: TableRef) -> None:
    """Row 2 of every seed is nothing but NULLs where NULLs are allowed.

    A source that turns NULL into an empty string, or into the string 'None',
    corrupts the target quietly — the row count matches and the data does not.
    """
    names, rows = _read_all(name, ref)
    row = _row(names, rows, 2)

    # Every column the seed leaves NULL, not "at least one of them". `any()`
    # passed while seventeen of eighteen NULLs were corrupted, which is the same
    # blind spot as counting rows and calling it data.
    not_null = {c: v for c, v in row.items() if c in NULLABLE_IN_ROW_TWO[name] and v is not None}
    assert not not_null, f"{name}: these came back non-NULL: {not_null}"


#: Columns the second seeded row leaves NULL. Not "everything but the id": MySQL
#: has a few that cannot be NULL in that row (its zero date and zero datetime are
#: seeded as ``0000-00-00``, which is a different trap and is checked on its own).
NULLABLE_IN_ROW_TWO: dict[str, tuple[str, ...]] = {
    "mysql": ("type", "name", "order", "binary_blob", "payload", "enum_value"),
    "mssql": (
        "type",
        "name",
        "order",
        "money_amount",
        "small_money",
        "precise_value",
        "legacy_datetime",
        "modern_datetime",
        "zoned_datetime",
        "day_date",
        "time_of_day",
        "flag",
        "cp1250_text",
        "unicode_text",
        "binary_blob",
        "key_uuid",
        "xml_payload",
        "doubled",
    ),
    "firebird": (
        "TYPE",
        "NAME",
        "ORDER",
        "MONEY_AMOUNT",
        "SMALL_SCALE",
        "NO_SCALE",
        "DECIMAL_VALUE",
        "DAY_DATE",
        "STAMP",
        "TIME_OF_DAY",
        "FLAG_CHAR",
        "FLAG_SMALLINT",
        "PADDED",
        "NOT_PADDED",
        "CZECH_TEXT",
        "BLOB_TEXT",
        "BLOB_BINARY",
        "BIG_VALUE",
        "FLOAT_VALUE",
    ),
}


# --- values ------------------------------------------------------------------
#
# The file existed to prove that values survive the trip and did not check a
# single one of them: the assertions were on counts and on column names, so the
# mutation ``datetime -> TEXT`` in the MySQL type map survived even the live
# tests. What follows is the missing half — the seeded values, spelled out.
#
# ``utf8mb4_text`` in MySQL used to come back double-encoded: neither the
# official image's docker-entrypoint-initdb.d runner nor CI's plain
# ``mysql ... < file`` invocation ever passed a client charset, so the session
# negotiated ``latin1`` and the server decoded the file's UTF-8 bytes as
# latin1 before re-encoding them as utf8mb4 on storage. Fixed in
# ``docker/seed/mysql/001_types.sql`` with a leading ``SET NAMES utf8mb4;`` —
# that statement governs the session regardless of which client loads the
# file, so both the local and the CI seeding paths pick it up. Asserted below.
#
# ``unicode_text`` in MSSQL is a different animal and is **not** a seed defect.
# It is an inherited limitation of the client charset, pinned by the tests under
# "the MSSQL client charset" further down and written up in
# ``docs/legacy-compat.md``.


@pytest.mark.needs_mysql
def test_mysql_values_arrive_intact() -> None:
    """MySQL's own traps: a signed interval past 24 hours, an unsigned BIGINT
    that does not fit a signed one, and fixed point that must not go via float."""
    names, rows = _read_all("mysql", TableRef(name="types_wide"))
    first, third = _row(names, rows, 1), _row(names, rows, 3)

    assert first["money_amount"] == Decimal("1234.5678")
    assert first["big_unsigned"] == 18446744073709551615
    assert first["zero_date"] == dt.date(2026, 1, 31)
    assert first["zero_datetime"] == dt.datetime(2026, 1, 31, 12, 34, 56)
    assert first["binary_blob"] == b"\xde\xad\xbe\xef"
    assert first["enum_value"] == "a"
    assert first["set_value"] == {"x", "y"}
    assert first["bit_value"] == 170
    assert first["year_value"] == 2026
    # Four-byte characters and Czech diacritics both round-trip; the seed used
    # to hand this back double-encoded (`SET NAMES utf8mb4;` fixed it).
    assert first["utf8mb4_text"] == "příliš žluťoučký kůň 😀"

    # TIME is an interval, not a wall clock: 25 hours and -1:30 are both legal
    # and neither is representable as a `datetime.time`.
    assert first["long_time"] == dt.timedelta(hours=25)
    assert first["negative_time"] == -dt.timedelta(hours=1, minutes=30)
    assert third["long_time"] == dt.timedelta(hours=838, minutes=59, seconds=59)
    assert third["negative_time"] == -dt.timedelta(hours=838, minutes=59, seconds=59)

    assert third["big_unsigned"] == 9223372036854775808
    assert third["money_amount"] == Decimal("-0.0001")
    assert third["year_value"] == 1901


@pytest.mark.needs_mysql
def test_a_mysql_zero_date_arrives_as_null() -> None:
    """``0000-00-00`` is not NULL and is not a date either.

    PostgreSQL rejects it outright, so it has to be gone before the target sees
    it. It is gone at the driver already — but only because the connection asks
    for that, which is exactly the kind of thing that stops being true after an
    unrelated change to the URL.
    """
    names, rows = _read_all("mysql", TableRef(name="types_wide"))
    second = _row(names, rows, 2)

    assert second["zero_date"] is None
    assert second["zero_datetime"] is None


@pytest.mark.needs_mysql
def test_mysql_reaches_the_conversion_layer_as_pandas_types() -> None:
    """The production path, not the driver path — and they do not agree.

    A load never sees the driver's objects; it sees whatever pandas made of the
    column, and that is what the conversion layer, the type mapping and `COPY`
    are handed. Reading only through ``exec_driver_sql``, as this file did
    throughout, cannot see any of these three:

    - ``TIME`` becomes ``timedelta64[ns]`` — an interval, still not a clock;
    - ``BIGINT UNSIGNED`` becomes ``uint64``, a dtype whose maximum does not fit
      the ``BIGINT`` the type map sends it to;
    - an integer column with a NULL in it is promoted to ``float64``, which is
      the promotion the conversion layer exists to undo.
    """
    import pandas as pd

    frame = pd.concat(_read_batches("mysql", TableRef(name="types_wide")), ignore_index=True)
    dtypes = dict(frame.dtypes.astype(str))
    row = frame[frame["id"] == 1].iloc[0]

    assert dtypes["long_time"] == "timedelta64[ns]"
    assert dtypes["big_unsigned"] == "uint64"
    assert dtypes["order"] == "float64", "the NULL in row 2 promotes the whole column"
    assert dtypes["zero_datetime"] == "datetime64[ns]"

    assert row["long_time"] == pd.Timedelta(hours=25)
    assert row["negative_time"] == -pd.Timedelta(hours=1, minutes=30)


@pytest.mark.parametrize(
    ("name", "table"),
    [
        pytest.param("mysql", "paged", marks=pytest.mark.needs_mysql, id="mysql"),
        pytest.param("mssql", "paged", marks=pytest.mark.needs_mssql, id="mssql"),
        pytest.param("firebird", "PAGED", marks=pytest.mark.needs_firebird, id="firebird"),
    ],
)
def test_the_production_read_path_really_batches(name: str, table: str) -> None:
    """Five rows at a batch size of two are three batches, not one.

    Batching is what keeps a table of millions of rows out of the client's RAM,
    and until now nothing checked that it happens: the strategy tests use a fake
    dialect whose ``iter_batches`` ignores ``batch_size`` and yields everything
    in one go, so a dialect that quietly stopped batching would pass the suite
    and only show up as a container killed for memory.
    """
    ref = TableRef(name=table, schema="dbo" if name == "mssql" else None)
    batches = _read_batches(name, ref, batch_size=2)

    assert [len(batch) for batch in batches] == [2, 2, 1]


@pytest.mark.needs_mssql
def test_mssql_values_arrive_intact() -> None:
    """Three date/time families with three precisions, and MONEY, which is fixed
    point without being DECIMAL."""
    ref = TableRef(name="types_wide", schema="dbo")
    names, rows = _read_all("mssql", ref)
    first, third = _row(names, rows, 1), _row(names, rows, 3)

    assert first["money_amount"] == Decimal("1234.5678")
    assert first["small_money"] == Decimal("12.34")
    assert first["precise_value"] == Decimal("1234567890.0123456789")
    # DATETIME rounds to 3.33 ms — .997 is what .997 becomes, and it is not a
    # rounding error to be smoothed away.
    assert first["legacy_datetime"] == dt.datetime(2026, 1, 31, 12, 34, 56, 997000)
    assert first["modern_datetime"] == dt.datetime(2026, 1, 31, 12, 34, 56, 123456)
    assert first["day_date"] == dt.date(2026, 1, 31)
    assert first["time_of_day"] == dt.time(12, 34, 56, 123456)
    assert first["flag"] is True
    assert first["cp1250_text"] == "prilis zlutoucky kun"
    # Not a typo and not a broken seed: the connection charset truncates every
    # N-type. Pinned rather than asserted intact — see the section below.
    assert first["unicode_text"] == "p"
    assert str(first["key_uuid"]) == "6f9619ff-8b86-d011-b42d-00c04fc964ff"
    assert first["xml_payload"] == "<r><a>1</a></r>"
    # The computed column is read, not skipped: order * 2.
    assert first["doubled"] == 20

    # DATETIMEOFFSET carries a zone the other two families do not have.
    assert first["zoned_datetime"].utcoffset() == dt.timedelta(hours=2)

    # The extremes of each family: DATETIME starts in 1753, DATETIME2 in year 1.
    assert third["legacy_datetime"] == dt.datetime(1753, 1, 1)
    assert third["modern_datetime"] == dt.datetime(1, 1, 1)
    assert third["day_date"] == dt.date(9999, 12, 31)
    assert third["time_of_day"] == dt.time(23, 59, 59, 999999)
    assert third["precise_value"] == Decimal("-0.0000000001")
    assert third["flag"] is False
    assert third["doubled"] == -10


# --- The MSSQL client charset -------------------------------------------------
#
# ``MSSQLDialect.connect_args`` opens the connection with ``charset='cp1250'``,
# inherited character for character from both predecessor extractors. That
# setting is load-bearing for the legacy single-byte columns and destructive for
# the N-types, and no charset value gets both right — measured, see
# ``docs/legacy-compat.md``. What follows pins the boundary from both sides so
# that neither half can be "improved" without the other half failing loudly.
#
# The order of these tests is the order in which the question has to be asked:
# first that the fixture holds what it claims to (otherwise every finding below
# is a seeding accident), then what reading does to it.

#: The bytes ``docker/seed/mssql/001_types.sql`` writes, as the hex MSSQL prints.
#: Duplicated from the seed on purpose — a fixture that is only ever compared
#: against itself cannot detect that the loader mangled it.
MSSQL_SEEDED_BYTES = {
    # 'příliš žluťoučký kůň' as UTF-16LE, twenty characters, forty bytes.
    "types_wide.unicode_text": (
        "70005901ED006C006900610120007E016C00750065016F0075000D016B00FD0020006B006F014801"
    ),
    # The same twenty characters as single-byte CP1250.
    "unicode_edge.varchar_value": "70F8ED6C699A209E6C759D6F75E86BFD206BF9F2",
}


def _mssql_scalar(sql: str):
    """One value straight from the source, through the package's own engine."""
    require("mssql")
    eng = engine_for("mssql")
    try:
        with eng.connect() as conn:
            return conn.exec_driver_sql(sql).scalar()
    finally:
        eng.dispose()


def _mssql_edge_rows(*, convert_nchar: bool = False) -> dict:
    """``dbo.unicode_edge`` through the **production** path, keyed by its label.

    ``iter_batches`` rather than a driver call: the claim being pinned is about
    what a load delivers, and a driver-level read is one layer below that.
    """
    import pandas as pd

    ref = TableRef(name="unicode_edge", schema="dbo")
    frame = pd.concat(_read_batches("mssql", ref, convert_nchar=convert_nchar), ignore_index=True)
    return {row["label"]: row for _, row in frame.iterrows()}


@pytest.mark.needs_mssql
def test_the_mssql_unicode_fixtures_are_stored_correctly() -> None:
    """The seed is not the suspect. Establish that before blaming the read path.

    Everything below asserts that correctly stored Unicode does **not** come
    back intact. That is only a statement about reading if the bytes in the
    column are known-good first — otherwise it is indistinguishable from a seed
    that was loaded through the wrong code page, which is a defect this
    repository has had before (see ``SET NAMES utf8mb4`` in the MySQL seed).

    The hex comes back as ASCII, so this assertion is itself immune to the
    charset it is investigating.
    """
    stored = _mssql_scalar(
        "SELECT CONVERT(VARCHAR(300), CONVERT(VARBINARY(200), unicode_text), 2) "
        "FROM [dbo].[types_wide] WHERE id = 1"
    )
    assert stored == MSSQL_SEEDED_BYTES["types_wide.unicode_text"]
    assert _mssql_scalar("SELECT LEN(unicode_text) FROM [dbo].[types_wide] WHERE id = 1") == 20

    legacy = _mssql_scalar(
        "SELECT CONVERT(VARCHAR(300), CONVERT(VARBINARY(200), varchar_value), 2) "
        "FROM [dbo].[unicode_edge] WHERE label = 'czech'"
    )
    assert legacy == MSSQL_SEEDED_BYTES["unicode_edge.varchar_value"]


@pytest.mark.needs_mssql
def test_an_mssql_nvarchar_is_cut_at_the_first_character_outside_latin1() -> None:
    """The blast radius, pinned: this is truncation, and it is not total.

    With ``charset='cp1250'`` the driver converts UCS-2 to the single-byte
    charset as **latin-1**, and a character latin-1 cannot express ends the
    string there — silently, with no error and no replacement character. So:

    - pure ASCII arrives whole, which is why most columns look fine;
    - a Czech string starting with 'p' arrives as ``'p'``;
    - a mostly-ASCII string loses only its tail, from the offending character on.

    That is the difference between "every Unicode column is destroyed" and "only
    the values that contain a character outside latin-1 are", and it is worth
    a test of its own because the two are very different sizes of problem.
    """
    rows = _mssql_edge_rows()

    assert rows["ascii"]["nvarchar_value"] == "plain ascii"
    assert rows["czech"]["nvarchar_value"] == "p"
    assert rows["czech_mid"]["nvarchar_value"] == "abcdef-"


@pytest.mark.needs_mssql
def test_every_mssql_n_type_is_cut_the_same_way() -> None:
    """``NCHAR`` and ``NTEXT`` fare exactly as ``NVARCHAR`` does.

    The conversion happens on the wire, so it is a property of the connection
    and not of the column type. Worth pinning explicitly: a limitation written
    up as "NVARCHAR" invites the reading that the other two are safe.
    """
    rows = _mssql_edge_rows()

    assert rows["czech"]["nchar_value"] == "p"
    assert rows["czech"]["ntext_value"] == "p"
    # NCHAR pads to its declared width, and the padding survives what the text
    # did not — further evidence that the cut happens during conversion rather
    # than at the column.
    assert rows["ascii"]["nchar_value"] == "plain ascii".ljust(20)


@pytest.mark.needs_mssql
def test_an_mssql_nvarchar_character_inside_latin1_arrives_as_a_different_one() -> None:
    """U+00F8 'ø' comes back as 'ř', and that is what identifies the conversion.

    'ø' is in latin-1 and not in CP1250; 'ř' is in CP1250 and not in latin-1;
    both are byte ``0xF8`` in their own code page. The value arriving whole and
    wrong — rather than cut short — is what proves the driver converts UCS-2 to
    latin-1 while the client decodes the result as CP1250. Every other result on
    this page follows from that one mismatch.
    """
    rows = _mssql_edge_rows()

    assert rows["latin1_only"]["nvarchar_value"] == "a-ř-b"


@pytest.mark.needs_mssql
def test_an_mssql_legacy_varchar_is_why_the_charset_is_cp1250() -> None:
    """The other half of the trade, and the reason the setting cannot just be dropped.

    A ``VARCHAR`` under the server's CP1250 collation holds single-byte text,
    and it arrives **correct** exactly because the connection asks for CP1250.
    Set the connection to UTF-8 and this value comes back as
    ``'pøíli\\x9a \\x9elu\\x9douèký kùò'`` — measured, not assumed.

    So the fix that suggests itself for the N-types regresses the columns that
    are the majority of a CP1250-collated source. That is the whole argument for
    pinning rather than changing; see ``docs/legacy-compat.md``.
    """
    rows = _mssql_edge_rows()

    assert rows["czech"]["varchar_value"] == "příliš žluťoučký kůň"
    assert rows["ascii"]["varchar_value"] == "plain ascii"


# --- convert_nchar_to_varchar, the way out ------------------------------------
#
# Everything above is the inherited behaviour, and it stays the default. What
# follows is the opt-in: with `LOAD_SETTINGS.convert_nchar_to_varchar` the server
# converts the N-types through their own collation before they reach the wire, so
# they arrive over the same cp1250 connection whole.
#
# These tests are the reason the key can be trusted at all — the conversion
# happens inside MSSQL, so no amount of reasoning about pymssql settles what
# comes back. Each of them has a counterpart above with the key off; the pair is
# the point.


@pytest.mark.needs_mssql
def test_the_nchar_conversion_brings_the_whole_czech_value_back() -> None:
    """``'p'`` becomes ``'příliš žluťoučký kůň'`` — the whole point of the key.

    Read through ``iter_batches``, i.e. what a load actually receives, not
    through a driver call one layer below it.
    """
    rows = _mssql_edge_rows(convert_nchar=True)

    assert rows["czech"]["nvarchar_value"] == "příliš žluťoučký kůň"
    assert rows["czech_mid"]["nvarchar_value"] == "abcdef-ř-ghijkl"
    assert rows["ascii"]["nvarchar_value"] == "plain ascii"


@pytest.mark.needs_mssql
def test_the_conversion_covers_every_n_type() -> None:
    """``NCHAR`` and ``NTEXT`` too, not only ``NVARCHAR``.

    ``NCHAR`` keeps padding to its declared width, exactly as it does without the
    key: the conversion changes the encoding of the value, not the column.
    """
    rows = _mssql_edge_rows(convert_nchar=True)

    assert rows["czech"]["nchar_value"] == "příliš žluťoučký kůň"
    assert rows["czech"]["ntext_value"] == "příliš žluťoučký kůň"
    assert rows["ascii"]["nchar_value"] == "plain ascii".ljust(20)


@pytest.mark.needs_mssql
def test_the_legacy_varchar_is_the_same_with_the_key_on_and_off() -> None:
    """The half that must not move, and the one worth a test of its own.

    The single-byte columns are the majority of a CP1250-collated source and the
    reason the connection charset cannot simply be changed. If the key touched
    them it would be the same trade in the other direction, so this asserts them
    identical on both sides rather than merely correct on one.
    """
    off = _mssql_edge_rows()
    on = _mssql_edge_rows(convert_nchar=True)

    for label in ("ascii", "czech"):
        assert on[label]["varchar_value"] == off[label]["varchar_value"]
    assert on["czech"]["varchar_value"] == "příliš žluťoučký kůň"
    # And the non-text columns are untouched as well — the wrap is driven by the
    # introspected type, so a number must not have been dragged through it.
    assert list(on["czech"].index) == list(off["czech"].index)
    assert on["czech"]["id"] == off["czech"]["id"]


@pytest.mark.needs_mssql
def test_what_cp1250_cannot_express_is_degraded_and_not_truncated() -> None:
    """The cost of the key, pinned so that nobody discovers it in production.

    The conversion is to a single-byte charset, so a character CP1250 has no room
    for cannot survive. What it does instead is the whole difference from the
    default: the rest of the string stays.

    Two different degradations, and both are the collation's decision rather than
    ours: U+0416 (Cyrillic Ж) has no CP1250 equivalent at all and becomes ``?``,
    while U+00F8 ``ø`` is folded onto its base letter ``o``. Without the key both
    values are cut instead — ``'a-ř-b'`` is what the same ``ø`` row looks like
    then, one character long in the Czech case.
    """
    on = _mssql_edge_rows(convert_nchar=True)
    off = _mssql_edge_rows()

    assert on["outside_cp1250"]["nvarchar_value"] == "a-?-b"
    assert on["latin1_only"]["nvarchar_value"] == "a-o-b"

    # The default, for contrast: same rows, same connection, no conversion.
    assert off["latin1_only"]["nvarchar_value"] == "a-ř-b"
    assert off["czech"]["nvarchar_value"] == "p"


@pytest.mark.needs_firebird
def test_firebird_values_arrive_intact() -> None:
    """Negative NUMERIC scale, CHAR padded with spaces, and days counted from
    1858 — all three change the value rather than failing."""
    names, rows = _read_all("firebird", TableRef(name="TYPES_WIDE"))
    first, third = _row(names, rows, 1), _row(names, rows, 3)

    # The scale the driver reports negated. Reading it with the wrong sign does
    # not fail; it moves the decimal point, and 1234.5678 becomes 12345678.
    assert first["MONEY_AMOUNT"] == Decimal("1234.5678")
    assert first["SMALL_SCALE"] == Decimal("12.34")
    assert first["NO_SCALE"] == Decimal("1234")
    assert first["DECIMAL_VALUE"] == Decimal("123456.789012")

    assert first["DAY_DATE"] == dt.date(2026, 1, 31)
    assert first["STAMP"] == dt.datetime(2026, 1, 31, 12, 34, 56, 123000)
    assert first["TIME_OF_DAY"] == dt.time(12, 34, 56, 123000)

    # CHAR pads to its declared width, VARCHAR does not. The padding survives
    # into the target unless something trims it, so it has to be visible here.
    assert first["PADDED"] == "padded    "
    assert first["NOT_PADDED"] == "nopad"
    assert first["FLAG_CHAR"] == "Y"
    assert first["CZECH_TEXT"] == "prilis zlutoucky kun"
    assert first["BLOB_TEXT"] == "a text blob"
    assert first["BIG_VALUE"] == 9223372036854775807
    assert first["FLOAT_VALUE"] == 1.7976931348623157e308

    # Firebird's epoch, seeded on purpose: a day count read against the Unix
    # epoch would come back as 1970 and nothing would fail.
    assert third["DAY_DATE"] == dt.date(1858, 11, 17)
    assert third["STAMP"] == dt.datetime(1858, 11, 17, 0, 0)
    assert third["MONEY_AMOUNT"] == Decimal("-0.0001")
    assert third["NO_SCALE"] == Decimal("-1")
    assert third["BIG_VALUE"] == -9223372036854775808
    assert third["PADDED"] == " " * 10
    assert third["NOT_PADDED"] == ""


@pytest.mark.needs_mysql
def test_the_session_timeouts_reach_the_live_server() -> None:
    """The mitigation for the two extractions that died as "2013 Lost connection".

    Everything else about `session_sql` can be checked without a server; this
    cannot. The statement has to be accepted by the server, and it has to be
    applied to the connection the reading actually happens on — a listener
    registered on the wrong event, or SQL a newer MySQL rejects, both look
    exactly like a working setup until a long extraction dies again.
    """
    from dbextractors import entrypoint

    require("mysql")
    dialect = resolve_dialect("mysql")
    eng = engine_for("mysql")
    try:
        with eng.connect() as conn:
            before = conn.exec_driver_sql("SELECT @@SESSION.net_write_timeout").scalar()

        entrypoint._attach_session_sql(eng, dialect)
        with eng.connect() as conn:
            after = conn.exec_driver_sql(
                "SELECT @@SESSION.net_write_timeout, @@SESSION.net_read_timeout"
            ).fetchone()
    finally:
        eng.dispose()

    # Check the premise: a server that already defaulted to 600 would make the
    # assertion below true without anything having been applied.
    assert int(before) != 600, "the server default is already 600 — this proves nothing"
    assert tuple(int(v) for v in after) == (600, 600)


# --- size estimate ------------------------------------------------------------


@pytest.mark.needs_mysql
def test_the_mysql_size_estimate_counts_the_live_table() -> None:
    """`estimate_size` against a real server, not a stand-in.

    The unit tests run the same code over SQLite, which cannot show that
    ``COUNT(*)`` over the dialect's own quoting reaches the right table on the
    right server — the database is in the URL, not in the query.
    """
    require("mysql")
    dialect = resolve_dialect("mysql")
    eng = engine_for("mysql")
    try:
        estimate = dialect.estimate_size(eng, TableRef(name="paged"))
        narrowed = dialect.estimate_size(eng, TableRef(name="paged"), where="id > 3")
    finally:
        eng.dispose()

    assert estimate.rows == 5
    assert estimate.method == "count_and_sample"
    assert narrowed.rows == 2


@pytest.mark.needs_mysql
def test_a_live_estimate_of_a_table_that_is_not_there_fails_loudly() -> None:
    """**Rule 5**, against the server that would really answer.

    A typo in ``source_name`` must not come back as an empty table: with
    ``empty_rows_ok`` that truncates the target and the run reports success.
    """
    require("mysql")
    dialect = resolve_dialect("mysql")
    eng = engine_for("mysql")
    try:
        with pytest.raises(Exception, match="(?i)doesn't exist|does not exist"):
            dialect.estimate_size(eng, TableRef(name="no_such_table"))
    finally:
        eng.dispose()


# --- DSN parsing -------------------------------------------------------------


def test_dsn_parsing_keeps_a_password_with_spaces_intact() -> None:
    """The libpq spelling is used precisely so a path or an odd password works.

    Not a live test, but it belongs next to the ones that depend on it: when
    this is wrong, every source test above fails at connect time and the reason
    is nowhere near the DSN.
    """
    parsed = parse_dsn("host=127.0.0.1 port=53050 database=/firebird/data/x.fdb user=SYSDBA")

    assert parsed["database"] == "/firebird/data/x.fdb"
    assert parsed["port"] == 53050
    assert isinstance(parsed["port"], int)


def test_dsn_parsing_rejects_a_token_that_is_not_a_pair() -> None:
    with pytest.raises(ValueError, match="not key=value"):
        parse_dsn("host=127.0.0.1 nonsense")
