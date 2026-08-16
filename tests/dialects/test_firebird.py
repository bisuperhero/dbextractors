"""Tests for `FirebirdDialect` — mainly the parts that can be compared verbatim
with the predecessor.

Of the four dialects Firebird is the most distant one: types are numbers rather
than names, paging goes through ``FIRST``/``SKIP`` inside the SELECT list,
introspection goes through the ``RDB$`` tables, and a date in the incremental
window is a count of days since 1899. Each of those things only shows up in a run
against the source — and no live Firebird was available while this was written,
so the tests here guard it in that much more detail.

``map_type`` deserves special attention: it is decided by the **sign** of
``scale``, so taking the wrong branch does not fail, it merely shifts the decimal
point in silence.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, List

import pytest

from dbextractors.dialects.base import SurrogateKey, TableRef, UnsupportedFeature
from dbextractors.dialects.firebird import (
    DEFAULT_CHARSET,
    FB_TYPE_NAMES,
    FIREBIRD_EPOCH,
    FirebirdDialect,
    fb_type_to_pg,
)


@pytest.fixture()
def dialect() -> FirebirdDialect:
    return FirebirdDialect()


# --- Fake engine ------------------------------------------------------------


class _Result:
    def __init__(self, rows: List[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> List[tuple]:
        return self._rows

    def scalar(self) -> Any:
        return self._rows[0][0] if self._rows else None


class _Connection:
    def __init__(self, answers: List[Any]) -> None:
        self._answers = list(answers)
        self.queries: List[tuple] = []

    def execute(self, query, params=None):
        self.queries.append((str(query), params))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Result(answer)


class _Engine:
    """A minimal stand-in for a SQLAlchemy engine — just ``connect()`` as a context."""

    def __init__(self, *answers: Any) -> None:
        self.connection = _Connection(list(answers))

    def connect(self):
        engine = self

        class _Ctx:
            def __enter__(self):
                return engine.connection

            def __exit__(self, *args):
                return False

        return _Ctx()


# --- Type mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    ("fb_type", "fb_subtype", "fb_scale", "expected"),
    [
        (7, 0, 0, "INTEGER"),
        (7, 0, -2, "NUMERIC"),
        (8, 0, 0, "INTEGER"),
        (8, 0, -4, "NUMERIC"),
        (16, 0, 0, "BIGINT"),
        (16, 0, -2, "NUMERIC"),
        (16, 1, 0, "NUMERIC"),
        (16, 2, 0, "NUMERIC"),
        (10, 0, 0, "REAL"),
        (27, 0, 0, "DOUBLE PRECISION"),
        (14, 0, 0, "TEXT"),
        (37, 0, 0, "VARCHAR"),
        (261, 1, 0, "TEXT"),
        (261, 0, 0, "BYTEA"),
        (35, 0, 0, "TIMESTAMP"),
        (12, 0, 0, "DATE"),
        (13, 0, 0, "TIME"),
        (23, 0, 0, "BOOLEAN"),
    ],
)
def test_the_type_map_is_verbatim_from_the_predecessor(
    fb_type, fb_subtype, fb_scale, expected
) -> None:
    """``_fb_type_to_pg`` as the predecessor's Firebird extractor writes it.

    All eighteen branches, including both where ``scale`` decides and both where
    ``subtype`` does.
    """
    assert fb_type_to_pg(fb_type, fb_subtype, fb_scale) == expected


def test_a_negative_scale_means_a_decimal_number() -> None:
    """The trickiest branch of the whole dialect.

    ``NUMERIC(15,2)`` is stored in Firebird as an ``INT64`` with ``scale=-2``.
    Were the scale ignored, the target would end up with ``BIGINT`` and the
    amount 1234.56 would be stored as 123456 — a silent shift by orders of
    magnitude, not a failure.
    """
    assert fb_type_to_pg(16, 0, -2) == "NUMERIC"
    assert fb_type_to_pg(16, 0, 0) == "BIGINT"


def test_an_unknown_code_falls_back_to_text() -> None:
    """FB 4 added codes (24/25 DECFLOAT, 26 INT128) the predecessor does not know."""
    assert fb_type_to_pg(26, 0, 0) == "TEXT"
    assert fb_type_to_pg(None, None, None) == "TEXT"


def test_map_type_reads_from_meta(dialect) -> None:
    from dbextractors.dialects.base import ColumnDef

    column = ColumnDef(
        name="PRICE",
        source_type="bigint",
        pg_type="",
        meta={"fb_type": 16, "fb_subtype": 0, "fb_scale": -2},
    )
    assert dialect.map_type(column) == "NUMERIC"


def test_map_type_without_meta_is_text(dialect) -> None:
    """A column that did not go through Firebird introspection has nothing to type from."""
    from dbextractors.dialects.base import ColumnDef

    assert dialect.map_type(ColumnDef(name="X", source_type="varchar", pg_type="")) == "TEXT"


# --- Type names against the rest of the package -----------------------------


def test_coerce_knows_the_date_names() -> None:
    """``source_type`` has to be a name `coerce` actually recognises.

    This is the whole reason the code is translated into a name: were a number
    left in ``source_type``, `extract_date_columns` would find **no** date column
    at all on Firebird and `fix_invalid_dates` would never run.
    """
    from dbextractors.core.coerce import DATE_LIKE_TYPES

    assert FB_TYPE_NAMES[12] in DATE_LIKE_TYPES
    assert FB_TYPE_NAMES[35] in DATE_LIKE_TYPES


def test_coerce_knows_time() -> None:
    from dbextractors.core import coerce

    assert FB_TYPE_NAMES[13] in coerce._TIME_TYPES


def test_the_type_names_are_lower_case() -> None:
    """The comparison goes through ``.lower()``, so a capital letter would never match."""
    assert all(name == name.lower() for name in FB_TYPE_NAMES.values())


# --- Introspection ----------------------------------------------------------


def test_introspection_looks_the_table_up_in_upper_case(dialect) -> None:
    """Firebird stores unquoted identifiers as ``UPPERCASE``."""
    engine = _Engine([("ID", 8, 0, 0)])
    dialect.introspect_columns(engine, TableRef(name="invoices"))
    assert engine.connection.queries[0][1] == {"tbl": "INVOICES"}


def test_introspection_returns_the_order_and_the_meta(dialect) -> None:
    engine = _Engine([("ID", 8, 0, 0), ("PRICE", 16, 0, -2), ("NOTE", 261, 1, 0)])
    columns = dialect.introspect_columns(engine, TableRef(name="FA"))

    assert [c.name for c in columns] == ["ID", "PRICE", "NOTE"]
    assert [c.ordinal for c in columns] == [1, 2, 3]
    assert [c.pg_type for c in columns] == ["INTEGER", "NUMERIC", "TEXT"]
    assert [c.source_type for c in columns] == ["integer", "bigint", "blob"]
    assert columns[1].meta == {"fb_type": 16, "fb_subtype": 0, "fb_scale": -2}


def test_introspection_trims_spaces_in_the_name(dialect) -> None:
    """``RDB$FIELD_NAME`` is a ``CHAR(31)``, so it arrives padded with spaces."""
    engine = _Engine([("ID   ", 8, 0, 0)])
    assert dialect.introspect_columns(engine, TableRef(name="T"))[0].name == "ID"


def test_introspection_leaves_target_name_empty(dialect) -> None:
    """Target names are filled in by `core.naming`, not by the dialect — see the
    column naming section of ``docs/legacy-compat.md``."""
    engine = _Engine([("TYPE", 37, 0, 0)])
    assert dialect.introspect_columns(engine, TableRef(name="T"))[0].target_name is None


def test_a_missing_table_is_an_error(dialect) -> None:
    """The predecessor returns "0 rows, data_present=False" and the run comes out green.

    That is exactly the silent path the error-handling rules forbid — an empty
    result caused by a typo in ``source_name`` must not pass for an empty table.
    """
    engine = _Engine([])
    with pytest.raises(RuntimeError, match="does not exist"):
        dialect.introspect_columns(engine, TableRef(name="nothing"))


def test_the_error_suggests_upper_case(dialect) -> None:
    engine = _Engine([])
    with pytest.raises(RuntimeError, match="INVOICES"):
        dialect.introspect_columns(engine, TableRef(name="invoices"))


def test_blob_columns_are_reported(dialect, caplog) -> None:
    engine = _Engine([("ID", 8, 0, 0), ("TEXT_", 261, 1, 0), ("DATA", 261, 0, 0)])
    with caplog.at_level("WARNING"):
        dialect.introspect_columns(engine, TableRef(name="T"))
    assert "BLOB" in caplog.text
    assert "DATA" in caplog.text and "TEXT_" in caplog.text


def test_without_blobs_nothing_is_reported(dialect, caplog) -> None:
    engine = _Engine([("ID", 8, 0, 0)])
    with caplog.at_level("WARNING"):
        dialect.introspect_columns(engine, TableRef(name="T"))
    assert "BLOB" not in caplog.text


# --- Size estimate ----------------------------------------------------------


def test_the_estimate_comes_from_metadata_not_from_a_sample(dialect) -> None:
    """Firebird is the only one that reads no data at all — it sums the declared lengths."""
    engine = _Engine([(1000,)], [(512,)])
    estimate = dialect.estimate_size(engine, TableRef(name="T"))

    assert estimate.rows == 1000
    assert estimate.method == "count_and_schema"
    assert estimate.size_mb == round(512 * 1000 / 1024 / 1024, 2)
    assert "SELECT *" not in " ".join(q[0] for q in engine.connection.queries)


def test_without_metadata_the_estimate_uses_256(dialect) -> None:
    """A relation with no readable lengths (typically a view) — the predecessor's fallback."""
    engine = _Engine([(10,)], [(None,)])
    assert dialect.estimate_size(engine, TableRef(name="V")).size_mb == round(
        256 * 10 / 1024 / 1024, 2
    )


def test_an_empty_table_is_not_asked_about_its_size(dialect) -> None:
    engine = _Engine([(0,)])
    estimate = dialect.estimate_size(engine, TableRef(name="T"))
    assert estimate.rows == 0 and estimate.size_mb == 0.0
    assert len(engine.connection.queries) == 1


def test_a_known_row_count_saves_the_count(dialect) -> None:
    engine = _Engine([(512,)])
    estimate = dialect.estimate_size(engine, TableRef(name="T"), known_total_rows=7)
    assert estimate.rows == 7
    assert estimate.method == "known_total_rows"
    assert "COUNT" not in engine.connection.queries[0][0].upper()


def test_where_makes_it_into_the_count(dialect) -> None:
    engine = _Engine([(5,)], [(8,)])
    dialect.estimate_size(engine, TableRef(name="T"), where="A = 1")
    assert "WHERE (A = 1)" in engine.connection.queries[0][0]


def test_a_failed_count_fails_loudly(dialect, monkeypatch) -> None:
    """Never ``rows=0`` when asking for the count did not succeed."""
    monkeypatch.setattr("dbextractors.dialects.firebird.time.sleep", lambda _s: None)
    engine = _Engine(RuntimeError("failed"), RuntimeError("failed"), RuntimeError("failed"))
    with pytest.raises(RuntimeError, match="failed"):
        dialect.estimate_size(engine, TableRef(name="T"))


# --- Retrying ---------------------------------------------------------------


def test_a_retry_succeeds_on_the_second_attempt(dialect, monkeypatch) -> None:
    monkeypatch.setattr("dbextractors.dialects.firebird.time.sleep", lambda _s: None)
    engine = _Engine(RuntimeError("reset"), [("ID", 8, 0, 0)])
    assert len(dialect.introspect_columns(engine, TableRef(name="T"))) == 1


def test_there_are_three_attempts(dialect, monkeypatch) -> None:
    naps: List[float] = []
    monkeypatch.setattr("dbextractors.dialects.firebird.time.sleep", naps.append)
    engine = _Engine(*[RuntimeError("reset")] * 3)
    with pytest.raises(RuntimeError):
        dialect.introspect_columns(engine, TableRef(name="T"))
    assert len(naps) == 2  # after the last attempt there is no sleeping


def test_the_delay_grows(dialect, monkeypatch) -> None:
    """Exponentially, with jitter — the orchestrator runs three pipelines at once."""
    naps: List[float] = []
    monkeypatch.setattr("dbextractors.dialects.firebird.time.sleep", naps.append)
    engine = _Engine(*[RuntimeError("reset")] * 3)
    with pytest.raises(RuntimeError):
        dialect.introspect_columns(engine, TableRef(name="T"))
    assert 1.5 <= naps[0] <= 2.5
    assert 3.0 <= naps[1] <= 5.0


# --- Connecting -------------------------------------------------------------


def test_the_password_is_escaped(dialect) -> None:
    url = dialect.build_conn_str(
        {"user": "u", "password": "a@b/c", "database": "/var/db/x.fdb"}, "h", 3050
    )
    assert "a%40b%2Fc" in url
    assert "a@b/c" not in url


def test_an_absolute_path_gives_a_double_slash(dialect) -> None:
    """``fdb`` recognises an absolute path precisely by the double slash.

    Escaping the path would turn it into ``%2Fvar%2Fdb…`` and the database would
    not be found — which is why it is the only part of the URL left as it is.
    """
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "/var/db/x.fdb"}, "h", 3050
    )
    assert url.startswith("firebird+fdb://u:p@h:3050//var/db/x.fdb?")


def test_doubled_backslashes_in_the_path_are_flattened(dialect) -> None:
    """Passing the config between Mage blocks occasionally doubles a Windows path."""
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "D:\\\\db_data\\\\x.fdb"}, "h", 3050
    )
    assert "D:\\db_data\\x.fdb" in url


def test_single_backslashes_are_left_alone(dialect) -> None:
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "D:\\db\\x.fdb"}, "h", 3050
    )
    assert "D:\\db\\x.fdb" in url


def test_charset_from_the_configuration_is_used(dialect) -> None:
    """Unlike MSSQL, this key **is** read — the predecessor puts it into the URL."""
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "x.fdb", "charset": "UTF8"}, "h", 3050
    )
    assert url.endswith("?charset=UTF8")


def test_the_default_charset_is_win1250(dialect) -> None:
    url = dialect.build_conn_str({"user": "u", "password": "p", "database": "x.fdb"}, "h", 3050)
    assert url.endswith(f"?charset={DEFAULT_CHARSET}")
    assert DEFAULT_CHARSET == "WIN1250"


def test_a_probe_against_a_closed_port_is_false(dialect) -> None:
    assert dialect.probe("127.0.0.1", 1, timeout=0.2) is False


def test_connect_args_are_empty(dialect) -> None:
    """The encoding travels in the URL, not in ``connect_args`` — unlike MSSQL."""
    assert dialect.connect_args == {}


# --- SQL generation ---------------------------------------------------------


def test_quotes_with_double_quotes(dialect) -> None:
    assert dialect.quote_ident("ID") == '"ID"'


def test_select_is_not_qualified_by_a_schema(dialect) -> None:
    """Firebird has no schemas and the predecessor never qualifies the table."""
    sql = dialect.render_select(["ID"], TableRef(name="FA", schema="nonsense"))
    assert sql == 'SELECT "ID" FROM "FA"'


def test_select_with_a_surrogate_key(dialect) -> None:
    sql = dialect.render_select(
        ["ID", "A"],
        TableRef(name="T"),
        surrogate=SurrogateKey(enabled=True, alias="ID", expr="A || B"),
    )
    assert sql == 'SELECT A || B AS "ID", "A" FROM "T"'


def test_select_with_where_and_ordering(dialect) -> None:
    sql = dialect.render_select(["A"], TableRef(name="T"), where="A > 1", order_by=["A"])
    assert sql == 'SELECT "A" FROM "T" WHERE (A > 1) ORDER BY "A"'


def test_there_is_no_hash_on_the_source_side(dialect) -> None:
    """Firebird has no ``build_hash_expr`` variant among the predecessors at all."""
    with pytest.raises(UnsupportedFeature):
        dialect.render_hash_expr(["A"], "row_hash")


def test_pandas_computes_the_hash(dialect) -> None:
    """When the source cannot do it, pandas has to — otherwise the column would be missing."""
    assert dialect.hash_in_pandas is True


# --- Cutoff -----------------------------------------------------------------


def test_the_cutoff_is_a_count_of_days_since_the_epoch(dialect) -> None:
    """The date columns of this source are not ``DATE`` but a count of days since
    30 December 1899."""
    assert dialect.cutoff_value(date(2025, 1, 1)) == 45658
    assert dialect.cutoff_value(date(2024, 7, 15)) == 45488


def test_the_cutoff_from_a_datetime(dialect) -> None:
    assert dialect.cutoff_value(datetime(2025, 1, 1, 13, 45)) == 45658


def test_the_epoch_is_the_one_from_the_predecessor() -> None:
    assert date(1899, 12, 30) == FIREBIRD_EPOCH


def test_the_cutoff_goes_into_sql_as_a_number(dialect) -> None:
    """It has to come out as ``12345``, not ``'12345'`` — the column is numeric."""
    assert dialect.sql_literal(dialect.cutoff_value(date(2025, 1, 1))) == "45658"


def test_the_other_dialects_do_not_compute_the_date() -> None:
    """The counterpart: the conversion is knowledge about Firebird, not about the strategy."""
    from dbextractors.dialects.mysql import MySQLDialect

    assert MySQLDialect().cutoff_value(date(2025, 1, 1)) == "2025-01-01"


# --- Capabilities -----------------------------------------------------------


def test_it_can_do_keyset_and_parent_incremental(dialect) -> None:
    """``parent_incremental`` exists today **only** on Firebird (24 tables)."""
    assert dialect.supports("keyset")
    assert dialect.supports("parent_incremental")


def test_it_can_do_neither_hash_diff_nor_partitioning(dialect) -> None:
    assert not dialect.supports("hash_diff")
    assert not dialect.supports("partition_by_source")


def test_require_explains_what_is_supported(dialect) -> None:
    with pytest.raises(UnsupportedFeature, match="hash_diff"):
        dialect.require("hash_diff")


def test_it_is_the_only_one_with_parent_incremental() -> None:
    from dbextractors.dialects.mssql import MSSQLDialect
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    others = [MySQLDialect(), MSSQLDialect(), PostgresDialect()]
    assert not any(d.supports("parent_incremental") for d in others)


def test_the_dialect_registry_knows_it() -> None:
    from dbextractors.entrypoint import resolve_dialect

    assert isinstance(resolve_dialect("firebird"), FirebirdDialect)


# --- Decoding ---------------------------------------------------------------


def test_the_decoding_cascade_starts_with_cp1250(dialect) -> None:
    assert dialect.decode_encodings == ("cp1250", "utf-8", "latin-1")


def test_the_textual_types_are_empty_on_purpose(dialect) -> None:
    """The Firebird predecessor has no "read as text" list at all.

    Adding one would change values against what is there today, and that is
    exactly what the golden test guards. The counterpart: MySQL has one, because
    its predecessor has one too.
    """
    from dbextractors.dialects.mysql import MySQLDialect

    assert dialect.text_like_types == frozenset()
    assert MySQLDialect().text_like_types


# --- Going through the strategies -------------------------------------------
#
# Without a live Firebird the golden test cannot be run, but the riskiest part of
# how the dialect meets the strategies can still be tested: on a dialect that
# cannot hash, `full` must not ask for a hash expression at all. Were it to ask,
# every Firebird run would fail on ``UnsupportedFeature`` — and only in
# production.


def _ctx(dialect, settings=None):
    from fakes import make_columns, make_context

    return make_context(
        None,
        "s",
        dialect=dialect,
        columns=make_columns([("ID", "integer", "INTEGER"), ("DESCR", "varchar", "VARCHAR")]),
        settings=dict(settings or {}),
    )


def test_full_load_does_not_ask_for_a_hash_expression(dialect) -> None:
    """``hash_in_pandas`` has to divert `full` before it reaches ``render_hash_expr``."""
    from dbextractors.core.strategies.full import FullLoadStrategy

    sql = FullLoadStrategy()._build_select(_ctx(dialect), "row_hash")
    assert sql == 'SELECT "ID", "DESCR" FROM "zdroj"'


def test_the_hash_is_computed_in_pandas(dialect) -> None:
    """The counterpart of the previous test: the column in the target **is** there,
    it is just pandas that fills it."""
    from dbextractors.core.strategies.full import _pandas_hash_columns

    assert _pandas_hash_columns(_ctx(dialect), "row_hash") == ["id", "descr"]


def test_parent_incremental_compares_with_a_number(dialect) -> None:
    """The condition on the parent has to be ``"CHANGED_ON" >= 45658``, not a quoted date.

    The column is numeric in the source, so a text literal would either be
    rejected by Firebird or (worse) compared as a string.
    """
    from dbextractors.core.strategies.parent_incremental import ParentIncrementalStrategy

    condition = ParentIncrementalStrategy()._parent_condition(
        _ctx(dialect), date(2025, 1, 1), ["CHANGED_ON"]
    )
    assert condition == '("CHANGED_ON" >= 45658)'


def test_parent_incremental_joins_the_columns_with_or(dialect) -> None:
    """In some tables the change date is filled in in only one of the two columns."""
    from dbextractors.core.strategies.parent_incremental import ParentIncrementalStrategy

    condition = ParentIncrementalStrategy()._parent_condition(
        _ctx(dialect), date(2025, 1, 1), ["CHANGED_ON", "CHANGED_AT"]
    )
    assert condition == '("CHANGED_ON" >= 45658 OR "CHANGED_AT" >= 45658)'


# --- Letter case coming from the driver -------------------------------------
#
# SQLAlchemy sets `requires_name_normalize` for Firebird, so a name stored in the
# database in upper case reaches the caller in lower case. Introspection through
# the ``RDB$`` tables returns upper case, however — and `ctx.name_map` rests on
# those.


class _NormalisingDialect:
    """Imitates a SQLAlchemy dialect that normalises names."""

    @staticmethod
    def denormalize_name(name):
        return name.upper() if isinstance(name, str) and name.islower() else name


class _EngineWithDialect:
    def __init__(self, dialect=None):
        self.dialect = dialect


def test_denormalisation_restores_upper_case_from_the_dialect() -> None:
    import pandas as pd

    from dbextractors.dialects.firebird import FirebirdDialect

    batch = pd.DataFrame({"id": [1], "name": ["Example"], "address_id": [7]})

    result = FirebirdDialect._denormalize_columns(batch, _EngineWithDialect(_NormalisingDialect()))

    assert list(result.columns) == ["ID", "NAME", "ADDRESS_ID"]


def test_denormalisation_works_without_a_dialect_too() -> None:
    """The fallback rule, in case the engine has no `denormalize_name`."""
    import pandas as pd

    from dbextractors.dialects.firebird import FirebirdDialect

    batch = pd.DataFrame({"id": [1], "note": ["x"]})

    result = FirebirdDialect._denormalize_columns(batch, _EngineWithDialect(None))

    assert list(result.columns) == ["ID", "NOTE"]


def test_denormalisation_leaves_mixed_case_names_alone() -> None:
    """A column created in quotes as ``"mixedCase"`` comes back exactly as it is."""
    import pandas as pd

    from dbextractors.dialects.firebird import FirebirdDialect

    batch = pd.DataFrame({"mixedCase": [1], "UPPER": [2], "lower": [3]})

    result = FirebirdDialect._denormalize_columns(batch, _EngineWithDialect(None))

    assert list(result.columns) == ["mixedCase", "UPPER", "LOWER"]


def test_denormalisation_lines_up_with_the_name_map() -> None:
    """The real reason this exists: without lining them up, `name_map` misses.

    One source table has a column ``NAME``, which in the target is ``_name`` (a
    reserved word). The driver, however, returns ``name``, so the rename is
    silently skipped and the failure comes two steps later as
    ``KeyError: "['_name'] not in index"``.
    """
    import pandas as pd

    from dbextractors.core import naming
    from dbextractors.dialects.firebird import FirebirdDialect

    from_source = ["ID", "NAME", "LOGINNAME"]
    name_map = dict(zip(from_source, naming.target_column_names(from_source, None), strict=False))
    assert name_map["NAME"] == "_name"

    from_driver = pd.DataFrame({"id": [1], "name": ["x"], "loginname": ["y"]})
    lined_up = FirebirdDialect._denormalize_columns(from_driver, _EngineWithDialect(None))

    assert list(lined_up.rename(columns=name_map).columns) == ["id", "_name", "loginname"]


# --- source_ident -----------------------------------------------------------


def test_source_ident_turns_a_lower_case_name_into_upper_case(dialect) -> None:
    """A name written in lower case in the configuration has to be flipped to upper case.

    Firebird keeps unquoted identifiers in upper case, so `"id"` is a different —
    and non-existent — column from `"ID"`. Seen in production: ``SELECT "id" FROM
    "DOCUMENTS"`` ended with ``SQLCODE -206, Column unknown - id``.
    """
    assert dialect.source_ident("id") == '"ID"'
    assert dialect.source_ident("parent_id") == '"PARENT_ID"'
    assert dialect.source_ident(" documents ") == '"DOCUMENTS"'


def test_source_ident_leaves_mixed_case_alone(dialect) -> None:
    """A column created in quotes as ``"mixedCase"`` must not be rewritten — in
    Firebird it is a valid name and upper-casing would break it. ``denormalize_name``
    in SQLAlchemy follows the same rule."""
    assert dialect.source_ident("mixedCase") == '"mixedCase"'
    assert dialect.source_ident("ID") == '"ID"'


def test_source_ident_equals_quote_ident_in_the_other_dialects() -> None:
    """Dialects that do not flip identifiers must be untouched by the change."""
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    for d in (MySQLDialect(), PostgresDialect()):
        assert d.source_ident("id") == d.quote_ident("id")
        assert d.source_ident("parent_id") == d.quote_ident("parent_id")


# --- 64-bit integers against the live source --------------------------------


@pytest.mark.needs_firebird
def test_a_bigint_column_with_a_null_arrives_from_the_source_already_rounded() -> None:
    """A pinning test for a known, deliberate limitation. It is not a wish list.

    ``TYPES_WIDE.BIG_VALUE`` holds the maximum signed 64-bit integer in row 1 and
    a NULL in row 2. Because of the NULL the column cannot be ``int64``, so
    ``pd.read_sql`` infers ``float64`` — 53 bits of mantissa — and the value is
    rounded before a single line of this package touches it.

    Verified end to end into the live PostgreSQL target as well: a Firebird
    ``BIGINT`` of ``9007199254740993`` lands in the target as
    ``9007199254740992``, with the right row count and a green run.

    **Why it is not fixed here.** The predecessor reads exactly the same way —
    ``pd.read_sql(text(sql), engine, params=params)`` at
    ``reference/variant-b/a_firebird_streaming_extractor.py:957``, no ``dtype=``, no
    ``coerce_float`` — so correcting the read would put a different value in the
    target than the old side and break golden parity. The reasoning and the rest
    of the blast radius are in ``tests/coerce/test_int64_precision.py``.
    """
    import numpy as np
    from source_db import engine_for

    engine = engine_for("firebird")
    dialect = FirebirdDialect()
    sql = 'SELECT "ID", "BIG_VALUE" FROM "TYPES_WIDE" ORDER BY "ID"'
    batches = list(dialect.iter_batches(engine, sql, 100))

    frame = batches[0]
    assert frame["BIG_VALUE"].dtype == np.dtype("float64")

    # The source holds 9223372036854775807. What comes back is 2**63, i.e. one
    # more than the source can even store — the rounding went the wrong way past
    # the top of the range.
    assert int(frame["BIG_VALUE"][0]) == 9223372036854775808
    assert int(frame["BIG_VALUE"][0]) != 9223372036854775807

    # ``ID`` has no NULL, so it is not demoted. That contrast is the whole point:
    # the NULL is the trigger, not the width of the value.
    assert frame["ID"].dtype == np.dtype("int64")
