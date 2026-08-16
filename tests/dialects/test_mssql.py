"""Tests for `MSSQLDialect` — mainly the parts that can be compared verbatim
with the predecessor.

The hash expression is a special case: the **only** proof that ``row_hash`` will
not change is that the generated SQL is byte-for-byte identical to the
predecessor's. The same expression in the same engine over the same data returns
the same digest — and conversely, any difference in the expression silently
recomputes the digests and the next run transfers the whole table.

Of the four dialects MSSQL has the most syntax of its own: square brackets,
``TOP`` inside the SELECT list, ``OFFSET`` without ``LIMIT``, ``COUNT_BIG``. The
tests below guard each of those separately, because each only shows up in a run
against the source.
"""

from __future__ import annotations

import pytest

from dbextractors.dialects.base import SurrogateKey, TableRef
from dbextractors.dialects.mssql import DEFAULT_SOURCE_SCHEMA, MSSQLDialect


@pytest.fixture()
def dialect() -> MSSQLDialect:
    return MSSQLDialect()


# --- Hash expression --------------------------------------------------------


def test_hash_expression_is_byte_identical_to_the_predecessor(dialect) -> None:
    """``build_hash_expr`` as the predecessor's MSSQL extractor writes it.

    A second variant has the same function character for character, so there was
    nothing to merge.
    """
    expected = (
        "CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONCAT_WS('||', "
        "ISNULL(CAST([id] AS NVARCHAR(MAX)), ''), "
        "ISNULL(CAST([name] AS NVARCHAR(MAX)), ''))), 2) AS [row_hash]"
    )
    assert dialect.render_hash_expr(["id", "name"], "row_hash") == expected


def test_a_binary_column_goes_through_hex(dialect) -> None:
    """``image``/``binary``/``varbinary`` cannot be cast to ``NVARCHAR(MAX)``.

    It has to go the long way through ``VARBINARY(MAX)`` with style ``1``, that
    is, hex. Were a binary column sent down the ordinary branch, the hash would
    be computed from something other than it is today and the table would be
    transferred forever.
    """
    sql = dialect.render_hash_expr(["photo"], "row_hash", {"photo": "image"})
    assert "ISNULL(CONVERT(NVARCHAR(MAX), CONVERT(VARBINARY(MAX), [photo]), 1), '')" in sql


@pytest.mark.parametrize("source_type", ["image", "binary", "varbinary", "VarBinary"])
def test_every_binary_type_takes_the_hex_branch(dialect, source_type) -> None:
    sql = dialect.render_hash_expr(["c"], "h", {"c": source_type})
    assert "VARBINARY(MAX)" in sql


def test_a_non_binary_type_stays_in_the_ordinary_branch(dialect) -> None:
    sql = dialect.render_hash_expr(["c"], "h", {"c": "nvarchar"})
    assert sql == (
        "CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONCAT_WS('||', "
        "ISNULL(CAST([c] AS NVARCHAR(MAX)), ''))), 2) AS [h]"
    )


def test_without_types_everything_is_hashed_as_text(dialect) -> None:
    """``column_types`` is optional — missing them must not fail, it only means a
    binary column goes unrecognised. The predecessor has ``(col_types or {})``
    for the same reason."""
    assert dialect.render_hash_expr(["c"], "h") == dialect.render_hash_expr(["c"], "h", {})


def test_the_source_computes_the_hash_not_pandas(dialect) -> None:
    """The opposite of PostgreSQL, where it is the other way round — and it is not
    a choice but what is stored in the target."""
    assert dialect.hash_in_pandas is False


# --- Quoting and qualification ----------------------------------------------


def test_quotes_with_square_brackets(dialect) -> None:
    """``quote_char`` is a single character, which is not enough for a ``[`` … ``]`` pair."""
    assert dialect.quote_ident("Number") == "[Number]"
    assert dialect.quote_ident("  Number  ") == "[Number]"


def test_an_empty_schema_means_dbo(dialect) -> None:
    """Across the predecessors ``src_schema='dbo'`` is the default parameter value."""
    assert dialect.qualified(TableRef(name="t")) == "[dbo].[t]"
    assert dialect.qualified(TableRef(name="t", schema="")) == "[dbo].[t]"
    assert DEFAULT_SOURCE_SCHEMA == "dbo"


def test_a_filled_in_schema_is_used(dialect) -> None:
    assert dialect.qualified(TableRef(name="t", schema="stg")) == "[stg].[t]"


def test_select_is_qualified(dialect) -> None:
    sql = dialect.render_select(["a", "b"], TableRef(name="t", schema="stg"))
    assert sql == "SELECT [a], [b] FROM [stg].[t]"


def test_select_with_a_surrogate_key(dialect) -> None:
    surrogate = SurrogateKey(enabled=True, alias="sk", expr="CONCAT(a, '-', b)")
    sql = dialect.render_select(["sk", "a"], TableRef(name="t"), surrogate=surrogate)
    assert sql == "SELECT CONCAT(a, '-', b) AS [sk], [a] FROM [dbo].[t]"


# --- convert_nchar_to_varchar -----------------------------------------------
#
# The generated SQL is the whole of this feature: what it does to the values is
# a property of the server and is pinned live in `tests/dialects/test_source_db.py`.
# What can be settled here is *which* columns it touches — the half that can
# regress in silence, because wrapping a legacy single-byte VARCHAR would corrupt
# exactly the columns the cp1250 connection reads correctly today.

#: One column of every type the question can be asked about.
NCHAR_TABLE_TYPES = {
    "id": "int",
    "nvarchar_value": "nvarchar",
    "nchar_value": "nchar",
    "ntext_value": "ntext",
    "varchar_value": "varchar",
    "char_value": "char",
    "text_value": "text",
}


def _select_all(dialect, **kwargs) -> str:
    return dialect.render_select(
        list(NCHAR_TABLE_TYPES), TableRef(name="t"), column_types=NCHAR_TABLE_TYPES, **kwargs
    )


def test_the_select_is_unchanged_unless_the_key_is_set(dialect) -> None:
    """Off is the default, and off has to mean *byte-identical* to before.

    The key changes what an N-typed column holds in the target, so a table that
    does not ask for it must generate the query it generated yesterday — down to
    the text, because the text is what the golden comparison compares.
    """
    plain = (
        "SELECT [id], [nvarchar_value], [nchar_value], [ntext_value], "
        "[varchar_value], [char_value], [text_value] FROM [dbo].[t]"
    )

    assert _select_all(dialect) == plain
    assert _select_all(dialect, convert_nchar=False) == plain


def test_only_the_n_types_are_converted(dialect) -> None:
    """The wrap is driven by the introspected type, never by the column's name.

    ``varchar``/``char``/``text`` are single-byte and arrive correct today; they
    are the reason the connection charset is cp1250 in the first place. Wrapping
    them would trade one corruption for another.
    """
    sql = _select_all(dialect, convert_nchar=True)

    assert sql == (
        "SELECT [id], "
        "CONVERT(VARCHAR(MAX), [nvarchar_value]) AS [nvarchar_value], "
        "CONVERT(VARCHAR(MAX), [nchar_value]) AS [nchar_value], "
        "CONVERT(VARCHAR(MAX), [ntext_value]) AS [ntext_value], "
        "[varchar_value], [char_value], [text_value] FROM [dbo].[t]"
    )


def test_a_converted_column_keeps_its_name(dialect) -> None:
    """Without the alias the batch would come back with a nameless column.

    Everything downstream is keyed by the source column name — ``ctx.name_map``,
    the hash column selection, `apply_text_dtypes` — so an expression without
    ``AS`` would not merely look untidy, it would lose the column.
    """
    assert dialect.render_nchar_convert("Note") == "CONVERT(VARCHAR(MAX), [Note]) AS [Note]"


def test_a_column_with_no_introspected_type_is_not_converted(dialect) -> None:
    """A column missing from the type map is left alone rather than guessed at.

    A surrogate key is exactly that case: it does not exist in the source, so
    introspection has no type for it, and it has to keep its own expression.
    """
    surrogate = SurrogateKey(enabled=True, alias="sk", expr="CONCAT(a, '-', b)")
    sql = dialect.render_select(
        ["sk", "note"],
        TableRef(name="t"),
        surrogate=surrogate,
        column_types={"note": "nvarchar"},
        convert_nchar=True,
    )

    assert sql == (
        "SELECT CONCAT(a, '-', b) AS [sk], CONVERT(VARCHAR(MAX), [note]) AS [note] FROM [dbo].[t]"
    )


def test_the_conversion_does_not_reach_the_hash_expression(dialect) -> None:
    """`row_hash` is computed by the server from the column itself, and stays that way.

    The digests in the target were computed by ``HASHBYTES`` over the *raw*
    ``NVARCHAR``, so they are already right — it is the text stored alongside
    them that was truncated. Putting the conversion into the hash as well would
    recompute every digest of every table that switches the key on, on top of the
    reload the key already implies.
    """
    expr = dialect.render_hash_expr(["note"], "row_hash", {"note": "nvarchar"})

    assert expr == (
        "CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CONCAT_WS('||', "
        "ISNULL(CAST([note] AS NVARCHAR(MAX)), ''))), 2) AS [row_hash]"
    )


# --- Literals ---------------------------------------------------------------


def test_keyset_does_not_double_the_backslash(dialect) -> None:
    """Unlike MySQL: there ``\\`` is an escape character, here it would change the value."""
    assert dialect.sql_literal("path\\") == "'path\\'"


# --- Fingerprint ------------------------------------------------------------


def test_the_fingerprint_uses_count_big(dialect) -> None:
    """``COUNT(*)`` is an ``INT`` in MSSQL and **fails** above 2.1 billion rows."""
    sql = dialect.render_fingerprint(TableRef(name="t", schema="stg"), pk="id")
    assert "COUNT_BIG(*) AS fp_count" in sql
    assert "COUNT(*)" not in sql
    assert "FROM [stg].[t]" in sql


def test_the_fingerprint_always_has_four_columns(dialect) -> None:
    """The caller should not have to work out which mode it is — the order and the
    count are fixed."""
    sql = dialect.render_fingerprint(TableRef(name="t"))
    assert sql == (
        "SELECT COUNT_BIG(*) AS fp_count, NULL AS fp_max_id, NULL AS fp_max_ts, "
        "NULL AS fp_agg FROM [dbo].[t]"
    )


def test_the_fingerprint_with_sums(dialect) -> None:
    """The ``aggregate`` mode is here for tables where the source leaves the change
    timestamp NULL."""
    sql = dialect.render_fingerprint(TableRef(name="t"), pk="id", aggregate_columns=["net", "vat"])
    assert "COALESCE(SUM(CAST([net] AS DECIMAL(38,6))), 0)" in sql
    assert "COALESCE(SUM(CAST([vat] AS DECIMAL(38,6))), 0)" in sql


def test_the_fingerprint_honours_where(dialect) -> None:
    sql = dialect.render_fingerprint(TableRef(name="t"), where="year = 2024")
    assert sql.endswith("FROM [dbo].[t] WHERE (year = 2024)")


# --- Type map ---------------------------------------------------------------


def test_the_type_map_is_verbatim_from_the_predecessor(dialect) -> None:
    assert dialect.type_map["tinyint"] == "INTEGER"
    assert dialect.type_map["bit"] == "BOOLEAN"
    assert dialect.type_map["float"] == "DOUBLE PRECISION"
    assert dialect.type_map["nvarchar"] == "TEXT"
    assert dialect.type_map["varchar"] == "VARCHAR"


def test_money_is_numeric_unlike_a_pg_source(dialect) -> None:
    """It looks like an inconsistency with the PostgreSQL source, and it is not.

    There ``money`` maps to ``TEXT``, because psycopg2 returns a localised string
    (``'3,50 EUR'``). pymssql returns a ``Decimal``, so ``NUMERIC`` fits — and it
    is also what is in the target today.
    """
    from dbextractors.dialects.postgres import PostgresDialect

    assert dialect.type_map["money"] == "NUMERIC"
    assert dialect.type_map["smallmoney"] == "NUMERIC"
    assert PostgresDialect().type_map["money"] == "TEXT"


def test_binary_types_go_to_text_not_bytea(dialect) -> None:
    """Verbatim from the predecessor. ``BYTEA`` would change the column type in
    hundreds of tables."""
    assert dialect.type_map["image"] == "TEXT"
    assert dialect.type_map["binary"] == "TEXT"
    assert dialect.type_map["varbinary"] == "TEXT"


def test_an_unknown_type_falls_back_to_text(dialect) -> None:
    from dbextractors.dialects.base import ColumnDef

    assert dialect.map_type(ColumnDef(name="x", source_type="geography", pg_type="")) == "TEXT"


# --- Encoding ---------------------------------------------------------------


def test_the_decoding_cascade_starts_with_cp1250(dialect) -> None:
    """The source system keeps its databases in cp1250."""
    assert dialect.decode_encodings == ("cp1250", "utf-8", "latin-1")


def test_the_cascade_was_not_generalised_to_the_others(dialect) -> None:
    """cp1250 rejects only five bytes — UTF-8 text would pass through it as mojibake."""
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    assert MySQLDialect().decode_encodings == ("utf-8",)
    assert PostgresDialect().decode_encodings == ("utf-8",)


def test_binary_types_are_not_among_the_textual_ones(dialect) -> None:
    """``apply_text_dtypes`` decodes bytes as UTF-8 with replacement characters.

    The predecessor, however, runs them through the ``decode_encodings`` cascade,
    which yields different text. Were binary types among the textual ones, the
    contents of the target would change for columns of type ``image``.
    """
    assert dialect.text_like_types.isdisjoint({"image", "binary", "varbinary"})
    assert "nvarchar" in dialect.text_like_types


# --- Connecting -------------------------------------------------------------


def test_the_password_is_escaped(dialect) -> None:
    """An ``@`` in the password would break the URL and the connection would go
    somewhere else."""
    url = dialect.build_conn_str({"user": "u", "password": "a@b", "database": "d"}, "h", 1433)
    assert url == "mssql+pymssql://u:a%40b@h:1433/d"


def test_the_database_name_is_escaped(dialect) -> None:
    """In a multi-source run the name comes from ``DATABASES`` in the configuration."""
    url = dialect.build_conn_str({"user": "u", "password": "p", "database": "a b"}, "h", 1433)
    assert url.endswith("/a+b")


def test_charset_from_the_configuration_is_ignored(dialect) -> None:
    """**It looks like a breach of the frozen contract, and it is the opposite.**

    Today's MSSQL extractor has cp1250 hard-coded in ``connect_args`` and never
    reads the ``SOURCE_DB.charset`` key. Yet all 16 pipelines of that source have
    ``utf8mb4`` in it — the name of a *MySQL* encoding, which pymssql does not
    know. Were the dialect to start honouring it, the decoding of text would
    change in all 16 tables at once.
    """
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "d", "charset": "utf8mb4"}, "h", 1433
    )
    assert "charset" not in url
    assert dialect.connect_args["charset"] == "cp1250"


def test_the_connection_charset_is_pinned_with_its_cost_written_down(dialect) -> None:
    """cp1250 on the connection is a **trade**, and this pins the side that was chosen.

    It truncates every ``NVARCHAR``/``NCHAR``/``NTEXT`` value at the first
    character latin-1 cannot express — ``'příliš žluťoučký kůň'`` arrives as
    ``'p'`` — while keeping the legacy single-byte columns readable. Both halves
    are measured against a live source in
    ``tests/dialects/test_source_db.py``; this test is the one that runs without
    a database, so it is where somebody changing the constant will land first.

    Changing it is a parity decision, not a bugfix: both predecessors pass the
    same charset, so the target has held the truncated text since it was
    created. The reasoning is in ``docs/legacy-compat.md``.
    """
    from dbextractors.dialects.mssql import DEFAULT_CHARSET

    assert DEFAULT_CHARSET == "cp1250"
    assert dialect.connect_args["charset"] == DEFAULT_CHARSET
    # The N-types stay among the textual ones despite the truncation: excluding
    # them would change the target's dtypes on top of the truncation, which is a
    # second deviation rather than a repair of the first.
    assert {"nvarchar", "nchar", "ntext"} <= dialect.text_like_types


def test_mysql_by_contrast_does_read_charset_from_the_configuration(dialect) -> None:
    """Keep it from being generalised: on MySQL ``charset`` in the URL is live and used."""
    from dbextractors.dialects.mysql import MySQLDialect

    url = MySQLDialect().build_conn_str(
        {"user": "u", "password": "p", "database": "d", "charset": "utf8mb4"}, "h", 3306
    )
    assert url.endswith("?charset=utf8mb4")


def test_login_timeout_is_in_connect_args(dialect) -> None:
    """Without it, logging in to a starved server hangs.

    The TCP probe passes meanwhile — the port is listening, only the login
    handshake never completes. One deployment paid for that with error 20002.
    """
    assert dialect.connect_args["login_timeout"] == 30


def test_a_probe_against_a_closed_port_is_false(dialect) -> None:
    assert dialect.probe("127.0.0.1", 1, timeout=0.5, attempts=1) is False


def test_the_probe_retries(dialect, monkeypatch) -> None:
    """A short TCP outage would otherwise drop into the SSH branch with a
    confusing error.

    It is the only dialect that retries the probe — and it is deliberate, not an
    oversight.
    """
    attempts = []
    monkeypatch.setattr("dbextractors.dialects.mssql.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "dbextractors.dialects.mssql.socket.create_connection",
        lambda *a, **kw: attempts.append(1) or (_ for _ in ()).throw(OSError("nothing")),
    )

    assert dialect.probe("h", 1433, timeout=0.1) is False
    assert len(attempts) == 3


# --- Capabilities -----------------------------------------------------------


def test_it_is_the_only_one_that_can_partition(dialect) -> None:
    """MSSQL is the only source that today assembles the target from several databases."""
    from dbextractors.dialects.base import FEATURE_PARTITION_BY_SOURCE
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    assert dialect.supports(FEATURE_PARTITION_BY_SOURCE)
    assert not MySQLDialect().supports(FEATURE_PARTITION_BY_SOURCE)
    assert not PostgresDialect().supports(FEATURE_PARTITION_BY_SOURCE)


def test_it_can_do_hash_diff_and_keyset(dialect) -> None:
    from dbextractors.dialects.base import FEATURE_HASH_DIFF, FEATURE_KEYSET

    assert dialect.supports(FEATURE_HASH_DIFF)
    assert dialect.supports(FEATURE_KEYSET)


# --- Binary columns: a warning, not a failure -------------------------------


def test_a_binary_column_is_reported(dialect, caplog) -> None:
    """A table with an ``image`` column will most likely not load — and that should
    be visible.

    The predecessor's type map sends binary data to TEXT, but it does not fit
    into CSV and `COPY` ends with "need to escape, but no escapechar set".
    **Today's extractor fails on it too** — verified on a real table, both
    components with the same error. Letting it fail only in `COPY` is needlessly
    obscure: the message then talks about CSV and not about a column.
    """
    from dbextractors.dialects.base import ColumnDef

    columns = [
        ColumnDef(name="ID", source_type="int", pg_type="INTEGER"),
        ColumnDef(name="Data", source_type="image", pg_type="TEXT"),
    ]
    with caplog.at_level("WARNING", logger="dbextractors.dialects.mssql"):
        dialect._warn_about_binary(TableRef(name="documents"), columns)

    assert "Data" in caplog.text
    assert "excluded_columns" in caplog.text, "the message has to say what to do about it"


def test_without_a_binary_column_nothing_is_warned_about(dialect, caplog) -> None:
    from dbextractors.dialects.base import ColumnDef

    with caplog.at_level("WARNING", logger="dbextractors.dialects.mssql"):
        dialect._warn_about_binary(
            TableRef(name="t"), [ColumnDef(name="ID", source_type="int", pg_type="INTEGER")]
        )

    assert caplog.text == ""
