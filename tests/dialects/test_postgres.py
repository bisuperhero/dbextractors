"""Tests for `PostgresDialect` — above all for the fact that pandas computes the
hash, not the source.

That last point is not a dialect detail: the pandas digest and the SQL digest do
not agree (4 of 51 tables), so if PostgreSQL started hashing in SQL, the first
run after the migration would evaluate the whole table as changed for some 58
pipelines. The tests below guard both halves of that agreement.
"""

from __future__ import annotations

import pytest

from dbextractors.dialects.base import TableRef, UnsupportedFeature
from dbextractors.dialects.postgres import DEFAULT_SOURCE_SCHEMA, PostgresDialect


@pytest.fixture()
def dialect() -> PostgresDialect:
    return PostgresDialect()


# --- Source schema ----------------------------------------------------------


def test_an_empty_schema_means_public(dialect) -> None:
    """One predecessor has ``get('source_schema', 'public')``, another
    ``get(...) or 'public'``.

    Those are not the same: when the key **is** in the configuration and is
    empty, the first one takes the empty string and a query against
    ``""."table"`` fails. Variant A wins — an empty value in YAML means "not
    filled in".
    """
    assert dialect.qualified(TableRef(name="t", schema="")) == f'"{DEFAULT_SOURCE_SCHEMA}"."t"'
    assert dialect.qualified(TableRef(name="t", schema=None)) == f'"{DEFAULT_SOURCE_SCHEMA}"."t"'


def test_a_filled_in_schema_is_used(dialect) -> None:
    assert dialect.qualified(TableRef(name="t", schema="stg")) == '"stg"."t"'


def test_select_is_qualified(dialect) -> None:
    sql = dialect.render_select(["a", "b"], TableRef(name="t", schema="stg"))
    assert sql == 'SELECT "a", "b" FROM "stg"."t"'


# --- Hash -------------------------------------------------------------------


def test_there_is_no_hash_on_the_source_side(dialect) -> None:
    """Not an omission — it is what is stored in the target."""
    assert dialect.hash_in_pandas is True
    with pytest.raises(UnsupportedFeature, match="pandas"):
        dialect.render_hash_expr(["a"], "row_hash")


def test_hash_diff_is_supported_regardless(dialect) -> None:
    """The hash_diff strategy does run on PostgreSQL — it just computes the hash itself."""
    from dbextractors.dialects.base import FEATURE_HASH_DIFF

    assert dialect.supports(FEATURE_HASH_DIFF)


def test_mysql_hashes_on_the_source(dialect) -> None:
    """The counterpart: on MySQL the hash **has to** be computed in SQL — keep it
    from being generalised."""
    from dbextractors.dialects.mysql import MySQLDialect

    assert MySQLDialect().hash_in_pandas is False


# --- Type map ---------------------------------------------------------------


def test_the_type_map_is_not_the_identity(dialect) -> None:
    """Source and target are both PostgreSQL, yet the map still changes types."""
    assert dialect.type_map["smallint"] == "INTEGER"
    assert dialect.type_map["character"] == "TEXT"
    assert dialect.type_map["character varying"] == "VARCHAR"


def test_money_is_text_not_numeric(dialect) -> None:
    """The only departure from the predecessor's type map, and the driver forces it.

    psycopg2 returns ``money`` as a localised string (``'3,50 EUR'``), so `COPY`
    into NUMERIC would fail. Verified against a live database in
    `test_postgres_db.py::test_money_goes_into_text`.
    """
    assert dialect.type_map["money"] == "TEXT"


def test_an_unknown_type_falls_back_to_text(dialect) -> None:
    from dbextractors.dialects.base import ColumnDef

    column = ColumnDef(name="x", source_type="hstore", pg_type="")
    assert dialect.map_type(column) == "TEXT"


# --- Connecting -------------------------------------------------------------


def test_the_password_is_escaped(dialect) -> None:
    """An ``@`` in the password would break the URL and the connection would go
    somewhere else."""
    url = dialect.build_conn_str({"user": "u", "password": "a@b", "database": "d"}, "h", 5432)
    assert url == "postgresql+psycopg2://u:a%40b@h:5432/d"


def test_charset_is_not_passed_on(dialect) -> None:
    """Unlike MySQL: the encoding is decided by the database, not by a connection
    parameter."""
    url = dialect.build_conn_str(
        {"user": "u", "password": "p", "database": "d", "charset": "utf8mb4"}, "h", 5432
    )
    assert "charset" not in url


def test_a_probe_against_a_closed_port_is_false(dialect) -> None:
    assert dialect.probe("127.0.0.1", 1, timeout=0.5) is False


# --- Fingerprint ------------------------------------------------------------


def test_the_fingerprint_is_qualified(dialect) -> None:
    sql = dialect.render_fingerprint(TableRef(name="t", schema="stg"), pk="id")
    assert 'FROM "stg"."t"' in sql
    assert "COUNT(*) AS fp_count" in sql
    assert 'MAX("id") AS fp_max_id' in sql
