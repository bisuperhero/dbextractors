"""PostgreSQL as a source against a **live** database, all the way through.

The other dialect tests work on strings. This file exists for the one class of
bug that cannot be found on strings: a type that maps fine, but whose **value**
does not survive `COPY` into the target.

That is exactly what happened with ``money``. The predecessor maps it to
``NUMERIC``, but psycopg2 returns it as a *localised string* (``'3,50 EUR'``)
and `COPY` into NUMERIC fails with "invalid input syntax". In the golden test it
only showed up as ``InFailedSqlTransaction``, because cleaning up the temporary
table masked the real error — the tests below fix both.

The source is created in a scratch schema prefixed with ``dbx_golden_``, just
like the target. Nothing else is touched.
"""

from __future__ import annotations

from typing import Iterator

import pytest

pytestmark = pytest.mark.needs_pg

#: Columns picked deliberately to cover the types that behave neither like
#: numbers nor like text: array, JSON, binary, UUID, interval, money, network
#: address. ``data`` and ``value`` are reserved words on top of that (see the
#: column naming section of ``docs/legacy-compat.md``).
DDL = """
    id integer PRIMARY KEY,
    "data" jsonb,
    "value" numeric(14,4),
    binary_blob bytea,
    flag boolean,
    labels text[],
    key_uuid uuid,
    duration interval,
    money_amount money,
    changed_at timestamp,
    day_date date,
    net_address inet,
    time_of_day time
"""

INSERT = """
    SELECT i,
           jsonb_build_object('a', i, 'b', 'text ' || i),
           (i * 1.2345)::numeric(14,4),
           decode(md5(i::text), 'hex'),
           (i %% 2 = 0),
           ARRAY['a' || i, 'b' || i],
           md5(i::text)::uuid,
           (i || ' days')::interval,
           (i * 3.5)::money,
           timestamp '2024-01-01 00:00:00' + (i || ' hours')::interval,
           date '2024-01-01' + i,
           ('10.0.0.' || (i %% 255))::inet,
           time '08:00' + (i || ' minutes')::interval
    FROM generate_series(1, %s) i
"""


@pytest.fixture()
def source_table(conn, schema) -> Iterator[str]:
    """A source table with awkward types, in a scratch schema."""
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{schema}"."odd_types" ({DDL})')
        cur.execute(f'INSERT INTO "{schema}"."odd_types" {INSERT}', (200,))
    conn.commit()
    yield "odd_types"


def _config(schema: str, cil: str, table: str, **load) -> dict:
    from psycopg2.extensions import parse_dsn

    from dbextractors.golden import session

    # Access details come **only** from `resolve_dsn()`, not half from here and
    # half from `POSTGRES_*`. Two sources of truth have already brought this file
    # down once: `DBX_GOLDEN_DSN` was enough for `resolve_dsn()`, but
    # `os.environ["POSTGRES_USER"]` failed with `KeyError` — so the tests
    # required both sets at once without that being written down anywhere.
    # `parse_dsn` also copes with a password containing a space.
    dsn = parse_dsn(session.resolve_dsn())
    return {
        "TABLE": {
            "source_name": table,
            "source_schema": schema,
            "output_name": table,
            "output_schema": cil,
            "empty_rows_ok": True,
        },
        "SOURCE_DB": {
            "user": dsn.get("user", ""),
            "password": dsn.get("password", ""),
            "database": dsn["dbname"],
            "host": dsn.get("host", "127.0.0.1"),
            "port": int(dsn.get("port", 5432)),
        },
        "LOAD_SETTINGS": {"load_method": "full", "primary_column": "id", **load},
    }


def _columns(conn, cil: str, table: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (cil, table),
        )
        return dict(cur.fetchall())


def test_odd_types_survive_copy(conn, schema, cil, source_table) -> None:
    """The whole table gets transferred — this is the regression being guarded.

    Before the fix ``money`` failed here: it mapped to NUMERIC and `COPY`
    rejected the value ``'3,50 EUR'``.
    """
    import dbextractors

    state = dbextractors.run(_config(schema, cil, source_table), dialect="postgres")

    assert int(state["rows_written"].sum()) == 200
    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{cil}"."odd_types"')
        assert cur.fetchone()[0] == 200


def test_money_goes_into_text(conn, schema, cil, source_table) -> None:
    """``money`` is the only departure from the predecessor's type map — with a reason.

    Were it mapped to NUMERIC (as the predecessor has it), `COPY` would fail on
    the localised string psycopg2 returns.
    """
    import dbextractors

    dbextractors.run(_config(schema, cil, source_table), dialect="postgres")

    assert _columns(conn, cil, "odd_types")["money_amount"] == "text"
    with conn.cursor() as cur:
        cur.execute(f'SELECT money_amount FROM "{cil}"."odd_types" WHERE id = 1')
        assert cur.fetchone()[0] is not None


def test_reserved_words_get_an_underscore(conn, schema, cil, source_table) -> None:
    """``data`` and ``value`` are both in the list of 825 words — see the column
    naming section of ``docs/legacy-compat.md``."""
    import dbextractors

    dbextractors.run(_config(schema, cil, source_table), dialect="postgres")
    columns = _columns(conn, cil, "odd_types")

    assert "_data" in columns and "data" not in columns
    assert "_value" in columns and "value" not in columns


def test_values_of_complex_types_arrive(conn, schema, cil, source_table) -> None:
    """It is not enough that it goes through — the values have to be in the
    target, not NULL."""
    import dbextractors

    dbextractors.run(_config(schema, cil, source_table), dialect="postgres")

    with conn.cursor() as cur:
        cur.execute(
            f'SELECT count("_data"), count(binary_blob), count(labels), count(key_uuid), '
            f'count(duration), count(net_address) FROM "{cil}"."odd_types"'
        )
        assert cur.fetchone() == (200, 200, 200, 200, 200, 200)


def test_hash_diff_over_odd_types_settles_down(conn, schema, cil, source_table) -> None:
    """A second run with no changes must transfer nothing, these types included.

    On PostgreSQL the hash is computed in pandas, so values of type ``interval``
    or ``text[]`` pass through ``str()`` — and it only takes that representation
    differing between the seed and the diff for the table to be transferred
    forever.
    """
    import dbextractors

    dbextractors.run(_config(schema, cil, source_table), dialect="postgres")
    state = dbextractors.run(
        _config(schema, cil, source_table, load_method="hash"), dialect="postgres"
    )

    assert int(state["rows_written"].sum()) == 0
    assert state["load_method"].iloc[0] == "hash"


def test_cleanup_does_not_mask_a_write_error(conn, schema, cil) -> None:
    """`SeenKeys.close()` runs inside the ``try``, hence **before** the ``except``.

    When the write fails, the transaction is aborted and ``DROP`` of the
    temporary table ends in ``InFailedSqlTransaction`` — and that error would
    replace the original one. That is exactly how the real cause was lost on the
    table with ``money``, leaving only "current transaction is aborted" in the
    report.
    """
    import psycopg2

    from dbextractors.core import target_pg

    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{cil}"."t" (id integer)')
    conn.commit()

    def fails_midway() -> None:
        with target_pg.SeenKeys(conn, "id"), conn.cursor() as cur:
            cur.execute(f'SELECT 1 FROM "{cil}"."does_not_exist"')

    with pytest.raises(psycopg2.errors.UndefinedTable) as error:
        fails_midway()

    assert "does_not_exist" in str(error.value), "cleanup must not mask the original error"
