"""``strict_integer_precision`` end to end, from ``LOAD_SETTINGS`` to the target.

The check itself is covered in ``tests/coerce/test_strict_integer_precision.py``.
What these tests add is the wiring: that the key reaches the check at all (there
is exactly one plumbing point, `full.FullLoadStrategy._prepare`, which every
strategy funnels through), and that with the key off a real ``COPY`` into a real
PostgreSQL still stores the same rounded number it has always stored.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core import coerce
from dbextractors.core.strategies.full import FullLoadStrategy
from fakes import FakeDialect, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [("id", "int", "INTEGER"), ("big_value", "bigint", "BIGINT")]

#: The 19-digit key from the end-to-end measurement, and what ``float64`` makes
#: of it once the NULL in the column has forced the whole column to float.
WIDE_KEY = 1234567890123456789
ROUNDED_WIDE_KEY = 1234567890123456768


def _batch() -> pd.DataFrame:
    """A nullable ``BIGINT`` as ``pd.read_sql`` hands it over — already float64."""
    return pd.DataFrame.from_records([(1, WIDE_KEY), (2, None)], columns=["id", "big_value"])


def test_the_run_still_writes_the_rounded_value_by_default(conn, schema) -> None:
    """The inherited behaviour. Silent, wrong, and deliberately kept."""
    dialect = FakeDialect([_batch()])
    ctx = make_context(conn, schema, dialect=dialect, columns=make_columns(COLUMNS))

    result = FullLoadStrategy().run(ctx)

    assert result.rows_written == 2
    with conn.cursor() as cur:
        cur.execute(f'SELECT big_value FROM "{schema}"."cil" WHERE id = 1')
        assert cur.fetchone()[0] == ROUNDED_WIDE_KEY


def test_the_key_stops_the_run_instead_of_writing_a_wrong_number(conn, schema) -> None:
    dialect = FakeDialect([_batch()])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"strict_integer_precision": True},
    )

    with pytest.raises(coerce.IntegerPrecisionError, match="big_value"):
        FullLoadStrategy().run(ctx)


def test_the_key_accepts_the_configuration_spellings_of_true(conn, schema) -> None:
    """``LOAD_SETTINGS`` values arrive as strings from YAML as well as bools, and
    `config.is_truthy` is what the whole contract is parsed with."""
    dialect = FakeDialect([_batch()])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"strict_integer_precision": "ano"},
    )

    with pytest.raises(coerce.IntegerPrecisionError):
        FullLoadStrategy().run(ctx)


def test_the_key_on_lets_an_ordinary_table_load(conn, schema) -> None:
    """Nothing in this batch is near 2**53, so the key must be invisible."""
    batch = pd.DataFrame.from_records([(1, 10), (2, None)], columns=["id", "big_value"])
    dialect = FakeDialect([batch])
    ctx = make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={"strict_integer_precision": True},
    )

    result = FullLoadStrategy().run(ctx)

    assert result.rows_written == 2
    with conn.cursor() as cur:
        cur.execute(f'SELECT big_value FROM "{schema}"."cil" ORDER BY id')
        assert [r[0] for r in cur.fetchall()] == [10, None]
