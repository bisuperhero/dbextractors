"""Tests for `AbraWatermarkStrategy` against a live target PostgreSQL.

The IDs below are ABRA-shaped: seven characters of a base-36 counter written
least significant digit first, then a three-character database suffix. They are
chosen so that the ID as a string and the counter disagree — ``Z000000101``
(counter 35) sorts above ``0100000101`` (counter 36). A strategy that compared
the IDs themselves would get these tests wrong.

The source is `FakeHashSource`, which evaluates the strategy's condition the way
a database would; the real renderings are tested per dialect in
``tests/dialects/test_abra_counter.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dbextractors.core import config
from dbextractors.core.strategies.abra_watermark import AbraWatermarkStrategy, abra_id_suffixes
from dbextractors.core.strategies.base import StrategyError, SurrogateKey, resolve_strategy
from dbextractors.core.strategies.full import FullLoadStrategy
from fakes import FakeHashSource, make_columns, make_context

pytestmark = pytest.mark.needs_pg

COLUMNS = [
    ("id", "varchar", "TEXT"),
    ("name", "varchar", "TEXT"),
]

SETTINGS = {"primary_column": "id", "abra_id_suffixes": ["101"]}


def abra_id(counter: int, suffix: str = "101") -> str:
    """An ABRA record ID: the counter in base 36, seven digits, reversed, then the suffix."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    for _ in range(7):
        counter, digit = divmod(counter, 36)
        out += digits[digit]
    return out + suffix


def _source(*ids: str) -> FakeHashSource:
    return FakeHashSource(pd.DataFrame({"id": list(ids), "name": list(ids)}), pk="id")


def _ctx(conn, schema, dialect, *, settings=None, where=None):
    return make_context(
        conn,
        schema,
        dialect=dialect,
        columns=make_columns(COLUMNS),
        settings={**SETTINGS, **(settings or {})},
        where=where,
    )


def _ids(conn, schema) -> set:
    with conn.cursor() as cur:
        cur.execute(f'SELECT id FROM "{schema}"."cil"')
        return {row[0] for row in cur.fetchall()}


def _delete(conn, schema, *ids: str) -> None:
    """Take rows out of the target, so the next run has something new to find.

    The same as the source having gained them since the last run, without a
    source that can be written to.
    """
    with conn.cursor() as cur:
        cur.execute(f'DELETE FROM "{schema}"."cil" WHERE id = ANY(%s)', (list(ids),))
    conn.commit()


def test_the_id_helper_writes_the_counter_backwards() -> None:
    assert abra_id(1) == "1000000101"
    assert abra_id(35) == "Z000000101"
    assert abra_id(36) == "0100000101"


def test_the_registry_hands_out_the_strategy() -> None:
    assert isinstance(resolve_strategy("abra_watermark"), AbraWatermarkStrategy)


def test_only_counters_above_the_watermark_are_transferred(conn, schema) -> None:
    """The ID of the newer row is the *smaller* string; the counter decides, not the ID."""
    source = _source(abra_id(1), abra_id(35))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": abra_id(36), "name": "new"})
    source.insert({"id": abra_id(1296), "name": "newer"})

    result = AbraWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 2, "only the two new counters may be read"
    assert result.rows_written == 2
    assert result.phase_metrics["watermark"] == "101:000000Z"
    assert _ids(conn, schema) == {abra_id(n) for n in (1, 35, 36, 1296)}


def test_a_second_run_transfers_nothing(conn, schema) -> None:
    source = _source(abra_id(1), abra_id(2))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": abra_id(3), "name": "new"})

    first = AbraWatermarkStrategy().run(_ctx(conn, schema, source))
    second = AbraWatermarkStrategy().run(_ctx(conn, schema, source))

    assert first.rows_written == 1
    assert second.rows_read == 0
    assert second.data_present is False
    assert len(_ids(conn, schema)) == 3


def test_a_row_read_twice_is_not_duplicated(conn, schema) -> None:
    """A rerun after a failure can bring a row that is already there; the upsert absorbs it."""
    source = _source(abra_id(1), abra_id(2), abra_id(3))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _delete(conn, schema, abra_id(2), abra_id(3))
    source.update(abra_id(3), "name", "changed at the source")

    AbraWatermarkStrategy().run(_ctx(conn, schema, source))

    with conn.cursor() as cur:
        cur.execute(f'SELECT id, "_name" FROM "{schema}"."cil" ORDER BY id')
        rows = cur.fetchall()
    assert sorted(r[0] for r in rows) == sorted({abra_id(1), abra_id(2), abra_id(3)})
    assert (abra_id(3), "changed at the source") in rows


def test_other_suffixes_are_left_to_the_full_load(conn, schema) -> None:
    """A ``000`` record is outside the counter; this strategy never reads it."""
    source = _source(abra_id(1), abra_id(2))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": abra_id(99, "000"), "name": "system record"})
    source.insert({"id": abra_id(3), "name": "new"})

    result = AbraWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.rows_read == 1
    assert abra_id(99, "000") not in _ids(conn, schema)


def test_each_suffix_has_its_own_watermark(conn, schema) -> None:
    """Two databases count separately; one's high counter must not hide the other's rows."""
    source = _source(abra_id(500, "101"), abra_id(5, "102"))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.insert({"id": abra_id(6, "102"), "name": "new in 102"})

    result = AbraWatermarkStrategy().run(
        _ctx(conn, schema, source, settings={"abra_id_suffixes": ["101", "102"]})
    )

    assert result.rows_read == 1
    assert abra_id(6, "102") in _ids(conn, schema)
    assert result.phase_metrics["watermark"] == f"101:{abra_id(500)[:7][::-1]}, 102:0000005"


def test_a_suffix_missing_from_the_target_is_read_whole(conn, schema) -> None:
    source = _source(abra_id(1), abra_id(2, "102"), abra_id(3, "102"))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    _delete(conn, schema, abra_id(2, "102"), abra_id(3, "102"))

    result = AbraWatermarkStrategy().run(
        _ctx(conn, schema, source, settings={"abra_id_suffixes": ["101", "102"]})
    )

    assert result.rows_read == 2
    assert result.phase_metrics["watermark"] == "101:0000001, 102:-"


def test_the_where_clause_from_the_configuration_applies(conn, schema) -> None:
    source = _source(abra_id(1))
    FullLoadStrategy().run(_ctx(conn, schema, source, where="1=1"))
    source.insert({"id": abra_id(2), "name": "new"})

    AbraWatermarkStrategy().run(_ctx(conn, schema, source, where="1=1"))

    assert any("1=1" in sql and "ABRA_COUNTER(`id`) > '0000001'" in sql for sql in source.seen_sql)


def test_the_source_is_not_counted_before_the_read(conn, schema) -> None:
    """A ``COUNT(*)`` over the counter would be a second full scan of the source."""
    source = _source(abra_id(1))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    source.estimate_calls.clear()
    source.insert({"id": abra_id(2), "name": "new"})

    result = AbraWatermarkStrategy().run(_ctx(conn, schema, source))

    assert source.estimate_calls == []
    assert result.rows_written == 1


def test_an_empty_target_falls_back_to_full(conn, schema) -> None:
    source = _source(abra_id(1), abra_id(2))
    FullLoadStrategy().run(_ctx(conn, schema, source))
    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE "{schema}"."cil"')
    conn.commit()

    result = AbraWatermarkStrategy().run(_ctx(conn, schema, source))

    assert result.fallback_reason is not None
    assert "is empty" in result.fallback_reason
    assert len(_ids(conn, schema)) == 2


def test_a_missing_target_falls_back_to_full(conn, schema) -> None:
    result = AbraWatermarkStrategy().run(_ctx(conn, schema, _source(abra_id(1))))

    assert result.fallback_reason == "the target table does not exist"


def test_without_suffixes_it_fails(conn, schema) -> None:
    ctx = _ctx(conn, schema, _source(abra_id(1)), settings={"abra_id_suffixes": None})

    with pytest.raises(StrategyError, match="abra_id_suffixes"):
        AbraWatermarkStrategy().run(ctx)


def test_without_a_primary_key_it_fails(conn, schema) -> None:
    ctx = make_context(
        conn,
        schema,
        dialect=_source(abra_id(1)),
        columns=make_columns(COLUMNS),
        settings={"abra_id_suffixes": ["101"]},
    )

    with pytest.raises(StrategyError, match="primary_column"):
        AbraWatermarkStrategy().run(ctx)


def test_a_surrogate_key_is_refused(conn, schema) -> None:
    ctx = _ctx(conn, schema, _source(abra_id(1)))
    ctx.surrogate = SurrogateKey(enabled=True, alias="id", expr="`a` || `b`")

    with pytest.raises(StrategyError, match="surrogate"):
        AbraWatermarkStrategy().run(ctx)


def test_a_dialect_without_the_feature_fails(conn, schema) -> None:
    """``required_features`` is checked before the source is touched."""
    from dbextractors.dialects.base import UnsupportedFeature

    source = _source(abra_id(1))
    source.FEATURES = frozenset({"keyset"})

    with pytest.raises(UnsupportedFeature, match="abra_watermark"):
        AbraWatermarkStrategy().run(_ctx(conn, schema, source))


# --- abra_id_suffixes, without a database ------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("101", ["101"]), (["101"], ["101"]), (("101", "102", "101"), ["101", "102"])],
)
def test_suffixes_are_read_as_strings(raw, expected) -> None:
    assert abra_id_suffixes({"abra_id_suffixes": raw}) == expected


@pytest.mark.parametrize("raw", [None, [], "", [101], ["101", ""], ["  "]])
def test_bad_suffixes_are_refused(raw) -> None:
    """A number is refused rather than converted: ``001`` unquoted arrives as ``1``."""
    with pytest.raises(StrategyError, match="abra_id_suffixes"):
        abra_id_suffixes({"abra_id_suffixes": raw})


def test_a_single_suffix_survives_config_parsing() -> None:
    """``_as_tuple`` would split ``'101'`` into characters; this key must not be."""
    parsed = config.parse(
        {
            "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
            "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
            "LOAD_SETTINGS": {"load_method": "abra_watermark", "abra_id_suffixes": "101"},
        }
    )

    assert abra_id_suffixes({"abra_id_suffixes": parsed.load_settings.abra_id_suffixes}) == ["101"]
