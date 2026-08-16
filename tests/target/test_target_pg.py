"""Tests for `core.target_pg` that need no database.

Whatever can be compared with the old implementation is compared through
`reference_oracle` — `drop_mage_cols`, `align_df_columns_to_db` and
`prepare_export_df` have a single variant across all the predecessors and must
give character-for-character the same result.

The rest of the module is SQL and is tested against a live PostgreSQL in
``test_target_pg_db.py``. What is here are only the functions a database would
verify nothing extra about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reference_oracle as ro
from dbextractors.core import target_pg
from dbextractors.core.strategies.base import TargetRef

TARGET = TargetRef(schema="raw_erp", table="invoices")


# --- Identifiers ------------------------------------------------------------


def test_quote_ident_matches_the_old_one() -> None:
    # The inputs are pinned by the recorded oracle: change a value and there is
    # no recording for it any more. The last one is a multi-byte identifier.
    old = ro.get("A-pg-ld", "quote_ident")
    for name in ["cena", "Cena", "_type", "a b", 'uvozovka"uvnitr', "", "ji\u017e"]:
        assert target_pg.quote_ident(name) == old(name)


def test_quote_ident_doubles_the_quotes() -> None:
    """Without this a column name could be used to break the query."""
    assert target_pg.quote_ident('a"b') == '"a""b"'


def test_qualify() -> None:
    assert target_pg.qualify(TARGET) == '"raw_erp"."invoices"'


# --- drop_mage_cols ---------------------------------------------------------


def test_drop_mage_cols_matches_the_old_one() -> None:
    old = ro.get("A-mysql", "drop_mage_cols")
    df = pd.DataFrame(
        {
            "id": [1],
            "_mage_created_at": ["x"],
            "_mage_updated_at": ["y"],
            "_mage_cokoliv": ["z"],
            "_timestamp": ["t"],
        }
    )
    assert list(target_pg.drop_mage_cols(df.copy()).columns) == list(old(df.copy()).columns)


def test_drop_mage_cols_drops_by_prefix_not_by_a_list() -> None:
    """The predecessors drop ``startswith('_mage_')``, not a fixed list of two
    columns."""
    df = pd.DataFrame({"id": [1], "_mage_something_new": [2]})
    assert list(target_pg.drop_mage_cols(df).columns) == ["id"]


# --- align_df_columns_to_db -------------------------------------------------


@pytest.fixture(scope="module")
def old_align():
    return ro.get("A-mysql", "align_df_columns_to_db")


ALIGN_CASES = [
    # (the batch's columns, the target's columns). The names are pinned by the
    # recorded oracle — an "extra" and a "missing" column here go by the names
    # the recording was made with.
    (["b", "a"], ["a", "b"]),
    (["A", "B"], ["a", "b"]),
    (["type"], ["_type"]),
    (["id", "navic"], ["id"]),
    (["id"], ["id", "chybejici"]),
    (["id"], []),
    (["_type"], ["_type"]),
    (["id", "type", "name"], ["id", "_type", "_name"]),
]


@pytest.mark.parametrize(("df_cols", "db_cols"), ALIGN_CASES)
def test_align_matches_the_old_one(old_align, df_cols, db_cols) -> None:
    data = {c: [1, 2] for c in df_cols}
    new = target_pg.align_df_columns_to_db(pd.DataFrame(data), db_cols)
    original = old_align(pd.DataFrame(data), db_cols)
    pd.testing.assert_frame_equal(new, original)


def test_align_fixes_the_order() -> None:
    """The order is part of the contract, not cosmetics — the golden test
    compares it."""
    df = pd.DataFrame({"b": [1], "a": [2]})
    assert list(target_pg.align_df_columns_to_db(df, ["a", "b"]).columns) == ["a", "b"]


def test_align_supplies_the_underscore_prefix() -> None:
    """The target has ``_type`` (the mage prefix), the batch has ``type``. They
    have to be matched up."""
    df = pd.DataFrame({"type": ["x"]})
    out = target_pg.align_df_columns_to_db(df, ["_type"])
    assert list(out.columns) == ["_type"]
    assert out["_type"].tolist() == ["x"]


# --- prepare_export_df ------------------------------------------------------


def _batch() -> pd.DataFrame:
    # The column names are pinned by the recorded oracle, so they stay as they
    # were recorded: name, amount, date, flag.
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nazev": ["a", None, "c"],
            "castka": [1.5, 2.5, None],
            "datum": ["2026-08-11", "0000-00-00", None],
            "priznak": [1, 0, None],
            "_mage_created_at": ["x", "y", "z"],
        }
    )


PREPARE_ARGS = {
    "meta": {"_deleted_in_source": False},
    "all_columns": ["id", "nazev", "castka", "datum", "priznak", "row_hash", "_timestamp"],
    "date_like_columns": {"datum"},
    "integer_columns_mapping": {"id": "INTEGER"},
    "overwrite_types": {
        "id": "INTEGER",
        "nazev": "TEXT",
        "castka": "NUMERIC",
        "priznak": "BOOLEAN",
    },
    "orig_type_map": {"datum": "date"},
}


def test_prepare_export_df_matches_the_old_one() -> None:
    old = ro.get("A-mysql", "prepare_export_df")
    new = target_pg.prepare_export_df(batch_df=_batch(), **PREPARE_ARGS)
    original = old(batch_df=_batch(), **PREPARE_ARGS)
    pd.testing.assert_frame_equal(new, original)


def test_prepare_export_df_drops_the_mage_columns() -> None:
    out = target_pg.prepare_export_df(batch_df=_batch(), **PREPARE_ARGS)
    assert not [c for c in out.columns if c.startswith("_mage_")]


def test_prepare_export_df_adds_the_missing_columns() -> None:
    out = target_pg.prepare_export_df(batch_df=_batch(), **PREPARE_ARGS)
    assert "row_hash" in out.columns
    assert out["row_hash"].isna().all()


# --- Type inference ---------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected_type"),
    [
        ([1, 2, 3], "smallint"),
        ([1, 100_000], "integer"),
        ([1, 10_000_000_000], "bigint"),
        ([1.5, 2.5], "double precision"),
        (["a", "b"], "text"),
        ([True, False], "boolean"),
        ([None, None], "text"),
        ([{"a": 1}], "JSONB"),
        ([["a", "b"]], "text[]"),
    ],
)
def test_infer_pg_type(values, expected_type) -> None:
    from pandas.api.types import infer_dtype

    series = pd.Series(values, name="column")
    assert target_pg.infer_pg_type(series, infer_dtype(series, skipna=True)) == expected_type


def test_build_create_table_respects_overwrite_types() -> None:
    df = pd.DataFrame({"id": [1], "price": [1.5]})
    sql = target_pg.build_create_table(df, TARGET, overwrite_types={"price": "NUMERIC"})
    assert '"price" NUMERIC' in sql
    assert '"id" smallint' in sql


def test_build_create_table_leaves_the_names_alone() -> None:
    """Sanitisation belongs to `core.naming`; here it could only change
    something quietly."""
    df = pd.DataFrame({"_type": [1], "_deleted_in_source": [True]})
    sql = target_pg.build_create_table(df, TARGET)
    assert '"_type"' in sql
    assert '"_deleted_in_source"' in sql


def test_build_create_table_fails_on_a_column_without_a_type() -> None:
    df = pd.DataFrame({"id": [1]})
    with pytest.raises(target_pg.TargetError, match="row_hash"):
        target_pg.build_create_table(df, TARGET, columns=["id", "row_hash"])


def test_an_empty_object_column_fails_instead_of_guessing() -> None:
    """Mage raises `BadConversionError` in this case; so do we, with our own
    message."""
    series = pd.Series([None, None], dtype=object, name="x")
    with pytest.raises(target_pg.TargetError, match="nothing but empty values"):
        target_pg.infer_pg_type(series, "mixed")


# --- COPY: format and serialisation ----------------------------------------


def test_the_copy_sql_has_the_same_shape_as_mages() -> None:
    sql = target_pg.build_copy_sql(TARGET, ["id", "_type"])
    assert sql == (
        'COPY "raw_erp"."invoices" ("id", "_type") FROM STDIN '
        "(FORMAT csv, DELIMITER ',', NULL '', FORCE_NULL(\"id\", \"_type\"))"
    )


def _csv(df: pd.DataFrame, chunk_rows: int = 10_000) -> str:
    reader = target_pg._CsvChunkReader(df, chunk_rows)
    out = []
    while True:
        piece = reader.read(64)
        if not piece:
            break
        out.append(piece)
    return "".join(out)


def test_the_csv_is_identical_to_to_csv() -> None:
    """Hand-rolled quoting would have to match ``to_csv`` character for
    character.

    Missing means changing the data in the target, not merely formatting it
    differently — which is why `to_csv` is called rather than imitated.
    """
    import csv as _csv_mod

    df = pd.DataFrame(
        {
            "a": ['quote " inside', "comma, inside", "tab\tinside", "", None],
            "b": ["line\nbreak", "CR\r\nbreak", "back \\ slash", "naïve", "日本語"],
        }
    )
    assert _csv(df) == df.to_csv(header=False, index=False, na_rep="", quoting=_csv_mod.QUOTE_ALL)


def test_a_lone_cr_is_quoted() -> None:
    """With the default ``QUOTE_MINIMAL`` pandas does not quote it and ``COPY``
    rejects it.

    Mage suffers from this too — only `normalize_carriage_returns` one step
    above protects it. `copy_from_stdin` must not rely on that.
    """
    line = _csv(pd.DataFrame({"a": ["lone\rCR"]}))
    assert line == '"lone\rCR"\n'


def test_the_csv_in_pieces_equals_the_csv_in_one_go() -> None:
    df = pd.DataFrame({"a": [f"line {i}" for i in range(2500)]})
    assert _csv(df, chunk_rows=100) == _csv(df, chunk_rows=10_000)


def test_the_reader_returns_empty_at_the_end() -> None:
    reader = target_pg._CsvChunkReader(pd.DataFrame({"a": [1]}))
    assert reader.read(1000) == '"1"\n'
    assert reader.read(1000) == ""


def test_serialisation_of_dicts_and_arrays() -> None:
    """``COPY`` into ``text[]`` fails on ``[a, b]`` — it needs ``{a, b}``."""
    df = pd.DataFrame({"j": [{"a": 1}], "p": [["a", "b"]]})
    out = target_pg._serialize_objects(df.copy())
    assert out["j"].iloc[0] == '{"a": 1}'
    assert out["p"].iloc[0] == '{"a", "b"}'


def test_serialisation_leaves_text_alone() -> None:
    df = pd.DataFrame({"t": ["[not an array]"], "c": [np.int64(5)]})
    out = target_pg._serialize_objects(df.copy())
    assert out["t"].iloc[0] == "[not an array]"


def test_copy_without_columns_fails() -> None:
    with pytest.raises(target_pg.TargetError, match="column order is part of the contract"):
        target_pg.copy_from_stdin(None, TARGET, pd.DataFrame({"a": [1]}), [])


# --- Names derived from the table -------------------------------------------


def test_index_name() -> None:
    assert target_pg.index_name(TARGET, "id") == "idx_invoices_id_unique"


def test_index_name_is_not_truncated_quietly() -> None:
    """Cutting at 63 characters would give two tables the same index name."""
    long_ref = TargetRef(schema="s", table="t" * 60)
    name = target_pg.index_name(long_ref, "primary_key")
    assert len(name) == target_pg.MAX_IDENTIFIER_LENGTH
    other = target_pg.index_name(TargetRef(schema="s", table="t" * 59 + "u"), "primary_key")
    assert name != other


def test_the_shadow_ref_is_different_every_time() -> None:
    """The orchestrator runs 3 pipelines at once — a fixed name would clash."""
    a = target_pg.shadow_ref(TARGET)
    b = target_pg.shadow_ref(TARGET)
    assert a.table != b.table
    assert a.schema == b.schema == TARGET.schema
    assert a.table.startswith(target_pg.SHADOW_PREFIX)
    assert len(a.table) <= target_pg.MAX_IDENTIFIER_LENGTH


def test_the_shadow_ref_fits_even_for_a_long_name() -> None:
    long_ref = TargetRef(schema="s", table="t" * 63)
    assert len(target_pg.shadow_ref(long_ref).table) <= target_pg.MAX_IDENTIFIER_LENGTH


def test_drop_shadow_refuses_anything_that_is_not_a_shadow() -> None:
    """The safeguard against dropping the target table. It cannot be switched
    off."""
    with pytest.raises(target_pg.TargetError, match="is not a shadow table"):
        target_pg.drop_shadow_table(None, TARGET)


def test_the_reader_is_not_quadratic() -> None:
    """A regression: `_CsvChunkReader` re-sliced its buffer on **every** read.

    ``psycopg2`` asks for 8 kB at a time, so on a hundred-megabyte chunk that
    meant twelve thousand copies of the whole buffer. Measured on a heavy batch:
    209 s against 2 s after the fix. The test catches it without timing
    anything — a large chunk has to be read as fast as a small one, so comparing
    the number of copied characters is enough.

    Were the re-slicing to come back, the ratio would run away with the chunk
    size.
    """
    big = pd.DataFrame({"a": ["x" * 5_000] * 400})

    def read_all(chunk_rows: int) -> int:
        reader = target_pg._CsvChunkReader(big, chunk_rows)
        chars = 0
        while True:
            piece = reader.read(8192)
            if not piece:
                break
            chars += len(piece)
        return chars

    in_small_chunks = read_all(1)
    in_large_chunks = read_all(400)
    assert in_small_chunks == in_large_chunks
    assert in_large_chunks > 2_000_000
