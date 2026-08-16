"""Pinning tests: 64-bit integers lose precision when the column contains a NULL.

**This is a known, deliberate limitation, not a bug that these tests are waiting
to see fixed.** They exist so that the loss stays exactly the size it is today
and nobody "repairs" it by accident.

## The mechanism

An integer column that also holds a NULL cannot be ``int64``, so pandas infers
``float64`` — and ``float64`` carries 53 bits of mantissa. Every integer above
2**53 is rounded on the way in. The row count stays right, the load reports
success, and the value in the target is wrong.

Measured end to end against the live containers (source -> ``COPY`` -> target,
value read back out of PostgreSQL), for **all four** dialects — Firebird, MySQL,
MSSQL and PostgreSQL alike, because the mechanism is pandas, not any one driver::

    source 9007199254740993     -> target 9007199254740992
    source 1234567890123456789  -> target 1234567890123456768

The same column with no NULL in it arrives intact, which is what makes the NULL
the trigger rather than the magnitude.

## Why it is not fixed

The predecessor loses the very same bits, in the very same place. Every one of
the 15 reference extractors reads through a bare ``pd.read_sql(...)`` with no
``dtype=`` and no ``coerce_float`` — e.g.
``reference/variant-b/a_firebird_streaming_extractor.py:957`` and
``reference/variant-b/a_mysql_streaming_extractor.py:873`` — and its conversion chain
(``convert_integer_columns`` at ``a_firebird_streaming_extractor.py:506``) only
ever does ``int(float(x))`` afterwards, which cannot recover what the read
already threw away. Its ``replace_pdna_with_none``
(``a_firebird_streaming_extractor.py:351``) is character for character the same
``.apply`` as ours and promotes ``Int64`` to ``float64`` identically.

Correcting the read would therefore make the new side write a **different value
into the target than the old side** — that is the definition of a golden-parity
failure, and parity outranks correctness here by project rule.

It is worse than "one column changes". The pandas hash path renders the value
straight out of the frame, so a plain nullable ``INTEGER`` column is hashed today
as ``'1||10.0'`` and would become ``'1||10'`` — see
`test_the_hash_follows_the_float_even_for_ordinary_integers`. Changing the read
dtype would recompute ``row_hash`` for every table that has a nullable numeric
column, hash-diff would call all of them changed, and roughly 530 tables would
demand a full reload. Silently, with a green run.

Lifting this needs an explicit decision and a coordinated reload of the affected
tables, recorded in ``docs/merge-decisions.md`` — not a quiet repair here.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from dbextractors.core import coerce, hashing

#: 2**53 + 1 — the smallest integer ``float64`` cannot represent.
JUST_ABOVE_MANTISSA = 9007199254740993
ROUNDED_MANTISSA = 9007199254740992

#: A realistic 19-digit key, far enough above 2**53 that it rounds by 21.
WIDE_KEY = 1234567890123456789
ROUNDED_WIDE_KEY = 1234567890123456768

#: The maximum signed 64-bit integer, which is what a Firebird/MySQL/MSSQL
#: ``BIGINT`` column may legitimately hold.
MAX_BIGINT = 9223372036854775807


# --- Lossy point 1: the read ------------------------------------------------
#
# ``pd.read_sql`` builds the frame through ``DataFrame.from_records`` (see
# ``pandas.io.sql._wrap_result``), so calling it directly reproduces the read
# faithfully without needing a database. The live end-to-end evidence sits in
# ``tests/dialects/test_firebird.py``.


def test_a_null_in_the_column_rounds_every_integer_above_the_mantissa() -> None:
    frame = pd.DataFrame.from_records(
        [(1, JUST_ABOVE_MANTISSA), (2, None), (3, WIDE_KEY)],
        columns=["id", "big_value"],
    )

    assert frame["big_value"].dtype == np.dtype("float64")
    assert int(frame["big_value"][0]) == ROUNDED_MANTISSA
    assert int(frame["big_value"][2]) == ROUNDED_WIDE_KEY


def test_the_same_column_without_a_null_arrives_intact() -> None:
    """The control. The NULL is the trigger; the magnitude on its own is fine."""
    frame = pd.DataFrame.from_records(
        [(1, JUST_ABOVE_MANTISSA), (3, WIDE_KEY)],
        columns=["id", "big_value"],
    )

    assert frame["big_value"].dtype == np.dtype("int64")
    assert list(frame["big_value"]) == [JUST_ABOVE_MANTISSA, WIDE_KEY]


def test_coerce_cannot_recover_a_value_the_read_already_rounded() -> None:
    """No amount of conversion downstream puts the lost bits back."""
    assert coerce.to_int_or_na(float(JUST_ABOVE_MANTISSA)) == ROUNDED_MANTISSA
    assert coerce.to_int_or_none(float(WIDE_KEY)) == ROUNDED_WIDE_KEY
    assert coerce.fmt_int_for_csv(float(WIDE_KEY)) == str(ROUNDED_WIDE_KEY)


# --- Lossy point 2: replace_pdna_with_none ----------------------------------
#
# Independent of the read: even a column that arrived with full precision is
# demoted here, and `target_pg.prepare_export_df` calls it four times. Its own
# docstring already records this; these tests hold it to the exact numbers.


def test_replace_pdna_with_none_demotes_an_intact_int64_column() -> None:
    frame = pd.DataFrame({"big_value": pd.array([JUST_ABOVE_MANTISSA, None], dtype="Int64")})

    result = coerce.replace_pdna_with_none(frame)

    assert result["big_value"].dtype == np.dtype("float64")
    assert int(result["big_value"][0]) == ROUNDED_MANTISSA


def test_replace_pdna_with_none_demotes_object_columns_of_python_ints() -> None:
    """The shape a ``NUMERIC``/``Decimal`` column arrives in is no safer."""
    frame = pd.DataFrame({"big_value": pd.Series([WIDE_KEY, None], dtype="object")})

    result = coerce.replace_pdna_with_none(frame)

    assert result["big_value"].dtype == np.dtype("float64")
    assert int(result["big_value"][0]) == ROUNDED_WIDE_KEY


# --- The blast radius on row_hash -------------------------------------------


def test_the_hash_follows_the_float_even_for_ordinary_integers() -> None:
    """Why the read dtype cannot simply be corrected.

    This column is a plain ``INTEGER`` holding ``10``. Nothing here is anywhere
    near 2**53 — and the hash is still taken over ``'10.0'``, because the NULL in
    the column made it ``float64``. Every nullable numeric column in every table
    is hashed this way today, so changing the read dtype changes all of them.
    """
    frame = pd.DataFrame.from_records([(1, 10), (2, None)], columns=["id", "qty"])
    frame["row_hash"] = None

    hashes = hashing.compute_row_hashes(frame, ["id", "qty"])

    assert hashes[0] == hashlib.sha256(b"1||10.0").hexdigest()
    assert hashes[0] != hashlib.sha256(b"1||10").hexdigest()


def test_the_hash_of_a_wide_key_is_taken_over_the_rounded_float() -> None:
    """On the pandas hash path the digest records the corruption, not the source.

    Two consequences worth naming, both inherited:

    1. the digest is of a value the source never held, so it can never be
       reconciled against a hash the source computed in SQL;
    2. the digest of an unchanged row **flips** depending on whether the batch
       happens to contain a NULL in that column — a false "row changed".
    """
    frame = pd.DataFrame.from_records(
        [(3, WIDE_KEY), (2, None)],
        columns=["id", "big_value"],
    )
    frame["row_hash"] = None

    hashes = hashing.compute_row_hashes(frame, ["id", "big_value"])

    assert hashes[0] == hashlib.sha256(b"3||1.2345678901234568e+18").hexdigest()
    assert hashes[0] != hashlib.sha256(str(f"3||{WIDE_KEY}").encode()).hexdigest()

    # The same row, in a batch whose column has no NULL: a different digest.
    intact = pd.DataFrame.from_records([(3, WIDE_KEY)], columns=["id", "big_value"])
    intact["row_hash"] = None
    assert hashing.compute_row_hashes(intact, ["id", "big_value"])[0] == (
        hashlib.sha256(f"3||{WIDE_KEY}".encode()).hexdigest()
    )


# --- The one case that is not silent ----------------------------------------


def test_the_maximum_bigint_with_a_null_fails_loudly_instead() -> None:
    """At the very top of the range the rounding overflows and the run stops.

    ``float64(9223372036854775807)`` is ``2**63`` exactly, which no longer fits a
    signed 64-bit integer, so the ``Int64`` cast raises rather than storing a
    wrong number. The predecessor reaches the same wall from the other side —
    ``df[col].apply(to_int_or_na).astype('Int64')`` at
    ``reference/variant-b/a_firebird_streaming_extractor.py:532`` — so a table with a
    nullable ``BIGINT`` at full range does not load on either side.

    Pinned because it is the one boundary where the corruption is *not* silent,
    and that property must not be lost.
    """
    frame = pd.DataFrame({"big_value": pd.Series([float(MAX_BIGINT), np.nan])})

    with pytest.raises(OverflowError):
        coerce.convert_integer_columns(frame, {"big_value": "BIGINT"})
