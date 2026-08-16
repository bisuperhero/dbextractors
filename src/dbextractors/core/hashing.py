"""``row_hash`` and the choice of hashed columns.

**No ``build_hash_expr`` here.** It moved to ``SourceDialect.render_hash_expr``,
because it is source-side SQL and differs in every dialect. Only what is genuinely
shared stays in this module.

## Who actually computes the hash

In most runs **not Python but the source database**. The expression is embedded
straight into the ``SELECT`` (``SHA2(CONCAT_WS('||', …), 256)`` on MySQL,
``HASHBYTES('SHA2_256', …)`` on MSSQL), and `add_hash_and_timestamp` then sees the
column already populated, so it does not recompute it.

The pandas path is used in only two cases:

1. the hash column is **missing** from the batch (the query carried no expression),
2. ``force_recompute=True`` — across all 15 predecessors this appeared in **only
   four places**, all in variant B's PostgreSQL extractor.

The practical consequence: on the main path, agreement with the old behaviour is
guaranteed by agreement of the **SQL string**, not by agreement of this module.
What is guarded here is the other path.

## Two traps that must be preserved

**1. The pandas path and the SQL path DO NOT MATCH.** Measured against a real
source (MySQL, 51 tables, `scripts/verify_hash_parity.py`): **4 out of 51** agree —
and precisely those **without binary columns**. Three documented causes:

- ``varbinary``/``blob`` arrives in pandas as ``bytes`` and ``str(b'x')`` yields
  ``"b'x'"``, prefix included, whereas ``CAST(… AS CHAR)`` yields ``x``. The
  predecessor hashes **before** `coerce.decode_bytes_columns`, so it sees this form
  too.
- Zero dates: ``CAST('0000-00-00 00:00:00' AS CHAR)`` returns that string, but the
  driver hands pandas ``None`` -> ``''``.
- Decimal scale: ``float(10,4)`` gives ``'27.0600'`` in MySQL, ``str(27.06)`` gives
  ``'27.06'`` in Python.

On MSSQL the divergence runs deeper still. Verified against a live SQL Server 2022:
``HASHBYTES('SHA2_256', CAST(… AS NVARCHAR(MAX)))`` hashes **UTF-16LE** bytes and
``CONVERT(…, 2)`` returns hex in **upper case**. Python hashes UTF-8 and returns
lower case — two completely different digests, not merely two spellings of one.

**Consequence:** for a table whose hashes were computed by the source, the code must
never fall back to the pandas path. One such fallback marks the whole table as
changed and forces a full reload. Seeding ``row_hash`` on the first run must
therefore go through the source expression, not through `compute_row_hashes`.

**2. ``.apply(axis=1)`` changes types.** pandas builds a row as a ``Series`` with a
common type for all columns, so in a frame of ``int64`` + ``float64``,
``str(row['id'])`` comes out as ``'1.0'``, not ``'1'``. The hash of a row therefore
depends on what **other** columns are in the batch — and changes when a column has
no NULLs in one batch and some in the next (``int64`` -> ``float64``). It is a source
of silent false changes, but it is the existing behaviour and `compute_row_hashes`
reproduces it deliberately; fixing it would recompute the hashes of ~530 tables.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

#: Separator between values in the hashed string. It has to match
#: ``CONCAT_WS('||', …)`` in the source SQL — this is part of the contract, not
#: formatting.
HASH_SEPARATOR = "||"

#: Columns left out of the hash when it is computed in pandas.
#: ``_deleted_in_source`` is deliberately **not** among them — the predecessor does
#: not exclude it, and adding it would change the hash everywhere it is computed
#: locally.
HASH_EXCLUDED_ALWAYS: frozenset = frozenset({"_timestamp"})


# --- Column selection -------------------------------------------------------


def compute_hash_columns(
    column_names: Sequence[str],
    hash_column_cfg: Optional[str],
    table_settings: dict,
    load_settings: dict,
) -> List[str]:
    """Picks the source columns that go into ``row_hash``.

    There were two genuine variants across 13 predecessor files, differing **only**
    by one debug log line. The behaviour was identical; the log is carried over at
    ``DEBUG`` level.

    The order is the order from ``column_names``, not from ``hash_include_columns`` —
    and order matters for a hash.
    """
    hash_include = load_settings.get("hash_include_columns")
    hash_exclude = load_settings.get("hash_exclude_columns", []) or []

    if table_settings.get("only_selected_columns") and table_settings.get("selected_columns"):
        hash_include = hash_include or table_settings.get("selected_columns")

    if hash_include:
        include = set(hash_include)
        hashed = [c for c in column_names if c in include and c != hash_column_cfg]
    else:
        hashed = [c for c in column_names if c != hash_column_cfg]

    if hash_exclude:
        exclude = set(hash_exclude)
        hashed = [c for c in hashed if c not in exclude]

    _log.debug("Hashed columns: %s", hashed)
    return hashed


# --- Row hash ---------------------------------------------------------------


def row_hash(values: Sequence[Any]) -> str:
    """SHA-256 over the joined values of a single row.

    The reference implementation for one row. **It is not called in a real run** —
    `compute_row_hashes` does that. It stays as the thing the vectorised path is
    measured against.
    """
    parts = ["" if _is_missing(v) else str(v) for v in values]
    return hashlib.sha256(HASH_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


def _is_missing(value: Any) -> bool:
    """``pd.isna`` on a scalar, without falling over on lists and arrays.

    The predecessor calls ``pd.isna(row[col])`` inside ``.apply``, which for a list
    returns an array and makes ``'' if <array> else …`` raise. That is handled here —
    it is the one place where this code departs from the predecessor, and only where
    the predecessor crashes.
    """
    result = pd.isna(value)
    return bool(result) if np.ndim(result) == 0 else False


def _row_series_dtype(frame: pd.DataFrame) -> Optional[np.dtype]:
    """The type pandas promotes a row to under ``.apply(axis=1)``.

    ``DataFrame.values`` uses the same common-type mechanism, so a single row is
    enough — taking the full frame would copy it for nothing. Returns ``None`` when
    the common type is ``object`` (i.e. promotion changes no value).
    """
    if frame.empty or len(frame.columns) == 0:
        return None
    try:
        dtype = frame.iloc[:1].values.dtype
    except (ValueError, TypeError):
        return None
    return None if dtype is np.dtype(object) else dtype


def _as_text(series: pd.Series) -> pd.Series:
    """``str(value)`` element by element, the way the predecessor does inside ``.apply``.

    For ``datetime64`` one **must not** use ``astype(str)``: for a column where every
    time is midnight, pandas prints only the date (``'2026-08-11'``), whereas
    ``str(Timestamp)`` gives ``'2026-08-11 00:00:00'``. The difference feeds straight
    into the hash.
    """
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return series.map(str)
    return series.astype(str)


def compute_row_hashes(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """``row_hash`` for a whole batch at once.

    Replaces ``df.apply(_row_hash, axis=1)`` — a row-wise ``apply`` is the most
    expensive option in pandas and the project's performance rules forbid it.

    **The result has to be identical bit for bit**, including the type promotion
    described in the module docstring.

    Watch one detail that is not obvious: the common type is looked for over the
    **whole frame**, not over the hashed columns. ``.apply(axis=1)`` builds a row out
    of all columns, so even a column that is not hashed influences how the hashed
    values are rendered. In practice this means ``add_hash_and_timestamp`` first
    creates an empty hash column (dtype ``object``) and thereby switches promotion
    off — and that has to be preserved.

    Verified by `tests/hashing/test_hashing.py` against the row-wise version on a
    battery of frames whose mix of types forces pandas to promote.
    """
    cols = list(columns)
    if not cols:
        empty = hashlib.sha256(b"").hexdigest()
        return pd.Series([empty] * len(df), index=df.index, dtype=object)

    common = _row_series_dtype(df)
    frame = df[cols]
    if common is not None:
        frame = frame.astype(common)

    parts = []
    for name in cols:
        series = frame[name]
        parts.append(_as_text(series).mask(series.isna(), ""))

    joined = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=HASH_SEPARATOR)
    return joined.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())


def add_hash_and_timestamp(
    df: Optional[pd.DataFrame],
    normalized_hash_col: str,
    updated_col_norm: Optional[str] = None,
    *,
    force_recompute: bool = False,
    hashed_columns_override: Optional[Sequence[str]] = None,
    compute_hash: bool = True,
) -> Optional[pd.DataFrame]:
    """Adds ``row_hash`` and ``_timestamp`` to the batch.

    The hash is computed only when the column is **missing** from the batch or when
    ``force_recompute`` is set. Otherwise whatever the source sent is kept — and that
    is the usual path.

    ``compute_hash`` existed in only one predecessor (a MySQL variant); the other 12
    files did not have it. Following the project rule that a new capability is an
    option with a safe default rather than a parallel ``_v2`` module, it is taken over
    with the default ``True``, i.e. the behaviour of the other twelve.

    ``updated_col_norm`` is accepted by the predecessor and **never used** — it
    appears in the body of none of the 13 copies. Kept because the signature is
    frozen.
    """
    if df is None:
        return df

    if compute_hash:
        hash_missing = normalized_hash_col not in df.columns
        if hash_missing:
            df[normalized_hash_col] = None

        if hash_missing or force_recompute:
            if hashed_columns_override is not None:
                hashed_columns = list(hashed_columns_override)
            else:
                hashed_columns = [
                    c
                    for c in df.columns
                    if c != normalized_hash_col and c not in HASH_EXCLUDED_ALWAYS
                ]
            df[normalized_hash_col] = compute_row_hashes(df, hashed_columns)

    if "_timestamp" not in df.columns:
        df["_timestamp"] = datetime.now(timezone.utc).isoformat()

    return df


# --- Primary key as text ----------------------------------------------------


def pk_to_str(value: Any) -> str:
    """The primary key as text, so that it can be compared across types.

    A single variant across 8 predecessor files — the three "variants" an AST diff
    reported differ only in their docstring. Bytes are decoded, because binary
    columns from both MySQL and MSSQL arrive as ``bytes`` and ``str(b'x')`` would
    give ``"b'x'"``.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def pk_series_to_str(series: pd.Series) -> pd.Series:
    """`pk_to_str` over a whole column. Vectorised equivalent, identical output."""
    if series.dtype != object:
        text = series.astype(str)
        return text.mask(series.isna(), "")
    return series.map(pk_to_str)


# --- Tracking changed keys --------------------------------------------------
#
# The predecessors did this with four closures (`flush_changed_keys` /
# `iter_changed_keys` / `cleanup_changed_keys_file` / `_finalize_live_pk`, 9 variants
# in total) over a local set and a file in ``/tmp``. The replacement is **not** a
# function in this module: the list of changed keys is the result of a SQL query
# against the target, so the whole of that bookkeeping moved into
# `target_pg.SourceHashSnapshot`. Here only an empty wrapper would be left.
