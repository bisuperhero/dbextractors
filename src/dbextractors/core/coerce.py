"""Type conversion and batch sanitising.

## The non-negotiable condition

**The output must be identical character for character with the previous one.**
Not "equivalent" — identical. ``row_hash`` is computed from the string
representation of the values, so if ``to_date_str`` started returning
``2026-08-07`` instead of ``2026-08-07 00:00:00``, every hash would change, the
hash diff would judge the whole table as changed and ~530 tables would demand a
full reload. Silently, with a green run.

``tests/coerce/test_characterization_coerce.py`` verifies it by running this
implementation and the predecessor's through the very same battery of inputs.

## The performance mandate

The predecessor's conversion layer holds **18 ``.apply()`` calls** and is one of
the reasons the full load runs at 4 430 rows/s against a ceiling of 34 200 rows/s.

Vectorisation is used where it does not change the result — typically through a
mask that skips a whole column with no relevant values. Where vectorisation would
change the result, it is not used: ``pd.to_datetime`` over a *series* infers one
common format for the whole column, whereas over a scalar it parses each value on
its own — with mixed formats that yields a different result.

## Two decisions that change the output and are therefore parameters

Both agreed.

1. **Encoding of byte columns.** MSSQL and Firebird decode ``cp1250`` first,
   MySQL and PostgreSQL ``utf-8``. This is not an oversight: the order matches
   what that particular source really stores. Unifying them is impossible —
   ``cp1250`` is a single-byte encoding that rejects only five bytes
   (``0x81 0x83 0x88 0x90 0x98``), so UTF-8 text passes through it as mojibake.
   Documented: out of six accented Czech words, exactly one survives the
   ``cp1250``-first cascade.
2. **``ensure_ascii`` in ``to_jsonb``.** The PostgreSQL sources write accented
   characters literally, the other seven files as ``\\uXXXX``.

``coerce`` knows nothing about dialects — it is handed both as parameters and the
dialect supplies the values.

Mind pandas 1.5.3 (not 2.0): no ``dtype_backend``.
"""

from __future__ import annotations

import json
import math
import re
from datetime import time as dt_time
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# --- Constants ---------------------------------------------------------------

#: Values that read as "nothing" in text. Identical across the predecessor.
_NULL_STRINGS: frozenset[str] = frozenset({"", "nan", "none", "null"})

#: Sentinels for invalid dates. **The union of all three variants** — MySQL
#: produces `0000-00-00`, the other dialects `9999-12-31` as "never". The union is
#: safe, because a source never produces another dialect's sentinel.
INVALID_DATE_STRINGS: frozenset[str] = frozenset({"0000-00-00", "9999-12-31"})

INVALID_DATETIME_STRINGS: frozenset[str] = frozenset(
    {
        "0000-00-00 00:00:00",
        "0000-00-00T00:00:00",
        "9999-12-31 00:00:00",
        "9999-12-31 23:59:59",
        "9999-12-31T00:00:00",
        "9999-12-31T23:59:59",
    }
)

#: Patterns of invalid dates for `fix_invalid_dates`. The union of the variants.
INVALID_DATE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^0{4}-0{2}-0{2}$"),
    re.compile(r"^0{4}-0{2}-0{2} 0{2}:0{2}:0{2}$"),
    re.compile(r"^0{4}-0{2}-0{2}T0{2}:0{2}:0{2}"),
    re.compile(r"^9999-12-31"),
)

#: Texts read as true / false. The union of both variants; `t`/`f` come from the
#: PostgreSQL sources, because that is how PG prints booleans.
TRUE_STRINGS: frozenset[str] = frozenset({"true", "1", "yes", "ano", "t"})
FALSE_STRINGS: frozenset[str] = frozenset({"false", "0", "no", "ne", "f"})

#: Default encoding order. The dialect overrides it — see the module docstring.
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1250", "latin-1")

#: Type names treated as date-like. **The union of all four variants.** Every
#: dialect uses its own names (`datetime` in MySQL, `timestamp without time zone`
#: in PostgreSQL, `datetime2`/`smalldatetime`/`datetimeoffset` in MSSQL), so they
#: cannot overlap and the union breaks nothing.
DATE_LIKE_TYPES: frozenset[str] = frozenset(
    {
        "date",
        "datetime",
        "datetime2",
        "smalldatetime",
        "datetimeoffset",
        "timestamp",
        "timestamp without time zone",
        "timestamp with time zone",
    }
)

#: Column-name fragments used to guess date columns when the caller supplies
#: none. Taken over from the predecessor.
_DATE_NAME_HINTS: tuple[str, ...] = ("date", "datum", "cas", "time")

#: Target type names that mean "an integer column". Public because
#: `target_pg._integer_bound_columns` needs the very same list.
INTEGER_TYPES: frozenset[str] = frozenset({"INTEGER", "BIGINT", "SMALLINT", "MEDIUMINT"})
_TIME_TYPES: frozenset[str] = frozenset({"time", "time without time zone", "time with time zone"})

#: The largest integer every smaller integer is still exactly representable in
#: ``float64``. ``float64`` has a 53-bit mantissa, so every integer in
#: ``[-2**53, 2**53]`` survives a round trip through it and ``2**53 + 1`` does
#: not. The boundary is therefore **inclusive**: exactly ``2**53`` is safe.
FLOAT64_EXACT_INTEGER_LIMIT: int = 2**53

#: The name of the ``LOAD_SETTINGS`` key that turns the check below on. Quoted in
#: the error message so whoever reads a failed run knows which key produced it
#: and can switch it back off.
INTEGER_PRECISION_FLAG = "strict_integer_precision"

_TIME_EXACT = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_TIME_WITH_DAYS = re.compile(r"^(\d+ days )?(\d{1,2}:\d{2}:\d{2})$")


# --- Recognising "nothing" ---------------------------------------------------


def is_pd_na(x: Any) -> bool:
    """Is the value missing? The predecessor's ``_is_pd_na``.

    It also has to cope with arrays and lists, on which ``pd.isna`` returns an
    array rather than a boolean. The old version relied on that blowing up inside
    the ``try`` and falling into ``except`` -> ``False``; here it is handled
    explicitly, but the result is the same.

    A single variant everywhere in the predecessor.
    """
    try:
        if x is None or x is pd.NA:
            return True
        result = pd.isna(x)
    except (TypeError, ValueError):
        return False
    return result if isinstance(result, bool) else False


def _is_missing(x: Any) -> bool:
    """A shortcut inside the module. Unifies ``None``, ``NaN`` and ``pd.NA``."""
    return x is None or (isinstance(x, float) and math.isnan(x)) or is_pd_na(x)


# --- Scalar converters -------------------------------------------------------


def to_int_or_none(x: Any) -> Optional[int]:
    """To an integer, otherwise ``None``.

    A non-integral float (``3.5``) yields ``None``, not a rounded value — the
    predecessor's deliberate behaviour.

    A single variant everywhere in the predecessor.
    """
    if x is None or is_pd_na(x):
        return None
    # NOTE: `np.bool_` is a subclass of neither `int` nor `np.integer`, so it does
    # not land here and falls all the way through to `return None`. Python `bool`
    # is a subclass of `int`, so `True` yields 1. Verified by the characterisation
    # test.
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if pd.isna(x):
            return None
        return int(x) if float(x).is_integer() else None
    if isinstance(x, str):
        text = x.strip().lower()
        if text in _NULL_STRINGS:
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return int(value) if value.is_integer() else None
    return None


def to_int_or_na(x: Any) -> Optional[int]:
    """Like ``to_int_or_none``, but it also tries types outside the list.

    The difference is the last branch: anything that is not ``int``, ``float`` or
    ``str`` is still pushed through ``float(x)``. That catches ``Decimal``, for
    instance.

    A single variant everywhere in the predecessor.
    """
    if _is_missing(x):
        return None
    # NOTE: `np.bool_` is a subclass of neither `int` nor `np.integer`, so it does
    # not land here and falls all the way through to `return None`. Python `bool`
    # is a subclass of `int`, so `True` yields 1. Verified by the characterisation
    # test.
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x) if float(x).is_integer() else None
    if isinstance(x, str):
        text = x.strip()
        if text == "" or text.lower() in _NULL_STRINGS:
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return int(value) if value.is_integer() else None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return int(value) if value.is_integer() else None


def to_str_or_none(x: Any) -> Optional[str]:
    """To text; an empty string and ``nan`` become ``None``.

    An integral float is printed without the decimal part (``3.0`` -> ``'3'``),
    otherwise ``3.0`` would reach the target where the source had a whole number.

    A single variant everywhere in the predecessor.
    """
    if _is_missing(x):
        return None
    if isinstance(x, (float, np.floating)) and not pd.isna(x) and float(x).is_integer():
        text = str(int(x))
    else:
        text = str(x)
    stripped = text.strip()
    return None if stripped == "" or stripped.lower() == "nan" else text


def to_bool(x: Any) -> Optional[bool]:
    """To a boolean. Unknown text yields ``None``, not ``False``.

    ``t``/``f`` are in the list because that is how PostgreSQL prints booleans.

    2 variants in the predecessor, 14 lines.
    """
    if _is_missing(x):
        return None
    # `np.bool_` lands neither here nor in the next branch and ends up as `None` —
    # the inherited behaviour, verified by the characterisation test.
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(int(x))
    if isinstance(x, str):
        text = x.strip().lower()
        if text in TRUE_STRINGS:
            return True
        if text in FALSE_STRINGS:
            return False
    return None


def to_date_str(x: Any) -> Optional[str]:
    """To ``YYYY-MM-DD``. Invalid and sentinel dates become ``None``.

    2 variants in the predecessor, 12 lines.
    """
    if _is_missing(x):
        return None
    if isinstance(x, str):
        text = x.strip()
        if text == "" or text in INVALID_DATE_STRINGS:
            return None
        parsed = pd.to_datetime(text, errors="coerce")
        return parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else None
    if hasattr(x, "strftime"):
        return x.strftime("%Y-%m-%d")
    return None


def to_datetime_str(x: Any) -> Optional[str]:
    """To ``YYYY-MM-DD HH:MM:SS``. Invalid and sentinel values become ``None``.

    2 variants in the predecessor, 12 lines.
    """
    if _is_missing(x):
        return None
    if isinstance(x, str):
        text = x.strip()
        if text == "" or text in INVALID_DATETIME_STRINGS:
            return None
        parsed = pd.to_datetime(text, errors="coerce")
        return parsed.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(parsed) else None
    if hasattr(x, "strftime"):
        return x.strftime("%Y-%m-%d %H:%M:%S")
    return None


def to_time_str(x: Any) -> Optional[str]:
    """To ``HH:MM:SS``.

    It also copes with ``timedelta`` (that is how MySQL returns ``TIME``, which
    may exceed 24 hours) and with ``datetime.time``. Only one variant has the
    ``dt_time`` branch; it is a clean superset, so it is taken over.

    2 variants in the predecessor, 26 lines.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    if isinstance(x, str):
        stripped = x.strip()
        if _TIME_EXACT.match(stripped):
            return stripped
        match = _TIME_WITH_DAYS.match(stripped)
        if match:
            return match.group(2)
        try:
            total_seconds = int(pd.to_timedelta(x).total_seconds())
        except (ValueError, TypeError):
            return None
    elif isinstance(x, pd.Timedelta):
        total_seconds = int(x.total_seconds())
    elif isinstance(x, dt_time):
        return f"{x.hour:02d}:{x.minute:02d}:{x.second:02d}"
    elif isinstance(x, (int, float)):
        total_seconds = int(x)
    else:
        return None

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def to_jsonb(x: Any, *, ensure_ascii: bool = True) -> Optional[str]:
    """To text for a ``jsonb`` column.

    ``ensure_ascii`` is a parameter because two PostgreSQL extractors set it to
    ``False`` while the other seven files keep the default ``True``. Unifying that
    is impossible without recomputing ``row_hash``. The default ``True`` follows
    the majority (7 of 9).

    2 variants in the predecessor, 6 lines.
    """
    if _is_missing(x):
        return None
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=ensure_ascii)
    return str(x)


def _fast_temporal_series(series: pd.Series, invalid: frozenset[str], fmt: str) -> Optional[list]:
    """Vectorised conversion of a date column, or ``None`` when it is not possible.

    ``pd.to_datetime`` over a whole series is an order of magnitude faster than
    calling it per element (measured at 3.0 s -> 0.1 s on 100 000 rows), but it is
    **not universally interchangeable**. Verified against the per-element pass:

    - **time zones**: with a mixed set it returns an ``object`` series without
      ``.dt``, so ``.dt.strftime`` fails,
    - **numbers**: ``pd.to_datetime(1234567890)`` is ``1970-01-01``, whereas
      ``to_date_str`` returns ``None`` on a number (it has no ``strftime``).

    The fast path is therefore used only when every non-empty value is text and
    the result comes out as ``datetime64``. Otherwise ``None`` is returned and the
    caller goes element by element.
    """
    non_null = series[series.notna()]
    if len(non_null) and not non_null.map(lambda v: isinstance(v, str)).all():
        return None

    stripped = series.map(lambda v: v.strip() if isinstance(v, str) else v)
    blocked = stripped.isin(list(invalid)) | (stripped == "")
    candidates = stripped.where(~blocked, None)

    try:
        parsed = pd.to_datetime(candidates, errors="coerce")
    except (ValueError, TypeError, OverflowError):
        return None
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        return None

    formatted = parsed.dt.strftime(fmt)
    return [None if pd.isna(v) else v for v in formatted]


def to_date_str_series(series: pd.Series) -> list:
    """The column form of ``to_date_str``. Same result, only faster."""
    fast = _fast_temporal_series(series, INVALID_DATE_STRINGS, "%Y-%m-%d")
    return fast if fast is not None else [to_date_str(v) for v in series]


def to_datetime_str_series(series: pd.Series) -> list:
    """The column form of ``to_datetime_str``. Same result, only faster."""
    fast = _fast_temporal_series(series, INVALID_DATETIME_STRINGS, "%Y-%m-%d %H:%M:%S")
    return fast if fast is not None else [to_datetime_str(v) for v in series]


def fmt_int_for_csv(x: Any) -> Optional[str]:
    """An integer for ``COPY``. The predecessor's ``_fmt_int_for_csv``.

    Mind the last branch: for an unknown type it returns an **empty string**, not
    ``None``. That differs from ``to_int_or_none`` and is deliberate — in CSV for
    ``COPY``, empty means something other than ``\\N``.

    A single variant everywhere in the predecessor.
    """
    if x is None:
        return None
    # `np.bool_` does not land here — see the note at `to_int_or_none`.
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return str(int(x)) if float(x).is_integer() else None
    if isinstance(x, str):
        text = x.strip()
        if text == "":
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return str(int(value)) if value.is_integer() else None
    return ""


def bytes_to_pg_hex(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Binary columns to the hexadecimal literal ``COPY`` understands.

    It has to run **after** the hash is computed (the hash is taken over the raw
    bytes, exactly as MySQL computes it over them) and **instead of** decoding:
    the target column is ``BYTEA``, so binary data must not turn into text.

    Without it the Python ``repr`` (``b'\x01\x02'``) ends up in the CSV and `COPY`
    rejects it with "invalid input syntax for type bytea".
    """
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if series.dtype != object:
            continue
        mask = series.map(lambda value: isinstance(value, (bytes, bytearray, memoryview)))
        if not mask.any():
            continue
        converted = series.copy()
        converted.loc[mask] = ["\\x" + bytes(value).hex() for value in series[mask]]
        df[column] = converted
    return df


def normalize_memoryviews(df: pd.DataFrame) -> pd.DataFrame:
    """``memoryview`` -> ``bytes``. Must run **before** the hash is computed.

    psycopg2 returns ``bytea`` as a ``memoryview`` and ``str()`` turns that into a
    **memory address** (``<memory at 0x7f…>``) which changes on every read. That
    breaks the hash so thoroughly that the seed and the diff never agree and the
    table is transferred forever — on a table with a binary column the hash
    strategy therefore never settles.

    Decoding is not an option: the target column is ``BYTEA``, so binary data has
    to reach the target as binary data. Converting to ``bytes`` is enough —
    ``bytes`` already hashes stably and it is also the shape in which MySQL
    returns binary data.
    """
    for position in range(len(df.columns)):
        series = df.iloc[:, position]
        if series.dtype != object:
            continue
        mask = series.map(lambda value: isinstance(value, memoryview))
        if not mask.any():
            continue
        converted = series.copy()
        converted.loc[mask] = [value.tobytes() for value in series[mask]]
        df.iloc[:, position] = converted
    return df


def decode_value(x: Any, encodings: Sequence[str] = DEFAULT_ENCODINGS) -> Any:
    """Bytes to text. The predecessor's ``_decode_value``.

    Tries the encodings in the given order and returns the first one that works.
    **The order matters critically**: ``cp1250`` is a single-byte encoding that
    rejects only five bytes, so when it comes first, UTF-8 text passes through it
    as mojibake. The order is therefore supplied by the dialect, according to what
    its source really stores.

    A single variant everywhere in the predecessor.
    """
    if not isinstance(x, (bytes, bytearray)):
        return x
    for encoding in encodings:
        try:
            return x.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return x.decode(encodings[-1] if encodings else "latin-1", errors="replace")


def clean_invalid_date(value: Any) -> Any:
    """Invalid date to ``None``, a valid one unchanged. The predecessor's ``_clean``.

    It does more than check the sentinels: it also **tries to parse** the value and
    returns ``None`` when that fails. Text that does not resemble a date at all
    therefore disappears from the column. It looks like a detail, but it is
    essential — if it stayed, `COPY` would fail on it.

    In the predecessor this is an emergency branch inside ``fix_invalid_dates``;
    it is pulled out because it is a function in its own right.

    A single variant everywhere in the predecessor.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or any(p.match(stripped) for p in INVALID_DATE_PATTERNS):
            return None
        try:
            pd.to_datetime(stripped, errors="raise")
        except (ValueError, TypeError, OverflowError, pd.errors.ParserError):
            return None
        # The **stripped** value is returned, not the original one. `'  2026-08-07  '`
        # therefore comes out as `'2026-08-07'`. It feeds into `row_hash`, so this
        # is not cosmetics.
        return stripped
    try:
        pd.to_datetime(value, errors="raise")
    except (ValueError, TypeError, OverflowError, pd.errors.ParserError):
        return None
    return value


# --- Whole-column operations -------------------------------------------------


def _object_series(values: Iterable, index: Any) -> pd.Series:
    """An ``object`` series, so that ``None`` does not flip to ``NaN``.

    Without an explicit ``dtype='object'``, pandas infers ``float64`` for numeric
    values and ``None`` turns into ``NaN``.
    """
    return pd.Series(list(values), index=index, dtype="object")


def decode_bytes_columns(
    df: pd.DataFrame,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    skip: Sequence[str] = (),
) -> pd.DataFrame:
    """Decode the columns that contain bytes.

    ``skip`` leaves out columns whose target is binary (``BYTEA``). Decoding those
    would lose data: binary content would turn into text with replacement
    characters — and that text also holds control characters on which `COPY`
    itself fails ("need to escape, but no escapechar set"). The predecessor does
    not tell the two apart, because nobody has sent binary data into a BYTEA
    column so far.

    Vectorisation: columns without bytes are skipped with a single mask instead of
    a call per element, and only the subset that really holds bytes is decoded. The
    predecessor calls ``.apply()`` twice over every ``object`` column.

    2 variants in the predecessor, 17 lines.
    """
    skipped = set(skip)
    for position in range(len(df.columns)):
        if df.columns[position] in skipped:
            continue
        series = df.iloc[:, position]
        if series.dtype != object:
            continue
        mask = series.map(lambda value: isinstance(value, (bytes, bytearray)))
        if not mask.any():
            continue
        decoded = series.copy()
        decoded.loc[mask] = [decode_value(value, encodings) for value in series[mask]]
        df.iloc[:, position] = decoded
    return df


def replace_pdna_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``pd.NA`` with ``None``.

    **Mind the trap this function carries over unchanged:** in numeric columns
    ``None`` does not survive. The values pass through Python and pandas infers the
    type again — ``float64`` turns ``None`` back into ``NaN`` and ``Int64`` even
    flips to ``float64``, so precision is lost on large integers.

    Kept deliberately. Fixing it would change ``row_hash`` for every table with a
    nullable numeric column — that belongs to a conscious decision, not to a silent
    repair.

    **And fixing it here alone would fix nothing.** This is the *second* place the
    precision goes, not the first: ``pd.read_sql`` already infers ``float64`` for
    any integer column that contains a NULL, so a ``BIGINT`` above 2**53 arrives
    rounded before this function ever sees it. Measured end to end on all four
    dialects — a source value of ``9007199254740993`` lands in the target as
    ``9007199254740992``. The predecessor reads exactly the same way, so the loss
    is inherited, and the whole reasoning is pinned in
    ``tests/coerce/test_int64_precision.py``.

    Vectorising it through ``Series.where`` was tried, which does preserve the
    dtype of ``object`` columns — but that is precisely what makes the result
    differ from the inherited one. Hence the per-element pass stays here.

    A single variant everywhere in the predecessor.
    """
    for column in df.columns:
        series = df[column]
        df[column] = series.map(lambda value: None if is_pd_na(value) else value)
    return df


def normalize_carriage_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify the line endings. Required before ``COPY``.

    **Mind the trap:** ``\\r\\n`` becomes ``\\n``, but a lone ``\\r`` becomes a
    **space**, not ``\\n``. It looks like a typo in the predecessor, but it is
    consistent across all of its files and it changes the cell contents — hence
    also ``row_hash``. Kept unchanged.

    Vectorisation: a column that holds no ``\\r`` at all is skipped with a single
    mask. The predecessor does the same.

    A single variant everywhere in the predecessor.
    """
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        if not series.dropna().astype(str).str.contains("\r").any():
            continue
        replaced = series.map(_fix_carriage_returns)
        if not replaced.equals(series):
            df[column] = replaced
    return df


def _fix_carriage_returns(value: Any) -> Any:
    """The predecessor's ``_fix``. See the note at ``normalize_carriage_returns``.

    A single variant everywhere in the predecessor.
    """
    if value is None:
        return None
    if isinstance(value, str) and "\r" in value:
        return value.replace("\r\n", "\n").replace("\r", " ")
    return value


def fix_invalid_dates(df: pd.DataFrame, date_like_columns: Optional[set] = None) -> pd.DataFrame:
    """Invalid and sentinel dates to ``None``.

    This is not a plain sentinel substitution but a **four-step pass**, and the
    order of the steps matters:

    1. sentinels matching the patterns (``0000-00-00``, ``9999-12-31…``) to
       ``None``,
    2. whatever ``pd.to_datetime`` cannot parse, to ``None``,
    3. substitution of the remaining sentinels including ``'0'`` and ``''`` (step 2
       leaves those alone, because ``'0'`` *can* be parsed as a date),
    4. cleanup of ``pd.NA`` and ``NaN``.

    **Step 2 is vectorised deliberately.** ``pd.to_datetime`` over a whole series
    is two orders of magnitude faster than calling it per element — measured at
    0.32 s vs 8.9 s on 100 000 rows. The predecessor does the same; there the
    per-element version is only an emergency branch in an ``except``, which is what
    ``clean_invalid_date`` serves.

    When ``date_like_columns`` is not supplied, the columns are guessed from their
    names.

    3 variants in the predecessor, 35 lines. The winner is the **union** of the
    sentinels; the variants differed only in which ones they knew. Agreed.
    """
    if not date_like_columns:
        date_like_columns = {
            column
            for column in df.columns
            if any(hint in str(column).lower() for hint in _DATE_NAME_HINTS)
        }

    for column in df.columns:
        if column not in date_like_columns:
            continue

        series = df[column].map(_strip_sentinel_dates)

        try:
            parsed = pd.to_datetime(series, errors="coerce")
            keep = parsed.notna() | series.isna() | (series.astype(str).str.strip() == "")
            series = series.where(keep, None)
        except (ValueError, TypeError, OverflowError) as err:
            # A column that cannot be parsed as a whole (mixed types, for example).
            # Falls back to the per-element path — slower, but reliable.
            _log_fallback(column, err)
            series = series.map(clean_invalid_date)

        series = series.replace(dict.fromkeys(_REPLACED_DATE_VALUES, None))
        df[column] = series.map(lambda value: None if _is_missing(value) else value)
    return df


def _strip_sentinel_dates(value: Any) -> Any:
    """Step 1: sentinel text to ``None``, everything else unchanged."""
    if isinstance(value, str) and any(p.match(value.strip()) for p in INVALID_DATE_PATTERNS):
        return None
    return value


def _log_fallback(column: Any, err: Exception) -> None:
    """A swallowed exception has to end up in the log."""
    import logging

    logging.getLogger(__name__).warning(
        "⚠️ Column %s cannot be parsed as a whole (%s), falling back to per-element handling.",
        column,
        err,
    )


#: Step 3. Besides the sentinels it holds `'0'` and `''` — step 2 lets those pass,
#: because `pd.to_datetime('0')` *can* be parsed.
_REPLACED_DATE_VALUES: tuple[str, ...] = (
    "0000-00-00",
    "0000-00-00 00:00:00",
    "0000-00-00T00:00:00",
    "9999-12-31",
    "9999-12-31 00:00:00",
    "9999-12-31 23:59:59",
    "9999-12-31T00:00:00",
    "9999-12-31T23:59:59",
    "0",
    "",
)


def convert_time_columns(
    df: pd.DataFrame, overwrite_types: dict, orig_type_map: Optional[dict] = None
) -> pd.DataFrame:
    """``TIME`` columns to ``HH:MM:SS``.

    The type names are unified across the dialects — a superset of both variants.

    2 variants in the predecessor, 36 lines.
    """
    if not overwrite_types:
        return df

    for column, target_type in overwrite_types.items():
        original = str((orig_type_map or {}).get(column, "")).lower()
        if target_type != "TIME" and original not in _TIME_TYPES:
            continue
        if column not in df.columns:
            continue
        series = df[column]
        df[column] = _object_series((to_time_str(value) for value in series), series.index)
    return df


def convert_integer_columns(df: pd.DataFrame, dtype_mapping: dict) -> pd.DataFrame:
    """Convert columns mapped to an integer type to ``Int64``.

    ``Int64`` (capital I) is nullable, so ``None`` survives — unlike ``int64``,
    where the column would have to flip to ``float64``.

    A single variant everywhere in the predecessor.
    """
    for column, target_type in (dtype_mapping or {}).items():
        if column not in df.columns:
            continue
        if str(target_type).upper() not in INTEGER_TYPES:
            continue
        series = df[column]
        df[column] = pd.array([to_int_or_na(value) for value in series], dtype="Int64")
    return df


class IntegerPrecisionError(ValueError):
    """A value bound for an integer column is outside the exact ``float64`` range.

    Raised only when ``LOAD_SETTINGS.strict_integer_precision`` is on. With the
    key off — the default, and the state of every existing pipeline — the value
    is written rounded, exactly as the predecessor writes it.
    """


def check_integer_precision(df: pd.DataFrame, integer_columns: Iterable[str]) -> None:
    """Fail loudly instead of writing an integer that ``float64`` already rounded.

    An integer column that also holds a NULL cannot be ``int64``, so pandas
    infers ``float64`` — already inside ``pd.read_sql``, before any code of this
    package runs — and ``float64`` carries 53 bits of mantissa. A source value of
    ``9007199254740993`` therefore lands in the target as ``9007199254740992``,
    with a correct row count and a green run. `replace_pdna_with_none` is the
    second place the same thing happens, to a column that arrived intact.

    **The rounding itself is not fixed and must not be**: the predecessor loses
    the very same bits in the very same place, so correcting the read would break
    golden parity and recompute ``row_hash`` for every table with a nullable
    numeric column. The whole reasoning is pinned in
    ``tests/coerce/test_int64_precision.py``. This function does not repair
    anything — it only refuses to write a number nobody can trust.

    **The check is deliberately conservative.** It flags every ``|value|`` above
    `FLOAT64_EXACT_INTEGER_LIMIT`, not only the values that really were rounded:
    ``2**53 + 2`` is representable exactly, yet it is reported too. Telling the
    two apart is impossible after the fact — the rounded and the intact value are
    the same ``float64`` — so the safe side is the loud one. The boundary is
    inclusive: exactly ``2**53`` passes, ``2**53 + 1`` does not.

    Vectorised on purpose. Throughput here is per row *times* column, so the scan
    is one ``numpy`` comparison over the whole column and columns that are not
    floats are skipped by their dtype alone. The caller must not call this at all
    when the flag is off — see `target_pg.prepare_export_df`.
    """
    for column in integer_columns or ():
        if column not in df.columns:
            continue
        series = df[column]
        # Only float columns can carry the loss. ``int64``/``Int64`` and object
        # columns of Python ints hold the value exactly, whatever its size.
        if not pd.api.types.is_float_dtype(series.dtype):
            continue
        if isinstance(series.dtype, np.dtype):
            # The ordinary case, and no copy: this is exactly what the read
            # produces for an integer column that holds a NULL.
            values = series.to_numpy(dtype="float64", copy=False)
        else:
            # A nullable extension float (``Float64``). ``pd.NA`` cannot go into a
            # numpy array, so it has to be spelled as ``NaN`` on the way out.
            values = series.to_numpy(dtype="float64", na_value=np.nan)
        # ``NaN`` compares false, so nulls need no separate mask.
        unsafe = np.abs(values) > FLOAT64_EXACT_INTEGER_LIMIT
        if not unsafe.any():
            continue
        offender = values[unsafe][0]
        raise IntegerPrecisionError(
            f"Column '{column}' is bound for an integer column in the target, but it "
            f"arrived as float64 and holds {offender:.0f}, whose absolute value "
            f"exceeds 2**53 ({FLOAT64_EXACT_INTEGER_LIMIT}). float64 has a 53-bit "
            f"mantissa, so the value has most likely already been rounded and the "
            f"number written to the target would differ from the source. Caused by "
            f"a NULL in an integer column, which makes pandas read the whole column "
            f"as float64. Reported because LOAD_SETTINGS.{INTEGER_PRECISION_FLAG} "
            f"is on; with it off the value is written rounded, as it has always been."
        )


def sanitize_integer_columns(
    df: pd.DataFrame, integer_columns: Iterable[str], show_debug: bool = False
) -> pd.DataFrame:
    """Sanitise the values in integer columns.

    The resulting column is deliberately ``object``, not ``Int64`` — that is what
    the predecessor does, and how a value is printed into ``COPY`` depends on the
    dtype.

    A single variant everywhere in the predecessor.
    """
    for column in integer_columns or ():
        if column not in df.columns:
            continue
        series = df[column]
        df[column] = _object_series((to_int_or_none(value) for value in series), series.index)
    return df


def extract_date_columns(original_column_types: Any) -> set:
    """Pick the columns that are to be treated as date-like.

    Accepts both a DataFrame of definitions and a ``name -> type`` dict.

    4 variants in the predecessor, 15 lines. The winner is the **union** of the
    type names; the variants differed only in which dialect names they knew, and
    those do not overlap with each other.
    """
    found: set = set()
    if original_column_types is None:
        return found

    for column, data_type in _iter_column_types(original_column_types):
        if column and str(data_type).lower() in DATE_LIKE_TYPES:
            found.add(column)
    return found


def _iter_column_types(definitions: Any) -> Iterable[tuple[Any, Any]]:
    """Unify the two shapes of column definitions the predecessor uses."""
    if isinstance(definitions, pd.DataFrame):
        for _, row in definitions.iterrows():
            column = row.get("normalized_column_name") or row.get("original_column_name")
            yield column, row.get("data_type")
    else:
        yield from dict(definitions).items()


def create_safe_dtype_mapping_from_definitions(
    all_columns: Sequence[str],
    original_column_types: Any,
    type_map: Optional[dict] = None,
    show_debug: bool = False,
) -> dict:
    """Turn column definitions into a ``column -> target type`` map.

    ``type_map`` is **a parameter, not a constant**. The variants differed in
    exactly one thing: which map they consulted (``MYSQL_TO_PG_TYPES``,
    ``MSSQL_TO_PG_TYPES``, ``PG_TO_PG_TYPES``) — and that is dialect knowledge,
    which per ARCHITECTURE.md does not belong in the core. ``SourceDialect.map_type``
    supplies it.

    Without a map the type is used as it arrived — the behaviour of the fourth
    variant.

    4 variants in the predecessor, 20 lines.
    """
    mapping: dict = {}
    if original_column_types is None:
        return mapping

    known = set(all_columns or ())
    for column, data_type in _iter_column_types(original_column_types):
        if not column or (known and column not in known):
            continue
        if data_type is None or str(data_type) in ("nan", "None", "", "generated"):
            continue
        mapping[column] = (
            type_map.get(str(data_type).lower(), "TEXT") if type_map else str(data_type)
        )
    return mapping


def fix_column_values(
    df: pd.DataFrame,
    safe_mapping: dict,
    show_debug: bool = False,
    *,
    ensure_ascii: bool = True,
) -> pd.DataFrame:
    """Convert the values according to the target type of each column.

    **In the predecessor this is 98 lines**, because it carries inline copies of
    ``to_bool``, ``to_date_str``, ``to_datetime_str`` and ``to_jsonb``. Those
    copies are exactly why it has three variants — they differ precisely in them.
    Here the already decided scalar functions are called, so the decision is made
    once and in one place. Agreed.

    Four things that look like details and are not:

    1. **The loop goes over the frame's columns, not over the map.** A column that
       is not in the map gets ``'TEXT'`` and goes through the last branch — that
       is, it is converted to a string. Iterating over the map instead would leave
       it untouched and a different value would end up in the target (``2.0``
       instead of ``2``).
    2. **The type is compared as it arrived** — without ``.upper()``. A type
       written in lower case falls into the last branch, not into its own.
    3. **``TIME`` is not handled here**, `convert_time_columns` does it. Handling it
       in both places would hand the second pass an already formatted string.
    4. **Integer columns end up as ``Int64``**, not as ``object``.
       `sanitize_integer_columns` and `fmt_int_for_csv` build on that.

    Identical in all 9 files that have the function (verified by the
    characterisation test over the frozen variants).

    3 variants in the predecessor, 98 lines.
    """
    mapping = safe_mapping or {}
    for column in df.columns:
        column_type = mapping.get(column, "TEXT")
        series = df[column]

        if column_type in INTEGER_TYPES:
            df[column] = pd.array([to_int_or_na(v) for v in series], dtype="Int64")
            continue
        if column_type == "TIME":
            continue

        converted: Iterable
        if column_type == "DATE":
            converted = to_date_str_series(series)
        elif column_type in ("DATETIME", "TIMESTAMP"):
            converted = to_datetime_str_series(series)
        elif column_type == "BOOLEAN":
            converted = (to_bool(v) for v in series)
        elif column_type == "JSONB":
            converted = (to_jsonb(v, ensure_ascii=ensure_ascii) for v in series)
        else:
            converted = (to_str_or_none(v) for v in series)

        df[column] = _object_series(converted, series.index)
    return df


def update_stats(col_name: str, series: Any, all_columns: set, col_stats: dict) -> None:
    """Column statistics from which the **type is promoted** after the first batch.

    A single variant across 7 files. This is not mere telemetry: the promotion
    ``INTEGER`` -> ``BIGINT`` rests on `int_max`, and the promotion ``enum(0,1)``
    -> ``BOOLEAN`` on the trio ``seen_integer``/``int_min``/``int_max`` (see
    `strategies.full`).

    ``all_columns`` is **filled**, not read — it is an output parameter. (The
    original implementation had the condition ``if col_name not in all_columns:
    return`` here, which is the inverted meaning: with an empty set on input
    nothing was ever counted and the type promotion silently never happened.)

    ``has_non_numeric`` takes precedence: as soon as a non-numeric value is found
    in the column, statistics are no longer collected and the type is not promoted.
    """
    all_columns.add(col_name)
    stats = col_stats.setdefault(
        col_name,
        {
            "id_like": any(
                part in col_name.lower() for part in ["id", "_id", "key", "_key", "id_", "key_"]
            ),
            "seen_float": False,
            "seen_integer": False,
            "int_min": None,
            "int_max": None,
            "has_non_numeric": False,
        },
    )

    nonnull = series.dropna()
    if nonnull.empty:
        return

    numeric_like = nonnull.map(_is_numeric_like)
    if not numeric_like.all():
        stats["has_non_numeric"] = True
        return

    num = pd.to_numeric(nonnull, errors="coerce")
    if not num.notna().any():
        return

    if (num % 1 != 0).any():
        stats["seen_float"] = True
        return

    stats["seen_integer"] = True
    vmin, vmax = int(num.min()), int(num.max())
    stats["int_min"] = vmin if stats["int_min"] is None else min(stats["int_min"], vmin)
    stats["int_max"] = vmax if stats["int_max"] is None else max(stats["int_max"], vmax)


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return isinstance(value, str) and value.replace(".", "", 1).isdigit()
