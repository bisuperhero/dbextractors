"""Serialising calls to the predecessor functions — storage for `reference_oracle`.

The characterisation tests need the **old** functions, but their sources are
not part of this repository. Without them three test files would not collect at
all and another 154 tests would fail.

The answer is record and replay. Where the sources exist, calls are recorded
(input -> output) into fixtures. Where they do not, the old function is replaced
by a replayer that finds the recorded answer for the same input.

**The format is JSON, not pickle**, even though pickle would be shorter. Three
reasons: the fixtures can be read and reviewed by eye, a diff shows what
changed, and loading them does not execute anything.

Type fidelity is a requirement here, not a nicety: the tests compare through
``pd.testing.assert_frame_equal``, which checks ``dtype`` as well. An ``Int64``
coming back as ``int64`` fails on a difference that was never in the old
function.
"""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "oracle"
INDEX_FILE = FIXTURES / "index.json"

#: The key carrying the type tag in the encoded form.
TAG = "__t__"


# --- encoding ----------------------------------------------------------------


def encode(value: Any) -> Any:
    """A value in a form `json.dumps` accepts and `decode` can restore."""
    # The order matters: `bool` subclasses `int`, `np.bool_` is not `bool`,
    # and `pd.NA` / `NaT` must never reach `math.isnan`.
    if value is None:
        return {TAG: "none"}
    if value is pd.NaT:
        return {TAG: "nat"}
    if value is pd.NA:
        return {TAG: "na"}
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, np.integer):
        return value
    if isinstance(value, float):
        # JSON only knows NaN and Infinity as an extension that other readers
        # reject, so they get a tag instead.
        if math.isnan(value):
            return {TAG: "nan"}
        if math.isinf(value):
            return {TAG: "inf", "v": 1 if value > 0 else -1}
        return value
    # `bytes` and `bytearray` must carry **different** tags even though the
    # bytes are the same. Sharing a tag gives them the same call fingerprint,
    # so the recordings overwrite each other and replaying `b'abc'` returns the
    # answer that belonged to `bytearray(b'abc')`. The old `to_jsonb` tells them
    # apart (`json.dumps` fails and the fallback does `str(x)`), which is how
    # this was found.
    if isinstance(value, bytearray):
        return {TAG: "bytearray", "v": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, bytes):
        return {TAG: "bytes", "v": base64.b64encode(value).decode("ascii")}
    # A function passed as an argument (`with_retry(operation, …)`). The
    # default `repr` carries a memory address, so the fingerprint would differ
    # between runs and replay would never match. The qualified name is stable.
    if callable(value) and not isinstance(value, type):
        return {
            TAG: "callable",
            "name": getattr(value, "__qualname__", getattr(value, "__name__", "?")),
        }
    if isinstance(value, np.bool_):
        return {TAG: "np", "dtype": "bool", "v": bool(value)}
    if isinstance(value, np.integer):
        return {TAG: "np", "dtype": str(value.dtype), "v": int(value)}
    if isinstance(value, np.floating):
        return {TAG: "np", "dtype": str(value.dtype), "v": encode(float(value))}
    if isinstance(value, decimal.Decimal):
        return {TAG: "decimal", "v": str(value)}
    if isinstance(value, pd.Timestamp):
        return {TAG: "timestamp", "v": value.isoformat(), "tz": str(value.tz) if value.tz else None}
    if isinstance(value, dt.datetime):
        return {TAG: "datetime", "v": value.isoformat()}
    if isinstance(value, dt.date):
        return {TAG: "date", "v": value.isoformat()}
    if isinstance(value, dt.time):
        return {TAG: "time", "v": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {TAG: "timedelta", "v": value.total_seconds()}
    if isinstance(value, pd.DataFrame):
        return _encode_frame(value)
    if isinstance(value, pd.Series):
        return _encode_series(value)
    if isinstance(value, pd.Index):
        return {TAG: "index", "dtype": str(value.dtype), "v": [encode(x) for x in value]}
    if isinstance(value, np.ndarray):
        return {TAG: "ndarray", "dtype": str(value.dtype), "v": [encode(x) for x in value.tolist()]}
    if isinstance(value, tuple):
        return {TAG: "tuple", "v": [encode(x) for x in value]}
    # A set has no order, but the call fingerprint must be stable, so the
    # encoded elements are sorted. `key=repr`, because an encoded element may
    # be a dict, and dicts cannot be compared to each other.
    if isinstance(value, frozenset):
        return {TAG: "frozenset", "v": sorted((encode(x) for x in value), key=repr)}
    if isinstance(value, set):
        return {TAG: "set", "v": sorted((encode(x) for x in value), key=repr)}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        # Keys need not be strings, so pairs are stored rather than an object.
        return {TAG: "dict", "v": [[encode(k), encode(v)] for k, v in value.items()]}
    # Last resort: store at least the type and repr. The original value cannot
    # be restored from it, but the fixtures still show what was there, and
    # `decode` says so clearly instead of failing on a KeyError.
    return {TAG: "opaque", "type": type(value).__name__, "repr": repr(value)}


def _encode_series(series: pd.Series) -> dict:
    return {
        TAG: "series",
        "name": series.name,
        "dtype": str(series.dtype),
        "index": [encode(x) for x in series.index],
        "v": [encode(x) for x in series.tolist()],
    }


def _encode_frame(frame: pd.DataFrame) -> dict:
    return {
        TAG: "frame",
        "index": [encode(x) for x in frame.index],
        "index_name": frame.index.name,
        "columns": [
            {
                "name": encode(name),
                "dtype": str(frame[name].dtype),
                "v": [encode(x) for x in frame[name].tolist()],
            }
            for name in frame.columns
        ],
    }


class DecodeError(RuntimeError):
    """An encoded value cannot be restored."""


def decode(value: Any) -> Any:
    """The inverse of `encode`."""
    if isinstance(value, list):
        return [decode(x) for x in value]
    if not isinstance(value, dict):
        return value

    tag = value.get(TAG)
    if tag is None:
        raise DecodeError(f"Object without a {TAG!r} tag: {value!r}")

    if tag == "none":
        return None
    if tag == "nat":
        return pd.NaT
    if tag == "na":
        return pd.NA
    if tag == "nan":
        return float("nan")
    if tag == "inf":
        return float("inf") * value["v"]
    if tag == "bytes":
        return base64.b64decode(value["v"])
    if tag == "bytearray":
        return bytearray(base64.b64decode(value["v"]))
    if tag == "callable":
        raise DecodeError(
            f"The function {value['name']!r} cannot be replayed — it cannot be "
            "restored from a fingerprint. A test that passes a function as an "
            "argument belongs in reference_oracle.LIVE_ONLY."
        )
    if tag == "np":
        return np.dtype(value["dtype"]).type(decode(value["v"]))
    if tag == "decimal":
        return decimal.Decimal(value["v"])
    if tag == "timestamp":
        return pd.Timestamp(value["v"])
    if tag == "datetime":
        return dt.datetime.fromisoformat(value["v"])
    if tag == "date":
        return dt.date.fromisoformat(value["v"])
    if tag == "time":
        return dt.time.fromisoformat(value["v"])
    if tag == "timedelta":
        return dt.timedelta(seconds=value["v"])
    if tag == "tuple":
        return tuple(decode(x) for x in value["v"])
    if tag == "set":
        return {decode(x) for x in value["v"]}
    if tag == "frozenset":
        return frozenset(decode(x) for x in value["v"])
    if tag == "dict":
        return {decode(k): decode(v) for k, v in value["v"]}
    if tag == "index":
        return pd.Index([decode(x) for x in value["v"]], dtype=value["dtype"])
    if tag == "ndarray":
        return np.array([decode(x) for x in value["v"]], dtype=value["dtype"])
    if tag == "series":
        return _decode_series(value)
    if tag == "frame":
        return _decode_frame(value)
    if tag == "opaque":
        raise DecodeError(
            f"A value of type {value['type']} could not be serialised "
            f"({value['repr']}). Add support for it in tests/oracle_store.py."
        )
    raise DecodeError(f"Unknown tag {tag!r}.")


def _rebuild(values: list, dtype: str, index: Any = None) -> pd.Series:
    """A Series with its dtype **preserved**.

    Through `astype` rather than `pd.Series(..., dtype=…)`, because for an
    empty list the latter quietly falls back to `object` for some dtypes.
    """
    series = pd.Series(values, index=index, dtype="object")
    if dtype == "object":
        return series
    try:
        return series.astype(dtype)
    except (TypeError, ValueError) as err:
        raise DecodeError(f"Series cannot be restored to dtype {dtype!r}: {err}") from err


def _decode_series(value: dict) -> pd.Series:
    index = pd.Index([decode(x) for x in value["index"]])
    series = _rebuild([decode(x) for x in value["v"]], value["dtype"], index)
    series.name = value["name"]
    return series


def _decode_frame(value: dict) -> pd.DataFrame:
    index = pd.Index([decode(x) for x in value["index"]])
    data = {}
    for column in value["columns"]:
        data[decode(column["name"])] = _rebuild(
            [decode(x) for x in column["v"]], column["dtype"], index
        )
    frame = pd.DataFrame(data, index=index)
    frame.index.name = value["index_name"]
    return frame


# --- call fingerprint ------------------------------------------------------------


def fingerprint(args: tuple, kwargs: dict) -> str:
    """A stable key for one call.

    It has to be identical when recording and when replaying, so it is computed
    from the **encoded** form — not from `repr`, which for a DataFrame changes
    with terminal width and pandas display settings.
    """
    payload = json.dumps(
        {"args": encode(list(args)), "kwargs": encode(kwargs)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- files -----------------------------------------------------------------


def path_for(short: str, function: str) -> Path:
    return FIXTURES / short / f"{function}.json"


def load(short: str, function: str) -> dict:
    """Recorded calls for one function. An empty dict when there are none."""
    path = path_for(short, function)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save(short: str, function: str, calls: dict) -> None:
    path = path_for(short, function)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calls, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_index() -> dict:
    if not INDEX_FILE.is_file():
        return {"variants": {}}
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def save_index(index: dict) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(
        json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
