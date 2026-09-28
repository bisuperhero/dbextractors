"""Golden ``row_hash`` values: the hashes themselves, hard-coded.

`test_hashing.py` compares the vectorised path with the row-wise ``apply`` — but
both run on whatever pandas is installed, so a change in pandas that moves both
the same way passes it unnoticed. This file does not compare two computations;
it compares one computation with **numbers**.

Every expected hash below was computed by dbextractors v1.0.2 under the stack it
was released for (Python 3.10, pandas 1.5.3, numpy 1.26.4). Those are the hashes
sitting in ``row_hash`` of roughly 670 production tables, and a hash-based load
treats any row whose hash differs as changed. So the rule for this file is:

**Never regenerate these values.** If a test here fails, the code is wrong, not
the constant — whatever pandas, numpy or Python version it runs on.

The frames are frozen for the same reason: changing one changes its hashes.
Non-ASCII values are written as escapes so the file stays ASCII.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from dbextractors.core import hashing


def _kitchen_sink() -> pd.DataFrame:
    """One column per type a source hands pandas, missing values included.

    Contains ``object`` columns, so the row type is ``object`` and no promotion
    happens — each column is rendered on its own.
    """
    return pd.DataFrame(
        {
            "i": [1, -2, 3, 0],
            "ni": pd.array([10, None, -30, 2**53 + 1], dtype="Int64"),
            "f": [1.5, np.nan, -0.1, 1e20],
            "dec": [Decimal("1.50"), Decimal("-3.25"), None, Decimal("1E+3")],
            "b": [b"abc", b"", None, b"\xc5\xa1\x00"],
            "s": ["př\xedliš žluťoučk\xfd kůň", "", None, "a||b"],
            "d": [dt.date(2026, 8, 11), None, dt.date(1900, 1, 1), dt.date(2000, 2, 29)],
            "ts": pd.Series(
                [
                    dt.datetime(2026, 8, 11, 12, 34, 56, 789000),
                    None,
                    dt.datetime(2000, 1, 1),
                    dt.datetime(1999, 12, 31, 23, 59, 59),
                ],
                dtype="datetime64[ns]",
            ),
            "midnight": pd.to_datetime(["2026-08-11", "2020-01-01", None, "1999-12-31"]),
            "td": pd.to_timedelta(["1 days 02:03:04", None, "0s", "-1s"]),
            "flag": [True, False, True, False],
            "nothing": [None, None, None, None],
        }
    )


FRAMES = {
    "kitchen sink": _kitchen_sink,
    # int64 + float64 promotes the row to float64: `1` is hashed as `1.0`.
    "promoted to float": lambda: pd.DataFrame(
        {"id": [1, 2, 3], "cena": [1.5, np.nan, -2.25], "pocet": [0, 7, -1]}
    ),
    # Nullable integers next to numpy ones, with and without a missing value.
    "Int64 without NA + float64": lambda: pd.DataFrame(
        {"ni": pd.array([1, 2, 3], dtype="Int64"), "f": [0.5, 1.0, 2.5]}
    ),
    "Int64 with NA + int64": lambda: pd.DataFrame(
        {"ni": pd.array([1, None, 3], dtype="Int64"), "i": [4, 5, 6]}
    ),
    "Int64 without NA + int64": lambda: pd.DataFrame(
        {"ni": pd.array([1, 2, 3], dtype="Int64"), "i": [4, 5, 6]}
    ),
    "bool + int64": lambda: pd.DataFrame({"flag": [True, False, True], "i": [1, 2, 3]}),
    "float32 + int32": lambda: pd.DataFrame(
        {
            "f": np.array([0.1, 1.5, -2.0], dtype=np.float32),
            "i": np.array([1, 2, 3], dtype=np.int32),
        }
    ),
    "bytes only": lambda: pd.DataFrame({"b": [b"abc", b"\xff\xfe", None]}),
}

#: `compute_row_hashes(df, all columns)` — the frame's own row type applies.
EXPECTED_COMPUTE = {
    "Int64 with NA + int64": [
        "df491782f7766bdee3f6abd0ae2cc6a933afc166bededaa3ba40a7aa69509c99",
        "7aae993ef8b9c0da120da9d56ac86e9d8f0a05143c5254ec15ee1703f641b2d7",
        "693c17f4fd3a7f0b7202e17f63293d8e5e3b0c4c6aeb86b0b96194c8eac199bf",
    ],
    "Int64 without NA + float64": [
        "00b5cd92da382b5453bd018ed05e9140f0e78f609209f7d8ea0e2d254b8cfea5",
        "8aa2ecf9f211700226ef85d9b3a609bbdbb734dd64d1d0ac7716994fc98bb34e",
        "0f8fd841a1d1ac9ec1ef3acae69cec2faf4b56a70803fd9103996ca7033a84d9",
    ],
    "Int64 without NA + int64": [
        "df491782f7766bdee3f6abd0ae2cc6a933afc166bededaa3ba40a7aa69509c99",
        "f42f565f8e65c19b3e311e53ac0a0563fa304644d039088defe3cf8c19657d8a",
        "693c17f4fd3a7f0b7202e17f63293d8e5e3b0c4c6aeb86b0b96194c8eac199bf",
    ],
    "bool + int64": [
        "2e04de001e51f4a55b94e5d1c6f62aa0b1d11bc5affce97c8d346779ebb7eda9",
        "ad99bd7651ced28b27763410e32466f1d51ebdda3b20b5dd57d47f49175370c6",
        "c737e2d77d9a4727468c6ed140c595b0b4a57c2841d4133f59487e23ab2b68a5",
    ],
    "bytes only": [
        "a5525591b56dd3716b7bfcaca8e1bf0800efd81b6d8abb807624aad9ebd8c4ea",
        "86e01160baad6cbe506c22055b96f58c0b5c2c0c877025a4100f850b33a3ab9d",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ],
    "float32 + int32": [
        "6a61a74287cf30ab5920eb757f08e1083ebd84ee0fb3380bf0074ceb8f72b261",
        "7f500599a86ef6197a05f4c9abac89024dd4e1333d800f2dd3a9d6c41621f7fe",
        "bfb4223703cbafb8861673d79fb4ece9ad5317931ef4f63234817446cfcc194b",
    ],
    "kitchen sink": [
        "240ec72e5b4dfab5c76a83eff949ade9083c72cee1566d0edd8ab6c1c2d0da7f",
        "61207c01d52475c3aefc6bcb45dd03c9dcbb84971e1d4e8c126c9cb02d243111",
        "254bf2a3add5799a00ec39cb30e891c7fc3a011763f38ea4afcfb420b3dcd6f6",
        "d9adc1bfa1d8bb6c824bdf067c444961a90c76e4a1c27e5bc8f68e9ee1f3eb8f",
    ],
    "promoted to float": [
        "960ef14d3df57305ca978d119754a4b8e4d93013c7dbd7e0372c15f361231d90",
        "ecf49cebe61747cad77b92d47f5946e4df9950fe3edaa8fbd06fd15ea076f468",
        "a7e89ab4d52f649d5477b38ea3ed40dbeee30a93d668aca5b390582af1c2bd40",
    ],
}

#: `add_hash_and_timestamp` — the `object` hash column switches promotion off,
#: which is why "promoted to float" and "float32 + int32" differ from above.
EXPECTED_ADD = {
    **EXPECTED_COMPUTE,
    "float32 + int32": [
        "76ac2d9d9fc0480fd62ef59f6c3a85f6874724d34ccfd959d0c4f8290ea0e0d5",
        "47c07661bd32ffff62bb8cf223e442bcbd376fc8f4bc0f1c6a7b2820f3abe2fc",
        "185e79404f1e6902b322b33ca4b353336640ac163a005b7d0585a67730aae7aa",
    ],
    "promoted to float": [
        "9ff7f2f9d8bcabfc534cea719cb648a8fe095093562945fc06287e258f32b965",
        "2b4a980476ccacc3d017c46ca56e90b1745b0b032e74070415401f223627b364",
        "3b3e6f923178fdebf838a33a51b7c2c11f038599799876c7d3d8c6f9046c94e9",
    ],
}


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_compute_row_hashes_matches_the_released_hashes(name: str) -> None:
    df = FRAMES[name]()
    got = hashing.compute_row_hashes(df, list(df.columns)).tolist()
    assert got == EXPECTED_COMPUTE[name]


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_add_hash_and_timestamp_matches_the_released_hashes(name: str) -> None:
    """The path a run takes: the hash column is missing, so it is created first
    (as ``object``, which switches type promotion off) and then filled."""
    df = hashing.add_hash_and_timestamp(FRAMES[name](), "row_hash")
    assert df is not None
    assert df["row_hash"].tolist() == EXPECTED_ADD[name]


def test_row_hash_matches_the_released_hash() -> None:
    """The one-row reference implementation over the same kinds of value."""
    values = [
        1,
        np.int64(-2),
        1.5,
        np.nan,
        Decimal("1.50"),
        b"abc",
        "př\xedliš žluťoučk\xfd",
        dt.date(2026, 8, 11),
        dt.datetime(2026, 8, 11, 12, 34, 56, 789000),
        pd.Timestamp("2026-08-11"),
        dt.timedelta(days=1, seconds=3784),
        True,
        None,
        pd.NA,
        pd.NaT,
    ]
    assert (
        hashing.row_hash(values)
        == "121d37d86076e3bebe987c53d03a352589162481adc07faa0e3143161fb5819f"
    )
