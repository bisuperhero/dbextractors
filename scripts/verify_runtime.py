"""Verify that the runtime matches the production Mage image.

An earlier attempt built on the `dlt` library failed because it targeted a newer
ecosystem than the one actually available. This script is the wall that stops the
same mistake happening again: it runs in CI and locally via `make verify-runtime`,
and exits with a non-zero status on any mismatch.

The expected versions are not guesses — they are read off a running
`mageai/mageai:0.9.79` container (`pip list` inside the image).
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

# Core dependencies. The package pins them with `==` in pyproject.toml, so what is
# checked here is that pip actually honoured those pins.
EXPECTED: dict[str, str] = {
    "SQLAlchemy": "1.4.54",
    "pandas": "1.5.3",
    "numpy": "1.26.4",
}

# Optional. Checked only when installed — they are always present in the Mage image,
# and in CI they depend on which extras were installed.
EXPECTED_OPTIONAL: dict[str, str] = {
    "mysql-connector-python": "8.4.0",
    "pymssql": "2.3.13",
    "fdb": "2.0.4",
}

EXPECTED_PYTHON = (3, 10)

# The driver to the target, which is the one every run depends on — and the one
# this script used to say nothing about. It needs its own check because it is
# the only dependency that arrives under **two different distribution names**:
# the image builds `psycopg2` from source, while outside the image the `target`
# extra installs `psycopg2-binary`, deliberately (the two collide if both are
# present, which is why `dbextractors` is installed bare into the image).
#
# So a plain equality check is impossible here, and its absence was worse than
# it looked: the script printed "the runtime matches the production image" over
# a `psycopg2-binary` that had resolved nine patch versions past what the image
# ships, without ever looking. The range below is the one `pyproject.toml`
# declares; the version is printed either way so the drift is visible rather
# than merely permitted.
PSYCOPG2_IMAGE_VERSION = "2.9.3"
PSYCOPG2_RANGE = ((2, 9, 3), (2, 10))


def _release(raw: str) -> tuple[int, ...]:
    """The numeric release part of a version, ``2.9.12`` -> ``(2, 9, 12)``.

    Deliberately hand-rolled rather than using `packaging`: this script has to
    run in an environment that holds nothing but the three pinned dependencies,
    and `packaging` is not one of them. Anything after the digits (`rc1`,
    `.post0`) is not needed to answer "is this inside the declared range".
    """
    parts: list[int] = []
    for chunk in raw.split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_psycopg2(problems: list[str]) -> None:
    for name in ("psycopg2", "psycopg2-binary"):
        try:
            actual = version(name)
        except PackageNotFoundError:
            continue

        if name == "psycopg2":
            # Inside the image. Here an exact match is the whole point.
            if actual != PSYCOPG2_IMAGE_VERSION:
                problems.append(f"psycopg2 {actual} != {PSYCOPG2_IMAGE_VERSION} (image)")
            else:
                print(f"  ok  psycopg2 {actual}")
            return

        low, high = PSYCOPG2_RANGE
        release = _release(actual)
        if not (low <= release < high):
            problems.append(
                f"psycopg2-binary {actual} is outside the declared range "
                f"(>={'.'.join(map(str, low))},<{'.'.join(map(str, high))})"
            )
        elif release != _release(PSYCOPG2_IMAGE_VERSION):
            # Not a failure: outside the image `psycopg2-binary` is the right
            # choice and the range is intentional. But it is a difference from
            # production, and a difference nobody is told about is the kind that
            # explains a bug three months later.
            print(f"  ok  psycopg2-binary {actual} (image ships psycopg2 {PSYCOPG2_IMAGE_VERSION})")
        else:
            print(f"  ok  psycopg2-binary {actual}")
        return

    print("  --  psycopg2 not installed (optional)")


def main() -> int:
    problems: list[str] = []

    actual_python = sys.version_info[:2]
    if actual_python != EXPECTED_PYTHON:
        problems.append(
            f"Python {'.'.join(map(str, actual_python))} != "
            f"{'.'.join(map(str, EXPECTED_PYTHON))} (production Mage image)"
        )
    else:
        print(f"  ok  Python {sys.version.split()[0]}")

    for pkg, expected in EXPECTED.items():
        try:
            actual = version(pkg)
        except PackageNotFoundError:
            problems.append(f"{pkg} is not installed (expected {expected})")
            continue
        if actual != expected:
            problems.append(f"{pkg} {actual} != {expected}")
        else:
            print(f"  ok  {pkg} {actual}")

    _check_psycopg2(problems)

    for pkg, expected in EXPECTED_OPTIONAL.items():
        try:
            actual = version(pkg)
        except PackageNotFoundError:
            print(f"  --  {pkg} not installed (optional)")
            continue
        if actual != expected:
            problems.append(f"{pkg} {actual} != {expected}")
        else:
            print(f"  ok  {pkg} {actual}")

    if problems:
        print("\nThe runtime does not match the production Mage image:", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        print(
            "\nThese versions cannot be raised — the image is production and upgrading "
            "it is out of scope for this package.",
            file=sys.stderr,
        )
        return 1

    print("\nThe runtime matches the production Mage image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
