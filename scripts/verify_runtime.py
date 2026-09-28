"""Verify that the runtime satisfies what the installed package declares.

pip enforces these ranges when it resolves an install, so in a clean
environment this script has nothing to find. It exists for the other kind:
an orchestrator image that already ships pandas, SQLAlchemy or a driver, into
which the package was installed with ``--no-deps`` or next to pins pip could
not reconcile. There a wrong version does not fail the install — it fails the
first run, or worse, it does not fail and produces different values. This
script runs in CI after installing the built wheel and locally via
``make verify-runtime``, and exits non-zero on any mismatch.

The ranges are **not** repeated here: they are read from the installed
distribution's own metadata, so they cannot drift from ``pyproject.toml``.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, metadata, requires, version

DISTRIBUTION = "dbextractors"

#: `psycopg2-binary` is what the `target` extra installs; an image that builds
#: `psycopg2` from source ships it under this other name. Either satisfies the
#: same range, and both at once collide.
ALTERNATIVE_NAMES: dict[str, tuple[str, ...]] = {"psycopg2-binary": ("psycopg2",)}

_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*([^;]*?)\s*(?:;\s*(.*))?$")


def _release(raw: str) -> tuple[int, ...]:
    """The numeric release part of a version, ``2.9.12`` -> ``(2, 9, 12)``.

    Deliberately hand-rolled rather than using `packaging`: this script has to
    run in an environment that holds nothing but the package's own
    dependencies, and `packaging` is not one of them. Anything after the digits
    (`rc1`, `.post0`) is not needed to answer "is this inside the range".
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


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)), b + (0,) * (width - len(b))


def _satisfies(actual: str, specifier: str) -> bool:
    """``>=``, ``<``, ``<=``, ``>``, ``==`` and ``!=`` clauses, comma-separated."""
    have = _release(actual)
    for clause in filter(None, (c.strip() for c in specifier.split(","))):
        match = re.match(r"(>=|<=|==|!=|<|>)\s*(.+)", clause)
        if not match:
            raise ValueError(f"unsupported version clause {clause!r}")
        op, bound = match.groups()
        left, right = _pad(have, _release(bound))
        ok = {
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
            "==": left == right,
            "!=": left != right,
        }[op]
        if not ok:
            return False
    return True


def _requirements() -> list[tuple[str, str, str | None]]:
    """``(name, specifier, extra)`` for every declared requirement."""
    parsed = []
    for raw in requires(DISTRIBUTION) or []:
        match = _REQUIREMENT.match(raw)
        if not match:
            continue
        name, specifier, marker = match.groups()
        extra = None
        if marker:
            found = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker)
            extra = found.group(1) if found else None
        if name == DISTRIBUTION:
            continue  # `dev` pulls in the other extras by name
        parsed.append((name, specifier, extra))
    return parsed


def _installed(name: str) -> tuple[str, str] | None:
    for candidate in (name, *ALTERNATIVE_NAMES.get(name, ())):
        try:
            return candidate, version(candidate)
        except PackageNotFoundError:
            continue
    return None


def main() -> int:
    problems: list[str] = []

    try:
        declared_python = metadata(DISTRIBUTION)["Requires-Python"] or ""
    except PackageNotFoundError:
        print(f"{DISTRIBUTION} is not installed.", file=sys.stderr)
        return 1

    running = ".".join(map(str, sys.version_info[:3]))
    if _satisfies(running, declared_python):
        print(f"  ok  Python {running} ({declared_python})")
    else:
        problems.append(f"Python {running} is outside {declared_python}")

    seen: set[str] = set()
    for name, specifier, extra in _requirements():
        if name in seen or extra == "dev":
            continue
        seen.add(name)
        found = _installed(name)
        if found is None:
            if extra is None:
                problems.append(f"{name} is not installed (required: {specifier})")
            else:
                print(f"  --  {name} not installed (extra '{extra}')")
            continue
        installed_as, actual = found
        label = installed_as if installed_as == name else f"{installed_as} (for {name})"
        if not specifier or _satisfies(actual, specifier):
            print(f"  ok  {label} {actual} ({specifier or 'any'})")
        else:
            problems.append(f"{label} {actual} is outside {specifier}")

    if problems:
        print("\nThe runtime does not satisfy what dbextractors declares:", file=sys.stderr)
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        return 1

    print("\nThe runtime satisfies what dbextractors declares.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
