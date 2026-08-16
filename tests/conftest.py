"""Shared plumbing for every test.

Two things, both of which have to happen before collection rather than inside a
fixture:

- the ``tests/`` directory goes on ``sys.path`` so that ``import
  reference_oracle`` works from any subdirectory. Pytest only inserts the
  directory the test file itself lives in, so without this
  ``tests/coerce/test_x.py`` could not reach ``tests/``.
- ``.env`` is loaded into the environment. See ``_load_dotenv`` for why this
  cannot live in the fixtures that consume it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


def _load_dotenv() -> None:
    """Load ``.env`` from the repo root, without overriding what is already set.

    ``setdefault``, not assignment: an exported variable is an explicit choice
    by whoever ran pytest and outranks the file. CI exports and ships no
    ``.env``, so there this is a no-op.

    This has to run at **import** time of the root conftest, which is the
    earliest point every test path passes through. The per-directory conftests
    each used to carry their own copy, called from inside the ``dsn`` fixture —
    and by then the fixture had already asked ``os.environ`` for
    ``DBX_GOLDEN_TEST_DSN``, found nothing, and committed to the
    ``resolve_dsn()`` fallback. The result was a ``.env`` that could not switch
    on the tests it exists to switch on: with a correct ``.env`` and nothing
    exported, the whole database suite skipped. The source switches
    (``DBX_TEST_MYSQL_DSN`` and friends) were never loaded at all, because the
    modules that read them do so directly and never called the loader.
    """
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# Pinning the target that `dbextractors.run()` writes to lives in
# `tests/target_pin.py`, not here — a copy of it used to sit in this file too,
# and nothing imported it. `conftest` is a name pytest reserves per directory,
# so `from conftest import ...` reaches the conftest of the importing
# subdirectory rather than this one; a helper defined here is unreachable by
# name from anywhere but the root.
