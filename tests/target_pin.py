"""Where ``dbextractors.run()`` is allowed to write during tests.

Since 0.2.6 the package resolves the target itself (``DBX_TARGET_DSN`` ->
``io_config.yaml`` -> ``POSTGRES_*``) and **deliberately never redirects it**:
production code must not write anywhere other than where it was told to.

In a test that address need not be reachable. When PostgreSQL runs on Windows
and pytest under WSL, the LAN address is blocked by the Windows firewall and
traffic goes through the gateway that `dbextractors.golden.session.resolve_dsn()`
substitutes. Without pinning, `dbextractors.run()` aims at the original address and
waits out the full TCP timeout — five such tests stretched the suite from ~50 s
to 698 s and ended in five failures.

Pinning belongs in the fixtures that build the DSN: the DSN a test writes to is
the one that fixture resolved, and pinning any earlier would guess at it.

This lives in its own module because ``conftest`` is a name pytest reserves per
directory — ``from conftest import ...`` picks up the conftest of the current
subdirectory, not the root one.
"""

from __future__ import annotations

import os


def pin_target(dsn: str) -> str:
    """Tell ``dbextractors.run()`` to write into the same database the test reads from."""
    os.environ["DBX_TARGET_DSN"] = dsn
    return dsn
