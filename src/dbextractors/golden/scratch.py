"""Scratch schemas for the golden test — and the guard against touching production.

The tool has to be idempotent and safe: it creates and cleans up its own target
schemas, and it never touches a production schema. Schema names are derived so
that a collision cannot happen.

The guard is deliberately **dumb and unpersuadable**: the only thing this module
can drop is a schema whose name starts with ``dbx_golden_``. There is no switch
that turns it off and no branch that goes around it. If a production schema ever
found its way into the configuration by mistake, the ``DROP`` does not happen —
it ends with an exception.

Name collisions: the name carries a timestamp and an eight-character digest of
the inputs (database, user, label). Two concurrent runs over the same table can
only collide if they started in the same second with the same inputs — and even
then there is a random component in the digest.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:  # pragma: no cover
    import psycopg2.extensions

#: The only prefix this module treats as its own.
SCRATCH_PREFIX = "dbx_golden_"

_SAFE_NAME = re.compile(rf"^{SCRATCH_PREFIX}[a-z0-9_]{{1,50}}$")


class UnsafeSchemaError(RuntimeError):
    """An attempt to touch a schema that does not belong to the golden test."""


def scratch_schema_name(label: str = "", *, now: Optional[datetime] = None) -> str:
    """Builds a scratch schema name for which a collision is practically ruled out.

    Args:
        label: Optional label (typically the table name), so that it is possible
            to tell in the database what a schema belongs to.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M%S")
    seed = f"{label}|{os.getpid()}|{uuid.uuid4()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]

    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:20]
    parts = [SCRATCH_PREFIX.rstrip("_"), stamp, digest]
    if slug:
        parts.insert(2, slug)
    name = "_".join(parts)

    assert_safe(name)
    return name


def assert_safe(schema: str) -> None:
    """Checks that a name may be created and dropped.

    Called before **every** ``CREATE`` and ``DROP``. Not once at startup.
    """
    if not _SAFE_NAME.match(schema or ""):
        raise UnsafeSchemaError(
            f"Schema {schema!r} does not belong to the golden test. Only schemas "
            f"with the {SCRATCH_PREFIX!r} prefix may be created and dropped. "
            "This guards against touching production and cannot be turned off."
        )


def create_schema(conn: psycopg2.extensions.connection, schema: str) -> None:
    assert_safe(schema)
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')


def drop_schema(conn: psycopg2.extensions.connection, schema: str) -> None:
    """Drops a scratch schema together with its contents.

    ``CASCADE`` is fine here — unlike on a production target, where it is
    forbidden because it takes down the views built on top of the table (102 of
    them in one deployment). Here it only drops what the harness itself created
    moments ago.
    """
    assert_safe(schema)
    with conn.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@contextmanager
def scratch_schema(
    conn: psycopg2.extensions.connection, label: str = "", *, keep: bool = False
) -> Iterator[str]:
    """Creates a scratch schema and cleans it up afterwards.

    Args:
        keep: Leave the schema behind. For debugging, when the result needs to be
            inspected by hand. The name is printed in that case, so that it can
            be cleaned up later.
    """
    schema = scratch_schema_name(label)
    create_schema(conn, schema)
    try:
        yield schema
    finally:
        if not keep:
            drop_schema(conn, schema)


def list_leftovers(conn: psycopg2.extensions.connection) -> list[str]:
    """Scratch schemas left behind by some earlier run.

    A run interrupted half way through leaves its schema behind. This helps find
    it.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname LIKE %s ORDER BY nspname",
            (SCRATCH_PREFIX + "%",),
        )
        return [row[0] for row in cur.fetchall()]
