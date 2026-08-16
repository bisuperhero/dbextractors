"""Source database adapters.

A new source means a new file in this package and nothing else. A dialect knows
nothing about the target.

The migration order was deliberately chosen from the smallest blast radius
upwards: PostgreSQL (21 tables) → MSSQL (16) → a second, larger PostgreSQL
deployment (58) → MySQL (254 + 237) → Firebird (80). MySQL is 73 % of the volume
yet comes fourth on purpose — if the golden test turns out to have holes, let
that show on 21 tables rather than on 254.
"""

from dbextractors.dialects.base import (
    ColumnDef,
    DictTypeMapDialect,
    SizeEstimate,
    SourceDialect,
    SurrogateKey,
    TableRef,
    UnsupportedFeature,
)

__all__ = [
    "ColumnDef",
    "DictTypeMapDialect",
    "SizeEstimate",
    "SourceDialect",
    "SurrogateKey",
    "TableRef",
    "UnsupportedFeature",
    "get_dialect",
]


def get_dialect(name: str) -> SourceDialect:
    """Return the dialect instance named in ``dbextractors.run(dialect=...)``.

    There is a single registry and it lives in `dbextractors.entrypoint`, where
    `run` needs it before anything else. This function is only a pass-through
    name for a caller that would rather not import `entrypoint` itself.

    The import is deliberately made inside the function: the drivers (`fdb`,
    `pymssql`) need not be installed in every environment, and there is no point
    pulling one in for a dialect nobody asked for.
    """
    from dbextractors.entrypoint import resolve_dialect

    return resolve_dialect(name)
