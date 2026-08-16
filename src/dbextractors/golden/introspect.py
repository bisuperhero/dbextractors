"""Reading table metadata from PostgreSQL.

Everything here is a plain ``SELECT``. This module never writes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dbextractors.golden.model import Relation

if TYPE_CHECKING:  # pragma: no cover
    import psycopg2.extensions


@dataclass(frozen=True)
class ColumnInfo:
    """One target column as ``information_schema`` sees it."""

    name: str
    ordinal: int
    data_type: str
    udt_name: str
    is_nullable: bool
    char_max_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    datetime_precision: Optional[int] = None
    #: Collation affects `min`/`max` over text. If two otherwise identical
    #: tables differed in collation, level 4 would report a difference with no
    #: visible cause — so it is checked at level 3, where the reason shows.
    collation: Optional[str] = None

    def type_signature(self) -> tuple:
        """What level 3 compares."""
        return (
            self.data_type,
            self.udt_name,
            self.is_nullable,
            self.char_max_length,
            self.numeric_precision,
            self.numeric_scale,
            self.datetime_precision,
            self.collation,
        )

    def describe_type(self) -> str:
        """Readable rendering of the type, for the report."""
        text = self.data_type
        if self.char_max_length is not None:
            text += f"({self.char_max_length})"
        elif self.numeric_precision is not None and self.data_type in ("numeric", "decimal"):
            scale = self.numeric_scale if self.numeric_scale is not None else 0
            text += f"({self.numeric_precision},{scale})"
        text += " NULL" if self.is_nullable else " NOT NULL"
        if self.collation:
            text += f" COLLATE {self.collation}"
        return text


def table_exists(conn: psycopg2.extensions.connection, relation: Relation) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (str(relation),))
        return bool(cur.fetchone()[0])


def row_count(conn: psycopg2.extensions.connection, relation: Relation) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {relation.qualified()}")
        return int(cur.fetchone()[0])


def columns(conn: psycopg2.extensions.connection, relation: Relation) -> list[ColumnInfo]:
    """Columns **ordered by ``ordinal_position``**.

    That order is not cosmetic — it is part of the contract the dbt layer rests
    on, and level 2 compares it. See the column naming section of
    ``docs/legacy-compat.md``.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, ordinal_position, data_type, udt_name, is_nullable,
                   character_maximum_length, numeric_precision, numeric_scale,
                   datetime_precision, collation_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (relation.schema, relation.table),
        )
        return [
            ColumnInfo(
                name=row[0],
                ordinal=row[1],
                data_type=row[2],
                udt_name=row[3],
                is_nullable=(row[4] == "YES"),
                char_max_length=row[5],
                numeric_precision=row[6],
                numeric_scale=row[7],
                datetime_precision=row[8],
                collation=row[9],
            )
            for row in cur.fetchall()
        ]


def unique_key(conn: psycopg2.extensions.connection, relation: Relation) -> Optional[list[str]]:
    """Finds a key that rows can be matched on.

    The primary key wins; without one, the narrowest valid unique index is used.
    Partial indexes (``WHERE …``) are skipped — they do not cover the whole
    table, so a comparison built on one would silently ignore some of the rows.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT array_agg(a.attname ORDER BY k.ord)
            FROM pg_index i
            CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
            WHERE i.indrelid = %s::regclass
              AND i.indisunique AND i.indisvalid AND i.indpred IS NULL
            GROUP BY i.indexrelid, i.indisprimary, i.indnatts
            ORDER BY i.indisprimary DESC, i.indnatts ASC
            LIMIT 1
            """,
            (str(relation),),
        )
        row = cur.fetchone()
        return list(row[0]) if row else None
