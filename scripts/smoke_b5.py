"""End-to-end pass of `dbextractors.run` against a live PostgreSQL.

This checks the wiring that unit tests with fakes and doubles cannot reach:

1. `run` picks a connection in ``auto`` mode — the probe succeeds and it goes direct,
2. the status frame carries the newer columns and ``connection_mode`` holds the mode
   that was actually used,
3. a multi-source run across two databases returns two rows with a ``source`` column,
4. ``connection_mode: direct`` skips the probe entirely,
5. one source failing does not stop the others, and the run ends with `SourceExtractionError`.

Writes go **only** into schemas prefixed with `dbx_golden_`, and reads only from those.
"""

from __future__ import annotations

import re

import psycopg2

from dbextractors import entrypoint
from dbextractors.golden import scratch, session

DSN = session.resolve_dsn()


def _connect():
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    return conn


def _create_source_table(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{schema}".source_table ('
            "  id integer PRIMARY KEY,"
            "  name text,"  # reserved word -> has to land as _name in the target
            "  amount numeric(12,2)"
            ")"
        )
        cur.execute(
            f'INSERT INTO "{schema}".source_table (id, name, amount) '
            "SELECT g, 'row ' || g, g * 1.5 FROM generate_series(1, 250) g"
        )
    conn.commit()


def _source_db() -> dict:
    """SOURCE_DB built from the **resolved** DSN, not straight from .env.

    `golden.session.resolve_dsn` falls back to the WSL gateway address: POSTGRES_HOST
    in .env is the address as seen from Windows, and from WSL it does not resolve. That
    fallback covers the target, but not the source — and `run` behaves correctly in
    ``auto`` mode there: the probe fails, SSH is tried next, and the error says plainly
    that no tunnel is configured.
    """
    fields = dict(part.split("=", 1) for part in DSN.split() if "=" in part)
    return {
        "user": fields.get("user"),
        "password": fields.get("password"),
        "host": fields.get("host"),
        "port": int(fields.get("port", 5432)),
        "database": fields.get("dbname"),
    }


def _config(source_schema: str, target_schema: str, **extra) -> dict:
    cfg = {
        "TABLE": {
            "source_name": "source_table",
            "source_schema": source_schema,
            "output_schema": target_schema,
            "output_name": "target_table",
        },
        "LOAD_SETTINGS": {"load_method": "full", "primary_column": "id", "batch_size": 100},
        "SOURCE_DB": dict(_source_db()),
    }
    cfg.update(extra)
    return cfg


def main() -> int:
    conn = _connect()
    source_schema = scratch.scratch_schema_name("smokesrc")
    target_schema = scratch.scratch_schema_name("smoketgt")
    scratch.create_schema(conn, source_schema)
    scratch.create_schema(conn, target_schema)
    conn.commit()

    try:
        _create_source_table(conn, source_schema)

        print("1) auto - the probe succeeds, the run goes direct")
        frame = entrypoint.run(_config(source_schema, target_schema), "postgres")
        print(frame.to_string(index=False))
        assert list(frame["connection_mode"]) == ["direct"], frame["connection_mode"].tolist()
        assert int(frame["rows_written"].iloc[0]) == 250
        assert bool(frame["success"].iloc[0]) is True
        print("   ok: 250 rows, connection_mode=direct\n")

        print("2) target columns - 'name' is reserved, so it must land as '_name'")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=%s AND table_name='target_table' ORDER BY ordinal_position",
                (target_schema,),
            )
            columns = [r[0] for r in cur.fetchall()]
        conn.commit()
        print("  ", columns)
        assert "_name" in columns, columns
        assert "_deleted_in_source" in columns and "_timestamp" in columns
        print("   ok\n")

        print("3) connection_mode: direct - the probe is never called")
        frame = entrypoint.run(
            _config(source_schema, target_schema, SOURCE_DB_MODE=None)
            | {"connection_mode": "direct"},
            "postgres",
        )
        assert list(frame["connection_mode"]) == ["direct"]
        print("   ok\n")

        print("4) multi-source - two rows carrying a source column")
        db = _source_db()["database"]
        cfg = _config(source_schema, target_schema)
        cfg["DATABASES"] = [db, db]
        cfg["LOAD_SETTINGS"] = {
            **cfg["LOAD_SETTINGS"],
            "multi_source": True,
        }
        frame = entrypoint.run(cfg, "postgres")
        print(frame.to_string(index=False))
        assert len(frame) == 2, len(frame)
        assert frame["source"].tolist() == [db, db]
        print("   ok\n")

        print("5) one source fails - the rest finish and the run ends red")
        cfg = _config(source_schema, target_schema)
        cfg["DATABASES"] = [db, "database_that_does_not_exist", db]
        cfg["LOAD_SETTINGS"] = {
            **cfg["LOAD_SETTINGS"],
            "multi_source": True,
        }
        try:
            entrypoint.run(cfg, "postgres")
        except entrypoint.SourceExtractionError as err:
            text = str(err)
            print("  ", text[:200].replace("\n", " "))
            # Matched by shape, not by wording: the summary has to say how many of how
            # many sources failed, and name the one that did. Asserting the exact
            # sentence would make this smoke test hostage to any rephrasing of the
            # message.
            assert re.search(r"\b1\b.*\b3\b", text), text
            assert "database_that_does_not_exist" in text
            print("   ok: two databases finished, one failed, the exception sums it up\n")
        else:
            raise AssertionError("expected SourceExtractionError to be raised")

        print("DONE: the wiring holds against a live database.")
        return 0
    finally:
        conn.rollback()
        scratch.drop_schema(conn, source_schema)
        scratch.drop_schema(conn, target_schema)
        conn.commit()
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
