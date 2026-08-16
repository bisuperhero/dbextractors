#!/usr/bin/env python3
"""Golden test: the **legacy and the new** component over the same source table.

This is the gate that every migration step has to pass. Both sides read the same
source and write into two working schemas; the results are then compared on all
five levels.

The source is **read only**. Writes go exclusively into schemas prefixed with
``dbx_golden_`` — the safeguard lives in `dbextractors.golden.scratch` and cannot
be turned off.

The legacy component needs the Mage runtime, so the whole script has to run
inside the production image.

Usage::

    python scripts/golden_batch.py --manifest tables.json
    python scripts/golden_batch.py --manifest tables.json --keep
    python scripts/golden_batch.py --manifest hash.json --perturb 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dbextractors.golden import compare, runners, scratch, session  # noqa: E402
from dbextractors.golden.model import VERDICT_MATCH, Relation  # noqa: E402


def _dsn() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ.get('POSTGRES_PASSWORD', '')}"
    )


_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def expand_env(value: Any) -> Any:
    """``${VARIABLE}`` -> value from the environment. Recursively over the whole tree.

    It used to live in ``run_legacy.py`` and this module imported it from there.
    That dependency pointed the wrong way: ``run_legacy`` is a migration tool that
    runs the legacy component from its own sources, whereas this is a general
    batch run. In a checkout without those sources the import then also brought
    down ``tests/golden/test_perturb.py``, which has nothing to do with the legacy
    component.
    """
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, str):
        match = _ENV_REF.match(value)
        if match:
            name = match.group(1)
            if name not in os.environ:
                raise SystemExit(f"The config references ${{{name}}}, but it is not in the env.")
            return os.environ[name]
    return value


#: The value the target is broken with before a hash run. It has to be
#: recognisable — after the repair not a single one may be left in the target.
PERTURB_MARKER = "DBX_PERTURB"


def _text_column(conn, schema: str, table: str, pk: str) -> Optional[str]:
    """A text column that can be broken. Not the PK and nothing the package manages."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "AND data_type IN ('text', 'character varying') "
            "AND column_name NOT IN (%s, 'row_hash', '_timestamp') "
            "AND column_name NOT LIKE '\\_%%' "
            "ORDER BY ordinal_position LIMIT 1",
            (schema, table, pk),
        )
        row = cur.fetchone()
    return row[0] if row else None


def perturb(conn, schema: str, table: str, pk: str, limit: int) -> Tuple[Optional[str], int]:
    """Break part of the target so that the hash diff has something to repair.

    With no change in the source a hash run **transfers nothing** — which is the
    main property we want to demonstrate, but it also means the path that fetches
    changed rows never runs against real data at all. And the source may only be
    read from.

    So the target is broken instead: one text column of the selected rows is
    overwritten with a marker and their hash is thrown away. The hash diff then
    has to find them, fetch them from the source and write the true value back.
    After the run not a single marker may be left in the target — and that is an
    **absolute** check, not a comparison of two sides that could both be wrong in
    the same way.

    At most half the rows are broken: if nothing matched, the safety brake would
    fire and the run would fall back to a full load, so the fetch path would again
    go untested.
    """
    column = _text_column(conn, schema, table, pk)
    if column is None:
        return None, 0

    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
        total = int(cur.fetchone()[0])
        count = min(limit, total // 2)
        if count <= 0:
            # The table is too small to break. Returns **with** the column name so
            # that the caller does not confuse it with "nothing to break" — that
            # is a different reason.
            return column, 0

        cur.execute(
            f'UPDATE "{schema}"."{table}" SET "{column}" = %s, row_hash = %s '
            f'WHERE "{pk}" IN (SELECT "{pk}" FROM "{schema}"."{table}" '
            f'ORDER BY "{pk}" LIMIT %s)',
            (PERTURB_MARKER, f"{PERTURB_MARKER}_hash", count),
        )
        return column, int(cur.rowcount)


def _target_pk(config: dict) -> str:
    """The primary key in the **target** naming.

    The config states it under its source name, and that differs from the target
    one: a source can have a column ``ID`` while the target has ``id``. Without the
    conversion the breaking step asks for a column that does not exist and the
    hash profile ends in an error instead of a measurement.
    """
    from dbextractors.core.naming import clean_column_name

    raw = config["LOAD_SETTINGS"].get("primary_column") or "id"
    return clean_column_name(str(raw))


def count_marker(conn, schema: str, table: str, column: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT count(*) FROM "{schema}"."{table}" WHERE "{column}" = %s',
            (PERTURB_MARKER,),
        )
        return int(cur.fetchone()[0])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON: a list of {label, block, config}")
    parser.add_argument("--dialect", default="mysql")
    parser.add_argument("--keep", action="store_true", help="keep the working schemas")
    parser.add_argument("--json", dest="json_path", help="where to write the machine report")
    parser.add_argument(
        "--perturb",
        type=int,
        default=0,
        help="for hash tables, break this many target rows so the diff has something to repair",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    # The legacy component calls `logger.warning` on its very first line and
    # without a logger it dies with AttributeError — Mage always supplies one in
    # kwargs.
    log = logging.getLogger("golden")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    import psycopg2

    conn = psycopg2.connect(_dsn())
    conn.autocommit = True

    legacy_schema = scratch.scratch_schema_name("legacy")
    new_schema = scratch.scratch_schema_name("new")
    scratch.create_schema(conn, legacy_schema)
    scratch.create_schema(conn, new_schema)

    print("GOLDEN TEST — LEGACY vs NEW COMPONENT")
    print("=" * 78)
    print(f"legacy schema: {legacy_schema}")
    print(f"new schema:    {new_schema}")
    print(f"tables:        {len(manifest)}")
    print()

    dsn = session.resolve_dsn(_dsn())
    results: List[dict] = []

    try:
        for item in manifest:
            label = item["label"]
            config = expand_env(item["config"])
            block = str(REPO_ROOT / item["block"])

            row = {"label": label, "load_method": config["LOAD_SETTINGS"]["load_method"]}
            print(f"{label}  ({row['load_method']})")

            try:
                # Both the incremental and the hash run need a non-empty target
                # with hashes, otherwise both sides fall back to a full load and
                # the full load is what gets tested. For hash it is on top of that
                # the condition for the test to mean anything at all: the hash in
                # the target has to be produced by the **same path** the diff later
                # computes it with.
                if row["load_method"] in ("incremental", "hash"):
                    seed = json.loads(json.dumps(config))
                    seed["LOAD_SETTINGS"]["load_method"] = "full"
                    runners.LegacyBlockRunner(block).run(seed, legacy_schema, logger=log)
                    runners.PackageRunner(args.dialect).run(seed, new_schema, logger=log)
                    print(
                        f"  (target pre-filled by a full load so that "
                        f"{row['load_method']} is what gets tested)"
                    )

                broken_column: Optional[str] = None
                if row["load_method"] == "hash" and args.perturb:
                    table = runners.target_table_name(config)
                    pk = _target_pk(config)
                    column = count = None
                    for schema in (legacy_schema, new_schema):
                        column, count = perturb(conn, schema, table, pk, args.perturb)
                    row["broken"] = count or 0
                    broken_column = column if count else None
                    if broken_column:
                        print(
                            f"  (broke {count} rows in column {broken_column}, "
                            f"so the diff has something to repair)"
                        )
                    elif column is None:
                        print("  (no text column to break, only the match is tested)")
                    else:
                        print("  (too small to break, only the match is tested)")

                start = time.perf_counter()
                legacy = runners.LegacyBlockRunner(block).run(
                    dict(config), legacy_schema, logger=log
                )
                row["legacy_rows"] = legacy.rows
                row["legacy_s"] = round(time.perf_counter() - start, 2)

                start = time.perf_counter()
                new = runners.PackageRunner(args.dialect).run(dict(config), new_schema, logger=log)
                row["new_rows"] = new.rows
                row["new_s"] = round(time.perf_counter() - start, 2)

                # The gate: a second run with no change in the source must transfer
                # nothing. If the hash were seeded by a different path than the one
                # the diff computes it with, this is the only place where real data
                # would reveal it — otherwise the run is green and simply keeps
                # transferring everything, every time.
                if broken_column:
                    # Absolute check: after the repair no marker may be left in the
                    # target. A comparison of the two sides would not catch this —
                    # both could be wrong in the same way.
                    table = runners.target_table_name(config)
                    row["markers_left"] = {
                        "legacy": count_marker(conn, legacy_schema, table, broken_column),
                        "new": count_marker(conn, new_schema, table, broken_column),
                    }
                    left = row["markers_left"]
                    mark = "ok " if left["new"] == 0 else "ERROR"
                    print(
                        f"  {mark} repaired: markers left — legacy {left['legacy']}, "
                        f"new {left['new']}"
                    )

                if row["load_method"] == "hash":
                    start = time.perf_counter()
                    again = runners.PackageRunner(args.dialect).run(
                        dict(config), new_schema, logger=log
                    )
                    row["second_run_rows"] = again.rows
                    row["second_run_s"] = round(time.perf_counter() - start, 2)
                    mark = "ok " if again.rows == 0 else "ERROR"
                    print(
                        f"  {mark} second run with no changes transferred {again.rows} rows "
                        f"/ {row['second_run_s']}s"
                    )
            except Exception as err:
                row["verdict"] = "ERROR"
                row["error"] = f"{type(err).__name__}: {err}"
                print(f"  ERROR {row['error']}")
                results.append(row)
                continue

            print(
                f"  legacy {row['legacy_rows']} rows / {row['legacy_s']}s      "
                f"new {row['new_rows']} rows / {row['new_s']}s"
            )

            with session.connect(dsn, read_only=True) as ro:
                report = compare.compare_tables(
                    ro,
                    Relation(legacy_schema, legacy.table),
                    Relation(new_schema, new.table),
                    label=label,
                )
            row["verdict"] = report.verdict
            if report.error:
                # Without this the table ends up as ERROR and the report does not
                # say why — `compare_tables` hides the exception in `report.error`
                # and carries on.
                row["error"] = report.error
                print(f"  ERROR in the comparison: {report.error}")
            row["levels"] = [
                {"level": int(u.level), "passed": u.passed, "skipped": u.skipped}
                for u in report.levels
            ]
            for level in report.levels:
                if level.skipped:
                    mark = "--  "
                elif level.deviations and level.passed:
                    mark = "ok* "
                else:
                    mark = "ok  " if level.passed else "ERROR"
                print(f"  {mark} {int(level.level)}. {level.title}")
                if not level.passed and not level.skipped:
                    for difference in level.differences[:4]:
                        print(f"        {difference}")
                for deviation in level.deviations[:4]:
                    print(
                        f"        deviation: [{deviation.where}] "
                        f"{deviation.left} -> {deviation.right}"
                    )
            results.append(row)
            print()
    finally:
        if args.keep:
            print(f"Schemas {legacy_schema} and {new_schema} are kept.")
        else:
            scratch.drop_schema(conn, legacy_schema)
            scratch.drop_schema(conn, new_schema)
        conn.close()

    print("=" * 78)
    for row in results:
        second = row.get("second_run_rows")
        note = "" if second is None else f"   2nd run: {second} rows"
        print(f"  {row.get('verdict', '?'):<7} {row['label']:<32} " f"{row['load_method']}{note}")

    from dbextractors.golden.model import VERDICT_MATCH_WITH_DEVIATIONS

    ok = {VERDICT_MATCH, VERDICT_MATCH_WITH_DEVIATIONS}
    matched = sum(1 for r in results if r.get("verdict") in ok)
    clean = sum(1 for r in results if r.get("verdict") == VERDICT_MATCH)
    print(
        f"\nMATCH on {matched} of {len(results)} tables "
        f"({clean} with no deviations, {matched - clean} with agreed deviations)."
    )

    repaired = [r for r in results if r.get("markers_left")]
    if repaired:
        fully_repaired = [r for r in repaired if r["markers_left"]["new"] == 0]
        broken = sum(r.get("broken", 0) for r in repaired)
        print(
            f"The new side repaired every broken row in {len(fully_repaired)} "
            f"of {len(repaired)} tables ({broken:,} rows in total)."
        )
        for row in repaired:
            if row["markers_left"]["new"]:
                print(f"  WARNING {row['label']}: {row['markers_left']['new']} markers left")

    second_runs = [r for r in results if r.get("second_run_rows") is not None]
    silent = [r for r in second_runs if r["second_run_rows"] == 0]
    if second_runs:
        print(
            f"A second run with no changes transferred nothing in {len(silent)} "
            f"of {len(second_runs)} hash tables."
        )
        for row in second_runs:
            if row["second_run_rows"]:
                print(f"  WARNING {row['label']}: {row['second_run_rows']} rows")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    all_repaired = all(r["markers_left"]["new"] == 0 for r in repaired)
    return 0 if matched == len(results) and len(silent) == len(second_runs) and all_repaired else 1


if __name__ == "__main__":
    raise SystemExit(main())
