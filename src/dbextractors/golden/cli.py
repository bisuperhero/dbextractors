"""Command line of the golden test: ``dbx-golden``.

Subcommands:

``compare``    compares two tables that already exist (read only)
``batch``      the same over a list of tables taken from a manifest
``leftovers``  lists (and optionally cleans up) scratch schemas from interrupted runs

Exit code: ``0`` = MATCH, ``1`` = DIFF, ``2`` = ERROR. A script that migrates
group by group can be driven by it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Sequence

from dbextractors.golden import progress as progress_mod
from dbextractors.golden import report as report_mod
from dbextractors.golden import scratch, session
from dbextractors.golden.compare import CompareOptions, compare_tables
from dbextractors.golden.model import (
    VERDICT_DIFF,
    VERDICT_ERROR,
    BatchReport,
    Relation,
)

EXIT_MATCH = 0
EXIT_DIFF = 1
EXIT_ERROR = 2


def _exit_code(verdict: str) -> int:
    if verdict == VERDICT_ERROR:
        return EXIT_ERROR
    return EXIT_DIFF if verdict == VERDICT_DIFF else EXIT_MATCH


def _options_from_args(args: argparse.Namespace) -> CompareOptions:
    return CompareOptions(
        key_columns=args.key.split(",") if args.key else None,
        hash_column=args.hash_column,
        sample_limit=args.samples,
        skip_row_level=args.skip_row_level,
        compare_type_mismatch_as_text=args.compare_as_text,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbx-golden",
        description=(
            "Golden test — compares the output of the old and the new component. "
            "It only reads from the production database."
        ),
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN. Otherwise DBX_GOLDEN_DSN / POSTGRES_*.")
    parser.add_argument("-v", "--verbose", action="store_true", help="detailed output")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_compare_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--key", help="comma-separated key columns; otherwise the PK is looked up")
        p.add_argument(
            "--hash-column",
            default="row_hash",
            help="column holding the row hash (row_hash by default)",
        )
        p.add_argument("--samples", type=int, default=10, help="how many differing keys to print")
        p.add_argument(
            "--skip-row-level",
            action="store_true",
            help="skips level 5 (FULL OUTER JOIN) on genuinely large tables",
        )
        p.add_argument(
            "--compare-as-text",
            dest="compare_as_text",
            action="store_true",
            help=(
                "also compares columns whose type differs between the two sides, "
                "through their textual form. Level 3 still reports the type "
                "difference; this only adds a comparison of the values. Meant for "
                "PostgreSQL sources, where most columns would otherwise be skipped"
            ),
        )
        # Original Czech spelling of the switch, kept so that existing scripts
        # keep working. Hidden from --help; --compare-as-text is the name to use.
        p.add_argument(
            "--porovnat-jako-text",
            dest="compare_as_text",
            action="store_true",
            help=argparse.SUPPRESS,
        )
        p.add_argument("--json", dest="json_path", help="where to write the machine report")
        p.add_argument(
            "--no-progress",
            dest="no_progress",
            action="store_true",
            help=(
                "do not print progress. Progress goes to stderr and only turns "
                "itself on in a terminal, so it reaches neither a file nor a CI "
                "log even without this switch"
            ),
        )
        # Original Czech spelling, kept for existing scripts. Hidden from --help.
        p.add_argument(
            "--bez-pokroku",
            dest="no_progress",
            action="store_true",
            help=argparse.SUPPRESS,
        )

    compare = sub.add_parser("compare", help="compares two existing tables")
    compare.add_argument("left", help="reference table, 'schema.table'")
    compare.add_argument("right", help="table under test, 'schema.table'")
    add_compare_options(compare)

    batch = sub.add_parser("batch", help="compares a list of tables from a manifest")
    batch.add_argument("manifest", help="YAML or JSON holding a list of pairs")
    add_compare_options(batch)

    leftovers = sub.add_parser("leftovers", help="scratch schemas from interrupted runs")
    leftovers.add_argument(
        "--clean", action="store_true", help="drops them (only schemas prefixed dbx_golden_)"
    )
    return parser


def _load_manifest(path: str) -> list[dict]:
    """A manifest is a list of table pairs.

    JSON, or YAML when PyYAML is installed::

        - label: invoices
          left:  legacy_schema.invoices
          right: dbx_golden_x.invoices
          key:   id
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "A YAML manifest needs PyYAML: pip install 'mage-db-extractors[golden]'"
            ) from exc
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)

    if isinstance(data, dict) and "tables" in data:
        data = data["tables"]
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of table pairs.")
    return data


def _cmd_compare(args: argparse.Namespace) -> int:
    left = Relation.parse(args.left)
    right = Relation.parse(args.right)

    progress = progress_mod.build(False if args.no_progress else None)
    with session.connect(args.dsn, read_only=True) as conn:
        info = session.describe_session(conn)
        result = compare_tables(conn, left, right, _options_from_args(args), progress=progress)

    print(report_mod.render_table(result, verbose=args.verbose))
    print()
    print(
        f"  {info['server_version']} · {info['database']} as {info['user']} · "
        f"read-only: {'yes' if info['read_only'] else 'NO'}"
    )

    if args.json_path:
        report_mod.to_json(result, args.json_path)
        print(f"  machine-readable report: {args.json_path}")

    return _exit_code(result.verdict)


def _cmd_batch(args: argparse.Namespace) -> int:
    entries = _load_manifest(args.manifest)
    batch = BatchReport(started_at=datetime.now(timezone.utc).isoformat())

    progress = progress_mod.build(False if args.no_progress else None)
    with session.connect(args.dsn, read_only=True) as conn:
        info = session.describe_session(conn)
        for index, entry in enumerate(entries, start=1):
            options = _options_from_args(args)
            if entry.get("key"):
                options.key_columns = [c.strip() for c in str(entry["key"]).split(",")]
            if entry.get("hash_column"):
                options.hash_column = entry["hash_column"]

            # Over 79 tables, "which one out of how many" is the single most
            # important thing on screen — without it there is no telling whether
            # to wait a minute or an hour.
            progress.message(f"[{index}/{len(entries)}] {entry.get('label') or entry['left']}")
            batch.reports.append(
                compare_tables(
                    conn,
                    Relation.parse(entry["left"]),
                    Relation.parse(entry["right"]),
                    options,
                    label=entry.get("label"),
                    progress=progress,
                )
            )
    batch.finished_at = datetime.now(timezone.utc).isoformat()

    print(report_mod.render_batch(batch, verbose=args.verbose))
    print()
    print(f"  {info['server_version']} · {info['database']} as {info['user']}")

    if args.json_path:
        report_mod.to_json(batch, args.json_path)
        print(f"  machine-readable report: {args.json_path}")

    return _exit_code(batch.verdict)


def _cmd_leftovers(args: argparse.Namespace) -> int:
    # Cleaning up writes, hence read_only=False. Only schemas prefixed
    # dbx_golden_ can be dropped — the guard lives in scratch.assert_safe.
    with session.connect(args.dsn, read_only=not args.clean) as conn:
        names = scratch.list_leftovers(conn)
        if not names:
            print("No leftover scratch schemas.")
            return EXIT_MATCH
        for name in names:
            if args.clean:
                scratch.drop_schema(conn, name)
                print(f"dropped    {name}")
            else:
                print(f"left over  {name}")
    if not args.clean:
        print("\nClean up with: dbx-golden leftovers --clean")
    return EXIT_MATCH


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "leftovers":
            return _cmd_leftovers(args)
    except (session.SessionError, scratch.UnsafeSchemaError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
