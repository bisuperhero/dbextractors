#!/usr/bin/env python3
"""How much memory a hash run costs — the legacy path against the new one.

The performance rules forbid ``seen_keys = set()`` holding every PK in RAM
(*"2-3 GB at 11 M rows"*), and the predecessor additionally keeps the **whole
target snapshot** in memory (``target_hashes = {}``). Both are claims about
memory, and a claim about memory should be measured, not estimated — the
orchestrator runs 3 pipelines side by side, so the peak is threefold and the
difference between 0.5 GB and 6 GB decides whether it fits on the machine.

The script runs **only the new** component (the legacy one is measured by
`golden_batch.py`, which runs both in the same process, so their peaks cannot
be told apart) and samples ``VmRSS`` from ``/proc/self/status`` on a side
thread.

It writes exclusively into a schema prefixed ``dbx_golden_``.

Usage::

    python scripts/bench_hash_memory.py --manifest manifest.json
    python scripts/bench_hash_memory.py --manifest manifest.json --interval 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from golden_batch import expand_env  # noqa: E402  (module local to scripts/)

from dbextractors.golden import runners, scratch  # noqa: E402


def rss_mb() -> float:
    """Resident memory of the process. Not ``docker stats`` — that counts the page cache too."""
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


class RssSampler(threading.Thread):
    """Samples RSS alongside the run. A thread is enough — the measured code is I/O bound."""

    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: List[float] = []
        # NOT `self._stop`: `threading.Thread._stop` is a method that `join()`
        # calls internally. Shadowing it with an attribute makes the code fail
        # with `TypeError: 'Event' object is not callable` — and only at
        # shutdown, that is, after the whole measurement is already done.
        self._finished = threading.Event()

    def run(self) -> None:
        while not self._finished.is_set():
            self.samples.append(rss_mb())
            self._finished.wait(self.interval)

    def stop(self) -> float:
        self._finished.set()
        self.join(timeout=5)
        return max(self.samples, default=0.0)


def _dsn() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ.get('POSTGRES_PASSWORD', '')}"
    )


def _run(runner, config: dict, schema: str, log, interval: float) -> dict:
    sampler = RssSampler(interval)
    baseline = rss_mb()
    sampler.start()
    start = time.perf_counter()
    try:
        result = runner.run(dict(config), schema, logger=log)
    finally:
        peak = sampler.stop()
    return {
        "rows": result.rows,
        "s": round(time.perf_counter() - start, 2),
        "rss_start_mb": round(baseline, 1),
        "rss_peak_mb": round(peak, 1),
        "rss_growth_mb": round(peak - baseline, 1),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dialect", default="mysql")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    log = logging.getLogger("bench")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    import psycopg2

    conn = psycopg2.connect(_dsn())
    conn.autocommit = True
    schema = scratch.scratch_schema_name("memory")
    scratch.create_schema(conn, schema)

    print("HASH RUN MEMORY — NEW COMPONENT")
    print("=" * 72)
    print(f"schema: {schema}\n")

    runner = runners.PackageRunner(args.dialect)
    results = []
    try:
        for item in manifest:
            config = expand_env(item["config"])
            label = item["label"]
            print(f"{label}")

            seed = json.loads(json.dumps(config))
            seed["LOAD_SETTINGS"]["load_method"] = "full"
            full_run = _run(runner, seed, schema, log, args.interval)
            print(
                f"  full  {full_run['rows']:>10,} rows / {full_run['s']:>7.2f}s   "
                f"peak RSS {full_run['rss_peak_mb']:>8.1f} MB "
                f"(+{full_run['rss_growth_mb']:.1f})"
            )

            hash_run = _run(runner, config, schema, log, args.interval)
            print(
                f"  hash  {hash_run['rows']:>10,} rows / {hash_run['s']:>7.2f}s   "
                f"peak RSS {hash_run['rss_peak_mb']:>8.1f} MB "
                f"(+{hash_run['rss_growth_mb']:.1f})"
            )
            results.append({"label": label, "full": full_run, "hash": hash_run})
    finally:
        scratch.drop_schema(conn, schema)
        conn.close()

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
