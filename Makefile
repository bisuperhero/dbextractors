.PHONY: venv install lint fix types test test-mage bench bench-write check verify-runtime db-up db-down clean

# The image `core/naming.py` and the write path were replicated from. It is no
# longer the runtime (that is v1.0.x), but the parity tests for the replicas
# cannot run anywhere else — see NOTICE.
MAGE_IMAGE := mageai/mageai:0.9.79

PY := 3.11
VENV := .venv
BIN := $(VENV)/bin

# The oldest supported Python, so that development does not quietly depend on
# something newer. CI covers the newer ones.
venv:
	uv venv --python $(PY) $(VENV)

install: venv
	uv pip install --python $(BIN)/python -e ".[dev]"

lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

fix:
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/ruff format src tests scripts

types:
	$(BIN)/mypy

# Most of the suite runs without a database. The rest reads its DSNs from a
# local .env; see .env.example. Anything without a DSN skips rather than fails,
# which is right locally — CI requires them to actually run.
test:
	$(BIN)/pytest

# Brings up the target and all four sources, seeded, and waits for health.
#
# The sources exist for the one bug class that string tests cannot find: a type
# that maps correctly and whose value still does not survive `COPY`. Three of
# the four had no live test at all before this.
# The servers are started and waited on first; the two one-shot seed containers
# run afterwards, because `--wait` treats a container that exits — even with
# status 0 — as a failure.
db-up:
	docker compose -f docker/compose.yml up -d --wait postgres mysql mssql firebird
	docker compose -f docker/compose.yml up --exit-code-from mssql-seed mssql-seed
	docker compose -f docker/compose.yml up --exit-code-from firebird-seed firebird-seed
	@echo "databases are up; copy .env.example to .env if you have not yet"

# `-v` on purpose: the seeds only run on an empty volume, so keeping them
# between runs would mean editing a seed and not seeing the change.
db-down:
	docker compose -f docker/compose.yml down -v

# Verifies that the local copy of the 825 reserved words still matches the
# live library. Locally the `needs_mage` tests are skipped because `mage_ai`
# is not installed, so this runs them inside the image instead. CI does the
# same in the `mage-parity` job.
test-mage:
	docker run --rm -v "$(CURDIR)":/work -w /work --entrypoint bash $(MAGE_IMAGE) -c '\
		pip install --quiet --disable-pip-version-check pytest==7.4.4 && \
		PYTHONPATH=/work/src python -m pytest tests/naming -m needs_mage -q -p no:cacheprovider'

# Benchmarks. `bench-write` has to run inside the image because it compares
# against the live library's write path.
bench:
	$(BIN)/python scripts/bench_coerce.py
	$(BIN)/python scripts/bench_hashing.py

bench-write:
	$(BIN)/python scripts/bench_write_path.py
	docker run --rm -v "$(CURDIR)":/work -w /work --entrypoint bash $(MAGE_IMAGE) -c '\
		python scripts/bench_write_path.py --no-db'

check: lint types test verify-runtime

# Verifies that what is installed satisfies the ranges the package declares —
# the check that matters in an image that already ships pandas or a driver.
verify-runtime:
	$(BIN)/python scripts/verify_runtime.py

clean:
	rm -rf $(VENV) build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
