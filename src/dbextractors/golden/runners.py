"""Running the old and the new component into scratch schemas.

The input is the same thing a pipeline already has: the table's
``PIPELINE_CONFIG`` dict as used by the Mage config blocks, plus the name of the
old block and the dialect name for the new package.

The module has two halves, and it is worth knowing how each of them is covered:

**Redirecting the configuration** (``redirect_output``) is a pure function over a
dict. It is tested and needs neither a database nor Mage. It is also the half
where a mistake would be most expensive — if the redirect did not take effect,
the old component would write into a production schema. That is why
``redirect_output`` **verifies** its own result and refuses to continue when
anything is ambiguous.

**Actually running it** (``LegacyBlockRunner``) needs the Mage runtime: all 15
predecessor blocks import ``mage_ai`` at module level. Two of them additionally
import modules from their own deployment repository, which are not part of this
tree — a Firebird resolution helper in one, a source-database registry in
another (the latter inside a ``try``).

The old component therefore cannot be run from a plain pytest — it has to run
inside the deployment's own Mage image. The runner says so in a comprehensible
way instead of failing on an ``ImportError`` inside somebody else's file.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    import logging

#: Keys through which a configuration names the target schema. The first one is
#: current, the rest are legacy top-level variants (see ``docs/legacy-compat.md``).
SCHEMA_KEYS: tuple[str, ...] = ("output_schema", "OUTPUT_SCHEMA")


class RunnerError(RuntimeError):
    """The component could not be run."""


class UnsafeConfigError(RunnerError):
    """The configuration could not be safely redirected away from production."""


@dataclass
class RunOutcome:
    """What a run produced."""

    schema: str
    table: str
    rows: Optional[int] = None
    detail: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {}


def target_table_name(config: dict) -> str:
    """Works out which table a configuration writes into.

    The order matches how the predecessor reads it: ``TABLE.output_name`` first,
    then ``TABLE.output_table``, then the legacy top-level ``OUTPUT_TABLE``.
    """
    table_cfg = config.get("TABLE") or {}
    name = (
        table_cfg.get("output_name") or table_cfg.get("output_table") or config.get("OUTPUT_TABLE")
    )
    if not name:
        raise UnsafeConfigError(
            "The configuration does not name a target table "
            "(TABLE.output_name / TABLE.output_table / OUTPUT_TABLE)."
        )
    return str(name)


def redirect_output(config: dict, schema: str) -> dict:
    """Returns a copy of the configuration that writes into ``schema``, not production.

    The input dict is not modified — the caller keeps its own configuration
    untouched.

    **Every** schema-bearing key present in the configuration is rewritten, not
    just the first one found. If a configuration had both ``TABLE.output_schema``
    and the legacy ``OUTPUT_SCHEMA`` and only one of them were rewritten, the
    component could still reach production through the other.

    Raises:
        UnsafeConfigError: When the schema could not be set, or when the original
            value survived somewhere after the rewrite.
    """
    from dbextractors.golden.scratch import assert_safe

    # The guard is here as well, not only in scratch.py. Redirecting into a
    # production schema is precisely the scenario being prevented.
    assert_safe(schema)

    patched = copy.deepcopy(config)
    changed: list[str] = []

    table_cfg = patched.get("TABLE")
    if isinstance(table_cfg, dict) and "output_schema" in table_cfg:
        table_cfg["output_schema"] = schema
        changed.append("TABLE.output_schema")
    if "OUTPUT_SCHEMA" in patched:
        patched["OUTPUT_SCHEMA"] = schema
        changed.append("OUTPUT_SCHEMA")

    if not changed:
        # The configuration does not mention a schema at all. Set it where the
        # current generation of extractors reads it from — but not silently.
        patched.setdefault("TABLE", {})["output_schema"] = schema
        changed.append("TABLE.output_schema (added)")

    leftovers = _schemas_in(patched) - {schema}
    if leftovers:
        raise UnsafeConfigError(
            f"Foreign schemas survived the redirect: {sorted(leftovers)}. "
            "Refusing to continue — the component could write into production."
        )
    return patched


def _schemas_in(config: dict) -> set[str]:
    """Every value in the configuration that names a **target** schema.

    The source schema (``TABLE.source_schema``) deliberately does not count —
    that one is only read from.
    """
    found: set[str] = set()
    table_cfg = config.get("TABLE")
    if isinstance(table_cfg, dict) and table_cfg.get("output_schema"):
        found.add(str(table_cfg["output_schema"]))
    if config.get("OUTPUT_SCHEMA"):
        found.add(str(config["OUTPUT_SCHEMA"]))
    return found


class Runner(ABC):
    """Something that can fill a target table from a configuration."""

    name: str

    @abstractmethod
    def run(self, config: dict, schema: str, logger: Optional[logging.Logger] = None) -> RunOutcome:
        """Processes the table into ``schema``."""


class LegacyBlockRunner(Runner):
    """The old component — one of the predecessor extractor blocks.

    Imports the file as a module and calls its ``load_data``. It needs an
    environment in which ``mage_ai`` is importable, that is, the deployment's
    Mage image; it will not run under a plain pytest.
    """

    def __init__(self, block_path: str) -> None:
        self.block_path = block_path
        self.name = f"legacy:{block_path}"

    def run(self, config: dict, schema: str, logger: Optional[logging.Logger] = None) -> RunOutcome:
        patched = redirect_output(config, schema)
        module = self._load_module()

        load_data = getattr(module, "load_data", None)
        if load_data is None:
            raise RunnerError(f"{self.block_path} has no load_data function.")

        # Mage's `@data_loader` decorator keeps the original function reachable;
        # when it does not, whatever lives under the name gets called.
        target = getattr(load_data, "__wrapped__", load_data)
        result = target(patched, logger=logger)

        return RunOutcome(
            schema=schema,
            table=target_table_name(patched),
            rows=_rows_from_status(result),
            detail={"runner": self.name},
        )

    def _load_module(self) -> Any:
        import importlib.util
        import pathlib

        path = pathlib.Path(self.block_path)
        if not path.is_file():
            raise RunnerError(f"File {self.block_path} does not exist.")

        spec = importlib.util.spec_from_file_location(f"dbx_golden_legacy_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise RunnerError(f"{self.block_path} cannot be loaded as a module.")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            raise RunnerError(
                f"{self.block_path} cannot be imported: {exc}\n"
                "All 15 predecessor blocks import mage_ai at module level, and two "
                "of them also import modules from their own deployment repository. "
                "The old component therefore has to be run inside that deployment's "
                "Mage image, not under a plain pytest — see the golden test "
                "documentation."
            ) from exc
        return module


class PackageRunner(Runner):
    """The new component — ``dbextractors.run``."""

    def __init__(self, dialect: str) -> None:
        self.dialect = dialect
        self.name = f"dbextractors:{dialect}"

    def run(self, config: dict, schema: str, logger: Optional[logging.Logger] = None) -> RunOutcome:
        from dbextractors import run as dbx_run

        patched = redirect_output(config, schema)
        result = dbx_run(patched, dialect=self.dialect, logger=logger)
        return RunOutcome(
            schema=schema,
            table=target_table_name(patched),
            rows=_rows_from_status(result),
            detail={"runner": self.name},
        )


def _rows_from_status(status: Any) -> Optional[int]:
    """Pulls the row count out of the returned status DataFrame, when it is there.

    The shape of that DataFrame differs slightly between the predecessor blocks,
    so this probes carefully and returns ``None`` when it fails. The row count is
    supplementary information anyway — the binding number comes from level 1 of
    the comparison, which reads it straight from the database.
    """
    try:
        for column in ("rows_written", "rows", "row_count", "total_rows"):
            if column in getattr(status, "columns", []):
                return int(status[column].sum())
    except Exception:
        return None
    return None
