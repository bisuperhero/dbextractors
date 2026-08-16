"""Tests of the package skeleton.

These were written before there was anything to test in substance — no function was
implemented yet. They therefore guard what falls apart most quietly in an empty
skeleton:

1. that every module imports at all (a typo in an import otherwise only shows up
   much later, the first time something reaches into that module),
2. that the public interface behaves like an interface — abstract classes cannot be
   instantiated, and stubs raise instead of returning ``None``,
3. that the package metadata has not drifted away from the production runtime.

Point 2 matters more than it looks: a method marked ``@abstractmethod`` whose body is
only a docstring returns ``None`` when called. Anything wired to such an interface
would quietly receive ``None`` instead of an error.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

import dbextractors

# --- Importing everything -------------------------------------------------

MODULES = [
    "dbextractors",
    "dbextractors.entrypoint",
    "dbextractors.core",
    "dbextractors.core.coerce",
    "dbextractors.core.config",
    "dbextractors.core.hashing",
    "dbextractors.core.logging",
    "dbextractors.core.naming",
    "dbextractors.core.partitioning",
    "dbextractors.core.reading",
    "dbextractors.core.retry",
    "dbextractors.core.secrets",
    "dbextractors.core.status",
    "dbextractors.core.target_conn",
    "dbextractors.core.target_pg",
    "dbextractors.core.tunnel",
    "dbextractors.core.strategies",
    "dbextractors.core.strategies.base",
    "dbextractors.core.strategies.full",
    "dbextractors.core.strategies.full_by_source",
    "dbextractors.core.strategies.hash_diff",
    "dbextractors.core.strategies.id_watermark",
    "dbextractors.core.strategies.incremental",
    "dbextractors.core.strategies.parent_incremental",
    "dbextractors.dialects",
    "dbextractors.dialects.base",
    "dbextractors.dialects.firebird",
    "dbextractors.dialects.mssql",
    "dbextractors.dialects.mysql",
    "dbextractors.dialects.postgres",
    # The golden test. Unlike the rest of the package this was never a skeleton —
    # it is finished and has its own tests in tests/golden/.
    "dbextractors.golden",
    "dbextractors.golden.cli",
    "dbextractors.golden.compare",
    "dbextractors.golden.deviations",
    "dbextractors.golden.introspect",
    "dbextractors.golden.model",
    "dbextractors.golden.progress",
    "dbextractors.golden.report",
    "dbextractors.golden.runners",
    "dbextractors.golden.scratch",
    "dbextractors.golden.session",
    "dbextractors.golden.sqlgen",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_can_be_imported(name: str) -> None:
    importlib.import_module(name)


def test_module_list_is_complete() -> None:
    """A new module has to appear in ``MODULES`` too, or it is never tested.

    Without this it would be possible to add a module, forget to list it here and
    never find out that it does not import.
    """
    found = {
        info.name
        for info in pkgutil.walk_packages(dbextractors.__path__, prefix="dbextractors.")
        if not info.name.rpartition(".")[2].startswith("_")
    }
    assert found | {"dbextractors"} == set(MODULES)


# --- The interface behaves like an interface ------------------------------


def test_source_dialect_cannot_be_instantiated() -> None:
    from dbextractors.dialects.base import SourceDialect

    with pytest.raises(TypeError):
        SourceDialect()  # type: ignore[abstract]


def test_load_strategy_cannot_be_instantiated() -> None:
    from dbextractors.core.strategies.base import LoadStrategy

    with pytest.raises(TypeError):
        LoadStrategy()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("dbextractors.dialects.mysql", "MySQLDialect"),
        ("dbextractors.dialects.mssql", "MSSQLDialect"),
        ("dbextractors.dialects.postgres", "PostgresDialect"),
        ("dbextractors.dialects.firebird", "FirebirdDialect"),
    ],
)
def test_dialect_has_its_metadata_filled_in(module: str, cls: str) -> None:
    """Name, port and driver are not stubs — they had to be filled in from the start.

    The driver string is checked against what the predecessor extractors used
    verbatim.
    """
    dialect = getattr(importlib.import_module(module), cls)()
    assert dialect.name in {"mysql", "mssql", "postgres", "firebird"}
    assert dialect.default_port > 0
    assert "+" in dialect.sqlalchemy_driver


DRIVERS = {
    "mysql": ("mysql+mysqlconnector", 3306),
    "mssql": ("mssql+pymssql", 1433),
    "postgres": ("postgresql+psycopg2", 5432),
    "firebird": ("firebird+fdb", 3050),
}


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("dbextractors.dialects.mysql", "MySQLDialect"),
        ("dbextractors.dialects.mssql", "MSSQLDialect"),
        ("dbextractors.dialects.postgres", "PostgresDialect"),
        ("dbextractors.dialects.firebird", "FirebirdDialect"),
    ],
)
def test_driver_and_port_match_the_predecessors(module: str, cls: str) -> None:
    dialect = getattr(importlib.import_module(module), cls)()
    expected_driver, expected_port = DRIVERS[dialect.name]
    assert dialect.sqlalchemy_driver == expected_driver
    assert dialect.default_port == expected_port


def test_firebird_cannot_do_hash_diff() -> None:
    """Firebird has no hash mode at all in any predecessor.

    It must fail during validation rather than halfway through a run — hence
    ``UnsupportedFeature``.
    """
    from dbextractors.dialects.base import FEATURE_HASH_DIFF, UnsupportedFeature
    from dbextractors.dialects.firebird import FirebirdDialect

    dialect = FirebirdDialect()
    assert dialect.supports(FEATURE_HASH_DIFF) is False

    with pytest.raises(UnsupportedFeature):
        dialect.require(FEATURE_HASH_DIFF)

    with pytest.raises(UnsupportedFeature):
        dialect.render_hash_expr(["a", "b"], "row_hash")


def test_mssql_is_the_only_dialect_with_partitioning() -> None:
    """See ARCHITECTURE.md: MSSQL alone has multi-source and partitioning."""
    from dbextractors.dialects.base import FEATURE_PARTITION_BY_SOURCE
    from dbextractors.dialects.firebird import FirebirdDialect
    from dbextractors.dialects.mssql import MSSQLDialect
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    with_partitioning = [
        d.name
        for d in (MySQLDialect(), MSSQLDialect(), PostgresDialect(), FirebirdDialect())
        if d.supports(FEATURE_PARTITION_BY_SOURCE)
    ]
    assert with_partitioning == ["mssql"]


# --- Quoting --------------------------------------------------------------
# An agreed departure: `render_column_expr` is concrete and derives from
# `quote_ident`. This test checks that it holds for all four dialects.


def test_quoting_follows_the_dialect() -> None:
    from dbextractors.dialects.firebird import FirebirdDialect
    from dbextractors.dialects.mssql import MSSQLDialect
    from dbextractors.dialects.mysql import MySQLDialect
    from dbextractors.dialects.postgres import PostgresDialect

    assert MySQLDialect().quote_ident("price") == "`price`"
    assert MSSQLDialect().quote_ident("price") == "[price]"
    assert PostgresDialect().quote_ident("price") == '"price"'
    assert FirebirdDialect().quote_ident("price") == '"price"'


def test_render_select_clause_uses_quote_ident() -> None:
    from dbextractors.dialects.mssql import MSSQLDialect

    assert MSSQLDialect().render_select_clause(["a", "b"]) == "[a], [b]"


def test_surrogate_key_overrides_the_expression() -> None:
    from dbextractors.dialects.base import SurrogateKey
    from dbextractors.dialects.mysql import MySQLDialect

    surrogate = SurrogateKey(enabled=True, alias="sk", expr="CONCAT(a, b)")
    dialect = MySQLDialect()
    assert dialect.render_column_expr("sk", surrogate) == "CONCAT(a, b) AS `sk`"
    # It must not leak onto any other column.
    assert dialect.render_column_expr("other", surrogate) == "`other`"


def test_disabled_surrogate_key_has_no_effect() -> None:
    from dbextractors.dialects.base import SurrogateKey
    from dbextractors.dialects.mysql import MySQLDialect

    surrogate = SurrogateKey(enabled=False, alias="sk", expr="CONCAT(a, b)")
    assert MySQLDialect().render_column_expr("sk", surrogate) == "`sk`"


# --- Stubs raise, they do not return None ---------------------------------


def test_run_rejects_an_empty_configuration() -> None:
    """``dbextractors.run`` is no longer a stub — it assembles a real run.

    An empty configuration therefore fails validation in `core.config` rather than
    with ``NotImplementedError``.
    """
    from dbextractors.core.config import ConfigError

    with pytest.raises(ConfigError):
        dbextractors.run({}, dialect="mysql")


def test_no_module_is_a_stub_any_more() -> None:
    """The skeleton is fully filled in — `core.status` was the last module with stubs.

    This test used to guard the opposite: that a stub **raises**
    ``NotImplementedError`` instead of quietly returning ``None``. Module after module
    dropped out of it (``config``, ``retry``, ``tunnel``, ``coerce``, then ``naming``,
    ``target_pg``, ``hashing``), and ``status`` was finished last, so the list ended up
    empty. Instead of an empty list, what remains here is a check that the function
    finished last really does compute something.

    If a new stub is ever added to the package, its check belongs back in here — the
    test should not be deleted.
    """
    from dbextractors.core import status

    assert status.fmt_duration(61.0) == "1m 1s"

    unfinished = [
        f"{module.__name__}.{name}"
        for module in (status,)
        for name, fn in vars(module).items()
        if callable(fn) and getattr(fn, "__doc__", "") and "TODO(" in (fn.__doc__ or "")
    ]
    assert not unfinished, f"Unfinished functions remain: {unfinished}"


# --- Runtime --------------------------------------------------------------


def test_runs_on_the_production_python() -> None:
    """The production Mage image is Python 3.10 and cannot be upgraded."""
    assert sys.version_info[:2] == (3, 10)


def test_package_version_matches_the_metadata() -> None:
    from importlib.metadata import version

    assert dbextractors.__version__ == version("dbextractors")
