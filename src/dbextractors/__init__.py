"""Table extraction from MySQL, MSSQL, PostgreSQL and Firebird into PostgreSQL.

    from dbextractors import run

    df = run(config, dialect="mysql")

``config`` is a plain dict with three sections — ``TABLE``, ``LOAD_SETTINGS``
and ``SOURCE_DB``. See the README for what goes in them, and ``ARCHITECTURE.md``
for how the pieces fit together.

The package logs to the standard ``dbextractors`` loggers and configures no
handlers. Pass the orchestrator's logger to have a run's records forwarded to it,
e.g. in a Dagster asset::

    run(config, dialect="mysql", logger=context.log)
"""

import logging

from dbextractors.entrypoint import run

# A library adds no handlers of its own; this only keeps Python's last-resort
# handler from printing the package's warnings to stderr when the host has not
# configured logging at all.
logging.getLogger("dbextractors").addHandler(logging.NullHandler())

__all__ = ["__version__", "run"]

__version__ = "2.0.0"
