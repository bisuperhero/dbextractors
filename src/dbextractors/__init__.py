"""Table extraction from MySQL, MSSQL, PostgreSQL and Firebird into PostgreSQL.

    from dbextractors import run

    df = run(config, dialect="mysql")

``config`` is a plain dict with three sections — ``TABLE``, ``LOAD_SETTINGS``
and ``SOURCE_DB``. See the README for what goes in them, and ``ARCHITECTURE.md``
for how the pieces fit together.

Under Mage, ``run`` is what a ``data_loader`` block calls; pass the block's
logger through so its output lands in the pipeline log::

    @data_loader
    def load_data(config, *args, **kwargs):
        return run(config, dialect="mysql", logger=kwargs.get("logger"))
"""

from dbextractors.entrypoint import run

__all__ = ["run", "__version__"]

__version__ = "1.0.0"
