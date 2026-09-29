"""How the package logs, and how a run's records reach the host's logger.

The package logs to standard loggers under ``dbextractors`` at the level each
message deserves (see `core.logging`). A host that passes ``run(logger=...)`` —
Dagster's ``context.log``, say — gets the run's records forwarded to it. These
tests pin the forwarding: that ``INFO`` gets through although the root logger
defaults to ``WARNING``, that nothing is printed twice, that another thread's
records stay out, and that everything is put back afterwards.
"""

from __future__ import annotations

import logging
import threading

import pytest

from dbextractors.core.logging import PACKAGE_LOGGER, forward_to

_pkg = logging.getLogger(PACKAGE_LOGGER)
_child = logging.getLogger(f"{PACKAGE_LOGGER}.core.test_logging_probe")


class Collecting(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelname, record.getMessage()))


@pytest.fixture()
def host() -> tuple:
    """A host logger outside the package, as Dagster's would be."""
    logger = logging.getLogger("test.dbextractors.host")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Collecting()
    logger.addHandler(handler)
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)


def test_info_reaches_the_host_and_keeps_its_level(host) -> None:
    logger, handler = host

    with forward_to(logger):
        _child.info("🚀 Full load: %s", "orders")
        _child.warning("⚠️ Falling back to a full load: %s", "empty target")

    assert handler.records == [
        ("INFO", "🚀 Full load: orders"),
        ("WARNING", "⚠️ Falling back to a full load: empty target"),
    ]


def test_a_host_logger_with_no_level_gets_everything() -> None:
    """Dagster's ``context.log`` is a `logging.Logger` built outside the hierarchy
    and left at ``NOTSET``, so its effective level is 0.

    Taking that 0 as the package level means "inherit", i.e. the root's
    ``WARNING`` — and every INFO record was dropped before it could be
    forwarded. That is what 2.1.0 did under Dagster: warnings came through,
    progress did not.
    """
    target = logging.Logger("dagster")  # no parent, NOTSET — like DagsterLogManager
    handler = Collecting()
    target.addHandler(handler)
    assert target.getEffectiveLevel() == logging.NOTSET

    with forward_to(target):
        _child.debug("sql")
        _child.info("progress")
        _child.warning("fallback")

    assert handler.records == [
        ("DEBUG", "sql"),
        ("INFO", "progress"),
        ("WARNING", "fallback"),
    ]


def test_a_real_dagster_log_manager_gets_info() -> None:
    """The same against the real thing, where Dagster is installed."""
    pytest.importorskip("dagster")
    from dagster._core.log_manager import DagsterLogManager

    received: list = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            received.append((record.levelname, record.getMessage()))

    manager = DagsterLogManager.create(loggers=[], handlers=[_Handler()])
    assert manager.getEffectiveLevel() == logging.NOTSET

    with forward_to(manager):
        _child.info("progress")

    assert any(level == "INFO" and "progress" in message for level, message in received)


def test_the_host_level_decides_what_is_forwarded(host) -> None:
    logger, handler = host
    logger.setLevel(logging.INFO)

    with forward_to(logger):
        _child.debug("SELECT …")
        _child.info("rows: %d", 3)

    assert handler.records == [("INFO", "rows: 3")]


def test_forwarded_records_do_not_also_propagate(host) -> None:
    """A host that captures standard loggers *and* passes one would see every line twice."""
    logger, _ = host
    root = Collecting()
    logging.getLogger().addHandler(root)
    try:
        with forward_to(logger):
            _child.warning("once")
    finally:
        logging.getLogger().removeHandler(root)

    assert root.records == []


def test_everything_is_put_back_afterwards(host) -> None:
    logger, _ = host
    before = (list(_pkg.handlers), _pkg.level, _pkg.propagate)

    with pytest.raises(RuntimeError), forward_to(logger):
        raise RuntimeError("the run failed")

    assert (list(_pkg.handlers), _pkg.level, _pkg.propagate) == before


def test_another_threads_records_stay_out(host) -> None:
    """Two runs in two threads of one process must not end up in each other's log."""
    logger, handler = host

    with forward_to(logger):
        other = threading.Thread(target=lambda: _child.warning("from another run"))
        other.start()
        other.join()
        _child.warning("from this run")

    assert handler.records == [("WARNING", "from this run")]


def _host(name: str) -> tuple:
    logger = logging.getLogger(f"test.dbextractors.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = Collecting()
    logger.addHandler(handler)
    return logger, handler


def test_two_runs_in_two_threads_do_not_undo_each_other() -> None:
    """Run A starts, run B starts, A ends while B is still going — B must keep its INFO.

    Each run saving and restoring the shared logger's state on its own would let
    A's exit reset the level to ``WARNING`` under B, or leave it lowered for good.
    """
    first, first_seen = _host("first")
    second, second_seen = _host("second")
    before = (list(_pkg.handlers), _pkg.level, _pkg.propagate)
    b_started, a_ended = threading.Event(), threading.Event()

    def run_b() -> None:
        with forward_to(second):
            b_started.set()
            a_ended.wait(5)
            _child.info("b after a ended")

    other = threading.Thread(target=run_b)
    with forward_to(first):
        other.start()
        b_started.wait(5)
        _child.info("a")
    a_ended.set()
    other.join(5)

    assert first_seen.records == [("INFO", "a")]
    assert second_seen.records == [("INFO", "b after a ended")]
    assert (list(_pkg.handlers), _pkg.level, _pkg.propagate) == before


def test_a_thread_that_forwards_nothing_propagates_as_before(host) -> None:
    """While another run forwards, a plain thread's records still reach the root —
    at the level they would have anyway, not the lowered one."""
    logger, _ = host
    root = Collecting()
    logging.getLogger().addHandler(root)
    try:
        with forward_to(logger):

            def plain() -> None:
                _child.debug("not before, not now")
                _child.warning("as before")

            other = threading.Thread(target=plain)
            other.start()
            other.join(5)
    finally:
        logging.getLogger().removeHandler(root)

    assert root.records == [("WARNING", "as before")]


def test_a_logger_without_positional_arguments_gets_finished_text() -> None:
    """A host logger that is not a `logging.Logger` gets the message already formatted."""
    received: list = []

    class Minimal:
        def info(self, message) -> None:
            received.append(("info", message))

        def warning(self, message) -> None:
            received.append(("warning", message))

    with forward_to(Minimal()):
        _child.info("source=%s", "erp")
        _child.debug("not forwarded: a plain object gets INFO and above")
        _child.critical("no critical() on the host — falls back to %s", "info")

    assert received == [
        ("info", "source=erp"),
        ("info", "no critical() on the host — falls back to info"),
    ]


def test_a_broken_format_string_does_not_bring_the_run_down(host) -> None:
    logger, handler = host

    with forward_to(logger):
        _child.warning("a placeholder is missing", "extra")

    level, message = handler.records[0]
    assert level == "WARNING"
    assert "a placeholder is missing" in message and "extra" in message


def test_none_and_the_packages_own_logger_forward_nothing() -> None:
    before = list(_pkg.handlers)

    with forward_to(None):
        assert _pkg.handlers == before
    with forward_to(_child):
        assert _pkg.handlers == before, "forwarding to itself would loop"


def test_the_package_has_a_null_handler() -> None:
    import dbextractors  # noqa: F401 — the import is what installs it

    assert any(isinstance(h, logging.NullHandler) for h in _pkg.handlers)


def test_run_forwards_to_the_host_logger(host) -> None:
    """The entry point opens the forwarding before it does anything else."""
    from dbextractors import entrypoint

    logger, handler = host
    config = {
        "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
        "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
        "LOAD_SETTINGS": {"load_method": "full", "no_such_key": 1},
    }

    # The configuration is parsed first, and the unknown key logged; the unknown
    # dialect then stops the run before anything touches a database.
    with pytest.raises(Exception, match="no_such_dialect"):
        entrypoint.run(config, dialect="no_such_dialect", logger=logger)

    assert ("WARNING", "Unknown key 'no_such_key' in the LOAD_SETTINGS section — ignoring.") in (
        handler.records
    )


# --- Progress has to be visible ----------------------------------------------
#
# One pipeline went 13.7 minutes without a word, while the old side reported
# every batch with an `n/12` alongside. On a run that takes tens of minutes
# there is then no way to tell whether it is working or stuck.


def test_incremental_staging_reports_progress(caplog) -> None:
    """The batch loop has to tick at INFO, not stay silent until the end."""
    from dbextractors.core.status import BatchProgress

    progress = BatchProgress(_child, total_rows=4_865_972, phase="staging")
    with caplog.at_level(logging.INFO, logger=PACKAGE_LOGGER):
        progress.tick(500_000)
        progress.tick(1_000_000)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(messages) == 2
    # Which batch, how many out of how many, and what share we are at.
    assert "batch 2" in messages[1]
    assert "1,000,000" in messages[1]
    assert "4,865,972" in messages[1]
    assert "20.6%" in messages[1]


def test_debug_prints_the_type_maps_only_when_it_is_switched_on(caplog) -> None:
    """`DEBUG` governs what is printed — never what is done. The type maps go at DEBUG."""
    from types import SimpleNamespace

    from dbextractors.entrypoint import _log_columns

    def ctx(debug: bool):
        return SimpleNamespace(
            target_names=["id", "_name"],
            overwrite_types={"id": "TEXT", "_name": "TEXT"},
            orig_type_map={"id": "varchar", "_name": "varchar"},
            surrogate=None,
            where=None,
            debug=debug,
        )

    with caplog.at_level(logging.DEBUG, logger=PACKAGE_LOGGER):
        _log_columns(ctx(False))
    quiet = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert len(quiet) == 1
    assert quiet[0][0] == "INFO"
    assert "Columns after selection (2)" in quiet[0][1]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=PACKAGE_LOGGER):
        _log_columns(ctx(True))
    loud = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert any(level == "DEBUG" and "overwrite_types" in m for level, m in loud)
    assert any(level == "DEBUG" and "orig_type_map" in m for level, m in loud)
