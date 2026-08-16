"""The logger Mage hands over takes no positional arguments — and dbextractors
has to survive that.

What is reproduced here is the exact failure that took down a production
pipeline: ``DictLogger.warning() takes 2 positional arguments but 3 were
given``.
"""

from __future__ import annotations

import logging

import pytest

from dbextractors.core.logging import LoggerAdapter, adapt


class DictLoggerLookalike:
    """The signature Mage's ``DictLogger`` has — a message and **only** keyword
    arguments."""

    def __init__(self) -> None:
        self.records: list = []

    def _store(self, level: str, message, **kwargs) -> None:
        self.records.append((level, message, kwargs))

    def debug(self, message, **kwargs) -> None:
        self._store("debug", message, **kwargs)

    def info(self, message, **kwargs) -> None:
        self._store("info", message, **kwargs)

    def warning(self, message, **kwargs) -> None:
        self._store("warning", message, **kwargs)

    def error(self, message, **kwargs) -> None:
        self._store("error", message, **kwargs)

    def exception(self, message, **kwargs) -> None:
        self._store("exception", message, **kwargs)


def test_the_lookalike_really_does_fail_without_the_adapter() -> None:
    """A check that the test tests what it thinks it tests."""
    logger = DictLoggerLookalike()

    with pytest.raises(TypeError, match="positional"):
        logger.warning("source=%s", "erp_2025")  # type: ignore[call-arg]


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "exception", "critical"])
def test_the_adapter_interpolates_the_arguments(level: str) -> None:
    logger = DictLoggerLookalike()
    adapter = adapt(logger)

    getattr(adapter, level)("source=%s, rows=%d", "erp_2025", 42)

    # The lookalike has no `critical` — that falls back to `info`, not to an
    # exception.
    expected_level = level if hasattr(logger, level) else "info"
    assert logger.records == [(expected_level, "source=erp_2025, rows=42", {})]


def test_keyword_arguments_pass_through_unchanged() -> None:
    """Mage uses keyword arguments to carry things like `error=` into its JSON
    output."""
    logger = DictLoggerLookalike()

    adapt(logger).error("%s failed", "the source", error="ValueError")

    assert logger.records == [("error", "the source failed", {"error": "ValueError"})]


def test_a_message_without_arguments_is_left_alone() -> None:
    logger = DictLoggerLookalike()

    adapt(logger).warning("100 % done")

    assert logger.records == [("warning", "100 % done", {})]


def test_a_broken_format_string_does_not_bring_the_run_down() -> None:
    """A typo in a format string must not cost an extraction that otherwise
    finished."""
    logger = DictLoggerLookalike()

    adapt(logger).warning("a placeholder is missing", "extra")

    level, message, _ = logger.records[0]
    assert level == "warning"
    assert "a placeholder is missing" in message
    assert "extra" in message


def test_a_standard_logger_is_not_wrapped() -> None:
    """`logging.Logger` handles positional arguments, and its lazy formatting is
    an advantage there is no reason to give up."""
    logger = logging.getLogger("test.dbextractors.do_not_wrap")

    assert adapt(logger) is logger


def test_none_stays_none() -> None:
    assert adapt(None) is None


def test_wrapping_twice_is_not_possible() -> None:
    adapter = adapt(DictLoggerLookalike())

    assert isinstance(adapter, LoggerAdapter)
    assert adapt(adapter) is adapter


def test_the_entrypoint_wraps_the_logger_right_at_the_door() -> None:
    """Without that the first `%s` message would bring the run down before
    anything was transferred at all."""
    from dbextractors import entrypoint
    from dbextractors.core.config import ConfigError

    logger = DictLoggerLookalike()

    # The configuration is empty, so validation fails — the point is that it
    # fails **there**, not on a `TypeError` out of logging.
    with pytest.raises(ConfigError):
        entrypoint.run({}, dialect="mysql", logger=logger)


# --- Progress has to be visible ----------------------------------------------
#
# One pipeline went 13.7 minutes without a word, while the old side reported
# every batch with an `n/12` alongside. On a run that takes tens of minutes
# there is then no way to tell whether it is working or stuck.


def test_incremental_staging_reports_progress() -> None:
    """The batch loop has to tick, not stay silent until the end."""
    from dbextractors.core.status import BatchProgress

    messages: list = []
    progress = BatchProgress(
        lambda level, message, *args: messages.append(str(message) % args if args else message),
        total_rows=4_865_972,
        phase="staging",
    )

    progress.tick(500_000)
    progress.tick(1_000_000)

    assert len(messages) == 2
    # Which batch, how many out of how many, and what share we are at.
    assert "batch 2" in messages[1]
    assert "1,000,000" in messages[1]
    assert "4,865,972" in messages[1]
    assert "20.6%" in messages[1]


def test_debug_prints_the_type_maps_only_when_it_is_switched_on() -> None:
    """`DEBUG` governs what is printed — never what is done."""
    from types import SimpleNamespace

    from dbextractors.entrypoint import _log_columns

    def ctx(debug: bool):
        messages: list = []
        return messages, SimpleNamespace(
            log=lambda level, message, *args: messages.append(
                str(message) % args if args else str(message)
            ),
            target_names=["id", "_name"],
            overwrite_types={"id": "TEXT", "_name": "TEXT"},
            orig_type_map={"id": "varchar", "_name": "varchar"},
            surrogate=None,
            where=None,
            debug=debug,
        )

    quiet, c = ctx(False)
    _log_columns(c)
    assert len(quiet) == 1
    assert "Columns after selection (2)" in quiet[0]
    assert not any("DEBUG" in m for m in quiet)

    loud, c = ctx(True)
    _log_columns(c)
    assert any("overwrite_types" in m for m in loud)
    assert any("orig_type_map" in m for m in loud)
