"""Checks that the condition over the source uses names in the form the source knows."""

from __future__ import annotations

from datetime import date

from dbextractors.core.strategies.base import LoadContext, TargetRef
from dbextractors.core.strategies.parent_incremental import ParentIncrementalStrategy
from dbextractors.dialects.base import TableRef
from dbextractors.dialects.firebird import FirebirdDialect


def _ctx(**extra_settings) -> LoadContext:
    settings = {
        "incremental_parent_table": "ORDERS",
        "incremental_parent_date_column": "CORRECTED_AT",
        "incremental_parent_date_column_fallback": "CREATED_AT",
    }
    settings.update(extra_settings)
    return LoadContext(
        dialect=FirebirdDialect(),
        engine=None,
        source=TableRef(name="ORDER_LINES"),
        target=TargetRef(schema="raw_source", table="order_lines"),
        target_conn=None,
        columns=[],
        settings=settings,
        table_cfg={},
        where=None,
    )


def test_the_parent_subquery_uses_upper_case_names() -> None:
    sql = ParentIncrementalStrategy()._source_where(_ctx(), date(2026, 1, 1))
    # This is exactly the shape that once failed in production with `Column unknown - id`.
    assert '"id"' not in sql
    assert 'SELECT "ID" FROM "ORDERS"' in sql
    assert '"PARENT_ID" IN' in sql


# --- incremental_parent_key_column / incremental_parent_id_column ------------
#
# Both were read by this strategy long before they were part of the contract,
# and `entrypoint._settings_dict` builds ``ctx.settings`` with
# ``dataclasses.asdict`` — so a key that is not a field of `LoadSettingsConfig`
# could never arrive and the defaults always won. The strategy's own error
# message named one of them, which made it advice that could not be followed.


def test_the_defaults_are_the_hardcoded_names_of_the_predecessor() -> None:
    """A configuration that says nothing must load exactly as it did before.

    ``parent_id`` / ``id`` are what the predecessor hard-codes, and adding the
    keys to the contract must not move them.
    """
    sql = ParentIncrementalStrategy()._source_where(_ctx(), date(2026, 1, 1))

    assert '"PARENT_ID" IN (SELECT "ID" FROM "ORDERS"' in sql


def test_a_configured_foreign_key_reaches_the_rendered_sql() -> None:
    """The child's reference to its parent, where it is not called ``parent_id``."""
    ctx = _ctx(incremental_parent_key_column="ORDER_ID")

    sql = ParentIncrementalStrategy()._source_where(ctx, date(2026, 1, 1))

    assert '"ORDER_ID" IN (SELECT "ID" FROM "ORDERS"' in sql
    assert "PARENT_ID" not in sql


def test_a_configured_parent_key_reaches_the_rendered_sql() -> None:
    """The parent's own key, where the parent is not keyed by ``id``."""
    ctx = _ctx(incremental_parent_id_column="ORDER_NO")

    sql = ParentIncrementalStrategy()._source_where(ctx, date(2026, 1, 1))

    assert '"PARENT_ID" IN (SELECT "ORDER_NO" FROM "ORDERS"' in sql


def test_both_keys_survive_the_journey_from_the_configuration() -> None:
    """The gap that made the keys unusable was in `entrypoint._settings_dict`.

    Rendering the SQL from a hand-written ``settings`` dict would have passed
    even while the keys were unreachable — the dict has to come from
    `config.parse` the way a real run builds it.
    """
    from dbextractors.core import config
    from dbextractors.entrypoint import _settings_dict

    parsed = config.parse(
        {
            "TABLE": {"source_name": "ORDER_LINES", "output_schema": "raw", "output_name": "ol"},
            "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
            "LOAD_SETTINGS": {
                "load_method": "incremental",
                "incremental_parent_table": "ORDERS",
                "incremental_parent_date_column": "CORRECTED_AT",
                "incremental_parent_key_column": "ORDER_ID",
                "incremental_parent_id_column": "ORDER_NO",
            },
        }
    )
    ctx = _ctx()
    ctx.settings = _settings_dict(parsed.load_settings)

    sql = ParentIncrementalStrategy()._source_where(ctx, date(2026, 1, 1))

    assert '"ORDER_ID" IN (SELECT "ORDER_NO" FROM "ORDERS"' in sql


class _RecordingConn:
    """A target connection that records the `DELETE` instead of running it.

    The delete is the **other** half of both keys: `_source_where` addresses the
    source, this addresses the target, and the two use different spellings of the
    same names (the target's are normalised by `core.naming`). A test that only
    read the source SQL would let the target half regress unnoticed — which is
    exactly what happened while these tests were being written.
    """

    rowcount = 0

    def __init__(self) -> None:
        self.sql: str = ""
        self.params: tuple = ()

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self.sql, self.params = sql, params or ()


def _delete_sql(**extra_settings) -> str:
    ctx = _ctx(**extra_settings)
    ctx.target_conn = _RecordingConn()
    ParentIncrementalStrategy()._delete_window(
        ctx, date(2026, 1, 1), TargetRef(schema="raw_source", table="orders")
    )
    return ctx.target_conn.sql


def test_the_delete_against_the_target_uses_the_default_names() -> None:
    """``parent_id`` / ``id``, normalised the way the target spells them."""
    sql = _delete_sql()

    assert '"parent_id" IN (SELECT "id" FROM "raw_source"."orders"' in sql


def test_a_configured_foreign_key_reaches_the_delete_as_well() -> None:
    """The **target** spelling, not the source one — the two are different names.

    Both go through `core.naming`, which lower-cases and, for a reserved word,
    prefixes an underscore. Addressing the target by the source's name would
    delete nothing and the run would still come out green.
    """
    sql = _delete_sql(
        incremental_parent_key_column="ORDER_ID", incremental_parent_id_column="ORDER_NO"
    )

    assert '"order_id" IN (SELECT "order_no" FROM "raw_source"."orders"' in sql
    assert "parent_id" not in sql


def test_the_keys_are_not_reported_as_unknown(caplog) -> None:
    """`config.parse` warned about them, which is how they were found.

    A key that the strategy reads and the parser calls unknown is the worst of
    both worlds: the log says it was ignored, and it really was.
    """
    from dbextractors.core import config

    with caplog.at_level("WARNING", logger="dbextractors.core.config"):
        config.parse(
            {
                "TABLE": {"source_name": "t", "output_schema": "raw", "output_name": "t"},
                "SOURCE_DB": {"user": "u", "password": "p", "database": "d"},
                "LOAD_SETTINGS": {
                    "incremental_parent_key_column": "ORDER_ID",
                    "incremental_parent_id_column": "ORDER_NO",
                },
            }
        )

    assert caplog.text == ""
