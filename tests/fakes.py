"""Fake dialect and context builders for the strategy tests.

A strategy must not know the SQL dialect — which means it can be tested with a
fake one. `FakeDialect` hands out prepared batches instead of touching a source,
so these tests only need the target PostgreSQL. Source databases (MySQL, MSSQL,
Firebird) are exercised by the dialect tests instead.

This lives in ``tests/`` rather than ``tests/strategies/`` because ``tests/`` is on
``sys.path`` (see ``tests/conftest.py``) and can therefore be imported when the
whole suite runs, not just one subdirectory.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import pandas as pd

from dbextractors.core.strategies.base import ColumnDef, LoadContext, TableRef, TargetRef
from dbextractors.dialects.base import SizeEstimate, SourceDialect

#: Prefix of the hashes the fake source computes "on its side".
#:
#: It exists so that the source path is **distinguishable** from the pandas path.
#: The two real paths genuinely produce different digests (see the module
#: docstring of ``core/hashing.py`` — 4 of 51 tables agree), and the contract is
#: that the seed and the diff both take the *source* path. When the fake computed
#: its hash with the very same ``hashing.row_hash`` the pandas path uses, a
#: mutation that forced the seed onto the pandas path (``force_recompute=True``
#: in ``full._prepare``) was invisible: both sides produced the same string and
#: every diff still settled. With the prefix, any switch of path shows up as a
#: full-table mismatch.
SOURCE_HASH_PREFIX = "src:"


def source_row_hash(values: Sequence) -> str:
    """The row hash as the fake *source database* computes it.

    Deliberately **not** equal to ``hashing.row_hash`` alone — see
    `SOURCE_HASH_PREFIX`. Tests that pre-fill a target with "hashes the source
    would have produced" must use this function, not ``hashing.row_hash``.
    """
    from dbextractors.core import hashing

    return SOURCE_HASH_PREFIX + hashing.row_hash(values)


class FakeDialect(SourceDialect):
    """A dialect that returns prepared batches instead of reading a source.

    It records the SQL it was handed, so a test can check that the strategy
    assembles the query the way it should — for instance that the hash expression
    is appended after the column list, and that the incremental window carries the
    right condition.
    """

    name = "fake"
    default_port = 1
    sqlalchemy_driver = "fake+fake"
    quote_char = "`"
    FEATURES = frozenset({"hash_diff", "keyset", "parent_incremental", "partition_by_source"})
    zero_datetime_literal = "0000-00-00 00:00:00"
    text_like_types = frozenset({"varchar", "text"})

    def __init__(
        self, batches: Optional[Sequence[pd.DataFrame]] = None, rows: Optional[int] = None
    ):
        self.batches: List[pd.DataFrame] = list(batches or [])
        self._rows = rows
        self.seen_sql: List[str] = []
        self.estimate_calls: List[Optional[str]] = []
        self.estimate_error: Optional[Exception] = None

    # -- what the strategies really call ------------------------------------

    def estimate_size(self, engine, ref, where=None, *, known_total_rows=None, sample_size=100):
        self.estimate_calls.append(where)
        if self.estimate_error is not None:
            raise self.estimate_error
        rows = self._rows if self._rows is not None else sum(len(b) for b in self.batches)
        return SizeEstimate(rows=rows, size_mb=0.5, method="fake")

    def render_select(
        self,
        columns,
        ref,
        where=None,
        order_by=None,
        surrogate=None,
        *,
        column_types=None,
        convert_nchar=False,
    ):
        # The conversion arguments are forwarded rather than dropped. The fake
        # declares no `NCHAR_TYPES`, so the clause comes out unchanged — and that
        # is the claim worth keeping true: a strategy passing them must not alter
        # the SQL of a dialect that has no national-character types.
        clause = self.render_select_clause(
            columns, surrogate, column_types=column_types, convert_nchar=convert_nchar
        )
        sql = f"SELECT {clause} FROM {self.quote_ident(ref.name)}"
        if where:
            sql += f" WHERE {where}"
        # ORDER BY **must** be rendered: resuming a read after a lost connection
        # depends on it (`core.reading`), and a fake that dropped it would claim
        # everything was fine even if the strategy stopped emitting it.
        if order_by:
            sql += " ORDER BY " + ", ".join(self.quote_ident(c) for c in order_by)
        return sql

    def render_hash_expr(self, hashed_columns, alias, column_types=None):
        parts = ", ".join(
            f"COALESCE(CAST({self.quote_ident(c)} AS CHAR), '')" for c in hashed_columns
        )
        return f"SHA2(CONCAT_WS('||', {parts}), 256) AS {self.quote_ident(alias)}"

    def iter_batches(self, engine, sql, batch_size):
        self.seen_sql.append(sql)
        for batch in self.batches:
            yield batch.copy()

    # -- the rest of the interface; the strategies do not touch it ----------

    def build_conn_str(self, params, host, port):
        return "fake://"

    def probe(self, host, port, timeout=10.0):
        return True

    def introspect_columns(self, engine, ref):
        return []

    def map_type(self, column):
        return column.pg_type


class FakeHashSource(FakeDialect):
    """A fake source that answers a query the way a real database would.

    `HashDiffStrategy` sends the source **two different queries** — a
    ``SELECT pk, hash`` scan over the whole table, and then a
    ``... WHERE pk IN (...)`` fetch — so testing it against a dialect that keeps
    returning the same batch proves nothing: it would not show that only the
    changed rows are actually fetched.

    This fake therefore keeps the source table as a DataFrame and, depending on
    the shape of the query, returns either ``(pk, hash)`` pairs or full rows. It
    computes the hash itself, which is exactly what ``SHA2`` does in the source —
    and, crucially, it is the **same path** for the scan and for the fetch, since
    otherwise the diff would never settle.

    The source table can be mutated (`update`, `delete`, `insert`), so the second
    run sees different data from the first.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        pk: str,
        hash_alias: str = "row_hash",
        parents: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.rows = rows.reset_index(drop=True)
        self.pk = pk
        self.hash_alias = hash_alias
        #: The parent table **in the source** — needed by `parent_incremental`,
        #: which narrows the child through a subquery over the parent.
        self.parents = None if parents is None else parents.reset_index(drop=True)
        #: Queries split by kind, so a test can also check how many there were.
        self.scan_sql: List[str] = []
        self.download_sql: List[str] = []
        #: When set, the scan fails — simulating a connection lost mid-way.
        self.scan_error: Optional[Exception] = None

    # -- mutating the source between runs -----------------------------------

    def update(self, key, column: str, value) -> None:
        self.rows.loc[self.rows[self.pk] == key, column] = value

    def delete(self, key) -> None:
        self.rows = self.rows[self.rows[self.pk] != key].reset_index(drop=True)

    def insert(self, row: dict) -> None:
        self.rows = pd.concat([self.rows, pd.DataFrame([row])], ignore_index=True)

    # -- answering queries ---------------------------------------------------

    def _hashes(self, frame: pd.DataFrame) -> List[str]:
        """Hashes "computed by the source" — distinguishable from the pandas path.

        Both the scan and the fetch flow through here, so the diff settles as
        long as the *target* was seeded through the source path too. The prefix
        (`SOURCE_HASH_PREFIX`) is what makes a seed that secretly recomputes in
        pandas visible — see the constant's docstring.
        """
        columns = list(self.rows.columns)
        return [source_row_hash([row[c] for c in columns]) for _, row in frame.iterrows()]

    @staticmethod
    def _keys_from_sql(sql: str) -> List[str]:
        inner = sql.partition(" IN (")[2].partition(")")[0]
        return [_unquote(part) for part in inner.split(",")]

    def _filter_rows(self, where: Optional[str]) -> pd.DataFrame:
        """Narrow the source by the condition, the way a database would.

        It understands the two shapes the strategies assemble: ``pk IN (...)``
        (`hash_diff` fetching the changed rows) and ``pk > value``
        (`id_watermark`). Anything else — ``1=1`` from the configuration, say —
        passes through; the fake is not an SQL engine, and tests that would need
        one belong against a real database.
        """
        selected = self.rows
        if not where:
            return selected

        subquery = re.search(
            r"(\S+)\s+IN \(SELECT\s+(\S+)\s+FROM\s+\S+\s+WHERE\s+(.*?)\)\s*$", where, re.S
        )
        if subquery is not None:
            selected = self._filter_by_parent(selected, subquery)
        elif " IN (" in where:
            keys = self._keys_from_sql(where)
            selected = selected[selected[self.pk].astype(str).isin(keys)]

        greater = re.search(
            rf"{re.escape(self.quote_ident(self.pk))}\s*>\s*('(?:[^']|'')*'|[-\d.]+)", where
        )
        if greater:
            value = _unquote(greater.group(1))
            if pd.api.types.is_numeric_dtype(selected[self.pk]):
                selected = selected[selected[self.pk] > float(value)]
            else:
                selected = selected[selected[self.pk].astype(str) > value]

        return selected

    def _filter_by_parent(self, selected: pd.DataFrame, subquery) -> pd.DataFrame:
        """Evaluate ``child IN (SELECT id FROM parent WHERE date >= ...)``.

        `ParentIncrementalStrategy` does not narrow the child directly but through
        its parent — and without the fake understanding that subquery, the tests
        would only be checking that the strategy transferred something. The parent
        is passed in as `parents` (a DataFrame with a key column and date columns);
        without it the subquery narrows nothing.

        There can be more than one date condition, joined with ``OR`` the way
        `_parent_condition` assembles them.
        """
        if self.parents is None:
            return selected

        child_column = _unquote_ident(subquery.group(1))
        parent_id = _unquote_ident(subquery.group(2))
        mask = pd.Series(False, index=self.parents.index)
        for column, value in re.findall(
            r"[`\"]?(\w+)[`\"]?\s*>=\s*('(?:[^']|'')*'|[-\d.]+)", subquery.group(3)
        ):
            boundary = pd.Timestamp(_unquote(value))
            mask = mask | (pd.to_datetime(self.parents[column], errors="coerce") >= boundary)
        return selected[selected[child_column].isin(self.parents.loc[mask, parent_id])]

    def _je_sken(self, sql: str) -> bool:
        """A scan is recognised by there being a single column before the hash expression.

        The hash expression itself contains the names of every hashed column, so
        counting identifiers across the whole SELECT list would not work.
        """
        head = sql.partition(" FROM ")[0]
        return head.partition("SHA2(")[0].count(self.quote_char) == 2

    def iter_batches(self, engine, sql, batch_size):
        self.seen_sql.append(sql)

        if "fp_count" in sql:
            yield self._fingerprint_row(sql)
            return

        if " IN (" in sql:
            self.download_sql.append(sql)
        elif self._je_sken(sql):
            self.scan_sql.append(sql)
            if self.scan_error is not None:
                raise self.scan_error

        selected = self._filter_rows(_where_clause(sql)).copy()
        # The hash is added **only when the query asked for it**. A real database
        # will not return a column that is not in the SELECT either — and without
        # this condition `full_by_source` would look like it computes the hash
        # even when it does not.
        if "SHA2(" in sql:
            selected[self.hash_alias] = self._hashes(selected)
        if self._je_sken(sql):
            selected = selected[[self.pk, self.hash_alias]]

        step = max(1, int(batch_size))
        for start in range(0, len(selected), step):
            yield selected.iloc[start : start + step].reset_index(drop=True)

    def _fingerprint_row(self, sql: str) -> pd.DataFrame:
        """The single-row fingerprint answer, as a database would return it.

        `FullBySourceStrategy` uses it to tell whether the source changed, so it
        has to react to the real data — otherwise the "unchanged source is
        skipped" test would also pass an implementation that skips every time.
        """
        rows = self._filter_rows(sql.partition(" WHERE ")[2])
        return pd.DataFrame(
            [
                {
                    "fp_count": len(rows),
                    "fp_max_id": rows[self.pk].max() if len(rows) else None,
                    "fp_max_ts": None,
                    "fp_agg": None,
                }
            ]
        )

    def estimate_size(self, engine, ref, where=None, *, known_total_rows=None, sample_size=100):
        """The estimate must honour the condition — otherwise `id_watermark` always sees work.

        ``rows=`` in the constructor overrides the estimate; it exists so a test
        can create the contradiction "the estimate reports rows, the source
        streams none".
        """
        self.estimate_calls.append(where)
        if self.estimate_error is not None:
            raise self.estimate_error
        count = self._rows if self._rows is not None else len(self._filter_rows(where))
        return SizeEstimate(rows=count, size_mb=0.1, method="fake")


def _where_clause(sql: str) -> str:
    """The condition from a SELECT, **without** a trailing ``ORDER BY``.

    Without cutting it off, the subquery parser used by `parent_incremental` would
    stop working: it anchors on the closing parenthesis at the end of the string,
    which the ``ORDER BY`` added by resumable reading (`core.reading`) breaks.
    """
    where = sql.partition(" WHERE ")[2]
    return where.partition(" ORDER BY ")[0]


def _unquote_ident(ident: str) -> str:
    """Quoted identifier -> bare name: backticks as well as square brackets."""
    return ident.strip().strip('`"[]')


def _unquote(literal: str) -> str:
    """SQL literal -> value. ``'b''c'`` -> ``b'c``."""
    part = literal.strip()
    if part.startswith("'") and part.endswith("'"):
        return part[1:-1].replace("''", "'")
    return part


def make_columns(spec: Sequence[tuple]) -> List[ColumnDef]:
    """``[(name, source type, pg type)]`` -> columns with the target name filled in."""
    from dbextractors.core import naming

    out = []
    for ordinal, (name, source_type, pg_type) in enumerate(spec):
        out.append(
            ColumnDef(
                name=name,
                source_type=source_type,
                pg_type=pg_type,
                ordinal=ordinal,
                target_name=naming.clean_column_name(naming.create_column_mapping([name])[name]),
            )
        )
    return out


def make_context(
    conn,
    schema: str,
    *,
    dialect: FakeDialect,
    columns: Sequence[ColumnDef],
    settings: Optional[dict] = None,
    table_cfg: Optional[dict] = None,
    table: str = "cil",
    where: Optional[str] = None,
) -> LoadContext:
    return LoadContext(
        dialect=dialect,
        engine=None,
        source=TableRef(name="zdroj"),
        target=TargetRef(schema=schema, table=table),
        target_conn=conn,
        columns=list(columns),
        settings=dict(settings or {}),
        table_cfg=dict(table_cfg or {}),
        where=where,
    )
