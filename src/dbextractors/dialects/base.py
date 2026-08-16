"""``SourceDialect`` — the only thing you write for a new type of source database.

The principle from ARCHITECTURE.md holds unchanged: **a dialect knows nothing
about the target.** It can say what columns a table has, how to read from it in
batches and how its types map onto PostgreSQL. What happens to the data
afterwards is none of its business.

There are four deviations from the design in ARCHITECTURE.md. All of them were
agreed and are backed by evidence from the predecessors:

1. ``estimate_size`` returns ``SizeEstimate``, not ``tuple[float, int]``
2. ``render_hash_expr`` lives here, not in ``core/hashing.py``
3. ``map_type()`` is a method, not a ``type_map: dict``
4. ``iter_batches()`` was added
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from sqlalchemy.engine import Engine

#: Names accepted by ``dbextractors.run(dialect=...)``.
KNOWN_DIALECTS: tuple[str, ...] = ("mysql", "mssql", "postgres", "firebird")

#: Capabilities that can be asked about through ``supports()``.
FEATURE_KEYSET = "keyset"
FEATURE_HASH_DIFF = "hash_diff"
FEATURE_PARTITION_BY_SOURCE = "partition_by_source"
FEATURE_PARENT_INCREMENTAL = "parent_incremental"


class UnsupportedFeature(NotImplementedError):
    """The dialect does not have this capability — Firebird and ``hash_diff``, say.

    A dedicated type rather than a bare ``NotImplementedError`` so that it can be
    caught during configuration validation, before the source is touched at all.
    """


# --- Value types on the interface -----------------------------------------


@dataclass(frozen=True)
class TableRef:
    """The address of a table in the source.

    Introduced because ``estimate_result_size_mb`` has **6 different signatures**
    across the predecessors and names the table in three different ways:
    ``table_or_view_name`` (MySQL, Firebird), ``table_sql_name`` (the older
    PostgreSQL loaders) and the pair ``source_schema`` + ``table_name`` (the
    PostgreSQL streaming extractor).
    """

    name: str
    schema: Optional[str] = None
    database: Optional[str] = None


@dataclass(frozen=True)
class ColumnDef:
    """One column of the source table as introspection sees it.

    ``source_type`` is the raw source type (``varchar``, ``datetime``, a numeric
    code in Firebird). ``pg_type`` is the result of mapping it onto the target;
    the dialect fills it in, because only the dialect knows what its own types
    mean.

    ``target_name`` is the name after normalisation for the target. **The dialect
    leaves it None.** ``core.naming`` fills it in, and the frozen naming rule
    applies to it: an underscore prefix for the 825 reserved words the dbt layer
    is built on.
    """

    name: str
    source_type: str
    pg_type: str
    nullable: bool = True
    ordinal: int = 0
    #: Firebird needs the subtype and scale, MSSQL the length of binary types,
    #: MySQL the unsigned flag. It goes in here so the signature does not swell.
    meta: dict = field(default_factory=dict)
    target_name: Optional[str] = None


@dataclass(frozen=True)
class SizeEstimate:
    """The result of estimating the size of the source.

    A failed estimate **must fail loudly**. Never return ``rows=0`` — combined
    with ``empty_rows_ok`` that overwrites the target table with nothing and the
    run still goes green. That is why there is no default here and why ``method``
    is mandatory: the log and the report have to show what the number was
    measured with.
    """

    rows: int
    size_mb: float
    method: str  # 'count_and_sample' | 'metadata' | 'known_total_rows'


@dataclass(frozen=True)
class SurrogateKey:
    """Surrogate key configuration (``surrogate_key_enabled``/``_definition``)."""

    enabled: bool
    alias: Optional[str] = None
    expr: Optional[str] = None


# --- The interface ---------------------------------------------------------


class SourceDialect(ABC):
    """Adapter for one type of source database."""

    #: 'mysql' | 'mssql' | 'postgres' | 'firebird'
    name: str
    #: The default port, used when the configuration does not give one.
    default_port: int
    #: SQLAlchemy dialect+driver exactly as the predecessors use it:
    #: mysql+mysqlconnector | mssql+pymssql | postgresql+psycopg2 | firebird+fdb
    sqlalchemy_driver: str
    #: The character identifiers are quoted with. MSSQL overrides `quote_ident`.
    quote_char: str = '"'
    #: Which optional capabilities the dialect has.
    FEATURES: frozenset[str] = frozenset()

    #: Encoding order for `coerce.decode_bytes_columns`. **This is not a global
    #: winner.** The cascade `cp1250 -> utf-8` must not be generalised: cp1250
    #: rejects only five bytes, so UTF-8 text passes through it as mojibake.
    #: MSSQL and Firebird override it.
    decode_encodings: tuple = ("utf-8",)

    #: Types to be read as text even though they look numeric. Without this,
    #: pandas infers `float64` for an eighteen-digit EAN in a `varchar` column
    #: and the number loses precision.
    text_like_types: frozenset[str] = frozenset()

    #: National-character types whose value the client encoding cannot carry
    #: intact, and which ``LOAD_SETTINGS.convert_nchar_to_varchar`` therefore
    #: converts on the server. **Empty means the dialect has no such problem** —
    #: only MSSQL fills it in, because only MSSQL is read over a single-byte
    #: connection charset it does not agree with the driver about. Reading it is
    #: also how `LoadStrategy.validate` can tell that the key would do nothing
    #: here and say so instead of ignoring it in silence.
    NCHAR_TYPES: frozenset[str] = frozenset()

    #: How the source writes a "zero" date that is not NULL. MySQL can store
    #: `0000-00-00 00:00:00`; the incremental window has to flatten it to NULL,
    #: otherwise `COALESCE` would never reach for `created_at`. The other
    #: dialects have no zero date and leave `None`.
    zero_datetime_literal: Optional[str] = None

    #: Is ``row_hash`` computed in pandas instead of on the source side? For
    #: PostgreSQL it is, and **that is not a choice**: the pandas digest is the
    #: one stored in the target today, and the SQL path produces a different one
    #: (4 of 51 tables agree). The strategy picks its path from this — and above
    #: all uses the **same** one for seeding and for the diff.
    hash_in_pandas: bool = False

    #: What is passed into ``create_engine(connect_args=…)``. Empty for dialects
    #: happy with the driver's own behaviour. It is deliberately not part of the
    #: URL — these are driver parameters, not an address.
    connect_args: ClassVar[dict] = {}

    #: SQL run immediately after a connection is established. For session
    #: settings that cannot be passed through ``connect_args``.
    session_sql: ClassVar[tuple] = ()

    # -- connecting --------------------------------------------------------

    @abstractmethod
    def build_conn_str(self, params: dict, host: str, port: int) -> str:
        """Build the SQLAlchemy URL. The password must be URL-encoded."""

    @abstractmethod
    def probe(self, host: str, port: int, timeout: float = 10.0) -> bool:
        """Is the source reachable? (the predecessors' ``is_*_accessible``)"""

    # -- introspection -----------------------------------------------------

    @abstractmethod
    def introspect_columns(self, engine: Engine, ref: TableRef) -> list[ColumnDef]:
        """Return the table's columns, including the mapped ``pg_type``.

        MySQL, MSSQL and PostgreSQL go through ``information_schema``, Firebird
        through ``RDB$``.
        """

    @abstractmethod
    def map_type(self, column: ColumnDef) -> str:
        """Source type -> PostgreSQL type.

        **Deviation 3** from ARCHITECTURE.md, where this is a
        ``type_map: dict[str, str]``. For Firebird a dict cannot express it:
        ``_fb_type_to_pg(fb_type, fb_subtype, fb_scale)`` decides on the sign of
        ``scale`` — type 7 (SMALLINT) is ``NUMERIC`` when ``scale < 0`` and
        ``INTEGER`` otherwise.

        Three dialects out of four stay declarative through
        ``DictTypeMapDialect``.
        """

    @abstractmethod
    def estimate_size(
        self,
        engine: Engine,
        ref: TableRef,
        where: Optional[str] = None,
        *,
        known_total_rows: Optional[int] = None,
        sample_size: int = 100,
    ) -> SizeEstimate:
        """Estimate the row count and the volume of data.

        **Deviation 1** from ARCHITECTURE.md (``-> tuple[float, int]`` there). A
        named type unifies 6 signatures and makes it impossible to swap the two
        members of the pair.

        ``known_total_rows`` is used only by MSSQL today (it saves a second
        ``COUNT(*)``), but it is a legitimate optimisation for all of them.

        Raises:
            RuntimeError: When the estimate fails. **Never return rows=0** — an
                unreachable source must not pass itself off as an empty table.
        """

    # -- SQL generation ----------------------------------------------------

    def quote_ident(self, name: str) -> str:
        """Quote an identifier. MSSQL overrides this (``[`` … ``]``)."""
        q = self.quote_char
        return f"{q}{name.strip()}{q}"

    def source_ident(self, name: str) -> str:
        """An identifier that **does not come from source introspection**.

        Most names in the generated SQL come from introspection, so they have
        exactly the shape the source uses. A few come from the configuration or
        from a default in the code (`parent_id`, `id`) and those are written in
        lower case. In a dialect that folds identifiers (Firebird keeps them
        upper case) they would not be found once quoted.

        By default this is the same as :meth:`quote_ident` — dialects that fold
        nothing are unaffected.
        """
        return self.quote_ident(name)

    def render_column_expr(self, column_name: str, surrogate: Optional[SurrogateKey]) -> str:
        """One expression for the SELECT list, honouring the surrogate key.

        The predecessors have 5 variants of this, but they all differ **only in
        the quoting character** — which is why this is a concrete method and not
        an abstract one.
        """
        if surrogate and surrogate.enabled and surrogate.alias == column_name and surrogate.expr:
            return f"{surrogate.expr} AS {self.quote_ident(column_name)}"
        return self.quote_ident(column_name)

    def render_nchar_convert(self, column_name: str) -> str:
        """One national-character column converted on the server, aliased to its own name.

        Only a dialect that declares `NCHAR_TYPES` has this, and today that is
        MSSQL alone. The alias is not decoration: the batch is keyed by column
        name all the way to ``df.rename(columns=ctx.name_map)``, so an expression
        without one would arrive nameless.

        Raises:
            UnsupportedFeature: When the dialect has no such conversion, which
                is every dialect whose `NCHAR_TYPES` is empty.
        """
        raise UnsupportedFeature(
            f"dialect '{self.name}' has no national-character types to convert"
        )

    def render_select_clause(
        self,
        column_names: Sequence[str],
        surrogate: Optional[SurrogateKey] = None,
        *,
        column_types: Optional[dict] = None,
        convert_nchar: bool = False,
    ) -> str:
        """The SELECT list. Identical across all 15 predecessor files bar the quoting.

        ``convert_nchar`` carries ``LOAD_SETTINGS.convert_nchar_to_varchar``, and
        ``column_types`` maps a **source** column name to its **source** type.
        Which columns get converted is decided from those introspected types and
        nowhere else — wrapping a legacy single-byte ``VARCHAR`` would corrupt
        exactly the columns the connection charset gets right today.

        A dialect with no `NCHAR_TYPES` renders the same clause it always did,
        whatever the flag says.
        """
        if not convert_nchar or not self.NCHAR_TYPES:
            return ", ".join(self.render_column_expr(c, surrogate) for c in column_names)

        types = column_types or {}
        return ", ".join(
            self.render_nchar_convert(c)
            if str(types.get(c, "")).lower() in self.NCHAR_TYPES
            else self.render_column_expr(c, surrogate)
            for c in column_names
        )

    def sql_literal(self, value: Any) -> str:
        """Render a value as a SQL literal. Text with doubled apostrophes.

        MySQL overrides this — it is the only one of the four that by default
        treats a backslash as an escape character, so it has to double that too.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    def render_fingerprint(
        self,
        ref: TableRef,
        *,
        pk: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        aggregate_columns: Sequence[str] = (),
        where: Optional[str] = None,
    ) -> str:
        """A single aggregate query that tells whether the source has changed.

        ``COUNT`` catches deletes, ``MAX(pk)`` catches inserts, and the third
        term (a modification time or a sum of columns) catches in-place edits.
        Four columns always come back in the same order, so the strategy does not
        have to work out which mode it is in.

        MSSQL overrides this because of ``COUNT_BIG`` — plain ``COUNT`` overflows
        there on tables above 2 billion rows.
        """
        tabulka = self.quote_ident(ref.name)
        if ref.schema:
            tabulka = f"{self.quote_ident(ref.schema)}.{tabulka}"

        casti = ["COUNT(*) AS fp_count"]
        casti.append(f"MAX({self.quote_ident(pk)}) AS fp_max_id" if pk else "NULL AS fp_max_id")
        casti.append(
            f"MAX({self.quote_ident(timestamp_column)}) AS fp_max_ts"
            if timestamp_column
            else "NULL AS fp_max_ts"
        )
        if aggregate_columns:
            soucet = " + ".join(
                f"COALESCE(SUM(CAST({self.quote_ident(c)} AS DECIMAL(38,6))), 0)"
                for c in aggregate_columns
            )
            casti.append(f"({soucet}) AS fp_agg")
        else:
            casti.append("NULL AS fp_agg")

        sql = f"SELECT {', '.join(casti)} FROM {tabulka}"
        return f"{sql} WHERE ({where})" if where else sql

    def cutoff_value(self, cutoff: Any) -> Any:
        """How a cutoff date is compared against a date column **in this source**.

        Most dialects hold a date as a date, so an ISO string is enough. Firebird
        holds it, in part of its tables, as a **day count since an epoch**, so it
        overrides this method and returns a number — and with that the last piece
        of Firebird knowledge disappears from `strategies.parent_incremental`.

        The predecessors do it the other way round: ``cutoff_numeric`` is
        computed by the shared ``_compute_incremental_cutoff``, so code that has
        nothing to do with Firebird still knows about the Firebird epoch.
        """
        return cutoff.strftime("%Y-%m-%d") if hasattr(cutoff, "strftime") else cutoff

    def render_key_expr(self, pk: str, surrogate: Optional[SurrogateKey] = None) -> str:
        """The expression that addresses the primary key in a ``WHERE`` clause.

        It differs from `render_column_expr` in having no ``AS`` — an alias is a
        syntax error in a condition, and with a surrogate key it is the only way
        to refer to the key at all (no column of that name exists in the source).
        """
        if surrogate and surrogate.enabled and surrogate.alias == pk and surrogate.expr:
            return f"({surrogate.expr})"
        return self.quote_ident(pk)

    def render_key_filter(
        self, pk: str, keys: Sequence[Any], surrogate: Optional[SurrogateKey] = None
    ) -> str:
        """``pk IN (…)`` for fetching specific rows by key.

        The keys go in as literals rather than as bound parameters: the query
        then travels through ``iter_batches(engine, sql, batch_size)``, which
        knows nothing about parameters — and should not, because paging in
        batches is dialect knowledge, not value transport.

        The key arrives from `target_pg.SourceHashSnapshot` as text (the column
        is ``TEXT`` there so that it can be compared across types). All three
        dialects that have this path convert a text literal against a numeric
        column by themselves — and the predecessors do the same, only through
        ``%s`` parameters filled from a file in ``/tmp``.
        """
        if not keys:
            raise ValueError("render_key_filter was given an empty list of keys.")
        values = ", ".join(self.sql_literal(k) for k in keys)
        return f"{self.render_key_expr(pk, surrogate)} IN ({values})"

    @abstractmethod
    def render_select(
        self,
        columns: Sequence[str],
        ref: TableRef,
        where: Optional[str] = None,
        order_by: Optional[Sequence[str]] = None,
        surrogate: Optional[SurrogateKey] = None,
        *,
        column_types: Optional[dict] = None,
        convert_nchar: bool = False,
    ) -> str:
        """The whole SELECT against the source.

        ``column_types`` and ``convert_nchar`` exist for
        ``LOAD_SETTINGS.convert_nchar_to_varchar`` and are handed straight to
        `render_select_clause`; three of the four dialects have nothing to do
        with them and accept them for the sake of the shared signature, the same
        way they accept ``column_types`` in `render_hash_expr`.
        """

    def render_hash_expr(
        self,
        hashed_columns: Sequence[str],
        alias: str,
        column_types: Optional[dict] = None,
    ) -> str:
        """A SQL expression that computes ``row_hash`` **on the source side**.

        **Deviation 2**: this was originally filed under ``core/hashing.py``. It
        does not belong there, it is pure source SQL:

        - MySQL: ``SHA2(CONCAT_WS('||', COALESCE(CAST(`c` AS CHAR), '')…), 256)``
        - MSSQL: ``HASHBYTES`` over ``ISNULL(CAST([c] AS NVARCHAR(MAX)), '')``,
          plus a separate branch for ``image``/``binary``/``varbinary`` — which is
          why it is the only one with ``col_types`` in its signature
        - Firebird: does not exist at all, hence ``supports('hash_diff') is False``

        What stays in ``core/hashing.py`` is the pandas ``_row_hash`` and the
        column selection (``compute_hash_columns``) — that part really is shared.

        Raises:
            UnsupportedFeature: When the dialect cannot hash on the source side.
        """
        raise UnsupportedFeature(f"{self.name} cannot compute row_hash on the source side")

    # -- reading -----------------------------------------------------------

    @abstractmethod
    def iter_batches(
        self,
        engine: Engine,
        sql: str,
        batch_size: int,
    ) -> Iterator[pd.DataFrame]:
        """Read the result in batches.

        **Deviation 4** — not in ARCHITECTURE.md. Without it a strategy would
        have to know whether reading goes through ``pd.read_sql(chunksize=)``, a
        server-side cursor or by hand, and that is exactly dialect knowledge.
        """

    # -- capabilities ------------------------------------------------------

    def supports(self, feature: str) -> bool:
        """``'keyset' | 'hash_diff' | 'partition_by_source' | 'parent_incremental'``"""
        return feature in self.FEATURES

    def require(self, feature: str) -> None:
        """Like ``supports``, but raises a readable error instead of returning ``False``.

        Called during configuration validation so that a mismatch shows up before
        the first read from the source rather than half-way through a run.
        """
        if not self.supports(feature):
            raise UnsupportedFeature(
                f"dialect '{self.name}' does not support '{feature}'; "
                f"it supports: {sorted(self.FEATURES) or '—'}"
            )


class DictTypeMapDialect(SourceDialect):
    """Intermediate class for dialects whose type mapping fits into a dict.

    That is MySQL, MSSQL and PostgreSQL. Firebird inherits straight from
    ``SourceDialect`` and writes its own ``map_type``.
    """

    #: source type (lowercase) -> PostgreSQL type
    type_map: ClassVar[dict[str, str]] = {}
    #: Used when the type is not in the map.
    fallback_pg_type: str = "TEXT"

    def map_type(self, column: ColumnDef) -> str:
        return self.type_map.get(column.source_type.lower(), self.fallback_pg_type)
