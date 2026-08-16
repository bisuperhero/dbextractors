# Enabling partitioning on an existing table

`partition_by` is applied only to a target table that **does not exist yet**. On a
table that is already there it is logged and skipped, and the run finishes without
partitioning. Converting an existing table is therefore a manual migration, and this
page is the procedure for it.

This is a DBA job you do once per table, against the target PostgreSQL. What
`partition_by` does and which modes exist is in the
[README](../README.md#target-partitioning); you do not need this page for a table
that is being created for the first time.

---

Adding `partition_by` to the configuration is **not enough** on its own — for an
existing table it merely logs that it was not applied. The reason lies in
PostgreSQL: `relkind` cannot be changed, so a plain table is not converted into a
partitioned one by `ALTER TABLE` or by `RENAME`. A **new relation with a new OID**
is always created, which is why dependent views have to be dropped and recreated —
a view holds an OID, not a name.

First look at what depends on the table and how big it is:

```sql
SELECT v.relname
FROM pg_depend d JOIN pg_rewrite r ON r.oid = d.objid
JOIN pg_class v ON v.oid = r.ev_class
WHERE d.refobjid = 'raw_source.orders'::regclass
  AND v.relkind IN ('v','m') AND d.deptype = 'n'
GROUP BY v.relname;
```

## Procedure A — a small table with no dependent views

Drop it and let dbextractors create it again:

```sql
DROP TABLE raw_source.orders;
```

Add `partition_by` to the configuration and run the pipeline. The incremental
strategies (`incremental`, `hash_diff`, `id_watermark`) **detect the missing target
themselves** and fall back to a full load (`fallback_reason: 'target table does not
exist'`), so nothing has to be switched over by hand. dbextractors creates the
indexes itself at the end of the run.

The price is re-reading the whole source. For a 15M-row table that is hours — which
is what procedure B is for.

## Procedure B — a large table, or a table with views

Rebuild it from the data already in the target, without touching the source. All of
it belongs in **one transaction**:

```sql
BEGIN;

-- 1. partitioned parent shaped like the old table
CREATE TABLE raw_source.orders_part (LIKE raw_source.orders INCLUDING DEFAULTS)
    PARTITION BY RANGE (day);

-- 2. partitions for the periods actually present in the data, plus the catch-all
CREATE TABLE raw_source.orders_part__default
    PARTITION OF raw_source.orders_part DEFAULT;
CREATE TABLE raw_source.orders_part__2026_08
    PARTITION OF raw_source.orders_part FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
-- … and so on; the list of periods comes from:
--   SELECT DISTINCT date_trunc('month', day::timestamp)::date
--   FROM raw_source.orders WHERE day IS NOT NULL ORDER BY 1;

-- 3. move the data over
INSERT INTO raw_source.orders_part SELECT * FROM raw_source.orders;

-- 4. views down, swap, views back (save the definitions first
--    with pg_get_viewdef(oid, true))
DROP VIEW raw_source.v_orders;
DROP TABLE raw_source.orders;
ALTER TABLE raw_source.orders_part RENAME TO orders;
CREATE VIEW raw_source.v_orders AS …;

COMMIT;
```

Then **create the indexes**, because `LIKE … INCLUDING DEFAULTS` does not copy them
and without them an incremental run fails straight away at the check
(`… is missing a UNIQUE/PRIMARY KEY`):

```sql
CREATE UNIQUE INDEX idx_orders_id_day_unique ON raw_source.orders (id, day);
CREATE INDEX idx_orders_id ON raw_source.orders (id);
```

The unique one **must contain the partition key** — PostgreSQL will not allow it
otherwise. The second, plain one is needed for deletion by key; without it that is a
sequential scan across every partition. `CONCURRENTLY` does not work over a
partitioned table.

Finally add `partition_by` to the configuration so the layout is kept from then on.
If the key in the configuration did not match the one the table is really
partitioned by, the run **fails** — quietly writing into a differently partitioned
table would be worse.

Verified on a daily traffic table: 716,815 rows, 29 columns and 43 monthly
partitions rebuilt in 3.6 s, with a matching checksum and a working view.
