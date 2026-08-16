-- Target fixtures.
--
-- The suite creates everything it needs in scratch schemas prefixed with
-- `dbx_golden_`, so almost nothing has to exist up front. What does:
--
--   1. the extensions the tests rely on
--   2. a schema the scratch machinery is allowed to write into

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- `dbx_golden_` is the only prefix the safety guard in
-- `dbextractors.golden.scratch` will write to, and that guard cannot be turned
-- off. The suite creates and drops these schemas itself; this one exists so
-- that a first run has somewhere to land even before it does.
CREATE SCHEMA IF NOT EXISTS dbx_golden_seed;

-- A wide table for the conversion-layer benchmark. Throughput is quoted per
-- row *times column*, not per row, so width is what the measurement is about.
CREATE TABLE IF NOT EXISTS dbx_golden_seed.wide_48 (
    id integer PRIMARY KEY,
    c01 text, c02 text, c03 text, c04 text, c05 text, c06 text,
    c07 numeric(14,4), c08 numeric(14,4), c09 numeric(14,4), c10 numeric(14,4),
    c11 timestamp, c12 timestamp, c13 timestamp, c14 timestamp,
    c15 date, c16 date, c17 date, c18 date,
    c19 boolean, c20 boolean, c21 boolean, c22 boolean,
    c23 bigint, c24 bigint, c25 bigint, c26 bigint,
    c27 text, c28 text, c29 text, c30 text, c31 text, c32 text,
    c33 numeric(18,6), c34 numeric(18,6), c35 numeric(18,6), c36 numeric(18,6),
    c37 jsonb, c38 jsonb,
    c39 bytea, c40 bytea,
    c41 text, c42 text, c43 text, c44 text, c45 text, c46 text, c47 text
);
