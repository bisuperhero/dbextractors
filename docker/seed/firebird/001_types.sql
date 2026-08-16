-- Firebird source fixtures.
--
-- Firebird is the oldest and least forgiving of the four sources, and until
-- this file it had no live test at all. Two of its habits drive the columns
-- below: it upper-cases unquoted identifiers, and its NUMERIC scale arrives
-- from the driver as a **negative** number, which is what the type mapping has
-- to read to tell 14,2 from 14,4.

CREATE TABLE TYPES_WIDE (
    ID              INTEGER NOT NULL PRIMARY KEY,

    -- Unquoted, so Firebird stores them upper-case. They are reserved words in
    -- the target and gain an underscore prefix there; see the column naming
    -- section of docs/legacy-compat.md.
    "TYPE"          VARCHAR(32),
    "NAME"          VARCHAR(64),
    "ORDER"         INTEGER,

    -- Scale, which the driver reports negated.
    MONEY_AMOUNT    NUMERIC(14,4),
    SMALL_SCALE     NUMERIC(14,2),
    NO_SCALE        NUMERIC(14,0),
    DECIMAL_VALUE   DECIMAL(18,6),

    -- Firebird counts days from 1858-11-17, not from the Unix epoch, and its
    -- DATE has no time part while TIMESTAMP does.
    DAY_DATE        DATE,
    STAMP           TIMESTAMP,
    TIME_OF_DAY     TIME,

    -- No native boolean in 2.5: it is spelled as a one-character flag or a
    -- SMALLINT, and both spellings are in the wild.
    FLAG_CHAR       CHAR(1),
    FLAG_SMALLINT   SMALLINT,

    -- CHAR pads with spaces, VARCHAR does not. The difference survives all the
    -- way into the target unless it is trimmed on the way.
    PADDED          CHAR(10),
    NOT_PADDED      VARCHAR(10),

    -- WIN1250 text, which is the charset these deployments actually use.
    CZECH_TEXT      VARCHAR(64),

    BLOB_TEXT       BLOB SUB_TYPE TEXT,
    BLOB_BINARY     BLOB SUB_TYPE BINARY,
    BIG_VALUE       BIGINT,
    FLOAT_VALUE     DOUBLE PRECISION
);

INSERT INTO TYPES_WIDE VALUES
    (1, 'alpha', 'first', 10,
     1234.5678, 12.34, 1234, 123456.789012,
     '2026-01-31', '2026-01-31 12:34:56.1230', '12:34:56.1230',
     'Y', 1,
     'padded', 'nopad',
     'prilis zlutoucky kun',
     'a text blob', NULL,
     9223372036854775807, 1.7976931348623157E308);

INSERT INTO TYPES_WIDE VALUES
    (2, NULL, NULL, NULL,
     NULL, NULL, NULL, NULL,
     NULL, NULL, NULL,
     NULL, NULL,
     NULL, NULL,
     NULL,
     NULL, NULL,
     NULL, NULL);

INSERT INTO TYPES_WIDE VALUES
    (3, 'gamma', 'third', -5,
     -0.0001, -0.01, -1, -0.000001,
     '1858-11-17', '1858-11-17 00:00:00.0000', '00:00:00.0000',
     'N', 0,
     '', '',
     '',
     '', NULL,
     -9223372036854775808, -1.7976931348623157E308);

CREATE TABLE PAGED (
    ID          INTEGER NOT NULL PRIMARY KEY,
    LABEL       VARCHAR(32),
    UPDATED_AT  TIMESTAMP,
    CREATED_AT  TIMESTAMP
);

INSERT INTO PAGED VALUES (1, 'one',   '2026-08-01', '2026-07-01');
INSERT INTO PAGED VALUES (2, 'two',   '2026-08-05', '2026-07-01');
INSERT INTO PAGED VALUES (3, 'three', '2026-08-10', '2026-07-15');
INSERT INTO PAGED VALUES (4, 'four',  '2026-08-15', '2026-08-01');
INSERT INTO PAGED VALUES (5, 'five',  NULL,         '2026-08-02');

COMMIT;
