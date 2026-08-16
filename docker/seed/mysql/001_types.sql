-- MySQL source fixtures.
--
-- Every column here exists because of a type that behaves neither like a number
-- nor like text once it has been through the conversion layer and `COPY`.

-- Neither the official image's docker-entrypoint-initdb.d runner nor CI's
-- plain `mysql ... < file` invocation passes `--default-character-set`, so the
-- client negotiates `latin1` (verified: `SHOW VARIABLES LIKE 'character_set%'`
-- reports `character_set_client = latin1` even though the server itself
-- defaults to utf8mb4). The client does not re-encode the UTF-8 bytes it reads
-- from this file — it forwards them as-is — so without this, the server
-- decodes them as latin1 and re-encodes as utf8mb4 on storage, i.e. exactly
-- one round of double-encoding. `SET NAMES` has to be the first statement,
-- before any literal with non-ASCII text is sent.
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS types_wide (
    id                INT PRIMARY KEY,

    -- Reserved words in the target. These gain an underscore prefix on the way
    -- in (`_type`, `_name`, `_order`); see the column naming section of
    -- docs/legacy-compat.md.
    `type`            VARCHAR(32),
    `name`            VARCHAR(64),
    `order`           INT,

    -- MySQL's zero date is not a date. It has to become NULL rather than
    -- reaching PostgreSQL, which rejects it outright.
    zero_date         DATE NULL,
    zero_datetime     DATETIME NULL,

    -- TIME in MySQL is a signed interval, not a wall clock: it legitimately
    -- holds values past 24 hours and negative ones.
    long_time         TIME,
    negative_time     TIME,

    -- Unsigned BIGINT exceeds a signed 64-bit integer, which is what the target
    -- column would otherwise be.
    big_unsigned      BIGINT UNSIGNED,

    -- TINYINT(1) is how MySQL spells a boolean.
    flag              TINYINT(1),

    -- Fixed point must not go through float on the way.
    money_amount      DECIMAL(14,4),

    -- Binary payloads, and text that needs four-byte characters.
    binary_blob       BLOB,
    utf8mb4_text      VARCHAR(255),

    payload           JSON,
    enum_value        ENUM('a','b','c'),
    set_value         SET('x','y'),
    bit_value         BIT(8),
    year_value        YEAR
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Zero dates are rejected by default in MySQL 8; the seed needs them, because
-- the point is precisely that the source may contain them.
SET SESSION sql_mode = '';

INSERT INTO types_wide VALUES
    (1, 'alpha', 'first',  10, '2026-01-31', '2026-01-31 12:34:56',
        '25:00:00', '-01:30:00', 18446744073709551615, 1, 1234.5678,
        UNHEX('DEADBEEF'), 'příliš žluťoučký kůň 😀', '{"a": 1}', 'a', 'x,y',
        b'10101010', 2026),

    -- The zero date and the empty-ish row: NULLs everywhere they are allowed.
    (2, NULL, NULL, NULL, '0000-00-00', '0000-00-00 00:00:00',
        '00:00:00', '00:00:00', 0, 0, 0.0000,
        NULL, '', NULL, NULL, '', b'00000000', 0),

    (3, 'gamma', 'third', -5, NULL, NULL,
        '838:59:59', '-838:59:59', 9223372036854775808, NULL, -0.0001,
        UNHEX('00'), 'ascii only', '[]', 'c', 'x',
        b'11111111', 1901);

-- A second table, deliberately narrow, for pagination and watermark tests.
CREATE TABLE IF NOT EXISTS paged (
    id            INT PRIMARY KEY,
    label         VARCHAR(32),
    updated_at    DATETIME,
    created_at    DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO paged (id, label, updated_at, created_at) VALUES
    (1, 'one',   '2026-08-01 00:00:00', '2026-07-01 00:00:00'),
    (2, 'two',   '2026-08-05 00:00:00', '2026-07-01 00:00:00'),
    (3, 'three', '2026-08-10 00:00:00', '2026-07-15 00:00:00'),
    (4, 'four',  '2026-08-15 00:00:00', '2026-08-01 00:00:00'),
    (5, 'five',  NULL,                  '2026-08-02 00:00:00');
