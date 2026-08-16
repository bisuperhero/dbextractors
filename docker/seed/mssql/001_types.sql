-- MSSQL source fixtures.
--
-- The server runs a CP1250 collation on purpose: a type that maps correctly
-- under UTF-8 and mangles Czech text under a legacy collation is exactly the
-- bug that string tests cannot see.

IF DB_ID('dbx_src') IS NULL
    CREATE DATABASE dbx_src;
GO

USE dbx_src;
GO

IF OBJECT_ID('dbo.types_wide', 'U') IS NOT NULL DROP TABLE dbo.types_wide;
GO

CREATE TABLE dbo.types_wide (
    id                INT PRIMARY KEY,

    -- Reserved words in the target; they gain an underscore prefix on the way
    -- in. See the column naming section of docs/legacy-compat.md.
    [type]            NVARCHAR(32),
    [name]            NVARCHAR(64),
    [order]           INT,

    -- MONEY and SMALLMONEY are fixed point but not DECIMAL, and drivers are
    -- inconsistent about how they hand them over.
    money_amount      MONEY,
    small_money       SMALLMONEY,

    -- NUMERIC with a large scale must not go through float.
    precise_value     NUMERIC(28,10),

    -- The three date/time families behave differently: DATETIME rounds to
    -- 3.33 ms, DATETIME2 does not, and DATETIMEOFFSET carries a zone.
    legacy_datetime   DATETIME,
    modern_datetime   DATETIME2(7),
    zoned_datetime    DATETIMEOFFSET(7),
    day_date          DATE,
    time_of_day       TIME(7),

    -- BIT is the boolean.
    flag              BIT,

    -- CP1250 text next to Unicode text: the same characters, two encodings.
    cp1250_text       VARCHAR(255),
    unicode_text      NVARCHAR(255),

    binary_blob       VARBINARY(MAX),
    key_uuid          UNIQUEIDENTIFIER,
    xml_payload       XML,

    -- A computed column: it must be read like any other, not skipped.
    doubled           AS ([order] * 2)
);
GO

INSERT INTO dbo.types_wide
    (id, [type], [name], [order], money_amount, small_money, precise_value,
     legacy_datetime, modern_datetime, zoned_datetime, day_date, time_of_day,
     flag, cp1250_text, unicode_text, binary_blob, key_uuid, xml_payload)
VALUES
    (1, N'alpha', N'first', 10, 1234.5678, 12.34, 1234567890.0123456789,
     '2026-01-31T12:34:56.997', '2026-01-31T12:34:56.1234567',
     '2026-01-31T12:34:56.1234567+02:00', '2026-01-31', '12:34:56.1234567',
     1, 'prilis zlutoucky kun', N'příliš žluťoučký kůň',
     0xDEADBEEF, '6F9619FF-8B86-D011-B42D-00C04FC964FF', N'<r><a>1</a></r>'),

    (2, NULL, NULL, NULL, NULL, NULL, NULL,
     NULL, NULL, NULL, NULL, NULL,
     NULL, NULL, NULL, NULL, NULL, NULL),

    -- The extremes: MSSQL's DATETIME starts in 1753, DATETIME2 in year 1.
    -- The offset on the maximum DATETIMEOFFSET has to be **positive**: the
    -- server validates the UTC equivalent, and a negative offset there would
    -- push it past year 9999.
    (3, N'gamma', N'third', -5, -0.0001, -0.01, -0.0000000001,
     '1753-01-01T00:00:00', '0001-01-01T00:00:00.0000000',
     '9999-12-31T23:59:59.9999999+14:00', '9999-12-31', '23:59:59.9999999',
     0, '', N'', 0x, '00000000-0000-0000-0000-000000000000', NULL);
GO

IF OBJECT_ID('dbo.paged', 'U') IS NOT NULL DROP TABLE dbo.paged;
GO

CREATE TABLE dbo.paged (
    id          INT PRIMARY KEY,
    label       NVARCHAR(32),
    updated_at  DATETIME2(7),
    created_at  DATETIME2(7)
);
GO

INSERT INTO dbo.paged (id, label, updated_at, created_at) VALUES
    (1, N'one',   '2026-08-01', '2026-07-01'),
    (2, N'two',   '2026-08-05', '2026-07-01'),
    (3, N'three', '2026-08-10', '2026-07-15'),
    (4, N'four',  '2026-08-15', '2026-08-01'),
    (5, N'five',  NULL,         '2026-08-02');
GO

-- The client-charset boundary.
--
-- `types_wide.unicode_text` shows that an NVARCHAR value can be stored
-- perfectly and still not arrive: the connection is opened with
-- `charset='cp1250'`, inherited from both predecessors, and that mangles the
-- N-types. One column cannot show *where* the boundary runs, and without that
-- the limitation reads as "Unicode is destroyed" instead of the much narrower
-- thing it is. See the NVARCHAR section of docs/legacy-compat.md.
--
-- Every non-ASCII value goes in as an explicit byte string rather than as a
-- readable literal. A literal would depend on how the loader (sqlcmd) decoded
-- this file, and a seed whose contents are a function of the loader is exactly
-- the accident this table exists to rule out — the bytes below *are* the
-- fixture, and the tests assert them before asserting anything about reading.
IF OBJECT_ID('dbo.unicode_edge', 'U') IS NOT NULL DROP TABLE dbo.unicode_edge;
GO

CREATE TABLE dbo.unicode_edge (
    id             INT PRIMARY KEY,
    label          VARCHAR(32) NOT NULL,

    -- The same text in all three N-types: the boundary is a property of the
    -- connection, not of the column type, and that has to be visible.
    nvarchar_value NVARCHAR(64),
    nchar_value    NCHAR(20),
    ntext_value    NTEXT,

    -- The legacy side, and the whole reason cp1250 is on the connection: a
    -- VARCHAR under the server's CP1250 collation holds single-byte text.
    varchar_value  VARCHAR(64)
);
GO

INSERT INTO dbo.unicode_edge (id, label, nvarchar_value, nchar_value, ntext_value, varchar_value)
VALUES
    -- Pure ASCII. Identical in every code page involved, so it survives
    -- whatever the connection is set to — the control case.
    (1, 'ascii', N'plain ascii', N'plain ascii', N'plain ascii', 'plain ascii'),

    -- 'příliš žluťoučký kůň'. UTF-16LE on the N-side, CP1250 bytes on the
    -- VARCHAR side — the same twenty characters written the two ways a
    -- CP1250-collated database really writes them.
    (2, 'czech',
     CONVERT(NVARCHAR(64), 0x70005901ED006C006900610120007E016C00750065016F0075000D016B00FD0020006B006F014801),
     CONVERT(NCHAR(20),    0x70005901ED006C006900610120007E016C00750065016F0075000D016B00FD0020006B006F014801),
     CONVERT(NTEXT, CONVERT(NVARCHAR(64), 0x70005901ED006C006900610120007E016C00750065016F0075000D016B00FD0020006B006F014801)),
     CONVERT(VARCHAR(64),  0x70F8ED6C699A209E6C759D6F75E86BFD206BF9F2)),

    -- 'abcdef-ř-ghijkl'. One character out of place in an otherwise ASCII
    -- string: it shows that the loss is a truncation at that character, not a
    -- substitution of it and not a loss of the whole value.
    (3, 'czech_mid',
     CONVERT(NVARCHAR(64), 0x6100620063006400650066002D0059012D006700680069006A006B006C00),
     CONVERT(NCHAR(20),    0x6100620063006400650066002D0059012D006700680069006A006B006C00),
     CONVERT(NTEXT, CONVERT(NVARCHAR(64), 0x6100620063006400650066002D0059012D006700680069006A006B006C00)),
     NULL),

    -- 'a-ø-b'. U+00F8 is in latin-1 but not in CP1250, and it is the case that
    -- proves the conversion is latin-1: it arrives whole and as the *wrong*
    -- character rather than being cut off.
    (4, 'latin1_only',
     CONVERT(NVARCHAR(64), 0x61002D00F8002D006200),
     CONVERT(NCHAR(20),    0x61002D00F8002D006200),
     CONVERT(NTEXT, CONVERT(NVARCHAR(64), 0x61002D00F8002D006200)),
     NULL),

    -- 'a-Ж-b'. U+0416 is in neither latin-1 nor CP1250, and unlike 'ø' the
    -- CP1250 collation has no base letter to fold it onto. It is the case that
    -- shows what `convert_nchar_to_varchar` costs: the server writes '?' where
    -- it cannot express the character, which is lossy but keeps the rest of the
    -- string — as against the truncation that happens without the key.
    (5, 'outside_cp1250',
     CONVERT(NVARCHAR(64), 0x61002D0016042D006200),
     CONVERT(NCHAR(20),    0x61002D0016042D006200),
     CONVERT(NTEXT, CONVERT(NVARCHAR(64), 0x61002D0016042D006200)),
     NULL);
GO
