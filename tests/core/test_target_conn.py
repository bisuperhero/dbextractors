"""Where the package writes.

This came out of a production failure::

    [one source] extraction failed: PostgreSQL at postgres:54333 is not
    answering.

`POSTGRES_PORT` there means the **port published to the host**
(``ports: - ${POSTGRES_PORT}:5432``), not the port to connect on. Inside the
docker network the database listens on 5432, which is exactly why
`io_config.yaml` has the port written out. The old extractors went through
`io_config.yaml` and hit it; this package assembled the target from `POSTGRES_*`
and missed.

So what the tests guard above all is the **order of the sources** — that is what
the failure came down to.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

from dbextractors.core import target_conn


@pytest.fixture()
def without_io_config(monkeypatch):
    """`io_config.yaml` is not available — a run outside Mage."""
    monkeypatch.setattr(target_conn, "_from_io_config", lambda _profile: None)


ENV = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "54333",  # the port on the host, not the one to connect on
    "POSTGRES_DB": "warehouse",
    "POSTGRES_USER": "mage",
    "POSTGRES_PASSWORD": "secret",
}


def test_io_config_beats_the_environment_variables(monkeypatch) -> None:
    """The heart of the fix: `io_config.yaml` takes precedence over `POSTGRES_*`.

    Without that order the package would still be pointing at port 54333.
    """
    monkeypatch.setattr(
        target_conn,
        "_from_io_config",
        lambda _p: "host=postgres port=5432 dbname=warehouse user=mage password=secret",
    )

    dsn = target_conn.resolve_target_dsn(ENV)

    # Unquoted on purpose: the stub above returns a literal and never reaches
    # `build_dsn`, so what this pins is the resolution order, not the spelling.
    # The tests that do exercise the builder assert the quoted form.
    assert "port=5432" in dsn
    assert "54333" not in dsn


def test_without_io_config_the_environment_variables_are_used(without_io_config) -> None:
    """Outside Mage there is no `io_config.yaml` and `POSTGRES_*` is the only
    source."""
    dsn = target_conn.resolve_target_dsn(ENV)

    assert "host='postgres'" in dsn and "port='54333'" in dsn


def test_an_explicit_dsn_beats_io_config_too(monkeypatch) -> None:
    monkeypatch.setattr(
        target_conn,
        "_from_io_config",
        lambda _p: "host=from-io-config port=5432 dbname=d user=u password=p",
    )
    env = dict(ENV, DBX_TARGET_DSN="host=elsewhere port=5432 dbname=d user=u password=p")

    assert "elsewhere" in target_conn.resolve_target_dsn(env)


def test_the_golden_dsn_is_only_the_last_fallback(without_io_config) -> None:
    """Deployments set `DBX_GOLDEN_DSN` because of this failure — it has to keep
    working.

    But it is a **fallback**, not the main route: as soon as `POSTGRES_*` is
    complete, those win, because they are the variables meant for the job.
    """
    env = dict(ENV, DBX_GOLDEN_DSN="host=from-golden port=5432 dbname=d user=u password=p")
    assert "from-golden" not in target_conn.resolve_target_dsn(env)

    golden_only = {"DBX_GOLDEN_DSN": "host=from-golden port=5432 dbname=d user=u password=p"}
    assert "from-golden" in target_conn.resolve_target_dsn(golden_only)


def test_the_profile_can_be_changed(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(target_conn, "_from_io_config", lambda p: seen.append(p) or None)

    target_conn.resolve_target_dsn({"DBX_TARGET_PROFILE": "other", **ENV})
    target_conn.resolve_target_dsn(ENV)

    assert seen == ["other", target_conn.DEFAULT_PROFILE]
    assert target_conn.DEFAULT_PROFILE == "warehouse", "the profile of the extractors replaced"


def test_with_nothing_at_all_it_fails_and_says_what_it_tried(without_io_config) -> None:
    """A loud failure beats writing who-knows-where."""
    with pytest.raises(target_conn.TargetConnectionError) as err:
        target_conn.resolve_target_dsn({})

    text = str(err.value)
    assert "DBX_TARGET_DSN" in text and "io_config.yaml" in text and "POSTGRES_HOST" in text


def test_the_password_never_reaches_the_log() -> None:
    """What matters is the **value**, not the key.

    The `password=…` token used to be dropped from the description whole. Since
    `core.secrets` came in it is masked instead (`password=***`) — so the log
    still shows that the DSN carried a password, which helps when debugging "why
    did it not connect". The secret is gone either way.
    """
    dsn = "host=postgres port=5432 dbname=w user=mage password=secret"

    description = target_conn.describe_dsn(dsn)

    assert "secret" not in description
    assert "password=***" in description
    assert "host=postgres" in description and "port=5432" in description


# --- The profile from the configuration -------------------------------------
#
# `TARGET_DB` in the configuration blocks declared a host, a user and a
# password, and **nothing read it** — neither this package nor any of the four
# old extractors (those have `config_profile = 'warehouse'` hard-coded). So 342
# blocks carried configuration that looked authoritative and was not.
# `TARGET_PROFILE` replaces it: it states only what is really used, and repeats
# no password anywhere.


def test_a_profile_in_the_configuration_beats_the_environment_variable(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(target_conn, "_from_io_config", lambda p: seen.append(p) or None)

    target_conn.resolve_target_dsn(
        dict(ENV, DBX_TARGET_PROFILE="from-environment"), profile="from-configuration"
    )

    assert seen == ["from-configuration"], "the more specific source (one pipeline) wins"


def test_without_a_profile_in_the_configuration_the_variable_applies(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(target_conn, "_from_io_config", lambda p: seen.append(p) or None)

    target_conn.resolve_target_dsn(dict(ENV, DBX_TARGET_PROFILE="from-environment"), profile=None)

    assert seen == ["from-environment"]


def test_target_profile_travels_through_the_configuration() -> None:
    """The top-level key has to reach `ParsedConfig`."""
    from dbextractors.core.config import parse

    base = {
        "TABLE": {"source_name": "t", "output_schema": "s", "output_table": "t"},
        "LOAD_SETTINGS": {"load_method": "full"},
        "SOURCE_DB": {"host": "h", "database": "d", "user": "u", "password": "p"},
    }
    assert parse(base).target_profile is None, "without the key the default decides"
    assert parse({**base, "TARGET_PROFILE": "store"}).target_profile == "store"


# --- Reading io_config.yaml --------------------------------------------------
#
# Every test above replaces `_from_io_config` with a lambda, which is right for
# testing the **order** of the sources and leaves the function itself — the
# primary production route to the target — with no coverage at all. Outside Mage
# it returns `None` on the first line and never reaches the parsing, so nothing
# a developer or CI runs ever executes the body.
#
# The stand-in below supplies the three names the function imports. It proves
# what our side of the contract does with the answers: which profile it asks
# for, which keys it reads, what it does with a partial answer and with a loader
# that throws. What it cannot prove is that the real library behaves this way —
# that is the `needs_mage` test at the end of the file, which runs in the CI job
# on the production image.


class _FakeConfigKey:
    """The four keys `_from_io_config` reads. The real ones are an enum whose
    values are these same strings."""

    POSTGRES_HOST = "POSTGRES_HOST"
    POSTGRES_PORT = "POSTGRES_PORT"
    POSTGRES_DBNAME = "POSTGRES_DBNAME"
    POSTGRES_USER = "POSTGRES_USER"
    POSTGRES_PASSWORD = "POSTGRES_PASSWORD"


WAREHOUSE = {
    "POSTGRES_HOST": "postgres",
    # The port is written out in `io_config.yaml` and taken from a variable in
    # the compose file — that difference is the failure this module came out of.
    "POSTGRES_PORT": 5432,
    "POSTGRES_DBNAME": "warehouse",
    "POSTGRES_USER": "mage",
    "POSTGRES_PASSWORD": "secret",
}


@pytest.fixture()
def mage_repo(tmp_path):
    """A directory that looks like a Mage project: it holds an `io_config.yaml`.

    The file's contents do not matter to the stand-in loader, but its existence
    does — `_from_io_config` gives up before parsing when the file is not there,
    and that branch is what a run outside Mage takes.
    """
    (tmp_path / "io_config.yaml").write_text("version: 0.1.1\n", encoding="utf-8")
    return tmp_path


def _install_fake_mage(monkeypatch, repo_path, profiles, *, explodes: bool = False) -> list:
    """Put a stand-in for `mage_ai` into ``sys.modules`` and return the calls made.

    Injected as modules rather than by patching `_from_io_config`, because the
    ``from mage_ai… import`` inside the function is itself part of the behaviour:
    it is what makes the whole route disappear outside Mage.
    """
    calls: list = []

    class _FakeLoader:
        def __init__(self, path, profile):
            calls.append((path, profile))
            if explodes:
                raise ValueError(f"profile {profile!r} is malformed")
            self._values = profiles.get(profile, {})

        def get(self, key):
            return self._values.get(key)

    config_module = types.ModuleType("mage_ai.io.config")
    config_module.ConfigFileLoader = _FakeLoader
    config_module.ConfigKey = _FakeConfigKey

    repo_module = types.ModuleType("mage_ai.settings.repo")
    repo_module.get_repo_path = lambda: str(repo_path)

    for name, module in (
        ("mage_ai", types.ModuleType("mage_ai")),
        ("mage_ai.io", types.ModuleType("mage_ai.io")),
        ("mage_ai.io.config", config_module),
        ("mage_ai.settings", types.ModuleType("mage_ai.settings")),
        ("mage_ai.settings.repo", repo_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return calls


def test_the_profile_becomes_a_dsn(monkeypatch, mage_repo) -> None:
    """The whole point of the module: the target is assembled out of the same
    profile the replaced extractors used, port included."""
    _install_fake_mage(monkeypatch, mage_repo, {"warehouse": WAREHOUSE})

    assert (
        target_conn._from_io_config("warehouse")
        == "host='postgres' port='5432' dbname='warehouse' user='mage' password='secret'"
    )


def test_the_profile_asked_for_is_the_one_that_is_read(monkeypatch, mage_repo) -> None:
    """`TARGET_PROFILE` is worth nothing if the loader is always handed
    ``warehouse`` anyway."""
    calls = _install_fake_mage(monkeypatch, mage_repo, {"store": WAREHOUSE})

    target_conn._from_io_config("store")

    assert calls == [(os.path.join(str(mage_repo), "io_config.yaml"), "store")]


def test_io_config_wins_over_the_environment_for_real(monkeypatch, mage_repo) -> None:
    """The same ordering the tests above assert, but through the real function.

    Those replace `_from_io_config` with a lambda, so they would still pass if the
    function could never return a DSN at all — which is exactly the state it was
    in: untested, and reachable only inside Mage.
    """
    _install_fake_mage(monkeypatch, mage_repo, {"warehouse": WAREHOUSE})

    dsn = target_conn.resolve_target_dsn(ENV)

    assert "port='5432'" in dsn and "54333" not in dsn


@pytest.mark.parametrize("missing", ["POSTGRES_HOST", "POSTGRES_DBNAME", "POSTGRES_USER"])
def test_an_incomplete_profile_is_passed_over(monkeypatch, mage_repo, missing: str) -> None:
    """A DSN without a host, a database or a user connects nowhere.

    Returning half of one would send the run to whatever libpq defaults to —
    typically a local socket — which is a far worse failure than falling through
    to `POSTGRES_*`.
    """
    profile = {k: v for k, v in WAREHOUSE.items() if k != missing}
    _install_fake_mage(monkeypatch, mage_repo, {"warehouse": profile})

    assert target_conn._from_io_config("warehouse") is None


def test_a_profile_without_a_port_falls_back_to_5432(monkeypatch, mage_repo) -> None:
    profile = {k: v for k, v in WAREHOUSE.items() if k != "POSTGRES_PORT"}
    _install_fake_mage(monkeypatch, mage_repo, {"warehouse": profile})

    assert "port='5432'" in target_conn._from_io_config("warehouse")


def test_a_profile_without_a_password_is_still_usable(monkeypatch, mage_repo) -> None:
    """Trust authentication and `.pgpass` are both legitimate; an absent password
    is not an absent profile."""
    profile = {k: v for k, v in WAREHOUSE.items() if k != "POSTGRES_PASSWORD"}
    _install_fake_mage(monkeypatch, mage_repo, {"warehouse": profile})

    assert target_conn._from_io_config("warehouse") == (
        "host='postgres' port='5432' dbname='warehouse' user='mage' password=''"
    )


def test_a_broken_profile_is_logged_rather_than_swallowed(monkeypatch, mage_repo, caplog) -> None:
    """Moving on in silence is what produced the failure this module is named for.

    So the fallthrough stays — a broken profile must not bring the run down —
    but it has to leave a line in the log naming the profile.
    """
    _install_fake_mage(monkeypatch, mage_repo, {"warehouse": WAREHOUSE}, explodes=True)

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.target_conn"):
        assert target_conn._from_io_config("warehouse") is None

    assert any("warehouse" in record.getMessage() for record in caplog.records), caplog.text


def test_without_an_io_config_file_there_is_no_dsn(monkeypatch, tmp_path) -> None:
    """A run under Mage whose project directory holds no `io_config.yaml`."""
    _install_fake_mage(monkeypatch, tmp_path, {"warehouse": WAREHOUSE})

    assert target_conn._from_io_config("warehouse") is None


def test_without_the_library_there_is_no_dsn(monkeypatch) -> None:
    """Outside Mage the import fails and the caller moves on to `POSTGRES_*`.

    ``None`` in ``sys.modules`` makes the import fail even where the library is
    installed, so this branch is checked in the CI job on the production image
    too and not only where it is trivially true.
    """
    monkeypatch.setitem(sys.modules, "mage_ai.io.config", None)

    assert target_conn._from_io_config("warehouse") is None


# --- Against the live library ------------------------------------------------
#
# Everything above tests our side of the contract against a stand-in. This one
# tests the contract itself: that `ConfigFileLoader` accepts a path and a
# profile, that `ConfigKey.POSTGRES_*` name the keys `io_config.yaml` actually
# uses, and that `get` returns the values rather than, say, a wrapper. It runs
# only where `mage_ai` is installed — locally that is nowhere, so the
# `mage-parity` job on the 0.9.79 image is what it is written for.


@pytest.mark.needs_mage
def test_the_live_loader_reads_the_warehouse_profile(monkeypatch, tmp_path) -> None:
    pytest.importorskip(
        "mage_ai.io.config",
        reason="mage_ai is not installed; the mage-parity job on the production image covers this",
    )
    from mage_ai.settings import repo as mage_repo_settings

    (tmp_path / "io_config.yaml").write_text(
        "version: 0.1.1\n"
        "warehouse:\n"
        "  POSTGRES_HOST: postgres\n"
        "  POSTGRES_PORT: 5432\n"
        "  POSTGRES_DBNAME: warehouse\n"
        "  POSTGRES_USER: mage\n"
        "  POSTGRES_PASSWORD: secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mage_repo_settings, "get_repo_path", lambda *a, **k: str(tmp_path))

    assert (
        target_conn._from_io_config("warehouse")
        == "host='postgres' port='5432' dbname='warehouse' user='mage' password='secret'"
    )


@pytest.mark.needs_mage
def test_the_live_loader_passes_over_a_profile_that_is_not_there(monkeypatch, tmp_path) -> None:
    """A profile named in `TARGET_PROFILE` and missing from the file must not
    take the run down — it must fall through to the next source."""
    pytest.importorskip(
        "mage_ai.io.config",
        reason="mage_ai is not installed; the mage-parity job on the production image covers this",
    )
    from mage_ai.settings import repo as mage_repo_settings

    (tmp_path / "io_config.yaml").write_text("version: 0.1.1\ndefault:\n  X: 1\n", encoding="utf-8")
    monkeypatch.setattr(mage_repo_settings, "get_repo_path", lambda *a, **k: str(tmp_path))

    assert target_conn._from_io_config("nonexistent") is None


# ---------------------------------------------------------------------------
# `build_dsn` — quoting, and the bug that made it necessary
# ---------------------------------------------------------------------------
#
# Before this, the builders interpolated every value bare. A password with a
# space then produced a DSN libpq cannot parse: it reads `password=my secret`
# as a password of `my` followed by a keyword `secret`, and reports
# `invalid dsn: missing "=" after "secret"` — quoting the offending fragment,
# which is how part of a password reached a log. The redaction fixed the
# symptom; these tests pin the cause.


@pytest.mark.parametrize(
    ("password", "expected"),
    [
        ("plain", "password='plain'"),
        ("", "password=''"),
        ("with space", "password='with space'"),
        ("it's", r"password='it\'s'"),
        ("back\\slash", r"password='back\\slash'"),
        ("both ' and \\", r"password='both \' and \\'"),
    ],
)
def test_a_password_is_quoted_and_escaped(password, expected) -> None:
    assert expected in target_conn.build_dsn("h", 5432, "d", "u", password)


def test_every_value_is_quoted_not_only_the_password() -> None:
    """A database name or user with a space breaks the DSN the same way."""
    dsn = target_conn.build_dsn("h", 5432, "my db", "some user", "p")
    assert "dbname='my db'" in dsn
    assert "user='some user'" in dsn


@pytest.mark.needs_pg
def test_a_password_with_a_space_can_actually_connect() -> None:
    """The point of the quoting, proven against a live server rather than argued.

    A role is created with a password containing a space, a quote and a
    backslash — all three of the characters that broke the old builder — and the
    DSN built for it has to connect. Before the fix psycopg2 rejected the string
    without ever reaching the server.
    """
    import psycopg2

    from dbextractors.golden import session

    # This file has no `dsn` fixture — that one lives beside the tests whose
    # whole subject is the database. Reading the switch directly keeps the test
    # here, next to the builder it is about.
    admin_dsn = os.environ.get("DBX_GOLDEN_TEST_DSN")
    if not admin_dsn:
        pytest.skip("no target PostgreSQL; set DBX_GOLDEN_TEST_DSN (see .env.example)")

    hostile = "sp ace'quote\\slash"
    role = "dbx_golden_quoting_probe"

    admin = psycopg2.connect(admin_dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f'DROP ROLE IF EXISTS "{role}"')
            cur.execute(f'CREATE ROLE "{role}" LOGIN PASSWORD %s', (hostile,))
        built = target_conn.build_dsn(
            session._dsn_field(admin_dsn, "host"),
            session._dsn_field(admin_dsn, "port"),
            session._dsn_field(admin_dsn, "dbname"),
            role,
            hostile,
        )
        probe = psycopg2.connect(built)
        probe.close()
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP ROLE IF EXISTS "{role}"')
        admin.close()
