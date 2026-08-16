"""Tests for `dbextractors.core.tunnel`.

Cover: characterisation of `find_free_port`/`ensure_private_key_permissions`/
`normalize_private_key_contents` against the oracle, `ssh_tunnel_preexec`
without a real fork (a fake `ctypes`/`os.setsid`), `resolve_connection_mode`,
and `open_tunnel` as a context manager (direct/ssh/auto, cleanup on an
exception included).

Nothing here starts a real `ssh` or opens a connection off localhost:
`subprocess.Popen` is always faked, and `_default_probe`/`wait_for_port` are
tested either against localhost (a closed port -> a local RST, no traffic
leaves the machine) or through a monkeypatch on `socket.create_connection`.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

import reference_oracle as ro
from dbextractors.core import tunnel

# --- find_free_port -----------------------------------------------------------
# Side-effecting (it binds a socket) -> characterised through its properties
# rather than through an exact value.


def test_find_free_port_returns_a_usable_port() -> None:
    port = tunnel.find_free_port()
    assert isinstance(port, int)
    assert 1 <= port <= 65535
    # It must bind straight away -> it really is free, not just "some number".
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def test_find_free_port_the_old_version_has_the_same_property() -> None:
    old = ro.get("A-mysql", "find_free_port")
    port = old()
    assert isinstance(port, int)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def test_find_free_port_repeated_calls_all_give_usable_ports() -> None:
    ports = {tunnel.find_free_port() for _ in range(5)}
    assert all(1 <= p <= 65535 for p in ports)


# --- ensure_private_key_permissions --------------------------------------------


@ro.live_only("ensure_private_key_permissions")
def test_ensure_private_key_permissions_characterisation_fixes_the_mode(tmp_path: Path) -> None:
    old = ro.get("A-mysql", "ensure_private_key_permissions")

    key_old = tmp_path / "old_key"
    key_new = tmp_path / "new_key"
    key_old.write_text("secret")
    key_new.write_text("secret")
    key_old.chmod(0o644)
    key_new.chmod(0o644)

    old(str(key_old))
    tunnel.ensure_private_key_permissions(str(key_new))

    assert oct(key_old.stat().st_mode)[-3:] == "600"
    assert oct(key_new.stat().st_mode)[-3:] == "600"


@ro.live_only("ensure_private_key_permissions")
def test_ensure_private_key_permissions_characterisation_mode_already_right(
    tmp_path: Path,
) -> None:
    old = ro.get("A-mysql", "ensure_private_key_permissions")

    key_old = tmp_path / "old_key"
    key_new = tmp_path / "new_key"
    key_old.write_text("secret")
    key_new.write_text("secret")
    key_old.chmod(0o600)
    key_new.chmod(0o600)

    old(str(key_old))
    tunnel.ensure_private_key_permissions(str(key_new))

    assert oct(key_old.stat().st_mode)[-3:] == "600"
    assert oct(key_new.stat().st_mode)[-3:] == "600"


@pytest.mark.parametrize("key_path", ["", None])
def test_ensure_private_key_permissions_characterisation_empty_path(
    key_path: Optional[str],
) -> None:
    old = ro.get("A-mysql", "ensure_private_key_permissions")
    # It must not fail, and it must not do anything.
    assert old(key_path) is None
    assert tunnel.ensure_private_key_permissions(key_path) is None  # type: ignore[arg-type]


@ro.live_only("ensure_private_key_permissions")
def test_ensure_private_key_permissions_characterisation_missing_file(tmp_path: Path) -> None:
    old = ro.get("A-mysql", "ensure_private_key_permissions")
    missing_old = str(tmp_path / "missing_old")
    missing_new = str(tmp_path / "missing_new")

    old(missing_old)  # must not raise
    tunnel.ensure_private_key_permissions(missing_new)  # must not raise

    assert not os.path.exists(missing_old)
    assert not os.path.exists(missing_new)


def test_ensure_private_key_permissions_logs_a_missing_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="dbextractors.core.tunnel"):
        tunnel.ensure_private_key_permissions(str(tmp_path / "missing"))
    assert any("not found" in r.getMessage() for r in caplog.records)


# --- normalize_private_key_contents --------------------------------------------


CRLF_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\r\nabc\r\ndef\r\n-----END OPENSSH PRIVATE KEY-----\r\n"
)
LITERAL_NEWLINE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\\nabc\\ndef\\n-----END OPENSSH PRIVATE KEY-----\\n"
)
NO_TRAILING_NEWLINE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----"
)
ALREADY_OK_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n"


@pytest.mark.parametrize(
    "content",
    [CRLF_KEY, LITERAL_NEWLINE_KEY, NO_TRAILING_NEWLINE_KEY, ALREADY_OK_KEY, "no marker at all"],
)
@ro.live_only("normalize_private_key_contents")
def test_normalize_private_key_contents_characterisation(tmp_path: Path, content: str) -> None:
    old = ro.get("A-mysql", "normalize_private_key_contents")

    key_old = tmp_path / "old_key"
    key_new = tmp_path / "new_key"
    key_old.write_text(content, newline="")
    key_new.write_text(content, newline="")

    old(str(key_old))
    tunnel.normalize_private_key_contents(str(key_new))

    assert key_old.read_bytes() == key_new.read_bytes()


@pytest.mark.parametrize("key_path", ["", None])
def test_normalize_private_key_contents_characterisation_empty_path(
    key_path: Optional[str],
) -> None:
    old = ro.get("A-mysql", "normalize_private_key_contents")
    assert old(key_path) is None
    assert tunnel.normalize_private_key_contents(key_path) is None  # type: ignore[arg-type]


@ro.live_only("normalize_private_key_contents")
def test_normalize_private_key_contents_characterisation_missing_file(tmp_path: Path) -> None:
    old = ro.get("A-mysql", "normalize_private_key_contents")
    missing = str(tmp_path / "missing")
    assert old(missing) is None
    assert tunnel.normalize_private_key_contents(missing) is None


def test_normalize_private_key_contents_leaves_the_file_alone_with_nothing_to_change(
    tmp_path: Path,
) -> None:
    key = tmp_path / "key"
    key.write_text(ALREADY_OK_KEY, newline="")
    mtime_before = key.stat().st_mtime_ns

    tunnel.normalize_private_key_contents(str(key))

    assert key.read_text() == ALREADY_OK_KEY
    # The contents are identical -> the file must not be written again.
    assert key.stat().st_mtime_ns == mtime_before


# --- ssh_tunnel_preexec ---------------------------------------------------
# Never start a real fork/ssh: `os.setsid`/`ctypes.CDLL` are faked.


def test_ssh_tunnel_preexec_sets_pdeathsig(monkeypatch: pytest.MonkeyPatch) -> None:
    setsid_calls = []
    monkeypatch.setattr(os, "setsid", lambda: setsid_calls.append(1))

    prctl_calls = []

    class _FakeLibc:
        def prctl(self, option: int, sig: int) -> int:
            prctl_calls.append((option, sig))
            return 0

    cdll_calls = []

    def fake_cdll(name: str, use_errno: bool = False) -> _FakeLibc:
        cdll_calls.append((name, use_errno))
        return _FakeLibc()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    tunnel.ssh_tunnel_preexec()

    assert setsid_calls == [1]
    assert cdll_calls == [("libc.so.6", True)]
    assert prctl_calls == [(1, signal.SIGTERM)]  # PR_SET_PDEATHSIG == 1


def test_ssh_tunnel_preexec_survives_a_failing_prctl(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A swallowed exception must always leave a trace in the log — eleven of
    the fifteen predecessors swallowed this one without a word."""
    monkeypatch.setattr(os, "setsid", lambda: None)

    def fake_cdll(name: str, use_errno: bool = False) -> Any:
        raise OSError("no libc here")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    with caplog.at_level(logging.WARNING, logger="dbextractors.core.tunnel"):
        tunnel.ssh_tunnel_preexec()  # must not raise

    assert any("PR_SET_PDEATHSIG" in r.getMessage() for r in caplog.records)


# --- resolve_connection_mode ---------------------------------------------------


def test_resolve_connection_mode_defaults_to_auto() -> None:
    assert tunnel.resolve_connection_mode({}) == "auto"
    assert tunnel.resolve_connection_mode({"SOURCE_DB": {}}) == "auto"


@pytest.mark.parametrize("mode", ["direct", "ssh", "auto"])
def test_resolve_connection_mode_from_source_db(mode: str) -> None:
    cfg = {"SOURCE_DB": {"connection_mode": mode}}
    assert tunnel.resolve_connection_mode(cfg) == mode


def test_resolve_connection_mode_top_level_as_a_shorthand() -> None:
    cfg = {"connection_mode": "direct"}
    assert tunnel.resolve_connection_mode(cfg) == "direct"


def test_resolve_connection_mode_source_db_beats_top_level() -> None:
    cfg = {"SOURCE_DB": {"connection_mode": "ssh"}, "connection_mode": "direct"}
    assert tunnel.resolve_connection_mode(cfg) == "ssh"


def test_resolve_connection_mode_invalid_value_falls_back_to_auto(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="dbextractors.core.tunnel"):
        result = tunnel.resolve_connection_mode({"connection_mode": "vpn"})
    assert result == "auto"
    assert any("Unknown connection_mode" in r.getMessage() for r in caplog.records)


def test_resolve_connection_mode_is_case_insensitive_and_trims_whitespace() -> None:
    assert tunnel.resolve_connection_mode({"connection_mode": " Direct "}) == "direct"


# --- open_tunnel: connection_mode='direct' -------------------------------------


def test_open_tunnel_direct_does_not_call_the_probe() -> None:
    probe = MagicMock(return_value=True)
    source_db = {"host": "10.0.0.5", "port": 5432}

    with tunnel.open_tunnel(source_db, "direct", probe=probe) as addr:
        assert addr == tunnel.TunnelAddress(host="10.0.0.5", port=5432, mode="direct")

    probe.assert_not_called()


def test_open_tunnel_direct_without_a_host_fails() -> None:
    with (
        pytest.raises(ValueError, match="connection_mode='direct'"),
        tunnel.open_tunnel({}, "direct"),
    ):
        pass


# --- open_tunnel: connection_mode='auto' ---------------------------------------


def test_open_tunnel_auto_a_successful_probe_goes_direct() -> None:
    probe = MagicMock(return_value=True)
    source_db = {"host": "10.0.0.5", "port": 3306}

    with tunnel.open_tunnel(source_db, "auto", probe=probe) as addr:
        assert addr == tunnel.TunnelAddress(host="10.0.0.5", port=3306, mode="direct")

    probe.assert_called_once_with("10.0.0.5", 3306)


def test_open_tunnel_auto_without_a_host_goes_straight_to_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = MagicMock(return_value=True)
    fake_proc = _make_fake_proc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    _patch_killpg(monkeypatch)

    source_db = {
        "ssh_address_or_host": "bastion",
        "ssh_username": "u",
        "ssh_pkey": "/tmp/key",
        "local_bind_address": ["127.0.0.1", 15432],
        "remote_bind_address": ["dbhost", 5432],
    }
    monkeypatch.setattr(tunnel, "normalize_private_key_contents", lambda *a, **kw: None)
    monkeypatch.setattr(tunnel, "ensure_private_key_permissions", lambda *a, **kw: None)

    with tunnel.open_tunnel(source_db, "auto", probe=probe) as addr:
        assert addr.mode == "ssh"

    probe.assert_not_called()


# --- open_tunnel: SSH branch -----------------------------------------------


class _FakeProc:
    """A fake `subprocess.Popen[bytes]` — no real process is ever started."""

    def __init__(self) -> None:
        self.pid = 424242
        self.terminate_calls = 0
        self.wait_calls: list[Optional[float]] = []
        self.communicate_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def wait(self, timeout: Optional[float] = None) -> int:
        self.wait_calls.append(timeout)
        return 0

    def communicate(self, timeout: Optional[float] = None) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        return (b"", b"")


def _make_fake_proc() -> _FakeProc:
    return _FakeProc()


def _patch_killpg(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []

    def fake_getpgid(pid: int) -> int:
        return pid

    def fake_killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    return calls


def _ssh_source_db(**overrides: Any) -> dict:
    base = {
        "ssh_address_or_host": "bastion.example.com",
        "ssh_port": 22,
        "ssh_username": "deploy",
        "ssh_pkey": "/tmp/does-not-matter",
        "local_bind_address": ["127.0.0.1", 15432],
        "remote_bind_address": ["dbhost.internal", 5432],
        "ssh_wait_timeout": 5,
    }
    base.update(overrides)
    return base


def _patch_key_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tunnel, "normalize_private_key_contents", lambda *a, **kw: None)
    monkeypatch.setattr(tunnel, "ensure_private_key_permissions", lambda *a, **kw: None)


def test_open_tunnel_ssh_success_returns_the_local_bind_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _make_fake_proc()
    popen_calls = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: (popen_calls.append((a, kw)), fake_proc)[1]
    )
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    _patch_key_helpers(monkeypatch)
    kill_calls = _patch_killpg(monkeypatch)

    source_db = _ssh_source_db()

    with tunnel.open_tunnel(source_db, "ssh") as addr:
        assert addr == tunnel.TunnelAddress(host="127.0.0.1", port=15432, mode="ssh")

    # preexec_fn has to be the PDEATHSIG variant.
    _, popen_kwargs = popen_calls[0]
    assert popen_kwargs["preexec_fn"] is tunnel.ssh_tunnel_preexec

    # On leaving the `with` the tunnel was cleaned up (SIGTERM to the group).
    assert (fake_proc.pid, signal.SIGTERM) in kill_calls
    assert fake_proc.wait_calls


def test_open_tunnel_ssh_missing_details_fail_without_a_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: popen_calls.append(1))

    with (
        pytest.raises(ValueError, match="no SSH fallback"),
        tunnel.open_tunnel({"host": "10.0.0.1"}, "ssh"),
    ):
        pass

    assert popen_calls == []


def test_open_tunnel_ssh_a_failed_handshake_raises_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _make_fake_proc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: False)
    _patch_key_helpers(monkeypatch)
    kill_calls = _patch_killpg(monkeypatch)

    source_db = _ssh_source_db()

    with (
        pytest.raises(RuntimeError, match="could not be established"),
        tunnel.open_tunnel(source_db, "ssh"),
    ):
        pass

    # The failed process has to be terminated (terminate in _drain_stderr) and
    # the group cleaned up in the `finally` as well (_terminate_tunnel) -> both
    # happened.
    assert fake_proc.terminate_calls == 1
    assert (fake_proc.pid, signal.SIGTERM) in kill_calls


def test_open_tunnel_cleans_up_even_when_the_block_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _make_fake_proc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    _patch_key_helpers(monkeypatch)
    kill_calls = _patch_killpg(monkeypatch)

    source_db = _ssh_source_db()

    class _BoomError(Exception):
        pass

    with pytest.raises(_BoomError), tunnel.open_tunnel(source_db, "ssh"):
        raise _BoomError("it failed a level up, but the tunnel still has to be cleaned up")

    assert (fake_proc.pid, signal.SIGTERM) in kill_calls


def test_open_tunnel_ssh_uses_find_free_port_without_a_local_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _make_fake_proc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    monkeypatch.setattr(tunnel, "find_free_port", lambda: 55555)
    _patch_key_helpers(monkeypatch)
    _patch_killpg(monkeypatch)

    source_db = _ssh_source_db()
    del source_db["local_bind_address"]

    with tunnel.open_tunnel(source_db, "ssh") as addr:
        assert addr.host == "127.0.0.1"
        assert addr.port == 55555


def test_open_tunnel_ssh_passes_the_right_command(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_proc = _make_fake_proc()
    captured = {}

    def fake_popen(cmd: list, **kwargs: Any) -> _FakeProc:
        captured["cmd"] = cmd
        return fake_proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    _patch_key_helpers(monkeypatch)
    _patch_killpg(monkeypatch)

    source_db = _ssh_source_db()

    with tunnel.open_tunnel(source_db, "ssh"):
        pass

    cmd = captured["cmd"]
    assert cmd[0] == "ssh"
    assert "-i" in cmd and "/tmp/does-not-matter" in cmd
    assert "-L" in cmd
    assert "127.0.0.1:15432:dbhost.internal:5432" in cmd
    assert cmd[-1] == "deploy@bastion.example.com"


# --- ssh_host_key_checking ------------------------------------------------
#
# The predecessors opened the tunnel without verifying the far end. That stays
# the default behaviour, because the contract is frozen — but it has to be a
# **choice**, not the only option. These tests guard both: that the default has
# not changed, and that it can be switched.


def _capture_ssh_command(monkeypatch: pytest.MonkeyPatch, source_db: dict) -> list:
    fake_proc = _make_fake_proc()
    captured: dict = {}

    def fake_popen(cmd: list, **kwargs: Any) -> _FakeProc:
        captured["cmd"] = cmd
        return fake_proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    _patch_key_helpers(monkeypatch)
    _patch_killpg(monkeypatch)

    with tunnel.open_tunnel(source_db, "ssh"):
        pass

    return captured["cmd"]


def test_host_key_checking_defaults_to_the_inherited_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the key the tunnel behaves exactly as it does today — otherwise
    existing deployments would break."""
    cmd = _capture_ssh_command(monkeypatch, _ssh_source_db())

    assert "StrictHostKeyChecking=no" in cmd
    assert "UserKnownHostsFile=/dev/null" in cmd


def test_host_key_checking_strict_switches_verification_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd = _capture_ssh_command(monkeypatch, _ssh_source_db(ssh_host_key_checking="strict"))

    assert "StrictHostKeyChecking=yes" in cmd
    # Discarding known_hosts has to go — otherwise verification would run
    # against an empty file and the connection would never come up.
    assert "UserKnownHostsFile=/dev/null" not in cmd


def test_host_key_checking_accept_new(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = _capture_ssh_command(monkeypatch, _ssh_source_db(ssh_host_key_checking="accept-new"))

    assert "StrictHostKeyChecking=accept-new" in cmd
    assert "UserKnownHostsFile=/dev/null" not in cmd


@pytest.mark.parametrize("value", ["STRICT", " strict ", "Accept-New"])
def test_host_key_checking_is_case_insensitive_and_trims_whitespace(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    cmd = _capture_ssh_command(monkeypatch, _ssh_source_db(ssh_host_key_checking=value))

    assert "UserKnownHostsFile=/dev/null" not in cmd


@pytest.mark.parametrize("value", [None, "", "nonsense"])
def test_host_key_checking_an_unknown_value_behaves_like_off(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    """`config.validate` catches an invalid value earlier; were one to slip
    through anyway, the tunnel should rather come up than have the extraction
    fail on a typo."""
    cmd = _capture_ssh_command(monkeypatch, _ssh_source_db(ssh_host_key_checking=value))

    assert "StrictHostKeyChecking=no" in cmd


# --- _default_probe (without a real network) ------------------------------


def test_default_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ctx:
        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: _Ctx())
    assert tunnel._default_probe("10.0.0.1", 5432) is True


def test_default_probe_failure_after_every_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fail(*a: object, **kw: object) -> None:
        calls.append(1)
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    assert tunnel._default_probe("10.0.0.1", 5432) is False
    assert len(calls) == tunnel._DEFAULT_PROBE_ATTEMPTS


def test_open_tunnel_auto_uses_the_default_probe_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an injected `probe` the built-in TCP fallback is used — verified
    against a genuinely closed local port, so no traffic leaves localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_sock:
        probe_sock.bind(("127.0.0.1", 0))
        closed_port = probe_sock.getsockname()[1]

    monkeypatch.setattr(tunnel, "_DEFAULT_PROBE_ATTEMPTS", 1)
    monkeypatch.setattr(tunnel, "_DEFAULT_PROBE_TIMEOUT", 0.2)

    fake_proc = _make_fake_proc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(tunnel, "wait_for_port", lambda *a, **kw: True)
    _patch_key_helpers(monkeypatch)
    _patch_killpg(monkeypatch)

    source_db = _ssh_source_db(host="127.0.0.1", port=closed_port)

    with tunnel.open_tunnel(source_db, "auto") as addr:
        assert addr.mode == "ssh"
