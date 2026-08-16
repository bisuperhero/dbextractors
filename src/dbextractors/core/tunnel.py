"""SSH tunnel, private key handling and the choice of connection method.

Two project rules bear directly on this module:

- start the SSH tunnel with ``PR_SET_PDEATHSIG``, otherwise orphaned processes are
  left behind;
- the choice of connection is explicit: ``connection_mode: direct | ssh | auto``,
  with ``auto`` only as the legacy default.

The predecessors do not build the tunnel with the ``sshtunnel`` library but with
``subprocess`` and a hand-written ``preexec_fn`` — hence ``ctypes`` and ``signal``
among the imports.

**The winning variant of ``ssh_tunnel_preexec`` comes from variant A** (its MySQL
extractor and three further files) — those four out of fifteen are the only ones
whose ``preexec_fn`` sets ``PR_SET_PDEATHSIG``. The remaining eleven files with SSH
have a bare ``preexec_fn=os.setsid`` and leave orphaned ``ssh`` processes on the host
when the block is OOM-killed.

``find_free_port``, ``ensure_private_key_permissions`` and
``normalize_private_key_contents`` show up as two "variants" each, but again they
differ only in docstring and formatting, not in behaviour — the same variant A file
wins, for consistency with ``ssh_tunnel_preexec``.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Optional, cast

from dbextractors.core import secrets
from dbextractors.core.retry import wait_for_port

_log = logging.getLogger(__name__)

#: Explicit choice of connection method. `auto` is the legacy default: it tries a
#: direct connection first and builds the tunnel only if that does not work.
ConnectionMode = Literal["direct", "ssh", "auto"]

DEFAULT_CONNECTION_MODE: ConnectionMode = "auto"

_VALID_CONNECTION_MODES: frozenset[str] = frozenset({"direct", "ssh", "auto"})

#: `PR_SET_PDEATHSIG` from <linux/prctl.h> — the kernel sends this signal to the
#: child when the parent thread that started it dies (e.g. the block is OOM-killed).
_PR_SET_PDEATHSIG = 1

#: Timing of the existing probe-and-fallback behaviour in `auto` mode: 3 attempts,
#: a 10 s timeout, 3 s between attempts. Used only as the default `probe` when the
#: caller supplies none — a real dialect (`SourceDialect.probe`, see
#: ARCHITECTURE.md) always takes precedence. This is the safe fallback that lets
#: `core.tunnel` be tested and used on its own, without depending on the dialects.
_DEFAULT_PROBE_ATTEMPTS = 3
_DEFAULT_PROBE_TIMEOUT = 10.0
_DEFAULT_PROBE_DELAY = 3.0


@dataclass(frozen=True)
class TunnelAddress:
    """The address and port the SQLAlchemy engine should connect to.

    ``mode`` is the method *actually used* (``"direct"`` or ``"ssh"``) — with
    ``connection_mode="auto"`` it is only known at run time, depending on whether the
    probe succeeded. Useful for the log and the status row.
    """

    host: str
    port: int
    mode: Literal["direct", "ssh"]


def find_free_port() -> int:
    """Finds a free local port for ``local_bind_address``.

    Opens a socket on port 0 (the kernel allocates a free one), reads it back and
    closes the socket immediately. There is a theoretical window between reading and
    using it (TOCTOU), but that is true of the existing implementation too — nothing
    new is introduced here.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return cast(int, sock.getsockname()[1])


def ensure_private_key_permissions(key_path: str, show_debug: bool = False) -> None:
    """Sets 0600 on the private key, otherwise ssh refuses it.

    A missing key, or being unable to read or set the permissions, is not fatal (ssh
    fails a little later with a comprehensible message anyway) — but the project rule
    is that it has to be logged, not silently swallowed.

    ``key_path`` goes through `secrets.redact` before it is logged. A path is not a
    secret and comes out unchanged; a configuration that put the **key itself** in
    ``ssh_pkey`` is what this guards against — ``os.stat`` then fails with
    ``ENAMETOOLONG`` and this line used to print the whole private key. See
    `secrets._PRIVATE_KEY_BLOCK`.
    """
    if not key_path:
        return
    safe_path = secrets.redact(key_path)
    try:
        current_mode = stat.S_IMODE(os.stat(key_path).st_mode)
    except FileNotFoundError:
        _log.warning("⚠️ SSH key %s not found while checking permissions.", safe_path)
        return
    except OSError as err:
        _log.error("⚠️ Could not inspect permissions for %s: %s", safe_path, secrets.redact(err))
        return

    if current_mode != 0o600:
        if show_debug:
            _log.warning(
                "🔧 Adjusting SSH key permissions for %s: %s -> 0o600", safe_path, oct(current_mode)
            )
        try:
            os.chmod(key_path, 0o600)
        except OSError as err:
            _log.error("⚠️ Failed to set 0600 permissions on %s: %s", safe_path, secrets.redact(err))


def normalize_private_key_contents(key_path: str, show_debug: bool = False) -> None:
    """Normalises line endings and the missing trailing newline in the key file.

    Handles two common breakages from copy-pasting through YAML or a UI: a literal
    ``\\n`` instead of a real line break inside the ``-----BEGIN...`` block, and CRLF
    line endings. The file is rewritten only when something actually changed.
    """
    if not key_path or not os.path.exists(key_path):
        return
    safe_path = secrets.redact(key_path)
    try:
        with open(key_path, encoding="utf-8") as handle:
            original = handle.read()
    except OSError as err:
        _log.error("⚠️ Could not read SSH key %s: %s", safe_path, secrets.redact(err))
        return

    normalized = original.replace("\r\n", "\n")
    replaced_literal_newlines = False
    if "-----BEGIN" in normalized and "\\n" in normalized:
        normalized = normalized.replace("\\n", "\n")
        replaced_literal_newlines = True
    if not normalized.endswith("\n"):
        normalized += "\n"

    if normalized != original:
        try:
            with open(key_path, "w", encoding="utf-8") as handle:
                handle.write(normalized)
            if show_debug:
                _log.warning(
                    "🔧 Normalized SSH key file %s (literal \\n replaced: %s).",
                    safe_path,
                    replaced_literal_newlines,
                )
        except OSError as err:
            _log.error("⚠️ Could not rewrite SSH key %s: %s", safe_path, secrets.redact(err))


def ssh_tunnel_preexec() -> None:
    """``preexec_fn`` for ``subprocess`` — sets ``PR_SET_PDEATHSIG``.

    Runs in the forked child (between ``fork`` and ``exec``): a new session (so the
    tunnel can be killed as a whole process group) plus a request to the kernel to
    take this ssh process down with SIGTERM when the parent — the block that started
    it — dies unexpectedly, e.g. from an OOM-kill. Without this, orphaned ssh
    processes are left on the host after a pipeline crash. This is the existing
    ``_ssh_tunnel_preexec``, in the single variant that variant A's files carry.

    ``prctl`` exists only on Linux with glibc (exactly what runs in the production
    Mage image) — on another platform it fails. That is deliberately a secondary,
    best-effort step: the main protection is the ``os.setsid`` one line above. Hence
    the broad catch that only logs, rather than the ``except Exception: pass`` found
    in eleven of the fifteen predecessor files.
    """
    os.setsid()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception as err:
        _log.warning("⚠️ Could not set PR_SET_PDEATHSIG on the SSH tunnel process: %s", err)


def resolve_connection_mode(config: dict) -> ConnectionMode:
    """Works out from the configuration how to connect.

    ``connection_mode`` is a new key with a safe default — the frozen contract allows
    keys to be **added**, not existing ones renamed. Its placement (identical to
    ``core/config.py``): primarily ``SOURCE_DB.connection_mode``, since it is a
    sibling of the ``ssh_*`` keys and concerns how the source is reached, with a
    top-level ``connection_mode`` as a convenient shorthand. The key exists in none
    of the 15 predecessor files, so there is no legacy placement to honour.

    This function is deliberately more forgiving than ``core/config.py``: that one
    raises ``ConfigError`` on an invalid value during ``parse()`` (validation at the
    entry point), which is the main, strict path. This function serves the tunnel
    when it is used on its own — without a full pass through ``config.parse()``, e.g.
    in a test or a future dialect — where an invalid value (a typo, an old
    ``"tunnel"``/``"auto-detect"`` and the like) is no reason to fail: it falls back
    to ``auto``, the behaviour from before the switch existed, and logs a warning.
    """
    source_db_cfg = config.get("SOURCE_DB") or {}
    raw = source_db_cfg.get("connection_mode") or config.get("connection_mode")
    if not raw:
        return DEFAULT_CONNECTION_MODE

    value = str(raw).strip().lower()
    if value not in _VALID_CONNECTION_MODES:
        _log.warning(
            "⚠️ Unknown connection_mode=%r, falling back to %r.",
            raw,
            DEFAULT_CONNECTION_MODE,
        )
        return DEFAULT_CONNECTION_MODE
    return cast(ConnectionMode, value)


@contextmanager
def open_tunnel(
    source_db: dict,
    mode: Optional[ConnectionMode] = None,
    *,
    probe: Optional[Callable[[str, int], bool]] = None,
    show_debug: bool = False,
) -> Iterator[TunnelAddress]:
    """Builds the tunnel (or does not) and returns the address to connect to.

    A context manager: ``with open_tunnel(source_db, mode) as addr: ...``. Inside the
    block ``addr.host``/``addr.port`` is a valid address; on leaving the block — by
    an exception too — the SSH process is always cleaned up. Previously this was a
    pair of "start the tunnel" / "remember to kill it at the end" statements
    scattered through ``load_data``.

    ``mode=None`` means ``DEFAULT_CONNECTION_MODE`` (``"auto"``).

    - ``"direct"`` / ``"ssh"``: the probe is not called at all, the chosen path is
      taken straight away. The earlier behaviour always tried a direct connection,
      even where it was clear that SSH was needed anyway — up to ~36 s of extra delay
      (3 attempts × 10 s timeout + 3 s waits) and, for a deployment without SSH, a
      confusing error message only after all of that.
    - ``"auto"``: tries a direct connection first (if ``SOURCE_DB.host`` is filled in)
      through ``probe`` (or the built-in fallback, see ``_DEFAULT_PROBE_*``), and
      builds the SSH tunnel when that does not succeed. This is the behaviour of all
      12 predecessor files with SSH, unchanged.

    ``probe`` carries the same responsibility as ``SourceDialect.probe`` in
    ARCHITECTURE.md — a dialect may pass its own version (different timeouts or ports
    per database). Without one, a generic TCP probe is used, so that this module can
    be tested and used before the dialects exist.

    The predecessor's ``direct_conn_success = True`` is deliberately not carried
    over — it is an unreachable line after a ``try/except`` in which both branches
    ``return``. Here the `yield` inside the `with` does the same thing one level more
    cleanly: once the address has been handed over, there is nothing left to "mark as
    successful".
    """
    resolved: ConnectionMode = mode or DEFAULT_CONNECTION_MODE
    host = source_db.get("host")
    port = source_db.get("port")

    if resolved == "direct":
        if not host or not port:
            raise ValueError(
                "connection_mode='direct' requires SOURCE_DB.host and SOURCE_DB.port to be set."
            )
        yield TunnelAddress(host=str(host), port=int(port), mode="direct")
        return

    if (
        resolved == "auto"
        and host
        and port
        and _probe_direct(str(host), int(port), probe, show_debug=show_debug)
    ):
        yield TunnelAddress(host=str(host), port=int(port), mode="direct")
        return

    with _open_ssh_tunnel(source_db, host, port, show_debug=show_debug) as addr:
        yield addr


def _probe_direct(
    host: str,
    port: int,
    probe: Optional[Callable[[str, int], bool]],
    *,
    show_debug: bool,
) -> bool:
    """Tries a direct connection through ``probe`` (or the built-in fallback)."""
    if show_debug:
        _log.warning("🔌 Trying direct connection to %s:%s...", host, port)
    check = probe or _default_probe
    reachable = check(host, port)
    if reachable:
        if show_debug:
            _log.warning("✅ Direct connection to %s:%s succeeded.", host, port)
    else:
        _log.warning("🛡️ Direct connection to %s:%s not reachable, falling back to SSH.", host, port)
    return reachable


def _default_probe(host: str, port: int) -> bool:
    """Generic TCP probe for ``auto`` mode when the caller supplies none.

    The timing matches the existing ``is_*_accessible`` functions across the
    predecessors: 3 attempts, a 10 s timeout each, 3 s between them. The real
    ``is_mysql_accessible`` / ``is_pg_accessible`` / ``is_mssql_accessible`` belong to
    the dialects — this is only a safe TCP-level default.
    """
    for attempt in range(1, _DEFAULT_PROBE_ATTEMPTS + 1):
        try:
            with socket.create_connection((host, port), timeout=_DEFAULT_PROBE_TIMEOUT):
                return True
        except OSError as err:
            _log.debug(
                "%s:%s not reachable (attempt %d/%d): %s",
                host,
                port,
                attempt,
                _DEFAULT_PROBE_ATTEMPTS,
                err,
            )
            if attempt < _DEFAULT_PROBE_ATTEMPTS:
                time.sleep(_DEFAULT_PROBE_DELAY)
    return False


def _host_key_options(mode: Any) -> list:
    """``ssh`` options for verifying the host key.

    The predecessors open the tunnel with ``StrictHostKeyChecking=no`` and
    ``UserKnownHostsFile=/dev/null``, so the other end is not verified at all. That
    stays the default behaviour (``off``) — the contract is frozen and changing the
    default would break runs that work today. Anyone who wants verification asks for
    it with the ``SOURCE_DB.ssh_host_key_checking`` key.

    ``accept-new`` remembers the host key on the first connection and refuses it if
    it changes; ``strict`` requires the host to be in ``known_hosts`` beforehand. Both
    use the default ``known_hosts`` of the user Mage runs as.

    An unknown value never reaches this function — `config.validate` catches it
    earlier. If one did, we behave as ``off``, so that the tunnel is more likely to
    come up than the extraction to fail on a typo in the configuration.
    """
    normalized = str(mode or "off").strip().lower()
    if normalized == "strict":
        return ["-o", "StrictHostKeyChecking=yes"]
    if normalized == "accept-new":
        return ["-o", "StrictHostKeyChecking=accept-new"]
    return ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]


@contextmanager
def _open_ssh_tunnel(
    source_db: dict,
    private_host: Any,
    private_port: Any,
    *,
    show_debug: bool,
) -> Iterator[TunnelAddress]:
    """Builds the SSH tunnel with ``subprocess`` and tears it down on leaving the block."""
    ssh_host = source_db.get("ssh_address_or_host")
    ssh_port = int(source_db.get("ssh_port") or 22)
    ssh_user = source_db.get("ssh_username")
    ssh_pkey = source_db.get("ssh_pkey")
    ssh_wait_timeout = float(source_db.get("ssh_wait_timeout") or 20)

    if not all([ssh_host, ssh_user, ssh_pkey]):
        raise ValueError(
            "Direct connection failed or was skipped, and no SSH fallback is configured "
            "(ssh_address_or_host/ssh_username/ssh_pkey empty in SOURCE_DB)."
        )
    ssh_pkey = str(ssh_pkey)

    normalize_private_key_contents(ssh_pkey, show_debug=show_debug)
    ensure_private_key_permissions(ssh_pkey, show_debug=show_debug)

    remote_bind = source_db.get("remote_bind_address")
    if remote_bind:
        remote_host, remote_port = remote_bind
    else:
        remote_host, remote_port = private_host or "localhost", private_port

    local_bind = source_db.get("local_bind_address")
    if local_bind:
        local_host, local_port = local_bind
    else:
        local_host, local_port = "127.0.0.1", find_free_port()
    if not local_port or int(local_port) == 0:
        local_port = find_free_port()
    local_host = str(local_host)
    local_port = int(local_port)

    ssh_cmd = [
        "ssh",
        *_host_key_options(source_db.get("ssh_host_key_checking")),
        "-i", str(ssh_pkey),
        "-N",
        "-L", f"{local_host}:{local_port}:{remote_host}:{remote_port}",
        "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}",
    ]  # fmt: skip

    if show_debug:
        # The argv is logged as it is built, minus anything that reads as a private
        # key. The ssh user, host, ports and key **path** stay — that line exists so
        # that a failing tunnel can be reproduced by hand, and without them it would
        # not be reproducible.
        _log.warning("🛡️ Starting SSH tunnel: %s", secrets.redact(" ".join(ssh_cmd)))

    proc = subprocess.Popen(
        ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=ssh_tunnel_preexec
    )
    try:
        if not wait_for_port(local_port, host=local_host, timeout=ssh_wait_timeout):
            stderr_text = _drain_stderr(proc)
            detail = f" stderr: {stderr_text}" if stderr_text else ""
            raise RuntimeError(
                f"SSH tunnel could not be established within {ssh_wait_timeout}s "
                f"on {local_host}:{local_port}.{detail}"
            )
        if show_debug:
            _log.warning("✅ SSH tunnel established on port %s.", local_port)
        yield TunnelAddress(host=local_host, port=local_port, mode="ssh")
    finally:
        _terminate_tunnel(proc, show_debug=show_debug)


def _drain_stderr(proc: subprocess.Popen[bytes]) -> str:
    """Terminates a failed tunnel and pulls its ``stderr`` out for the error message.

    The text is redacted before it is returned. ``ssh`` echoes its own ``-i``
    argument back on failure (``Warning: Identity file … not accessible``), so a
    configuration that put the key itself in ``ssh_pkey`` gets the whole private
    key into the ``RuntimeError`` this feeds — triggered against OpenSSH, see
    `tests/core/test_credential_leaks.py`.
    """
    try:
        proc.terminate()
    except OSError as err:
        _log.warning("⚠️ Could not terminate the failed SSH process: %s", secrets.redact(err))

    try:
        _, stderr_data = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc, sig=signal.SIGKILL)
        try:
            _, stderr_data = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _log.warning("⚠️ SSH process did not exit even after SIGKILL; giving up on its stderr.")
            stderr_data = b""

    if not stderr_data:
        return ""
    return secrets.redact(stderr_data.decode("utf-8", errors="ignore").strip())


def _terminate_tunnel(proc: subprocess.Popen[bytes], *, show_debug: bool) -> None:
    """Terminates the whole tunnel process group — SIGTERM, then SIGKILL if needed.

    It is called from a ``finally``, so it must never raise — that would mask the
    original exception (or bring down an otherwise successful run during teardown).
    Anything that goes wrong here is logged as a warning.
    """
    try:
        _kill_process_group(proc, sig=signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc, sig=signal.SIGKILL)
            proc.wait(timeout=5)
        if show_debug:
            _log.warning("🛡️ SSH tunnel terminated.")
    except Exception as err:
        _log.warning("⚠️ Error while terminating the SSH tunnel: %s", err)


def _kill_process_group(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    """Signals the whole tunnel process group, not just ``ssh`` itself.

    The group exists thanks to ``os.setsid()`` in ``ssh_tunnel_preexec`` — which also
    means any subprocesses ssh spawned along the way are killed with it.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        _log.debug("🔍 SSH tunnel process group %s already gone.", proc.pid)
