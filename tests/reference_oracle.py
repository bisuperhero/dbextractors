"""The oracle for the characterisation tests: the **predecessor** functions.

Every conversion in this package was written to match behaviour that already
existed, and the tests prove it by running both implementations over the same
inputs. That requires calling the old functions — which cannot simply be
imported, because in the predecessor extractors they are:

1. **closures inside a ``load_data`` body**, not module-level functions
2. in files whose first line imports a runtime that is not installed here
3. in two cases, importing modules from deployment-specific repositories

So the file is parsed as an AST, only what is safe to evaluate is executed
(imports and module constants), and **every nested function is flattened** into
one namespace. Mutual calls keep working, because Python resolves names at call
time rather than at definition time.

What this does **not** recover: functions that reach for a live connection or
for other locals of ``load_data``. They get defined but raise ``NameError``
when called. That is fine — the ones under test are pure functions over data.

This module never edits or runs anything in production. It reads files and
evaluates them in an isolated dict.

Two modes
---------

The predecessor sources are not part of this repository. The module therefore
runs in two modes, and the tests do not need to know which:

``live``
    The sources are present. Behaves as described above, extracting from the
    AST. With ``DBX_ORACLE_RECORD=1`` it additionally **records** every call
    into ``tests/fixtures/oracle/``.

``replay``
    The sources are absent, which is the normal case here. Instead of the old
    function a replayer is returned, which looks up the recorded answer for the
    same input. A call that was never recorded fails with an explanation rather
    than a ``FileNotFoundError``.

The mode is detected from whether the directory exists; ``DBX_ORACLE_MODE``
forces it, which is how CI checks that the fixtures really are sufficient
without having to delete anything.

Fixtures are recorded where the predecessor sources are available::

    DBX_ORACLE_RECORD=1 pytest

If an old function or a test's inputs change, that has to be re-run — otherwise
replay reports a missing call, which is the intended failure rather than a
silent pass.
"""

from __future__ import annotations

import ast
import atexit
import builtins
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable

import oracle_store

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = REPO_ROOT / "reference"

#: Modules that are not installed here. Their imports are skipped.
#:
#: Only relevant in ``live`` mode. Two of the predecessor sources also import
#: deployment-specific modules; wherever those sources are kept, their module
#: prefixes are added here through ``DBX_ORACLE_SKIP_IMPORTS`` (comma-separated)
#: rather than being written into the package.
UNAVAILABLE_PREFIXES: tuple[str, ...] = (
    "mage_ai",
    *(p.strip() for p in os.environ.get("DBX_ORACLE_SKIP_IMPORTS", "").split(",") if p.strip()),
)

#: Short keys for the predecessor sources. The prefix letter denotes the variant.
FILES: dict[str, str] = {
    "C-mssql": "variant-c/a_mssql_streaming_extractor.py",
    "C-mysql": "variant-c/a_mysql_streaming_extractor.py",
    "C-my-v2": "variant-c/a_mysql_streaming_extractor_v2.py",
    "C-my-uni": "variant-c/a_mysql_universal_extractor.py",
    "C-pgread": "variant-c/a_postgresql_table_reader.py",
    "A-my-ld": "variant-a/a_mysql_loader.py",
    "A-mysql": "variant-a/a_mysql_streaming_extractor.py",
    "A-pg-ld": "variant-a/a_postgresql_loader.py",
    "A-pg": "variant-a/a_postgresql_streaming_extractor.py",
    "B-fbird": "variant-b/a_firebird_streaming_extractor.py",
    "B-mssql": "variant-b/a_mssql_streaming_extractor.py",
    "B-mysql": "variant-b/a_mysql_streaming_extractor.py",
    "B-my-uni": "variant-b/a_mysql_universal_extractor.py",
    "B-pg-ld": "variant-b/a_postgresql_loader.py",
    "B-pg": "variant-b/a_postgresql_streaming_extractor.py",
}


class OracleError(RuntimeError):
    """The function could not be extracted or replayed."""


# --- mode -------------------------------------------------------------------


def _resolve_mode() -> str:
    forced = (os.environ.get("DBX_ORACLE_MODE") or "").strip().lower()
    if forced in ("live", "replay"):
        return forced
    if forced:
        raise OracleError(f"DBX_ORACLE_MODE must be 'live' or 'replay', got {forced!r}.")
    return "live" if REFERENCE.is_dir() else "replay"


MODE = _resolve_mode()
RECORDING = MODE == "live" and os.environ.get("DBX_ORACLE_RECORD") == "1"

if MODE == "live" and not REFERENCE.is_dir():
    raise OracleError(
        "DBX_ORACLE_MODE=live, but the predecessor sources are not present. "
        "Without them the fixtures are used instead (replay)."
    )

#: Functions that **cannot** be recorded and replayed, with the reason.
#:
#: Recording handles input -> output and mutation of passed-in collections. It
#: does not stretch to two cases, and saying so is better than pretending the
#: coverage exists:
#:
#: 1. **Side effects outside the arguments.** ``ensure_private_key_permissions``
#:    changes a file's mode on disk and ``normalize_private_key_contents``
#:    rewrites its contents. The test inspects that file, not the return value,
#:    and a replayer never touches it.
#: 2. **A function passed as an argument.** ``with_retry(operation, …)`` is
#:    entirely about how many times ``operation`` is called and with what
#:    delays. Replaying a "result" would verify nothing.
#:
#: Tests that rely on these are skipped in ``replay`` mode. They are covered
#: where the predecessor sources exist, and the new implementations have their
#: own direct tests that do not depend on the oracle.
LIVE_ONLY: dict[str, str] = {
    "ensure_private_key_permissions": "changes a file mode on disk",
    "normalize_private_key_contents": "rewrites a file on disk",
    "find_free_port": "binds a socket; the result differs every time",
    "wait_for_port": "touches the network",
    "with_retry": "takes a function as an argument",
    "_ssh_tunnel_preexec": "calls prctl/setsid in the process",
}


def live_only(function: str) -> Any:
    """A `skipif` marker for a test that makes no sense in ``replay`` mode.

    Usage::

        @ro.live_only("with_retry")
        def test_something() -> None: ...
    """
    import pytest

    duvod = LIVE_ONLY.get(function, "requires the predecessor sources")
    return pytest.mark.skipif(
        MODE == "replay",
        reason=f"{function}: {duvod} — the oracle cannot be replayed",
    )


#: What this run recorded. Written out at the end so the files are not
#: rewritten on every call — there are over a thousand tests.
_recorded: dict[tuple[str, str], dict] = {}
_recorded_variants: dict[str, list[str]] = {}


def _flush_recording() -> None:
    """Write the recorded calls, merging with what the fixtures already hold.
    A single ``pytest -k …`` run must not erase the rest of the suite."""
    if not RECORDING:
        return
    for (short, function), calls in _recorded.items():
        merged = oracle_store.load(short, function)
        merged.update(calls)
        oracle_store.save(short, function, merged)

    index = oracle_store.load_index()
    index.setdefault("variants", {}).update(_recorded_variants)
    oracle_store.save_index(index)


atexit.register(_flush_recording)


class _RecordingLogger:
    """Stand-in for the runtime logger.

    The old functions log through a closed-over ``logger``. Discarding the
    messages would work, but they help when a difference has to be explained,
    so they are kept.
    """

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def _record(self, level: str) -> Callable[..., None]:
        def write(message: Any = "", *args: Any, **kwargs: Any) -> None:
            self.messages.append((level, str(message)))

        return write

    def __getattr__(self, name: str) -> Callable[..., None]:
        return self._record(name)


def _safe_import_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Imports, except those whose modules are not installed here.

    Walks the **whole tree**, not just module level: some sources have
    ``import json`` inside ``load_data``, where a top-level scan misses it.
    """
    keep: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(n.startswith(UNAVAILABLE_PREFIXES) for n in names):
            continue
        keep.append(node)
    return keep


#: Calls that may be evaluated while extracting constants. They must be free
#: of side effects — see `_constant_nodes`.
_PURE_CALLS: frozenset[str] = frozenset(
    {"frozenset", "set", "dict", "list", "tuple", "str", "int", "float", "bool", "compile"}
)


def _is_pure_value(node: ast.expr) -> bool:
    """Can this expression be evaluated without side effects?

    **This is a safety guard, not an optimisation.** The predecessor sources
    have lines like ``engine = create_engine(...)`` and
    ``conn = psycopg2.connect(...)`` inside ``load_data``. If constants were
    extracted by blindly trying and catching exceptions, those lines would
    succeed — and every test run would try to connect to a production database.
    """
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass

    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _PURE_CALLS:
            return all(_is_pure_value(arg) for arg in node.args)
        return False

    # A container is pure when all of its elements are. This is why:
    #     invalid_date_patterns = [re.compile(r'^0{4}-0{2}-0{2}$'), ...]
    # a list of calls, which `literal_eval` alone cannot handle.
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_is_pure_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_pure_value(k) for k in node.keys) and all(
            _is_pure_value(v) for v in node.values
        )
    return False


def _constant_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Constant assignments, at module level and inside ``load_data``.

    Only what passes ``_is_pure_value`` is taken. That yields things like
    ``INT_MAX``, type maps and ``invalid_date_patterns`` (a list of regular
    expressions defined inside ``load_data``) while running nothing that
    reaches outside the process.
    """
    keep: list[ast.stmt] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and _is_pure_value(node.value)
            or (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and _is_pure_value(node.value)
            )
        ):
            keep.append(node)
    return keep


def _all_function_defs(tree: ast.Module) -> list[ast.FunctionDef]:
    """Every ``def`` in the file, at any nesting depth.

    Depth matters: ``to_int_or_none`` is nested inside
    ``sanitize_integer_columns`` and ``to_time_str`` inside
    ``convert_time_columns``. Taking only the definitions directly under
    ``load_data`` would miss half of the functions under test.
    """
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


@functools.cache
def _namespace(short: str) -> dict:
    """The namespace of one predecessor source. Computed once."""
    if short not in FILES:
        raise OracleError(f"Unknown source {short!r}. Known: {', '.join(sorted(FILES))}")

    path = REFERENCE / FILES[short]
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))

    namespace: dict[str, Any] = {
        "__name__": f"dbx_oracle_{short.replace('-', '_')}",
        "__file__": str(path),
        "logger": _RecordingLogger(),
        "logging": logging,
        # Runtime decorators. The old files put them on `load_data` /
        # `test_output`, which are not needed — but without them the `def`
        # would not evaluate.
        "data_loader": lambda fn: fn,
        "test": lambda fn: fn,
        "kwargs": {},
        # Closed-over variables the old functions read from `load_data`. In
        # production their value comes from the configuration; for
        # characterisation a default is enough, but it has to exist or the
        # call raises NameError.
        "show_debug": False,
    }

    for node in _safe_import_nodes(tree):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)

    for node in _constant_nodes(tree):
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        except Exception:
            # A constant that depends on something unavailable. Skipped
            # deliberately: the missing name surfaces when a function that
            # needs it is called, where the error is easier to read.
            continue

    for node in _all_function_defs(tree):
        if node.name in ("load_data", "test_output"):
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        except Exception:
            continue

    return namespace


def _recording_wrapper(short: str, function: str, fn: Callable) -> Callable:
    """The old function, wrapped so that it records its input and output.

    Exceptions are recorded too: some characterisation tests assert precisely
    that the old function fails on a given input, and without that the replay
    would behave differently from the live oracle.
    """

    @functools.wraps(fn)
    def zaznamenavajici(*args: Any, **kwargs: Any) -> Any:
        key = oracle_store.fingerprint(args, kwargs)
        calls = _recorded.setdefault((short, function), {})
        try:
            result = fn(*args, **kwargs)
        except Exception as err:
            calls[key] = {
                "args": oracle_store.encode(list(args)),
                "kwargs": oracle_store.encode(kwargs),
                "raises": {"type": type(err).__name__, "msg": str(err)},
            }
            raise
        calls[key] = {
            "args": oracle_store.encode(list(args)),
            "kwargs": oracle_store.encode(kwargs),
            # The state of mutable arguments **after** the call. Some old
            # functions return nothing and deliver their result by mutating a
            # passed-in `set` or `dict` — `update_stats(name, series,
            # all_columns, stats)` is the case this exists for. Without it the
            # replay would return `None` and the caller would get empty
            # collections.
            "args_after": oracle_store.encode(list(args)),
            "result": oracle_store.encode(result),
        }
        return result

    return zaznamenavajici


def _apply_out_params(args: tuple, after: list) -> None:
    """Bring mutable arguments to the state they had after the live call.

    Only ``set``, ``dict`` and ``list`` — what the old functions actually use
    as output parameters. Mutated **in place**, because the caller holds the
    same reference; replacing it with a new object would never reach them.
    """
    for original, encoded in zip(args, after, strict=False):
        if not isinstance(original, set | dict | list):
            continue
        updated = oracle_store.decode(encoded)
        if type(updated) is not type(original):
            continue
        original.clear()
        if isinstance(original, list):
            original.extend(updated)
        else:
            original.update(updated)


def _replay_function(short: str, function: str) -> Callable:
    """A stand-in for the old function that answers from the fixtures."""
    calls = oracle_store.load(short, function)
    if not calls:
        raise OracleError(
            f"{function!r} in {short} has not been recorded. Fixtures are "
            f"recorded with DBX_ORACLE_RECORD=1 where the predecessor sources exist."
        )

    def prehravajici(*args: Any, **kwargs: Any) -> Any:
        key = oracle_store.fingerprint(args, kwargs)
        call = calls.get(key)
        if call is None:
            raise OracleError(
                f"No recording of {short}.{function} for this input "
                f"(fingerprint {key}); {len(calls)} calls are recorded. If the test "
                f"or its inputs changed, re-record with DBX_ORACLE_RECORD=1 where "
                f"the predecessor sources exist."
            )
        if "raises" in call:
            typ = getattr(builtins, call["raises"]["type"], None)
            raise (typ or RuntimeError)(call["raises"]["msg"])

        _apply_out_params(args, call.get("args_after", []))
        return oracle_store.decode(call["result"])

    prehravajici.__name__ = function
    prehravajici.__qualname__ = f"{short}.{function}"
    return prehravajici


def get(short: str, function: str) -> Callable:
    """Return the old function ``function`` from source ``short``.

    In ``replay`` mode this is a replayer over the fixtures rather than the
    real function. The caller cannot tell, and should not have to — otherwise
    every test would need to know about both.

    Args:
        short: Source key, e.g. ``'A-mysql'``.
        function: Function name.

    Raises:
        OracleError: When the function could not be extracted or replayed.
    """
    if short not in FILES:
        raise OracleError(f"Unknown source {short!r}. Known: {', '.join(sorted(FILES))}")

    if MODE == "replay":
        return _replay_function(short, function)

    namespace = _namespace(short)
    found = namespace.get(function)
    if not callable(found):
        raise OracleError(
            f"{function!r} could not be extracted from {short}. Either it is not "
            "in that source, or it depends on something not installed here."
        )
    return _recording_wrapper(short, function, found) if RECORDING else found


def variants(function: str) -> dict[str, Callable]:
    """Every available variant of one function, keyed by source.

    Used where the variants are compared against each other — that is, for
    functions that differ between the predecessors.

    Called inside `pytest.mark.parametrize`, so **at collection time**. In
    ``replay`` mode the list therefore comes from the recorded index rather
    than from whichever fixture files happen to exist: otherwise a repository
    without the sources would collect a different number of tests than the one
    that recorded them, and nobody would notice.
    """
    if MODE == "replay":
        shorts = oracle_store.load_index().get("variants", {}).get(function, [])
        return {short: _replay_function(short, function) for short in shorts}

    found: dict[str, Callable] = {}
    for short in FILES:
        try:
            found[short] = get(short, function)
        except OracleError:
            continue
    if RECORDING:
        _recorded_variants[function] = sorted(found)
    return found


def logger_of(short: str) -> _RecordingLogger:
    """Log messages captured for one source. For debugging.

    ``live`` mode only — messages are not recorded, because the single test
    that reads them exercises the AST extraction itself, not the behaviour of
    any old function.
    """
    if MODE == "replay":
        raise OracleError("logger_of() needs the predecessor sources; replay has nothing to read.")
    return _namespace(short)["logger"]
