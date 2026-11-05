"""Standalone macOS proc_pidinfo probe; run with python -I on a Mac host."""
from __future__ import annotations

import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _load_adapter():
    adapter_path = Path(__file__).resolve().with_name("_orphan_guard.py")
    spec = importlib.util.spec_from_file_location("_orphan_guard_mac_probe_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load orphan guard adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _LoggedProcPidInfo:
    def __init__(self, function, calls: list[dict[str, object]], identity_struct) -> None:
        self._function = function
        self._calls = calls
        self._identity_struct = identity_struct

    @property
    def argtypes(self):
        return self._function.argtypes

    @argtypes.setter
    def argtypes(self, value) -> None:
        self._function.argtypes = value

    @property
    def restype(self):
        return self._function.restype

    @restype.setter
    def restype(self, value) -> None:
        self._function.restype = value

    def __call__(self, *args):
        result = self._function(*args)
        call: dict[str, object] = {
            "pid": int(args[0]),
            "flavor": int(args[1]),
            "arg": int(args[2]),
            "buffer_size": int(args[4]),
            "return_count": int(result),
            "errno": ctypes.get_errno(),
        }
        if int(result) == int(args[4]):
            info = ctypes.cast(
                args[3], ctypes.POINTER(self._identity_struct)).contents
            call["identity"] = {
                "pid": int(info.bsd.pid),
                "uid": int(info.bsd.uid),
                "start_sec": int(info.bsd.start_sec),
                "start_usec": int(info.bsd.start_usec),
                "pid_version": int(info.unique.idversion),
                "zombie": info.bsd.status == 5,
            }
        self._calls.append(call)
        return result


class _LoggedLibproc:
    def __init__(self, library, calls: list[dict[str, object]], identity_struct) -> None:
        self._library = library
        self._calls = calls
        self._identity_struct = identity_struct
        self._proc_pidinfo = None

    def __getattr__(self, name: str):
        function = getattr(self._library, name)
        if name != "proc_pidinfo":
            return function
        if self._proc_pidinfo is None:
            self._proc_pidinfo = _LoggedProcPidInfo(
                function, self._calls, self._identity_struct)
        return self._proc_pidinfo


def _show(stage: str, calls: list[dict[str, object]], callback) -> None:
    first_call = len(calls)
    try:
        verdict = repr(callback())
    except Exception as error:
        verdict = f"{type(error).__name__}: {error}"
    print(json.dumps({
        "stage": stage,
        "verdict": verdict,
        "proc_pidinfo": calls[first_call:],
    }, sort_keys=True))


def main() -> int:
    if sys.platform != "darwin":
        print("This diagnostic requires macOS (darwin).", file=sys.stderr)
        return 2

    adapter = _load_adapter()
    calls: list[dict[str, object]] = []
    original_cdll = ctypes.CDLL

    def logged_cdll(name, *args, **kwargs):
        library = original_cdll(name, *args, **kwargs)
        if name == "/usr/lib/libproc.dylib":
            return _LoggedLibproc(library, calls, adapter._BSDInfoWithUnique)
        return library

    ctypes.CDLL = logged_cdll
    guard = adapter.OrphanGuard(Path.cwd(), platform="darwin")
    scope = guard.scope("macos-probe")
    process: subprocess.Popen[bytes] | None = None
    try:
        launch = scope.launch(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        process = launch.process
        print(json.dumps({
            "stage": "registration",
            "pid": launch.pid,
            "captured_identity": repr(launch.identity),
            "handle_type": type(launch.bound_handle).__name__,
            "handle_error": repr(getattr(launch.bound_handle, "_error", None)),
            "proc_pidinfo": calls,
        }, sort_keys=True))
        _show("is_current", calls, launch.bound_handle.is_current)
        _show("is_alive", calls, launch.bound_handle.is_alive)
        scope.end()
        _show("scope_end_check", calls, guard.check)
        print(json.dumps({
            "stage": "final",
            "captured_identity": repr(launch.identity),
            "outcomes": [repr(outcome) for outcome in guard.check().outcomes],
            "unit_outcomes": [repr(outcome) for outcome in guard.check().unit_outcomes],
            "proc_pidinfo": calls,
        }, sort_keys=True))
        return 0
    finally:
        try:
            guard.close()
        finally:
            ctypes.CDLL = original_cdll
            if process is not None and process.poll() is None:
                process.wait(timeout=35.0)


if __name__ == "__main__":
    raise SystemExit(main())
