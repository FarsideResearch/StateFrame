"""One-shot mutation boot gate and framed unit runner; standard library only."""
from __future__ import annotations

import ast
import ctypes
import errno
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback


class _StderrRing(io.TextIOBase):
    def __init__(self):
        self.data = bytearray()

    def writable(self):
        return True

    def write(self, value):
        data = str(value).encode("utf-8", "replace")
        self.data.extend(data)
        del self.data[:max(0, len(self.data) - (32 << 10))]
        return len(value)

    def getvalue(self):
        return self.data.decode("utf-8", "replace")


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate stdin key")
        value[key] = item
    return value


def _read_doc():
    data = os.read(0, (1 << 20) + 1)
    if len(data) > 1 << 20:
        raise ValueError("stdin document exceeds 1 MiB")
    return strict_loads(data)


def strict_loads(data: bytes):
    """Reject duplicate, nonfinite, oversized and deeply nested stdin values."""
    def bounded_int(value):
        if len(value.lstrip("-")) > 40:
            raise ValueError("JSON integer exceeds 40 digits")
        return int(value)

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON float")
        return result

    try:
        value = json.loads(data.decode("utf-8", "strict"), object_pairs_hook=_unique_pairs,
                           parse_int=bounded_int, parse_float=finite_float,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except RecursionError as error:
        raise ValueError("stdin JSON nesting exceeds limit") from error
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 16:
            raise ValueError("stdin JSON nesting exceeds 16")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
            pending.extend((key, depth + 1) for key in item)
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            item.encode("utf-8", "strict")
    return value


def wait_for_release(ctl: Path, launch: str, nonce: str, *, timeout: float = 30.0) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = (Path(ctl) / "release.json").read_bytes()
            if len(raw) > 4096:
                return None
            record = json.loads(raw, object_pairs_hook=_unique_pairs)
            if (record.get("launch") != launch or record.get("nonce") != nonce
                    or type(record.get("deadline")) not in {int, float}
                    or time.monotonic() > record["deadline"]
                    or not isinstance(record.get("identity"), dict)
                    or not record["identity"].get("start")):
                return None
            return record
        except FileNotFoundError:
            time.sleep(0.01)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    return None


def _startup_wait_seconds(value: object) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid mutation startup wait") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid mutation startup wait")
    return timeout


def macos_profile(read_paths: tuple[str, ...], *, network_full: bool = False) -> str:
    """Single final deny-default Seatbelt profile, assembled before unit import."""
    paths = " ".join(f"(subpath {json.dumps(path)})" for path in read_paths)
    ancestors = {str(parent) for path in read_paths for parent in Path(path).parents}
    metadata = " ".join(f"(literal {json.dumps(path)})" for path in sorted(ancestors))
    network_rule = "(allow network*)" if network_full else "(deny network*)"
    return ("(version 1) (deny default) "
            f"(allow file-read* (require-any {paths} {metadata})) "
            '(allow file-write* (literal "/dev/null")) '
            f"(deny process-fork) (deny process-exec) (deny mach-lookup) {network_rule}")


def activate_macos_sandbox(read_paths: tuple[str, ...], *, network_full: bool = False) -> None:
    profile = macos_profile(read_paths, network_full=network_full).encode("utf-8")
    library = None
    for name in ("/usr/lib/libsandbox.1.dylib", "/usr/lib/libSystem.B.dylib"):
        try:
            library = ctypes.CDLL(name, use_errno=True)
            break
        except OSError:
            continue
    if library is None or not hasattr(library, "sandbox_init_with_parameters"):
        raise RuntimeError("macOS sandbox_init_with_parameters unavailable")
    method = library.sandbox_init_with_parameters
    method.argtypes = [ctypes.c_char_p, ctypes.c_uint64,
                       ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    method.restype = ctypes.c_int
    error = ctypes.c_char_p()
    result = method(profile, 0, None, ctypes.byref(error))
    if result != 0:
        if error and hasattr(library, "sandbox_free_error"):
            library.sandbox_free_error(error)
        raise RuntimeError("macOS sandbox activation failed")


def _write_frame(body: dict) -> None:
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    frame = b"SFMW1 " + str(len(data)).encode() + b"\n" + data
    while frame:
        frame = frame[os.write(1, frame):]


def _error_frame(call: str, category: str, error: BaseException, ring: _StderrRing) -> None:
    _write_frame({"wire": 1, "call": call, "kind": "error",
                  "error": {"type": category, "message": str(error)[:2048],
                            "trace": traceback.format_exc()[-8192:]},
                  "stderr": ring.getvalue()})


def _unit_files(unit: Path) -> tuple[dict, list[Path]]:
    manifest = json.loads((unit / "unit.json").read_text("utf-8"), object_pairs_hook=_unique_pairs)
    entry = manifest.get("entry")
    if not isinstance(entry, str) or re.fullmatch(r"[a-z0-9_]{1,40}\.py", entry) is None:
        raise ValueError("invalid unit entry")
    files = sorted(unit.glob("*.py"))
    if unit / entry not in files or len(files) > 8:
        raise ValueError("unit files violate bundle contract")
    import sys as _sys
    reserved = set(_sys.stdlib_module_names) | {"sf_mutate_sdk", "sf_mutate_boot", "sf_launch"}
    if any(path.stem in reserved for path in files):
        raise ValueError("unit module shadows trusted module")
    return manifest, files


def _validate(files: list[Path]) -> dict:
    errors = []
    for path in files:
        try:
            source = path.read_text("utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec", dont_inherit=True)
        except (SyntaxError, UnicodeError) as error:
            errors.append({"file": path.name, "line": getattr(error, "lineno", 0) or 0,
                           "col": getattr(error, "offset", 0) or 0, "msg": str(error)[:2048]})
    return {"ok": not errors, "errors": errors, "hints": []}


def _close_payload_fds() -> None:
    """Drop loader and import descriptors before untrusted handler code runs."""
    try:
        names = os.listdir("/proc/self/fd")
    except OSError as error:
        raise RuntimeError("cannot enumerate mutation payload descriptors") from error
    for name in names:
        if name.isdecimal() and int(name) > 2:
            try:
                os.close(int(name))
            except OSError as error:
                if error.errno != errno.EBADF:  # enumeration can close its own fd
                    raise


def main() -> int:
    null = os.open("/dev/null", os.O_WRONLY)
    os.dup2(null, 2)
    os.close(null)
    ring = _StderrRing()
    sys.stderr = ring
    call = ""
    activated = False
    try:
        if len(sys.argv) in {4, 5} and sys.argv[1] == "--broker":
            launch, nonce = sys.argv[2:4]
            ctl = (Path("/sf/ctl") if sys.platform != "darwin"
                   else Path(__file__).resolve().parent.parent / "stage" / launch / "ctl")
            timeout = (_startup_wait_seconds(sys.argv[4]) if len(sys.argv) == 5 else 30.0)
            if wait_for_release(ctl, launch, nonce, timeout=timeout) is None:
                return 4
            import runpy
            boot = Path(__file__).resolve().parent
            if sys.platform == "darwin":
                import sysconfig
                stdlib = sysconfig.get_path("stdlib")
                activate_macos_sandbox((stdlib, str(Path(stdlib) / "lib-dynload"),
                                        str(boot), str(ctl), "/etc/resolv.conf", "/etc/ssl/certs"),
                                       network_full=True)
            sys.path.insert(0, str(boot))
            script = boot / "sf_mutate_fetch.py"
            sys.argv = [str(script)]
            runpy.run_path(str(script), run_name="__main__")
            return 0
        doc = _read_doc()
        if not isinstance(doc, dict) or doc.get("wire") != 1:
            raise ValueError("invalid stdin document")
        call = doc["call"]
        launch, nonce = doc["launch"], doc["nonce"]
        stage = Path("/sf/ctl") if sys.platform != "darwin" else Path(__file__).resolve().parent.parent / "stage" / launch / "ctl"
        hook_ready = doc.get("hook_ready") is True and doc.get("role") == "payload"
        timeout = (_startup_wait_seconds(doc["startup_wait_s"])
                   if not hook_ready and "startup_wait_s" in doc else 30.0)
        if not hook_ready and wait_for_release(stage, launch, nonce, timeout=timeout) is None:
            return 4
        boot_path = Path("/sf/boot") if sys.platform != "darwin" else Path(__file__).resolve().parent
        unit_path = Path("/sf/unit") if sys.platform != "darwin" else stage.parent / "unit"
        if sys.platform.startswith("linux"):
            if len(os.listdir("/proc/self/task")) != 1:
                raise RuntimeError("mutation boot requires one native thread")
            sys.path.insert(0, str(boot_path))
            from sf_launch import install_syscall_filter
            install_syscall_filter(network_none=True, post_init=True, deny_keyring=True)
        elif sys.platform == "darwin":
            import sysconfig
            stdlib = sysconfig.get_path("stdlib")
            activate_macos_sandbox((stdlib, str(Path(stdlib) / "lib-dynload"),
                                    str(boot_path), str(unit_path), str(stage.parent / "in"),
                                    str(stage.parent / "resp"), str(stage)))
        else:
            raise RuntimeError("mutation platform unsupported")
        activated = True
        stdlib_paths = [path for path in sys.path if path and path != str(boot_path)
                        and ("python" in path or path.startswith("/System/Library"))]
        sys.path = [*stdlib_paths, str(boot_path)]
        from sf_mutate_sdk import MutationContext, NeedMore
        manifest, files = _unit_files(unit_path)
        sys.path.append(str(unit_path))
        if sys.platform.startswith("linux"):
            _close_payload_fds()
        if hook_ready:
            cpu_s = doc.get("hook_cpu_s")
            if type(cpu_s) not in {int, float} or not 0 < cpu_s <= 1.0:
                raise ValueError("invalid hook CPU budget")
            os.write(1, b"SFMR1\n")
            if wait_for_release(stage, launch, nonce) is None:
                return 4
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_limit = math.ceil(usage.ru_utime + usage.ru_stime + cpu_s)
            _, hard_limit = resource.getrlimit(resource.RLIMIT_CPU)
            cpu_limit = min(cpu_limit, hard_limit)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        if doc.get("role") == "validator":
            result = _validate(files)
        else:
            import importlib
            module = importlib.import_module(manifest["entry"][:-3])
            if sys.platform.startswith("linux"):
                _close_payload_fds()
            handler = getattr(module, "handle")
            ctx = MutationContext(doc)
            try:
                result = handler(doc["input"], ctx)
            except NeedMore:
                if not ctx.pending_fetches:
                    raise ValueError("need without new fetch")
                _write_frame({"wire": 1, "call": call, "kind": "need", "fetch": ctx.pending_fetches})
                return 0
            if not isinstance(result, dict):
                raise ValueError("handler result must be an object")
        frame = {"wire": 1, "call": call, "kind": "result", "result": result}
        if doc.get("role") != "validator" and ctx.edits:
            frame["edits"] = ctx.edits
        _write_frame(frame)
        return 0
    except Exception as error:
        _error_frame(call, "HandlerError" if activated else "BootError", error, ring)
        return 3 if activated else 4


if __name__ == "__main__":
    raise SystemExit(main())
