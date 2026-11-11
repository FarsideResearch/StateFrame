"""Host-bound mutation qualification using actual isolated boot launches."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import tempfile
import uuid
import stat

if __package__:
    from .kernel_mutate_build import admit, stage_unit
    from .kernel_safe_paths import safe_home
    from .kernel_mutate_exec import (LaunchSpec, MutationExecutor, MutationLimits,
                                     MutationPlatformError, _ENGINE, _remove_tree, _runtime_closure,
                                     _runtime_roots)
    from .sf_mutate_boot import macos_profile
else:
    from kernel_mutate_build import admit, stage_unit
    from kernel_safe_paths import safe_home
    from kernel_mutate_exec import (LaunchSpec, MutationExecutor, MutationLimits,
                                    MutationPlatformError, _ENGINE, _remove_tree, _runtime_closure,
                                    _runtime_roots)
    from sf_mutate_boot import macos_profile


@dataclass(frozen=True)
class Qualification:
    ok: bool
    platform: str
    reasons: list[str]
    record: dict


_QUALIFICATION_WALL_SECONDS = 60.0
_QUALIFICATION_TIMEOUT_RETRIES = 1
_UNQUALIFIED_TIMEOUT = "MUTATION_PLATFORM_UNQUALIFIED_TIMEOUT"
_UNQUALIFIED_FAILURE = "MUTATION_PLATFORM_UNQUALIFIED_FAILURE"


def _run_qualification_check(executor, make_spec: Callable[[], object], *, name: str,
                             accepts: Callable[[object], bool],
                             detail: Callable[[object], str]) -> dict:
    """Retry once only when the isolated launch explicitly reports a timeout."""
    attempts = 0
    while True:
        outcome = executor.run(make_spec())
        attempts += 1
        timed_out = outcome.timed_out and not outcome.stdout_overflow
        cleanup_ok = getattr(getattr(outcome, "cleanup", None), "ok", False)
        retryable_timeout = timed_out and cleanup_ok
        if retryable_timeout and attempts <= _QUALIFICATION_TIMEOUT_RETRIES:
            continue
        passed = (not outcome.timed_out and not outcome.stdout_overflow
                  and cleanup_ok and accepts(outcome))
        if passed:
            return {"name": name, "ok": True, "detail": detail(outcome), "attempts": attempts}
        return {"name": name, "ok": False,
                "reason": _UNQUALIFIED_TIMEOUT if retryable_timeout else _UNQUALIFIED_FAILURE,
                "detail": detail(outcome), "attempts": attempts}


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding() -> dict:
    roots = _runtime_roots()
    runtime = _runtime_closure()
    runtime_excluded = [[name, list(missing)] for name, missing in runtime.excluded]
    hidden = set(runtime.hidden)
    closure = []
    for root in roots:
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            if path in hidden or any(parent in hidden for parent in path.parents):
                continue
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                closure.append((str(path), info.st_size, _digest_file(path)))
            elif stat.S_ISLNK(info.st_mode):
                closure.append((str(path), "link", os.readlink(path)))
            elif not stat.S_ISDIR(info.st_mode):
                raise MutationPlatformError("unsupported runtime closure entry")
    policy = {"macos": macos_profile(("/sf/boot", "/sf/unit", "/in", "/resp")),
              "isolation": _digest_file(_ENGINE / "kernel_test_isolation.py"),
              "launcher": _digest_file(_ENGINE / "sf_launch.py")}
    return {"os": platform.system(), "kernel": platform.release(),
            "arch": platform.machine(), "python": sys.version,
            "runtime_excluded": runtime_excluded,
            "closure_digest": hashlib.sha256(json.dumps(closure, separators=(",", ":")).encode()).hexdigest(),
            "boot_digest": _digest_file(_ENGINE / "sf_mutate_boot.py"),
            "sdk_digest": _digest_file(_ENGINE / "sf_mutate_sdk.py"),
            "profile_digest": hashlib.sha256(json.dumps(
                policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _unit(root: Path, name: str, body: str) -> Path:
    compile(body, name + "/main.py", "exec")
    manifest = {"schema": "stateframe-mutation-unit/1", "name": name,
                "revision": 1, "parent": None, "entry": "main.py",
                "summary": "Platform qualification probe", "kind": "reader",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "requires": {"network": False, "writes": False},
                "smoke": [{"input": {}, "expect": ({"ok": True} if name == "positive"
                                                   else {"denied": True})}],
                "limits": {"wall_s": _QUALIFICATION_WALL_SECONDS, "cpu_s": 5, "mem_mb": 256}}
    return stage_unit({"main.py": body.encode("utf-8")}, manifest, root).dir


def _spec(unit: Path, role: str = "payload") -> LaunchSpec:
    launch = uuid.uuid4().hex
    doc = {"wire": 1, "call": "selftest", "launch": launch, "phase": 0, "role": role,
           "input": {}, "inputs": [], "responses": [],
           "grants": {"network": False, "writes": False}, "max_fetch": 4}
    return LaunchSpec(role, "selftest", launch, unit, {}, {},
                      json.dumps(doc, separators=(",", ":")).encode(),
                      MutationLimits(wall_s=_QUALIFICATION_WALL_SECONDS, cpu_s=5, mem_mb=256,
                                     output_bytes=8192))


def _probe_body(operation: str, *, exception: str = "Exception") -> str:
    return ("def handle(input, ctx):\n"
            "    try:\n"
            + "\n".join("        " + line for line in operation.splitlines()) + "\n"
            f"    except {exception} as error:\n"
            "        return {'denied': True, 'detail': type(error).__name__}\n"
            "    return {'denied': False}\n")


def _excluded_extension_operation(name: str, host_path: Path) -> str:
    return (f"import importlib\ntry:\n    importlib.import_module({name.split('.')[0]!r})\n"
            f"except ImportError:\n    if open({str(host_path)!r}, 'rb').read(1) == b'':\n"
            "        raise OSError('masked')\n"
            "except BaseException:\n    pass")


def _negative_operations(supervisor_pid: int, outside_lib: str, platform_name: str) -> dict[str, str]:
    python = str(Path(sys.executable).resolve())
    engine = str(_ENGINE / "kernel_exec_supervisor.py")
    home = safe_home()
    if home is None:
        raise MutationPlatformError("cannot determine home directory for mutation qualification")
    home_ssh = str(home / ".ssh")
    operations = {
        "fork": "import os\nos.fork()",
        "posix_spawn": f"import os\nos.posix_spawn({python!r}, [{python!r}], {{}})",
        "subprocess": f"import subprocess\nsubprocess.run([{python!r}, '-c', 'pass'])",
        "af_inet": "import socket\nsocket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        "af_inet6": "import socket\nsocket.socket(socket.AF_INET6, socket.SOCK_STREAM)",
        "af_unix": "import socket\nsocket.socket(socket.AF_UNIX, socket.SOCK_STREAM)",
        "abstract_unix": "import socket\ns=socket.socket(socket.AF_UNIX)\ns.bind('\\0sf_mutate_probe')",
        "home_ssh": f"import os\nos.listdir({home_ssh!r})",
        "engine_source": f"open({engine!r}, 'rb').read(1)",
        "etc_shadow": "open('/etc/shadow', 'rb').read(1)",
        "write_tmp": "open('/tmp/sf_mutate_probe', 'wb').write(b'x')",
        "write_dev_shm": "open('/dev/shm/sf_mutate_probe', 'wb').write(b'x')",
        "write_unit": ("open('/sf/unit/sf_mutate_probe', 'wb').write(b'x')" if platform_name == "linux"
                       else "from pathlib import Path\nopen(Path(__file__).parent/'sf_mutate_probe','wb').write(b'x')"),
        "write_in": ("open('/in/sf_mutate_probe', 'wb').write(b'x')" if platform_name == "linux"
                     else "from pathlib import Path\nopen(Path(__file__).parent.parent/'in'/'sf_mutate_probe','wb').write(b'x')"),
        "write_stdin": "import os\nos.write(0,b'x')",
        "dlopen_outside": f"import ctypes\nctypes.CDLL({outside_lib!r})",
        "signal_supervisor": f"import os,signal\nos.kill({supervisor_pid},signal.SIGTERM)",
    }
    if platform_name == "linux":
        operations.update({
            "clone3": "import ctypes, errno\nlib=ctypes.CDLL(None,use_errno=True)\n"
                      "result=lib.syscall(435,0,0)\n"
                      "if result == -1 and ctypes.get_errno() == errno.EPERM: raise OSError(errno.EPERM,'denied')",
            "execveat": "import ctypes, errno\nlib=ctypes.CDLL(None,use_errno=True)\n"
                        "result=lib.syscall(322,-1,0,0,0,0)\n"
                        "if result == -1 and ctypes.get_errno() == errno.EPERM: raise OSError(errno.EPERM,'denied')",
            "thread": "import threading\nt=threading.Thread(target=lambda: None)\nt.start()\nt.join()",
            "write_proc_comm": "open('/proc/self/comm','w').write('changed')",
        })
    return operations


def _outside_library(roots: tuple[Path, ...]) -> str:
    for directory in (Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib/aarch64-linux-gnu"),
                      Path("/usr/lib")):
        if not directory.is_dir():
            continue
        for path in (*directory.glob("*.so*"), *directory.glob("*.dylib")):
            if (path.is_file() and not path.is_symlink()
                    and not any(path == root or root in path.parents for root in roots)):
                return str(path)
    raise MutationPlatformError("no outside library available for dlopen qualification")


def selftest(home: Path) -> dict:
    """Run a positive boot, then deny probes as separate physical launches."""
    platform_name = "macos" if platform.system().lower() == "darwin" else platform.system().lower()
    checks = []
    runtime_excluded = []
    if platform_name not in {"linux", "macos"}:
        return {"ok": False, "platform": platform_name,
                "runtime_excluded": runtime_excluded,
                "checks": [{"name": "platform", "ok": False, "detail": "unsupported platform"}]}
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="mutate-selftest-", dir=home))
    closed_clean = False
    try:
        runtime_excluded = [[name, list(missing)] for name, missing in _runtime_closure().excluded]
        detail = "; ".join(f"{name} (missing {', '.join(missing)})"
                           for name, missing in runtime_excluded) or "none"
        checks.append({"name": "runtime_exclusions", "ok": True, "detail": detail})
        units = root / "units"
        units.mkdir()
        runtime = root / "runtime"
        runtime.mkdir()
        # The catalog includes a broker-capable unit; payload probes still run
        # with network denied by their per-launch capability.
        with MutationExecutor(runtime, network_envelope="full") as executor:
            positive = _unit(units, "positive", "def handle(input, ctx): return {'ok': True}\n")
            positive_check = _run_qualification_check(
                executor, lambda: _spec(positive), name="positive",
                accepts=lambda result: (result.exit_code == 0 and result.cleanup.ok
                                        and b'"ok":true' in result.stdout),
                detail=lambda result: f"exit={result.exit_code} cleanup={result.cleanup.ok}",
            )
            checks.append(positive_check)
            if positive_check["ok"]:
                outside = _outside_library(executor._roots)
                supervisor_pid = executor._session._process.pid
                operations = _negative_operations(supervisor_pid, outside, platform_name)
                if platform_name == "linux" and runtime_excluded:
                    name = runtime_excluded[0][0]
                    host_path = Path(sysconfig.get_path("stdlib")).resolve(strict=True) / "lib-dynload" / name
                    operations["excluded_extension"] = _excluded_extension_operation(name, host_path)
                for name, operation in operations.items():
                    body = _probe_body(operation, exception="BaseException")
                    unit = _unit(units, name, body)
                    try:
                        check = _run_qualification_check(
                            executor, lambda unit=unit: _spec(unit), name=name,
                            accepts=lambda result: (result.exit_code == 0 and result.cleanup.ok
                                                    and b'"denied":true' in result.stdout),
                            detail=lambda result: (
                                "exit={} denied={} cleanup={}".format(
                                    result.exit_code, b'"denied":true' in result.stdout,
                                    result.cleanup.ok)
                            ),
                        )
                    except BaseException as error:
                        check = {"name": name, "ok": False, "reason": _UNQUALIFIED_FAILURE,
                                 "detail": f"{type(error).__name__}: {error}"}
                    checks.append(check)
                prebuilt = _ENGINE.parent / "mutations_prebuilt"
                for unit_dir in sorted(path for path in prebuilt.iterdir() if path.is_dir()):
                    try:
                        manifest = json.loads((unit_dir / "unit.json").read_text("utf-8"))
                        files = {path.name: path.read_bytes() for path in unit_dir.glob("*.py")}
                        staged = stage_unit(files, manifest, runtime, fixture_root=unit_dir)
                        admit(staged, executor, runtime_closure_digest="selftest",
                              sdk_digest="selftest", policy_version="selftest",
                              grants_digest="selftest",
                              timeout_retries=_QUALIFICATION_TIMEOUT_RETRIES,
                              wall_timeout_s=_QUALIFICATION_WALL_SECONDS)
                        checks.append({"name": "prebuilt:" + unit_dir.name,
                                       "ok": True, "detail": staged.source_digest})
                    except BaseException as error:
                        reason = (_UNQUALIFIED_TIMEOUT if getattr(error, "code", None) == "TIMEOUT"
                                  else _UNQUALIFIED_FAILURE)
                        checks.append({"name": "prebuilt:" + unit_dir.name,
                                       "ok": False, "reason": reason,
                                       "detail": f"{type(error).__name__}: {error}"})
        closed_clean = True
        return {"ok": all(check["ok"] for check in checks),
                "platform": platform_name, "runtime_excluded": runtime_excluded, "checks": checks}
    except BaseException as error:
        checks.append({"name": "setup", "ok": False,
                       "reason": _UNQUALIFIED_FAILURE,
                       "detail": f"{type(error).__name__}: {error}"})
        return {"ok": False, "platform": platform_name,
                "runtime_excluded": runtime_excluded, "checks": checks}
    finally:
        if closed_clean:
            _remove_tree(root)


def qualify(home: Path, *, refresh: bool = False) -> Qualification:
    """Cache only a selftest result bound to this exact runtime and host build."""
    home = Path(home)
    binding = _binding()
    directory = home / "mutations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "qualified.json"
    if path.is_file() and not refresh:
        try:
            record = json.loads(path.read_text("utf-8"))
            if record.get("binding") == binding and isinstance(record.get("report"), dict):
                report = record["report"]
                return Qualification(report.get("ok") is True, report.get("platform", ""),
                                     [check["name"] for check in report.get("checks", []) if not check["ok"]],
                                     record)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    report = selftest(home)
    record = {"schema": "stateframe-mutation-qualified/1", "binding": binding, "report": report}
    temporary = directory / (".qualified-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(record, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return Qualification(report["ok"], report["platform"],
                         [check["name"] for check in report["checks"] if not check["ok"]], record)
