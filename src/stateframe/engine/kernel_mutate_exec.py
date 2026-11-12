"""Durable, one-shot mutation launches on a dedicated execution session."""
from __future__ import annotations

from collections.abc import Mapping
import base64
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import ssl
import stat
import subprocess
import sys
import sysconfig
import time
from typing import Callable, Literal
import uuid

if __package__:
    from .kernel_exec_capability import ExecCapability, ExecLimits, symlink_hops
    from .kernel_exec_session import CleanupReport, ExecSession, ExecSessionError, rpc_timeout_seconds
    from .kernel_exec_tracker import ProcessTable, process_cmdline_digest
    from .kernel_mutate_debug import _hook_timing
    from .kernel_mutate_build import (MAX_MUTATION_MEMORY_MB, MAX_MUTATION_WALL_S,
                                      MAX_MUTATION_CPU_S)
    from .kernel_mutate_sweep import _boot_id
    from .sf_mutate_boot import strict_loads
else:
    from kernel_exec_capability import ExecCapability, ExecLimits, symlink_hops
    from kernel_exec_session import CleanupReport, ExecSession, ExecSessionError, rpc_timeout_seconds
    from kernel_exec_tracker import ProcessTable, process_cmdline_digest
    from kernel_mutate_debug import _hook_timing
    from kernel_mutate_build import (MAX_MUTATION_MEMORY_MB, MAX_MUTATION_WALL_S,
                                     MAX_MUTATION_CPU_S)
    from kernel_mutate_sweep import _boot_id
    from sf_mutate_boot import strict_loads


_ENGINE = Path(__file__).resolve().parent
_ENV = {"LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/nonexistent", "TZ": "UTC"}
_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
_HOOK_READY = b"SFMR1\n"


def _hook_arm_deadline(trigger_deadline: float, *, launch_started: float,
                       ready_at: float) -> float:
    """Keep the staged hook budget, excluding isolated launch and readiness latency."""
    return trigger_deadline + max(0.0, ready_at - launch_started)


def _fixed_env_capability(name: str, unit_root: Path, read_roots: tuple[Path, ...],
                          network: Literal["none", "full"], limits: ExecLimits) -> ExecCapability:
    """Give every mutation child the same fixed environment."""
    return ExecCapability(name, unit_root, False, read_roots, (), network, _ENV, limits)


@dataclass(frozen=True)
class MutationLimits:
    wall_s: float = 10.0
    cpu_s: float = 5.0
    mem_mb: int = MAX_MUTATION_MEMORY_MB
    output_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        for name, maximum in (("wall_s", MAX_MUTATION_WALL_S), ("cpu_s", MAX_MUTATION_CPU_S)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= maximum or not math.isfinite(value):
                raise ValueError(f"mutation {name} limit must be between 0 and {maximum} seconds")
        if type(self.mem_mb) is not int or not 0 < self.mem_mb <= MAX_MUTATION_MEMORY_MB:
            raise ValueError(f"mutation memory limit must be between 1 and {MAX_MUTATION_MEMORY_MB} MiB")


@dataclass(frozen=True)
class LaunchSpec:
    role: Literal["payload", "validator", "smoke", "broker"]
    call: str
    launch: str
    unit_dir: Path | None
    inputs: Mapping[str, Path]
    responses: Mapping[str, Path]
    stdin_doc: bytes
    limits: MutationLimits
    budget_after_ready: bool = False
    trigger_deadline: float | None = None


@dataclass(frozen=True)
class LaunchOutcome:
    exit_code: int | None
    timed_out: bool
    stdout: bytes
    stdout_overflow: bool
    wall_s: float
    cpu_s: float | None
    cleanup: CleanupReport
    identity: dict | None


@dataclass(frozen=True)
class RuntimeClosure:
    roots: tuple[Path, ...]
    excluded: tuple[tuple[str, tuple[str, ...]], ...]
    hidden: tuple[Path, ...] = ()  # inside a root, masked in the Linux sandbox


class MutationPlatformError(RuntimeError):
    code = "MUTATION_PLATFORM_UNQUALIFIED"


class MutationContainmentError(BaseException):
    code = "MUTATION_CONTAINMENT"


class MutationStartupTimeout(Exception):
    code = "MUTATION_STARTUP_TIMEOUT"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_regular(source: Path, target: Path) -> None:
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("mutation snapshot source must be a private regular file")
        with os.fdopen(fd, "rb", closefd=False) as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        target.chmod(0o400)
    finally:
        os.close(fd)


def _verify_unit_bytes(directory: Path, expected: str) -> None:
    """Bind every copied launch image to the staged source digest."""
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("mutation unit has no source digest identity")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("mutation unit directory is not private")
    if __package__:
        from .kernel_mutate_build import source_digest
    else:
        from kernel_mutate_build import source_digest

    members: dict[str, bytes] = {}
    for path in directory.iterdir():
        if path.name != "unit.json" and re.fullmatch(r"[a-z0-9_]{1,40}\.py", path.name) is None:
            raise ValueError("mutation unit has an unexpected member")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 64 << 10:
                raise ValueError("mutation unit member is not a bounded private file")
            data = bytearray()
            while len(data) <= 64 << 10:
                block = os.read(fd, min(1 << 16, (64 << 10) + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
            if len(data) != info.st_size or os.fstat(fd).st_size != info.st_size:
                raise ValueError("mutation unit changed during verification")
            members[path.name] = bytes(data)
        finally:
            os.close(fd)
    manifest_bytes = members.pop("unit.json", None)
    if manifest_bytes is None:
        raise ValueError("mutation unit manifest is missing")
    manifest = strict_loads(manifest_bytes)
    if not isinstance(manifest, dict) or source_digest(members, manifest) != "sha256:" + expected:
        raise ValueError("mutation unit bytes differ from admitted source")


def _remove_tree_at(parent_fd: int, name: str, expected: tuple[int, int]) -> None:
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != expected:
            raise MutationContainmentError("mutation stage identity changed")
        if not info.st_mode & stat.S_IWUSR:
            os.fchmod(fd, info.st_mode | stat.S_IWUSR | stat.S_IXUSR)
        for entry in os.scandir(fd):
            child = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(child.st_mode):
                _remove_tree_at(fd, entry.name, (child.st_dev, child.st_ino))
            elif stat.S_ISREG(child.st_mode) and child.st_nlink == 1:
                os.unlink(entry.name, dir_fd=fd)
            else:
                raise MutationContainmentError("unexpected mutation stage entry")
        os.fsync(fd)
    finally:
        os.close(fd)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != expected:
        raise MutationContainmentError("mutation stage replaced during cleanup")
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _remove_tree(path: Path) -> None:
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        _remove_tree_at(parent, path.name, (info.st_dev, info.st_ino))
    finally:
        os.close(parent)


def _runtime_roots() -> tuple[Path, ...]:
    """Interpreter, stdlib and dynamic dependencies; site-packages is masked at launch."""
    return _runtime_closure().roots


def _runtime_closure() -> RuntimeClosure:
    """Cached read roots and unavailable optional extensions."""
    python = Path(sys.executable).resolve(strict=True)
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    dynamic = stdlib / "lib-dynload"
    if not dynamic.is_dir():
        raise MutationPlatformError("Python dynamic extension directory unavailable")
    return _runtime_roots_cached(str(python), str(stdlib),
                                 python.stat().st_mtime_ns, dynamic.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _runtime_roots_cached(python_name: str, stdlib_name: str, _python_mtime: int,
                          _dynamic_mtime: int) -> RuntimeClosure:
    python = Path(python_name)
    stdlib = Path(stdlib_name)
    dynamic = stdlib / "lib-dynload"
    binaries = [python, *sorted(dynamic.glob("*.so"))]
    if not sys.platform.startswith("linux"):
        return RuntimeClosure(tuple(sorted({python, stdlib})), ())
    try:
        output = subprocess.run(["/usr/bin/ldd", *map(str, binaries)],
                                capture_output=True, text=True, timeout=60, check=True).stdout
    except OSError as error:
        raise MutationPlatformError("cannot qualify Python runtime libraries: ldd is unavailable "
                                    f"({type(error).__name__} at /usr/bin/ldd)") from error
    except subprocess.CalledProcessError as error:
        raise MutationPlatformError("cannot qualify Python runtime libraries: ldd failed "
                                    f"(exit {error.returncode})") from error
    except subprocess.SubprocessError as error:
        raise MutationPlatformError("cannot qualify Python runtime libraries: ldd did not finish") from error
    sections: dict[str, list[str]] = {str(python): []} if len(binaries) == 1 else {}
    current = str(python) if len(binaries) == 1 else None
    expected = {str(binary) for binary in binaries}
    for line in output.splitlines():
        if line and not line[0].isspace():
            name = line[:-1] if line.endswith(":") else ""
            if name not in expected or (name in sections and len(binaries) > 1):
                raise MutationPlatformError("cannot qualify Python runtime libraries")
            sections[name] = []
            current = name
        elif line.strip():
            if current is None:
                raise MutationPlatformError("cannot qualify Python runtime libraries")
            sections[current].append(line)
    if set(sections) != expected or any(not lines for lines in sections.values()):
        raise MutationPlatformError("cannot qualify Python runtime libraries")
    excluded = []
    libraries: set[Path] = set()
    for binary in binaries:
        lines = sections[str(binary)]
        missing_names = set()
        for line in lines:
            if "not found" in line:
                match = re.match(r"\s*(\S+)\s+(?:=>\s+)?not found(?:\s|$)", line)
                if match is None:
                    raise MutationPlatformError("cannot qualify Python runtime libraries")
                missing_names.add(match.group(1))
        missing = sorted(missing_names)
        if missing:
            if binary == python:
                raise MutationPlatformError("runtime library dependency missing: "
                                            f"{python.name} needs {', '.join(missing)}")
            excluded.append((binary.name, tuple(missing)))
            continue
        paths = {Path(match) for match in re.findall(r"(/[\w.+/\-]+)\s*\(", "\n".join(lines))}
        try:
            libraries.update(paths | {path.resolve(strict=True) for path in paths})
        except (OSError, RuntimeError) as error:
            raise MutationPlatformError("cannot qualify Python runtime libraries") from error
    # One stdlib bind keeps the launch cheap; site-packages and unloadable extensions are masked
    # inside it (kernel_test_isolation._mutation_command), so they are absent from the sandbox.
    candidates = [stdlib / "site-packages", stdlib / "dist-packages",
                  *(dynamic / name for name, _ in excluded)]
    hidden: set[Path] = set()
    for path in candidates:
        if not (path.exists() or path.is_symlink()):
            continue
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if stdlib in target.parents:
            hidden.add(target)
    return RuntimeClosure(tuple(sorted({python, stdlib, *libraries})), tuple(excluded),
                          tuple(sorted(hidden)))


def _broker_config_roots() -> tuple[Path, ...]:
    """Only resolver and CA material needed by the trusted network broker."""
    roots: set[Path] = set()
    for name in ("/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/hosts"):
        path = Path(name)
        if path.exists():
            roots.update(symlink_hops(path))
    verify_paths = ssl.get_default_verify_paths()
    ca_files = ("/etc/ssl/cert.pem", verify_paths.cafile, verify_paths.openssl_cafile)
    ca_dirs = ("/etc/ssl/certs", verify_paths.capath, verify_paths.openssl_capath)
    paths = [Path(name).absolute() for name in ca_files if name]
    paths.extend(Path(name).absolute() for name in ca_dirs if name)
    for path in sorted(set(paths)):
        if path.exists():
            roots.update(symlink_hops(path))
            if path.is_dir():
                # OpenSSL follows hash links in CApath after opening the directory.
                # A read-only bind of CApath does not expose targets outside it.
                bound_directory = path.resolve(strict=True)
                for entry in path.iterdir():
                    if entry.is_symlink() and entry.is_file():
                        roots.update(hop for hop in symlink_hops(entry)
                                     if not hop.is_relative_to(path) and
                                     not hop.is_relative_to(bound_directory))
    return tuple(sorted(roots))


class MutationExecutor:
    def __init__(self, runtime_root: Path, *, network_envelope: Literal["none", "full"],
                 launch_stage_hook: Callable[[str], None] | None = None):
        if network_envelope not in {"none", "full"}:
            raise ValueError("invalid mutation network envelope")
        requested_root = Path(runtime_root).absolute()
        if requested_root.is_symlink():
            raise ValueError("mutation runtime root cannot be a symlink")
        self.root = requested_root.resolve(strict=True)
        if not self.root.is_dir() or self.root.stat().st_uid != os.getuid():
            raise ValueError("mutation runtime root must be a directory")
        if any(self.root.iterdir()):
            raise ValueError("mutation runtime root must be unused")
        self.root.chmod(0o700)
        self.network_envelope = network_envelope
        self._boot = self.root / "boot"
        self._stage = self.root / "stage"
        self._session_dir = self.root / "session"
        self._owner = self.root / "owner.json"
        self._closed = False
        self._seen: set[str] = set()
        self._launch_stage_hook = launch_stage_hook
        self._session = None
        self._ledger = None
        self._hidden: tuple[Path, ...] = ()
        try:
            _hook_timing("qualification begin")
            closure = _runtime_closure()
            self._roots = closure.roots
            self._hidden = closure.hidden if sys.platform.startswith("linux") else ()
            self._broker_roots = _broker_config_roots() if network_envelope == "full" else ()
            for path in (self._boot, self._stage, self._session_dir):
                path.mkdir(mode=0o700)
            for name in ("sf_mutate_boot.py", "sf_mutate_sdk.py", "sf_launch.py"):
                _copy_regular(_ENGINE / name, self._boot / name)
            if network_envelope == "full":
                fetch = _ENGINE / "sf_mutate_fetch.py"
                if not fetch.is_file():
                    raise MutationPlatformError("broker boot is unavailable")
                _copy_regular(fetch, self._boot / fetch.name)
                web_fetch = _ENGINE / "kernel_web_fetch.py"
                if web_fetch.is_file():
                    _copy_regular(web_fetch, self._boot / web_fetch.name)
            self._session = ExecSession(self._session_dir)
            envelope = _fixed_env_capability(
                "mutation", self.root, (*self._roots, *self._broker_roots, self.root),
                network_envelope, ExecLimits())
            self._session.configure(envelope)
            ledger_fd = os.open(self.root / "launches.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._ledger = os.fdopen(ledger_fd, "ab", buffering=0)
            self._session._start(envelope)
            # Popen is not a readiness signal. Await the supervisor's first reply
            # before a hook is allowed to start its one-second budget.
            self._session._rpc({"op": "all"})
            self._write_owner()
            _hook_timing("qualification end")
        except BaseException as error:
            try:
                if self._ledger is not None:
                    self._ledger.close()
                if self._session is not None:
                    self._session.close()
                _remove_tree(self.root)
            except BaseException as cleanup_error:
                raise MutationContainmentError(
                    f"mutation constructor cleanup failed: {cleanup_error}") from error
            raise

    def _record(self, spec: LaunchSpec, state: str, identity: dict | None = None) -> None:
        row = {"schema": "stateframe-mutation-launch/1", "launch": spec.launch,
               "call": spec.call, "role": spec.role, "state": state,
               "t": time.monotonic_ns(), "identity": identity}
        self._ledger.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
        os.fsync(self._ledger.fileno())
        if self._launch_stage_hook is not None:
            self._launch_stage_hook(state)

    def _write_owner(self) -> None:
        if self._owner.exists():
            return
        process = self._session._process
        row = ProcessTable().identity(process.pid)
        if row is None:
            raise MutationContainmentError("supervisor identity unavailable")
        data = {"pid": row.pid, "start": list(row.start),
                "boot_id": _boot_id(), "run_uuid": self.root.name}
        if sys.platform.startswith("linux"):
            table = ProcessTable()
            try:
                namespace = table.pid_namespace(row.pid)
                command_digest = process_cmdline_digest(row.pid)
            except (OSError, RuntimeError, ValueError) as error:
                raise MutationContainmentError("supervisor launch identity facts unavailable") from error
            if not namespace:
                raise MutationContainmentError("supervisor PID namespace identity unavailable")
            data["ns"] = namespace
            data["cmdline_sha256"] = command_digest
        fd = os.open(self._owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(self.root)

    def _stage_spec(self, spec: LaunchSpec) -> tuple[Path, dict]:
        stage = self._stage / spec.launch
        stage.mkdir(mode=0o700)
        for name in ("ctl", "in", "resp", "unit"):
            (stage / name).mkdir(mode=0o700)
        if spec.unit_dir is not None:
            for source in Path(spec.unit_dir).iterdir():
                if not source.is_file() or source.is_symlink():
                    raise ValueError("unit bundle contains nonregular file")
                _copy_regular(source, stage / "unit" / source.name)
        for key, source in spec.inputs.items():
            if not _NAME.fullmatch(key):
                raise ValueError("invalid input id")
            _copy_regular(Path(source), stage / "in" / key)
        for key, source in spec.responses.items():
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError("invalid response digest")
            _copy_regular(Path(source), stage / "resp" / key)
        for name in ("in", "resp", "unit", "ctl"):
            _fsync_dir(stage / name)
        _fsync_dir(stage)
        return stage, {name: str(stage / name) for name in ("ctl", "in", "resp", "unit")}

    def run(self, spec: LaunchSpec) -> LaunchOutcome:
        if self._closed:
            raise MutationContainmentError("mutation executor is closed")
        if spec.role not in {"payload", "validator", "smoke", "broker"}:
            raise ValueError("invalid launch role")
        if spec.budget_after_ready and (spec.role != "payload" or spec.limits.wall_s > 1.0
                                        or spec.limits.cpu_s > 1.0):
            raise ValueError("ready-based budget is reserved for one-second payload hooks")
        if spec.trigger_deadline is not None and (
                not spec.budget_after_ready or type(spec.trigger_deadline) not in (int, float)
                or not math.isfinite(spec.trigger_deadline)):
            raise ValueError("trigger deadline is reserved for payload-ready hooks")
        if not re.fullmatch(r"[0-9a-f]{32}", spec.launch) or spec.launch in self._seen:
            raise ValueError("launch must be a unique uuid4 hex value")
        if not _NAME.fullmatch(spec.call) or len(spec.stdin_doc) > 1 << 20:
            raise ValueError("mutation call or stdin document invalid")
        if spec.role == "broker" and self.network_envelope != "full":
            raise MutationPlatformError("broker exceeds network envelope")
        doc = strict_loads(spec.stdin_doc)
        if not isinstance(doc, dict):
            raise ValueError("mutation stdin document must be an object")
        if "hook_ready" in doc or "hook_cpu_s" in doc or "startup_wait_s" in doc:
            raise ValueError("mutation startup fields are host-reserved")
        if doc.get("launch") != spec.launch or doc.get("call") != spec.call or doc.get("role") != spec.role:
            raise ValueError("mutation stdin identity differs from LaunchSpec")
        if spec.role != "broker":
            grants = doc.get("grants")
            if (not isinstance(grants, dict) or set(grants) != {"network", "writes"}
                    or any(type(value) is not bool for value in grants.values())):
                raise ValueError("mutation launch grants are invalid")
            if grants["network"] and self.network_envelope != "full":
                raise MutationPlatformError("mutation network exceeds owner envelope")
        if spec.unit_dir is not None:
            _verify_unit_bytes(Path(spec.unit_dir), Path(spec.unit_dir).name)
        stage = self._stage / spec.launch
        self._seen.add(spec.launch)
        try:
            stage, mounts = self._stage_spec(spec)
            if spec.unit_dir is not None:
                _verify_unit_bytes(stage / "unit", Path(spec.unit_dir).name)
        except BaseException as error:
            try:
                if stage.exists() or stage.is_symlink():
                    _remove_tree(stage)
            except BaseException as cleanup_error:
                raise MutationContainmentError(
                    f"mutation stage cleanup could not be verified: {cleanup_error}") from error
            raise
        identity = None
        try:
            mounts["boot"] = str(self._boot)
            nonce = uuid.uuid4().hex
            startup_window = (10.0 * rpc_timeout_seconds() / 15.0
                              if not spec.budget_after_ready else None)
            if spec.role == "broker":
                if set(doc) != {"call", "launch", "role", "request"} or not isinstance(doc["request"], dict):
                    raise ValueError("broker launch needs an isolated request document")
                stdin = json.dumps(doc["request"], separators=(",", ":")).encode()
            else:
                doc["nonce"] = nonce
                if spec.budget_after_ready:
                    doc["hook_ready"] = True
                    doc["hook_cpu_s"] = spec.limits.cpu_s
                else:
                    doc["startup_wait_s"] = startup_window + 2.0
                stdin = json.dumps(doc, separators=(",", ":")).encode()
            if sys.platform == "darwin":
                for row in doc.get("inputs", []):
                    key = row.get("id")
                    if key not in spec.inputs or row.get("path") != "/in/" + key:
                        raise ValueError("mutation input path differs from staged snapshot")
                    row["path"] = str(stage / "in" / key)
                for row in doc.get("responses", []):
                    body = row.get("body")
                    if body is not None:
                        key = Path(body).name
                        if key not in spec.responses or body != "/resp/" + key:
                            raise ValueError("mutation response path differs from staged snapshot")
                        row["body"] = str(stage / "resp" / key)
            if len(stdin) > 1 << 20:
                raise ValueError("mutation stdin document exceeds 1 MiB")
            cap = _fixed_env_capability(
                "mutation_broker" if spec.role == "broker" else "mutation",
                stage / "unit",
                (*self._roots, *(self._broker_roots if spec.role == "broker" else ()),
                 *(Path(path) for path in mounts.values())),
                "full" if spec.role == "broker" else "none",
                ExecLimits(10.0 if spec.budget_after_ready else spec.limits.wall_s,
                           10.0 if spec.budget_after_ready else spec.limits.cpu_s,
                           spec.limits.mem_mb << 20, 1 << 20, 64, 8,
                           spec.limits.output_bytes + (len(_HOOK_READY) if spec.budget_after_ready else 0),
                           True))
            boot_script = (self._boot / "sf_mutate_boot.py" if sys.platform == "darwin"
                           else Path("/sf/boot/sf_mutate_boot.py"))
            command = [str(Path(sys.executable).resolve()), "-I", "-S", "-B", str(boot_script)]
            if spec.role == "broker":
                command += ["--broker", spec.launch, nonce, str(startup_window + 2.0)]
            self._record(spec, "PREPARED")
            if self._hidden:
                mounts = {**mounts, "hide": [str(path) for path in self._hidden]}
            launch_started = time.monotonic()
            startup_deadline = (launch_started + startup_window
                                if not spec.budget_after_ready else None)
            def startup_remaining() -> float:
                remaining = startup_deadline - time.monotonic()
                if remaining <= 0:
                    raise MutationStartupTimeout("mutation startup exceeded bound")
                return remaining

            if spec.budget_after_ready:
                _hook_timing("hook launch")
                handle = self._session._spawn(command, cap, stdin=stdin, mounts=mounts,
                                              armable_wall_s=spec.limits.wall_s)
            else:
                control = self._session._control
                original_timeout = control.gettimeout()
                try:
                    control.settimeout(startup_remaining())
                    handle = self._session._spawn(command, cap, stdin=stdin, mounts=mounts,
                                                  armable_wall_s=spec.limits.wall_s)
                    startup_remaining()
                except ExecSessionError as error:
                    if time.monotonic() >= startup_deadline:
                        raise MutationStartupTimeout("mutation startup exceeded bound") from error
                    raise
                finally:
                    control.settimeout(original_timeout)
            identity = handle.identity
            self._write_owner()
            self._record(spec, "STARTED", identity)
            if identity is None:
                raise MutationContainmentError("mutation launch identity missing")
            self._record(spec, "REGISTERED", identity)
            ready_at = self._await_hook_ready(handle) if spec.budget_after_ready else None
            if not spec.budget_after_ready:
                startup_remaining()
            # An ordinary arm reply can be slow while still inside the startup window.
            # Its gate record must remain valid through release and the payload budget.
            release = {"launch": spec.launch, "nonce": nonce,
                       "deadline": (time.monotonic() if spec.budget_after_ready else startup_deadline)
                       + spec.limits.wall_s + 2,
                       "identity": identity}
            target = stage / "ctl" / "release.json"
            temporary = stage / "ctl" / (".release-" + nonce)
            with temporary.open("x") as stream:
                json.dump(release, stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            expired = False
            if spec.budget_after_ready:
                arm_request = {"op": "arm", "id": handle.id, "ready_at": ready_at}
                if spec.trigger_deadline is not None:
                    arm_request["deadline"] = _hook_arm_deadline(
                        spec.trigger_deadline, launch_started=launch_started,
                        ready_at=ready_at)
                armed = self._session._rpc(arm_request)
                _hook_timing("budget clock start", at=armed["started"])
                expired = armed.get("expired", False)
            else:
                control = self._session._control
                original_timeout = control.gettimeout()
                try:
                    control.settimeout(startup_remaining())
                    armed = self._session._rpc({"op": "arm", "id": handle.id})
                except ExecSessionError as error:
                    if time.monotonic() >= startup_deadline:
                        raise MutationStartupTimeout("mutation startup exceeded bound") from error
                    raise
                finally:
                    control.settimeout(original_timeout)
                expired = armed.get("expired", False)
            if expired:
                temporary.unlink()
            else:
                if not spec.budget_after_ready:
                    startup_remaining()
                os.replace(temporary, target)
                _fsync_dir(stage / "ctl")
                self._record(spec, "RELEASED", identity)
            if spec.budget_after_ready:
                result = self._session._collect(
                    handle, cap,
                    on_first_byte=lambda: _hook_timing("payload first byte"),
                    on_timeout=lambda at: _hook_timing("timeout", at=at))
            else:
                result = self._session._collect(handle, cap)
            self._record(spec, "EXITED", identity)
            if not result.cleanup.ok:
                raise MutationContainmentError(result.cleanup.detail)
            current = ProcessTable().identity(identity["pid"])
            if current is not None and not current.zombie and list(current.start) == identity["start"]:
                raise MutationContainmentError("registered process survived cleanup")
            if spec.unit_dir is not None:
                _verify_unit_bytes(stage / "unit", Path(spec.unit_dir).name)
            _remove_tree(stage)
            self._record(spec, "REAPED", identity)
            return LaunchOutcome(result.exit_code, result.timed_out or expired, result.output,
                                 result.output_truncated or result.output_total_bytes > cap.limits.output_bytes,
                                 result.wall_seconds, None,
                                 result.cleanup, identity)
        except BaseException as error:
            try:
                report = self._session.stop_all(0)
                if not report.ok:
                    raise MutationContainmentError(report.detail)
                if stage.exists() or stage.is_symlink():
                    _remove_tree(stage)
                if isinstance(error, MutationStartupTimeout):
                    if identity is not None:
                        self._record(spec, "REAPED", identity)
                    self.close()
                elif identity is not None:
                    self._record(spec, "REAPED", identity)
                if isinstance(error, MutationContainmentError):
                    raise error
            except BaseException as cleanup_error:
                try:
                    self._record(spec, "CLEANUP_FAILED", identity)
                except BaseException as record_error:
                    raise MutationContainmentError(
                        f"mutation cleanup evidence could not be recorded: {record_error}") from error
                if isinstance(cleanup_error, MutationContainmentError):
                    raise
                raise MutationContainmentError(
                    f"mutation launch cleanup could not be verified: {cleanup_error}") from error
            raise

    def _await_hook_ready(self, handle) -> float:
        deadline = time.monotonic() + 9.0
        received = bytearray()
        while time.monotonic() < deadline:
            status = handle._status(True)
            if status.get("lost_bytes"):
                raise MutationContainmentError("mutation hook readiness bytes lost")
            received.extend(base64.b64decode(status["output"]))
            if received:
                if bytes(received) != _HOOK_READY[:len(received)]:
                    raise MutationContainmentError("mutation hook readiness protocol invalid")
                if len(received) == len(_HOOK_READY):
                    if status["finished"]:
                        raise MutationContainmentError("mutation hook exited before release")
                    ready_at = time.monotonic()
                    _hook_timing("ready signal", at=ready_at)
                    return ready_at
            if status["finished"]:
                raise MutationContainmentError("mutation hook exited before readiness")
            time.sleep(0.01)
        _hook_timing("timeout")
        raise MutationContainmentError("mutation hook startup exceeded bound")

    def close(self) -> CleanupReport:
        if self._closed:
            return CleanupReport(True, 0, (), "")
        # Stage bytes can still be in use until the supervisor and its children are gone.
        # Retain artifacts on unverified shutdown, but never retain the owned ledger fd.
        stop_failure: BaseException | None = None
        try:
            if not hasattr(self, "_close_report"):
                if not getattr(self._session, "_close_control_done", False):
                    try:
                        report = self._session.stop_all(0)
                    except BaseException as error:
                        stop_failure = error
                        report = None
                    if report is not None and report.ok:
                        self._close_report = report
                    elif report is not None:
                        stop_failure = MutationContainmentError(report.detail)
            self._session.close()
            if stop_failure is not None:
                raise stop_failure
            if not hasattr(self, "_close_report"):
                # ExecSession.close verified the launch unit even though the
                # control channel was closed by a prior cleanup attempt.
                self._close_report = CleanupReport(True, 0, (), "supervisor exit verified")
        except BaseException as error:
            if self._ledger is not None and not self._ledger.closed:
                try:
                    self._ledger.close()
                except BaseException as ledger_error:
                    raise MutationContainmentError(
                        f"mutation ledger close failed: {ledger_error}") from error
            if isinstance(error, MutationContainmentError):
                raise
            raise MutationContainmentError(
                f"mutation executor cleanup failed: {error}") from error

        errors: list[BaseException] = []

        def attempt(action) -> None:
            try:
                action()
            except BaseException as error:
                errors.append(error)

        for path in (self._session_dir, self._boot):
            def remove_if_present(path: Path = path) -> None:
                if path.exists() or path.is_symlink():
                    _remove_tree(path)

            attempt(remove_if_present)

        def sweep_stage() -> None:
            if not self._stage.is_dir():
                return
            for child in self._stage.iterdir():
                if not re.fullmatch(r"[0-9a-f]{32}", child.name):
                    raise MutationContainmentError("unexpected mutation stage at close")
                attempt(lambda child=child: _remove_tree(child))
            if not any(self._stage.iterdir()):
                _remove_tree(self._stage)

        attempt(sweep_stage)
        if not self._ledger.closed:
            attempt(self._ledger.close)
        if errors:
            error = errors[0]
            if isinstance(error, MutationContainmentError):
                raise error
            raise MutationContainmentError(f"mutation executor cleanup failed: {error}") from error
        self._closed = True
        return self._close_report

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def recover_orphaned_stages(runtime_root: Path) -> None:
    """Remove abandoned stage snapshots only after all recorded owners are gone.

    The full runtime root and ledger remain for the mutation sweeper to reconcile.
    """
    root = Path(runtime_root).resolve(strict=True)
    table = ProcessTable()
    owner = json.loads((root / "owner.json").read_text("utf-8"))
    try:
        boot_id = _boot_id()
    except (OSError, ValueError) as error:
        raise MutationContainmentError("cannot establish host boot identity") from error
    if owner.get("boot_id") != boot_id:
        raise MutationContainmentError("runtime owner belongs to a different boot")
    identities = [owner]
    with (root / "launches.jsonl").open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("identity") is not None:
                identities.append(row["identity"])
    for identity in identities:
        if identity.get("boot_id") != boot_id or type(identity.get("pid")) is not int:
            raise MutationContainmentError("invalid recorded process identity")
        current = table.identity(identity["pid"])
        if current is not None and not current.zombie and list(current.start) == identity.get("start"):
            raise MutationContainmentError("recorded mutation process is still alive")
    stage = root / "stage"
    if stage.is_dir():
        for child in stage.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", child.name):
                raise MutationContainmentError("unexpected orphaned stage entry")
            _remove_tree(child)
        _remove_tree(stage)
