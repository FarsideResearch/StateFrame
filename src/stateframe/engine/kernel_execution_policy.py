"""One fail-closed execution policy for model-influenced child processes.

This module is the only v46 entry point for new model-controlled subprocesses.
It deliberately separates host capability discovery from child execution so a
missing sandbox, unsupported platform, invalid limit, or output flood cannot
fall back to a raw host process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import platform
import re
import resource
import selectors
import signal
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Callable

from kernel_test_isolation import isolated_command, sandbox_executable
from sf_launch import _command_cgroup


MAX_WALL_SECONDS = 900.0
MAX_INPUT_BYTES = 4 * 1024 * 1024
DEFAULT_CPU_SECONDS = 30.0
DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_NOFILE = 128
DEFAULT_NPROC = 64
MAX_OUTPUT_BYTES = 256 * 1024
MAX_PROFILE_OUTPUT_BYTES = 16 * 1024
_PROFILE_FD_ENV = "STATEFRAME_TEST_RUN_SPANS_FD"

_ALLOWED_ENV = frozenset({
    "PATH", "LANG", "LC_ALL", "PYTHONHASHSEED", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
})


class ExecutionPolicyError(RuntimeError):
    """A child was refused because the required execution boundary is absent."""


@dataclass(frozen=True)
class CapabilityReport:
    platform: str
    backend: str | None
    available: bool
    network_isolated: bool
    path_contained: bool
    cpu_limited: bool
    memory_limited: bool
    output_bounded: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-execution-capabilities/1",
            "platform": self.platform,
            "backend": self.backend,
            "available": self.available,
            "network_isolated": self.network_isolated,
            "path_contained": self.path_contained,
            "cpu_limited": self.cpu_limited,
            "memory_limited": self.memory_limited,
            "output_bounded": self.output_bounded,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyResult:
    completed: subprocess.CompletedProcess[str]
    timed_out: bool
    process_group_killed: bool
    output_truncated: bool
    capabilities: CapabilityReport
    pid: int | None = None
    execution_trace: Mapping[str, int] = field(default_factory=dict)
    profile_payload: bytes | None = None


def capability_report() -> CapabilityReport:
    """Return the host capability matrix without starting a child process."""
    system = platform.system()
    backend_name = "sandbox-exec" if system == "Darwin" else "bwrap" if system == "Linux" else None
    executable = sandbox_executable(backend_name) if backend_name else None
    required_limits = (
        "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_NOFILE",
    )
    limits_available = all(hasattr(resource, name) for name in required_limits)
    memory_available = hasattr(resource, "RLIMIT_AS") or hasattr(resource, "RLIMIT_DATA")
    network_isolated = system in {"Darwin", "Linux"}
    available = bool(
        executable and network_isolated and limits_available and memory_available
    )
    reasons: list[str] = []
    if system not in {"Darwin", "Linux"}:
        reasons.append("unsupported platform")
    if executable is None:
        reasons.append(f"missing {backend_name or 'sandbox backend'}")
    if not limits_available:
        reasons.append("required POSIX resource limits unavailable")
    if not memory_available:
        reasons.append("memory resource limit unavailable")
    if system == "Darwin" and memory_available:
        reasons.append("Darwin address-space limit is best effort")
    return CapabilityReport(
        platform=system,
        backend=executable,
        available=available,
        network_isolated=network_isolated and executable is not None,
        path_contained=available,
        cpu_limited=limits_available,
        memory_limited=memory_available,
        output_bounded=True,
        reason="; ".join(reasons) if reasons else None,
    )


def available() -> bool:
    return capability_report().available


def require_capabilities() -> CapabilityReport:
    report = capability_report()
    if not report.available:
        raise ExecutionPolicyError(
            "model-influenced execution refused: sandbox capability unavailable "
            f"({report.reason or 'unknown reason'})"
        )
    return report


def _contained_root(root: str | os.PathLike[str]) -> Path:
    candidate = Path(root).resolve()
    if not candidate.is_dir() or candidate == Path(candidate.anchor):
        raise ExecutionPolicyError("execution root must be an existing non-root directory")
    return candidate


def _arguments(command: Sequence[object]) -> list[str]:
    if isinstance(command, (str, bytes, bytearray)):
        raise ExecutionPolicyError("execution command must be an argv sequence, not shell text")
    values = [str(item) for item in command]
    if not values or any("\x00" in value for value in values):
        raise ExecutionPolicyError("execution command is empty or contains NUL")
    if len(values) > 128 or sum(len(value) for value in values) > 64 * 1024:
        raise ExecutionPolicyError("execution command exceeds the policy bound")
    return values


# The HOST (never the model) may declare task-environment keys a project's tests need,
# e.g. a stage6 AHA worker sets STATEFRAME_TASK_ENV_KEYS=DATABASE_URL,TEST_DATABASE_URL,...
# 10-03: without this every AHA check died at settings import (pytest exit 4).
TASK_ENV_KEYS_VAR = "STATEFRAME_TASK_ENV_KEYS"
_TASK_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _parse_host_task_env_keys(raw: str) -> frozenset[str]:
    keys = {item.strip() for item in raw.split(",") if item.strip()}
    return frozenset(key for key in keys if _TASK_ENV_KEY_RE.match(key))


# These values are host policy, not model input. Capture them before any
# request/turn code can mutate os.environ and use the same snapshot for every
# child in this process.
_HOST_TASK_ENV_KEYS = _parse_host_task_env_keys(os.environ.get(TASK_ENV_KEYS_VAR, ""))
_HOST_TASK_ENV_VALUES = {
    key: os.environ[key] for key in _HOST_TASK_ENV_KEYS if key in os.environ
}


def _host_task_env_keys() -> frozenset[str]:
    return _HOST_TASK_ENV_KEYS


def _safe_environment(env: Mapping[object, object] | None) -> dict[str, str]:
    source = os.environ if env is None else env
    allowed = _ALLOWED_ENV | _host_task_env_keys()
    result = {str(key): str(value) for key, value in source.items() if str(key) in allowed}
    # Declared task keys come from the host process env and are AUTHORITATIVE: they
    # override caller-supplied values. 10-03: run_tests pre-fills DATABASE_URL with a
    # localhost default via setdefault, which silently beat the container's real value
    # and every AHA check still died at import (exit 4). The model cannot set host env.
    result.update(_HOST_TASK_ENV_VALUES)
    return result


def _limit_value(name: str, requested: float | int, *, maximum: float | int) -> float | int:
    try:
        value = float(requested)
    except (TypeError, ValueError) as error:
        raise ExecutionPolicyError(f"{name} must be finite and positive") from error
    if not math.isfinite(value) or value <= 0 or value > maximum:
        raise ExecutionPolicyError(f"{name} is outside the execution policy bound")
    return value


def _lowered_limits(current: tuple[int, int], limit: int) -> tuple[int, int]:
    """Return (soft, hard) no higher than `limit` and never soft > hard.

    RLIM_INFINITY is -1 on Linux, so a naive min(limit, soft) picks -1 ("unlimited")
    and setrlimit rejects soft=unlimited over a finite hard limit.  10-03: this killed
    every run_tests call in the Linux worker container (macOS reports finite values).
    """
    soft, hard = current
    infinity = resource.RLIM_INFINITY
    bound = limit if hard == infinity else min(limit, hard)
    new_soft = bound if soft == infinity else min(bound, soft)
    return new_soft, bound


def _resource_preexec(cpu_seconds: float, memory_bytes: int,
                      file_bytes: int, nofile: int, nproc: int,
                      cgroup: Path | None = None) -> Callable[[], None]:
    cpu = int(math.ceil(_limit_value("cpu_seconds", cpu_seconds, maximum=MAX_WALL_SECONDS)))
    memory = int(_limit_value("memory_bytes", memory_bytes, maximum=4 * 1024 * 1024 * 1024))
    file_limit = int(_limit_value("file_bytes", file_bytes, maximum=64 * 1024 * 1024))
    nofile_limit = int(_limit_value("nofile", nofile, maximum=1024))
    _limit_value("nproc", nproc, maximum=256)

    def _lower_only(which: int, value: int) -> None:
        # A non-root process may only LOWER a hard limit.  Containers often start
        # with hard limits below our policy values; raising them makes setrlimit
        # fail inside preexec_fn, which killed every L82 run at its first check.
        _soft, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(which, (value, value))

    def apply() -> None:
        if cgroup is not None:
            try:
                (cgroup / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
            except OSError:
                pass  # The in-namespace RLIMIT_NPROC is mandatory; cgroup is optional.
        _lower_only(resource.RLIMIT_CPU, cpu)
        _lower_only(resource.RLIMIT_FSIZE, file_limit)
        _lower_only(resource.RLIMIT_CORE, 0)
        resource.setrlimit(resource.RLIMIT_NOFILE,
                           _lowered_limits(resource.getrlimit(resource.RLIMIT_NOFILE), nofile_limit))
        if hasattr(resource, "RLIMIT_AS"):
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            except (OSError, ValueError):
                if platform.system() != "Darwin":
                    raise
        elif hasattr(resource, "RLIMIT_DATA"):
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
        else:  # Capability discovery should have rejected this before spawn.
            raise RuntimeError("memory resource limit unavailable")

    return apply


def _kill_group(process: subprocess.Popen[bytes]) -> bool:
    killed = False
    try:
        os.killpg(process.pid, signal.SIGKILL)
        killed = True
    except (OSError, ProcessLookupError):
        pass
    try:
        process.kill()
        killed = True
    except (OSError, ProcessLookupError):
        pass
    return killed


def _prepared(command: Sequence[object], root: Path, env: Mapping[object, object] | None,
              internal_env: Mapping[str, object] | None,
              readonly_workspace: bool, nproc: int,
              generic_check: bool = False,
              isolation_builder: Callable[..., tuple[list[str], dict[str, str]]] = isolated_command,
              ) -> tuple[list[str], dict[str, str], CapabilityReport]:
    report = require_capabilities()
    arguments = _arguments(command)
    isolation_options = {"readonly_workspace": readonly_workspace}
    if generic_check:
        isolation_options["generic_check"] = True
    if platform.system() == "Linux":
        isolation_options["nproc_budget"] = int(_limit_value("nproc", nproc, maximum=256))
    isolated, clean = isolation_builder(arguments, root, _safe_environment(env),
                                        **isolation_options)
    for key, value in (internal_env or {}).items():
        if not str(key).startswith("STATEFRAME_"):
            raise ExecutionPolicyError("internal execution variables must be STATEFRAME_*")
        clean[str(key)] = str(value)
    return isolated, clean, report


def sandboxed_popen(command: Sequence[object], root: str | os.PathLike[str], *,
                    env: Mapping[object, object] | None = None,
                    internal_env: Mapping[str, object] | None = None,
                    cpu_seconds: float = DEFAULT_CPU_SECONDS,
                    memory_bytes: int = DEFAULT_MEMORY_BYTES,
                    file_bytes: int = DEFAULT_FILE_BYTES,
                    nofile: int = DEFAULT_NOFILE,
                    nproc: int = DEFAULT_NPROC,
                    stdin: object = subprocess.PIPE,
                    stdout: object = subprocess.PIPE,
                    stderr: object = subprocess.PIPE,
                    readonly_workspace: bool = False,
                    generic_check: bool = False,
                    pass_fds: Sequence[int] = (),
                    cgroup: Path | None = None,
                    isolation_builder: Callable[..., tuple[list[str], dict[str, str]]] = isolated_command,
                    popen_factory: Callable[..., subprocess.Popen] | None = None) -> tuple[subprocess.Popen[bytes], CapabilityReport]:
    """Start an argv child behind the required sandbox and resource policy."""
    contained = _contained_root(root)
    command_line, clean, report = _prepared(
        command, contained, env, internal_env, readonly_workspace, nproc,
        generic_check=generic_check,
        isolation_builder=isolation_builder)
    factory = popen_factory or subprocess.Popen
    process = factory(
        command_line,
        cwd=str(contained),
        env=clean,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        start_new_session=True,
        pass_fds=tuple(pass_fds),
        preexec_fn=_resource_preexec(cpu_seconds, memory_bytes, file_bytes, nofile, nproc, cgroup)
        if os.name != "nt" else None,
    )
    return process, report


def _trace_mark(trace: dict[str, int] | None, name: str) -> None:
    if trace is not None and name not in trace:
        trace[name] = time.monotonic_ns()


def _collect(process: subprocess.Popen[bytes], input_data: bytes | None, deadline: float,
             output_limit: int, *, observer_fd: int | None = None,
             observer_buffer: bytearray | None = None,
             trace: dict[str, int] | None = None) -> tuple[str, str, bool, bool, bool]:
    if input_data is not None and len(input_data) > MAX_INPUT_BYTES:
        raise ExecutionPolicyError("child input exceeds the policy bound")
    selector = selectors.DefaultSelector()
    streams: dict[int, str] = {}
    handles: list[object] = []
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    truncated = False
    input_offset = 0
    group_killed = False
    if process.stdin is not None:
        handles.append(process.stdin)
        os.set_blocking(process.stdin.fileno(), False)
        if input_data:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        handles.append(stream)
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
        streams[stream.fileno()] = name
    if observer_fd is not None and observer_buffer is not None:
        try:
            os.set_blocking(observer_fd, False)
            selector.register(observer_fd, selectors.EVENT_READ, "observer")
        except (OSError, ValueError, KeyError):  # telemetry channel failure is fail-open
            observer_fd = None
            observer_buffer = None
    _trace_mark(trace, "pipe_drain_start_ns")
    timed_out = False
    while selector.get_map():
        if trace is not None and process.poll() is not None:
            _trace_mark(trace, "child_exit_observed_ns")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = process.poll() is None
            if timed_out:
                group_killed = _kill_group(process)
            break
        for key, mask in selector.select(min(remaining, 0.05)):
            stream = key.fileobj
            kind = key.data
            if kind == "stdin":
                broken_pipe = False
                try:
                    written = os.write(stream.fileno(), input_data[input_offset:input_offset + 8192])
                    input_offset += written
                except BlockingIOError:
                    written = 0
                except OSError:
                    broken_pipe = True
                if broken_pipe or input_data is None or input_offset >= len(input_data):
                    selector.unregister(stream)
                    stream.close()
                continue
            try:
                stream_fd = stream if isinstance(stream, int) else stream.fileno()
                data = os.read(stream_fd, 8192)
            except (BlockingIOError, OSError):
                data = b""
            if not data:
                if kind in {"stdout", "stderr"}:
                    _trace_mark(trace, f"{kind}_eof_ns")
                selector.unregister(stream)
                continue
            if kind == "observer":
                if observer_buffer is not None:
                    available = max(0, MAX_PROFILE_OUTPUT_BYTES - len(observer_buffer))
                    if available:
                        observer_buffer.extend(data[:available])
                    if len(data) > available and trace is not None:
                        trace["profile_report_truncated"] = 1
                continue
            available = max(0, output_limit - total)
            if available:
                buffers[kind].extend(data[:available])
            if len(data) > available:
                truncated = True
            total += len(data)
        if trace is not None and process.poll() is not None:
            _trace_mark(trace, "child_exit_observed_ns")
            # The main runner owns the profile writer. Drain all currently available
            # bytes, then stop observing this private channel so a test-spawned child
            # cannot extend the host wait by inheriting the descriptor.
            if observer_fd is not None and observer_buffer is not None:
                try:
                    observer_key = selector.get_key(observer_fd)
                except KeyError:
                    observer_key = None
                if observer_key is not None:
                    while True:
                        try:
                            data = os.read(observer_fd, 8192)
                        except (BlockingIOError, OSError):
                            data = b""
                        if not data:
                            break
                        available = max(0, MAX_PROFILE_OUTPUT_BYTES - len(observer_buffer))
                        if available:
                            observer_buffer.extend(data[:available])
                        if len(data) > available and trace is not None:
                            trace["profile_report_truncated"] = 1
                    try:
                        selector.unregister(observer_fd)
                    except (KeyError, OSError):
                        pass
    for key in list(selector.get_map().values()):
        try:
            selector.unregister(key.fileobj)
        except (KeyError, OSError):
            pass
    for handle in handles:
        try:
            handle.close()
        except OSError:
            pass
    selector.close()
    _trace_mark(trace, "pipe_drain_finished_ns")
    if timed_out:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            group_killed = _kill_group(process) or group_killed
    else:
        process.wait(timeout=max(1.0, deadline - time.monotonic() + 1.0))
    if trace is not None:
        _trace_mark(trace, "child_exit_observed_ns")
    return (
        bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        timed_out,
        truncated,
        group_killed,
    )


def run_sandboxed(command: Sequence[object], root: str | os.PathLike[str], *,
                  env: Mapping[object, object] | None = None,
                  internal_env: Mapping[str, object] | None = None,
                  input_data: str | bytes | None = None,
                  timeout: float = MAX_WALL_SECONDS,
                  output_limit: int = MAX_OUTPUT_BYTES,
                  readonly_workspace: bool = False,
                  generic_check: bool = False,
                  cpu_seconds: float = DEFAULT_CPU_SECONDS,
                  memory_bytes: int = DEFAULT_MEMORY_BYTES,
                  file_bytes: int = DEFAULT_FILE_BYTES,
                  nofile: int = DEFAULT_NOFILE,
                  nproc: int = DEFAULT_NPROC,
                  observe_pytest: bool = False) -> PolicyResult:
    """Run one bounded child; every failure is a refusal or typed result."""
    timeout = float(_limit_value("timeout", timeout, maximum=MAX_WALL_SECONDS))
    output_limit = int(_limit_value("output_limit", output_limit, maximum=4 * 1024 * 1024))
    nproc_budget = (int(_limit_value("nproc", nproc, maximum=256))
                    if platform.system() == "Linux" else None)
    payload = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
    trace: dict[str, int] | None = {} if observe_pytest else None
    profile_reader: int | None = None
    profile_writer: int | None = None
    profile_buffer: bytearray | None = None
    child_internal_env = dict(internal_env or {})
    pass_fds: tuple[int, ...] = ()
    if observe_pytest and _PROFILE_FD_ENV not in child_internal_env:
        try:
            profile_reader, profile_writer = os.pipe()
            os.set_inheritable(profile_writer, True)
            child_internal_env[_PROFILE_FD_ENV] = str(profile_writer)
            pass_fds = (profile_writer,)
            profile_buffer = bytearray()
        except OSError:
            for fd in (profile_reader, profile_writer):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            profile_reader = profile_writer = None
            profile_buffer = None
            pass_fds = ()
            child_internal_env.pop(_PROFILE_FD_ENV, None)
    if trace is not None:
        trace["sandbox_spawn_start_ns"] = time.monotonic_ns()
    cgroup = _command_cgroup(uuid.uuid4().hex, nproc_budget) if nproc_budget is not None else None
    try:
        spawn_options = {
            "env": env, "internal_env": child_internal_env,
            "readonly_workspace": readonly_workspace,
            "cpu_seconds": cpu_seconds, "memory_bytes": memory_bytes,
            "file_bytes": file_bytes, "nofile": nofile, "nproc": nproc,
        }
        if generic_check:
            spawn_options["generic_check"] = True
        if pass_fds:
            spawn_options["pass_fds"] = pass_fds
        if cgroup is not None:
            spawn_options["cgroup"] = cgroup
        process, report = sandboxed_popen(command, root, **spawn_options)
    except BaseException:
        for fd in (profile_reader, profile_writer):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if cgroup is not None:
            try:
                cgroup.rmdir()
            except OSError:
                pass
        raise
    if trace is not None:
        trace["sandbox_spawn_complete_ns"] = time.monotonic_ns()
    if profile_writer is not None:
        try:
            os.close(profile_writer)
        except OSError:
            pass
        profile_writer = None
    try:
        deadline = time.monotonic() + timeout
        if trace is None:
            stdout, stderr, timed_out, truncated, group_killed = _collect(
                process, payload, deadline, output_limit)
        else:
            stdout, stderr, timed_out, truncated, group_killed = _collect(
                process, payload, deadline, output_limit,
                observer_fd=profile_reader, observer_buffer=profile_buffer, trace=trace)
    except BaseException:
        _kill_group(process)
        raise
    finally:
        if profile_reader is not None:
            try:
                os.close(profile_reader)
            except OSError:
                pass
        if cgroup is not None:
            try:
                cgroup.rmdir()
            except OSError:
                pass
    profile_payload = bytes(profile_buffer) if profile_buffer else None
    return PolicyResult(
        completed=subprocess.CompletedProcess(
            _arguments(command), process.returncode, stdout, stderr),
        timed_out=timed_out,
        process_group_killed=group_killed,
        output_truncated=truncated,
        capabilities=report,
        pid=process.pid if isinstance(process.pid, int) else None,
            execution_trace=trace or {},
        profile_payload=profile_payload,
    )


__all__ = [
    "CapabilityReport", "DEFAULT_CPU_SECONDS", "DEFAULT_FILE_BYTES",
    "DEFAULT_MEMORY_BYTES", "ExecutionPolicyError", "MAX_OUTPUT_BYTES",
    "PolicyResult", "available", "capability_report", "require_capabilities",
    "run_sandboxed", "sandboxed_popen",
]
