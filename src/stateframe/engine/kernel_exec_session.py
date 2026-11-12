"""One supervised execution lifetime per run, with explicit cleanup results."""
from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

if __package__:
    from .kernel_exec_capability import ExecCapability
    from .kernel_exec_tracker import capture_launch_identity, kill_launch_unit, recorded_launch_identity
    from .kernel_test_isolation import sandbox_executable
    from .sf_launch import _current_cgroup
else:
    from kernel_exec_capability import ExecCapability
    from kernel_exec_tracker import capture_launch_identity, kill_launch_unit, recorded_launch_identity
    from kernel_test_isolation import sandbox_executable
    from sf_launch import _current_cgroup


_OWNER_PIPE_WRITES: set[int] = set()


def _close_owner_pipe_writers_after_fork() -> None:
    """Prevent forked engine helpers from extending execution-session lifetime."""
    descriptors = tuple(_OWNER_PIPE_WRITES)
    _OWNER_PIPE_WRITES.clear()
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                # A child that cannot close this end must not survive holding EOF open.
                os._exit(127)


def close_owner_pipe_writers_after_fork() -> None:
    """Explicitly close registered owner ends in known fork-child paths."""
    _close_owner_pipe_writers_after_fork()


os.register_at_fork(after_in_child=_close_owner_pipe_writers_after_fork)


def _owner_pipe() -> tuple[int, int]:
    """Create close-on-exec descriptors on platforms with and without pipe2."""
    if hasattr(os, "pipe2"):
        return os.pipe2(os.O_CLOEXEC)
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    return read_fd, write_fd


class ExecSessionError(RuntimeError):
    """The execution boundary could not be established or maintained."""


@dataclass(frozen=True)
class CleanupReport:
    ok: bool
    killed: int
    survivors: tuple[int, ...]
    detail: str


@dataclass(frozen=True)
class ExecResult:
    exit_code: int | None
    timed_out: bool
    output: bytes
    output_truncated: bool
    output_total_bytes: int
    wall_seconds: float
    cleanup: CleanupReport
    memory_limit_mechanism: str | None = None
    cgroup_leaf_pattern: str | None = None


def _cleanup(value: dict) -> CleanupReport:
    return CleanupReport(value["ok"], value["killed"], tuple(value["survivors"]), value["detail"])


def _supervisor_diagnostics(process: subprocess.Popen | None, log_path: Path | None,
                            process_role: str | None = None) -> str:
    if process is None:
        status = "execution process status unknown"
    else:
        role = process_role or "execution process"
        try:
            exit_code = process.poll()
        except OSError:
            exit_code = None
        status = (f"{role} exit status: {exit_code}" if exit_code is not None
                  else f"{role} status unknown")

    log_diagnostic = "supervisor.log absent"
    if (log_path is not None and hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "O_DIRECTORY")):
        flags = (os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            directory = os.open(log_path.parent, directory_flags)
            descriptor = None
            try:
                descriptor = os.open(log_path.name, flags, dir_fd=directory)
            finally:
                try:
                    os.close(directory)
                except OSError:
                    if descriptor is not None:
                        os.close(descriptor)
                    raise
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    return f"{status}\n{log_diagnostic}"
                end = os.lseek(descriptor, 0, os.SEEK_END)
                os.lseek(descriptor, max(0, end - 8192), os.SEEK_SET)
                tail = bytearray()
                while len(tail) < 8192:
                    chunk = os.read(descriptor, 8192 - len(tail))
                    if not chunk:
                        break
                    tail.extend(chunk)
                log_diagnostic = "supervisor.log tail (up to 8192 bytes):\n" + tail.decode(
                    "utf-8", "replace")
            finally:
                os.close(descriptor)
        except OSError:
            # Missing, linked, non-regular, or unreadable logs are not followed or trusted.
            log_diagnostic = "supervisor.log absent"
    return f"{status}\n{log_diagnostic}"


def _grace(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 10:
        raise ValueError("cleanup grace must be finite and between 0 and 10 seconds")
    return value


def _host_load() -> tuple[float, int]:
    cpus = os.cpu_count() or 1
    try:
        load = os.getloadavg()[0]
    except (AttributeError, OSError):
        load = 0.0
    return (load if math.isfinite(load) else 0.0), cpus


def rpc_timeout_seconds() -> float:
    """Allow longer supervisor waits when the host's runnable load exceeds its CPU count."""
    try:
        base = float(os.environ.get("STATEFRAME_EXEC_RPC_SECONDS", "15"))
    except ValueError:
        base = 15.0
    if not math.isfinite(base) or base <= 0:
        base = 15.0
    load, cpus = _host_load()
    return base * min(8.0, max(1.0, load / cpus))


class JobHandle:
    def __init__(self, ident: str, session: ExecSession,
                 memory_limit_mechanism: str | None = None,
                 cgroup_leaf_pattern: str | None = None):
        self.id = ident
        self._session = session
        self.lost_bytes = 0
        self.memory_limit_mechanism = memory_limit_mechanism
        self.cgroup_leaf_pattern = cgroup_leaf_pattern

    def _status(self, consume=False):
        return self._session._rpc({"op": "status", "id": self.id, "consume": consume})

    def poll(self) -> int | None:
        return self._status()["exit_code"]

    def read_available(self) -> bytes:
        result = self._status(True)
        self.lost_bytes += result["lost_bytes"]
        return base64.b64decode(result["output"])

    def stop(self, grace_seconds=2.0) -> CleanupReport:
        self._session._rpc({"op": "stop", "id": self.id, "grace": _grace(grace_seconds)})
        deadline = time.monotonic() + grace_seconds + 4
        while time.monotonic() < deadline:
            status = self._status()
            if status["finished"]:
                return _cleanup(status["cleanup"])
            time.sleep(0.02)
        return CleanupReport(False, 0, (), "supervisor did not establish quiescence before deadline")

    def wait(self, timeout: float) -> int | None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("wait timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        while True:
            status = self._status()
            if status["finished"]:
                if not status["cleanup"]["ok"]:
                    raise ExecSessionError(status["cleanup"]["detail"])
                return status["exit_code"]
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))


class RestoredJobHandle:
    """A job handle restored from a checkpoint. Checkpoints refuse live jobs, so the job had
    already finished in the saving process; nothing is left to read, poll or stop."""

    def __init__(self, ident: str, lost_bytes: int):
        self.id = ident
        self.lost_bytes = lost_bytes

    def poll(self) -> int | None:
        raise ExecSessionError("job finished before the checkpoint; it has no live process")

    def read_available(self) -> bytes:
        return b""

    def stop(self, grace_seconds=2.0) -> CleanupReport:
        return CleanupReport(True, 0, (), "job finished before the checkpoint")

    def wait(self, timeout: float) -> int | None:
        raise ExecSessionError("job finished before the checkpoint; it has no live process")


class ExecSession:
    def __init__(self, run_dir: Path, *, platform_backend: str | None = None):
        self.run_dir = Path(run_dir).resolve(strict=True)
        self.backend = platform_backend or ("sandbox-exec" if platform.system() == "Darwin" else "bwrap")
        if self.backend not in {"sandbox-exec", "bwrap"}:
            raise ValueError("platform_backend must be sandbox-exec or bwrap")
        # Replay deliberately repeats the UUID stream, but a previous attempt's
        # supervisor artifacts must remain intact. Allocate a fresh physical directory.
        self._directory = Path(tempfile.mkdtemp(
            prefix="exec-" + uuid.uuid4().hex + "-", dir=self.run_dir))
        self._process = None
        self._process_role: str | None = None
        self._control = None
        self._owner_write_fd: int | None = None
        self._receive = bytearray()
        self._next_rid = 0
        self._log = None
        self._lock = threading.RLock()
        self._closed = False
        self._sandboxed = None
        self._write_envelope: tuple[Path, ...] = ()
        self._handles: list[JobHandle] = []
        self._initial_capability: ExecCapability | None = None

    def configure(self, capability: ExecCapability) -> None:
        """Pin the run namespace envelope before any purpose launches a command."""
        if self._process is not None or self._initial_capability is not None:
            raise ExecSessionError("execution session authority is already configured")
        self._initial_capability = capability

    def startup_diagnostics(self) -> str:
        """Describe only retained execution-process state and the private supervisor log."""
        return _supervisor_diagnostics(
            self._process, self._directory / "supervisor.log", self._process_role)

    def _supervisor_error(self, message: str) -> ExecSessionError:
        return ExecSessionError(
            f"{message}\nstartup supervisor diagnostics:\n{self.startup_diagnostics()}")

    def _start(self, cap: ExecCapability) -> None:
        if self._closed:
            raise ExecSessionError("execution session is closed")
        if self._process is not None:
            if cap.sandboxed != self._sandboxed:
                raise ExecSessionError("cannot mix host and sandboxed execution in one session")
            if cap.sandboxed and self.backend == "bwrap":
                requested = (*cap.write_roots, *((cap.workspace,) if cap.workspace_writable else ()))
                if any(not any(p == root or root in p.parents for root in self._write_envelope) for p in requested):
                    raise ExecSessionError("write authority exceeds initial Linux session mount envelope")
            return
        if sys.platform.startswith("linux") and os.getuid() == 0:
            raise ExecSessionError("Linux execution requires a non-root host UID")
        self._sandboxed = cap.sandboxed
        bwrap_executable = None
        if cap.sandboxed and self.backend == "bwrap":
            bwrap_executable = sandbox_executable("bwrap")
            if bwrap_executable is None:
                raise ExecSessionError("execution capability requires bwrap; no unsandboxed fallback")
            network = "full" if cap.network == "full" else "loopback"
            self._write_envelope = (*cap.write_roots, cap.workspace, self.run_dir)
        else:
            network = "host"
        parent, child = socket.socketpair()
        self._log = (self._directory / "supervisor.log").open("xb", buffering=0)
        owner_read_fd = owner_write_fd = None
        try:
            owner_read_fd, owner_write_fd = _owner_pipe()
            self._owner_write_fd = owner_write_fd
            _OWNER_PIPE_WRITES.add(owner_write_fd)
            command = [sys.executable, "-I", str(Path(__file__).with_name("kernel_exec_supervisor.py")),
                       str(self._directory), self.backend, network, str(owner_read_fd)]
            if bwrap_executable is not None:
                # bwrap closes unlisted descriptors during setup. --sync-fd keeps this
                # owner channel through its monitor; --as-pid-1 passes it to the command.
                wrapper = [bwrap_executable, "--unshare-all", "--die-with-parent", "--new-session",
                           "--as-pid-1", "--sync-fd", str(owner_read_fd),
                           "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                           "--bind", str(self.run_dir), str(self.run_dir)]
                cgroup = _current_cgroup()
                try:
                    pids_delegated = (cgroup is not None
                                      and "pids" in (cgroup / "cgroup.subtree_control").read_text().split())
                except OSError:
                    pids_delegated = False
                if pids_delegated and os.access(cgroup, os.W_OK):
                    # Only the trusted supervisor can write this delegated subtree.
                    # The nested command's mount envelope does not include it.
                    wrapper += ["--bind", str(cgroup), "/sys/fs/cgroup"]
                if network == "full":
                    wrapper += ["--share-net"]
                for root in sorted(set(self._write_envelope), key=lambda p: (len(p.parts), str(p))):
                    wrapper += ["--bind", str(root), str(root)]
                command = [*wrapper, "--", *command]
            self._process = subprocess.Popen(command, stdin=child, stdout=subprocess.DEVNULL,
                                             stderr=self._log, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                                             close_fds=True, start_new_session=True,
                                             pass_fds=(owner_read_fd,))
            self._process_role = "launcher (bwrap)" if bwrap_executable is not None else "supervisor"
            capture_launch_identity(self._process, session_leader=True)
        except BaseException as error:  # noqa: BLE001 - release every owner fd on launch failure
            parent.close()
            self._log.close()
            self._close_owner_write_end()
            if isinstance(error, OSError):
                raise ExecSessionError(f"cannot launch execution supervisor: {error}") from error
            raise
        finally:
            child.close()
            if owner_read_fd is not None:
                os.close(owner_read_fd)
        parent.settimeout(rpc_timeout_seconds())
        self._control = parent

    def _close_owner_write_end(self) -> None:
        descriptor = self._owner_write_fd
        self._owner_write_fd = None
        if descriptor is None:
            return
        _OWNER_PIPE_WRITES.discard(descriptor)
        os.close(descriptor)

    def _rpc(self, request: dict) -> dict:
        with self._lock:
            if self._closed or self._control is None:
                raise ExecSessionError("execution session is not open")
            self._next_rid += 1
            payload = json.dumps({**request, "rid": self._next_rid}, separators=(",", ":")).encode() + b"\n"
            if len(payload) > 1 << 20:
                raise ValueError("execution control request exceeds 1 MiB")
            try:
                self._control.sendall(payload)
                while True:
                    if b"\n" not in self._receive:
                        try:
                            part = self._control.recv(65536)
                        except socket.timeout as error:
                            load, cpus = _host_load()
                            seconds = self._control.gettimeout()
                            raise self._supervisor_error(
                                f"execution supervisor did not answer within {seconds:g} s "
                                f"(host load {load:g} on {cpus} CPUs); retry when the machine is less busy "
                                "or raise STATEFRAME_EXEC_RPC_SECONDS") from error
                        if not part:
                            raise self._supervisor_error(
                                f"execution supervisor exited; inspect {self._directory / 'supervisor.log'}")
                        self._receive.extend(part)
                        if len(self._receive) > 6 << 20:
                            raise ExecSessionError("execution supervisor response exceeds bound")
                    if b"\n" not in self._receive:
                        continue
                    line, _, rest = self._receive.partition(b"\n")
                    self._receive = bytearray(rest)
                    value = json.loads(line)
                    if value.get("rid") != self._next_rid:
                        continue
                    if "error" in value:
                        raise ExecSessionError(value["error"])
                    return value
            except OSError as error:
                raise self._supervisor_error(
                    f"execution supervisor communication failed: {error}") from error

    def _spawn(self, argv, cap, *, cwd=None, stdin=None) -> JobHandle:
        if isinstance(argv, (str, bytes)):
            raise TypeError("argv must be a sequence, not shell text")
        with self._lock:
            self._start(self._initial_capability or cap)
            self._start(cap)
            value = self._rpc({"op": "start", "argv": list(argv), "cap": cap.as_dict(),
                               "cwd": str(cwd) if cwd is not None else None,
                               "stdin": base64.b64encode(stdin).decode() if stdin is not None else None})
            handle = JobHandle(value["id"], self, value.get("memory_limit_mechanism"),
                               value.get("cgroup_leaf_pattern"))
            self._handles.append(handle)
            return handle

    def spawn(self, argv, cap, *, cwd=None) -> JobHandle:
        if cap.workspace_writable:
            raise ValueError("background jobs require read-only workspace")
        if sum(handle.poll() is None for handle in self._handles) >= 4:
            raise ExecSessionError("at most four running background jobs")
        return self._spawn(argv, cap, cwd=cwd)

    def run(self, argv: Sequence[str], cap: ExecCapability, *, stdin: bytes | None = None,
            cwd: Path | None = None) -> ExecResult:
        handle = self._spawn(argv, cap, cwd=cwd, stdin=stdin)
        head = bytearray()
        tail = bytearray()
        half = cap.limits.output_bytes // 2
        total_received = 0
        lost = 0
        try:
            while True:
                status = handle._status(True)
                chunk = base64.b64decode(status["output"])
                lost += status["lost_bytes"]
                total_received += len(chunk)
                need = max(0, half - len(head))
                head.extend(chunk[:need])
                tail.extend(chunk[need:])
                if len(tail) > cap.limits.output_bytes - half:
                    del tail[:len(tail) - (cap.limits.output_bytes - half)]
                if status["finished"]:
                    return ExecResult(status["exit_code"], status["timed_out"], bytes(head + tail),
                                      bool(lost or total_received > cap.limits.output_bytes),
                                      status["output_total_bytes"], status["wall_seconds"],
                                      _cleanup(status["cleanup"]),
                                      status.get("memory_limit_mechanism"),
                                      handle.cgroup_leaf_pattern)
                time.sleep(0.01)
        except BaseException:
            handle.stop(0.1)
            raise

    def stop_all(self, grace_seconds: float = 2.0) -> CleanupReport:
        _grace(grace_seconds)
        if self._process is None:
            return CleanupReport(True, 0, (), "")
        self._rpc({"op": "stop_all", "grace": grace_seconds})
        deadline = time.monotonic() + grace_seconds + 4
        while time.monotonic() < deadline:
            statuses = self._rpc({"op": "all"})["jobs"].values()
            if all(status["finished"] for status in statuses):
                reports = [_cleanup(status["cleanup"]) for status in statuses]
                return CleanupReport(all(r.ok for r in reports), sum(r.killed for r in reports),
                                     tuple(sorted({p for r in reports for p in r.survivors})),
                                     "; ".join(r.detail for r in reports if r.detail))
            time.sleep(0.02)
        return CleanupReport(False, 0, (), "session cleanup deadline exceeded")

    def close(self) -> None:
        if self._closed:
            return
        report = None
        try:
            report = self.stop_all()
            if self._process is not None:
                self._rpc({"op": "close", "grace": 0})
        finally:
            self._closed = True
            self._close_owner_write_end()
            if self._control is not None:
                self._control.close()
            if self._process is not None:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Kill the whole launch unit: the outer bwrap alone can leave a
                    # supervisor that is still setting up its sandbox alive.
                    survivors = kill_launch_unit(
                        self._process, deadline_seconds=5,
                        launch_identity=recorded_launch_identity(self._process))
                    detail = f"; survivors {', '.join(map(str, survivors))}" if survivors else ""
                    raise ExecSessionError("execution supervisor did not terminate after cleanup" + detail)
            if self._log is not None:
                self._log.close()
        if report is not None and not report.ok:
            raise ExecSessionError(f"execution cleanup failed: {report.detail}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
