"""One supervised execution lifetime per run, with explicit cleanup results."""
from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid

from kernel_exec_capability import ExecCapability


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


def _cleanup(value: dict) -> CleanupReport:
    return CleanupReport(value["ok"], value["killed"], tuple(value["survivors"]), value["detail"])


def _grace(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 10:
        raise ValueError("cleanup grace must be finite and between 0 and 10 seconds")
    return value


class JobHandle:
    def __init__(self, ident: str, session: ExecSession):
        self.id = ident
        self._session = session
        self.lost_bytes = 0

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


class ExecSession:
    def __init__(self, run_dir: Path, *, platform_backend: str | None = None):
        self.run_dir = Path(run_dir).resolve(strict=True)
        self.backend = platform_backend or ("sandbox-exec" if platform.system() == "Darwin" else "bwrap")
        if self.backend not in {"sandbox-exec", "bwrap"}:
            raise ValueError("platform_backend must be sandbox-exec or bwrap")
        self._directory = self.run_dir / ("exec-" + uuid.uuid4().hex)
        self._directory.mkdir(mode=0o700)
        self._process = None
        self._control = None
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
            raise ExecSessionError("Linux execution requires a non-root host UID to enforce RLIMIT_NPROC")
        self._sandboxed = cap.sandboxed
        command = [sys.executable, "-I", str(Path(__file__).with_name("kernel_exec_supervisor.py")),
                   str(self._directory), self.backend, "host"]
        if cap.sandboxed and self.backend == "bwrap":
            executable = shutil.which("bwrap")
            if executable is None:
                raise ExecSessionError("execution capability requires bwrap; no unsandboxed fallback")
            network = "full" if cap.network == "full" else "loopback"
            command[-1] = network
            self._write_envelope = (*cap.write_roots, cap.workspace, self.run_dir)
            # Only the trusted supervisor sees this read-only host tree. Every
            # target gets a fresh mount/user namespace with capability-only binds.
            wrapper = [executable, "--unshare-all", "--die-with-parent", "--new-session", "--as-pid-1",
                       "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                       "--bind", str(self.run_dir), str(self.run_dir)]
            if network == "full":
                wrapper += ["--share-net"]
            for root in sorted(set(self._write_envelope), key=lambda p: (len(p.parts), str(p))):
                wrapper += ["--bind", str(root), str(root)]
            command = [*wrapper, "--", *command]
        parent, child = socket.socketpair()
        self._log = (self._directory / "supervisor.log").open("xb", buffering=0)
        try:
            self._process = subprocess.Popen(command, stdin=child, stdout=subprocess.DEVNULL,
                                             stderr=self._log, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                                             close_fds=True, start_new_session=True)
        except OSError as error:
            parent.close()
            self._log.close()
            raise ExecSessionError(f"cannot launch execution supervisor: {error}") from error
        finally:
            child.close()
        parent.settimeout(15)
        self._control = parent

    def _rpc(self, request: dict) -> dict:
        with self._lock:
            if self._closed or self._control is None:
                raise ExecSessionError("execution session is not open")
            payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
            if len(payload) > 1 << 20:
                raise ValueError("execution control request exceeds 1 MiB")
            try:
                self._control.sendall(payload)
                response = bytearray()
                while not response.endswith(b"\n"):
                    part = self._control.recv(65536)
                    if not part:
                        raise ExecSessionError(f"execution supervisor exited; inspect {self._directory / 'supervisor.log'}")
                    response.extend(part)
                    if len(response) > 6 << 20:
                        raise ExecSessionError("execution supervisor response exceeds bound")
            except OSError as error:
                raise ExecSessionError(f"execution supervisor communication failed: {error}") from error
            value = json.loads(response)
            if "error" in value:
                raise ExecSessionError(value["error"])
            return value

    def _spawn(self, argv, cap, *, cwd=None, stdin=None) -> JobHandle:
        if isinstance(argv, (str, bytes)):
            raise TypeError("argv must be a sequence, not shell text")
        with self._lock:
            self._start(self._initial_capability or cap)
            self._start(cap)
            value = self._rpc({"op": "start", "argv": list(argv), "cap": cap.as_dict(),
                               "cwd": str(cwd) if cwd is not None else None,
                               "stdin": base64.b64encode(stdin).decode() if stdin is not None else None})
            handle = JobHandle(value["id"], self)
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
                                      _cleanup(status["cleanup"]))
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
            if self._control is not None:
                self._control.close()
            if self._process is not None:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
                    raise ExecSessionError("execution supervisor did not terminate after cleanup")
            if self._log is not None:
                self._log.close()
        if report is not None and not report.ok:
            raise ExecSessionError(f"execution cleanup failed: {report.detail}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
