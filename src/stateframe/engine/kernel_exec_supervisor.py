"""Private execution-session server. The control socket is never inherited by jobs."""
from __future__ import annotations

import base64
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

# -I deliberately ignores ambient PYTHONPATH and the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_exec_capability import capability_from_dict
from kernel_exec_tracker import Descendants, ProcessTable
from kernel_job_output import JobOutputCollector
from kernel_test_isolation import capability_command
from sf_launch import _command_cgroup


class ManagedProcess:
    def __init__(self, ident, process, collector, cap, table, cgroup=None):
        self.id = ident
        self.process = process
        self.collector = collector
        self.cap = cap
        self.family = Descendants(process.pid, table)
        self.started = time.monotonic()
        self.stop_at = None
        self.kill_at = None
        self.term_sent: set[int] = set()
        self.killed: set[int] = set()
        self.timed_out = False
        self.cleanup = None
        self.eof = False
        self.cursor = 0
        self.exit_code = None
        self.ended = None
        self.limit_detail = ""
        self.containment_failed = False
        self.cgroup = cgroup

    def remove_cgroup(self):
        if self.cgroup is not None:
            try:
                self.cgroup.rmdir()
            except OSError:
                return  # A still-reaping task can temporarily keep it populated.
            self.cgroup = None

    def stop(self, grace):
        if self.cleanup is None and self.stop_at is None:
            self.stop_at = time.monotonic()
            self.kill_at = self.stop_at + grace

    def tick(self, rows):
        if self.cleanup is not None:
            self.remove_cgroup()
            return
        now = time.monotonic()
        if not self.eof:
            try:
                self.eof = self.collector.drain(self.process.stdout.fileno())
            except OSError as error:
                self.containment_failed = True
                self.limit_detail = f"output collection failed (errno {error.errno})"
                self.eof = True
                self.process.stdout.close()
                self.stop(0)
        live = self.family.update(rows)
        if (sum(self.family.table.resident_bytes(row) for row in live) > self.cap.limits.memory_bytes
                or len(live) > self.cap.limits.nproc):
            self.limit_detail = "aggregate resident-memory or process limit exceeded"
            self.stop(0)
        self.exit_code = self.process.poll()
        if self.exit_code is not None:
            self.stop(0.1)
        if now - self.started >= self.cap.limits.wall_seconds and self.stop_at is None:
            self.timed_out = True
            self.stop(0.1)
        if self.stop_at is not None:
            fresh = [p for p in live if p.pid not in self.term_sent]
            self.term_sent |= self.family.signal(fresh, signal.SIGTERM)
            if now >= self.kill_at:
                self.killed |= self.family.signal(live, signal.SIGKILL)
            if not live and self.process.poll() is not None:
                # No producers remain. Drain the finite pipe fully before publishing exit.
                while not self.eof:
                    before = self.collector.total_bytes
                    self.eof = self.collector.drain(self.process.stdout.fileno())
                    if self.collector.total_bytes == before and not self.eof:
                        break
                if self.eof:
                    self.complete(True, ())
                elif now > self.kill_at + 2:
                    self.complete(False, (), "output pipe held by untracked descendant")
            elif now > self.kill_at + 2:
                self.complete(False, tuple(p.pid for p in live), "descendants survived SIGKILL")

    def complete(self, ok, survivors, detail=""):
        if survivors and self.family.signal_errors:
            detail += "; signal permission denied"
        self.cleanup = {"ok": ok and not self.containment_failed, "killed": len(self.killed | self.term_sent),
                        "survivors": survivors, "detail": detail or self.limit_detail}
        self.exit_code = self.process.poll()
        self.ended = time.monotonic()
        self.process.stdout.close()
        self.collector.seal()
        self.remove_cgroup()

    def status(self, consume=False):
        output = b""
        lost = 0
        if consume:
            output, self.cursor, lost = self.collector.read_since(self.cursor)
        return {"exit_code": self.exit_code if self.cleanup is not None else None,
                "finished": self.cleanup is not None, "cleanup": self.cleanup,
                "timed_out": self.timed_out, "output": base64.b64encode(output).decode(),
                "output_total_bytes": self.collector.total_bytes,
                "lost_bytes": lost, "wall_seconds": (self.ended or time.monotonic()) - self.started}


class Supervisor:
    def __init__(self, control, directory, backend, namespace_network):
        self.control = control
        self.directory = directory
        self.backend = backend
        self.namespace_network = namespace_network
        self.table = ProcessTable()
        self.jobs: dict[str, ManagedProcess] = {}
        self.next_id = 0
        self.closing = False
        self.selector = selectors.DefaultSelector()
        self.control.setblocking(False)
        self.selector.register(control, selectors.EVENT_READ)
        self.incoming = bytearray()
        self.outgoing = bytearray()

    def start(self, request):
        if any(job.cleanup is not None and not job.cleanup["ok"] for job in self.jobs.values()):
            raise RuntimeError("session has unresolved cleanup failure")
        cap = capability_from_dict(request["cap"])
        if cap.purpose == "job" and cap.workspace_writable:
            raise ValueError("background jobs require read-only workspace")
        if sum(job.cleanup is None for job in self.jobs.values()) >= 5:
            raise ValueError("execution session has five active commands")
        private_loopback = (self.namespace_network == "full" and cap.network == "loopback"
                            and cap.purpose == "custom_tool")
        if (self.namespace_network != "host" and not private_loopback
                and cap.network not in {"none", self.namespace_network}):
            raise ValueError("capability network differs from persistent session network")
        argv = request["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or "\0" in v for v in argv):
            raise ValueError("argv must be a nonempty sequence of strings without NUL")
        cwd = Path(request.get("cwd") or cap.workspace).resolve(strict=True)
        spec = {"argv": argv, "cwd": str(cwd), "env": dict(cap.env), "limits": asdict(cap.limits),
                "memory_supervised": True}
        # The launcher is stdlib-only and isolated (-I): exec the real interpreter, not a venv
        # symlink, so the sandbox only has to admit the resolved runtime roots.
        launch = [os.path.realpath(sys.executable), "-I", str(Path(__file__).with_name("sf_launch.py").resolve()),
                  json.dumps(spec, separators=(",", ":"))]
        command, environment = capability_command(
            launch, cap, platform_backend=self.backend,
            shared_namespace=self.namespace_network != "host")
        if private_loopback:
            # A custom tool may narrow a full-network run. Keep the supervised
            # PID namespace, but give this command a disconnected loopback netns.
            command.insert(1, "--unshare-net")
        self.next_id += 1
        ident = f"j{self.next_id}"
        collector = JobOutputCollector(self.directory / f"{ident}.output", capacity=1 << 20)
        cgroup = _command_cgroup(ident, cap.limits.nproc) if cap.sandboxed and self.backend == "bwrap" else None
        if cgroup is not None:
            command = [os.path.realpath(sys.executable), "-I", str(Path(__file__).with_name("sf_launch.py")),
                       "--cgroup-enter", str(cgroup), *command]
        try:
            with tempfile.TemporaryFile(dir=self.directory) as input_file:
                supplied = request.get("stdin")
                if supplied is not None:
                    input_file.write(base64.b64decode(supplied, validate=True))
                    input_file.seek(0)
                process = subprocess.Popen(command, stdin=input_file if supplied is not None else subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           env=environment, close_fds=True)
        except BaseException:
            collector.close()
            if cgroup is not None:
                try:
                    cgroup.rmdir()
                except OSError:
                    pass
            raise
        job = ManagedProcess(ident, process, collector, cap, self.table, cgroup)
        self.jobs[ident] = job
        return {"id": ident}

    def dispatch(self, request):
        op = request["op"]
        if op == "start":
            return self.start(request)
        if op in {"status", "stop"}:
            job = self.jobs[request["id"]]
            if op == "stop":
                job.stop(min(10.0, max(0.0, float(request["grace"]))))
            return job.status(request.get("consume", False))
        if op in {"stop_all", "close"}:
            for job in self.jobs.values():
                job.stop(min(10.0, max(0.0, float(request.get("grace", 2.0)))))
            if op == "close":
                self.closing = True
            return {"jobs": {ident: job.status() for ident, job in self.jobs.items()}}
        if op == "all":
            return {"jobs": {ident: job.status() for ident, job in self.jobs.items()}}
        raise ValueError(f"unknown execution supervisor operation: {op}")

    def serve(self):
        connected = True
        try:
            while True:
                rows = self.table.snapshot()
                if self.namespace_network != "host":
                    leaders = {j.process.pid for j in self.jobs.values()}
                    for job in self.jobs.values():
                        if job.cleanup is None:
                            job.family.update(rows)
                    recorded = {pid for j in self.jobs.values() for pid in j.family.members}
                    for row in rows.values():
                        if row.ppid != os.getpid() or row.pid in leaders:
                            continue
                        if row.zombie:
                            try:
                                os.waitpid(row.pid, os.WNOHANG)
                            except ChildProcessError:
                                pass
                        elif row.pid not in recorded:
                            # A double-fork between sweeps cannot be attributed to
                            # one command. Stop the entire session, fail closed.
                            current = self.table.identity(row.pid)
                            if current is not None and current.start == row.start:
                                os.kill(row.pid, signal.SIGKILL)
                            for job in self.jobs.values():
                                if job.cleanup is None:
                                    job.containment_failed = True
                                    job.limit_detail = "unattributed namespace descendant; session stopped"
                                    job.stop(0)
                for job in self.jobs.values():
                    job.tick(rows)
                if self.closing and all(j.cleanup is not None for j in self.jobs.values()) and not self.outgoing:
                    break
                events = self.selector.select(0.02)
                for _, mask in events:
                    if mask & selectors.EVENT_READ:
                        try:
                            chunk = self.control.recv(65536)
                        except ConnectionResetError:
                            chunk = b""
                        if not chunk:
                            connected = False
                            self.selector.unregister(self.control)
                            self.outgoing.clear()
                            self.closing = True
                            for job in self.jobs.values():
                                job.stop(0.1)
                            continue
                        self.incoming.extend(chunk)
                        if len(self.incoming) > 1 << 20:
                            raise ValueError("execution control request exceeds 1 MiB")
                        if b"\n" in self.incoming:
                            line, _, rest = self.incoming.partition(b"\n")
                            self.incoming = bytearray(rest)
                            try:
                                response = self.dispatch(json.loads(line))
                            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                                response = {"error": f"{type(error).__name__}: {error}"}
                            self.outgoing.extend(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                    if connected and mask & selectors.EVENT_WRITE and self.outgoing:
                        try:
                            count = self.control.send(self.outgoing)
                            del self.outgoing[:count]
                        except (BrokenPipeError, ConnectionResetError):
                            connected = False
                            self.selector.unregister(self.control)
                            self.outgoing.clear()
                            self.closing = True
                            for job in self.jobs.values():
                                job.stop(0.1)
                if connected:
                    self.selector.modify(self.control, selectors.EVENT_READ |
                                         (selectors.EVENT_WRITE if self.outgoing else 0))
        finally:
            # On an internal failure, attempt cleanup before surfacing the error.
            deadline = time.monotonic() + 3
            for job in self.jobs.values():
                job.stop(0)
            while any(j.cleanup is None for j in self.jobs.values()) and time.monotonic() < deadline:
                rows = self.table.snapshot()
                for job in self.jobs.values():
                    job.tick(rows)
                time.sleep(0.01)
            for job in self.jobs.values():
                job.collector.close()
                job.remove_cgroup()
            self.selector.close()
            self.control.close()


def main():
    directory = Path(sys.argv[1])
    backend, network = sys.argv[2:4]
    if network == "loopback":
        # Bubblewrap configures private lo before dropping its setup privileges.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            flags = fcntl.ioctl(sock.fileno(), 0x8913, struct.pack("16sh", b"lo", 0))
            if not struct.unpack_from("h", flags, 16)[0] & 1:
                raise RuntimeError("persistent execution namespace has no loopback interface")
    Supervisor(socket.socket(fileno=0), directory, backend, network).serve()


if __name__ == "__main__":
    main()
