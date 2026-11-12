"""Private execution-session server. The control socket is never inherited by jobs."""
from __future__ import annotations

import base64
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
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
from kernel_exec_tracker import Descendants, ProcessTable, process_cmdline_digest
from kernel_execution_policy import (
    FULL_PROCESS_SCAN_INTERVAL_SECONDS,
    _remove_command_cgroup,
)
from kernel_job_output import JobOutputCollector
from kernel_test_isolation import capability_command
from sf_launch import (
    _cgroup_leaf_pattern_status,
    _cgroup_memory_limit_configured,
    _command_cgroup,
)


def _read_memory_join_result(fd: int, *, timeout: float = 10.0) -> bytes:
    """Wait only a bounded time for the pre-bwrap launcher to report its cgroup join."""
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise RuntimeError("command memory join result timed out")
    return os.read(fd, 1)


class ManagedProcess:
    def __init__(self, ident, process, collector, cap, table, cgroup=None,
                 memory_limit_mechanism=None, armable_wall_s=None):
        self.id = ident
        self.process = process
        self.collector = collector
        self.cap = cap
        self.family = Descendants(process.pid, table)
        self.started = time.monotonic()
        self.armable_wall_s = armable_wall_s
        self.armed = armable_wall_s is None
        self.wall_seconds = cap.limits.wall_seconds
        self.stop_at = None
        self.kill_at = None
        self.term_sent: set[int] = set()
        self.killed: set[int] = set()
        self.timed_out = False
        self.timeout_at = None
        self.cleanup = None
        self.eof = False
        self.cursor = 0
        self.exit_code = None
        self.ended = None
        self.limit_detail = ""
        self.containment_failed = False
        self.leader_reaped = False
        self.initial_scan_pending = True
        self.post_exit_scan_pending = False
        self.post_exit_scan_complete = False
        self.cgroup = cgroup
        self.memory_limit_mechanism = memory_limit_mechanism
        self.output_overflow = False

    def remove_cgroup(self) -> bool:
        if self.cgroup is None:
            return True
        if not _remove_command_cgroup(self.cgroup):
            return False
        self.cgroup = None
        return True

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
        if self.cap.limits.kill_on_output and self.collector.total_bytes > self.cap.limits.output_bytes:
            self.output_overflow = True
            self.limit_detail = "output byte limit exceeded"
            self.stop(0)
        live = self.family.update(rows)
        breach = self.family.table.limit_breach(live, self.cap.limits)
        if breach:
            self.limit_detail = breach
            self.stop(0)
        self.exit_code = self.process.poll()
        if self.exit_code is not None:
            # poll() has reaped the leader: from here its PID may be reused, so only group
            # members born by now still belong to this command.
            if not self.leader_reaped:
                self.family.leader_reaped()
                self.leader_reaped = True
                self.post_exit_scan_pending = True
            self.stop(0.1)
        if self.armed and now - self.started >= self.wall_seconds and self.stop_at is None:
            self.timed_out = True
            self.timeout_at = now
            self.stop(0.1)
        if self.stop_at is not None:
            fresh = [p for p in live if p.pid not in self.term_sent]
            self.term_sent |= self.family.signal(fresh, signal.SIGTERM)
            if now >= self.kill_at:
                self.killed |= self.family.signal(live, signal.SIGKILL)
            try:
                # Signalling and reaping can change liveness within this tick.
                live = self.family.update(self.family.table.snapshot())
                if getattr(self, "post_exit_scan_pending", False):
                    self.post_exit_scan_pending = False
                    self.post_exit_scan_complete = True
            except (OSError, RuntimeError, ValueError):
                if now > self.kill_at + 2:
                    self.complete(False, tuple(sorted(p.pid for p in live)),
                                  "descendant liveness unverified after signaling")
                return
            if (not live and self.process.poll() is not None
                    and self.post_exit_scan_complete):
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
                details = []
                for row in live[:8]:
                    try:
                        namespace = self.family.table.pid_namespace(row.pid) or "unknown"
                    except (OSError, AttributeError):
                        namespace = "unknown"
                    details.append(
                        f"{row.pid}(ppid={row.ppid},pgid={row.pgid},sid={row.sid},pidns={namespace})")
                suffix = f"; {len(live) - len(details)} more" if len(live) > len(details) else ""
                detail = "descendants survived SIGKILL"
                if details:
                    detail += ": " + "; ".join(details) + suffix
                self.complete(False, tuple(p.pid for p in live), detail[:600])

    def complete(self, ok, survivors, detail=""):
        if survivors and self.family.signal_errors:
            # Name the real failure: a missing pidfd or unreadable identity is not a permission error.
            reasons = sorted({self.family.signal_errors[pid] for pid in survivors
                              if pid in self.family.signal_errors} or set(self.family.signal_errors.values()))
            detail += "; " + "; ".join(reasons)[:300]
        signal_paths = self.family.signal_paths
        self.cleanup = {"ok": ok and not self.containment_failed,
                        "killed": len(self.killed | self.term_sent),
                        "survivors": survivors, "detail": detail or self.limit_detail}
        if signal_paths:
            self.cleanup["signal_paths"] = signal_paths.copy()
        self.exit_code = self.process.poll()
        self.ended = time.monotonic()
        self.process.stdout.close()
        self.collector.seal()
        if not self.remove_cgroup():
            self.containment_failed = True
            self.cleanup["ok"] = False
            cleanup_detail = self.cleanup["detail"]
            self.cleanup["detail"] = (
                cleanup_detail + "; " if cleanup_detail else "")
            self.cleanup["detail"] += "per-command cgroup cleanup failed"

    def status(self, consume=False):
        output = b""
        lost = 0
        if consume:
            output, self.cursor, lost = self.collector.read_since(self.cursor)
        value = {"exit_code": self.exit_code if self.cleanup is not None else None,
                "finished": self.cleanup is not None, "cleanup": self.cleanup,
                "timed_out": self.timed_out, "output": base64.b64encode(output).decode(),
                "memory_limit_mechanism": self.memory_limit_mechanism,
                "output_total_bytes": self.collector.total_bytes,
                "lost_bytes": lost, "wall_seconds": (self.ended or time.monotonic()) - self.started}
        if self.cap.limits.kill_on_output:
            value["output_overflow"] = self.output_overflow
        if self.timeout_at is not None:
            value["timeout_at"] = self.timeout_at
        return value


class Supervisor:
    def __init__(self, control, directory, backend, namespace_network, owner_fd):
        self.control = control
        self.directory = directory
        self.backend = backend
        self.namespace_network = namespace_network
        self.owner_fd = owner_fd
        self.table = ProcessTable()
        self.jobs: dict[str, ManagedProcess] = {}
        self._next_full_process_scan = time.monotonic()
        self.next_id = 0
        self.closing = False
        self.selector = selectors.DefaultSelector()
        self.control.setblocking(False)
        self.selector.register(control, selectors.EVENT_READ, "control")
        os.set_blocking(owner_fd, False)
        self.selector.register(owner_fd, selectors.EVENT_READ, "owner")
        self.incoming = bytearray()
        self.outgoing = bytearray()

    def _begin_close(self):
        self.closing = True
        self.outgoing.clear()
        for job in self.jobs.values():
            job.stop(0.1)

    def _sample_rows(self):
        """Refresh known trees every tick and scan procfs once per second."""
        active = [job for job in self.jobs.values() if job.cleanup is None]
        if sys.platform != "linux":
            rows = self.table.snapshot()
            for job in active:
                if getattr(job, "post_exit_scan_pending", False):
                    job.post_exit_scan_pending = False
                    job.post_exit_scan_complete = True
                job.initial_scan_pending = False
            return rows
        rows = {}
        for job in active:
            rows.update(self.table.refresh_members(
                job.family.members,
                namespace_inode=job.family.private_pid_namespace_inode,
            ))
        now = time.monotonic()
        force_scan = any(
            getattr(job, "initial_scan_pending", False)
            or getattr(job, "post_exit_scan_pending", False)
            for job in active
        )
        if active and (force_scan or now >= self._next_full_process_scan):
            rows.update(self.table.snapshot())
            self._next_full_process_scan = (
                time.monotonic() + FULL_PROCESS_SCAN_INTERVAL_SECONDS)
            for job in active:
                job.initial_scan_pending = False
                if getattr(job, "post_exit_scan_pending", False):
                    job.post_exit_scan_pending = False
                    job.post_exit_scan_complete = True
        return rows

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
        mutation = cap.purpose in {"mutation", "mutation_broker"}
        armable_wall_s = request.get("armable_wall_s")
        if armable_wall_s is not None:
            if (cap.purpose not in {"mutation", "mutation_broker"}
                    or type(armable_wall_s) not in {int, float}
                    or not math.isfinite(armable_wall_s)
                    or not 0 < armable_wall_s <= cap.limits.wall_seconds):
                raise ValueError("invalid mutation arm budget")
        cwd = (Path("/sf/unit") if mutation and cap.purpose == "mutation" and self.backend == "bwrap"
               else Path("/") if mutation and self.backend == "bwrap"
               else Path(request.get("cwd") or cap.workspace).resolve(strict=True))
        limits = asdict(cap.limits)
        if not cap.limits.kill_on_output:
            del limits["kill_on_output"]
        shared_namespace = self.namespace_network != "host"
        bwrap_starts_session = (
            self.backend == "bwrap" and cap.sandboxed
            and (mutation or not shared_namespace)
        )
        spec = {"argv": argv, "cwd": str(cwd), "env": dict(cap.env), "limits": limits,
                "memory_supervised": True, "new_session": False}
        # Popen or bwrap starts the session; sf_launch must not call setsid a second time.
        # Keep reparented descendants in that launch session so cleanup can still find them.
        # The launcher is stdlib-only and isolated (-I): exec the real interpreter, not a venv
        # symlink, so the sandbox only has to admit the resolved runtime roots.
        if mutation and self.backend == "bwrap":
            launcher_path = "/sf/boot/sf_launch.py"
        elif mutation:
            launcher_path = str(Path(request["mounts"]["boot"]) / "sf_launch.py")
        else:
            launcher_path = str(Path(__file__).with_name("sf_launch.py").resolve())
        launch = [os.path.realpath(sys.executable), "-I", launcher_path,
                  json.dumps(spec, separators=(",", ":"))]
        self.next_id += 1
        ident = f"j{self.next_id}"
        cgroup = (_command_cgroup(ident, cap.limits.nproc,
                                  memory_bytes=cap.limits.memory_bytes)
                  if cap.sandboxed and self.backend == "bwrap" else None)
        cgroup_memory_configured = _cgroup_memory_limit_configured(
            cgroup, cap.limits.memory_bytes)
        launch_memory_mechanism = (
            "supervised_rss" if sys.platform == "darwin" else
            "cgroup_memory_max" if cgroup_memory_configured else "rlimit_data")
        spec["memory_limit_mechanism"] = launch_memory_mechanism
        launch[-1] = json.dumps(spec, separators=(",", ":"))
        command, environment = capability_command(
            launch, cap, platform_backend=self.backend,
            shared_namespace=shared_namespace,
            mutation_mounts=request.get("mounts") if mutation else None)
        if private_loopback:
            # A custom tool may narrow a full-network run. Keep the supervised
            # PID namespace, but give this command a disconnected loopback netns.
            command.insert(1, "--unshare-net")
        final_command = tuple(command)
        memory_mechanism = launch_memory_mechanism
        memory_reader = memory_writer = None
        collector = None
        process = None
        try:
            if cgroup_memory_configured:
                memory_reader, memory_writer = os.pipe()
                os.set_inheritable(memory_writer, True)
            collector = JobOutputCollector(self.directory / f"{ident}.output", capacity=1 << 20)
            pass_fds = ()
            if cgroup is not None:
                enter = [os.path.realpath(sys.executable), "-I",
                         str(Path(__file__).with_name("sf_launch.py").resolve()),
                         "--cgroup-enter", str(cgroup), "--memory-budget",
                         str(cap.limits.memory_bytes)]
                if memory_writer is not None:
                    enter.extend(("--memory-result-fd", str(memory_writer)))
                    pass_fds = (memory_writer,)
                command = [*enter, "--", *command]
            supplied = request.get("stdin")
            if mutation and supplied is not None:
                fd, staged = tempfile.mkstemp(prefix="mutation-stdin-", dir=self.directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(base64.b64decode(supplied, validate=True))
                        stream.flush()
                        os.fsync(stream.fileno())
                    with open(staged, "rb") as input_file:
                        process = subprocess.Popen(command, stdin=input_file, stdout=subprocess.PIPE,
                                                   stderr=subprocess.DEVNULL, env=environment, close_fds=True,
                                                   pass_fds=pass_fds,
                                                   start_new_session=not bwrap_starts_session)
                finally:
                    os.unlink(staged)
            else:
                with tempfile.TemporaryFile(dir=self.directory) as input_file:
                    if supplied is not None:
                        input_file.write(base64.b64decode(supplied, validate=True))
                        input_file.seek(0)
                    process = subprocess.Popen(command, stdin=input_file if supplied is not None else subprocess.DEVNULL,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               env=environment, close_fds=True, pass_fds=pass_fds,
                                               start_new_session=not bwrap_starts_session)
            if memory_writer is not None:
                os.close(memory_writer)
                memory_writer = None
                marker = _read_memory_join_result(memory_reader)
                os.close(memory_reader)
                memory_reader = None
                reported = {b"C": "cgroup_memory_max", b"D": "rlimit_data"}.get(marker)
                if reported is None:
                    raise RuntimeError("command did not report its cgroup memory join result")
                memory_mechanism = reported
            if sys.platform.startswith("linux") and memory_mechanism == "rlimit_data":
                memory_mechanism = "rlimit_data+rss_supervisor"
        except BaseException:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if process.stdout is not None:
                    process.stdout.close()
            for fd in (memory_reader, memory_writer):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            if collector is not None:
                collector.close()
            if cgroup is not None:
                _remove_command_cgroup(cgroup)
            raise
        job = ManagedProcess(ident, process, collector, cap, self.table, cgroup,
                             memory_limit_mechanism=memory_mechanism,
                             armable_wall_s=armable_wall_s)
        self.jobs[ident] = job
        if mutation:
            if request.get("_mutation_fault") == "after_spawn_before_identity":
                if sys.platform == "linux":
                    self._wait_for_final_exec(process, final_command)
                os._exit(97)
            namespace = None
            command_digest = None
            if sys.platform == "linux":
                try:
                    row, namespace, command_digest = self._wait_for_final_exec(process, final_command)
                except (OSError, RuntimeError, ValueError) as error:
                    job.stop(0)
                    raise RuntimeError("mutation launch final-exec identity unavailable") from error
            else:
                row = self.table.identity(process.pid)
            if row is None:
                job.stop(0)
                raise RuntimeError("mutation launch identity unavailable")
            try:
                boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            except OSError:
                boot_id = ""
            identity = {"pid": row.pid, "start": list(row.start), "birth": list(row.start),
                        "boot_id": boot_id, "pgid": row.pgid, "sid": row.sid,
                        "ns": namespace if sys.platform == "linux" else self.table.pid_namespace(row.pid)}
            if sys.platform == "linux":
                # The launcher and final sandbox share a PID, but not a command line.
                identity["cmdline_sha256"] = command_digest
                if not identity["ns"]:
                    job.stop(0)
                    raise RuntimeError("mutation launch PID namespace identity unavailable")
            return {"id": ident, "identity": identity,
                    "memory_limit_mechanism": memory_mechanism,
                    "cgroup_leaf_pattern": _cgroup_leaf_pattern_status()}
        return {"id": ident, "memory_limit_mechanism": memory_mechanism,
                "cgroup_leaf_pattern": _cgroup_leaf_pattern_status()}

    def _wait_for_final_exec(self, process, final_command):
        """Wait until the supervised PID exposes the argv that will be identity-pinned."""
        expected = hashlib.sha256(
            b"\0".join(os.fsencode(value) for value in final_command) + b"\0").hexdigest()
        initial = self.table.identity(process.pid)
        if initial is None:
            raise RuntimeError("mutation launch identity unavailable")
        deadline = time.monotonic() + 10.0
        last_error = None
        while time.monotonic() < deadline:
            # Do not poll Popen here: poll() reaps the leader and releases its PID for reuse.
            row = self.table.identity(process.pid)
            if row is None or row.start != initial.start or row.zombie:
                raise RuntimeError("mutation launch exited before its final exec identity")
            try:
                command_digest = process_cmdline_digest(row.pid)
                if command_digest == expected:
                    namespace = self.table.pid_namespace(row.pid)
                    confirmed = self.table.identity(row.pid)
                    if (namespace and confirmed is not None and not confirmed.zombie
                            and confirmed.start == row.start
                            and self.table.pid_namespace(row.pid) == namespace
                            and process_cmdline_digest(row.pid) == expected):
                        return confirmed, namespace, expected
            except (OSError, RuntimeError, ValueError) as error:
                # /proc can be transiently empty while exec replaces the launcher image.
                last_error = error
            time.sleep(0.01)
        raise RuntimeError("mutation launch command line did not reach its final exec") from last_error

    def dispatch(self, request):
        op = request["op"]
        if op == "start":
            return self.start(request)
        if op == "arm":
            job = self.jobs[request["id"]]
            if job.armed or job.cleanup is not None or job.stop_at is not None:
                raise ValueError("mutation cannot be armed")
            deadline = request.get("deadline")
            if deadline is not None and (type(deadline) not in (int, float)
                                         or not math.isfinite(deadline)):
                raise ValueError("invalid mutation hook trigger deadline")
            now = time.monotonic()
            ready_at = request.get("ready_at", now)
            if (type(ready_at) not in (int, float) or not math.isfinite(ready_at)
                    or ready_at > now):
                raise ValueError("invalid mutation hook readiness time")
            job.started = ready_at
            remaining = (job.armable_wall_s if deadline is None else
                         min(job.armable_wall_s, deadline - job.started))
            job.wall_seconds = max(0.0, remaining)
            job.armed = True
            expired = remaining <= 0 or now >= job.started + job.wall_seconds
            if expired:
                job.timed_out = True
                job.timeout_at = now
                job.stop(0)
            return {"started": job.started, "expired": expired}
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

    def reply(self, request):
        has_rid = "rid" in request
        rid = request.pop("rid", None)
        try:
            response = self.dispatch(request)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        if has_rid:
            response["rid"] = rid
        return response

    def _queue_replies(self):
        while b"\n" in self.incoming:
            line, _, rest = self.incoming.partition(b"\n")
            self.incoming = bytearray(rest)
            if len(line) + 1 > 1 << 20:
                raise ValueError("execution control request exceeds 1 MiB")
            try:
                response = self.reply(json.loads(line))
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                response = {"error": f"{type(error).__name__}: {error}"}
            self.outgoing.extend(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        if len(self.incoming) > 1 << 20:
            raise ValueError("execution control request exceeds 1 MiB")

    def _unattributed(self, orphans):
        """Return reparented children that still belong to no job after a fresh look.

        An intermediate process can exit between two sweeps before its child was recorded, so
        the child is reparented here with no recorded parent. Re-snapshot once and attribute it
        by process group or session to a live job member before treating it as an escape.
        """
        if not orphans:
            return []
        rows = self.table.snapshot()
        for job in self.jobs.values():
            if job.cleanup is None:
                job.family.update(rows)
        own = (os.getpgrp(), os.getsid(0))
        escaped = []
        for row in orphans:
            current = rows.get(row.pid)
            if current is None or current.start != row.start or current.zombie:
                continue  # gone, reused or exiting: nothing to contain
            if not self._attribute_orphan(current, own):
                escaped.append(current)
        return escaped

    def _attribute_orphan(self, row, own) -> bool:
        for job in self.jobs.values():
            if job.cleanup is not None:
                continue
            members = job.family.members
            if row.pid in members:
                return True
            leader = members.get(job.family.leader)
            if leader is None or row.start < leader.start:
                continue
            for member in members.values():
                if member.pid == row.pid:
                    continue
                shares_group = row.pgid == member.pgid and row.pgid not in own
                shares_session = row.sid == member.sid and row.sid not in own
                if shares_group or shares_session:
                    members[row.pid] = row
                    return True
        return False

    def serve(self):
        connected = True
        try:
            while True:
                rows = self._sample_rows()
                if self.namespace_network != "host":
                    leaders = {j.process.pid for j in self.jobs.values()}
                    for job in self.jobs.values():
                        if job.cleanup is None:
                            job.family.update(rows)
                    recorded = {pid for j in self.jobs.values() for pid in j.family.members}
                    orphans = []
                    for row in rows.values():
                        if row.ppid != os.getpid() or row.pid in leaders:
                            continue
                        if row.zombie:
                            try:
                                os.waitpid(row.pid, os.WNOHANG)
                            except ChildProcessError:
                                pass
                        elif row.pid not in recorded:
                            orphans.append(row)
                    for row in self._unattributed(orphans):
                        # A double-fork between sweeps cannot be attributed to
                        # one command. Stop the entire session, fail closed.
                        current = self.table.identity(row.pid)
                        if current is not None and current.start == row.start:
                            # serve() alone reaps this direct child; its PID stays
                            # reserved until that waitpid, including after exit.
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
                for key, mask in events:
                    if key.data == "owner" and mask & selectors.EVENT_READ:
                        # Only the engine owns the write end. Any event is EOF or a
                        # protocol violation, and both must fail closed for the session.
                        try:
                            os.read(self.owner_fd, 1)
                        except BlockingIOError:
                            continue
                        self.selector.unregister(self.owner_fd)
                        os.close(self.owner_fd)
                        self.owner_fd = -1
                        self._begin_close()
                        continue
                    if mask & selectors.EVENT_READ:
                        try:
                            chunk = self.control.recv(65536)
                        except ConnectionResetError:
                            chunk = b""
                        if not chunk:
                            connected = False
                            self.selector.unregister(self.control)
                            self._begin_close()
                            continue
                        self.incoming.extend(chunk)
                        self._queue_replies()
                    if connected and mask & selectors.EVENT_WRITE and self.outgoing:
                        try:
                            count = self.control.send(self.outgoing)
                            del self.outgoing[:count]
                        except (BrokenPipeError, ConnectionResetError):
                            connected = False
                            self.selector.unregister(self.control)
                            self._begin_close()
                if connected:
                    self.selector.modify(self.control, selectors.EVENT_READ |
                                         (selectors.EVENT_WRITE if self.outgoing else 0))
        finally:
            # On an internal failure, attempt cleanup before surfacing the error.
            deadline = time.monotonic() + 3
            for job in self.jobs.values():
                job.stop(0)
            while any(j.cleanup is None for j in self.jobs.values()) and time.monotonic() < deadline:
                rows = self._sample_rows()
                for job in self.jobs.values():
                    job.tick(rows)
                time.sleep(0.01)
            for job in self.jobs.values():
                job.collector.close()
                job.remove_cgroup()
            self.selector.close()
            self.control.close()
            if self.owner_fd >= 0:
                os.close(self.owner_fd)


def main():
    directory = Path(sys.argv[1])
    backend, network = sys.argv[2:4]
    owner_fd = int(sys.argv[4])
    if network == "loopback":
        # Bubblewrap configures private lo before dropping its setup privileges.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            flags = fcntl.ioctl(sock.fileno(), 0x8913, struct.pack("16sh", b"lo", 0))
            if not struct.unpack_from("h", flags, 16)[0] & 1:
                raise RuntimeError("persistent execution namespace has no loopback interface")
    Supervisor(socket.socket(fileno=0), directory, backend, network, owner_fd).serve()


if __name__ == "__main__":
    main()
