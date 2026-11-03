"""Process identities and descendant tracking, including setsid escapees.

macOS has no PID namespace: an orphan that escapes between sweeps can be
missed. Start timestamps protect signals against PID reuse, not that race.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import ctypes
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import weakref


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    ppid: int
    pgid: int
    sid: int
    start: tuple[int, int]
    zombie: bool


@dataclass(frozen=True)
class LaunchIdentity:
    """Authority captured while the direct child still owns its PID."""

    pid: int
    start: tuple[int, int] | None
    boot_id: str | None
    pidfd: int | None = None


def _boot_id() -> str | None:
    if sys.platform != "linux":
        return None
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


class ContainmentUnverified(RuntimeError):
    """The supervisor cannot prove that every launched process was contained."""


def capture_launch_identity(process, table: ProcessTable | None = None, *,
                            session_leader: bool = False) -> LaunchIdentity | None:
    """Record the leader at spawn, including an explicitly requested new session."""
    table = table or ProcessTable()
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        row = table.identity(pid)
        boot_id = _boot_id()
    except (OSError, RuntimeError, ValueError):
        return None
    if row is None and (not session_leader or getattr(process, "_waitpid_lock", None) is None
                        or getattr(process, "returncode", None) is not None):
        return None
    if row is not None and row.pgid != pid:
        return None
    if sys.platform == "linux" and not boot_id:
        return None
    pidfd = None
    if row is not None and sys.platform == "linux" and hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(pid)
        except OSError:
            # A process-table start time remains useful when pidfds are disabled
            # by the host; the unreaped child pins this PID until wait().
            pidfd = None
        if pidfd is not None:
            try:
                again = table.identity(pid)
            except (OSError, RuntimeError, ValueError):
                os.close(pidfd)
                return None
            if again is None or again.start != row.start:
                os.close(pidfd)
                return None
            try:
                weakref.finalize(process, os.close, pidfd)
            except TypeError:
                os.close(pidfd)
                return None
    identity = LaunchIdentity(pid, row.start if row is not None else None, boot_id, pidfd)
    process._stateframe_launch_identity = identity
    return identity


def recorded_launch_identity(process) -> LaunchIdentity | None:
    """Read launch authority, strengthened only while its child remains unreaped."""
    identity = getattr(process, "_stateframe_launch_identity", None)
    return identity if isinstance(identity, LaunchIdentity) else None


def _unreaped_child(pid: int) -> bool:
    """An exited child left waitable by WNOWAIT still reserves its launch PID."""
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except (ChildProcessError, OSError):
        return False


def child_exited_unreaped(process: subprocess.Popen) -> bool:
    """Observe child exit without allowing Popen.poll to release its group PID."""
    if process.returncode is not None:
        return True
    return _unreaped_child(process.pid)


def signal_launch_group(process, launch_identity: LaunchIdentity | None, sig: signal.Signals,
                        *, table: ProcessTable | None = None) -> bool:
    """Signal a session leader only while its recorded kernel identity still matches."""
    if launch_identity is None or getattr(process, "pid", None) != launch_identity.pid:
        return False
    # Popen.poll()/wait() use this lock; holding it prevents a sibling thread
    # from reaping the direct child between identity verification and killpg.
    lock = getattr(process, "_waitpid_lock", None)
    if lock is None:
        return False
    with lock:
        if getattr(process, "returncode", None) is not None:
            return False
        table = table or ProcessTable()
        try:
            boot_id = _boot_id()
            row = table.identity(launch_identity.pid)
        except (OSError, RuntimeError, ValueError):
            return False
        if boot_id != launch_identity.boot_id:
            return False
        if row is None:
            try:
                group_still_owned = os.getpgid(launch_identity.pid) == launch_identity.pid
            except ProcessLookupError:
                group_still_owned = _unreaped_child(launch_identity.pid)
            if not group_still_owned:
                return False
        elif ((launch_identity.start is not None and row.start != launch_identity.start)
              or row.pgid != launch_identity.pid):
            return False
        transient_pidfd = None
        try:
            # An unreaped zombie still reserves its PID, and the group can
            # contain live descendants. pidfd signal-zero may reject zombies.
            if row is not None and not row.zombie and sys.platform == "linux" and hasattr(os, "pidfd_open"):
                pidfd = launch_identity.pidfd
                if pidfd is None:
                    try:
                        transient_pidfd = os.pidfd_open(launch_identity.pid)
                    except OSError:
                        transient_pidfd = None
                    pidfd = transient_pidfd
                    if pidfd is not None:
                        try:
                            again = table.identity(launch_identity.pid)
                        except (OSError, RuntimeError, ValueError):
                            return False
                        if again is None or (launch_identity.start is not None
                                             and again.start != launch_identity.start):
                            return False
                if pidfd is not None and launch_identity.start is not None:
                    try:
                        signal.pidfd_send_signal(pidfd, 0)
                    except OSError:
                        return False
            os.killpg(launch_identity.pid, sig)
        except ProcessLookupError:
            return False
        finally:
            if transient_pidfd is not None:
                os.close(transient_pidfd)
    return True


class _BSDInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid",
        "svuid", "svgid", "rfu")]
    _fields_ += [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
    _fields_ += [(name, ctypes.c_uint32) for name in (
        "nfiles", "pgid", "pjobc", "tdev", "tpgid", "nice")]
    _fields_ += [("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]


class _TaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "virtual", "resident", "user", "system", "threads_user", "threads_system")]
    _fields_ += [("counters", ctypes.c_int32 * 12)]


class ProcessTable:
    def __init__(self):
        self._lib = None
        if sys.platform == "darwin":
            self._lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            self._lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
            self._lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def identity(self, pid: int) -> ProcessIdentity | None:
        if self._lib is not None:
            info = _BSDInfo()
            count = self._lib.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
            if count == 0:
                error = ctypes.get_errno()
                if error in (errno.ESRCH, errno.ENOENT):
                    return None
                raise OSError(error, "cannot inspect execution descendant", pid)
            if count != ctypes.sizeof(info):
                raise RuntimeError("incomplete macOS process identity")
            try:
                sid = os.getsid(pid)
            except ProcessLookupError:
                return None
            return ProcessIdentity(pid, info.ppid, info.pgid, sid,
                                   (info.start_sec, info.start_usec), info.status == 5)
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        fields = raw[raw.rfind(")") + 2:].split()
        return ProcessIdentity(pid, int(fields[1]), int(fields[2]), int(fields[3]),
                               (int(fields[19]), 0), fields[0] == "Z")

    def resident_bytes(self, row: ProcessIdentity) -> int:
        current = self.identity(row.pid)
        if current is None or current.start != row.start or current.zombie:
            return 0
        if self._lib is not None:
            info = _TaskInfo()
            count = self._lib.proc_pidinfo(row.pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
            if count == 0 and ctypes.get_errno() == errno.ESRCH:
                return 0
            if count != ctypes.sizeof(info):
                raise RuntimeError("cannot measure execution descendant resident memory")
            return info.resident
        try:
            fields = (Path("/proc") / str(row.pid) / "statm").read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            return 0
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")

    def limit_breach(self, live: list[ProcessIdentity], limits) -> str:
        """Apply the same aggregate limits to every supervised containment unit."""
        if sum(self.resident_bytes(row) for row in live) > limits.memory_bytes:
            return "aggregate resident-memory limit exceeded"
        if len(live) > limits.nproc:
            return "process limit exceeded"
        return ""

    @staticmethod
    def pid_namespace(pid: int) -> str | None:
        try:
            return os.readlink(f"/proc/{pid}/ns/pid")
        except FileNotFoundError:
            return None

    def snapshot(self) -> dict[int, ProcessIdentity]:
        if self._lib is not None:
            count = self._lib.proc_listallpids(None, 0)
            if count <= 0:
                raise RuntimeError("cannot enumerate processes for execution cleanup")
            pids = (ctypes.c_int * (count + 1024))()
            actual = self._lib.proc_listallpids(pids, ctypes.sizeof(pids))
            if actual <= 0 or actual >= len(pids):
                raise RuntimeError("process table changed beyond enumeration capacity")
            candidates = list(pids[:actual])
        else:
            candidates = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdecimal()]
        result = {}
        for pid in candidates:
            if pid <= 0:
                continue
            try:
                row = self.identity(pid)
            except PermissionError:
                # Other users' processes are not children of this supervisor.
                continue
            if row is not None:
                result[pid] = row
        return result


class Descendants:
    def __init__(self, leader: int, table: ProcessTable,
                 launch_identity: LaunchIdentity | None = None):
        self.leader = leader
        self.table = table
        self.members: dict[int, ProcessIdentity] = {}
        self.signal_errors: dict[int, str] = {}
        row = table.identity(leader)
        if row is not None and (launch_identity is None or
                                (row.pid == launch_identity.pid and row.start == launch_identity.start)):
            self.members[leader] = row

    def update(self, rows: dict[int, ProcessIdentity]) -> list[ProcessIdentity]:
        for pid in self.members.keys() - rows.keys():
            current = self.table.identity(pid)
            if current is not None:
                rows[pid] = current
        active = {pid for pid, old in self.members.items()
                  if pid in rows and rows[pid].start == old.start}
        leader = self.members.get(self.leader)
        leader_owned = (leader is not None and self.leader in active)
        changed = True
        while changed:
            changed = False
            for pid, row in rows.items():
                if pid in active:
                    continue
                if row.ppid in active or (leader_owned and (row.pgid == self.leader or row.sid == self.leader)):
                    # A reused leader PID must not acquire a new unrelated family.
                    previous = self.members.get(pid)
                    if previous is not None and previous.start != row.start:
                        continue
                    self.members[pid] = row
                    active.add(pid)
                    changed = True
        return [rows[pid] for pid in active if not rows[pid].zombie]

    def signal(self, rows: list[ProcessIdentity], sig: signal.Signals) -> set[int]:
        signalled = set()
        for row in rows:
            pidfd = None
            try:
                if sys.platform == "linux" and not hasattr(os, "pidfd_open"):
                    raise OSError(errno.ENOSYS, "pidfd unavailable for descendant signal")
                if sys.platform == "linux":
                    pidfd = os.pidfd_open(row.pid)
                current = self.table.identity(row.pid)
                if current is None or current.start != row.start or current.zombie:
                    continue
                if pidfd is not None:
                    signal.pidfd_send_signal(pidfd, sig)
                else:
                    self.signal_errors[row.pid] = "detached descendant signal unavailable without pidfd"
                    continue
                signalled.add(row.pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Keep the member live so the supervisor reports a survivor.
                self.signal_errors[row.pid] = "signal permission denied"
            except OSError as error:
                self.signal_errors[row.pid] = f"signal failed (errno {error.errno})"
            finally:
                if pidfd is not None:
                    os.close(pidfd)
        return signalled


def kill_launch_unit(process, family: Descendants | None = None,
                     deadline_seconds: float = 2.0, *,
                     launch_identity: LaunchIdentity | None) -> tuple[int, ...]:
    """SIGKILL a launched process and every descendant; return the PIDs that survived.

    Killing only the launcher is not containment. A bwrap launcher killed while its inner
    child is still setting up the sandbox leaves that child alive (--die-with-parent had
    not taken effect); it reparents away and execs the payload with no supervisor. The
    family is snapshotted while the launcher is still alive, so children that already
    left its process group or session are still linked by parent PID; a monitor's family
    adds anything it saw earlier. Identities guard every signal against PID reuse.
    """
    table = family.table if family is not None else ProcessTable()
    if launch_identity is None:
        family = None
    if (launch_identity is not None and family is not None
            and (family.members.get(process.pid) is None
                 or family.members[process.pid].start != launch_identity.start)):
        family = None
    if family is None and launch_identity is not None:
        lock = getattr(process, "_waitpid_lock", None)
        with lock if lock is not None else nullcontext():
            if getattr(process, "returncode", None) is None:
                current = table.identity(process.pid)
                if (current is not None and (launch_identity.start is None
                                              or current.start == launch_identity.start)
                        and current.pgid == launch_identity.pid):
                    effective_identity = launch_identity
                    if launch_identity.start is None and _boot_id() == launch_identity.boot_id:
                        effective_identity = LaunchIdentity(
                            launch_identity.pid, current.start, launch_identity.boot_id,
                            launch_identity.pidfd)
                    candidate = Descendants(process.pid, table, effective_identity)
                    if effective_identity is not launch_identity:
                        candidate.members[process.pid] = current
                    if process.pid in candidate.members:
                        family = candidate
                        if effective_identity is not launch_identity:
                            launch_identity = effective_identity
                            process._stateframe_launch_identity = effective_identity
    if family is not None:
        family.update(table.snapshot())
    try:
        signal_launch_group(process, launch_identity, signal.SIGKILL, table=table)
    except PermissionError:
        pass
    if getattr(process, "returncode", None) is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    until = time.monotonic() + deadline_seconds
    while True:
        try:
            process.wait(timeout=.02)
        except subprocess.TimeoutExpired:
            pass
        live = ([row for row in family.update(table.snapshot()) if row.pid != process.pid]
                if family is not None else [])
        if family is not None:
            family.signal(live, signal.SIGKILL)
        if not live and process.poll() is not None:
            if family is None:
                raise ContainmentUnverified(
                    "containment unverified: launch identity or descendant family unavailable; survivors unknown")
            return ()
        if time.monotonic() >= until:
            survivors = {row.pid for row in live}
            if process.poll() is None:
                survivors.add(process.pid)
            if family is None:
                raise ContainmentUnverified(
                    "containment unverified: launch identity or descendant family unavailable; survivors unknown")
            return tuple(sorted(survivors))
