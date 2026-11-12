"""Process identities and descendant tracking, including setsid escapees.

macOS has no PID namespace: an orphan that escapes between sweeps can be
missed. Start timestamps protect signals against PID reuse, not that race.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import logging
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
                process._stateframe_pidfd_finalizer = weakref.finalize(process, os.close, pidfd)
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


class _UniqueInfo(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16), ("uniqueid", ctypes.c_uint64),
                ("parent_uniqueid", ctypes.c_uint64), ("idversion", ctypes.c_int32),
                ("reserved2", ctypes.c_uint32), ("reserved3", ctypes.c_uint64),
                ("reserved4", ctypes.c_uint64)]


class _BSDInfoWithUnique(ctypes.Structure):
    _fields_ = [("bsd", _BSDInfo), ("unique", _UniqueInfo)]


MAC_KERNEL_SIGNAL_PATH = "pid-generation kernel signal"
MAC_FALLBACK_SIGNAL_PATH = "fallback: identity re-verified os.kill (residual PID-reuse race)"


def _libproc():
    """Load libproc at use time so absent symbols do not become cached state."""
    return ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


def signal_mac_verified_process(expected: ProcessIdentity, sig: signal.Signals) -> str | bool:
    """Return the verified signal path, or False when the process is gone or changed."""
    if sys.platform != "darwin" or expected.pid <= 0 or expected.zombie:
        return False
    try:
        lib = _libproc()
    except OSError as error:
        raise ContainmentUnverified("macOS libproc unavailable: cannot read process identity") from error
    proc_pidinfo = getattr(lib, "proc_pidinfo", None)
    if proc_pidinfo is None:
        raise ContainmentUnverified("macOS libproc unavailable: cannot read process identity")
    proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                             ctypes.c_void_p, ctypes.c_int]

    def read_identity() -> _BSDInfoWithUnique | None:
        info = _BSDInfoWithUnique()
        count = proc_pidinfo(expected.pid, 18, 0, ctypes.byref(info), ctypes.sizeof(info))
        if count == 0 and ctypes.get_errno() in (errno.ESRCH, errno.ENOENT):
            return None
        if count != ctypes.sizeof(info):
            raise ContainmentUnverified("cannot read macOS process birth and PID version together")
        return info

    info = read_identity()
    if info is None or (info.bsd.pid != expected.pid or info.bsd.uid != os.getuid()
            or (info.bsd.start_sec, info.bsd.start_usec) != expected.start
            or info.unique.idversion == 0 or info.bsd.status == 5):
        return False

    def identity_still_matches() -> bool:
        again = read_identity()
        if again is None or (again.bsd.pid != info.bsd.pid or again.bsd.uid != info.bsd.uid
                or (again.bsd.start_sec, again.bsd.start_usec) != (info.bsd.start_sec, info.bsd.start_usec)
                or again.bsd.status != info.bsd.status
                or again.unique.idversion != info.unique.idversion):
            return False
        return True

    if int(sig) == 0:
        # proc_signal_with_audittoken rejects signal 0 with EINVAL; verify identity without calling it.
        return MAC_KERNEL_SIGNAL_PATH if identity_still_matches() else False

    def signal_after_recheck() -> str | bool:
        if not identity_still_matches():
            return False
        try:
            os.kill(expected.pid, sig)
        except ProcessLookupError:
            return False
        return MAC_FALLBACK_SIGNAL_PATH

    proc_signal = getattr(lib, "proc_signal_with_audittoken", None)
    if proc_signal is None:
        return signal_after_recheck()
    audit_token = (ctypes.c_uint32 * 8)()
    audit_token[5] = expected.pid
    audit_token[7] = info.unique.idversion & 0xffffffff
    proc_signal.argtypes = [ctypes.c_void_p, ctypes.c_int]
    result = proc_signal(ctypes.byref(audit_token), int(sig))
    if result in (errno.ENOTSUP, errno.ENOSYS):
        return signal_after_recheck()
    if result in (errno.ESRCH, errno.ENOENT):
        return False
    if result != 0:
        raise OSError(result, "cannot signal verified macOS process", expected.pid)
    return MAC_KERNEL_SIGNAL_PATH


class _TaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "virtual", "resident", "user", "system", "threads_user", "threads_system")]
    _fields_ += [("counters", ctypes.c_int32 * 12)]


def process_start_now() -> tuple[int, int]:
    """The current time in ProcessTable start units, rounded up so a process born by now compares <=.

    macOS reports wall-clock (seconds, microseconds). Linux /proc reports clock ticks since boot;
    CLOCK_BOOTTIME is never behind the clock older kernels used, so the bound stays inclusive.
    """
    if sys.platform == "linux":
        ticks = os.sysconf("SC_CLK_TCK")
        return (int(time.clock_gettime(time.CLOCK_BOOTTIME) * ticks) + 1, 0)
    now = time.time_ns() // 1000
    return (now // 1_000_000, now % 1_000_000 + 1)


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
                               (int(fields[19]), 0), fields[0] in {"Z", "X", "x"})

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
        except FileNotFoundError:
            return 0
        except OSError as error:
            if isinstance(error, ProcessLookupError) or error.errno == errno.ESRCH:
                return 0
            raise
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")

    def page_table_bytes(self, row: ProcessIdentity) -> int:
        """Return a live Linux process's page-table allocation from procfs."""
        if sys.platform != "linux":
            return 0
        current = self.identity(row.pid)
        if current is None or current.start != row.start or current.zombie:
            return 0
        try:
            status = (Path("/proc") / str(row.pid) / "status").read_text(
                encoding="ascii", errors="replace")
        except FileNotFoundError:
            return 0
        except OSError as error:
            if isinstance(error, ProcessLookupError) or error.errno == errno.ESRCH:
                return 0
            raise

        current = self.identity(row.pid)
        if current is None or current.start != row.start or current.zombie:
            return 0

        has_rss = False
        for line in status.splitlines():
            name, separator, value = line.partition(":")
            if name == "VmRSS" and separator:
                has_rss = True
            if name != "VmPTE" or not separator:
                continue
            fields = value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise RuntimeError("cannot measure execution descendant page-table memory")
            try:
                kibibytes = int(fields[0], 10)
            except ValueError as error:
                raise RuntimeError(
                    "cannot measure execution descendant page-table memory") from error
            if kibibytes < 0:
                raise RuntimeError("cannot measure execution descendant page-table memory")
            return kibibytes * 1024
        if not has_rss:
            # Linux omits all Vm* fields for tasks without an address space.
            return 0
        raise RuntimeError("Linux process status omits VmPTE")

    def accounted_memory_bytes(self, row: ProcessIdentity) -> int:
        """Return RSS plus page-table memory for one tracked process."""
        return self.resident_bytes(row) + self.page_table_bytes(row)

    def resident_memory_limit_breach(self, live: list[ProcessIdentity], memory_bytes: int) -> str:
        """Return the shared aggregate memory breach reason for one launch unit."""
        if sum(self.accounted_memory_bytes(row) for row in live) > memory_bytes:
            return "aggregate resident-memory limit exceeded"
        return ""

    def limit_breach(self, live: list[ProcessIdentity], limits) -> str:
        """Apply the same aggregate limits to every supervised containment unit."""
        breach = self.resident_memory_limit_breach(live, limits.memory_bytes)
        if breach:
            return breach
        if len(live) > limits.nproc:
            return "process limit exceeded"
        return ""

    @staticmethod
    def pid_namespace(pid: int) -> str | None:
        try:
            return os.readlink(f"/proc/{pid}/ns/pid")
        except FileNotFoundError:
            return None

    @staticmethod
    def pin_pid_namespace(pid: int, namespace: str) -> PinnedNamespace | None:
        """Hold ``pid``'s PID namespace open if it is still ``namespace`` (readlink form)."""
        handle = _open_pid_namespace(pid)
        if handle is not None and f"pid:[{handle.inode}]" != namespace:
            handle.close()
            return None
        return handle

    @staticmethod
    def pin_pid_namespace_inode(pid: int, inode: int) -> PinnedNamespace | None:
        """Hold ``pid``'s PID namespace open if its inode is still ``inode``.

        A namespace inode is reused once the namespace dies; comparing by inode is only sound
        while a handle like this keeps the namespace alive.
        """
        handle = _open_pid_namespace(pid)
        if handle is not None and handle.inode != inode:
            handle.close()
            return None
        return handle

    @staticmethod
    def pid_namespace_inode(pid: int) -> int | None:
        """Return a process PID namespace inode for exact namespace membership checks."""
        if sys.platform != "linux":
            return None
        try:
            return os.stat(f"/proc/{pid}/ns/pid").st_ino
        except OSError:
            return None

    @staticmethod
    def children(pid: int) -> tuple[int, ...]:
        """Read children created by every thread of a process from procfs."""
        task_root = Path("/proc") / str(pid) / "task"
        try:
            tasks = tuple(task_root.iterdir())
        except FileNotFoundError:
            return ()
        except OSError as error:
            if isinstance(error, ProcessLookupError) or error.errno == errno.ESRCH:
                return ()
            raise
        children: set[int] = set()
        for task in tasks:
            if not task.name.isdecimal():
                continue
            try:
                values = (task / "children").read_text(encoding="ascii").split()
            except FileNotFoundError:
                continue
            except OSError as error:
                if isinstance(error, ProcessLookupError) or error.errno == errno.ESRCH:
                    continue
                raise
            children.update(int(value) for value in values if value.isdecimal())
        return tuple(sorted(children))

    def refresh_members(self, members: dict[int, ProcessIdentity], *,
                        namespace_inode: int | None = None) -> dict[int, ProcessIdentity]:
        """Refresh known Linux members and discover their descendants without scanning /proc."""
        if sys.platform != "linux" or self._lib is not None:
            return self.snapshot()
        rows: dict[int, ProcessIdentity] = {}
        pending = deque((pid, None) for pid in members)
        visited: set[int] = set()
        while pending:
            pid, parent_pid = pending.popleft()
            if pid in visited:
                continue
            visited.add(pid)
            row = self.identity(pid)
            if row is None:
                continue
            expected = members.get(pid)
            if expected is not None and row.start != expected.start:
                continue
            same_namespace = (namespace_inode is not None
                              and self.pid_namespace_inode(pid) == namespace_inode)
            if parent_pid is not None and row.ppid != parent_pid and not same_namespace:
                continue
            rows[pid] = row
            child_pids = self.children(pid)
            current = self.identity(pid)
            if current is None or current.start != row.start:
                continue
            for child_pid in child_pids:
                if child_pid not in visited:
                    pending.append((child_pid, pid))
        return rows

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


class PinnedNamespace:
    """An open handle on a PID namespace; close() releases it (also on garbage collection)."""

    def __init__(self, descriptor: int):
        self.descriptor: int | None = descriptor
        self.inode = os.fstat(descriptor).st_ino
        self._finalizer = weakref.finalize(self, os.close, descriptor)

    def close(self) -> None:
        self.descriptor = None
        self._finalizer()


def _open_pid_namespace(pid: int) -> PinnedNamespace | None:
    if sys.platform != "linux":
        return None
    try:
        descriptor = os.open(f"/proc/{pid}/ns/pid", os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None
    return PinnedNamespace(descriptor)


def process_cmdline_digest(pid: int) -> str:
    """Return a bounded digest of a Linux process's NUL-delimited argv."""
    if sys.platform != "linux":
        raise RuntimeError("process command identity is Linux-only")
    fd = os.open(f"/proc/{pid}/cmdline", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        value = bytearray()
        limit = 1 << 20
        while len(value) <= limit:
            block = os.read(fd, min(8192, limit + 1 - len(value)))
            if not block:
                break
            value.extend(block)
        if not value or len(value) > limit or not value.endswith(b"\0"):
            raise ValueError("process command line is empty or exceeds its bound")
        return hashlib.sha256(value).hexdigest()
    finally:
        os.close(fd)


class Descendants:
    def __init__(self, leader: int, table: ProcessTable,
                 launch_identity: LaunchIdentity | None = None, *,
                 allow_unstamped_leader: bool = False):
        self.leader = leader
        self.table = table
        self.members: dict[int, ProcessIdentity] = {}
        self.launch_leader_start: tuple[int, int] | None = None
        self.signal_errors: dict[int, str] = {}
        self.signal_paths: dict[int, str] = {}
        namespace_inode = getattr(table, "pid_namespace_inode", None)
        self.launch_namespace_inode = namespace_inode(leader) if namespace_inode else None
        self.private_pid_namespace_inode: int | None = None
        # Holds the private namespace open so its inode cannot name another sandbox later.
        self.private_pid_namespace_pin: PinnedNamespace | None = None
        # A leader PID cannot be reused until the leader is reaped, so a group or session named
        # by it is ours until then. Afterwards only processes born by the reap belong to it: a
        # reused PID's new group starts later. `reaped_bound` is the reap time when the caller
        # reports it (leader_reaped); otherwise the newest start seen while the leader was live.
        self.owned_bound: tuple[int, int] | None = None
        self.reaped_bound: tuple[int, int] | None = None
        row = table.identity(leader)
        identity_matches = (
            launch_identity is None
            or (row is not None and row.pid == launch_identity.pid
                and (row.start == launch_identity.start
                     or (allow_unstamped_leader and launch_identity.start is None)))
        )
        if row is not None and identity_matches:
            self.members[leader] = row
            if launch_identity is not None:
                self.launch_leader_start = row.start

    def leader_reaped(self) -> None:
        """Record that the caller has just reaped the leader; call once, right after the reap."""
        if self.reaped_bound is None:
            self.reaped_bound = process_start_now()

    def update(self, rows: dict[int, ProcessIdentity]) -> list[ProcessIdentity]:
        for pid in self.members.keys() - rows.keys():
            current = self.table.identity(pid)
            if current is not None:
                rows[pid] = current
        active = {pid for pid, old in self.members.items()
                  if pid in rows and rows[pid].start == old.start}
        namespace_inode = getattr(self.table, "pid_namespace_inode", None)
        if (sys.platform == "linux" and self.private_pid_namespace_inode is None
                and self.launch_namespace_inode is not None and namespace_inode is not None):
            for row in rows.values():
                if row.pid == self.leader or row.ppid not in active:
                    continue
                candidate = namespace_inode(row.pid)
                if candidate is not None and candidate != self.launch_namespace_inode:
                    pin = getattr(self.table, "pin_pid_namespace_inode", None)
                    if pin is not None:
                        handle = pin(row.pid, candidate)
                        if handle is None:
                            continue  # gone or already a different namespace
                        self.private_pid_namespace_pin = handle
                    self.private_pid_namespace_inode = candidate
                    break
        if self.private_pid_namespace_inode is not None and namespace_inode is not None:
            launched = self.launch_leader_start or getattr(self.members.get(self.leader), "start", None)
            for pid, row in rows.items():
                if namespace_inode(pid) != self.private_pid_namespace_inode:
                    continue
                if launched is not None and row.start < launched and pid not in self.members:
                    # Older than our launch: it cannot live in our namespace. Never adopt it.
                    logging.getLogger(__name__).error(
                        "refused to adopt pid %d into launch unit %d: it shares PID namespace "
                        "inode %d but started before the launch", pid, self.leader,
                        self.private_pid_namespace_inode)
                    continue
                previous = self.members.get(pid)
                if previous is None or previous.start == row.start:
                    self.members[pid] = row
                    active.add(pid)
        leader = self.members.get(self.leader)
        leader_live = leader is not None and self.leader in active and self.reaped_bound is None
        if leader_live and rows:
            newest = max(row.start for row in rows.values())
            self.owned_bound = newest if self.owned_bound is None else max(self.owned_bound, newest)
        bound = self.reaped_bound if self.reaped_bound is not None else self.owned_bound
        changed = True
        while changed:
            changed = False
            for pid, row in rows.items():
                if pid in active:
                    continue
                in_group = row.pgid == self.leader or row.sid == self.leader
                if row.ppid in active or (in_group and (leader_live or (bound is not None and row.start <= bound))):
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
                elif sys.platform == "darwin":
                    try:
                        path = signal_mac_verified_process(row, sig)
                        if not path:
                            continue
                    except ContainmentUnverified as error:
                        self.signal_errors[row.pid] = str(error)
                        continue
                    self.signal_paths[row.pid] = path
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
    if launch_identity is not None and family is not None:
        leader = family.members.get(process.pid)
        if (leader is None or (launch_identity.start is not None
                               and leader.start != launch_identity.start)):
            family = None
        elif launch_identity.start is None:
            if getattr(family, "launch_leader_start", None) != leader.start:
                family = None
            else:
                # Only a family explicitly created by the unreaped spawn path may
                # complete this partial identity; arbitrary families cannot bless
                # a PID that may already have been reused.
                launch_identity = LaunchIdentity(
                    launch_identity.pid, leader.start, launch_identity.boot_id,
                    launch_identity.pidfd)
                process._stateframe_launch_identity = launch_identity
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
