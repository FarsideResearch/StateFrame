"""Process identities and descendant tracking, including setsid escapees.

macOS has no PID namespace: an orphan that escapes between sweeps can be
missed. Start timestamps protect signals against PID reuse, not that race.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import os
from pathlib import Path
import signal
import sys


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    ppid: int
    pgid: int
    sid: int
    start: tuple[int, int]
    zombie: bool


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
    def __init__(self, leader: int, table: ProcessTable):
        self.leader = leader
        self.table = table
        self.members: dict[int, ProcessIdentity] = {}
        self.signal_errors: dict[int, str] = {}
        row = table.identity(leader)
        if row is not None:
            self.members[leader] = row

    def update(self, rows: dict[int, ProcessIdentity]) -> list[ProcessIdentity]:
        for pid in self.members.keys() - rows.keys():
            current = self.table.identity(pid)
            if current is not None:
                rows[pid] = current
        active = {pid for pid, old in self.members.items()
                  if pid in rows and rows[pid].start == old.start}
        changed = True
        while changed:
            changed = False
            for pid, row in rows.items():
                if pid in active:
                    continue
                if row.ppid in active or row.pgid == self.leader or row.sid == self.leader:
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
            current = self.table.identity(row.pid)
            if current is None or current.start != row.start or current.zombie:
                continue
            try:
                os.kill(row.pid, sig)
                signalled.add(row.pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Keep the member live so the supervisor reports a survivor.
                self.signal_errors[row.pid] = "signal permission denied"
        return signalled
