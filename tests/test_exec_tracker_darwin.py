"""Darwin process visibility, with the separately measured libproc ABI."""
from dataclasses import dataclass
import ctypes
import errno
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_exec_tracker as tracker
from stateframe import parallel_host, parallel_worker
from stateframe.parallel_worker import TickProcessTable, WorkerHandle

OBSERVER = 4000
LEADER = 4001
CHILD = 4002
GRANDCHILD = 4003
ROOT = 4004
USER = 501


@dataclass
class Process:
    ppid: int
    pgid: int
    uid: int = USER
    birth: int = 100
    short_error: int | None = None
    short_size: int = 64
    returned_pid: int | None = None
    full_error: int | None = None


class Libproc:
    def __init__(self):
        self.rows = {
            pid: Process(1, pid, 0 if index < 173 else USER)
            for index, pid in enumerate(range(ROOT, ROOT + 621))
        }
        self.rows.update({
            OBSERVER: Process(1, OBSERVER),
            LEADER: Process(OBSERVER, LEADER),
            CHILD: Process(LEADER, LEADER),
            GRANDCHILD: Process(CHILD, GRANDCHILD),
        })
        self.calls = []
        self.sysctl_calls = []
        self.sysctl_error = None
        self.on_short_read = None
        self.proc_pidinfo = lambda *args: self.pidinfo(*args)
        self.proc_listallpids = lambda *args: self.listallpids(*args)
        self.sysctl = lambda *args: self.kern_proc_pid(*args)

    def kern_proc_pid(self, mib, length, buffer, size, new, new_len):
        assert length == 4 and list(mib)[:3] == [1, 14, 1]
        pid = mib[3]
        self.sysctl_calls.append(pid)
        if self.sysctl_error is not None:
            ctypes.set_errno(self.sysctl_error)
            return -1
        row = self.rows[pid]
        assert size._obj.value == 648
        struct.pack_into("=B3xi", buffer, 36, 2, pid)
        struct.pack_into("=I", buffer, 420, row.uid)
        struct.pack_into("=ii", buffer, 560, row.ppid, row.pgid)
        return 0

    def listallpids(self, buffer, size):
        if buffer is not None:
            for index, pid in enumerate(self.rows):
                buffer[index] = pid
        return len(self.rows)

    def pidinfo(self, pid, flavor, arg, pointer, size):
        self.calls.append((pid, flavor))
        row = self.rows.get(pid)
        if row is None:
            ctypes.set_errno(errno.ESRCH)
            return 0
        if flavor == 3:
            if row.full_error is not None or row.uid != USER:
                ctypes.set_errno(row.full_error or errno.EPERM)
                return 0
            info = pointer._obj
            info.pid, info.ppid, info.pgid = pid, row.ppid, row.pgid
            info.uid, info.status, info.start_sec = row.uid, 2, row.birth
            return size
        assert flavor == 13 and size == 64
        if self.on_short_read is not None:
            self.on_short_read(pid)
            row = self.rows[pid]
        if row.short_error is not None:
            ctypes.set_errno(row.short_error)
            return 0
        data = struct.pack("=4I16s8I", row.returned_pid or pid, row.ppid, row.pgid, 2,
                           b"process", 0, row.uid, 0, row.uid, 0, 0, 0, 0)
        ctypes.memmove(pointer, data, len(data))
        return row.short_size


@pytest.fixture
def lib(monkeypatch):
    lib = Libproc()
    monkeypatch.setattr(tracker, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(parallel_worker, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(parallel_host, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(tracker.ctypes, "CDLL", lambda *_args, **_kwargs: lib)
    monkeypatch.setattr(tracker.os, "getpid", lambda: OBSERVER)
    monkeypatch.setattr(tracker.os, "getuid", lambda: USER)
    monkeypatch.setattr(tracker.os, "geteuid", lambda: USER)
    monkeypatch.setattr(tracker.os, "getsid", lambda pid: lib.rows[pid].pgid)
    return lib


def _family(table=None):
    table = table if table is not None else tracker.ProcessTable()
    table.require_complete = True
    stamp = tracker.LaunchIdentity(LEADER, (100, 0), None)
    return tracker.Descendants(LEADER, table, stamp)


def _refresh(family):
    return {row.pid: row for row in family.update(family.table.refresh_members(family.members))}


def test_strict_snapshot_uses_short_info_for_173_unrelated_root_processes(lib):
    family = _family()
    assert len(lib.rows) == 625
    assert set(_refresh(family)) == {LEADER, CHILD, GRANDCHILD}
    assert sum(flavor == 13 for _, flavor in lib.calls) == 173


def test_unreadable_child_keeps_its_grandchild_and_counts_until_exit(lib):
    lib.rows[CHILD].uid = 0
    family = _family()
    live = _refresh(family)
    assert set(live) == {LEADER, CHILD, GRANDCHILD}
    assert live[CHILD].start is None and live[CHILD].uid == 0
    assert live[GRANDCHILD].start == (100, 0)

    family.leader_reaped()
    del lib.rows[LEADER], lib.rows[GRANDCHILD]
    lib.rows[CHILD].ppid = 1
    assert set(_refresh(family)) == {CHILD}
    del lib.rows[CHILD]
    assert _refresh(family) == {}


def test_replacement_between_info_calls_cannot_inherit_signal_authority(lib, monkeypatch):
    family = _family()
    _refresh(family)
    lib.rows[CHILD].uid = 0

    def replace_child(pid):
        if pid == CHILD:
            lib.rows[CHILD] = Process(ppid=1, pgid=CHILD, birth=200)

    lib.on_short_read = replace_child
    live = _refresh(family)
    assert live[CHILD].start is None and live[CHILD].uid == USER
    assert live[CHILD].ppid == 1 and live[CHILD].pgid == CHILD
    lib.rows[GRANDCHILD].ppid = 1
    assert CHILD not in _refresh(family)
    delivered = []
    monkeypatch.setattr(tracker, "signal_mac_verified_process",
                        lambda *args: delivered.append(args))
    assert family.signal([live[CHILD]], signal.SIGKILL) == set()
    assert "birth" in family.signal_errors[CHILD]
    assert delivered == []


def test_esrch_on_short_info_means_process_exited(lib):
    lib.rows[ROOT].short_error = errno.ESRCH
    family = _family()
    assert set(_refresh(family)) == {LEADER, CHILD, GRANDCHILD}
    assert ROOT not in family.table.snapshot()
    assert lib.sysctl_calls == []


@pytest.mark.parametrize("bad_reply", ["size", "pid"])
def test_malformed_short_info_cannot_manufacture_membership(lib, bad_reply):
    row = lib.rows[ROOT]
    if bad_reply == "size":
        row.short_size = 63
    else:
        row.returned_pid = CHILD
    assert set(_refresh(_family())) == {LEADER, CHILD, GRANDCHILD}
    assert lib.sysctl_calls == [ROOT]
    lib.sysctl_error = errno.EPERM
    with pytest.raises(tracker.ContainmentUnverified, match="short process information"):
        _refresh(_family())


def test_child_of_unstamped_parent_is_live_but_unverified_outside_group(lib, monkeypatch):
    lib.rows[CHILD].uid = 0
    family = _family()
    grandchild = _refresh(family)[GRANDCHILD]
    monkeypatch.setattr(tracker, "signal_mac_verified_process",
                        lambda *args: pytest.fail("unverified ancestry reached signal"))
    assert family.signal([grandchild], signal.SIGKILL) == set()
    assert "unverified parent" in family.signal_errors[GRANDCHILD]


def test_recycled_group_with_newer_leader_excludes_unstamped_stranger(lib):
    family = _family()
    _refresh(family)
    family.reaped_bound = (150, 0)
    lib.rows[LEADER] = Process(1, LEADER, birth=200)
    lib.rows[ROOT].ppid = LEADER
    lib.rows[ROOT].pgid = LEADER
    lib.rows[ROOT + 1] = Process(LEADER, LEADER)
    del lib.rows[CHILD], lib.rows[GRANDCHILD]
    assert _refresh(family) == {}
    assert not tracker.live_group_rows(family.table.snapshot(), LEADER, family.reaped_bound)


def test_signalling_refuses_unstamped_leader_even_with_unstamped_launch(lib, monkeypatch):
    lib.rows[LEADER].uid = 0
    process = SimpleNamespace(pid=LEADER, returncode=None, _waitpid_lock=threading.Lock())
    identity = tracker.LaunchIdentity(LEADER, None, None)
    delivered = []
    monkeypatch.setattr(tracker.os, "killpg", lambda *args: delivered.append(args))
    monkeypatch.setattr(tracker.os, "getpgid", lambda pid: pid)
    assert not tracker.signal_launch_group(process, identity, signal.SIGKILL)
    assert delivered == []
    with pytest.raises(tracker.ContainmentUnverified, match="birth"):
        tracker.ProcessTable().identity(LEADER)


@pytest.mark.parametrize("uid", [0, USER])
def test_denied_by_both_flavors_uses_sysctl_to_exclude_unrelated_process(lib, uid):
    lib.rows[ROOT].uid = uid
    lib.rows[ROOT].full_error = lib.rows[ROOT].short_error = errno.EPERM
    assert set(_refresh(_family())) == {LEADER, CHILD, GRANDCHILD}
    assert lib.sysctl_calls == [ROOT]


def test_sysctl_foreign_uid_can_still_be_a_family_member(lib):
    lib.rows[CHILD].uid = 0
    lib.rows[CHILD].short_error = errno.EPERM
    live = _refresh(_family())
    assert live[CHILD].uid == 0 and live[CHILD].start is None


def test_denial_without_proof_of_unrelated_membership_fails_closed(lib):
    lib.rows[CHILD].full_error = lib.rows[CHILD].short_error = errno.EPERM
    with pytest.raises(tracker.ProcessIdentityUnavailable, match="short process information"):
        _refresh(_family())
    lib.sysctl_error = errno.EPERM
    with pytest.raises(tracker.ProcessIdentityUnavailable, match="short process information"):
        _refresh(_family())


def test_same_uid_denial_with_missing_ancestor_cannot_prove_unrelated(lib):
    lib.rows[ROOT] = Process(ROOT + 1000, ROOT, full_error=errno.EPERM, short_error=errno.EPERM)
    with pytest.raises(tracker.ProcessIdentityUnavailable, match="short process information"):
        _refresh(_family())


def test_worker_tick_group_fallback_and_host_keep_unstamped_member_live(lib, tmp_path):
    table = TickProcessTable()
    stamp = tracker.LaunchIdentity(LEADER, (100, 0), None)
    process = SimpleNamespace(pid=LEADER, returncode=None)
    handle = WorkerHandle(process, LEADER, LEADER, 1, tmp_path, tmp_path / "worker.log",
                          0, 0, threading.Thread(), launch_identity=stamp)
    handle.attach_tick_table(table)
    lib.rows[CHILD].uid = 0
    with table.tick():
        assert {row.pid for row in handle._live_family(shared=True)} == {LEADER, CHILD, GRANDCHILD}
        assert table.scans == 1

    handle._family.leader_reaped()
    del lib.rows[LEADER], lib.rows[GRANDCHILD]
    lib.rows[CHILD].ppid = 1
    assert handle.group_live()
    assert handle._live_group()
    assert parallel_host._group_has_live_processes(LEADER, table)
    host = SimpleNamespace(_process_table=table)
    assert parallel_host.ParallelSupervisor._recorded_group_rows_live(host, LEADER)
    private = WorkerHandle(process, LEADER, LEADER, 1, tmp_path, tmp_path / "worker.log",
                           0, 0, threading.Thread(), launch_identity=stamp)
    assert private._live_group()
    with pytest.raises(parallel_host.ProcessInspectionDeferred):
        parallel_host._process_identity(CHILD)

    del lib.rows[CHILD]
    assert not handle.group_live()
    assert not parallel_host._group_has_live_processes(LEADER, table)
    assert not parallel_host.ParallelSupervisor._recorded_group_rows_live(host, LEADER, fresh=True)



@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS process inspection")
def test_startup_probe_and_worker_tick_with_root_processes(tmp_path):
    from stateframe.engine.kernel_safe_probe import run_probe

    table = TickProcessTable()
    with subprocess.Popen([sys.executable, "-I", "-c", "import sys; sys.stdin.read()"],
                          stdin=subprocess.PIPE, start_new_session=True) as child:
        try:
            with table.tick():
                assert any(row.uid == 0 and row.pid != os.getpid()
                           for row in table.tick_snapshot().values())
                handle = WorkerHandle(child, child.pid, child.pid, 1, tmp_path,
                                      tmp_path / "worker.log", time.time(), time.monotonic(),
                                      threading.Thread())
                handle.attach_tick_table(table)
                assert handle.poll() is None
                assert table.scans == 1
            result = run_probe([os.path.realpath(sys.executable), "-I", "-c", "print(42)"],
                               tmp_path, timeout=30)
            assert result.exit_code == 0 and result.output == b"42\n" and result.cleanup.ok
        finally:
            child.stdin.close()
