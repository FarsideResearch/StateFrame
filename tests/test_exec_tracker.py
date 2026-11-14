import os
import signal
import subprocess
import sys
from types import SimpleNamespace

from _timeouts import scaled
import kernel_exec_tracker as tracker
from kernel_exec_tracker import Descendants, ProcessTable
from fixtures.process_signals import signal_owned


def test_signal_permission_failure_preserves_live_member(tmp_path, monkeypatch):
    script = tmp_path / "wait.py"
    script.write_text("import time\ntime.sleep(10)\n")
    process = subprocess.Popen([sys.executable, "-I", str(script)], start_new_session=True)
    try:
        table = ProcessTable()
        family = Descendants(process.pid, table)
        live = family.update(table.snapshot())
        assert any(row.pid == process.pid for row in live)

        held_fd = os.open(os.devnull, os.O_RDONLY)
        monkeypatch.setattr(tracker, "sys", SimpleNamespace(platform="linux"))

        def open_pidfd(pid):
            assert pid == process.pid
            return os.dup(held_fd)

        monkeypatch.setattr(os, "pidfd_open", open_pidfd, raising=False)

        def denied(_pidfd, sig):
            assert sig == signal.SIGKILL
            raise PermissionError("test boundary denies signal")

        monkeypatch.setattr(signal, "pidfd_send_signal", denied, raising=False)
        try:
            assert family.signal(live, signal.SIGKILL) == set()
        finally:
            os.close(held_fd)
        assert family.signal_errors[process.pid] == "signal permission denied"
        assert any(row.pid == process.pid for row in family.update(table.snapshot()))
    finally:
        signal_owned(process, signal.SIGKILL)
        process.wait(timeout=scaled(3))


class _NamespaceTable:
    """A table whose per-process namespace reads are counted; (pid, start) names a process."""

    def __init__(self, rows, namespaces):
        self.rows = rows
        self.namespaces = namespaces
        self.reads: list[int] = []

    def snapshot(self):
        return dict(self.rows)

    def identity(self, pid):
        return self.rows.get(pid)

    def pid_namespace_inode(self, pid):
        self.reads.append(pid)
        return self.namespaces.get(pid)


def test_private_namespace_inode_is_read_once_per_process_not_once_per_update():
    leader = tracker.ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    init = tracker.ProcessIdentity(101, 100, 100, 100, (11, 0), False)
    bystanders = {pid: tracker.ProcessIdentity(pid, 1, pid, pid, (5, 0), False) for pid in range(200, 260)}
    table = _NamespaceTable({100: leader, 101: init, **bystanders},
                            {100: 10, 101: 20, **{pid: 30 for pid in bystanders}})
    family = Descendants(100, table)
    for _ in range(4):
        family.update(table.snapshot())
    assert family.private_pid_namespace_inode == 20
    steady = len(table.reads)
    for _ in range(5):
        family.update(table.snapshot())
    assert len(table.reads) == steady, "an update re-read /proc namespaces it already knew"


def test_namespace_cache_never_names_a_recycled_pid_by_its_predecessor():
    leader = tracker.ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    init = tracker.ProcessIdentity(101, 100, 100, 100, (11, 0), False)
    table = _NamespaceTable({100: leader, 101: init}, {100: 10, 101: 20})
    family = Descendants(100, table)
    family.update(table.snapshot())
    assert family.private_pid_namespace_inode == 20
    # pid 300 first lives outside the private namespace, then the pid is reused inside it.
    outsider = tracker.ProcessIdentity(300, 1, 300, 300, (50, 0), False)
    table.rows[300] = outsider
    table.namespaces[300] = 30
    family.update(table.snapshot())
    assert 300 not in family.members
    table.rows[300] = tracker.ProcessIdentity(300, 101, 300, 300, (60, 0), False)
    table.namespaces[300] = 20
    family.update(table.snapshot())
    assert family.members[300].start == (60, 0)


def test_namespace_cache_is_bounded_by_the_live_table():
    leader = tracker.ProcessIdentity(100, 1, 100, 100, (10, 0), False)
    init = tracker.ProcessIdentity(101, 100, 100, 100, (11, 0), False)
    table = _NamespaceTable({100: leader, 101: init}, {100: 10, 101: 20})
    family = Descendants(100, table)
    family.update(table.snapshot())
    for generation in range(300):
        pid = 1000 + generation
        table.rows = {100: leader, 101: init,
                      pid: tracker.ProcessIdentity(pid, 1, pid, pid, (generation, 0), False)}
        table.namespaces[pid] = 30
        family.update(table.snapshot())
    assert len(family._namespace_inodes) <= 4 * 3 + 64 + 3
