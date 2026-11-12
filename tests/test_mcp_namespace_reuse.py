"""Sandbox monitors must never adopt another sandbox's processes through a reused namespace.

Monitors recognised a sandbox's private PID namespace by its inode. Once that namespace ended the
kernel reused the inode for the next new sandbox, and a monitor SIGKILLed a neighbouring engine's
freshly started supervisor (its launcher then exited 137). Comparing by inode is sound only while
the namespace is held open, and a process older than the sandbox can never be inside it.
"""
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

from kernel_exec_tracker import Descendants, ProcessIdentity, ProcessTable
from kernel_mcp_driver import _ContainmentMonitor

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux PID namespaces")

HOST = "pid:[4026531836]"
PRIVATE = "pid:[4026539999]"


def _row(pid, ppid, start):
    return ProcessIdentity(pid, ppid, pid, pid, start, False)


# ---- real namespace handles -------------------------------------------------------------

def test_pin_holds_the_namespace_only_when_it_is_the_one_named():
    own_link = os.readlink(f"/proc/{os.getpid()}/ns/pid")
    own_inode = os.stat(f"/proc/{os.getpid()}/ns/pid").st_ino
    assert ProcessTable.pin_pid_namespace(os.getpid(), "pid:[1]") is None
    assert ProcessTable.pin_pid_namespace_inode(os.getpid(), own_inode + 1) is None
    for handle in (ProcessTable.pin_pid_namespace(os.getpid(), own_link),
                   ProcessTable.pin_pid_namespace_inode(os.getpid(), own_inode)):
        assert handle is not None and handle.inode == own_inode
        descriptor = handle.descriptor
        os.fstat(descriptor)
        handle.close()
        with pytest.raises(OSError):
            os.fstat(descriptor)


# ---- Descendants (shared by the exec, resident-memory and MCP supervisors) ---------------

class _Table:
    def __init__(self, rows, inodes, *, pin_ok=True):
        self.rows, self.inodes, self.pin_ok = rows, inodes, pin_ok
        self.pinned = []

    def identity(self, pid):
        return self.rows.get(pid)

    def pid_namespace_inode(self, pid):
        return self.inodes.get(pid)

    def pin_pid_namespace_inode(self, pid, inode):
        if not self.pin_ok:
            return None
        self.pinned.append((pid, inode))
        return SimpleNamespace(inode=inode, close=lambda: None)


def test_private_namespace_is_recorded_only_with_a_pin():
    leader, child = _row(500, 1, (10, 0)), _row(501, 500, (11, 0))
    table = _Table({500: leader, 501: child}, {500: 1, 501: 77}, pin_ok=False)
    family = Descendants(500, table)
    family.update(dict(table.rows))
    assert family.private_pid_namespace_inode is None

    table.pin_ok = True
    family.update(dict(table.rows))
    assert family.private_pid_namespace_inode == 77
    assert table.pinned == [(501, 77)]
    assert family.private_pid_namespace_pin is not None


def test_process_older_than_the_launch_is_never_adopted_by_namespace(caplog):
    leader, child = _row(500, 1, (10, 0)), _row(501, 500, (11, 0))
    stranger = _row(900, 2, (5, 0))  # started before our launch, same inode number
    table = _Table({500: leader, 501: child, 900: stranger}, {500: 1, 501: 77, 900: 77})
    family = Descendants(500, table)
    with caplog.at_level(logging.ERROR, logger="kernel_exec_tracker"):
        family.update(dict(table.rows))
    assert 900 not in family.members
    assert "refused to adopt pid 900" in caplog.text


# ---- MCP containment monitor ------------------------------------------------------------

class _McpTable:
    def __init__(self, rows, namespaces, *, pin_ok=True):
        self.rows, self.namespaces, self.pin_ok = rows, namespaces, pin_ok
        self.checked = []

    def snapshot(self):
        return dict(self.rows)

    def pid_namespace(self, pid):
        return self.namespaces.get(pid)

    def pin_pid_namespace(self, pid, namespace):
        return SimpleNamespace(close=lambda: None) if self.pin_ok else None

    def limit_breach(self, live, _limits):
        self.checked.append(sorted(row.pid for row in live))
        return ""


def _monitor(table, *, namespace, family_rows=(), launched=(50, 0)):
    monitor = _ContainmentMonitor.__new__(_ContainmentMonitor)
    monitor.process = SimpleNamespace(pid=600, returncode=None)
    monitor.cap = SimpleNamespace(limits=SimpleNamespace(wall_seconds=600))
    monitor.table = table
    monitor.launch_identity = SimpleNamespace(pid=600, start=launched)
    monitor.family = SimpleNamespace(members={600: None}, update=lambda rows: list(family_rows))
    monitor.started = time.monotonic()
    monitor._next_full_process_scan = monitor.started + 3600
    monitor.host_namespace = HOST
    monitor.namespace = namespace
    monitor.namespace_handle = None
    monitor.namespace_ready = SimpleNamespace(set=lambda: None)
    return monitor


def test_monitor_adopts_a_namespace_only_through_a_matching_handle():
    child = _row(601, 600, (60, 0))
    table = _McpTable({601: child}, {601: PRIVATE}, pin_ok=False)
    monitor = _monitor(table, namespace=None, family_rows=[child])
    monitor._sample()
    assert monitor.namespace is None

    table.pin_ok = True
    monitor._sample()
    assert monitor.namespace == PRIVATE and monitor.namespace_handle is not None
    monitor._release_namespace()
    assert monitor.namespace_handle is None


def test_monitor_refuses_and_logs_a_process_older_than_its_server(caplog):
    stranger = _row(900002, 900000, (1, 0))
    table = _McpTable({stranger.pid: stranger}, {stranger.pid: PRIVATE})
    monitor = _monitor(table, namespace=PRIVATE)
    with caplog.at_level(logging.ERROR, logger="kernel_mcp_driver"):
        assert monitor._sample() == ""
    assert table.checked == [[]]
    assert "refused to adopt pid 900002" in caplog.text
