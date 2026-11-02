"""MCP supervision waits long enough to find the server's private PID namespace on a busy host.

Regression (Linux release gate under load, 11-02): require_namespace waited at most 0.2 s after initialize. One
monitor sample walks the whole host process table; under load that took longer, and discovery
failed with "cannot identify private MCP PID namespace for supervision".
"""
import time
from types import SimpleNamespace

import pytest

import kernel_mcp_driver as driver
from kernel_exec_session import ExecSessionError
from kernel_exec_tracker import ProcessIdentity

SERVER, CHILD = 100, 101


def _monitor(monkeypatch, *, sample_seconds, private=True, alive=True):
    rows = {pid: ProcessIdentity(pid, parent, pid, pid, (pid, 0), False)
            for pid, parent in ((SERVER, 1), (CHILD, SERVER))}

    class Table:
        def identity(self, pid):
            return rows.get(pid)

        def snapshot(self):
            time.sleep(sample_seconds)  # a full host process table walk on a loaded host
            return rows

        def pid_namespace(self, pid):
            return "private" if private and pid == CHILD else "host"

        def limit_breach(self, live, limits):
            return None

    process = SimpleNamespace(pid=SERVER, poll=lambda: None if alive else 0)
    monkeypatch.setattr(driver, "ProcessTable", Table)
    monkeypatch.setattr(driver, "sys", SimpleNamespace(platform="linux"))
    monitor = driver._ContainmentMonitor(process, SimpleNamespace(limits=SimpleNamespace(wall_seconds=60)))
    monitor.start()
    return monitor


def test_slow_process_table_still_finds_the_private_namespace(monkeypatch):
    # An idle host: the unscaled wait must already cover a 0.6 s sample (0.2 s did not).
    monkeypatch.setattr(driver, "_host_load", lambda: (0.0, 4), raising=False)
    monitor = _monitor(monkeypatch, sample_seconds=.6)
    try:
        monitor.require_namespace(time.monotonic() + 30)
        assert monitor.namespace == "private"
    finally:
        monitor.stop()


def test_wait_is_bounded_when_no_private_namespace_appears(monkeypatch):
    monkeypatch.setattr(driver, "namespace_wait_seconds", lambda: .3)
    monitor = _monitor(monkeypatch, sample_seconds=.01, private=False)
    started = time.monotonic()
    try:
        with pytest.raises(ExecSessionError, match="private MCP PID namespace"):
            monitor.require_namespace(time.monotonic() + 30)
        assert time.monotonic() - started < 2
    finally:
        monitor.stop()


def test_call_deadline_caps_the_wait(monkeypatch):
    monkeypatch.setattr(driver, "namespace_wait_seconds", lambda: 30)
    monitor = _monitor(monkeypatch, sample_seconds=.01, private=False)
    started = time.monotonic()
    try:
        with pytest.raises(ExecSessionError):
            monitor.require_namespace(time.monotonic() + .3)
        assert time.monotonic() - started < 2
    finally:
        monitor.stop()


def test_exited_server_fails_without_waiting_out_the_bound(monkeypatch):
    monkeypatch.setattr(driver, "namespace_wait_seconds", lambda: 30)
    monitor = _monitor(monkeypatch, sample_seconds=.01, private=False, alive=False)
    started = time.monotonic()
    try:
        with pytest.raises(ExecSessionError):
            monitor.require_namespace(time.monotonic() + 30)
        assert time.monotonic() - started < 2
    finally:
        monitor.stop()


@pytest.mark.parametrize("load,cpus,expected", [(0.0, 8, 2.0), (16.0, 8, 4.0), (1000.0, 8, 16.0)])
def test_wait_scales_with_host_load_and_is_capped(monkeypatch, load, cpus, expected):
    monkeypatch.setattr(driver, "_host_load", lambda: (load, cpus))
    assert driver.namespace_wait_seconds() == expected


def test_non_linux_hosts_do_not_wait(monkeypatch):
    monitor = SimpleNamespace(namespace=None)
    monkeypatch.setattr(driver, "sys", SimpleNamespace(platform="darwin"))
    started = time.monotonic()
    driver._ContainmentMonitor.require_namespace(monitor, time.monotonic() + 30)
    assert time.monotonic() - started < .1
