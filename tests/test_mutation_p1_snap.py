"""A completed command must report only descendants still executable after cleanup."""
from __future__ import annotations

from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

import kernel_exec_tracker as tracker
from kernel_exec_supervisor import ManagedProcess


LEADER = 41234


class FakeTable:
    def __init__(self, row):
        self.rows = {row.pid: row}

    def identity(self, pid):
        return self.rows.get(pid)

    def snapshot(self):
        return self.rows.copy()

    def limit_breach(self, live, limits):
        return ""


class FakeCollector:
    total_bytes = 0

    def drain(self, fd):
        return True

    def seal(self):
        pass


class FakeProcess:
    pid = LEADER
    returncode = 0

    def __init__(self):
        self.stdout = SimpleNamespace(fileno=lambda: 1, close=lambda: None)

    def poll(self):
        return self.returncode


def make_job():
    row = tracker.ProcessIdentity(LEADER, 1, LEADER, LEADER, (100, 0), False)
    table = FakeTable(row)
    cap = SimpleNamespace(limits=SimpleNamespace(kill_on_output=False, output_bytes=100,
                                                 wall_seconds=60))
    job = ManagedProcess("j1", FakeProcess(), FakeCollector(), cap, table)
    job.stop(0)
    job.kill_at -= 3
    return job, table


def test_recheck_after_sigkill_does_not_report_a_dead_survivor(monkeypatch):
    job, table = make_job()

    def signal_member(rows, sig):
        if sig == signal.SIGKILL:
            table.rows.clear()
        return set()

    monkeypatch.setattr(job.family, "signal", signal_member)
    job.tick(table.snapshot())

    assert job.cleanup is not None
    assert job.cleanup["ok"] is True
    assert job.cleanup["survivors"] == ()


def test_recheck_keeps_a_live_unsignalled_descendant_failed(monkeypatch):
    job, table = make_job()
    monkeypatch.setattr(job.family, "signal", lambda rows, sig: set())
    job.tick(table.snapshot())

    assert job.cleanup is not None
    assert job.cleanup["ok"] is False
    assert job.cleanup["survivors"] == (LEADER,)


def test_uninspectable_descendant_is_a_failed_containment(monkeypatch):
    job, table = make_job()

    def lose_inspection(rows, sig):
        if sig == signal.SIGKILL:
            table.rows.clear()
        return set()

    def unreadable_identity(pid):
        raise PermissionError("cannot inspect descendant")

    monkeypatch.setattr(job.family, "signal", lose_inspection)
    monkeypatch.setattr(table, "identity", unreadable_identity)
    job.tick({LEADER: job.family.members[LEADER]})

    assert job.cleanup is not None
    assert job.cleanup["ok"] is False
    assert job.cleanup["survivors"] == (LEADER,)
    assert "unverified" in job.cleanup["detail"]


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_linux_terminal_process_states_are_not_executable(monkeypatch, tmp_path, state):
    if tracker.sys.platform != "linux":
        pytest.skip("Linux /proc state codes")
    proc = tmp_path / "proc"
    stat = proc / str(LEADER) / "stat"
    stat.parent.mkdir(parents=True)
    stat.write_text(f"{LEADER} (command) {state} 1 {LEADER} {LEADER} "
                    + "0 " * 15 + "100 0\n")
    real_path = Path
    monkeypatch.setattr(tracker, "Path",
                        lambda value: proc if value == "/proc" else real_path(value))

    row = tracker.ProcessTable().identity(LEADER)
    assert row is not None and row.zombie
    family = tracker.Descendants(LEADER, SimpleNamespace(identity=lambda pid: row))
    assert family.update({LEADER: row}) == []
