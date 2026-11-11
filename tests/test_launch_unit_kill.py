"""Class guard: a timed-out launch is killed as a whole unit, never only its launcher.

Regression (foundry gate, 11-02): the MCP timeout killed the outer bwrap while its inner
child was still setting up the sandbox. The child survived, reparented away and ran the
server for an hour with no supervisor while the engine reported a clean stop.

Every cleanup helper used by a sandbox launch path is driven against two launchers: a
child still in the launcher's process group (bwrap mid-setup) and a child that already
called setsid (left the group). A planted launcher-only kill must fail the same check.
"""
import ast
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from _timeouts import scaled
import pytest

import kernel_exec_tracker as tracker
import kernel_execution_policy as policy
import kernel_probe as probe

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"

# The child reports its PID, then outlives the launcher's own SIGKILL.
LAUNCHER = """
import os, sys, time
detach = sys.argv[1] == "setsid"
pid = os.fork()
if pid == 0:
    if detach:
        os.setsid()
    # Announce only once detached: the test must never act while a "setsid" child is still
    # in the launcher's process group (a group kill would take it and look like containment).
    sys.stdout.write(str(os.getpid()) + "\\n"); sys.stdout.flush()
    time.sleep(120)
    os._exit(0)
time.sleep(120)
"""


def _alive(pid):
    row = tracker.ProcessTable().identity(pid)
    return row is not None and not row.zombie


def _launch(kind):
    process = subprocess.Popen([sys.executable, "-c", LAUNCHER, kind], stdout=subprocess.PIPE,
                               start_new_session=True)
    assert tracker.capture_launch_identity(process) is not None
    child = int(process.stdout.readline())
    return process, child


def _cleanup(process, child):
    for pid in (child, process.pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
    process.stdout.close()


def _launcher_only(process):
    """Planted violation: the pre-fix MCP cleanup."""
    process.kill()
    process.wait()
    return True


CLEANUPS = {
    "kill_launch_unit": lambda process: tracker.kill_launch_unit(
        process, launch_identity=tracker.recorded_launch_identity(process)),
    "execution_policy._kill_group": policy._kill_group,
    "probe._kill_group": probe._kill_group,
}


@pytest.mark.parametrize("kind", ["same_group", "setsid"])
@pytest.mark.parametrize("name", sorted(CLEANUPS))
def test_cleanup_leaves_no_family_member(name, kind):
    process, child = _launch(kind)
    try:
        result = CLEANUPS[name](process)
        deadline = time.monotonic() + scaled(2)
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(.02)
        # INVERTED ASSERTION: this test previously asserted that on darwin a setsid child
        # SURVIVES cleanup (result == (child,) / _kill_group returned False). That encoded the
        # 0.3.15-0.3.18 macOS cleanup bug (detached descendants were never signalled). With the
        # fix (generation-checked kernel signal in Descendants.signal), every platform and kind
        # must leave no family member.
        assert not _alive(child), f"{name} left launch child {child} ({kind}) running"
        if name == "kill_launch_unit":
            assert result == ()
    finally:
        _cleanup(process, child)


@pytest.mark.parametrize("kind", ["same_group", "setsid"])
def test_planted_launcher_only_kill_is_caught(kind):
    process, child = _launch(kind)
    try:
        _launcher_only(process)
        time.sleep(.2)
        assert _alive(child), "control: a launcher-only kill must leave the child running"
    finally:
        _cleanup(process, child)


def test_monitor_family_survivor_is_killed_after_launcher_is_gone():
    # The MCP shape: the monitor recorded the child while the launcher lived; the launcher
    # is already dead and reaped when cleanup runs.
    process, child = _launch("setsid")
    try:
        family = tracker.Descendants(process.pid, tracker.ProcessTable())
        family.update(family.table.snapshot())
        process.kill()
        process.wait()
        survivors = tracker.kill_launch_unit(
            process, family, launch_identity=tracker.recorded_launch_identity(process))
        # INVERTED ASSERTION: previously on darwin this asserted survivors == (child,) and the
        # child alive, encoding the 0.3.15-0.3.18 macOS cleanup bug. With the generation-checked
        # kernel signal fix the monitored setsid child is killed on every platform.
        assert survivors == ()
        assert not _alive(child)
    finally:
        _cleanup(process, child)


def test_unkillable_survivor_is_reported(monkeypatch):
    process, child = _launch("setsid")
    try:
        monkeypatch.setattr(tracker.Descendants, "signal", lambda self, rows, sig: set())
        survivors = tracker.kill_launch_unit(
            process, deadline_seconds=.2,
            launch_identity=tracker.recorded_launch_identity(process))
        # The detached child cannot be signalled here, so it must still be alive AND reported.
        assert _alive(child), "control: the stubbed signal must leave the setsid child running"
        assert child in survivors
    finally:
        monkeypatch.undo()
        _cleanup(process, child)


def test_kill_launch_unit_never_signals_a_reused_group_leader(monkeypatch):
    original = tracker.ProcessIdentity(43210, 1, 43210, 43210, (1, 0), False)
    reused = tracker.ProcessIdentity(43210, 1, 43210, 43210, (2, 0), False)

    class Table:
        def __init__(self):
            self.current = original

        def identity(self, _pid):
            return self.current

        def snapshot(self):
            return {43210: self.current}

    class Process:
        pid = 43210

        def kill(self):
            pass

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    table = Table()
    family = tracker.Descendants(43210, table)
    table.current = reused
    signaled = []
    monkeypatch.setattr(tracker.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(tracker.os, "killpg", lambda pid, sig: signaled.append((pid, sig)))

    assert tracker.kill_launch_unit(
        Process(), family, launch_identity=tracker.LaunchIdentity(
            43210, original.start, None)) == ()
    assert signaled == []


def test_no_family_never_adopts_a_recycled_pid(monkeypatch):
    original = tracker.ProcessIdentity(43210, 1, 43210, 43210, (1, 0), False)
    recycled = tracker.ProcessIdentity(43210, 1, 43210, 43210, (2, 0), False)

    class Table:
        def identity(self, pid):
            assert pid == 43210
            return recycled

        def snapshot(self):
            return {43210: recycled}

    class Process:
        pid = 43210
        returncode = 0  # The original launcher has been reaped.

        def kill(self):
            raise AssertionError("a reaped launcher must not be signalled")

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    calls = []
    monkeypatch.setattr(tracker, "ProcessTable", Table)
    monkeypatch.setattr(tracker.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    process = Process()
    for identity in (None, tracker.LaunchIdentity(43210, original.start, tracker._boot_id())):
        with pytest.raises(tracker.ContainmentUnverified, match="survivors unknown"):
            tracker.kill_launch_unit(process, launch_identity=identity)
    assert calls == []


def test_missing_identity_kills_only_the_unreaped_direct_child(monkeypatch):
    class Process:
        pid = 43210
        returncode = None

        def __init__(self):
            self.killed = 0

        def kill(self):
            self.killed += 1
            self.returncode = -signal.SIGKILL

        def wait(self, timeout):
            return self.returncode

        def poll(self):
            return self.returncode

    calls = []
    monkeypatch.setattr(tracker.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    process = Process()
    with pytest.raises(tracker.ContainmentUnverified, match="survivors unknown"):
        tracker.kill_launch_unit(process, launch_identity=None)
    assert process.killed == 1
    assert calls == []


def test_runner_without_group_authority_kills_direct_child():
    from stateframe.runner import _stop_group

    class Process:
        pid = 43210
        returncode = None

        def __init__(self):
            self.killed = 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed += 1
            self.returncode = -signal.SIGKILL

        def wait(self, timeout):
            return self.returncode

    process = Process()
    _stop_group(process)
    assert process.killed == 1
    assert process.returncode == -signal.SIGKILL


def test_recycled_leader_cannot_adopt_new_group_members():
    original = tracker.ProcessIdentity(43210, 1, 43210, 43210, (1, 0), False)
    recycled = tracker.ProcessIdentity(43210, 1, 43210, 43210, (2, 0), False)
    unrelated = tracker.ProcessIdentity(43211, 43210, 43210, 43210, (3, 0), False)

    class Table:
        # The real constructor records the original leader; later lookups see the recycled PID.
        def __init__(self):
            self.current = {43210: original}

        def identity(self, pid):
            return self.current.get(pid)

    table = Table()
    family = tracker.Descendants(43210, table)
    assert family.members == {43210: original}
    table.current = {43210: recycled, 43211: unrelated}
    # No live snapshot of the original leader and no reported reap: nothing anchors group 43210,
    # so neither the recycled leader nor its child is adopted.
    assert family.update({43210: recycled, 43211: unrelated}) == []
    assert family.members == {43210: original}


def test_linux_group_signal_checks_launch_pidfd(monkeypatch):
    row = tracker.ProcessIdentity(43210, 1, 43210, 43210, (1, 0), False)

    class Table:
        def identity(self, pid):
            assert pid == row.pid
            return row

    class Process:
        pid = row.pid
        returncode = None
        _waitpid_lock = threading.Lock()

    calls = []
    monkeypatch.setattr(tracker.sys, "platform", "linux")
    monkeypatch.setattr(tracker, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(tracker.os, "pidfd_open", lambda pid: 17, raising=False)
    monkeypatch.setattr(tracker.signal, "pidfd_send_signal",
                        lambda fd, sig: calls.append(("pidfd", fd, sig)), raising=False)
    monkeypatch.setattr(tracker.os, "killpg",
                        lambda pid, sig: calls.append(("group", pid, sig)))

    identity = tracker.LaunchIdentity(row.pid, row.start, "boot-a", pidfd=17)
    assert tracker.signal_launch_group(Process(), identity, signal.SIGTERM, table=Table())
    assert calls == [("pidfd", 17, 0), ("group", row.pid, signal.SIGTERM)]

    calls.clear()

    def exited(_fd, _sig):
        calls.append(("pidfd", 17, 0))
        raise ProcessLookupError()

    monkeypatch.setattr(tracker.signal, "pidfd_send_signal", exited)
    assert not tracker.signal_launch_group(Process(), identity, signal.SIGKILL, table=Table())
    assert calls == [("pidfd", 17, 0)]


def test_unreaped_zombie_leader_still_authorizes_its_group(monkeypatch):
    leader = tracker.ProcessIdentity(43210, 1, 43210, 43210, (1, 0), True)

    class Table:
        def identity(self, pid):
            assert pid == leader.pid
            return leader

    class Process:
        pid = leader.pid
        returncode = None  # Popen has not waited; the PID cannot be recycled.
        _waitpid_lock = threading.Lock()

    calls = []
    monkeypatch.setattr(tracker.os, "killpg",
                        lambda pid, sig: calls.append((pid, sig)))
    identity = tracker.LaunchIdentity(leader.pid, leader.start, tracker._boot_id())
    assert tracker.signal_launch_group(Process(), identity, signal.SIGKILL, table=Table())
    assert calls == [(leader.pid, signal.SIGKILL)]


def test_waitable_exited_leader_still_owns_its_group(monkeypatch):
    class Table:
        def identity(self, pid):
            assert pid == 43210
            return None  # macOS may no longer expose an exited direct child.

    class Process:
        pid = 43210
        returncode = None
        _waitpid_lock = threading.Lock()

    calls = []
    monkeypatch.setattr(tracker.os, "waitid", lambda *_args: object())
    monkeypatch.setattr(tracker.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    identity = tracker.LaunchIdentity(43210, (1, 0), tracker._boot_id())
    assert tracker.signal_launch_group(Process(), identity, signal.SIGKILL, table=Table())
    assert calls == [(43210, signal.SIGKILL)]

    monkeypatch.setattr(tracker.os, "waitid", lambda *_args: None)
    calls.clear()
    assert not tracker.signal_launch_group(Process(), identity, signal.SIGKILL, table=Table())
    assert calls == []


# --- structural half: every sandbox launch module routes its kills through the helper ---

def _launch_kill_violations(source: str) -> list[int]:
    """Lines that kill a launched process directly (os.killpg / Popen.kill / Popen.terminate)."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        if isinstance(func.value, ast.Name) and func.value.id == "os" and func.attr == "killpg":
            lines.append(node.lineno)
        elif func.attr in {"kill", "terminate"} and not node.args and isinstance(func.value, (ast.Name, ast.Attribute)):
            lines.append(node.lineno)
    return lines


# Modules that launch a bwrap/sandbox-exec boundary and must never kill only its launcher.
SANDBOX_LAUNCHERS = ("kernel_mcp_driver.py", "kernel_exec_session.py", "kernel_execution_policy.py",
                     "kernel_probe.py", "kernel_public_probe_execution.py")


@pytest.mark.parametrize("name", SANDBOX_LAUNCHERS)
def test_sandbox_launch_modules_kill_through_the_unit_helper(name):
    source = (ENGINE / name).read_text()
    assert "kill_launch_unit" in source, f"{name} does not use kill_launch_unit"
    assert _launch_kill_violations(source) == [], f"{name} kills a launched process directly"


def test_structural_guard_catches_planted_violation():
    planted = "import os\ndef stop(process):\n    os.killpg(process.pid, 9)\n    process.kill()\n"
    assert _launch_kill_violations(planted) == [3, 4]
