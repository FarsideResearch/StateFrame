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
import time
from pathlib import Path

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
    time.sleep(120)
    os._exit(0)
sys.stdout.write(str(pid) + "\\n"); sys.stdout.flush()
time.sleep(120)
"""


def _alive(pid):
    row = tracker.ProcessTable().identity(pid)
    return row is not None and not row.zombie


def _launch(kind):
    process = subprocess.Popen([sys.executable, "-c", LAUNCHER, kind], stdout=subprocess.PIPE,
                               start_new_session=True)
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
    "kill_launch_unit": tracker.kill_launch_unit,
    "execution_policy._kill_group": policy._kill_group,
    "probe._kill_group": probe._kill_group,
}


@pytest.mark.parametrize("kind", ["same_group", "setsid"])
@pytest.mark.parametrize("name", sorted(CLEANUPS))
def test_cleanup_leaves_no_family_member(name, kind):
    process, child = _launch(kind)
    try:
        CLEANUPS[name](process)
        deadline = time.monotonic() + 2
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not _alive(child), f"{name} left launch child {child} ({kind}) running"
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
        assert tracker.kill_launch_unit(process, family) == ()
        assert not _alive(child)
    finally:
        _cleanup(process, child)


def test_unkillable_survivor_is_reported(monkeypatch):
    process, child = _launch("setsid")
    try:
        monkeypatch.setattr(tracker.Descendants, "signal", lambda self, rows, sig: set())
        assert child in tracker.kill_launch_unit(process, deadline_seconds=.2)
    finally:
        monkeypatch.undo()
        _cleanup(process, child)


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
