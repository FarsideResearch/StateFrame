"""Regressions for cleanup receipts and late process identity recovery."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import kernel_environment_state as environment
import kernel_exec_tracker as tracker
import kernel_execution_policy as policy
from stateframe import outer_checks


def test_outer_helper_keeps_successful_policy_cleanup_after_collection_error(
        tmp_path, monkeypatch):
    packets = []
    kills = []
    process = SimpleNamespace(returncode=None, wait=lambda timeout: 0)

    def kill_group(target):
        assert target is process
        kills.append(target)
        if len(kills) > 1:
            raise tracker.ContainmentUnverified("leader already reaped")
        process.returncode = -signal.SIGKILL
        return True

    def collect(*_args, **_kwargs):
        policy.sandboxed_popen([], tmp_path)
        policy._kill_group(process)
        raise ValueError("collection failed after cleanup")

    spec = {"command": "true", "cwd": str(tmp_path),
            "deadline": time.monotonic() + 30, "test_env": {},
            "test_python": None, "pytest_plugins": None,
            "runner_kind": "generic", "capture_output": False}
    monkeypatch.setattr(outer_checks.sys, "stdin", io.StringIO(json.dumps(spec)))
    monkeypatch.setattr(outer_checks.tempfile, "TemporaryDirectory",
                        lambda **_kw: SimpleNamespace(name=str(tmp_path / "scratch"), cleanup=lambda: None))
    monkeypatch.setattr(outer_checks.shutil, "copytree", lambda *_a, **_k: None)
    monkeypatch.setattr(policy, "require_capabilities", lambda: None)
    monkeypatch.setattr(policy, "sandboxed_popen", lambda *_a, **_k: (process, None))
    monkeypatch.setattr(policy, "run_sandboxed", collect)
    monkeypatch.setattr(policy, "_kill_group", kill_group)
    monkeypatch.setattr(outer_checks, "print", lambda packet, **_k: packets.append(json.loads(packet)),
                        raising=False)
    original_env = dict(os.environ)
    original_handler = signal.getsignal(signal.SIGTERM)
    original_path = list(sys.path)
    try:
        assert outer_checks._helper() == 2
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        signal.signal(signal.SIGTERM, original_handler)
        sys.path[:] = original_path

    assert packets == [{"error": "collection failed after cleanup"}, {"cleanup_ok": True}]
    assert kills == [process]


def test_environment_cleanup_then_close_reuses_verified_containment(tmp_path, monkeypatch):
    launched = []

    def spawn(command, root, **_kwargs):
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        tracker.capture_launch_identity(process, session_leader=True)
        launched.append(process)
        return process, None

    monkeypatch.setattr(environment, "sandboxed_popen", spawn)
    owner = environment.EnvironmentState(tmp_path)
    ready = tmp_path / "ready"
    row = owner.launch([sys.executable, "-c",
                        "import pathlib, signal, sys, time; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "pathlib.Path(sys.argv[1]).touch(); time.sleep(30)", str(ready)])
    process = launched[0]
    try:
        until = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert ready.exists()
        cleaned = owner.cleanup(grace_seconds=.2)
        assert cleaned[0].launch_id == row.launch_id
        assert cleaned[0].state == "killed"
        closed = owner.close()
        assert closed == cleaned
        assert process.returncode is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        process.stdout.close()


def test_recovered_launch_row_builds_family_under_wait_lock(monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    real_table = tracker.ProcessTable()

    class MissingThenPresent:
        def __init__(self):
            self.lookups = 0
            self.recovery_under_lock = False

        def identity(self, pid):
            self.lookups += 1
            if self.lookups == 1:
                return None
            if self.lookups == 2:
                self.recovery_under_lock = process._waitpid_lock.locked()
            return real_table.identity(pid)

        def snapshot(self):
            return real_table.snapshot()

    table = MissingThenPresent()
    try:
        identity = tracker.capture_launch_identity(process, table, session_leader=True)
        assert identity is not None and identity.start is None
        monkeypatch.setattr(tracker, "ProcessTable", lambda: table)
        assert tracker.kill_launch_unit(process, launch_identity=identity,
                                        deadline_seconds=1) == ()
        assert table.recovery_under_lock is True
        assert tracker.recorded_launch_identity(process).start is not None
        assert process.returncode is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
