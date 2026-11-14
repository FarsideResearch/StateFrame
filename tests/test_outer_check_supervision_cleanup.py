"""Outer-check interruption preserves the supervised process-family cleanup."""

import io
import json
import os
import signal
import sys
import time

import kernel_execution_policy as policy
from stateframe import outer_checks


def test_outer_check_interruption_forwards_recorded_family(tmp_path, monkeypatch):
    family = object()
    launch_identity = object()
    supervision_finished = []
    supervisor = type("Supervisor", (), {})()
    supervisor.family = family
    supervisor.launch_identity = launch_identity
    supervisor.containment_verified = False

    def finish_supervision(**kwargs):
        supervision_finished.append(kwargs)

    supervisor.finish = finish_supervision

    class Process:
        pid = 4242
        returncode = None
        _stateframe_memory_supervisor = supervisor

        def wait(self, timeout=None):
            self.returncode = -getattr(signal, "SIGKILL", 9)
            return self.returncode

    process = Process()
    kill_calls = []

    def kill_group(child, recorded_family=None, *, launch_identity=None, **_log_fields):
        kill_calls.append((child, recorded_family, launch_identity))
        child.returncode = -getattr(signal, "SIGKILL", 9)
        return True

    def interrupt_collection(*_args, **_kwargs):
        raise InterruptedError("interrupted during RSS supervision")

    monkeypatch.setattr(policy, "require_capabilities", lambda: None)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        policy, "sandboxed_popen", lambda *_args, **_kwargs: (process, object()))
    monkeypatch.setattr(policy, "_kill_group", kill_group)
    monkeypatch.setattr(policy, "_collect", interrupt_collection)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.setattr(signal, "setitimer", lambda *_args: None)

    spec = {
        "command": f"{sys.executable} -c pass",
        "cwd": str(tmp_path),
        "deadline": time.monotonic() + 10,
        "test_env": {},
        "pytest_plugins": None,
        "test_python": None,
        "runner_kind": "generic",
        "capture_output": False,
    }
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(spec)))
    monkeypatch.setattr(sys, "stdout", captured)

    previous_environment = os.environ.copy()
    previous_path = list(sys.path)
    try:
        helper_status = outer_checks._helper()
    finally:
        os.environ.clear()
        os.environ.update(previous_environment)
        sys.path[:] = previous_path

    packets = [json.loads(line) for line in captured.getvalue().splitlines()]
    assert helper_status == 2
    assert {"error": "interrupted during RSS supervision"} in packets
    assert packets[-1] == {"cleanup_ok": True}
    assert supervision_finished == [{"timeout": 0.1, "wait_for_post_exit": False}]
    assert kill_calls == [(process, family, launch_identity)]
