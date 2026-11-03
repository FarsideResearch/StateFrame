"""Process cleanup retains group authority until descendants are stopped."""

import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import capability_host_capacity as capacity
import capability_release_runner as release
import kernel_environment_state as environment
import kernel_execution_policy as policy
import kernel_exec_tracker as tracker
import kernel_job_watch as jobs
import kernel_probe as probe
from stateframe import outer_checks


LAUNCHER = """
import os, signal, sys, time
ready_read, ready_write = os.pipe()
child = os.fork()
if child == 0:
    os.close(ready_read)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(ready_write, b'1')
    os.close(ready_write)
    os.close(1)
    os.close(2)
    while True: time.sleep(1)
os.close(ready_write)
os.read(ready_read, 1)
os.close(ready_read)
print(child, flush=True)
while True: time.sleep(1)
"""


def _alive(pid):
    row = tracker.ProcessTable().identity(pid)
    return row is not None and not row.zombie


def _launch():
    process = subprocess.Popen([sys.executable, "-c", LAUNCHER],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    assert tracker.capture_launch_identity(process) is not None
    child = int(process.stdout.readline())
    return process, child


def _dispose(process, child):
    try:
        os.kill(child, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.returncode is None:
        process.kill()
    process.wait(timeout=2)
    process.stdout.close()
    process.stderr.close()


@pytest.mark.parametrize("caller", ["environment", "job", "capacity", "release"])
def test_term_exits_leader_but_kill_reaches_child(tmp_path, caller, monkeypatch):
    process, child = _launch()
    try:
        if caller == "capacity":
            capacity.terminate_process_group(process, pgid=process.pid, grace_s=.1)
        elif caller == "release":
            release._terminate_attempt(process, capacity_plan=SimpleNamespace(enabled=False),
                                       process_group=None)
        elif caller == "environment":
            owner = environment.EnvironmentState(tmp_path)
            owner._processes["P1"] = SimpleNamespace(
                process=process, record=SimpleNamespace(ownership="owned", launch_id="P1"),
                cleanup_requested=False)
            monkeypatch.setattr(owner, "_reconcile_one", lambda *_args, **_kwargs: None)
            monkeypatch.setattr(owner, "refresh", lambda: ())
            owner.cleanup(grace_seconds=.1)
        else:
            owner = jobs.JobWatch(tmp_path)
            record = jobs.JobRecord("J1", time.time(), expected_s=30)
            owner._jobs["J1"] = jobs._ManagedJob(record, process, time.monotonic(), b"", time.monotonic)
            owner.job_kill("J1")
        until = time.monotonic() + 2
        while _alive(child) and time.monotonic() < until:
            time.sleep(.01)
        assert not _alive(child), f"{caller} reaped the leader before KILL reached child {child}"
    finally:
        _dispose(process, child)


@pytest.mark.parametrize("collector", ["policy", "probe"])
def test_closed_pipes_live_leader_is_timeout(collector):
    process = subprocess.Popen([sys.executable, "-c",
                                "import os,time; os.close(1); os.close(2); time.sleep(30)"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    assert tracker.capture_launch_identity(process) is not None
    try:
        if collector == "policy":
            result = policy._collect(process, None, time.monotonic() + .1, 1024)
            timed_out, killed = result[2], result[4]
        else:
            result = probe._collect_process(process, .1, 1024)
            timed_out, killed = result[3], result[4]
        assert timed_out is True
        assert killed is True
    finally:
        if process.returncode is None:
            process.kill()
        process.wait(timeout=2)
        process.stdout.close()
        process.stderr.close()


def test_missing_process_row_retains_spawn_session_authority():
    class MissingTable:
        def identity(self, _pid):
            return None

    process = subprocess.Popen([sys.executable, "-c", LAUNCHER],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    child = int(process.stdout.readline())
    try:
        identity = tracker.capture_launch_identity(process, MissingTable(), session_leader=True)
        assert identity is not None
        assert tracker.signal_launch_group(process, identity, signal.SIGKILL, table=MissingTable())
        process.wait(timeout=2)
        assert process.returncode == -signal.SIGKILL
        until = time.monotonic() + 2
        while _alive(child) and time.monotonic() < until:
            time.sleep(.01)
        assert not _alive(child)
    finally:
        _dispose(process, child)


def test_missing_launch_authority_reports_unknown_containment():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               start_new_session=True)
    try:
        with pytest.raises(tracker.ContainmentUnverified, match="launch identity"):
            tracker.kill_launch_unit(process, launch_identity=None)
    finally:
        if process.returncode is None:
            process.kill()
        process.wait(timeout=2)


def test_environment_cleanup_deadline_uses_real_clock(tmp_path, monkeypatch):
    process, child = _launch()
    try:
        owner = environment.EnvironmentState(tmp_path, monotonic_clock=lambda: 0.0)
        owner._processes["P1"] = SimpleNamespace(
            process=process, record=SimpleNamespace(ownership="owned", launch_id="P1"),
            cleanup_requested=False)
        monkeypatch.setattr(owner, "_reconcile_one", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(owner, "refresh", lambda: ())
        thread = threading.Thread(target=owner.cleanup, kwargs={"grace_seconds": .1}, daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "cleanup waited on a frozen observation clock"
        until = time.monotonic() + 2
        while _alive(child) and time.monotonic() < until:
            time.sleep(.01)
        assert not _alive(child)
    finally:
        _dispose(process, child)


@pytest.mark.parametrize("acknowledges", [True, False])
def test_outer_check_waits_for_cleanup_ack_or_reports_uncontained(tmp_path, monkeypatch, acknowledges):
    original_popen = subprocess.Popen
    script = (
        "import os,signal,sys,time\n"
        "def stop(*_):\n"
        "    time.sleep(.8)\n"
        + ("    print('{\"cleanup_ok\":true}', flush=True)\n" if acknowledges else "")
        + ("    sys.exit(0)\n" if acknowledges else "    os._exit(3)\n")
        + "signal.signal(signal.SIGTERM, stop)\n"
        + "while True: time.sleep(1)\n"
    )

    def launch(_argv, **kwargs):
        return original_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(outer_checks.subprocess, "Popen", launch)
    if acknowledges:
        result = outer_checks.run("python -m pytest", tmp_path, python=sys.executable,
                                  timeout_s=1.0)
        assert result["timed_out"] is True
        assert result["passed"] is False
    else:
        with pytest.raises(outer_checks.OuterCheckError, match="containment unverified"):
            outer_checks.run("python -m pytest", tmp_path, python=sys.executable,
                             timeout_s=1.0)
