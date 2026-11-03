"""Regressions for cleanup when the launch group no longer proves containment."""

import os
import io
import json
import signal
import subprocess
import sys
import time
from functools import partial
from types import SimpleNamespace

import pytest

import capability_host_capacity as capacity
import capability_release_runner as release
import kernel_environment_state as environment
import kernel_exec_tracker as tracker
import kernel_execution_policy as policy
import kernel_job_watch as jobs
from stateframe import outer_checks


PROCESS_TABLE = tracker.ProcessTable


LAUNCHER = """
import os, signal, sys, time
path = sys.argv[1]
children = []
for escaped in (False, True):
    ready_read, ready_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        if escaped:
            os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Report readiness only after setsid, so no test signal can race it.
        os.write(ready_write, b'1')
        os.close(ready_write)
        os.close(1)
        os.close(2)
        while True: time.sleep(1)
    os.close(ready_write)
    os.read(ready_read, 1)
    os.close(ready_read)
    children.append(pid)
with open(path, 'w') as stream:
    stream.write(','.join(map(str, children)))
    stream.flush()
while True: time.sleep(1)
"""


def _alive(pid):
    row = PROCESS_TABLE().identity(pid)
    return row is not None and not row.zombie


def _launch(tmp_path, *, identity):
    pids_file = tmp_path / "pids"
    process = subprocess.Popen([sys.executable, "-c", LAUNCHER, str(pids_file)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    if identity:
        assert tracker.capture_launch_identity(process, session_leader=True) is not None
    end = time.monotonic() + 3
    contents = ""
    while time.monotonic() < end:
        if pids_file.exists():
            contents = pids_file.read_text()
            if len(contents.split(",")) == 2 and all(part.isdecimal() for part in contents.split(",")):
                break
        time.sleep(.01)
    assert contents and len(contents.split(",")) == 2
    children = tuple(int(item) for item in contents.split(","))
    return process, children


def _dispose(process, children):
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=2)
    process.stdout.close()
    process.stderr.close()


def test_missing_family_reports_unverified_even_after_group_kill(tmp_path, monkeypatch):
    class MissingTable:
        def identity(self, _pid):
            return None

    process, children = _launch(tmp_path, identity=False)
    try:
        identity = tracker.capture_launch_identity(process, MissingTable(), session_leader=True)
        assert identity is not None
        monkeypatch.setattr(tracker, "ProcessTable", MissingTable)
        with pytest.raises(tracker.ContainmentUnverified, match="descendant family"):
            tracker.kill_launch_unit(process, launch_identity=identity, deadline_seconds=.2)
        # The escaped child left the group before any signal, so cleanup must not reach it.
        assert _alive(children[1])
        until = time.monotonic() + 2
        while _alive(children[0]) and time.monotonic() < until:
            time.sleep(.01)
        assert not _alive(children[0])
        assert _alive(children[1])
    finally:
        _dispose(process, children)


def test_job_reader_cannot_reap_leader_before_group_escalation(tmp_path, monkeypatch):
    process_holder = []
    def spawn(_argv, _root, **_kwargs):
        process, children = _launch(tmp_path, identity=True)
        process_holder.append((process, children))
        return process, None

    monkeypatch.setattr(policy, "sandboxed_popen", spawn)
    owner = jobs.JobWatch(tmp_path)
    row = owner.launch_job("ignored", 30, "DONE")
    process, children = process_holder[0]
    try:
        killed = owner.job_kill(row["job_id"])
        assert killed["state"] == "killed"
        until = time.monotonic() + 2
        while _alive(children[0]) and time.monotonic() < until:
            time.sleep(.01)
        assert not _alive(children[0])
    finally:
        _dispose(process, children)


@pytest.mark.parametrize("caller", ["environment", "job", "capacity", "release"])
def test_identity_loss_is_reported_by_public_cleanup(tmp_path, monkeypatch, caller):
    process, children = _launch(tmp_path, identity=False)
    try:
        if caller == "environment":
            owner = environment.EnvironmentState(tmp_path)
            owner._processes["P1"] = SimpleNamespace(
                process=process, record=SimpleNamespace(ownership="owned", launch_id="P1"),
                cleanup_requested=False)
            monkeypatch.setattr(owner, "_reconcile_one", lambda *_a, **_k: None)
            monkeypatch.setattr(owner, "refresh", lambda: ())
            action = partial(owner.cleanup, grace_seconds=.1)
        elif caller == "job":
            owner = jobs.JobWatch(tmp_path)
            record = jobs.JobRecord("J1", time.time(), expected_s=30)
            owner._jobs["J1"] = jobs._ManagedJob(record, process, time.monotonic(), b"", time.monotonic)
            action = partial(owner.job_kill, "J1")
        elif caller == "capacity":
            action = partial(capacity.terminate_process_group, process, pgid=process.pid, grace_s=.1)
        else:
            action = partial(release._terminate_attempt, process,
                             capacity_plan=SimpleNamespace(enabled=False), process_group=None)
        error = jobs.JobWatchError if caller == "job" else tracker.ContainmentUnverified
        with pytest.raises(error, match="containment unverified"):
            action()
        assert _alive(children[1])
    finally:
        _dispose(process, children)


@pytest.mark.parametrize("already_cleaned", [False, True])
def test_outer_helper_acknowledges_process_cleanup_before_tree_removal(
        tmp_path, monkeypatch, already_cleaned):
    events = []
    class Scratch:
        name = str(tmp_path / "scratch")

        def cleanup(self):
            events.append("remove")

    process = SimpleNamespace(wait=lambda timeout: events.append("wait"),
                              returncode=0 if already_cleaned else None)
    def spawn(*_args, **_kwargs):
        return process, None

    def check(*_args, **_kwargs):
        policy.sandboxed_popen([], tmp_path)
        return SimpleNamespace(
            completed=SimpleNamespace(stderr="", stdout="", returncode=0),
            timed_out=False, output_truncated=False, process_group_killed=True)

    spec = {"command": "true", "cwd": str(tmp_path),
            "deadline": time.monotonic() + 30, "test_env": {},
            "test_python": None, "pytest_plugins": None,
            "runner_kind": "generic", "capture_output": False}
    monkeypatch.setattr(outer_checks.sys, "stdin", io.StringIO(json.dumps(spec)))
    monkeypatch.setattr(outer_checks.tempfile, "TemporaryDirectory", lambda **_kw: Scratch())
    monkeypatch.setattr(outer_checks.shutil, "copytree", lambda *_a, **_k: None)
    monkeypatch.setattr(policy, "require_capabilities", lambda: None)
    monkeypatch.setattr(policy, "sandboxed_popen", spawn)
    monkeypatch.setattr(policy, "run_sandboxed", check)
    monkeypatch.setattr(policy, "_kill_group", lambda _p: events.append("kill") or True)
    monkeypatch.setattr(outer_checks.signal, "setitimer",
                        lambda _which, seconds: events.append(("timer", seconds)))
    monkeypatch.setattr(outer_checks, "print", lambda packet, **_k: events.append(
        "ack" if "cleanup_ok" in json.loads(packet) else "result"), raising=False)
    old_env = dict(os.environ)
    old_handler = signal.getsignal(signal.SIGTERM)
    old_path = list(sys.path)
    try:
        assert outer_checks._helper() == 0
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        signal.signal(signal.SIGTERM, old_handler)
        sys.path[:] = old_path
    assert events.count("kill") == (0 if already_cleaned else 1)
    if not already_cleaned:
        assert events.index("kill") < events.index("ack")
    assert events.index("ack") < events.index("remove")
    assert ("timer", outer_checks.HELPER_PROCESS_CLEANUP_S) in events
    assert ("timer", 0) in events
    assert outer_checks.HELPER_PROCESS_CLEANUP_S < outer_checks.HELPER_CLEANUP_GRACE


def test_outer_helper_loss_reports_escaped_check_as_unverified(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    pid_file = tmp_path / "escaped-pid"
    script = (
        "import os,signal,sys,time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.setsid()\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    os.close(1); os.close(2)\n"
        "    while True: time.sleep(1)\n"
        "with open(sys.argv[1], 'w') as stream: stream.write(str(child))\n"
        "signal.signal(signal.SIGTERM, lambda *_: os._exit(2))\n"
        "while True: time.sleep(1)\n"
    )
    def launch(_argv, **kwargs):
        return original_popen([sys.executable, "-c", script, str(pid_file)], **kwargs)

    monkeypatch.setattr(outer_checks.subprocess, "Popen", launch)
    try:
        with pytest.raises(outer_checks.OuterCheckError, match="containment unverified"):
            outer_checks.run("true", tmp_path, python=sys.executable, timeout_s=.4)
        assert pid_file.exists()
        assert _alive(int(pid_file.read_text()))
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_release_cleanup_reports_leader_that_cannot_be_reaped():
    class Unreapable:
        pid = 0
        returncode = 0

        def terminate(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("unreapable", timeout)

    with pytest.raises(tracker.ContainmentUnverified, match="leader survived cleanup"):
        release._terminate_attempt(
            Unreapable(), capacity_plan=SimpleNamespace(enabled=False), process_group=None)
