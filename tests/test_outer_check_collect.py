"""Collection keeps launch identity available through normal cleanup."""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from _timeouts import scaled
import kernel_exec_tracker as tracker
import kernel_execution_policy as policy


def test_collect_cleans_normal_process_before_reaping(tmp_path, monkeypatch):
    (tmp_path / "check.py").write_text("print('check-value=0')\n", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "check.py"], cwd=tmp_path, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    assert tracker.capture_launch_identity(process, session_leader=True) is not None
    observed = []

    def sweep(child, identity, sent_signal):
        observed.append((child is process, identity.pid, sent_signal,
                         tracker.child_exited_unreaped(child), child.returncode))
        return True

    monkeypatch.setattr(policy, "signal_launch_group", sweep)

    stdout, stderr, timed_out, truncated, cleaned = policy._collect(
        process, None, time.monotonic() + scaled(5), 4096,
    )

    assert (stdout, stderr, timed_out, truncated, cleaned, process.returncode) == (
        "check-value=0\n", "", False, False, True, 0,
    )
    assert observed == [(True, process.pid, policy.signal.SIGKILL, True, None)]


def test_collect_accepts_empty_group_after_real_signal_attempt(tmp_path):
    (tmp_path / "check.py").write_text("print('check-value=0')\n", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "check.py"], cwd=tmp_path, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    assert tracker.capture_launch_identity(process, session_leader=True) is not None

    stdout, stderr, timed_out, truncated, cleaned = policy._collect(
        process, None, time.monotonic() + scaled(5), 4096,
    )

    assert (stdout, stderr, timed_out, truncated, cleaned, process.returncode) == (
        "check-value=0\n", "", False, False, True, 0,
    )


@pytest.mark.parametrize("signalled", [False, True])
def test_collect_refuses_unverified_normal_sweep(tmp_path, monkeypatch, signalled):
    (tmp_path / "check.py").write_text("print('check-value=0')\n", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "check.py"], cwd=tmp_path, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    assert tracker.capture_launch_identity(process, session_leader=True) is not None
    monkeypatch.setattr(policy, "signal_launch_group", lambda *_args: signalled)
    monkeypatch.setattr(policy, "ProcessTable", lambda: SimpleNamespace(
        snapshot=lambda: {process.pid + 1: SimpleNamespace(pgid=process.pid, zombie=False)}))

    try:
        with pytest.raises(tracker.ContainmentUnverified,
                           match="cannot sweep completed check process group"):
            policy._collect(process, None, time.monotonic() + scaled(5), 4096)
        assert process.returncode is None
    finally:
        process.wait(timeout=scaled(2))
