"""Context exit must bound process cleanup before RSS completion checks."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

import kernel_exec_tracker as tracker
import kernel_execution_policy as policy


@pytest.mark.skipif(os.name == "nt", reason="launch-unit identity requires POSIX")
def test_body_exception_kills_live_child_before_bounded_cleanup(monkeypatch):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert tracker.capture_launch_identity(process, session_leader=True) is not None

    events: list[tuple[str, float | None]] = []
    real_kill_group = policy._kill_group
    real_wait = process.wait

    def observe_kill(*args, **kwargs):
        events.append(("kill", None))
        return real_kill_group(*args, **kwargs)

    def fail_finish(_process, timeout=3.0):
        events.append(("finish", timeout))
        raise RuntimeError("forced completion bookkeeping failure")

    def observe_wait(timeout=None):
        events.append(("wait", timeout))
        return real_wait(timeout=timeout)

    monkeypatch.setattr(policy, "_kill_group", observe_kill)
    monkeypatch.setattr(policy, "finish_supervision", fail_finish)
    process.wait = observe_wait
    original = RuntimeError("caller body failed")
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="caller body failed") as raised:
        with policy.supervised_process(process, timeout=0.5):
            raise original

    elapsed = time.monotonic() - started
    assert raised.value is original
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "forced completion bookkeeping failure"
    assert process.returncode == -signal.SIGKILL
    assert all(stream is None or stream.closed
               for stream in (process.stdin, process.stdout, process.stderr))
    assert elapsed < 15.0

    kill_index = next(index for index, item in enumerate(events) if item[0] == "kill")
    finish_index = next(index for index, item in enumerate(events) if item[0] == "finish")
    final_wait_index = max(index for index, item in enumerate(events) if item[0] == "wait")
    assert kill_index < finish_index < final_wait_index
    assert events[final_wait_index] == ("wait", 0.5)
