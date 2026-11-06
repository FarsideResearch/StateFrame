"""Fast-exiting leaders must not leave detached RSS descendants behind."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import time

import pytest

import kernel_exec_supervisor
import kernel_execution_policy as policy
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux RSS supervision")
LIMIT = 32 << 20
HOLD_SECONDS = 2.0


def _require_bwrap():
    report = policy.capability_report()
    if (not report.available or report.backend is None
            or Path(report.backend).name != "bwrap"):
        pytest.skip("Linux bwrap execution boundary is unavailable")


def _detached_script(ready: Path, survived: Path, *, mapping_bytes: int) -> str:
    """Fork twice, detach, prove the child is live, then let the leader exit."""
    return f'''import mmap, os, time
first = os.fork()
if first == 0:
    second = os.fork()
    if second:
        os._exit(0)
    os.setsid()
    for fd in (0, 1, 2):
        try: os.close(fd)
        except OSError: pass
    size = {mapping_bytes}
    mapping = mmap.mmap(-1, size,
                        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
    for offset in range(0, size, 4096):
        mapping[offset:offset + 1] = b'x'
    with open({str(ready)!r}, 'w') as stream:
        stream.write(str(os.getpid()))
    time.sleep({HOLD_SECONDS!r})
    with open({str(survived)!r}, 'w') as stream:
        stream.write('still alive after leader exit')
    os._exit(0)
os.waitpid(first, 0)
deadline = time.monotonic() + 5
while not os.path.exists({str(ready)!r}) and time.monotonic() < deadline:
    time.sleep(0.001)
'''


def _run_check(root: Path, script: str, monkeypatch,
               memory_bytes: int) -> tuple[bool, str]:
    monkeypatch.setattr(policy, "_command_cgroup", lambda *_args, **_kwargs: None)
    result = policy.run_sandboxed(
        [sys.executable, "-c", script], root,
        timeout=10, cpu_seconds=10, memory_bytes=memory_bytes,
        file_bytes=64 << 20, nproc=32,
    )
    assert result.memory_limit_mechanism == "rlimit_data+rss_supervisor"
    return result.memory_limit_killed, result.memory_limit_reason or ""


def _run_job(tmp_path: Path, script: str, monkeypatch,
             memory_bytes: int) -> tuple[int, str]:
    workspace, run_env, home, run = (tmp_path / name for name in
                                     ("workspace", "env", "home", "run"))
    for directory in (workspace, run_env, home, run):
        directory.mkdir(exist_ok=True)
    cap = shell_capability(workspace=workspace, run_env_dir=run_env,
                           scratch_home=home, network="none", writable=False)
    limits = replace(cap.limits, wall_seconds=10, cpu_seconds=10,
                     memory_bytes=memory_bytes, nproc=32)
    cap = replace(cap, purpose="job", workspace_writable=False, limits=limits)
    monkeypatch.setattr(kernel_exec_supervisor, "_command_cgroup",
                        lambda *_args, **_kwargs: None)
    with ExecSession(run) as session:
        job = session.spawn([sys.executable, "-c", script], cap)
        deadline = time.monotonic() + 8
        while True:
            exit_code = job.wait(0.05)
            if exit_code is not None:
                break
            assert time.monotonic() < deadline, "job did not finish after leader exit"
        status = job._status()
        assert status["finished"] and status["cleanup"]["ok"]
        assert status["memory_limit_mechanism"] == "rlimit_data+rss_supervisor"
        detail = status["cleanup"]["detail"]
    return exit_code, detail


@pytest.mark.parametrize("runner", ["check", "job"])
@pytest.mark.parametrize("over_budget", [False, True])
def test_fast_double_fork_is_cleaned_before_completion(
        tmp_path, monkeypatch, runner, over_budget):
    _require_bwrap()
    marker_dir = tmp_path / "home"
    marker_dir.mkdir(exist_ok=True)
    ready = marker_dir / "grandchild-ready"
    survived = marker_dir / "grandchild-survived"
    mapping_bytes = 2 * LIMIT if over_budget else 8 << 20
    memory_bytes = LIMIT if over_budget else 128 << 20
    script = _detached_script(ready, survived, mapping_bytes=mapping_bytes)
    memory_killed = False

    if runner == "check":
        memory_killed, reason = _run_check(tmp_path, script, monkeypatch, memory_bytes)
        if over_budget and memory_killed:
            assert reason == "aggregate resident-memory limit exceeded"
    else:
        _exit_code, detail = _run_job(tmp_path, script, monkeypatch, memory_bytes)
        memory_killed = "aggregate resident-memory limit exceeded" in detail

    if over_budget:
        assert ready.exists() or memory_killed
    else:
        assert ready.exists(), "detached child never reached its hold period"

    # If completion was published while the detached child lived, it writes this marker
    # after two seconds. PID namespace teardown or the launch-unit sweep must prevent it.
    time.sleep(HOLD_SECONDS + 0.15)
    assert not survived.exists(), "detached grandchild survived command completion"
