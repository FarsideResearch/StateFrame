"""The Linux outer check must tolerate same-uid threads and still bound forks."""
import os
import select
import subprocess
import sys
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

import kernel_execution_policy as policy
from kernel_exec_capability import shell_capability
from kernel_exec_capability import ExecLimits
from kernel_exec_session import ExecSession
from sf_launch import _NPROC_HEADROOM

pytestmark = [pytest.mark.host_sandbox,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RLIMIT_NPROC semantics")]

THREAD_STORM_SCRIPT = ("import threading\n"
                       # Small stacks keep the storm cheap; arm64 Linux needs at least 128 KiB (PTHREAD_STACK_MIN).
                       "threading.stack_size(262144)\n"
                       "event = threading.Event()\nthreads = []\n"
                       "try:\n"
                       "    for _ in range(200):\n"
                       "        thread = threading.Thread(target=event.wait)\n"
                       "        thread.start()\n        threads.append(thread)\n"
                       "except RuntimeError:\n    print('bounded', len(threads), flush=True)\n"
                       "else:\n    print('unbounded', len(threads), flush=True)\n"
                       "finally:\n"
                       "    event.set()\n"
                       "    for thread in threads: thread.join()\n")


@pytest.fixture
def busy_same_uid():
    helper = subprocess.Popen(
        [sys.executable, "-c", "import sys, threading\n"
         "event = threading.Event()\n"
         "threads = [threading.Thread(target=event.wait, daemon=True) for _ in range(400)]\n"
         "for thread in threads: thread.start()\n"
         "print('ready', flush=True)\n"
         "sys.stdin.buffer.read(1)\n"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready, _, _ = select.select([helper.stdout], [], [], 10)
        assert ready, "same-uid thread helper did not start"
        assert helper.stdout.readline() == b"ready\n", helper.stderr.read() if helper.poll() is not None else ""
        assert len(os.listdir(f"/proc/{helper.pid}/task")) >= 401
        yield helper
    finally:
        helper.terminate()
        helper.communicate(timeout=10)


def test_check_sandbox_starts_with_many_user_threads(tmp_path, busy_same_uid):
    assert busy_same_uid.poll() is None
    result = policy.run_sandboxed(["/bin/sh", "-c", "echo ok"], tmp_path, timeout=60)
    assert result.completed.returncode == 0, result.completed.stderr
    assert "ok" in result.completed.stdout
    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    with ExecSession(tmp_path / "run") as session:
        command = session.run(["/bin/sh", "-c", "echo session-ok"], cap)
    assert command.exit_code == 0 and b"session-ok" in command.output, command


def test_check_sandbox_still_bounds_new_processes(tmp_path, busy_same_uid, monkeypatch):
    # The budget applies to processes the check creates: a fork loop beyond it must be refused.
    # Force the mandatory RLIMIT fallback so a delegated cgroup cannot mask a host-count regression.
    monkeypatch.setattr(policy, "_command_cgroup", lambda ident, nproc: None)
    script = ("import os,sys\nn=0\ntry:\n    for _ in range(400):\n        if os.fork()==0:\n"
              "            import time; time.sleep(5); os._exit(0)\n        n+=1\nexcept OSError:\n"
              "    print('bounded', n)\n    sys.exit(0)\nprint('unbounded', n)\n")
    result = policy.run_sandboxed([sys.executable, "-c", script], tmp_path, timeout=60, nproc=64)
    marker, count = result.completed.stdout.strip().split()
    assert busy_same_uid.poll() is None
    assert marker == "bounded" and 0 < int(count) <= 64 + _NPROC_HEADROOM, (
        result.completed.stdout + result.completed.stderr)
    normal = policy.run_sandboxed(["/bin/sh", "-c", "echo ok"], tmp_path, timeout=60)
    assert normal.completed.returncode == 0 and normal.completed.stdout.strip() == "ok", normal


def test_session_launcher_bounds_thread_creation(tmp_path):
    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    cap = replace(cap, limits=replace(cap.limits, nproc=8, wall_seconds=15))
    with ExecSession(tmp_path / "run") as session:
        result = session.run([sys.executable, "-c", THREAD_STORM_SCRIPT], cap)
    marker, count = result.output.decode().strip().split()
    assert result.exit_code == 0 and marker == "bounded" and 0 < int(count) < 200, result


def test_launcher_relative_limit_bounds_threads_without_cgroup(tmp_path):
    launcher = Path(policy.__file__).with_name("sf_launch.py")
    spec = {"argv": [sys.executable, "-c", THREAD_STORM_SCRIPT], "cwd": str(tmp_path), "env": {},
            "limits": asdict(ExecLimits(nproc=8)), "memory_supervised": True}
    result = subprocess.run([sys.executable, "-I", str(launcher), json.dumps(spec)],
                            capture_output=True, text=True, timeout=10)
    marker, count = result.stdout.strip().split()
    assert result.returncode == 0 and marker == "bounded" and 0 < int(count) < 200, result
