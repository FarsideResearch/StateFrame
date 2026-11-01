"""The Linux outer check must tolerate same-uid threads and still bound forks."""
import os
import select
import subprocess
import sys

import pytest

import kernel_execution_policy as policy
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession

pytestmark = [pytest.mark.host_sandbox,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RLIMIT_NPROC semantics")]


def test_check_sandbox_starts_with_many_user_threads(tmp_path):
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
    finally:
        helper.terminate()
        helper.communicate(timeout=10)


def test_check_sandbox_still_bounds_new_processes(tmp_path):
    # The budget applies to processes the check creates: a fork loop beyond it must be refused.
    script = ("import os,sys\nn=0\ntry:\n    for _ in range(400):\n        if os.fork()==0:\n"
              "            import time; time.sleep(5); os._exit(0)\n        n+=1\nexcept OSError:\n"
              "    print('bounded', n)\n    sys.exit(0)\nprint('unbounded', n)\n")
    result = policy.run_sandboxed([sys.executable, "-c", script], tmp_path, timeout=60, nproc=64)
    assert "bounded" in result.completed.stdout, result.completed.stdout + result.completed.stderr
