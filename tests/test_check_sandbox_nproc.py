"""Regression: the check sandbox must start for a user who already owns many processes.

RLIMIT_NPROC counts every process the user owns on Linux; an absolute cap of 64 made bwrap fail
("Creating new namespace failed: Resource temporarily unavailable") for typical desktop sessions,
so every run stopped at its first check. The ceiling is now the current count plus the budget.
"""
import subprocess
import sys

import pytest

import kernel_execution_policy as policy

pytestmark = [pytest.mark.host_sandbox,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RLIMIT_NPROC semantics")]


def test_check_sandbox_starts_with_many_user_processes(tmp_path):
    sleepers = [subprocess.Popen(["sleep", "120"]) for _ in range(80)]
    try:
        assert policy._user_process_count() >= 80
        result = policy.run_sandboxed(["/bin/sh", "-c", "echo ok"], tmp_path, timeout=60)
        assert result.completed.returncode == 0, result.completed.stderr
        assert "ok" in result.completed.stdout
    finally:
        for sleeper in sleepers:
            sleeper.kill()
            sleeper.wait()


def test_check_sandbox_still_bounds_new_processes(tmp_path):
    # The budget applies to processes the check creates: a fork loop beyond it must be refused.
    script = ("import os,sys\nn=0\ntry:\n    for _ in range(400):\n        if os.fork()==0:\n"
              "            import time; time.sleep(5); os._exit(0)\n        n+=1\nexcept OSError:\n"
              "    print('bounded', n)\n    sys.exit(0)\nprint('unbounded', n)\n")
    result = policy.run_sandboxed([sys.executable, "-c", script], tmp_path, timeout=60, nproc=64)
    assert "bounded" in result.completed.stdout, result.completed.stdout + result.completed.stderr
