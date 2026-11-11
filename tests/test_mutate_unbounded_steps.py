"""Mutation hunks of the ReAct loop accept an unbounded step limit (no --max-turns)."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import probes.react_repair as react_repair
from tools.compare_mutation_off_requests import EXAMPLES


CURRENT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("limit", "step", "left"), [(None, 3, 99), (0, 3, 99), (10, 3, 7), (2, 5, 0)])
def test_mutation_turns_left_handles_unbounded_and_bounded_limits(monkeypatch, limit, step, left):
    monkeypatch.setattr(react_repair, "MAX_STEPS", limit)
    assert react_repair._mutation_turns_left(step) == left


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation launches require the Linux sandbox")
def test_prebuilt_run_without_step_limit_submits(tmp_path):
    """The documented QUICKSTART opt-in command: no --max-turns, so the step limit is unbounded."""
    task, test_cmd = EXAMPLES["discount_bug"]
    repo = tmp_path / "repo"
    shutil.copytree(CURRENT / "examples" / "discount_bug", repo)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_MUTATION")}
    environment.update({"PYTHONPATH": str(CURRENT / "src"), "HOME": str(tmp_path / "home"),
                        "STATEFRAME_HOME": str(tmp_path / "home" / ".stateframe")})
    endpoint = subprocess.Popen([sys.executable, "-m", "stateframe.scripted_endpoint", "--script",
                                 str(CURRENT / "examples" / "discount_bug.script.json"), "--port", "0"],
                                env=environment, stdout=subprocess.PIPE, text=True)
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        run = tmp_path / "run"
        result = subprocess.run([sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", task,
                                 "--test-cmd", test_cmd, "--test-python", sys.executable,
                                 "--mutations", "prebuilt", "--endpoint", url, "--model", "scripted",
                                 "--run-dir", str(run), "--quiet"],
                                env=environment, capture_output=True, text=True, timeout=600)
        summary = json.loads((run / "summary.json").read_text())
        assert summary["engine_result"] == "SUBMITTED", (result.stdout[-3000:], result.stderr[-3000:])
        assert not (run / "child-traceback.txt").exists()
    finally:
        endpoint.terminate()
        endpoint.wait(timeout=10)
