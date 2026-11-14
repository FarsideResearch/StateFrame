"""A 16k mock-provider run must stop through the typed engine boundary."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled

from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.host_sandbox
def test_allocator_on_16k_context_runs_or_stops_typed(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    document = json.loads((ROOT / "examples/discount_bug.script.json").read_text())
    document["slots"] = [16_384]
    script = Script(document)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update(
        PYTHONPATH=str(ROOT / "src"),
        STATEFRAME_HOME=isolated_stateframe_home(),
        STATEFRAME_WORKING_SET_ALLOCATOR="on",
        STATEFRAME_CONTEXT_TOKENS="16384",
    )
    run = tmp_path / "run"
    server = serve(script)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
             "--max-turns", "1", "--quiet", "--json"],
            env=environment, capture_output=True, text=True, timeout=scaled(600), check=False)
    finally:
        server.shutdown()
        server.server_close()
    spec = json.loads((run / "spec.json").read_text())
    assert spec["environment"]["STATEFRAME_WORKING_SET_ALLOCATOR"] == "on"
    assert spec["environment"]["STATEFRAME_CONTEXT_TOKENS"] == "16384"
    output = completed.stdout + completed.stderr + (run / "engine.log").read_text()
    # Allocation failures must reach the typed stop boundary, never escape it.
    for error in ("FrameRequestAllocationError", "FrameFit5ImpossibleMinimum"):
        assert error not in output, output[-12000:]
    summary = json.loads(completed.stdout)
    assert summary["runtime_outcome"] in {"completed", "incomplete", "budget_stop", "suspended"}, summary
    assert summary["engine_result"] and not summary["engine_result"].startswith("ERR"), summary
    assert completed.returncode in (0, 1), summary
    if summary["engine_result"] == "BLOCKED":
        terminal = summary["engine_terminal"]
        assert summary["runtime_outcome"] == "suspended"
        assert terminal["resumable"] is True
        assert terminal["reason_code"] == "WORKING_SET_ALLOCATION_BLOCKED"
        assert terminal["suspension"]["proof"]["verified"] is True
    else:
        assert script.requests, "run neither reached the provider nor returned a typed suspension"
