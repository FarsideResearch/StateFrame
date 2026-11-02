"""End to end: with STATEFRAME_STABLE_TOOL_HEAD=on the provider tool list only ever grows by appending.

Runs the bundled scripted discount_bug example offline (needs the host sandbox and a local socket).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


def _run(tmp_path: Path, switch: str | None) -> tuple[subprocess.CompletedProcess, Path]:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if switch is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = switch
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, timeout=240)
        script.assert_consumed()
        return result, run
    finally:
        server.shutdown()
        server.server_close()


def _wire(run: Path) -> list[dict]:
    return [json.loads(path.read_bytes()) for path in sorted((run / "wire").glob("*.request.json"))]


def test_listed_tools_only_append_and_the_run_still_succeeds(tmp_path):
    result, run = _run(tmp_path, "on")
    assert result.returncode == 0, result.stdout + result.stderr
    requests = _wire(run)
    assert len(requests) >= 2
    for previous, current in zip(requests, requests[1:]):
        before, after = previous["tools"], current["tools"]
        # Every earlier tool keeps its exact bytes and position; new tools are appended.
        assert json.dumps(after[:len(before)]) == json.dumps(before)
    summary = json.loads((run / "summary.json").read_text())
    assert summary.get("succeeded") is True


def test_switch_off_keeps_per_turn_menus(tmp_path):
    result, run = _run(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(str(message.get("content", "")).startswith("[Tool menu | this turn]")
                   for request in _wire(run) for message in request["messages"])
