"""Host-only proof that the public CLI sends the first planned step to its route."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ._support import Endpoint, PYTHON, copy_repo, environment


pytestmark = [pytest.mark.host_sandbox,
              pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "src" / "stateframe" /
                                      "engine" / "kernel_frame_routes.py").is_file(),
                                 reason="routing engine lane not installed")]


def test_public_cli_routes_first_plan_send_to_second_endpoint(tmp_path):
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    cheap_script = tmp_path / "cheap.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "steps": [
        {"tool": "read", "arguments": {"path": "app.py"}}]}))
    default_script = tmp_path / "default.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "steps": []}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap:
        process = subprocess.run([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Inspect app.py",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "3", "--quiet", "--route", f"plan.endpoint={cheap.url}",
            "--route", "plan.model=cheap", "--route", f"read.endpoint={cheap.url}",
            "--route", "read.model=cheap"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    spec = json.loads((run_dir / "spec.json").read_text())
    first = json.loads((run_dir / "wire" / "0001.route.json").read_text())
    assert spec["routes"]["schema"] == "stateframe-routes/1"
    assert first["role"] == "plan"
    assert first["model"] == "cheap"
    assert first["endpoint_host"] == cheap.url.removeprefix("http://").removesuffix("/v1")
