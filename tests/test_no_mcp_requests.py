"""No-server runs retain the released request shape."""
import json
import shutil
import sys
from pathlib import Path

import pytest
from tools.compare_no_mcp_requests import capture, write_launcher
from kernel_tool_settings import ToolSettings
from stateframe.endpoint import Endpoint
from stateframe.runner import RunOptions, prepare

ROOT = Path(__file__).resolve().parents[1]


def test_no_mcp_spec_omits_empty_fields(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    run_dir, _, spec = prepare(RunOptions(repo=repo, task="Inspect sample", endpoint=endpoint,
                                          run_dir=tmp_path / "run",
                                          tool_settings=ToolSettings(run_env=False)))
    assert all(key not in spec for key in ("mcp_tools_digest", "mcp_omitted", "mcp_tools"))
    assert "mcp_files" not in spec["tool_settings"]
    assert json.loads((run_dir / "spec.json").read_text()) == spec


@pytest.mark.host_sandbox
def test_no_mcp_request_bodies_match_golden(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "fixed_seed_run.py"
    write_launcher(launcher)
    script = ROOT / "examples/discount_bug.script.json"
    run_dir = tmp_path / "run"
    first = capture(ROOT, sys.executable, launcher, script, repo, run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    assert all(key not in spec for key in ("mcp_tools_digest", "mcp_omitted", "mcp_tools"))
    golden = tmp_path / "request-golden.bin"
    golden.write_bytes(first)
    assert b"mcp" not in golden.read_bytes().lower()
    run_dir.rename(tmp_path / "golden-run")
    repeated = capture(ROOT, sys.executable, launcher, script, repo, run_dir)
    assert repeated == golden.read_bytes()
