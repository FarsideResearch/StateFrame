"""Regression checks for the host setup shared by skills and MCP."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from stateframe import runner
from stateframe.endpoint import Endpoint
from stateframe.engine.kernel_request_diet import RequestTokenMeter
from stateframe.engine.kernel_tool_settings import ToolSettings


def test_prepare_pins_skills_and_mcp_in_the_same_run(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")

    skill = tmp_path / "example-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: example-skill\ndescription: Review the example.\n---\nRead sample.py.\n"
    )
    mcp_file = tmp_path / "mcp.json"
    mcp_file.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [{
        "name": "example", "trust": "Local test executable", "argv": ["/bin/echo"],
        "network": "none", "writes": "none", "replay": "observation",
        "timeout_seconds": 5,
    }]}))

    monkeypatch.setattr(RequestTokenMeter, "measure", lambda self, body: (8, "calibrated_bytes"))
    monkeypatch.setattr(runner, "discover_mcp", lambda *args, **kwargs: SimpleNamespace(
        digest="discovered-mcp-digest", omitted=(), tools=()))
    settings = ToolSettings(git_tool=False, run_env=False,
                            mcp_files=({"path": str(mcp_file), "digest": "pending"},))
    options = runner.RunOptions(repo=repo, task="Inspect sample.py",
                                endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None),
                                run_dir=tmp_path / "run", skills=(str(skill),), tool_settings=settings)

    run, _, spec = runner.prepare(options)

    assert spec["skills"][0]["name"] == "example-skill"
    assert spec["skills_digest"] == json.loads((run / "skill-settings.json").read_text())["skills_digest"]
    assert spec["environment"]["STATEFRAME_SKILL_SETTINGS_FILE"] == str(run / "skill-settings.json")
    assert spec["mcp_tools_digest"] == "discovered-mcp-digest"
    assert spec["mcp_tools"] == [] and spec["mcp_omitted"] == []
    assert spec["tool_settings"]["mcp_files"] == [{
        "path": str(mcp_file), "digest": hashlib.sha256(mcp_file.read_bytes()).hexdigest()}]
    assert json.loads((run / "tool-settings.json").read_text()) == spec["tool_settings"]
