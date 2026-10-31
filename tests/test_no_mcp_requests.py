"""No-server runs retain the released request shape."""
import json
from kernel_tool_settings import ToolSettings
from stateframe.endpoint import Endpoint
from stateframe.runner import RunOptions, prepare

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
