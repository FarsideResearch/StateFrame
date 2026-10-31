"""Real MCP peer, execution sandbox and scripted provider integration."""
import json
from dataclasses import replace
from pathlib import Path
import socket
import sys

import pytest

from acceptance._support import copy_repo, run_case
from kernel_mcp_tools import McpTool, _schema, discover, load_mcp_files, run_driver, result_text
from kernel_exec_session import ExecSession
from kernel_shell_tool import ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import ToolReplayRefused, configure_replay, tape_for_run
from kernel_request_diet import RequestTokenMeter
from stateframe.endpoint import Endpoint
from stateframe.runner import RunOptions, prepare, resume_run
from stateframe.resume import ResumeRefused

pytestmark = pytest.mark.host_sandbox
FIXTURE = Path(__file__).parent / "fixtures/mcp_server.py"


def config(tmp_path, mode, *, replay="external", writes="none", argv_tail=()):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [{
        "name": "fixture", "trust": "Reviewed local fixture for this test",
        "argv": [sys.executable, str(FIXTURE), mode, *argv_tail], "network": "none",
        "writes": writes, "replay": replay, "timeout_seconds": 5}]}))
    return path


def test_discovery_and_budget(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    servers = load_mcp_files([config(tmp_path, "many")], shell_network="loopback")
    found = discover(servers, workspace, tmp_path, shell_network="loopback", host_authorization=False)
    assert found.tools and found.omitted
    assert len(found.tools) + len(found.omitted) == 40
    count, _ = RequestTokenMeter(template_overhead_tokens=0).measure(
        {"messages": [], "tools": found.schemas()})
    assert count <= 1500


def test_name_collision_is_omitted(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    servers = load_mcp_files([config(tmp_path, "normal")], shell_network="loopback")
    existing = {"type": "function", "function": {"name": "mcp_fixture__echo",
                "description": "Existing command", "parameters": {"type": "object",
                "properties": {}, "required": []}}}
    found = discover(servers, workspace, tmp_path, shell_network="loopback",
                     host_authorization=False, custom_schemas=[existing])
    assert not found.tools and "fixture:echo" in found.omitted


def test_readonly_workspace_and_network_none(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = load_mcp_files([config(tmp_path, "probe_write")], shell_network="loopback").servers[0]
    result = run_driver(server, "call", workspace, tmp_path, shell_network="loopback",
                        host_authorization=False, tool="echo", arguments={"value": "x"})
    assert result_text(result["result"]) != "wrote"
    assert not (workspace / "mcp-forbidden.txt").exists()
    outside = tmp_path / "outside-forbidden.txt"
    outside_server = load_mcp_files([config(tmp_path, "probe_outside", argv_tail=(str(outside),))],
                                    shell_network="loopback").servers[0]
    result = run_driver(outside_server, "call", workspace, tmp_path, shell_network="loopback",
                        host_authorization=False, tool="echo", arguments={"value": "x"})
    assert result_text(result["result"]) != "wrote"
    assert not outside.exists()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        network_server = load_mcp_files([config(tmp_path, "probe_network", argv_tail=(str(port),))],
                                        shell_network="loopback").servers[0]
        result = run_driver(network_server, "call", workspace, tmp_path, shell_network="loopback",
                            host_authorization=False, tool="echo", arguments={"value": "x"})
        assert result_text(result["result"]) != "connected"


def test_offline_run_observation_reaches_next_frame(tmp_path):
    repo = copy_repo(tmp_path)
    mcp = config(tmp_path, "normal", replay="observation")
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "mcp_fixture__echo", "arguments": {"value": "mcp-roundtrip"}},
        {"tool": "done", "arguments": {"summary": "Inspected the tool result."}}]}))
    case = run_case(tmp_path, str(script), repo=repo, flags=("--mcp", str(mcp), "--no-run-env"))
    assert case.process.returncode == 0, (case.process.stdout, case.process.stderr, case.summary)
    case.assert_offered("mcp_fixture__echo")
    case.assert_reply("mcp-roundtrip")
    assert case.summary and case.summary["succeeded"]


def test_tool_list_change_changes_resume_pin(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    servers = load_mcp_files([config(tmp_path, "switch_list")], shell_network="loopback")
    first = discover(servers, workspace, tmp_path, shell_network="loopback", host_authorization=False)
    (workspace / "variant.txt").write_text("changed")
    second = discover(servers, workspace, tmp_path, shell_network="loopback", host_authorization=False)
    assert second.digest != first.digest


def test_resume_refuses_changed_mcp_tool_list(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    marker = tmp_path / "variant"
    marker.write_text("base")
    mcp = config(tmp_path, "switch_list", argv_tail=(str(marker),))
    settings = ToolSettings(mcp_files=({"path": str(mcp), "digest": "pending"},))
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    run_dir, _, _ = prepare(RunOptions(repo=repo, task="inspect sample", endpoint=endpoint,
                                      run_dir=tmp_path / "run", tool_settings=settings))
    marker.write_text("changed")
    with pytest.raises(ResumeRefused, match="MCP tool list or schema changed"):
        resume_run(run_dir)


def test_external_call_replay_requires_flag(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n")
    server = load_mcp_files([config(tmp_path, "normal")], shell_network="none").servers[0]
    schema = _schema({"type": "object", "properties": {"value": {"type": "string"}},
                      "required": ["value"]})
    tool = McpTool("mcp_fixture__echo", server, "echo", "Echo", schema, "pinned-definition")
    settings = ToolSettings(run_env=False, shell_network="none", replay_external=False)
    with ExecSession(tmp_path) as session:
        shell = ShellTool(tmp_path, settings, {}, session)
        shell.bind(workspace, {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": False,
                               "allowed_product_paths": ["app.py"]}, set())
        configure_replay(tmp_path, active=False)
        original = shell._execute({"value": "hello"}, "call-1", custom=tool)
        assert "hello" in original.observation
        assert tape_for_run().lookup("call-1").replay_class == "external"
        configure_replay(tmp_path, active=True)
        with pytest.raises(ToolReplayRefused, match="replay-external"):
            shell._execute({"value": "hello"}, "call-1", custom=tool)
        shell.settings = replace(settings, replay_external=True)
        assert shell._execute({"value": "hello"}, "call-1", custom=tool).observation == original.observation
