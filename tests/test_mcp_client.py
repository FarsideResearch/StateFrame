"""MCP stdio tests use a real peer through the execution boundary."""
import json
import sys

import pytest

from kernel_mcp_tools import McpError, _schema, discover, load_mcp_files, run_driver, result_text, validate_arguments

SERVER = r'''#!/usr/bin/env python3
import json
import os
import sys
import time
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    if mode == "early":
        sys.exit(0)
    if mode == "hang":
        time.sleep(600)
    if mode == "malformed":
        sys.stdout.write("{not json}\n")
        sys.stdout.flush()
        continue
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fixture", "version": "1"}}
    elif method == "tools/list":
        if mode == "request":
            send({"jsonrpc": "2.0", "id": 100, "method": "roots/list", "params": {}})
            response = json.loads(sys.stdin.readline())
            assert response == {"jsonrpc": "2.0", "id": 100, "result": {"roots": []}}
        page = request.get("params", {}).get("cursor")
        tools = [{"name": "echo", "description": "Echo text", "inputSchema": {
                  "type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}},
                 {"name": "add", "inputSchema": {"type": "object", "properties": {
                    "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}]
        if mode == "many":
            tools = [{"name": "tool_" + str(i), "description": "x" * 200,
                      "inputSchema": {"type": "object", "properties": {
                          "value": {"type": "string", "description": "x" * 200}},
                          "required": ["value"]}} for i in range(40)]
        if mode == "pages" and page is None:
            result = {"tools": tools[:1], "nextCursor": "second"}
        else:
            result = {"tools": tools[1:] if mode == "pages" else tools}
    elif method == "tools/call":
        if mode == "huge":
            result = {"content": [{"type": "text", "text": "x" * (5 << 20)}]}
        elif mode == "tool_error":
            result = {"content": [{"type": "text", "text": "failure"}], "isError": True}
        else:
            args = request["params"]["arguments"]
            text = args.get("text", str(args.get("a", 0) + args.get("b", 0)))
            result = {"content": [{"type": "text", "text": text}]}
    else:
        continue
    send({"jsonrpc": "2.0", "id": request["id"] + (1 if mode == "wrong_id" else 0),
          "result": result})
'''


@pytest.fixture
def server(tmp_path):
    path = tmp_path / "server"
    path.write_text(SERVER)
    path.chmod(0o755)
    return path


def config(tmp_path, server, **changes):
    definition = {"name": "local", "trust": "This test server is authored by this repository.",
                  "argv": [str(server)], "timeout_seconds": 2}
    definition.update(changes)
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [definition]}))
    return load_mcp_files([path], shell_network="loopback")


def driver(server, mode, operation="list", **args):
    workspace = server.parent / "workspace"
    workspace.mkdir(exist_ok=True)
    definition = config(server.parent, server, argv=[sys.executable, str(server), mode],
                        timeout_seconds=1).servers[0]
    return run_driver(definition, operation, workspace, server.parent,
                      shell_network="none", host_authorization=False,
                      tool=args.get("tool", ""), arguments=args.get("arguments"))


@pytest.mark.parametrize("mode", ["normal", "request", "pages"])
@pytest.mark.host_sandbox
def test_driver_lists_real_server(server, mode):
    envelope = driver(server, mode)
    assert envelope["ok"] is True
    assert [tool["name"] for tool in envelope["tools"]] == ["echo", "add"]


@pytest.mark.host_sandbox
def test_driver_calls_and_maps_results(server):
    envelope = driver(server, "normal", "call", tool="echo", arguments={"text": "hello"})
    assert envelope["result"]["content"][0]["text"] == "hello"
    assert result_text(envelope["result"]) == "hello"
    error = driver(server, "tool_error", "call", tool="echo", arguments={"text": "hello"})
    assert error["result"]["isError"] is True
    assert result_text({"content": [{"type": "image"}, {"type": "resource", "uri": "file:///x"}]}) == (
        "[image omitted]\n[resource: file:///x]")


@pytest.mark.parametrize("mode,fragment", [
    ("malformed", "invalid JSON"), ("wrong_id", "response id"), ("early", "exited"),
    ("huge", "exceeds 4 MiB"), ("hang", "deadline exceeded")])
@pytest.mark.host_sandbox
def test_driver_fails_closed(server, mode, fragment):
    operation = "call" if mode == "huge" else "list"
    with pytest.raises(McpError, match=fragment):
        driver(server, mode, operation, tool="echo", arguments={"text": "x"})


@pytest.mark.parametrize("field,value", [
    ("name", "A"), ("trust", ""), ("argv", []), ("network", "full"),
    ("writes", "host"), ("replay", "bad"), ("timeout_seconds", 0),
    ("timeout_seconds", 601), ("tools", ["echo", "echo"]), ("unknown", 1)])
def test_bad_config_rejected(tmp_path, server, field, value):
    with pytest.raises(McpError):
        config(tmp_path, server, **{field: value})


def test_duplicate_and_noncanonical_json_rejected(tmp_path, server):
    file = tmp_path / "mcp.json"
    file.write_text('{"schema":"stateframe-mcp/1","servers":[],"servers":[]}')
    with pytest.raises(McpError, match="duplicate JSON key"):
        load_mcp_files([file], shell_network="none")
    file.write_text('{"schema":"stateframe-mcp/1","servers":[NaN]}')
    with pytest.raises(McpError, match="non-finite"):
        load_mcp_files([file], shell_network="none")


def test_schema_sanitizer_and_argument_validation():
    schema = _schema({"type": "object", "properties": {
        "text": {"type": "string", "description": "x" * 1000}},
        "required": ["text"]})
    assert schema["properties"]["text"]["description"].startswith("Untrusted server property metadata: ")
    assert "unknown" not in schema
    with pytest.raises(McpError, match="unsupported constraint"):
        _schema({"type": "object", "properties": {"text": {"type": "string", "unknown": True}}})
    validate_arguments(schema, {"text": "okay"})
    with pytest.raises(McpError, match="wrong type"):
        validate_arguments(schema, {"text": 3})
    with pytest.raises(McpError, match="missing or unknown"):
        validate_arguments(schema, {})


@pytest.mark.host_sandbox
def test_sandboxed_discovery_budget_and_digest(tmp_path, server):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server_set = config(tmp_path, server, argv=[sys.executable, str(server), "many"], replay="observation",
                        timeout_seconds=30)
    found = discover(server_set, workspace, tmp_path, shell_network="none",
                     host_authorization=False)
    assert 0 < len(found.tools) < 40
    assert found.omitted
    assert found.digest == discover(server_set, workspace, tmp_path,
                                    shell_network="none", host_authorization=False).digest
