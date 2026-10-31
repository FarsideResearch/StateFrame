"""MCP stdio tests use a real child server and the real protocol driver."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_mcp_tools import McpError, _schema, discover, load_mcp_files, result_text, validate_arguments

DRIVER = Path(__file__).resolve().parents[1] / "src/stateframe/engine/kernel_mcp_driver.py"
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
    request = {"argv": [str(server), mode], "operation": operation, "timeout_seconds": 1,
               **args}
    return subprocess.run([sys.executable, "-I", str(DRIVER)], input=json.dumps(request).encode(),
                          capture_output=True, timeout=5)


@pytest.mark.parametrize("mode", ["normal", "request", "pages"])
def test_driver_lists_real_server(server, mode):
    envelope = json.loads(driver(server, mode).stdout)
    assert envelope["ok"] is True
    assert [tool["name"] for tool in envelope["result"]] == ["echo", "add"]


def test_driver_calls_and_maps_results(server):
    envelope = json.loads(driver(server, "normal", "call", tool="echo",
                                 arguments={"text": "hello"}).stdout)
    assert envelope["result"]["content"][0]["text"] == "hello"
    assert result_text(envelope["result"]) == "hello"
    error = json.loads(driver(server, "tool_error", "call", tool="echo",
                              arguments={"text": "hello"}).stdout)
    assert error["result"]["isError"] is True
    assert result_text({"content": [{"type": "image"}, {"type": "resource", "uri": "file:///x"}]}) == (
        "[image omitted]\n[resource: file:///x]")


@pytest.mark.parametrize("mode,fragment", [
    ("malformed", "invalid JSON"), ("wrong_id", "response id"), ("early", "exited"),
    ("huge", "exceeds 4 MiB"), ("hang", "deadline exceeded")])
def test_driver_fails_closed(server, mode, fragment):
    operation = "call" if mode == "huge" else "list"
    envelope = json.loads(driver(server, mode, operation, tool="echo",
                                 arguments={"text": "x"}).stdout)
    assert envelope["ok"] is False and fragment in envelope["error"]


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
        "text": {"type": "string", "description": "x" * 1000, "unknown": True}},
        "required": ["text"], "unknown": True})
    assert len(schema["properties"]["text"]["description"]) == 200
    assert "unknown" not in schema
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
