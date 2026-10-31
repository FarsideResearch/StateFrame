import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kernel_custom_tools import CustomToolError
from kernel_mcp_tools import DRIVER, load_mcp_files, _schema, result_text
from stateframe import __version__

FIXTURE = Path(__file__).parent / "fixtures/mcp_server.py"


def call(mode, *, action="list", timeout=2, extra=()):
    request = {"argv": [sys.executable, str(FIXTURE), mode, *extra],
               "operation": action, "timeout_seconds": timeout,
               "client_version": __version__,
               "tool": "echo", "arguments": {"value": "hello"}}
    result = subprocess.run([sys.executable, "-I", str(DRIVER)], input=json.dumps(request),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and not result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("mode,fragment", [
    ("malformed", "invalid JSON"), ("wrong_id", "does not match"),
    ("early_exit", "exited before responding"), ("huge", "exceeds 4 MiB")])
def test_protocol_rejects_bad_peer(mode, fragment):
    response = call(mode, action="call" if mode == "huge" else "list")
    assert not response["ok"] and fragment in response["error"]


def test_normal_pagination_and_server_request():
    assert [x["name"] for x in call("pagination")["result"]] == ["echo", "add"]
    assert call("server_request")["result"][0]["name"] == "echo"
    assert result_text(call("normal", action="call")["result"]) == "hello"
    assert call("is_error", action="call")["result"]["isError"] is True


def test_timeout_kills_child(tmp_path):
    pidfile = tmp_path / "pid"
    response = call("hang", timeout=.2, extra=(str(pidfile),))
    assert response["code"] == "TIMEOUT"
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("mode", ["detached_hang", "malformed"])
def test_driver_cleans_detached_or_malformed_peer(tmp_path, mode):
    pidfile = tmp_path / "pid"
    response = call(mode, timeout=.3, extra=(str(pidfile),))
    assert response["ok"] is False
    assert not alive(int(pidfile.read_text()))


def alive(pid):
    """True for a running process; a killed orphan left unreaped by a container init counts as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return True


def test_driver_uses_product_version():
    assert call("version", extra=(__version__,))["ok"] is True


def definition():
    return {"name": "local_server", "trust": "Reviewed local fixture", "argv": [sys.executable, str(FIXTURE), "normal"],
            "timeout_seconds": 3}


def load(tmp_path, item):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [item]}))
    return load_mcp_files([path], shell_network="loopback")


@pytest.mark.parametrize("field,value", [
    ("name", "A"), ("trust", ""), ("trust", "x" * 301), ("argv", []),
    ("argv", ["/no/such/executable"]), ("network", "full"),
    ("writes", "outside"), ("replay", "unknown"), ("timeout_seconds", 0),
    ("timeout_seconds", 601), ("tools", ["same", "same"]), ("extra", 1)])
def test_config_rejects_invalid_fields(tmp_path, field, value):
    item = definition()
    item[field] = value
    with pytest.raises(CustomToolError):
        load(tmp_path, item)


def test_config_replay_constraints_and_json(tmp_path):
    item = definition()
    item["replay"] = "observation"
    assert load(tmp_path, item).servers[0].replay == "observation"
    item["writes"] = "workspace"
    with pytest.raises(CustomToolError):
        load(tmp_path, item)
    path = tmp_path / "mcp.json"
    path.write_text('{"schema":"stateframe-mcp/1","servers":[],"servers":[]}')
    with pytest.raises(CustomToolError, match="duplicate JSON key"):
        load_mcp_files([path], shell_network="loopback")


def test_schema_sanitizer_and_result_mapping():
    result = _schema({"type": "object", "properties": {
        "value": {"type": "string", "description": "a" * 500},
        "nested": {"type": "object"}}, "required": ["value"]})
    assert result["properties"]["value"]["description"].startswith("Untrusted server property metadata: ")
    assert result["properties"]["nested"]["type"] == "object"
    assert result_text({"content": [{"type": "image"}, {"type": "resource", "resource": {"uri": "x"}}]}) == "[image omitted]\n[resource: x]"
