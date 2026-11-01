"""Hooks, skills and MCP together in one offline run: every tool route passes the hook gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ._support import run_case
from .test_skills import script, skill_dir


pytestmark = pytest.mark.host_sandbox
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/mcp_server.py"
ECHO = "mcp_fixture__echo"

# A pre_tool guard that denies only the MCP call asking for "blocked", and a post_tool audit that
# leaves a note naming every tool it saw.
GUARD = ("import json,sys;a=json.load(sys.stdin)['data']['args'];"
         "sys.exit((sys.stderr.write('blocked by guard'),2)[1] if a.get('value')=='blocked' else 0)")
AUDIT = "import json,sys;print('audited '+json.load(sys.stdin)['data']['tool'])"


def _mcp_file(tmp_path: Path) -> Path:
    # why_required: the server declares a required `why`, which MCP dispatch sends as advertised.
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [{
        "name": "fixture", "trust": "Reviewed local fixture for this test",
        "argv": [sys.executable, str(FIXTURE), "why_required"], "network": "none",
        "writes": "none", "replay": "observation", "timeout_seconds": 15}]}))
    return path


def _hooks_file(tmp_path: Path) -> Path:
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "context", "event": "session_start",
         "argv": [sys.executable, "-c", "print('composition context')"]},
        {"name": "guard", "event": "pre_tool", "tools": ["mcp_*", "skill_load"],
         "argv": [sys.executable, "-c", GUARD]},
        {"name": "audit", "event": "post_tool", "argv": [sys.executable, "-c", AUDIT]},
    ]}))
    return path


def test_hooks_gate_skill_and_mcp_calls_in_one_run(tmp_path):
    selected = skill_dir(tmp_path)
    steps = [
        {"tool": "skill_load", "arguments": {"name": "edit-app"}},
        {"tool": ECHO, "arguments": {"value": "first-echo", "why": "read the fixture"}},
        {"tool": ECHO, "arguments": {"value": "blocked", "why": "the guard refuses this"}},
        {"tool": "str_replace", "arguments": {"path": "app.py", "old_str": 'VALUE = "before"',
                                                "new_str": 'VALUE = "after"'}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "arguments": {"summary": "Changed the app value."}},
    ]
    case = run_case(tmp_path, str(script(tmp_path, steps)), flags=(
        "--skill", str(selected), "--mcp", str(_mcp_file(tmp_path)),
        "--hooks", str(_hooks_file(tmp_path)), "--no-run-env"))
    assert case.summary is not None and case.summary["succeeded"] is True, (
        case.process.stdout + case.process.stderr)

    spec = json.loads((case.run_dir / "spec.json").read_text())
    assert spec["skills"][0]["name"] == "edit-app"
    assert spec["mcp_tools"] == [ECHO]
    assert [hook["name"] for hook in spec["hooks"]["definitions"]] == ["context", "guard", "audit"]
    case.assert_offered("skill_load")
    case.assert_offered(ECHO)

    records = [json.loads(line) for line in
               (case.run_dir / "hooks/records.jsonl").read_text().splitlines()]
    # Records carry no tool data; the order and decisions identify the calls: skill_load, the two MCP
    # calls (the second denied), then the native edit and test run, which only the audit matches.
    tool_events = [(r["event"], r["hook"], r["decision"]) for r in records
                   if r["event"] in {"pre_tool", "post_tool"}]
    assert tool_events == [
        ("pre_tool", "guard", "continue"), ("post_tool", "audit", "continue"),
        ("pre_tool", "guard", "continue"), ("post_tool", "audit", "continue"),
        ("pre_tool", "guard", "deny"), ("post_tool", "audit", "continue"),
        ("post_tool", "audit", "continue"), ("post_tool", "audit", "continue"),
    ]
    case.assert_reply("audited skill_load")
    case.assert_reply("audited str_replace")

    case.assert_reply("composition context")
    case.assert_reply("USER-PROVIDED SKILL INSTRUCTIONS: edit-app")
    case.assert_reply("first-echo")
    case.assert_reply("REFUSED HOOK_DENIED: guard: blocked by guard")
    case.assert_reply(f"audited {ECHO}")
    # The denied call never reached the server: only the admitted call is on the tool tape.
    tape = (case.run_dir / "tool-tape.jsonl").read_text()
    assert "first-echo" in tape and "the guard refuses this" not in tape
    assert (case.run_dir / "workspace/app.py").read_text().count('VALUE = "after"') == 1
