"""Project instructions composed with hooks, a selected skill and an MCP server in one run."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from acceptance._support import copy_repo, run_case

pytestmark = pytest.mark.host_sandbox
FIXTURE = Path(__file__).parent / "fixtures/mcp_server.py"
MARKER = "\n\nPROJECT INSTRUCTIONS ("


def _inputs(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    repo = copy_repo(tmp_path)
    (repo / "AGENTS.md").write_text("# Notes\nRun the tests with `python -m unittest discover -s tests -q`.\n")
    mcp = tmp_path / "mcp.json"
    mcp.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [{
        "name": "fixture", "trust": "Reviewed local fixture for this test",
        "argv": [sys.executable, str(FIXTURE), "normal"], "network": "none",
        "writes": "none", "replay": "observation", "processes": "single", "timeout_seconds": 15}]}))
    hooks = tmp_path / "hooks.json"
    hooks.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "start", "event": "session_start", "argv": [sys.executable, "-c", "print('start marker')"]},
        {"name": "prompt", "event": "prompt_submit", "argv": [sys.executable, "-c", "print('prompt marker')"]}]}))
    skill = tmp_path / "review-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: review-skill\ndescription: Review the change.\n---\n"
                                    "Read the changed file before finishing.\n")
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "mcp_fixture__echo", "arguments": {"value": "mcp-roundtrip"}},
        {"tool": "done", "arguments": {"summary": "Inspected the tool result."}}]}))
    flags = ("--mcp", str(mcp), "--hooks", str(hooks), "--skill", str(skill), "--no-run-env")
    return repo, flags, script


def _case(tmp_path: Path, name: str, *extra: str):
    base = tmp_path / name
    base.mkdir()
    repo, flags, script = _inputs(base)
    case = run_case(base, str(script), repo=repo, flags=(*flags, *extra))
    assert case.process.returncode == 0, (case.process.stdout, case.process.stderr, case.summary)
    assert case.summary and case.summary["succeeded"], case.summary
    return case


def _text(message: dict) -> str:
    return str(message.get("content") or "")


def test_instructions_compose_with_hooks_skills_and_mcp(tmp_path):
    plain = _case(tmp_path, "plain")
    framed = _case(tmp_path, "framed", "--instructions")
    assert framed.requests and len(framed.requests) == len(plain.requests)
    for request in framed.requests:
        messages = request["messages"]
        assert messages[0]["role"] == "system" and MARKER in _text(messages[0])
        assert sum(MARKER in _text(message) for message in messages) == 1
        assert _text(messages[0]).endswith("\nproject-instructions>>>")
        assert messages[1]["role"] == "user" and MARKER not in _text(messages[1])
    first = framed.requests[0]
    # Hook start context and the skill catalog land after the task; MCP tools are on the menu.
    hook_index = next(index for index, message in enumerate(first["messages"]) if "start marker" in _text(message))
    assert hook_index > 1
    assert any("review-skill" in _text(message) for message in first["messages"][1:])
    framed.assert_offered("mcp_fixture__echo")
    # Removing the one block gives back the run without --instructions, tools included.
    head = _text(first["messages"][0]).split(MARKER, 1)[0]
    without_slot = dict(first, messages=[dict(first["messages"][0], content=head), *first["messages"][1:]])
    normalise = lambda request: json.dumps(  # noqa: E731
        {"messages": request["messages"], "tools": request.get("tools")}, sort_keys=True
    ).replace(str(tmp_path / "framed"), "<BASE>").replace(str(tmp_path / "plain"), "<BASE>")
    left, right = normalise(without_slot), normalise(plain.requests[0])
    at = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), min(len(left), len(right)))
    assert left == right, (at, left[max(0, at - 200):at + 200], right[max(0, at - 200):at + 200])
