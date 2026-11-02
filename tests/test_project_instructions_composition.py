"""Project instructions composed with hooks, a selected skill, an MCP server and test-runner checks."""
from __future__ import annotations

import json
import re
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


# Test-runner modes: the default non-pytest --test-cmd (unittest) with each protection setting,
# and an explicit --check-cmd with no test command.
RUNNERS = {
    "test-cmd-protect-all": (("--protect-tests", "all"), True),
    "test-cmd-protect-task": (("--protect-tests", "task"), True),
    "check-cmd": (("--check-cmd", "python -m py_compile app.py"), False),
}


def _case(tmp_path: Path, name: str, runner: str, *extra: str):
    base = tmp_path / name
    base.mkdir()
    repo, flags, script = _inputs(base)
    runner_flags, default_test_cmd = RUNNERS[runner]
    case = run_case(base, str(script), repo=repo, flags=(*flags, *runner_flags, *extra),
                    default_test_cmd=default_test_cmd)
    assert case.process.returncode == 0, (case.process.stdout, case.process.stderr, case.summary)
    assert case.summary and case.summary["succeeded"], case.summary
    return case


def _text(message: dict) -> str:
    return str(message.get("content") or "")


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_instructions_compose_with_hooks_skills_mcp_and_test_runners(tmp_path, runner):
    plain = _case(tmp_path, "plain", runner)
    framed = _case(tmp_path, "framed", runner, "--instructions")
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


ROUTES = ("--route", "read.temperature=0.3", "--route", "plan.temperature=0.4",
          "--route", "finish.temperature=0.5")


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_instructions_compose_with_routes(tmp_path, runner):
    # Per-step models on top of instructions, hooks, a skill, MCP and each test-runner mode: the one
    # instructions block stays last in the system message, the task stays messages[1], every request
    # carries a route record, and routing changes only sampling, never what the model is shown.
    framed = _case(tmp_path, "framed", runner, "--instructions")
    routed = _case(tmp_path, "routed", runner, "--instructions", *ROUTES)
    # Per-role sampling is applied at send time, so check the bytes that went out (wire/), not the
    # engine-side request records.
    wire = lambda case: [json.loads(path.read_text())  # noqa: E731
                         for path in sorted((case.run_dir / "wire").glob("*.request.json"))]
    sent, framed_sent = wire(routed), wire(framed)
    assert sent and len(sent) == len(framed_sent)
    routes = [json.loads(path.read_text()) for path in sorted((routed.run_dir / "wire").glob("*.route.json"))]
    assert len(routes) == len(sent)
    assert routes[0]["role"] == "plan"
    for request, route in zip(sent, routes):
        messages = request["messages"]
        assert messages[0]["role"] == "system" and _text(messages[0]).endswith("\nproject-instructions>>>")
        assert sum(MARKER in _text(message) for message in messages) == 1
        assert messages[1]["role"] == "user" and MARKER not in _text(messages[1])
        expected = {"plan": 0.4, "read": 0.3, "finish": 0.5}.get(route["role"])
        if expected is not None:
            assert request.get("temperature") == expected, (route["role"], request.get("temperature"))
    # Two separate runs differ only where any two runs without routes do (the documented determinism
    # scope): run-scoped 20-hex ids (wire-truth handles, tool call ids, edit receipts), how long a tool
    # call took, and the receipt hash over that duration. Mask those; messages and tools must match.
    volatile = [(re.compile(r"(?<![0-9a-f])[0-9a-f]{20}(?![0-9a-f])"), "<ID>"),
                (re.compile(r"\| \d+\.\d+s"), "| <SECONDS>"),
                (re.compile(r'(result_sha256\\*":\\*")[0-9a-f]{64}'), r"\1<SHA>")]

    def normalise(request, base):
        text = json.dumps({"messages": request["messages"], "tools": request.get("tools")}, sort_keys=True)
        text = text.replace(str(tmp_path / base), "<BASE>")
        for pattern, replacement in volatile:
            text = pattern.sub(replacement, text)
        return text
    assert ([normalise(request, "routed") for request in sent]
            == [normalise(request, "framed") for request in framed_sent])
