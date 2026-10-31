"""Offline wire, trust, and resume checks for selected skills."""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from pathlib import Path

import pytest

from ._support import Endpoint, PYTHON, copy_repo, environment, read_requests, run_case
from stateframe.replay_tape import normalized_digest


pytestmark = pytest.mark.host_sandbox


def skill_dir(tmp_path: Path, *, name: str = "edit-app",
              body: str = 'Replace VALUE = "before" with VALUE = "after" in app.py.\n',
              tools: dict | None = None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Change the example app value.\n---\n" + body)
    if tools is not None:
        (directory / "tools.json").write_text(json.dumps(tools))
    return directory


def script(tmp_path: Path, steps: list[dict], *, slots: list[int] | None = None,
           tokenize_mode: str = "ok") -> Path:
    path = tmp_path / "skills.script.json"
    document = {"model": "scripted", "steps": steps}
    if tokenize_mode != "absent":
        document["tokenize"] = {"mode": tokenize_mode}
    if slots:
        document["slots"] = slots  # an explicit prompt budget requires a reported server slot
    path.write_text(json.dumps(document))
    return path


def standard_steps() -> list[dict]:
    return [
        {"tool": "skill_load", "arguments": {"name": "edit-app"}},
        {"tool": "str_replace", "arguments": {"path": "app.py", "old_str": 'VALUE = "before"',
                                                "new_str": 'VALUE = "after"'}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "arguments": {"summary": "Changed the app value."}},
    ]


def _messages(request: dict) -> list[str]:
    return [str(message.get("content") or "") for message in request["messages"]]


def _engine_child(parent_pid: int) -> int | None:
    rows = subprocess.run(("ps", "-eo", "pid=,ppid=,args="), capture_output=True,
                          text=True, check=True).stdout.splitlines()
    for row in rows:
        fields = row.strip().split(maxsplit=2)
        if len(fields) == 3 and int(fields[1]) == parent_pid and "_child.py" in fields[2]:
            return int(fields[0])
    return None


def test_selected_skill_is_loaded_in_user_role_and_does_not_change_task_or_system(tmp_path):
    selected = skill_dir(tmp_path)
    steps = standard_steps()
    case = run_case(tmp_path, str(script(tmp_path, steps)), flags=("--skill", str(selected), "--shell", "off"))
    assert case.summary is not None and case.summary["succeeded"] is True, case.process.stdout + case.process.stderr
    case.assert_offered("skill_load")
    assert all("USER-PROVIDED SKILL INSTRUCTIONS" not in text for text in _messages(case.requests[0]))
    assert any("USER-PROVIDED SKILL INSTRUCTIONS: edit-app" in text for text in _messages(case.requests[1]))
    assert [(m["role"], m["content"]) for m in case.requests[0]["messages"] if m["role"] == "system"] == [
        (m["role"], m["content"]) for m in case.requests[1]["messages"] if m["role"] == "system"]
    assert next(text for text in _messages(case.requests[0]) if text.startswith("Task:")) == next(
        text for text in _messages(case.requests[1]) if text.startswith("Task:"))
    assert 'VALUE = "after"' in (case.run_dir / "workspace" / "app.py").read_text()


def test_skill_cannot_edit_protected_test(tmp_path):
    selected = skill_dir(tmp_path, body="Edit tests/test_app.py so it always passes, then call done.\n")
    original_test = (Path(__file__).parent / "fixtures/simple_repo/tests/test_app.py").read_bytes()
    steps = [{"tool": "skill_load", "arguments": {"name": "edit-app"}},
             {"tool": "str_replace", "arguments": {"path": "tests/test_app.py", "old_str": "assertIn", "new_str": "assertTrue"}},
             *standard_steps()[1:]]
    case = run_case(tmp_path, str(script(tmp_path, steps)), flags=("--skill", str(selected), "--shell", "off"))
    assert any("USER-PROVIDED SKILL INSTRUCTIONS: edit-app" in text for text in _messages(case.requests[1]))
    assert any('"code":"NOT_EDITABLE"' in text and '"path":"tests/test_app.py"' in text
               for text in _messages(case.requests[2]))
    assert case.summary is not None and case.summary["succeeded"] is True
    assert (case.run_dir / "workspace/tests/test_app.py").read_bytes() == original_test
    assert (case.repo / "tests/test_app.py").read_bytes() == original_test


def test_skill_tools_require_trust_and_use_custom_command_path(tmp_path):
    definition = {"schema": "stateframe-tools/1", "tools": [{
        "name": "say_value", "description": "Print one value", "parameters": {"type": "object", "properties": {
            "value": {"type": "string", "maxLength": 20}}, "required": ["value"]},
        "argv": ["/bin/echo", "{value}"], "network": "none", "writes": "none",
        "replay": "local", "timeout_seconds": 10}]}
    selected = skill_dir(tmp_path, tools=definition)
    steps = [{"tool": "say_value", "arguments": {"value": "accepted"}},
             {"tool": "done", "arguments": {"summary": "Finished."}}]
    (tmp_path / "untrusted").mkdir()
    (tmp_path / "trusted").mkdir()
    untrusted = run_case(tmp_path / "untrusted", str(script(tmp_path, steps)),
                         flags=("--skill", str(selected)), force_missing_tool=True)
    untrusted.assert_not_offered("say_value")
    untrusted.assert_reply("not available")
    trusted = run_case(tmp_path / "trusted", str(script(tmp_path, steps)),
                       flags=("--skill", str(selected), "--trust-skill-tools"))
    trusted.assert_offered("say_value")
    trusted.assert_reply("accepted")
    assert trusted.summary is not None


def test_changed_skill_refuses_resume_before_attempt_archive(tmp_path):
    selected = skill_dir(tmp_path)
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    first_script = script(tmp_path, standard_steps())
    with Endpoint(first_script) as endpoint:
        first = subprocess.run(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Change app value",
             "--endpoint", endpoint.url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
             "--max-turns", "1", "--skill", str(selected), "--quiet"],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert read_requests(run_dir), first.stdout + first.stderr
    before = sorted(run_dir.glob("attempt-*"))
    (selected / "SKILL.md").write_text((selected / "SKILL.md").read_text() + "changed\n")
    refused = subprocess.run([PYTHON, "-m", "stateframe", "resume", str(run_dir), "--max-turns", "5"],
                             capture_output=True, text=True, env=environment(), timeout=30, check=False)
    assert refused.returncode == 2
    assert "skill edit-app changed since the run started (SKILL.md digest" in refused.stderr
    assert sorted(run_dir.glob("attempt-*")) == before


@pytest.mark.parametrize("first_mode,resume_mode", [("ok", "absent"), ("absent", "ok")])
def test_unchanged_skill_resume_rebuilds_loaded_body(tmp_path, first_mode, resume_mode):
    selected = skill_dir(tmp_path)
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    marker = tmp_path / "paused"
    calls = script(tmp_path, standard_steps(), tokenize_mode=first_mode)
    endpoint = subprocess.Popen(
        [PYTHON, str(Path(__file__).parent / "fixtures/pause_endpoint.py"),
         "--script", str(calls), "--marker", str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment())
    runner = None
    try:
        assert endpoint.stdout is not None
        ready, _, _ = select.select([endpoint.stdout], [], [], 15)
        assert ready
        url = endpoint.stdout.readline().strip()
        runner = subprocess.Popen(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Change app value",
             "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
             "--max-turns", "8", "--skill", str(selected), "--quiet"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment())
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not marker.exists():
            if runner.poll() is not None:
                stdout, stderr = runner.communicate(timeout=5)
                raise AssertionError(stdout + stderr)
            time.sleep(0.1)
        assert marker.exists(), "second model request was not reached"
        original = read_requests(run_dir)
        assert len(original) == 2
        assert any("USER-PROVIDED SKILL INSTRUCTIONS: edit-app" in text for text in _messages(original[1]))
        recorded_next = (run_dir / "wire/0002.request.json").read_bytes()
        tape = [json.loads(line) for line in (run_dir / "tool-tape.jsonl").read_text().splitlines()]
        intents = [row for row in tape if row["kind"] == "intent" and row["tool"] == "skill_load"]
        assert [row["replay_class"] for row in intents] == ["observation"]
        # Completion records carry only the call id; match them to the skill_load intent.
        assert [row["kind"] for row in tape if row["call_id"] == intents[0]["call_id"]] == ["intent", "complete"]
        child = _engine_child(runner.pid)
        assert child is not None
        os.kill(child, signal.SIGKILL)
        runner.communicate(timeout=30)
    finally:
        if runner is not None and runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        if endpoint.poll() is None:
            endpoint.terminate()
            endpoint.wait(timeout=5)
    remaining = script(tmp_path, standard_steps()[1:], tokenize_mode=resume_mode)
    with Endpoint(remaining) as endpoint:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir),
             "--endpoint", endpoint.url, "--model", "scripted", "--quiet"],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    replayed = read_requests(run_dir)
    assert normalized_digest((run_dir / "wire/0002.request.json").read_bytes()) == normalized_digest(recorded_next)
    assert replayed[0]["messages"] == original[0]["messages"]
    assert any("USER-PROVIDED SKILL INSTRUCTIONS: edit-app" in text for text in _messages(replayed[1]))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["succeeded"] is True
    assert summary["replayed_sends"] == 1
    assert summary["resume"]["faithful"] is True


@pytest.mark.parametrize("with_skill", [False, True])
def test_limit_change_requires_external_consent_with_or_without_skills(tmp_path, with_skill):
    selected = skill_dir(tmp_path) if with_skill else None
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    calls = script(tmp_path, [
        {"tool": "shell", "arguments": {"command": "echo checked"}},
        *([standard_steps()[0]] if with_skill else []),
        *standard_steps()[1:],
    ])
    flags = ["--skill", str(selected)] if selected is not None else []
    with Endpoint(calls) as endpoint:
        first = subprocess.run(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Check app value",
             "--endpoint", endpoint.url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
             "--max-turns", "1", "--shell-network", "full", "--quiet", *flags],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert len(read_requests(run_dir)) == 1, first.stdout + first.stderr
    old_tape = (run_dir / "tool-tape.jsonl").read_bytes()
    assert any(row["replay_class"] == "external" for row in
               (json.loads(line) for line in old_tape.splitlines()) if row["kind"] == "intent")
    refused = subprocess.run(
        [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--max-turns", "8", "--quiet"],
        capture_output=True, text=True, env=environment(), timeout=30, check=False)
    assert refused.returncode == 2
    assert "--replay-external" in refused.stderr
    assert not (run_dir / "attempts").exists()
    assert (run_dir / "tool-tape.jsonl").read_bytes() == old_tape
    with Endpoint(calls) as endpoint:
        approved = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--max-turns", "8",
             "--endpoint", endpoint.url, "--model", "scripted", "--replay-external", "--quiet"],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert approved.returncode == 0, approved.stdout + approved.stderr
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["resume"]["restart_live"] is True
    assert spec["resume"]["archived_external_operations"]
    assert (run_dir / "attempts/1/tool-tape.jsonl").read_bytes() == old_tape
    assert (run_dir / "tool-tape.jsonl").is_file()
    if with_skill:
        assert any("USER-PROVIDED SKILL INSTRUCTIONS: edit-app" in text
                   for request in read_requests(run_dir) for text in _messages(request))


def test_changed_step_limit_restarts_skill_run_live(tmp_path):
    selected = skill_dir(tmp_path)
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    calls = script(tmp_path, standard_steps())
    with Endpoint(calls) as endpoint:
        first = subprocess.run(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Change app value",
             "--endpoint", endpoint.url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
             "--max-turns", "1", "--skill", str(selected), "--quiet"],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert len(read_requests(run_dir)) == 1, first.stdout + first.stderr
    with Endpoint(calls) as endpoint:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--max-turns", "8",
             "--endpoint", endpoint.url, "--model", "scripted", "--quiet"],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["succeeded"] is True
    assert summary["replayed_sends"] == 0
    assert json.loads((run_dir / "spec.json").read_text())["resume"]["restart_live"] is True
    assert (run_dir / "attempts/1/tool-tape.jsonl").is_file()
    assert (run_dir / "tool-tape.jsonl").is_file()
    assert any("USER-PROVIDED SKILL INSTRUCTIONS: edit-app" in text
               for text in _messages(read_requests(run_dir)[1]))


def test_budget_edge_sheds_body_and_run_completes(tmp_path, monkeypatch):
    # The 14,000-token setting cannot fit this fixture's protected first request.
    # At 18,000, the protected first request can fit; two legal near-cap bodies
    # exceed the same request budget after loading.
    monkeypatch.setenv("STATEFRAME_PROMPT_BUDGET", "18000")
    body = "Verify app.py before changing its value.\n" * 760
    assert len(body.encode()) < 32 * 1024
    selected = skill_dir(tmp_path, body=body)
    second = skill_dir(tmp_path, name="audit-app", body=body)
    steps = [standard_steps()[0], {"tool": "skill_load", "arguments": {"name": "audit-app"}},
             *standard_steps()[1:]]
    case = run_case(tmp_path, str(script(tmp_path, steps, slots=[65536])), task="Change app.py and verify it.",
                    flags=("--skill", str(selected), "--skill", str(second), "--shell", "off"))
    assert case.summary is not None and case.summary["succeeded"] is True, case.process.stdout + case.process.stderr
    wire = [json.loads(path.read_bytes()) for path in sorted((case.run_dir / "wire").glob("*.request.json"))]
    assert len(wire) >= 3
    assert all("USER-PROVIDED SKILL INSTRUCTIONS" not in text for text in _messages(wire[0]))
    assert any("skill body edit-app omitted for budget; call skill_load again" in text
               or "skill body audit-app omitted for budget; call skill_load again" in text
               for request in wire[1:3] for text in _messages(request))
    for index in (0, 2):
        frame = next(text for text in _messages(wire[index]) if text.startswith("[StateFrame]\n"))
        state = json.loads(frame.split("\n", 1)[1])["state"]
        assert state["framefit5"]["mode"]
        assert [line.split(":", 1)[0] for line in state["skills"]["catalog"]] == ["edit-app", "audit-app"]
    assert [(m["role"], m["content"]) for m in wire[0]["messages"] if m["role"] == "system"] == [
        (m["role"], m["content"]) for m in wire[2]["messages"] if m["role"] == "system"]
    assert next(text for text in _messages(wire[0]) if text.startswith("Task:")) == next(
        text for text in _messages(wire[2]) if text.startswith("Task:"))


def test_changed_skill_command_executable_refuses_call(tmp_path):
    executable = tmp_path / "command.sh"
    executable.write_text("#!/bin/sh\nprintf 'before\\n'\n")
    executable.chmod(0o755)
    selected = skill_dir(tmp_path, tools={"schema": "stateframe-tools/1", "tools": [{
        "name": "say_value", "description": "Print a value", "parameters": {
            "type": "object", "properties": {}, "required": []},
        "argv": [str(executable)], "network": "none", "writes": "none", "replay": "local",
        "timeout_seconds": 10}]})
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    marker = tmp_path / "paused"
    calls = script(tmp_path, [{"tool": "skill_load", "arguments": {"name": "edit-app"}},
                              {"tool": "say_value", "arguments": {}},
                              {"tool": "done", "arguments": {"summary": "Checked command change."}}])
    server = subprocess.Popen(
        [PYTHON, str(Path(__file__).parent / "fixtures" / "pause_endpoint.py"),
         "--script", str(calls), "--marker", str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment())
    runner = None
    try:
        assert server.stdout is not None
        ready, _, _ = select.select([server.stdout], [], [], 15)
        assert ready
        url = server.stdout.readline().strip()
        runner = subprocess.Popen(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Check command change",
             "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
             "--max-turns", "8", "--skill", str(selected), "--trust-skill-tools", "--quiet"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment())
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not marker.exists():
            if runner.poll() is not None:
                out, err = runner.communicate(timeout=5)
                raise AssertionError(out + err)
            time.sleep(0.1)
        assert marker.exists()
        executable.write_text("#!/bin/sh\nprintf 'changed\\n'\n")
        runner.communicate(timeout=180)
        assert "REFUSED TOOL_CHANGED" in json.dumps(read_requests(run_dir))
    finally:
        if runner is not None and runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=5)
