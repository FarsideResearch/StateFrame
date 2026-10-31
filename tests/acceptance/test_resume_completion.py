"""Replay, completion ownership, disabled tools, and prompt budget."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time

import pytest

from ._support import Endpoint, FIXTURES, PYTHON, copy_repo, environment, read_requests, run_case


pytestmark = pytest.mark.host_sandbox


def _engine_child(parent_pid):
    rows = subprocess.run(("ps", "-eo", "pid=,ppid=,args="), capture_output=True,
                          text=True, check=True).stdout.splitlines()
    for row in rows:
        fields = row.strip().split(maxsplit=2)
        if len(fields) == 3 and int(fields[1]) == parent_pid and "_child.py" in fields[2]:
            return int(fields[0])
    return None


def _interrupted_run(tmp_path, *, network="none", script_name="resume.script.json", repo=None,
                     task="Format app.py and verify it"):
    repo = repo or copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    marker = tmp_path / "paused"
    script = FIXTURES / "scripts" / script_name
    endpoint = subprocess.Popen(
        [PYTHON, str(FIXTURES / "pause_endpoint.py"), "--script", str(script),
         "--marker", str(marker)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment())
    runner = None
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:")
        runner = subprocess.Popen(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task",
             task, "--endpoint", url, "--model", "scripted",
             "--run-dir", str(run_dir), "--test-python", PYTHON,
             "--test-cmd", "python -m unittest discover -s tests -q",
             "--max-turns", "32", "--quiet",
             "--shell-network", network], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment())
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not marker.exists():
            if runner.poll() is not None:
                stdout, stderr = runner.communicate(timeout=5)
                raise AssertionError(f"run ended before interruption: {stdout} {stderr}")
            time.sleep(0.1)
        assert marker.exists(), "second model request was not reached"
        requests_before = read_requests(run_dir)
        assert len(requests_before) >= 2
        child = _engine_child(runner.pid)
        assert child is not None
        os.kill(child, signal.SIGKILL)
        runner.communicate(timeout=30)
        return repo, run_dir, requests_before[1]
    finally:
        if runner is not None and runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        if endpoint.poll() is None:
            endpoint.terminate()
            try:
                endpoint.wait(timeout=5)
            except subprocess.TimeoutExpired:
                endpoint.kill()
                endpoint.wait(timeout=5)


def test_e16_local_shell_resume_replays_identical_next_request(tmp_path):
    repo, run_dir, expected = _interrupted_run(tmp_path)
    with Endpoint(FIXTURES / "scripts" / "resume_remaining.script.json") as endpoint:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr + str(summary)
    actual = read_requests(run_dir)[1]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]
    assert summary["succeeded"] is True
    assert 'VALUE = "after"' in (run_dir / "workspace" / "app.py").read_text()
    assert 'VALUE = "before"' in (repo / "app.py").read_text()


def test_e23_full_network_shell_requires_replay_external(tmp_path):
    _, run_dir, _ = _interrupted_run(tmp_path, network="full")
    with Endpoint(FIXTURES / "scripts" / "resume_remaining.script.json") as endpoint:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=60, check=False)
    assert resumed.returncode != 0
    assert "--replay-external" in resumed.stderr + resumed.stdout
    with Endpoint(FIXTURES / "scripts" / "resume_remaining.script.json") as endpoint:
        approved = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--replay-external", "--quiet"],
            capture_output=True, text=True, env=environment(), timeout=240, check=False)
    assert approved.returncode == 0, approved.stdout + approved.stderr
    assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True


def test_e19_volatile_shell_observation_is_byte_identical_on_resume(tmp_path):
    _, run_dir, expected = _interrupted_run(tmp_path, script_name="volatile.script.json")
    with Endpoint(FIXTURES / "scripts" / "volatile_remaining.script.json") as endpoint:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    actual = read_requests(run_dir)[1]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]


@pytest.mark.parametrize("operation", ["file_move", "file_delete"])
def test_file_move_kill_resume_replays_identical_request_and_sealed_rename(tmp_path, operation):
    repo = copy_repo(tmp_path)
    (repo / "obsolete.py").write_text("OLD = True\n")
    script = tmp_path / "move.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": operation, "required": True,
         "arguments": ({"source": "obsolete.py", "destination": "renamed.py"}
                       if operation == "file_move" else {"path": "obsolete.py"})},
        {"tool": "read", "arguments": {"path": "app.py"}}]}))
    _, run_dir, expected = _interrupted_run(
        tmp_path, script_name=script, repo=repo,
        task=("Rename obsolete.py to renamed.py and verify imports." if operation == "file_move"
              else "Delete obsolete.py and verify imports."))
    # Automatic verification may be the newest exchange. Prove the mutation
    # itself committed before killing, independently of which reply is newest.
    tape = [json.loads(line) for line in (run_dir / "tool-tape.jsonl").read_text().splitlines()]
    intent = next(row for row in tape if row.get("tool") == operation)
    assert intent["replay_class"] == "local"
    completion = next(row for row in tape if row["kind"] == "complete"
                      and row["call_id"] == intent["call_id"])
    assert completion["exit_code"] == 0 and operation + " applied" in completion["observation"]
    remaining = tmp_path / "move-remaining.json"
    remaining.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "arguments": {"summary": "Rename verified after restart."}}]}))
    with Endpoint(remaining) as endpoint:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    actual = read_requests(run_dir)[1]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["succeeded"] is True
    assert summary["resume"]["faithful"] is True
    assert set(summary["files_changed"]) == ({"obsolete.py", "renamed.py"}
                                            if operation == "file_move" else {"obsolete.py"})
    assert not (run_dir / "workspace/obsolete.py").exists()
    if operation == "file_move":
        assert (run_dir / "workspace/renamed.py").read_text() == "OLD = True\n"
    else:
        assert not (run_dir / "workspace/renamed.py").exists()
    assert (repo / "obsolete.py").exists() and not (repo / "renamed.py").exists()


def test_e21_shell_exit_zero_does_not_own_completion(tmp_path):
    repo = copy_repo(tmp_path)
    (repo / "tests/test_app.py").write_text(
        "import unittest\nclass RequiredCheck(unittest.TestCase):\n"
        "    def test_required(self):\n        self.fail('engine check must fail')\n")
    script = tmp_path / "completion.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "shell", "arguments": {"command": "true"}},
        {"tool": "done", "arguments": {"summary": "Shell returned zero."}}]}))
    case = run_case(tmp_path, str(script), repo=repo)
    case.assert_reply("exit 0")
    case.assert_offered("run_tests")
    assert case.summary is not None and case.summary["succeeded"] is False
    assert case.summary["tests_after"]["passed"] is False


def test_e24_shell_off_hallucinated_call_is_refused_and_run_continues(tmp_path):
    case = run_case(tmp_path, "one_shell.script.json", flags=("--shell", "off"),
                    force_missing_tool=True)
    case.assert_not_offered("shell")
    case.assert_reply("REFUSED SHELL_OFF:")
    assert len(case.requests) >= 2


def test_all_tool_schemas_fit_prompt_budget(tmp_path):
    repo = copy_repo(tmp_path)
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(("git", "-c", "user.name=Acceptance", "-c",
                    "user.email=acceptance@example.invalid", "commit", "--allow-empty", "-qm",
                    "baseline"), cwd=repo, check=True)
    case = run_case(tmp_path, "budget.script.json", repo=repo,
                    flags=("--tools", str(FIXTURES / "tools.json"), "--web"))
    for name in ("shell", "git", "web_fetch", "sample_tool"):
        case.assert_offered(name)
    sends = [json.loads(line) for line in (case.run_dir / "sends.jsonl").read_text().splitlines()]
    assert sends
    assert max(send["prompt_tokens"] for send in sends) <= 33_344
    assert case.summary is not None and case.summary["provider_sends"] == len(sends)
    assert all(isinstance(send["prompt_tokens"], int) for send in sends)
