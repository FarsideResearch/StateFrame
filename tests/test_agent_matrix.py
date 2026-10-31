"""Composed capabilities execute; unsupported lean runs refuse before requests."""
import json
import subprocess

import pytest

from acceptance._support import FIXTURES, copy_repo, run_case
from stateframe.profiles import load

pytestmark = pytest.mark.host_sandbox


@pytest.mark.parametrize("loop", ["composed", "lean"])
@pytest.mark.parametrize("name", ["job_wait", "job_output", "job_stop", "git", "web_fetch", "sample_tool", "generate_tool"])
def test_capability_matrix(tmp_path, loop, name):
    repo = copy_repo(tmp_path)
    profile = tmp_path / "profile.json"
    switches = load("r2hc3")
    switches["STATEFRAME_LEAN"] = "on" if loop == "lean" else "off"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}))
    flags = ["--profile", str(profile), "--no-run-env"]
    setup = []
    if name.startswith("job_"):
        setup = [{"tool": "shell", "arguments": {"command": "printf ready; sleep 30", "background": True}}]
        arguments = {"job": "j1"}
        if name == "job_wait":
            arguments.update(until="output", text="ready", timeout_seconds=5)
        expected = "job j1"
    elif name == "git":
        for argv in (("init", "-q"), ("config", "user.name", "Fixture"),
                     ("config", "user.email", "fixture@example.invalid"), ("add", "app.py"),
                     ("commit", "-qm", "matrix history")):
            subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True)
        arguments, expected = {"action": "log", "max_count": 1}, "matrix history"
    elif name == "web_fetch":
        flags += ["--web"]
        arguments, expected = {"url": "http://127.0.0.1/"}, "REFUSED WEB_REFUSED_ADDRESS:"
    else:
        flags += ["--tools", str(FIXTURES / "tools.json")]
        arguments = {"value": "accepted"} if name == "sample_tool" else {}
        expected = "accepted" if name == "sample_tool" else "files changed by this command:"
    script = tmp_path / "matrix.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        *setup, {"tool": name, "arguments": arguments},
        {"tool": "shell", "arguments": {"command": "printf matrix-next-frame"}},
        {"tool": "done", "arguments": {"summary": "Inspected the tool result."}}]}))
    case = run_case(tmp_path, str(script), repo=repo, flags=tuple(flags))
    if loop == "lean":
        assert case.process.returncode != 0
        assert "REFUSED LEAN_UNSUPPORTED:" in case.process.stderr
        assert not case.requests
        assert not (case.run_dir / "tool-tape.jsonl").exists()
        assert not (case.run_dir / "worker").exists()
        return
    assert case.summary is not None and case.summary["succeeded"], (
        case.summary, case.process.stdout, case.process.stderr)
    assert len(case.requests) > len(setup), case.summary
    case.assert_offered(name, request=len(setup))
    case.assert_reply(expected)
    case.assert_reply("matrix-next-frame")
    records = [json.loads(line) for path in (case.run_dir / "worker").rglob("events.jsonl")
               for line in path.read_text().splitlines()]
    assert any(record.get("kind") == "action_receipt" and
               (record.get("receipt", {}).get("kind") in {name, "shell_effect"}
                or record.get("data", {}).get("receipt", {}).get("kind") in {name, "shell_effect"})
               for record in records), f"missing typed receipt for {name}"
    tape = [json.loads(line) for line in (case.run_dir / "tool-tape.jsonl").read_text().splitlines()]
    assert any(row.get("tool") == name for row in tape)
