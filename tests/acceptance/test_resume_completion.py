"""Replay, completion ownership, disabled tools, and prompt budget."""
from __future__ import annotations

import json
import signal
import subprocess
import time

import pytest

from ._support import Endpoint, FIXTURES, budget, PYTHON, copy_repo, environment, read_requests, run_case
from _timeouts import scaled
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from engine_run import run_monitored, start_monitored


pytestmark = pytest.mark.host_sandbox


def _engine_child(parent_pid):
    # ps truncates args to terminal width under captured pytest output; the
    # _child.py suffix can disappear even while the child is alive.
    rows = subprocess.run(("ps", "-ww", "-eo", "pid=,ppid=,args="), capture_output=True,
                          text=True, check=True).stdout.splitlines()
    for row in rows:
        fields = row.strip().split(maxsplit=2)
        if len(fields) == 3 and int(fields[1]) == parent_pid and "_child.py" in fields[2]:
            return int(fields[0])
    return None


def _interrupted_run(tmp_path, *, network="none", script_name="resume.script.json", repo=None,
                     task="Format app.py and verify it", wait_for_checkpoint=False,
                     checkpoint_task=None, pause_at=None, tokenize_off_marker=None, shell=None,
                     wait_for_saved=False):
    repo = repo or copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    marker = tmp_path / "paused"
    shell = shell or ("off" if wait_for_checkpoint else "standard")
    script = FIXTURES / "scripts" / ("checkpoint_resume.script.json"
                                     if wait_for_checkpoint and shell == "off" else script_name)
    pause_at = pause_at or (3 if wait_for_checkpoint else 2)
    endpoint_args = [PYTHON, str(FIXTURES / "pause_endpoint.py"), "--script", str(script),
                     "--marker", str(marker), "--pause-at", str(pause_at),
                     "--pause-seconds", "600" if wait_for_checkpoint or wait_for_saved else "30"]
    if tokenize_off_marker is not None:
        endpoint_args.extend(("--tokenize-off-marker", str(tokenize_off_marker)))
    endpoint = subprocess.Popen(
        endpoint_args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment())
    runner = None
    retain_endpoint = False
    run_task = task
    if wait_for_checkpoint and shell == "off":
        run_task = checkpoint_task or "Set app.py VALUE to after and verify it"
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:")
        launched_after = time.time()
        runner = start_monitored(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", run_task,
             "--endpoint", url, "--model", "scripted",
             "--run-dir", str(run_dir), "--test-python", PYTHON,
             "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "32", "--quiet",
            "--shell-network", network, "--shell", shell],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment(), start_new_session=True,
            expected_seconds=scaled(600), step="interrupted run",
            watch_paths=(run_dir / "checkpoints",))
        deadline = time.monotonic() + scaled(600)
        while not marker.exists():
            if runner.poll() is not None:
                stdout, stderr = runner.communicate()
                raise AssertionError(f"run ended before interruption: {stdout} {stderr}")
            time.sleep(0.1)
        assert marker.exists(), "paused model request was not reached"
        requests_before = read_requests(run_dir)
        paused_index = pause_at - 1
        assert len(requests_before) > paused_index
        if wait_for_checkpoint:
            pointer = run_dir / "checkpoints" / "LATEST"
            while not pointer.is_file():
                assert runner.poll() is None, "run ended before the checkpoint was saved"
                time.sleep(0.1)
            assert pointer.is_file(), "checkpoint was not saved before the paused request"
        child = _engine_child(runner.pid)
        assert child is not None
        identity = record_identity(child, launched_after=launched_after, command_fragment="_child.py")
        signal_run_verified(run_dir, child, identity, signal.SIGKILL, deadline=deadline,
                            what="paused resume request")
        runner.communicate(timeout=scaled(30))
        if wait_for_checkpoint:
            retain_endpoint = True
            return repo, run_dir, requests_before[paused_index], endpoint, url
        return repo, run_dir, requests_before[paused_index]
    finally:
        if runner is not None and runner.poll() is None:
            # Failure cleanup may run before the first checkpoint exists.
            signal_run_owned(run_dir, runner, signal.SIGKILL, deadline=time.monotonic(),
                             what="interrupted run cleanup", require_checkpoint_saved=False)
            runner.wait(timeout=scaled(5))
        if not retain_endpoint and endpoint.poll() is None:
            endpoint.terminate()
            try:
                endpoint.wait(timeout=scaled(5))
            except subprocess.TimeoutExpired:
                endpoint.kill()
                endpoint.wait(timeout=scaled(5))


def test_e16_local_shell_resume_replays_identical_next_request(tmp_path):
    repo, run_dir, expected = _interrupted_run(tmp_path)
    with Endpoint(FIXTURES / "scripts" / "resume_remaining.script.json") as endpoint:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr + str(summary)
    actual = read_requests(run_dir)[1]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]
    assert summary["succeeded"] is True
    assert 'VALUE = "after"' in (run_dir / "workspace" / "app.py").read_text()
    assert 'VALUE = "before"' in (repo / "app.py").read_text()


def test_continuation_resume_sends_uncrashed_prepared_request_bytes(tmp_path, monkeypatch):
    """Compare the provider body rebuilt on resume with the real pre-crash request."""
    monkeypatch.setenv("KERNEL_CONTINUATION_EXECUTIVE", "1")
    _, run_dir, expected = _interrupted_run(tmp_path)
    prepared = (run_dir / "wire" / "0002.request.json").read_bytes()
    prepared_body = json.loads(prepared)
    assert prepared_body["messages"] == expected["messages"]
    assert prepared_body["tools"] == expected["tools"]
    request_log = tmp_path / "resumed-provider-requests.jsonl"
    script = json.loads((FIXTURES / "scripts" / "resume_remaining.script.json").read_text())
    script["request_log"] = str(request_log)
    script_path = tmp_path / "resume-with-request-log.script.json"
    script_path.write_text(json.dumps(script))
    with Endpoint(script_path) as endpoint:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    sent = tuple(json.loads(line) for line in request_log.read_text().splitlines())
    assert sent, "resume did not send a provider request"
    assert json.dumps(sent[0]).encode() == prepared
    assert not (run_dir / "resume-divergence.json").exists()


def test_checkpoint_restores_and_verifies_paused_request(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "save")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_EVERY", "1")
    _, run_dir, expected, endpoint, url = _interrupted_run(tmp_path, wait_for_checkpoint=True)
    try:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=scaled(5))
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=scaled(5))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr + (run_dir / "engine.log").read_text()
    actual = read_requests(run_dir)[2]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]
    assert '"kind":"checkpoint_verified"' in (run_dir / "engine.log").read_text()


def test_shell_run_restores_checkpoint_after_shell_step(tmp_path, monkeypatch):
    """A shell-enabled run killed at request 3 resumes from a checkpoint taken after its shell
    step (not by full replay) and rebuilds the paused request byte for byte."""
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_EVERY", "1")
    _, run_dir, expected, endpoint, url = _interrupted_run(
        tmp_path, wait_for_checkpoint=True, shell="standard", pause_at=3)
    try:
        started = time.monotonic()
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet", "--keep-checkpoints"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
        seconds = time.monotonic() - started
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=scaled(5))
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=scaled(5))
    log = (run_dir / "engine.log").read_text()
    print(f"shell resume: rc={resumed.returncode} in {seconds:.1f} s")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr + log
    actual = read_requests(run_dir)[2]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]
    events = [json.loads(line) for path in sorted(run_dir.rglob("checkpoints/events.jsonl"))
              for line in path.read_text().splitlines()]
    restored = [row for row in events if row.get("event") == "checkpoint_restored"]
    assert restored and restored[-1]["sends"] >= 2, events
    assert '"kind":"checkpoint_verified"' in log
    assert '"replay_sends":0' in log  # nothing before the checkpoint was replayed
    assert not [row for row in events if row.get("event") in {"checkpoint_failed", "checkpoint_disabled"}]


def test_full_replay_verifies_check_receipt_request_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    _, run_dir, _ = _interrupted_run(
        tmp_path, script_name="checkpoint_resume.script.json",
        task="Set app.py VALUE to after and verify it")
    recorded = (run_dir / "wire" / "0002.request.json").read_bytes()
    assert b"check-execution:" in recorded and b"check-receipt:" in recorded
    remaining = json.loads((FIXTURES / "scripts" / "resume_remaining.script.json").read_text())
    remaining.pop("tokenize")
    script = tmp_path / "remaining.json"
    script.write_text(json.dumps(remaining))
    with Endpoint(script) as endpoint:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (run_dir / "wire" / "0002.request.json").read_bytes() == recorded
    assert not (run_dir / "resume-divergence.json").exists()


def test_checkpoint_verifies_check_receipt_request_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "save")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_EVERY", "1")
    tokenize_off = tmp_path / "tokenize_off"
    _, run_dir, _, endpoint, url = _interrupted_run(
        tmp_path, wait_for_checkpoint=True, pause_at=2,
        tokenize_off_marker=tokenize_off)
    recorded = (run_dir / "wire" / "0002.request.json").read_bytes()
    assert b"check-execution:" in recorded and b"check-receipt:" in recorded
    tokenize_off.write_text("off\n")
    try:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
    finally:
        endpoint.terminate()
        endpoint.wait(timeout=scaled(5))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (run_dir / "wire" / "0002.request.json").read_bytes() == recorded
    assert '"kind":"checkpoint_verified"' in (run_dir / "engine.log").read_text()
    assert not (run_dir / "resume-divergence.json").exists()


def test_checkpoint_off_on_resume_skips_restore_and_later_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "save")
    _, run_dir, _, endpoint, url = _interrupted_run(tmp_path, wait_for_checkpoint=True)
    try:
        env = environment()
        env["STATEFRAME_CHECKPOINT"] = "off"
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=env, expected_seconds=scaled(240), check=False)
    finally:
        endpoint.terminate()
        endpoint.wait(timeout=scaled(5))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads((run_dir / "spec.json").read_text())["environment"]["STATEFRAME_CHECKPOINT"] == "off"
    assert not (run_dir / "checkpoints").exists()
    assert '"kind":"checkpoint_verified"' not in (run_dir / "engine.log").read_text()


def test_checkpoint_hex_mismatch_refuses_after_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "save")
    original_hex = "a" * 32
    task = f"Set app.py VALUE to after and verify it. Reference {original_hex}."
    _, run_dir, _, endpoint, url = _interrupted_run(
        tmp_path, wait_for_checkpoint=True, checkpoint_task=task)
    pending = run_dir / "wire" / "0003.request.json"
    raw = pending.read_bytes()
    assert original_hex.encode() in raw
    pending.write_bytes(raw.replace(original_hex.encode(), b"b" * 32))
    try:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
    finally:
        endpoint.terminate()
        endpoint.wait(timeout=scaled(5))
    assert resumed.returncode != 0, resumed.stdout + resumed.stderr
    spec = json.loads((run_dir / "spec.json").read_text())
    assert "CheckpointUnfaithful" in spec["resume"]["checkpoint_refused"]
    assert (run_dir / "attempts" / "2" / "checkpoint-refused.json").is_file()
    assert "ReplayUnfaithful" in (run_dir / "engine.log").read_text()
    assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is False


def test_e23_full_network_shell_requires_replay_external(tmp_path):
    _, run_dir, _ = _interrupted_run(tmp_path, network="full")
    with Endpoint(FIXTURES / "scripts" / "resume_remaining.script.json") as endpoint:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(60), check=False)
    assert resumed.returncode != 0
    assert "--replay-external" in resumed.stderr + resumed.stdout
    with Endpoint(FIXTURES / "scripts" / "resume_remaining.script.json") as endpoint:
        approved = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--replay-external", "--quiet"],
            capture_output=True, text=True, env=environment(), expected_seconds=scaled(240), check=False)
    assert approved.returncode == 0, approved.stdout + approved.stderr
    assert json.loads((run_dir / "summary.json").read_text())["succeeded"] is True


def test_e19_volatile_shell_observation_is_byte_identical_on_resume(tmp_path):
    _, run_dir, expected = _interrupted_run(tmp_path, script_name="volatile.script.json", wait_for_saved=True)
    with Endpoint(FIXTURES / "scripts" / "volatile_remaining.script.json") as endpoint:
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(budget("e19_resume")), check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    actual = read_requests(run_dir)[1]
    assert actual["messages"] == expected["messages"]
    assert actual["tools"] == expected["tools"]
    # A shell step leaves a live execution session in the frame; saving must still succeed.
    saves = [json.loads(line) for path in run_dir.rglob("checkpoints/events.jsonl")
             for line in path.read_text().splitlines()]
    assert not [row for row in saves if row.get("event") in {"checkpoint_failed", "checkpoint_disabled"}], saves
    assert any(row.get("event") == "checkpoint_saved" for row in saves)


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
        resumed = run_monitored(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", endpoint.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), expected_seconds=scaled(240), check=False)
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
