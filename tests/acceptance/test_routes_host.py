"""Host-only proof that the public CLI sends the first planned step to its route."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from ._support import Endpoint, PYTHON, ROOT, copy_repo, environment


pytestmark = [pytest.mark.host_sandbox,
              pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "src" / "stateframe" /
                                      "engine" / "kernel_frame_routes.py").is_file(),
                                 reason="routing engine lane not installed")]


def _kill_run(process: subprocess.Popen[str], run_dir: Path | None = None) -> None:
    """Kill the CLI and, like a machine crash, the engine: it runs in its own session and holds the run lock."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=5)
    if run_dir is None:
        return
    from stateframe.resume import ENGINE_LOCK, _lock_holder

    cutoff = time.monotonic() + 15
    while (holder := _lock_holder(run_dir / ENGINE_LOCK)) is not None and time.monotonic() < cutoff:
        try:
            os.killpg(os.getpgid(int(holder)), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass
        time.sleep(0.1)
    assert _lock_holder(run_dir / ENGINE_LOCK) is None, "engine survived the simulated crash"


def _crash_python(tmp_path: Path, point: str) -> tuple[Path, Path, Path]:
    """Inject a child-only hook through the public interpreter flag, then disarm on resume."""
    arm, marker, wrapper = (tmp_path / "crash-armed", tmp_path / "crash-ready",
                            tmp_path / "crash-python")
    arm.touch()
    hook_dir = Path(__file__).parent / "route_crash_hook"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"if [ -e {shlex.quote(str(arm))} ]; then\n"
        f"  ROUTE_CRASH_POINT={shlex.quote(point)} "
        f"ROUTE_CRASH_MARKER={shlex.quote(str(marker))} "
        f"PYTHONPATH={shlex.quote(str(hook_dir))} "
        f"exec {shlex.quote(PYTHON)} \"$@\"\n"
        "fi\n"
        f"exec {shlex.quote(PYTHON)} \"$@\"\n")
    wrapper.chmod(0o700)
    return wrapper, arm, marker


def test_public_cli_routes_first_plan_send_to_second_endpoint(tmp_path):
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    cheap_script = tmp_path / "cheap.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "steps": [
        {"tool": "read", "arguments": {"path": "app.py"}}]}))
    default_script = tmp_path / "default.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "steps": []}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap:
        process = subprocess.run([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Inspect app.py",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "3", "--quiet", "--route", f"plan.endpoint={cheap.url}",
            "--route", "plan.model=cheap", "--route", f"read.endpoint={cheap.url}",
            "--route", "read.model=cheap"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    spec = json.loads((run_dir / "spec.json").read_text())
    first = json.loads((run_dir / "wire" / "0001.route.json").read_text())
    assert spec["routes"]["schema"] == "stateframe-routes/1"
    assert first["role"] == "plan"
    assert first["model"] == "cheap"
    assert first["endpoint_host"] == cheap.url.removeprefix("http://").removesuffix("/v1")


def test_failed_check_replans_then_read_action_returns_to_read_route(tmp_path):
    repo = tmp_path / "discount_bug"
    shutil.copytree(ROOT / "examples" / "discount_bug", repo)
    run_dir = tmp_path / "run-failed-check"
    plan_script = tmp_path / "plan.script.json"
    plan_script.write_text(json.dumps({"model": "plan", "steps": [
        {"tool": "run_tests", "required": True,
         "arguments": {"target": "tests/test_pricing.py"}},
        {"tool": "read", "required": True,
         "arguments": {"path": "shop/pricing.py"}}]}))
    default_script = tmp_path / "default.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "steps": [
        {"tool": "read", "required": True,
         "arguments": {"path": "shop/pricing.py"}}]}))
    with Endpoint(default_script) as default, Endpoint(plan_script) as plan:
        process = subprocess.run([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Run tests/test_pricing.py, then inspect shop/pricing.py twice.",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--max-turns", "3", "--quiet",
            "--route", f"plan.endpoint={plan.url}", "--route", "plan.model=plan",
        ], capture_output=True, text=True, env=environment(), timeout=300, check=False)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    wire = run_dir / "wire"
    routes = [json.loads((wire / f"{index:04d}.route.json").read_text())
              for index in (1, 2, 3)]
    assert [route["role"] for route in routes] == ["plan", "plan", "read"]
    events = [json.loads(line) for path in (run_dir / "worker" / "measurement-runs").glob("*/events.jsonl")
              for line in path.read_text().splitlines()]
    assert any(event.get("kind") == "test_subprocess_completed" and event.get("exit_code") == 1
               for event in events)


def test_read_edit_reply_is_discarded_and_reasked_on_default(tmp_path):
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run-escalated"
    cheap_log, default_log = tmp_path / "cheap.jsonl", tmp_path / "default.jsonl"
    cheap_script = tmp_path / "cheap-escalate.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "request_log": str(cheap_log),
        "steps": [
            {"tool": "read", "arguments": {"path": "app.py"}},
            {"tool": "file_create", "arguments": {"path": "discarded.txt", "content": "wrong model"}},
            {"tool": "done", "arguments": {"summary": "complete"}}]}))
    default_script = tmp_path / "default-escalate.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "request_log": str(default_log),
        "steps": [
            {"tool": "file_create", "arguments": {"path": "accepted.txt", "content": "right model"}},
            {"tool": "done", "arguments": {"summary": "complete"}}]}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap:
        process = subprocess.run([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Read app.py, then create accepted.txt containing right model.",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "5", "--quiet", "--route", f"plan.endpoint={cheap.url}",
            "--route", "plan.model=cheap", "--route", f"read.endpoint={cheap.url}",
            "--route", "read.model=cheap"], capture_output=True, text=True,
            env=environment(), timeout=300, check=False)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    cheap_requests = [json.loads(line) for line in cheap_log.read_text().splitlines()]
    default_requests = [json.loads(line) for line in default_log.read_text().splitlines()]
    assert len(cheap_requests) >= 2 and default_requests
    assert all(request["model"] == "cheap" for request in cheap_requests)
    assert all(request["model"] == "scripted" for request in default_requests)
    import re

    def frame(request):
        """The typed frame, and the live owner budget line, which leg two's own charge changes by design."""
        texts = [message["content"] for message in request["messages"]
                 if isinstance(message.get("content"), str)
                 and message["content"].startswith("[StateFrame]\n")]
        assert len(texts) == 1
        parsed = json.loads(texts[0].split("\n", 1)[1])
        limits = parsed["state"]["framefit5"]["owner_protocol"].pop("continuation_limits")
        decisions = int(re.search(r"model_decisions_remaining=(\d+)", limits).group(1))
        return parsed, decisions
    first_frame, first_decisions = frame(cheap_requests[1])
    second_frame, second_decisions = frame(default_requests[0])
    assert first_frame == second_frame  # the typed frame is the handoff between models
    assert first_decisions - second_decisions == 1  # leg two charged its own model decision
    wire = run_dir / "wire"
    routes = [json.loads(path.read_text()) for path in sorted(wire.glob("*.route.json"))]
    cheap_host = cheap.url.removeprefix("http://").removesuffix("/v1")
    default_host = default.url.removeprefix("http://").removesuffix("/v1")
    assert routes[0]["role"] == "plan" and routes[0]["endpoint_host"] == cheap_host
    assert any(route["role"] == "edit" and route["endpoint_host"] == default_host
               for route in routes)
    event_files = list((run_dir / "worker" / "measurement-runs").glob("*/events.jsonl"))
    events = [json.loads(line) for path in event_files for line in path.read_text().splitlines()]
    discarded = [event for event in events if event.get("kind") == "route_escalation_discarded"]
    assert discarded
    receipts = [event for event in events if "receipt" in event.get("kind", "")]
    retained = [event for event in events if event.get("kind", "").startswith("passive_retention")]
    assert "discarded.txt" not in json.dumps(receipts)
    assert "discarded.txt" not in json.dumps(retained)
    assert any(event.get("kind") == "response_received" and
               event.get("request_id") == discarded[0].get("request_id") for event in events)
    assert "discarded.txt" not in (run_dir / "final.diff").read_text()
    assert "accepted.txt" in (run_dir / "final.diff").read_text()
    # Only the accepted leg can become the newest exchange on the next request.
    assert len(cheap_requests) >= 3
    assert "discarded.txt" not in json.dumps(cheap_requests[2]["messages"])
    tool_tape = run_dir / "tool-tape.jsonl"
    if tool_tape.is_file():
        assert "discarded.txt" not in tool_tape.read_text()


def test_killed_second_leg_replays_first_leg_and_verifies_pending_bytes(tmp_path):
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run-killed"
    cheap_script = tmp_path / "cheap-killed.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "steps": [
        {"tool": "file_create", "arguments": {"path": "discarded.txt", "content": "wrong model"}}]}))
    default_script = tmp_path / "default-killed.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "steps": [
        {"tool": "file_create", "arguments": {"path": "accepted.txt", "content": "right model"},
         "repeat": True, "delay_s": 5}]}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap:
        routes = ["--route", f"plan.endpoint={cheap.url}", "--route", "plan.model=cheap",
                  "--route", f"read.endpoint={cheap.url}", "--route", "read.model=cheap"]
        process = subprocess.Popen([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Create accepted.txt containing right model.",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "3", "--quiet", *routes], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=environment(), start_new_session=True)
        pending_route = run_dir / "wire" / "0002.route.json"
        pending_request = run_dir / "wire" / "0002.request.json"
        cutoff = time.monotonic() + 120
        try:
            while time.monotonic() < cutoff and process.poll() is None and not pending_request.is_file():
                time.sleep(0.02)
            assert pending_route.is_file() and pending_request.is_file(), process.communicate(timeout=3)
            first = (run_dir / "wire" / "0001.request.json").read_bytes()
            second = pending_request.read_bytes()
            route_bytes = pending_route.read_bytes()
            recorded_routes = json.loads((run_dir / "spec.json").read_text())["routes"]
            _kill_run(process, run_dir)
            resumed = subprocess.run([PYTHON, "-m", "stateframe", "resume", str(run_dir),
                                      "--quiet"], capture_output=True, text=True,
                                      env=environment(), timeout=300, check=False)
        finally:
            if process.poll() is None:
                _kill_run(process, run_dir)
    assert resumed.returncode in (0, 1), resumed.stdout + resumed.stderr
    assert (run_dir / "wire" / "0001.request.json").read_bytes() == first
    assert (run_dir / "wire" / "0002.request.json").read_bytes() == second
    assert (run_dir / "wire" / "0002.route.json").read_bytes() == route_bytes
    assert json.loads((run_dir / "spec.json").read_text())["routes"] == recorded_routes
    assert "discarded.txt" not in (run_dir / "final.diff").read_text()
    assert "accepted.txt" in (run_dir / "final.diff").read_text()


@pytest.mark.parametrize("crash_point", ["after_discard", "after_route:2"])
def test_routed_crash_boundary_replays_with_same_route_and_request_bytes(tmp_path, crash_point):
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run-crash"
    crash_python, arm, marker = _crash_python(tmp_path, crash_point)
    cheap_script = tmp_path / "cheap-crash.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "steps": [
        {"tool": "file_create", "arguments": {"path": "discarded.txt", "content": "wrong"}}]}))
    default_script = tmp_path / "default-crash.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "steps": [
        {"tool": "file_create", "arguments": {"path": "accepted.txt", "content": "right"}}]}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap:
        routes = ["--route", f"plan.endpoint={cheap.url}", "--route", "plan.model=cheap",
                  "--route", f"read.endpoint={cheap.url}", "--route", "read.model=cheap"]
        process = subprocess.Popen([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Create accepted.txt containing right.",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--python", str(crash_python), "--test-python", PYTHON,
            "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "3", "--quiet", *routes], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            env=environment(),
            start_new_session=True)
        wire = run_dir / "wire"
        route2 = wire / "0002.route.json"
        request2 = wire / "0002.request.json"
        cutoff = time.monotonic() + 120
        try:
            while time.monotonic() < cutoff and process.poll() is None and not marker.is_file():
                time.sleep(0.02)
            assert marker.is_file(), process.communicate(timeout=3)
            assert process.poll() is None, process.communicate(timeout=3)
            assert (wire / "0001.request.json").is_file()
            if crash_point == "after_route:2":
                assert route2.is_file() and not request2.exists()
            else:
                assert not route2.exists()
            saved = {path.name: path.read_bytes() for path in wire.glob("*.route.json")}
            saved.update({path.name: path.read_bytes() for path in wire.glob("*.request.json")})
            _kill_run(process, run_dir)
            arm.unlink()
            resumed = subprocess.run([PYTHON, "-m", "stateframe", "resume", str(run_dir),
                                      "--quiet", *routes], capture_output=True, text=True,
                                     env=environment(), timeout=300, check=False)
        finally:
            if process.poll() is None:
                _kill_run(process, run_dir)
    assert resumed.returncode in (0, 1), resumed.stdout + resumed.stderr
    for name, data in saved.items():
        assert (wire / name).read_bytes() == data
    assert request2.is_file()
    assert "discarded.txt" not in (run_dir / "final.diff").read_text()
    assert "accepted.txt" in (run_dir / "final.diff").read_text()


def test_escalation_without_second_response_allowance_stops_before_dispatch(tmp_path):
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run-no-allowance"
    cheap_script = tmp_path / "cheap-no-allowance.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "steps": [
        {"tool": "file_create", "arguments": {"path": "discarded.txt", "content": "wrong"}}]}))
    default_log = tmp_path / "default-no-allowance.jsonl"
    default_script = tmp_path / "default-no-allowance.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "request_log": str(default_log),
        "steps": [{"tool": "file_create", "arguments": {
            "path": "accepted.txt", "content": "right"}}]}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap:
        process = subprocess.run([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Create accepted.txt containing right.",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "1", "--quiet", "--route", f"plan.endpoint={cheap.url}",
            "--route", "plan.model=cheap"], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    assert (run_dir / "wire" / "0001.route.json").is_file()
    assert not (run_dir / "wire" / "0002.request.json").exists()
    assert not default_log.exists()
    assert "discarded.txt" not in (run_dir / "final.diff").read_text()
    event_files = list((run_dir / "worker" / "measurement-runs").glob("*/events.jsonl"))
    events = [json.loads(line) for path in event_files for line in path.read_text().splitlines()]
    retained = [event for event in events if event.get("kind", "").startswith("passive_retention")]
    assert "discarded.txt" not in json.dumps(retained)


def test_routed_tokenizer_survives_sealing_on_a_separate_server(tmp_path):
    # Regression (gate 3): final qualification replaced a route's tokenizer with its chat endpoint.
    # The routed chat server has no /tokenize, so only the route's own tokenizer can count it.
    repo = copy_repo(tmp_path)
    run_dir = tmp_path / "run"
    tok_log, base_tok_log = tmp_path / "tok.jsonl", tmp_path / "base-tok.jsonl"
    cheap_script = tmp_path / "cheap.script.json"
    cheap_script.write_text(json.dumps({"model": "cheap", "steps": [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "read", "arguments": {"path": "app.py"}}]}))
    tok_script = tmp_path / "tok.script.json"
    tok_script.write_text(json.dumps({"model": "cheap", "steps": [], "tokenize": {"mode": "ok"},
                                      "tokenize_log": str(tok_log)}))
    default_script = tmp_path / "default.script.json"
    default_script.write_text(json.dumps({"model": "scripted", "steps": [], "tokenize": {"mode": "ok"},
                                          "tokenize_log": str(base_tok_log)}))
    with Endpoint(default_script) as default, Endpoint(cheap_script) as cheap, Endpoint(tok_script) as tok:
        routes = []
        for role in ("plan", "read"):
            routes += ["--route", f"{role}.endpoint={cheap.url}", "--route", f"{role}.model=cheap",
                       "--route", f"{role}.tokenizer={tok.url}"]
        process = subprocess.run([
            PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Inspect app.py",
            "--endpoint", default.url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", PYTHON, "--test-cmd", "python -m unittest discover -s tests -q",
            "--max-turns", "2", "--quiet", *routes], capture_output=True, text=True,
            env=environment(), timeout=240, check=False)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    roles = [json.loads(path.read_text())["role"] for path in sorted((run_dir / "wire").glob("*.route.json"))]
    assert roles and set(roles) <= {"plan", "read"}
    assert tok_log.is_file() and tok_log.read_text().strip()
    assert '"kind":"token_counting_unavailable"' not in (run_dir / "engine.log").read_text()


_RECEIPT = re.compile(rb'(edit_receipt\\*":\\*")[0-9a-f]{20}')


def _default_wire(tmp_path: Path, monkeypatch, name: str, *extra: str) -> tuple[list[bytes], Path]:
    """One fixed-seed scripted run in process; returns each request with edit-receipt ids masked."""
    from stateframe import cli, runner

    monkeypatch.setattr(runner.secrets, "token_hex", lambda n=16: "0123456789abcdef" * 2)
    run_dir = tmp_path / name
    with Endpoint(ROOT / "examples" / "discount_bug.script.json") as server:
        code = cli.main(["run", "--repo", str(tmp_path / "discount_bug"), "--task",
                         "Fix the pricing code so the tests pass.", "--endpoint", server.url,
                         "--model", "scripted", "--run-dir", str(run_dir), "--quiet", "--json", *extra])
    assert code == 0, name
    return ([_RECEIPT.sub(rb"\1<RECEIPT>", path.read_bytes())
             for path in sorted((run_dir / "wire").glob("*.request.json"))], run_dir)


def test_inactive_routes_keep_requests_byte_identical(tmp_path, monkeypatch, capsys):
    """No route settings and the all-commented example file both leave the default path untouched."""
    shutil.copytree(ROOT / "examples" / "discount_bug", tmp_path / "discount_bug")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in [key for key in os.environ if key.startswith("STATEFRAME_ROUTE")]:
        monkeypatch.delenv(variable)
    plain, plain_dir = _default_wire(tmp_path, monkeypatch, "plain")
    example, example_dir = _default_wire(tmp_path, monkeypatch, "example", "--routes",
                                         str(ROOT / "examples" / "routes.example.toml"))
    capsys.readouterr()
    assert plain and plain == example
    for run_dir in (plain_dir, example_dir):
        assert "routes" not in json.loads((run_dir / "spec.json").read_text())
        assert not list((run_dir / "wire").glob("*.route.json"))
