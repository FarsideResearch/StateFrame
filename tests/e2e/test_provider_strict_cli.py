"""A strict OpenAI-shaped mock rejects unknown keys during a real CLI run."""

from __future__ import annotations

import json
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from _timeouts import scaled
from _journal import measurement_events
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, make_handler, serve


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.host_sandbox


@pytest.mark.parametrize("reasoning_effort", [None, "high"])
def test_shipped_profile_cli_completes_two_turns_on_strict_openai_mock(
        tmp_path, reasoning_effort):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    script = Script({
        "model": "strict-mock",
        "strict_openai": True,
        "steps": [
            {"tool": "read", "arguments": {"path": "app.py"}},
            {"tool": "done", "arguments": {"summary": "app.py sets VALUE to seven."}},
        ],
    })
    server = serve(script)
    run_dir = tmp_path / "run"
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("STATEFRAME_", "REACT_"))
    }
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if reasoning_effort is not None:
        environment["REACT_REASONING_EFFORT"] = reasoning_effort
    try:
        process = run_monitored(
            [
                sys.executable, "-m", "stateframe", "run",
                "--repo", str(repo),
                "--task", "Read app.py and report VALUE without changing files.",
                "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", "strict-mock",
                "--run-dir", str(run_dir),
                "--max-turns", "8",
                "--quiet",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            expected_seconds=240,
            run_dir=run_dir,
            check=False,
        )
        output = process.stdout + process.stderr
        traceback_path = run_dir / "child-traceback.txt"
        if traceback_path.is_file():
            output += "\n" + traceback_path.read_text(encoding="utf-8")
        if ("could not start in the isolated startup probe" in output
                and "sandbox_apply: Operation not permitted" in output):
            pytest.skip("sandbox-exec blocks engine startup; rerun this test on the host")
    finally:
        server.shutdown()
        server.server_close()

    first_line = process.stdout.splitlines()[0] if process.stdout.splitlines() else ""
    assert process.returncode == 0, output
    assert first_line.startswith("capabilities:")
    assert "provider dialect: openai-strict" in first_line
    assert "openai-strict because /props was 404" in first_line
    reasoning_disposition = (
        "reasoning controls omitted; proposal side-calls refused" if reasoning_effort is None
        else f"reasoning controls mapped to reasoning_effort={reasoning_effort}"
    )
    assert reasoning_disposition in first_line
    assert "provider dialects" in output and reasoning_disposition in output
    # One read, then three done replies: the shipped profile re-asks done twice on this task. The base
    # engine without strict-endpoint support makes the same 4 requests for this script.
    assert len(script.requests) == 4, output
    for request in script.requests:
        if reasoning_effort is None:
            assert "reasoning_effort" not in request
        else:
            assert request["reasoning_effort"] == reasoning_effort
    assert not script.strict_openai_errors
    # The strict mock has neither /props proof nor /slots proof. Its automatic profile pin
    # degrades once, is journaled, and leaves OpenAI-compatible request bodies.
    degradation_line = "cache: prompt-cache pin dropped because provider did not prove llama.cpp /slots"
    assert process.stdout.count(degradation_line) == 1
    assert "prompt-cache slot: degraded: provider did not prove llama.cpp /slots" in process.stdout
    assert all("id_slot" not in request and "cache_prompt" not in request for request in script.requests)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["prompt_cache_slot"]["degraded"] is True
    assert summary["prompt_cache_slot"]["reason"] == (
        "provider did not prove llama.cpp /slots; optional prompt-cache pin dropped"
    )
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["environment"]["STATEFRAME_PROMPT_CACHE"] == "off"
    journal_events = measurement_events(run_dir)
    notices = [event for event in journal_events if event.get("kind") == "provider_feature_degraded"]
    assert len(notices) == 1
    assert notices[0]["feature"] == "prompt-cache pinned slot"
    assert notices[0]["reason"] == summary["prompt_cache_slot"]["reason"]
    script.assert_consumed()


def test_slots_proof_honors_automatic_pin_when_props_is_missing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    script = Script({
        "model": "slots-mock",
        "slots": [65536],
        "steps": [
            {"tool": "read", "arguments": {"path": "app.py"}},
            {"tool": "done", "arguments": {"summary": "app.py sets VALUE to seven."}},
        ],
    })
    handler = make_handler(script)
    normal_get = handler.do_GET

    def missing_props(self):
        if self.path.rstrip("/") == "/props":
            self._send(404, {"error": {"message": "not found"}})
        else:
            normal_get(self)

    handler.do_GET = missing_props
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run_dir = tmp_path / "run"
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("STATEFRAME_", "REACT_"))
    }
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    try:
        process = run_monitored(
            [
                sys.executable, "-m", "stateframe", "run",
                "--repo", str(repo),
                "--task", "Read app.py and report VALUE without changing files.",
                "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", "slots-mock",
                "--run-dir", str(run_dir),
                "--max-turns", "8",
                "--quiet",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            expected_seconds=240,
            run_dir=run_dir,
            check=False,
        )
        output = process.stdout + process.stderr
        traceback_path = run_dir / "child-traceback.txt"
        if traceback_path.is_file():
            output += "\n" + traceback_path.read_text(encoding="utf-8")
        if ("could not start in the isolated startup probe" in output
                and "sandbox_apply: Operation not permitted" in output):
            pytest.skip("sandbox-exec blocks engine startup; rerun this test on the host")
    finally:
        server.shutdown()
        thread.join(timeout=scaled(2))
        server.server_close()

    assert process.returncode == 0, output
    first_line = process.stdout.splitlines()[0] if process.stdout.splitlines() else ""
    assert "provider dialect: llama.cpp (proven by /slots)" in first_line
    assert "cache: pinned slot 0 (cache_prompt on)" in process.stdout
    assert "cache: prompt-cache pin dropped" not in process.stdout
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["prompt_cache_slot"]["slot"] == 0
    assert summary["prompt_cache_slot"]["selection"] == "automatic_idle"
    assert "degraded" not in summary["prompt_cache_slot"]
    capability = next(iter(summary["provider_capabilities"].values()))
    assert capability["reason"] == "proven by /slots"
    assert capability["slots"] is True
    assert capability["slot_ids"] == [0]
    assert len(script.requests) == 4
    assert all(request["id_slot"] == 0 and request["cache_prompt"] is True
               for request in script.requests)
    assert not script.strict_openai_errors
    script.assert_consumed()


def test_explicit_cli_slot_still_exits_two_without_slot_proof(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    script = Script({"model": "strict-mock", "strict_openai": True, "steps": []})
    server = serve(script)
    run_dir = tmp_path / "run"
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("STATEFRAME_", "REACT_"))
    }
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    try:
        process = run_monitored(
            [
                sys.executable, "-m", "stateframe", "run",
                "--repo", str(repo),
                "--task", "Read app.py and report VALUE without changing files.",
                "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", "strict-mock",
                "--run-dir", str(run_dir),
                "--slot", "0",
                "--max-turns", "8",
                "--quiet",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            expected_seconds=240,
            run_dir=run_dir,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()

    output = process.stdout + process.stderr
    assert process.returncode == 2, output
    assert "provider_capability_missing" in output
    assert "prompt-cache pinned slot requires proven" in output
    assert script.requests == []


def test_operator_override_is_visible_when_props_is_unavailable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    script = Script({
        "model": "override-mock",
        "steps": [
            {"tool": "read", "arguments": {"path": "app.py"}},
            {"tool": "done", "arguments": {"summary": "app.py sets VALUE to seven."}},
        ],
    })
    handler = make_handler(script)
    normal_get = handler.do_GET

    def missing_props(self):
        if self.path.rstrip("/") == "/props":
            self._send(404, {"error": {"message": "not found"}})
        else:
            normal_get(self)

    handler.do_GET = missing_props
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run_dir = tmp_path / "run"
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("STATEFRAME_", "REACT_"))
    }
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATEFRAME_PROVIDER": "llama.cpp",
    })
    try:
        process = run_monitored(
            [
                sys.executable, "-m", "stateframe", "run",
                "--repo", str(repo),
                "--task", "Read app.py and report VALUE without changing files.",
                "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", "override-mock",
                "--run-dir", str(run_dir),
                "--max-turns", "8",
                "--quiet",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            expected_seconds=240,
            run_dir=run_dir,
            check=False,
        )
        output = process.stdout + process.stderr
        traceback_path = run_dir / "child-traceback.txt"
        if traceback_path.is_file():
            output += "\n" + traceback_path.read_text(encoding="utf-8")
        if ("could not start in the isolated startup probe" in output
                and "sandbox_apply: Operation not permitted" in output):
            pytest.skip("sandbox-exec blocks engine startup; rerun this test on the host")
    finally:
        server.shutdown()
        thread.join(timeout=scaled(2))
        server.server_close()

    first_line = process.stdout.splitlines()[0] if process.stdout.splitlines() else ""
    assert process.returncode == 0, output
    assert "provider dialect: llama.cpp" in first_line
    assert "operator override, /props answered 404" in first_line
    assert "llama.cpp reasoning controls unchanged" in first_line
    assert "provider dialects" in output
    assert "operator override, /props answered 404" in output
    assert "llama.cpp reasoning controls unchanged" in output
    # One read, then three done replies: the shipped profile re-asks done twice on this task. The base
    # engine without strict-endpoint support makes the same 4 requests for this script.
    assert len(script.requests) == 4, output
    assert "reasoning_budget_tokens" in script.requests[0]
    script.assert_consumed()
