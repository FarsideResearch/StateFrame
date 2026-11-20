"""Frozen legacy behavior for the workspace-state rollout."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from _journal import measurement_events
from tools.workspace_artifact_probe import compare_request_bodies, delivery_outcome, observe_artifacts

from stateframe import cli, runner
from stateframe.scripted_endpoint import Script, make_handler


@dataclass
class ProviderTrace:
    script: Script
    requests: list[tuple[int, bytes]] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)


WORKSPACE_SETTING = "STATEFRAME_WORKSPACE_OBJECTS"


def _recording_handler(script, trace):
    class Capture(make_handler(script)):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return super().do_POST()
            started = time.monotonic_ns()
            original = self.rfile.read

            def read(size):
                data = original(size)
                body = gzip.decompress(data) if self.headers.get("Content-Encoding") == "gzip" else data
                trace.requests.append((started, body))
                return data

            self.rfile.read = read
            try:
                return super().do_POST()
            finally:
                self.rfile.read = original

        def _send(self, status, payload):
            super()._send(status, payload)
            if self.path.rstrip("/") == "/v1/chat/completions" and status == 200:
                trace.responses.append(time.monotonic_ns())

    return Capture


@contextmanager
def scripted_provider(steps):
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    trace = ProviderTrace(script)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _recording_handler(script, trace))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", trace
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _oracle_run(tmp_path, monkeypatch, *, witness=None):
    examples = Path(__file__).resolve().parents[1] / "examples"
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    shutil.copytree(examples / "discount_bug", repo)
    if witness is not None:
        (repo / "inventory.bin").write_bytes(witness)
    steps = json.loads((examples / "discount_bug.script.json").read_text())["steps"]
    steps.insert(1, {"tool": "shell", "arguments": {"command": "touch forbidden.py"}})
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    monkeypatch.setattr(runner.secrets, "token_hex", lambda size=16: ("0123456789abcdef" * size)[:2 * size])

    with scripted_provider(steps) as (url, trace):
        result = cli.main([
            "run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
            "--endpoint", url, "--model", "scripted", "--run-dir", str(run), "--quiet",
            "--test-python", sys.executable, "--no-run-env",
            "--shell", "read-only", "--max-turns", "8", "--timeout", "2400",
        ])
    summary_path = run / "summary.json"
    assert summary_path.is_file(), f"scripted run exited with {result} after {len(trace.responses)} replies"
    summary = json.loads(summary_path.read_text())
    assert result == 0, summary
    trace.script.assert_consumed()
    assert len(trace.requests) == len(trace.responses) == 5
    return run, summary, [body for _, body in trace.requests]


LEGACY_READ = (
    "[read page] path=shop/pricing.py "
    "revision=sha256:74ed0a46a26be12ba133ef61742e1f1c773ff94381ec8c89ea6360d837952650 "
    "lines=1-13/13 requested=lines:1-120 effective=lines:1-13@0:48 clip=none "
    "chars=508 bytes=508 truncated=false decoded_with_replacement=false"
)
LEGACY_REFUSAL = "REFUSED READ_ONLY_POLICY: 'touch' is not an inspection program"
LEGACY_OLD_LINE = "    return round(price * percent / 100, 2)"
LEGACY_NEW_LINE = "    return round(price * (100 - percent) / 100, 2)"


def _visible_text(body):
    return "\n".join(
        message["content"]
        for message in json.loads(body)["messages"]
        if isinstance(message.get("content"), str)
    )


def _assert_legacy_outputs(run, summary, bodies):
    # Freeze workspace observations independently of host paths, clocks and
    # launch/resume turn limits. Existing OFF-byte tests pin the full envelope.
    assert LEGACY_READ in _visible_text(bodies[1]).splitlines()
    assert LEGACY_REFUSAL in _visible_text(bodies[2])
    assert LEGACY_NEW_LINE in _visible_text(bodies[3])
    assert b"check_status=PASS" in bodies[4]
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["succeeded"] is True

    workspace = run / "workspace"
    assert not (workspace / "forbidden.py").exists()
    original = (run.parent / "repo" / "shop" / "pricing.py").read_text()
    assert original.count(LEGACY_OLD_LINE) == 1
    assert (workspace / "shop" / "pricing.py").read_text() == original.replace(
        LEGACY_OLD_LINE, LEGACY_NEW_LINE,
    )
    patch = (run / "final.diff").read_text()
    assert [line for line in patch.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))] == [
        "-" + LEGACY_OLD_LINE, "+" + LEGACY_NEW_LINE,
    ]
    receipts = [event["receipt"] for event in measurement_events(run)
                if event.get("kind") == "action_receipt"]
    assert any((row["kind"], row["result_class"], row["detail"])
               == ("shell", "REFUSED", LEGACY_REFUSAL) for row in receipts)
    print("HTTP request SHA-256:", [hashlib.sha256(body).hexdigest() for body in bodies])


@pytest.mark.host_sandbox
def test_legacy_workspace_scripted_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKSPACE_SETTING, "off")
    _assert_legacy_outputs(*_oracle_run(tmp_path, monkeypatch))


@pytest.mark.host_sandbox
def test_legacy_workspace_checkpoint_restores_scripted_trajectory(tmp_path, monkeypatch):
    import test_checkpoint_b_resume_fidelity as checkpoint

    monkeypatch.setenv(WORKSPACE_SETTING, "off")
    wait_for_pause = checkpoint._wait_for_pause

    def wait_for_scripted_request(event, process, timeout, label):
        # Leave performance enforcement to the budget tests. A loaded host can
        # take several minutes to reach the checkpoint and the next HTTP send.
        wait_for_pause(event, process, max(timeout, 900), label)

    monkeypatch.setattr(checkpoint, "_wait_for_pause", wait_for_scripted_request)

    examples = Path(__file__).resolve().parents[1] / "examples"
    repo = tmp_path / "repo"
    shutil.copytree(examples / "discount_bug", repo)
    # The shared harness pauses the HTTP reply after the first read, waits for
    # a published save, and stops only the verified fixture-owned engine.
    outcome = checkpoint._run_interrupted(
        "discount_bug", repo, tmp_path / "run", checkpoint._write_launcher(tmp_path / "launch.py"),
        checkpoint._write_profile(tmp_path / "profile.json"), tmp_path / "home",
    )
    assert outcome.resumed_request == outcome.pending_request
    assert LEGACY_READ in _visible_text(outcome.resumed_request).splitlines()


@pytest.mark.host_sandbox
def test_workspace_switch_preserves_provider_and_delivery(tmp_path, monkeypatch):
    witness = random.Random(1009).randbytes(12 << 10)
    results = []
    for name, setting in (("default", None), ("off", "off"), ("on", "on")):
        folder = tmp_path / name
        with monkeypatch.context() as config:
            if setting is None:
                config.delenv(WORKSPACE_SETTING, raising=False)
            else:
                config.setenv(WORKSPACE_SETTING, setting)
            run, summary, requests = _oracle_run(folder, config, witness=witness)
            assert (run / ".workspace-objects.json").is_file() == (setting == "on")
            _assert_legacy_outputs(run, summary, requests)
            export = runner.final_candidate(run)
            assert export is not None and b"(100 - percent)" in export["shop/pricing.py"]
            outcome = delivery_outcome(run)
            assert outcome["engine_result"] == "SUBMITTED"
        results.append((requests, outcome, export))

    expected_requests, expected_outcome, expected_export = results[0]
    for requests, outcome, export in results[1:]:
        compare_request_bodies(expected_requests, requests)
        assert expected_outcome == outcome
        assert expected_export == export
    assert not (tmp_path / "default" / "home" / "workspace-state").exists()
    assert not (tmp_path / "off" / "home" / "workspace-state").exists()
    # These are the three live source worlds used by the runner: its frozen
    # baseline, the resumable execution checkout, and the isolated grade copy.
    # A retained checkpoint/candidate payload anywhere else is still a copy.
    sessions = list((run / "worker" / "continuation-workspaces").iterdir())
    assert len(sessions) == 1
    grading = list(sessions[0].glob("grading-*/inventory.bin"))
    assert len(grading) == 1
    live_sources = (run / "parent" / "inventory.bin",
                    sessions[0] / "repo" / "inventory.bin", grading[0])
    assert all(path.read_bytes() == witness for path in live_sources)
    sample = observe_artifacts(run, tmp_path / "on" / "home", payloads=(witness,),
                               fixture_inputs=live_sources)
    assert sample.candidates and not sample.invalid_records, sample.invalid_records
    assert sample.blob_copies == {hashlib.sha256(witness).hexdigest(): 1}, sample.counts()
    assert not sample.copies, sample.copies


@pytest.mark.host_sandbox
def test_workspace_objects_checkpoint_repeats_pending_provider_bytes(tmp_path, monkeypatch):
    import test_checkpoint_b_resume_fidelity as checkpoint

    environment = checkpoint._environment
    wait_for_pause = checkpoint._wait_for_pause
    wait_for_checkpoint = checkpoint._wait_for_checkpoint
    traces = []

    def object_environment(*args, **kwargs):
        return {**environment(*args, **kwargs), WORKSPACE_SETTING: "on"}

    def serve_recorded(script):
        trace = ProviderTrace(script)
        traces.append(trace)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _recording_handler(script, trace))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def wait_for_scripted_request(event, process, timeout, label):
        return wait_for_pause(event, process, max(timeout, 900), label)

    def wait_for_retained_checkpoint(run_dir, process, sends):
        retained = checkpoint._saved_checkpoint(run_dir, sends)
        if retained is not None:
            return retained
        events = checkpoint._rows(run_dir / "checkpoints" / "events.jsonl")
        failures = [event.get("failure_reason") or event.get("error") for event in events
                    if event.get("event") == "checkpoint_failed"]
        assert not failures, f"no checkpoint for pending send {sends + 1}: {failures[-1]}"
        return wait_for_checkpoint(run_dir, process, sends)

    monkeypatch.setattr(checkpoint, "_environment", object_environment)
    monkeypatch.setattr(checkpoint, "_wait_for_pause", wait_for_scripted_request)
    monkeypatch.setattr(checkpoint, "_wait_for_checkpoint", wait_for_retained_checkpoint)
    monkeypatch.setattr(checkpoint, "serve", serve_recorded)
    examples = Path(__file__).resolve().parents[1] / "examples"
    repo = tmp_path / "repo"
    shutil.copytree(examples / "discount_bug", repo)
    outcome = checkpoint._run_interrupted(
        "discount_bug", repo, tmp_path / "run", checkpoint._write_launcher(tmp_path / "launch.py"),
        checkpoint._write_profile(tmp_path / "profile.json"), tmp_path / "home",
    )
    assert len(traces) == 1
    requests = [body for _, body in traces[0].requests]
    assert len(requests) == 3, f"expected the first send and two copies of the pending send, got {len(requests)}"
    compare_request_bodies([requests[1]], [requests[2]], first_send=2)
    compare_request_bodies([outcome.pending_request], [outcome.resumed_request], first_send=2)
    assert LEGACY_READ in _visible_text(outcome.resumed_request).splitlines()
    assert outcome.checkpoint_meta["schema"] == "stateframe-engine-checkpoint/2"
    assert {path.name for path in outcome.checkpoint_dir.iterdir()} == {"checkpoint.json"}
