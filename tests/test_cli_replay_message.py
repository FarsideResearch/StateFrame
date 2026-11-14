"""Replay refusals retain diagnostics and tell users how to recover."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

import kernel_bounded_http
import kernel_cache_layout_candidate as layout_candidate
from stateframe import _child, cli, replay_tape, runner
from stateframe.scripted_endpoint import Script, serve

GUIDANCE = "This recording can't be replayed by this version. Start a fresh run. "
URL = "http://127.0.0.1:1/v1/chat/completions"
WIRE = b'{"model": "scripted", "messages": []}'
ROUTE = {"role": "read", "endpoint": URL, "model": "scripted"}


@pytest.mark.parametrize("site", ["earlier_request", "earlier_route", "routed", "checkpoint", "replay"])
def test_each_recorder_refusal_has_actionable_message(tmp_path, monkeypatch, site):
    request = tmp_path / "recorded.json"
    request.write_bytes(WIRE)
    response = tmp_path / "response.json"
    response.write_bytes(b'{"choices":[{}]}')
    tape = [{"send": 1, "request": str(request), "response": str(response),
             "request_digest": replay_tape.normalized_digest(WIRE),
             "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest()}]
    if site == "routed":
        tape[0]["route"] = {**ROUTE, "role": "plan"}
    player = replay_tape.Replayer(tape, strict=site in {"checkpoint", "earlier_request"})
    if site == "earlier_request":
        player.diverged = {"reason": "previous mismatch"}
        detail = "an earlier rebuilt request differed; nothing more is sent"
    elif site == "earlier_route":
        player.route_mismatch = True
        detail = "an earlier routed replay differed; nothing more is sent"
    elif site == "routed":
        detail = "send 1: routed send destination or model differs"
    else:
        detail = "send 1: the engine's rebuilt request differs from the recorded request"
    sent = []
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *a, **k: sent.append(a))
    for name in ("_REPLAY_LAYOUT_DECISION_PROVIDER", "_REPLAY_STABLE_WIRE_DECISION_PROVIDER",
                 "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER"):
        monkeypatch.setattr(layout_candidate, name, getattr(layout_candidate, name))
    _child.install_wire_recorder(tmp_path, URL, player, {URL} if site == "routed" else None)
    exception = replay_tape.ReplayUnfaithful if site == "replay" else replay_tape.CheckpointUnfaithful
    with pytest.raises(exception) as caught:
        kernel_bounded_http.request_bytes(
            WIRE.replace(b"[]", b"[1]"), URL, "", **({"route_meta": ROUTE} if site == "routed" else {}))
    assert str(caught.value) == GUIDANCE + detail
    assert sent == []


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_resume_divergence_guides_user_without_traceback(tmp_path, monkeypatch, capsys, json_output):
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    shutil.copytree(root / "examples/discount_bug", repo)
    run = tmp_path / "run"
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    server = serve(Script(json.loads((root / "examples/discount_bug.script.json").read_text())))
    try:
        assert cli.main(["run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
                  "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
                  "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "1", "--quiet"]) == 1
        capsys.readouterr()
        assert list((run / "wire").glob("*.response.json"))
        # Model an interrupted attempt using its real recorded exchange. Budget stops
        # require a larger turn limit, which deliberately restarts live instead of replaying.
        (run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}))
        # A valid recording whose old task bytes cannot match this version's rebuilt request.
        spec_path = run / "spec.json"
        spec = json.loads(spec_path.read_text())
        spec["public_row"]["msg"] += " A different task recorded by an older version."
        spec_path.write_text(json.dumps(spec))
        assert cli.main(["resume", str(run),
                         "--json" if json_output else "--quiet"]) == 2
        output = capsys.readouterr()
        assert GUIDANCE.strip() in output.out
        assert "send 1: the engine's rebuilt request differs from the recorded request" in output.out
        assert "Traceback" not in output.out + output.err
        assert "continue it with:" not in output.out
        summary = json.loads((run / "summary.json").read_text())
        assert summary["engine_error_reason"].startswith(GUIDANCE)
        if json_output:
            assert json.loads(output.out)["engine_error_reason"] == summary["engine_error_reason"]
        assert "ReplayUnfaithful" in (run / "child-traceback.txt").read_text()
    finally:
        server.shutdown()
        server.server_close()


def test_truncated_replay_detail_keeps_guidance_in_live_and_summary_output(tmp_path, monkeypatch, capsys):
    detail = (GUIDANCE + "send 3: " + "long diagnostic " * 40)[:300]
    child = tmp_path / "boundary.py"
    event = {"kind": "engine_error", "error": "ReplayUnfaithful", "detail": detail}
    child.write_text("print(" + repr(runner.LOG_PREFIX + json.dumps(event)) + ")\nraise SystemExit(1)\n")
    monkeypatch.setattr(runner, "CHILD", child)
    assert runner.drive(tmp_path, runner.RunOptions(repo=tmp_path, task="Resume", endpoint=None, quiet=True),
                        print) == 1
    output = capsys.readouterr()
    assert GUIDANCE.strip() in output.out
    assert "send 3: long diagnostic" in output.out
    assert "Traceback" not in output.out + output.err
    (tmp_path / "engine-result.json").write_text(json.dumps({
        "result": {"result": "ENGINE_ERROR", "error": "ReplayUnfaithful"}}))
    assert runner.summarize(tmp_path, time.monotonic())["engine_error_reason"] == detail


@pytest.mark.parametrize("error", [{"code": "unrelated"}, ["unrelated"], None])
def test_summary_ignores_unrelated_terminal_error_payload(tmp_path, error):
    (tmp_path / "engine-result.json").write_text(json.dumps({
        "result": {"result": "ENGINE_ERROR", "error": error}}))
    assert "engine_error_reason" not in runner.summarize(tmp_path, time.monotonic())
