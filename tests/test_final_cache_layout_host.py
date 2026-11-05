"""The scripted request must carry the cache layout on its sealed provider wire."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_cache_layout_view import CacheLayoutError, FINAL_LAYOUT_NAME, semantic_messages
from kernel_stateframe_page_log import _payload_has_authority_keys
from _journal import measurement_events
from stateframe.scripted_endpoint import Script, serve
from tools.immediate_baseline import baseline_checkout  # noqa: F401  (pytest fixture)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox
# Per scripted run. tiny_site measured ~420 s solo for the whole parity test on the foundry host;
# a 240 s per-call limit flaked under full-suite load, so allow ~1.5x the solo time per call.
RUN_TIMEOUT_SECONDS = 600


def _parity_launchers(tmp_path):
    launcher = tmp_path / "parity_launcher.py"
    wrapper = tmp_path / "parity_child.py"
    launcher.write_text(
        "import secrets, subprocess, sys\n"
        "from stateframe import cli\n"
        "wrapper, *args = sys.argv[1:]\n"
        "original = secrets.token_hex\n"
        "used = False\n"
        "def fixed_hex(size=None):\n"
        "    global used\n"
        "    if size == 16 and not used:\n"
        "        used = True\n"
        "        return 'a' * 32\n"
        "    return original(size)\n"
        "secrets.token_hex = fixed_hex\n"
        "popen = subprocess.Popen\n"
        "def child(command, *positional, **keywords):\n"
        "    if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "        command = [command[0], command[1], wrapper, command[2], command[3]]\n"
        "    return popen(command, *positional, **keywords)\n"
        "subprocess.Popen = child\n"
        "raise SystemExit(cli.main(args))\n")
    wrapper.write_text(
        "import os, runpy, sys\n"
        "child, run = sys.argv[1:]\n"
        "sys.path.insert(0, os.path.dirname(child))\n"
        "sys.argv = [child, run]\n"
        "runpy.run_path(child, run_name='__main__')\n")
    return launcher, wrapper


@pytest.mark.parametrize("example,task", [
    ("discount_bug", "Fix the pricing code so the tests pass."),
    ("tiny_site", "Build the small static shop site described in the README."),
    ("brief_to_report", "Write the report the brief asks for."),
])
def test_switch_off_bytes_match_base_release(tmp_path, baseline_checkout, example, task):
    # Immediate base release (SF_TEST_BASELINE_IMMEDIATE, else the merge base); see tools/immediate_baseline.py.
    baseline = baseline_checkout
    assert (baseline / "src/stateframe/engine/ENGINE_SOURCE.json").is_file()
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / example, repo)
    launcher, wrapper = _parity_launchers(tmp_path)
    outputs = []
    for checkout, label in ((baseline, "baseline"), (ROOT, "current")):
        script = Script(json.loads((ROOT / "examples" / f"{example}.script.json").read_text()))
        server = serve(script)
        run = tmp_path / "run"
        environment = {**os.environ, "PYTHONPATH": str(checkout / "src")}
        # Pinned off on both sides: switch-off bytes must match the immediate base release.
        environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = "off"
        environment["STATEFRAME_PAGE_LOG"] = "off"
        try:
            result = subprocess.run(
                [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
                 "--task", task, "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                 "--model", "scripted", "--run-dir", str(run),
                 "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
                env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS)
            assert result.returncode == 0, result.stdout + result.stderr
            script.assert_consumed()
            outputs.append([path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))])
            if label == "baseline":
                run.rename(tmp_path / "baseline_run")
        finally:
            server.shutdown()
            server.server_close()
    assert outputs[0]
    # The receipt-determinism change (11-03) moved request_id and the receipt IDs derived
    # like it; every other byte must still match the base release.
    from tools.compare_no_mcp_requests import (
        DERIVED_RESULT_FIELDS, MOVED_ID_FIELDS, planted_body_change_is_caught,
        request_id_handle_mismatches, unbacked_result_hash_changes, wire_id_drift)
    problems, kinds = wire_id_drift(outputs[0], outputs[1])
    assert not problems, problems
    assert set(kinds) <= MOVED_ID_FIELDS | DERIVED_RESULT_FIELDS, kinds
    assert unbacked_result_hash_changes(tmp_path / "baseline_run", tmp_path / "run") == []
    assert planted_body_change_is_caught(outputs[1][0])
    assert request_id_handle_mismatches(tmp_path / "run") == []


def _scripted_wire(tmp_path, label, layout, *, example="discount_bug",
                   task="Fix the pricing code so the tests pass.", tokenize=True,
                   return_details=False):
    """Request bodies from one scripted example run; None leaves the switch unset."""
    # Same repo and run paths for every arm (paths reach the request bytes); renamed afterwards.
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / example, repo)
    launcher, wrapper = _parity_launchers(tmp_path / f"launch-{label}")
    script_document = json.loads((ROOT / "examples" / f"{example}.script.json").read_text())
    if not tokenize:
        script_document.pop("tokenize", None)
    script = Script(script_document)
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment.pop("STATEFRAME_FINAL_CACHE_LAYOUT", None)
    if layout is not None:
        environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = layout
    environment["STATEFRAME_PAGE_LOG"] = "off"
    try:
        result = subprocess.run(
            [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
             "--task", task,
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run),
             "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS)
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    events = measurement_events(run)
    assert not [row for row in events if (row.get("kind") or row.get("event")) == "final_cache_layout_failed_open"]
    wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
    sends = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
    run.rename(tmp_path / f"run-{label}")
    repo.rename(tmp_path / f"repo-{label}")
    return (wires, events, sends) if return_details else wires


def test_default_request_order_is_the_layout_on_order(tmp_path):
    (tmp_path / "launch-default").mkdir()
    (tmp_path / "launch-on").mkdir()
    (tmp_path / "launch-off").mkdir()
    default = _scripted_wire(tmp_path, "default", None)
    on = _scripted_wire(tmp_path, "on", "on")
    off = _scripted_wire(tmp_path, "off", "off")
    assert default and default == on
    assert off != on  # the control: switching it off restores the other order


def test_unset_switch_applies_layout_with_estimated_meter(tmp_path):
    (tmp_path / "launch-estimate").mkdir()
    wires, events, sends = _scripted_wire(
        tmp_path, "estimate", None, tokenize=False, return_details=True)
    frame_requests = [json.loads(wire) for wire in wires
                      if any(str(row.get("content", "")).startswith("[StateFrame]\n")
                             for row in json.loads(wire).get("messages", []))]
    assert frame_requests
    assert all(request["messages"][-1].get("name") == FINAL_LAYOUT_NAME
               for request in frame_requests)
    assert not [row for row in events
                if (row.get("kind") or row.get("event")) == "final_cache_layout_failed_open"
                and row.get("reason") in {"exceeds_measured_envelope", "meter_unavailable"}]
    frame_send_numbers = {
        index + 1 for index, wire in enumerate(wires)
        if any(str(row.get("content", "")).startswith("[StateFrame]\n")
               for row in json.loads(wire).get("messages", []))
    }
    sends_by_number = {row["send"]: row for row in sends}
    assert frame_send_numbers
    assert all(sends_by_number[number].get("final_layout", "plain") == "applied"
               for number in frame_send_numbers)


def test_scripted_near_ceiling_layout_failure_rate_guard_and_legacy_control(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kernel_cache_layout_candidate as candidate
    from kernel_context import token_estimate
    from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER

    (tmp_path / "launch-guard").mkdir()
    wires = _scripted_wire(tmp_path, "guard", None)
    bodies = []
    for wire in wires:
        body = json.loads(wire)
        if body["messages"][-1].get("name") == FINAL_LAYOUT_NAME:
            body["messages"] = semantic_messages(body["messages"])
        frame_index = next((index for index, row in enumerate(body["messages"])
                            if str(row.get("content", "")).startswith("[StateFrame]\n")), None)
        if frame_index is None:
            continue
        body["messages"].insert(frame_index, {
            "role": "user", "content": "[Optional source]\n" + "x" * 16384,
        })
        bodies.append(body)
    assert len(bodies) >= 2, "scripted fixture must produce multiple frame requests"

    class NearCeilingMeter:
        def measure(self, body):
            count = token_estimate(body)
            if body["messages"][-1].get("name") == FINAL_LAYOUT_NAME:
                count += 2000
            return count, "estimate"

    def run_candidates(candidates, *, can_trim):
        results = []
        for index, original in enumerate(candidates, start=1):
            body = copy.deepcopy(original)
            token_ceiling = token_estimate(body)
            byte_ceiling = len(json.dumps(body).encode())
            journal = SimpleNamespace(events=[])

            def emit(kind, **fields):
                journal.events.append((kind, fields))

            journal.emit = emit
            capture = SimpleNamespace(journal=journal, request_id=f"guard-{index}",
                                      request_source_final_delivery=None, step=index)
            if can_trim:
                def trim_optional(_tokens, _bytes):
                    body["messages"] = [row for row in body["messages"]
                                        if not str(row.get("content", "")).startswith("[Optional source]")]

                trim = trim_optional
            else:
                def trim_without_change(_tokens, _bytes):
                    return None

                trim = trim_without_change
            result = candidate.render_final_cache_candidate(
                capture, None, body, meter=NearCeilingMeter(),
                inputs={"token_ceiling": token_ceiling, "byte_ceiling": byte_ceiling,
                        "source_plan": {}, "sources": {}, "receipt": None,
                        "_final_layout_trim": trim})
            failed = sum(kind == "final_cache_layout_failed_open" for kind, _fields in journal.events)
            results.append((result, failed))
        return results

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)

    def assert_failure_rate_guard(results):
        failures = sum(failed for _result, failed in results)
        assert failures / len(results) <= 0.10, (
            f"final_cache_layout_failed_open rate {failures}/{len(results)} exceeds 10%")

    good = run_candidates(bodies, can_trim=True)
    assert_failure_rate_guard(good)
    assert all(result is not None and result["messages"][-1].get("name") == FINAL_LAYOUT_NAME
               for result, _failed in good)

    real_project = candidate._project_final_cache_layout

    def planted_legacy_projection(body):
        projected = real_project(body)
        state = next(row for row in projected["messages"]
                     if str(row.get("content", "")).startswith("[StateFrame]\n"))
        payload = json.loads(state["content"].split("\n", 1)[1])
        state["content"] = "[StateFrame]\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        final = projected["messages"][-1]
        marker = AUTHORITY_OVERLAY_HEADER + "\n"
        prefix, encoded = final["content"].split(marker, 1)
        overlay, consumed = json.JSONDecoder().raw_decode(encoded)
        order = overlay.pop("order")
        moved_content = {item["index"]: prefix[item["start"]:item["end"]]
                         for item in order["moved"]}
        overlay["fields"] = [
            {"path": field["path"], "value": field["value"] if "value" in field
             else (moved_content[field["message"]][field["span"][0]:field["span"][1]]
                   if "span" in field else moved_content[field["message"]])}
            for field in overlay["fields"]
        ]
        final["content"] = prefix + marker + json.dumps(
            overlay, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + encoded[consumed:]
        return projected

    with monkeypatch.context() as control:
        control.setattr(candidate, "_final_layout_reserve", lambda *_args, **_kwargs: (0, 0))
        control.setattr(candidate, "_project_final_cache_layout", planted_legacy_projection)
        old_projection = run_candidates(bodies, can_trim=False)
        with pytest.raises(AssertionError, match="exceeds 10%"):
            assert_failure_rate_guard(old_projection)


def test_default_layout_steps_aside_for_page_log_and_explicit_on_still_refuses(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    events = []
    capture = type("Capture", (), {"request_id": "r1", "journal": type("J", (), {
        "emit": lambda self, kind, **fields: events.append((kind, fields))})()})()
    body = {"messages": [{"role": "system", "content": "s"}]}
    monkeypatch.setattr(candidate, "mode", lambda: "on")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    assert candidate.render_final_cache_candidate(capture, None, body, meter=None, inputs={}) is body
    assert events and events[0][0] == "final_cache_layout_failed_open" and events[0][1]["reason"] == "page_log_on"
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    with pytest.raises(ValueError, match="page log"):
        candidate.render_final_cache_candidate(capture, None, body, meter=None, inputs={})


def test_default_layout_fails_open_on_an_unprojectable_request(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    events = []
    capture = type("Capture", (), {"request_id": "r1", "journal": type("J", (), {
        "emit": lambda self, kind, **fields: events.append((kind, fields))})()})()
    body = {"messages": []}  # not a frame request: the projection refuses it
    monkeypatch.setattr(candidate, "mode", lambda: "off")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    assert candidate.render_final_cache_candidate(capture, None, body, meter=None, inputs={}) is body
    assert events[0][1]["reason"] == "cannot_project"
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    with pytest.raises(ValueError, match="message array"):
        candidate.render_final_cache_candidate(capture, None, body, meter=None, inputs={})


def test_scripted_final_sealed_wire_and_planted_violation(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = "on"
    environment["STATEFRAME_PAGE_LOG"] = "off"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run),
             "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS)
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()

    wires = sorted((run / "wire").glob("*.request.json"))
    assert wires
    digests = set()
    for path in wires:
        raw = path.read_bytes()
        digests.add("sha256:" + hashlib.sha256(raw).hexdigest())
        rows = json.loads(raw)["messages"]
        state_index = next(i for i, row in enumerate(rows)
                           if str(row.get("content") or "").startswith("[StateFrame]\n"))
        assert state_index > 2
        assert any(str(row.get("content") or "").startswith("[Current source:")
                   for row in rows[2:state_index])
        assert sum(str(row.get("content") or "").count("[CONTINUATION LIMITS |")
                   for row in rows) == 1
        assert "[CONTINUATION LIMITS |" in rows[-1]["content"]
        frame = json.loads(rows[state_index]["content"].split("\n", 1)[1])
        assert not _payload_has_authority_keys(frame, final=True)
        assert semantic_messages(rows)[2]["content"].startswith("[StateFrame]\n")

    events = measurement_events(run)
    final_events = [row for row in events if row.get("event") == "cache_layout_final_wire"
                    or row.get("kind") == "cache_layout_final_wire"]
    assert final_events
    assert digests.issubset({row.get("body_sha256") for row in final_events})
    assert all(row.get("authority_message_index") == len(row.get("message_order", ())) - 1
               for row in final_events)
    assert all(row["message_order"][0] == "system"
               and row["message_order"][-1] == "final_authority"
               and row["message_order"].index("source") < row["message_order"].index("state")
               for row in final_events)

    broken = copy.deepcopy(json.loads(wires[0].read_bytes())["messages"])
    broken[2]["content"] += "\n[CONTINUATION LIMITS | planted]"
    with pytest.raises(CacheLayoutError, match="physically final"):
        semantic_messages(broken)
