"""End to end: the default stable tool head keeps the provider tool list append-only.

Runs the bundled scripted discount_bug example offline (needs the host sandbox and a local socket).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe.scripted_endpoint import Script, serve
from engine_run import run_monitored
from test_hook_wiring_parity import baseline_checkout  # noqa: F401 -- pytest registers this fixture

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


def _run(tmp_path: Path, switch: str | None) -> tuple[subprocess.CompletedProcess, Path]:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if switch is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = switch
    try:
        result = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            env=environment, capture_output=True, text=True, expected_seconds=240)
        script.assert_consumed()
        return result, run
    finally:
        server.shutdown()
        server.server_close()


def _wire(run: Path) -> list[dict]:
    return [json.loads(path.read_bytes()) for path in sorted((run / "wire").glob("*.request.json"))]


def _raw_tool_arrays(run: Path) -> list[bytes]:
    arrays = []
    for path in sorted((run / "wire").glob("*.request.json")):
        source = path.read_bytes().decode("utf-8")
        decoder = json.JSONDecoder()
        offset = 1
        while True:
            while source[offset].isspace():
                offset += 1
            if source[offset] == "}":
                raise AssertionError(f"{path.name} has no top-level tools array")
            key, offset = decoder.raw_decode(source, offset)
            while source[offset].isspace():
                offset += 1
            assert source[offset] == ":"
            offset += 1
            while source[offset].isspace():
                offset += 1
            start = offset
            value, offset = decoder.raw_decode(source, offset)
            if key == "tools":
                assert isinstance(value, list)
                arrays.append(source[start:offset].encode("utf-8"))
                break
            while source[offset].isspace():
                offset += 1
            if source[offset] == ",":
                offset += 1
            elif source[offset] == "}":
                raise AssertionError(f"{path.name} has no top-level tools array")
    return arrays


def _tool_array_bytes(request: dict) -> bytes:
    return json.dumps(
        request["tools"], ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _assert_tool_arrays_identical(requests: list[dict]) -> None:
    assert requests
    first = _tool_array_bytes(requests[0])
    for index, request in enumerate(requests[1:], start=2):
        assert _tool_array_bytes(request) == first, f"request {index} changed the tool array"


def test_unset_defaults_on_and_the_run_still_succeeds(tmp_path):
    result, run = _run(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr
    requests = _wire(run)
    assert len(requests) >= 2
    _assert_tool_arrays_identical(requests)
    raw_arrays = _raw_tool_arrays(run)
    assert len(raw_arrays) == len(requests)
    assert all(array == raw_arrays[0] for array in raw_arrays[1:])
    first_names = {item["function"]["name"] for item in requests[0]["tools"]}
    assert "authored_check" in first_names
    summary = json.loads((run / "summary.json").read_text())
    assert summary.get("succeeded") is True


def test_identical_array_assertion_catches_a_planted_late_append():
    def tool(name: str) -> dict:
        return {"type": "function", "function": {
            "name": name, "description": "d",
            "parameters": {"type": "object", "properties": {}, "required": []}}}

    requests = [
        {"tools": [tool("read")]},
        {"tools": [tool("read"), tool("authored_check")]},
    ]
    with pytest.raises(AssertionError, match="request 2 changed the tool array"):
        _assert_tool_arrays_identical(requests)


def test_raw_tool_array_extractor_preserves_serialized_bytes(tmp_path):
    run = tmp_path / "run"
    wire = run / "wire"
    wire.mkdir(parents=True)
    request = b'{"model":"scripted","tools" : [ {"name":"read"} ],"messages":[]}'
    (wire / "0001.request.json").write_bytes(request)

    assert _raw_tool_arrays(run) == [b'[ {"name":"read"} ]']


def test_switch_off_requests_match_the_released_baseline(
        tmp_path, baseline_checkout):  # noqa: F811 -- injected fixture
    from tools.compare_no_mcp_requests import (
        capture, mask_wire_ids, unbacked_result_hash_changes, write_launcher)

    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "fixed_seed_run.py"
    write_launcher(launcher)
    script = ROOT / "examples/discount_bug.script.json"
    run_dir = tmp_path / "run"

    expected = capture(
        baseline_checkout, sys.executable, launcher, script, repo, run_dir,
        stable_tool_head="off",
    )
    run_dir.rename(tmp_path / "baseline-run")
    actual = capture(
        ROOT, sys.executable, launcher, script, repo, run_dir,
        stable_tool_head="off",
    )
    run_dir.rename(tmp_path / "candidate-run")

    # The masked result hash must still hash text that is sent and compared.
    assert unbacked_result_hash_changes(
        tmp_path / "baseline-run", tmp_path / "candidate-run") == []
    assert mask_wire_ids(actual) == mask_wire_ids(expected), (
        "STATEFRAME_STABLE_TOOL_HEAD=off changed requests vs immediate base")


def test_deferred_prefit_layout_off_keeps_released_request_bytes(
        tmp_path, monkeypatch, baseline_checkout):  # noqa: F811 -- injected fixture
    from tools.compare_no_mcp_requests import capture, write_launcher

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    monkeypatch.setenv("STATEFRAME_STABLE_TOOL_HEAD", "off")
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "fixed_seed_run.py"
    write_launcher(launcher)
    script = ROOT / "examples/discount_bug.script.json"
    run_dir = tmp_path / "run"

    released = capture(baseline_checkout, sys.executable, launcher, script, repo, run_dir,
                       stable_tool_head="off")
    run_dir.rename(tmp_path / "released-run")
    from _journal import measurement_events

    def budget_slots(events):
        return [(row.get("step"), row.get("sequence"))
                for row in events
                if row.get("kind") == "request_budget_final_wire"]

    released_slots = budget_slots(measurement_events(tmp_path / "released-run"))
    current = capture(ROOT, sys.executable, launcher, script, repo, run_dir,
                      stable_tool_head="off")
    assert len(_wire(run_dir)) >= 2
    events = measurement_events(run_dir)
    deferred = [row for row in events
                if row.get("kind") == "request_budget_final_wire"
                and row.get("status") == "PRE_FIT_DEFERRED"]
    proofs = [row for row in events
              if row.get("kind") == "request_budget_final_wire"
              and row.get("phase") == "frozen_transport_revalidation"]
    assert deferred and proofs
    assert all(row.get("status") == "READY" and row.get("measure_method") == "tokenize"
               for row in proofs)
    assert current == released  # Includes every provider-visible request_id.
    assert budget_slots(events) == released_slots

    # Planted control: dropping one deferred event must break the same slot
    # comparison. Request IDs can be content-derived, so check the journal too.
    skipped_events = list(events)
    skipped_events.remove(deferred[0])
    with pytest.raises(AssertionError):
        assert budget_slots(skipped_events) == released_slots


def test_switch_off_keeps_per_turn_menus(tmp_path):
    result, run = _run(tmp_path, "off")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(str(message.get("content", "")).startswith("[Tool menu | this turn]")
                   for request in _wire(run) for message in request["messages"])
