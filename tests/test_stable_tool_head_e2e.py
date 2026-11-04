"""End to end: the default stable tool head keeps the provider tool list append-only.

Runs the bundled scripted discount_bug example offline (needs the host sandbox and a local socket).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from stateframe.scripted_endpoint import Script, serve
from engine_run import run_monitored

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


def _archive_revision(revision: str, destination: Path) -> Path:
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", revision, "src", "examples"],
        capture_output=True, check=True,
    ).stdout
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as contents:
        for member in contents:
            target = destination / member.name
            if (not target.resolve().is_relative_to(destination.resolve())
                    or member.issym() or member.islnk()):
                raise AssertionError(f"unsafe baseline archive entry: {member.name}")
            contents.extract(member, destination)
    return destination


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


def test_switch_off_requests_match_the_released_baseline(tmp_path):
    from tools.compare_no_mcp_requests import (
        capture, mask_wire_ids, unbacked_result_hash_changes, write_launcher)

    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "fixed_seed_run.py"
    write_launcher(launcher)
    script = ROOT / "examples/discount_bug.script.json"
    run_dir = tmp_path / "run"

    # Only a published release commit: a local or unmerged revision is missing from other clones and CI.
    for index, revision in enumerate(("5140116",)):
        baseline = _archive_revision(revision, tmp_path / f"baseline-{index}")
        expected = capture(
            baseline, sys.executable, launcher, script, repo, run_dir,
            stable_tool_head="off",
        )
        run_dir.rename(tmp_path / f"baseline-run-{index}")
        actual = capture(
            ROOT, sys.executable, launcher, script, repo, run_dir,
            stable_tool_head="off",
        )
        run_dir.rename(tmp_path / f"candidate-run-{index}")

        # Wire IDs changed derivation on purpose (receipt determinism, 11-03); a masked
        # result hash must still hash text that is sent and compared (see mask_wire_ids).
        assert unbacked_result_hash_changes(
            tmp_path / f"baseline-run-{index}", tmp_path / f"candidate-run-{index}") == []
        assert mask_wire_ids(actual) == mask_wire_ids(expected), (
            f"STATEFRAME_STABLE_TOOL_HEAD=off changed requests vs {revision}")


def test_switch_off_keeps_per_turn_menus(tmp_path):
    result, run = _run(tmp_path, "off")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(str(message.get("content", "")).startswith("[Tool menu | this turn]")
                   for request in _wire(run) for message in request["messages"])
