import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
from dataclasses import replace
from unittest.mock import patch

import pytest

from engine_run import run_monitored
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot
from stateframe.engine.kernel_parallel import _snapshot
from stateframe.engine.kernel_custom_tools import load_tool_files
from stateframe.engine.kernel_effects import scan
from stateframe.engine.kernel_exec_tracker import LaunchIdentity
from stateframe.engine.kernel_hooks import create_bundle, load_hook_files, verify_bundle
from stateframe.engine.kernel_parallel_records import (CheckRecipe, ParallelLimits, UnitContract,
                                                       UnitSpec, canonical_json)
from stateframe.engine.kernel_unit_task import build_public_row
from stateframe.engine.kernel_tool_settings import ToolSettings
from stateframe.endpoint import Endpoint
from stateframe import profiles, routes
from stateframe.runner import RunOptions, _child_env, prepare, summarize
from stateframe._child import install_repo_context
from stateframe.parallel_worker import WorkerLaunchError, WorkerPreparationError, launch_worker, prepare_worker


_DEFAULT_ENGINE_TIMEOUT = 300
_OFF_RELEASE_COMMIT = "39f73ee6960e582e079312afb5982fd07bf1c1b4"  # StateFrame 0.8.5


def _engine_timeout_seconds() -> int:
    raw = os.environ.get("SF_TEST_ENGINE_TIMEOUT", str(_DEFAULT_ENGINE_TIMEOUT))
    try:
        timeout = int(raw)
    except ValueError:
        pytest.fail(f"SF_TEST_ENGINE_TIMEOUT must be a positive integer number of seconds, got {raw!r}",
                    pytrace=False)
    if timeout <= 0:
        pytest.fail(f"SF_TEST_ENGINE_TIMEOUT must be a positive integer number of seconds, got {raw!r}",
                    pytrace=False)
    return timeout


def _scaled_test_timeout(seconds: int) -> int:
    scaled = (seconds * _engine_timeout_seconds() + _DEFAULT_ENGINE_TIMEOUT - 1) // _DEFAULT_ENGINE_TIMEOUT
    return max(seconds, scaled)


def test_engine_timeout_defaults_to_300_seconds(monkeypatch):
    monkeypatch.delenv("SF_TEST_ENGINE_TIMEOUT", raising=False)
    assert _engine_timeout_seconds() == 300
    assert _scaled_test_timeout(60) == 60


def test_engine_timeout_accepts_positive_integer(monkeypatch):
    monkeypatch.setenv("SF_TEST_ENGINE_TIMEOUT", "1800")
    assert _engine_timeout_seconds() == 1800
    assert _scaled_test_timeout(30) == 180


def test_engine_timeout_rejects_invalid_values(monkeypatch):
    for value in ("invalid", "0", "-1", "1.5"):
        monkeypatch.setenv("SF_TEST_ENGINE_TIMEOUT", value)
        with pytest.raises(pytest.fail.Exception,
                           match="SF_TEST_ENGINE_TIMEOUT must be a positive integer number of seconds"):
            _engine_timeout_seconds()


@pytest.fixture(autouse=True)
def _unlock_fixture_snapshots(tmp_path):
    yield
    for path in sorted(tmp_path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)


def _unit_fixture(root: Path, *, include_symlink: bool = False,
                  executable_helper: bool = True,
                  run_seed: str = "0123456789abcdef0123456789abcdef"):
    parent_run = root / "parent-run"
    start_snapshot = parent_run / "parent"
    (start_snapshot / "pkg").mkdir(parents=True)
    (start_snapshot / "tests").mkdir()
    (start_snapshot / "pkg" / "a.py").write_bytes(b"VALUE = 1\n")
    (start_snapshot / "pkg" / "run.sh").write_bytes(b"#!/bin/sh\nexit 0\n")
    (start_snapshot / "pkg" / "run.sh").chmod(0o755 if executable_helper else 0o644)
    (start_snapshot / "tests" / "test_a.py").write_bytes(b"def test_value():\n    assert True\n")
    (start_snapshot / "pkg" / "a.py").chmod(0o444)
    (start_snapshot / "tests" / "test_a.py").chmod(0o444)
    (start_snapshot / "pkg").chmod(0o700)
    call_root = parent_run / "parallel" / "call"
    frozen = _snapshot(start_snapshot, call_root, 4)
    snapshot = frozen.root
    if include_symlink:
        (snapshot / "pkg").chmod(0o755)
        (snapshot / "pkg" / "alias.py").symlink_to("a.py")
        (snapshot / "pkg").chmod(0o555)
    manifest = frozen.manifest_digest
    contract = UnitContract("a", "change value", ("pkg/a.py",), ("pkg/new.py",), (), (), (), (),
                            CheckRecipe(("tests/test_a.py",), ("tests/test_a.py::test_value",), "pytest"))
    spec = UnitSpec("call", "a", "execution", run_seed, contract, (), (),
                    ("pkg/a.py", "pkg/new.py", "tests/test_a.py"), (), "failed assertion", 4, 4,
                    manifest, "unit-policy", ParallelLimits())
    run_dir = call_root / "units" / "a"
    run_dir.mkdir(mode=0o700, parents=True)
    spec_path = run_dir / "unit-spec.json"
    spec_path.write_bytes(canonical_json(spec.to_json()) + b"\n")
    parent_spec = {
        "schema": "stateframe-run/1",
        "engine_dir": str(Path(__file__).resolve().parents[1] / "src/stateframe/engine"),
        "engine_source": profiles.engine_source_summary(),
        "profile": "default",
        "chat_url": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "local-model",
        "max_turns": 16,
        "repo": str(root),
        "api_key_env": None,
        "environment": {"DS_URL": "http://127.0.0.1:8080/v1/chat/completions",
                         "DS_MODEL": "local-model", "PYTHONHASHSEED": "0", "PROFILE_VALUE": "retained"},
        "environment_sources": {"PROFILE_VALUE": "profile:test"},
        "public_row": {"msg": "parent task"},
        "tool_settings": ToolSettings(shell="standard", shell_ask=True, run_env=True, git_tool=True,
                                       web=True, web_allow=("example.com",)).as_dict(),
        "parent_snapshot": str(start_snapshot),
        "run_identity_seed": "parent-seed",
        "routes": routes.resolve({
            "endpoint": "http://127.0.0.1:8080/v1", "model": "local-model",
            "api_key_env": None, "temperature": 1.0, "max_tokens": 24000,
            "timeout_s": 1800.0, "slot": None, "context_tokens": 65536,
            "allow_remote": False,
        }, environ={}).public(),
        "test_cmd": "pytest -q",
        "resume": {"schema": "stateframe-resume/1"},
        "parallel_recovery": {"schema": "stateframe-parallel-recovery/1"},
        "capabilities": {"python": sys.executable},
        "snapshot_manifest_digest": frozen.snapshot_manifest_digest,
    }
    parent_spec["routes_digest"] = parent_spec["routes"]["digest"]
    return snapshot, run_dir, spec_path, spec, parent_spec, parent_run


def _wire_requests(run_dir: Path) -> dict[str, bytes]:
    wire_dir = run_dir / "wire"
    assert wire_dir.is_dir(), f"missing recorded wire directory: {wire_dir}"
    return {path.name: path.read_bytes() for path in sorted(wire_dir.glob("*.request.json"))}


def _first_wire_component(left: object, right: object, path: str = "") -> str:
    if type(left) is not type(right):
        return path or "body"
    if isinstance(left, dict):
        for key in dict.fromkeys((*left, *right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                return child
            if left[key] != right[key]:
                return _first_wire_component(left[key], right[key], child)
    elif isinstance(left, list):
        for index, (before, after) in enumerate(zip(left, right)):
            if before != after:
                return _first_wire_component(before, after, f"{path}[{index}]")
        if len(left) != len(right):
            return f"{path}[{min(len(left), len(right))}]"
    return path or "serialized JSON (format/key order)"


def _assert_wire_bytes(left: bytes, right: bytes, *, send: int, name: str) -> None:
    if left == right:
        return
    offset = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                  min(len(left), len(right)))
    try:
        component = _first_wire_component(json.loads(left), json.loads(right))
    except (ValueError, UnicodeDecodeError):
        component = "invalid serialized JSON"
    before = repr(left[offset:offset + 1]) if offset < len(left) else "EOF"
    after = repr(right[offset:offset + 1]) if offset < len(right) else "EOF"
    raise AssertionError(f"send {send} ({name}): first differing byte {offset} (zero-based), "
                         f"{before} -> {after}; component {component}; "
                         f"sizes {len(left)} -> {len(right)} bytes\n"
                         f"expected window {left[max(0, offset - 56):offset + 72]!r}\n"
                         f"actual window   {right[max(0, offset - 56):offset + 72]!r}")


def _assert_wire_parity(left_run: Path, _left_endpoint: str,
                        right_run: Path, _right_endpoint: str) -> None:
    left, right = _wire_requests(left_run), _wire_requests(right_run)
    assert left and right, "expected recorded requests on both sides"
    # Compare the common sends first: a later missing send must not hide an earlier wire change.
    for send, name in enumerate(sorted(left.keys() | right.keys()), 1):
        assert name in left and name in right, f"send {send} ({name}): missing request body"
        _assert_wire_bytes(left[name], right[name], send=send, name=name)


@pytest.mark.parametrize("field,component", [("content", "messages[0].content"),
                                           ("description", "tools[0].function.description")])
def test_wire_parity_reports_first_send_byte_and_component(tmp_path, field, component):
    body = {"messages": [{"role": "user", "content": "café"}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "read"}}]}
    before = json.dumps(body, ensure_ascii=False).encode()
    after = before.replace(b"caf\xc3\xa9", b"caf\xc3\xaa") if field == "content" else before.replace(
        b'"description": "read"', b'"description": "write"')
    offset = next(i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1])
    for label, payload in (("left", before), ("right", after)):
        wire = tmp_path / label / "wire"
        wire.mkdir(parents=True)
        (wire / "0001.request.json").write_bytes(before)
        (wire / "0002.request.json").write_bytes(payload)
        (wire / "0003.request.json").write_bytes(b'{}' if label == "left" else b'{"later": true}')
    with pytest.raises(AssertionError) as error:
        _assert_wire_parity(tmp_path / "left", "endpoint", tmp_path / "right", "endpoint")
    message = str(error.value)
    assert "send 2" in message
    assert f"byte {offset}" in message
    assert component in message


@pytest.mark.parametrize("change,component", [
    ("source", "messages[0].content"),
    ("hash", "messages[0].content"),
    ("tool-order", "tools[0].function.name"),
    ("format", "serialized JSON"),
    ("eof", "invalid serialized JSON"),
])
def test_exact_wire_comparison_preserves_content_order_sizes_and_hashes(change, component):
    body = {"messages": [{"content": 'source: VALUE = 1; sha256:' + "a" * 64}],
            "tools": [{"function": {"name": "read"}}, {"function": {"name": "edit"}}]}
    before = json.dumps(body).encode()
    if change == "source":
        after = before.replace(b"VALUE = 1", b"VALUE = 10")
    elif change == "hash":
        after = before.replace(b"a" * 64, b"b" * 64)
    elif change == "tool-order":
        body["tools"].reverse()
        after = json.dumps(body).encode()
    elif change == "format":
        after = json.dumps(body, separators=(",", ":")).encode()
    else:
        after = before[:-1]
    offset = next((i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]),
                  min(len(before), len(after)))
    with pytest.raises(AssertionError) as error:
        _assert_wire_bytes(before, after, send=3, name="applied-edit")
    message = str(error.value)
    assert f"send 3 (applied-edit): first differing byte {offset}" in message
    assert component in message
    assert f"sizes {len(before)} -> {len(after)} bytes" in message
    _assert_wire_bytes(before, before, send=3, name="applied-edit")


def _pin_off_cache_switches(switches: dict[str, str]) -> dict[str, str]:
    for name in (
        "STATEFRAME_WORKING_SET_STABLE_FIRST",
        "STATEFRAME_CONSTANT_TOOL_MENU",
        "STATEFRAME_SHARED_CHECK_CACHE",
        "STATEFRAME_PROMPT_CACHE_UNPINNED",
    ):
        switches.pop(name, None)
    return switches


def _write_off_cache_profile(tmp_path: Path, *, released_profile: Path | None = None) -> Path:
    source = released_profile or profiles.HERE / "r2hc3.json"
    profile = json.loads(source.read_text())
    switches = _pin_off_cache_switches(profile["switches"])
    # The historical reader defaults to off. Supplying the retired key to the
    # candidate creates a journal event for a setting this wire case does not use.
    if released_profile is not None and switches.get("STATEFRAME_READ_LOCKOUT") == "off":
        del switches["STATEFRAME_READ_LOCKOUT"]
    path = tmp_path / "r2hc3-off.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.mark.host_sandbox
@pytest.mark.parametrize("mode", ["omitted", "off", "2"])
def test_parallel_off_exact_main_wire_fixture_and_on_companion(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    helper = root / "tests/fixtures/parallel/off_wire_capture.py"
    profile = _write_off_cache_profile(tmp_path)
    environment = {**os.environ, "PYTHONHASHSEED": "0"}
    completed = run_monitored(
        [sys.executable, str(helper), str(root), str(tmp_path / "capture"), mode, str(profile)],
        cwd=root, env=environment, run_dir=tmp_path / "capture/run",
        expected_seconds=_engine_timeout_seconds(), step=f"pinned OFF wire fixture ({mode})",
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    captured = _wire_requests(tmp_path / "capture/run")
    assert len(captured) >= 3
    evidence = json.loads((tmp_path / "capture/evidence.json").read_bytes())
    assert evidence["checks"][0] == "FAIL" and evidence["checks"][-1] == "PASS"
    fixture = root / "tests/fixtures/parallel/off_wire_main_b520a637"
    for send, stage in enumerate(("initial", "failed-check", "applied-edit"), 1):
        name = f"{send:04}.request.json"
        raw = captured[name]
        body = json.loads(raw)
        names = [tool["function"]["name"] for tool in body["tools"]]
        assert ("parallel" in names) == (mode == "2"), stage
        if mode != "2":
            _assert_wire_bytes((fixture / name).read_bytes(), raw, send=send, name=stage)
    from test_request_determinism import _has_applied_edit_in_frame

    assert not _has_applied_edit_in_frame(json.loads(captured["0001.request.json"]))
    assert "TESTS FAILED" in captured["0002.request.json"].decode()
    assert _has_applied_edit_in_frame(json.loads(captured["0003.request.json"]))
    assert bool(evidence["dispatch"]) == (mode == "2")


def _assert_default_off_spec_parity(released: dict, candidate: dict,
                                    _released_run: Path, _released_endpoint: str,
                                    _candidate_run: Path, _candidate_endpoint: str) -> None:
    assert not set(candidate) - set(released), "candidate added unexpected run-spec keys"
    assert not set(released) - set(candidate), "candidate removed released run-spec keys"
    assert "parallel_mode" not in candidate and "parallel_limits" not in candidate
    assert "parallel_engine_python" not in candidate
    assert canonical_json(released) == canonical_json(candidate), (
        f"default-off run spec values differ at {_first_wire_component(released, candidate)}")


def _assert_default_off_timed_mapping_parity(released: dict, candidate: dict,
                                             seconds_key: str, label: str, *,
                                             candidate_only_keys: dict[str, str] | None = None) -> None:
    allowed_candidate_keys = set(candidate_only_keys or ())
    released_keys = set(released)
    candidate_keys = set(candidate)
    assert not released_keys - candidate_keys, f"default-off {label} removed reference keys"
    unexpected_candidate_keys = candidate_keys - released_keys - allowed_candidate_keys
    assert not unexpected_candidate_keys, f"default-off {label} added unexpected keys"

    assert seconds_key in released and seconds_key in candidate
    shared_candidate = {key: value for key, value in candidate.items() if key in released_keys}
    assert {key: type(value) for key, value in released.items()} == {
        key: type(value) for key, value in shared_candidate.items()}
    released_seconds = released[seconds_key]
    candidate_seconds = shared_candidate[seconds_key]
    assert type(released_seconds) is type(candidate_seconds) is float
    assert math.isfinite(released_seconds) and released_seconds >= 0.0
    assert math.isfinite(candidate_seconds) and candidate_seconds >= 0.0
    released_without_seconds = {key: value for key, value in released.items() if key != seconds_key}
    candidate_without_seconds = {key: value for key, value in shared_candidate.items() if key != seconds_key}
    assert canonical_json(released_without_seconds) == canonical_json(candidate_without_seconds), (
        f"default-off {label} differ at "
        f"{_first_wire_component(released_without_seconds, candidate_without_seconds)}")


# These intentional default-off summary fields postdate the 0.8.5 parity reference.
_DEFAULT_OFF_CANDIDATE_ONLY_SUMMARY_KEYS = {
    "no_change": "StateFrame 0.9.6 FALSE-SUCCESS",
    "start_files_skipped": "StateFrame 0.9.4 START",
    "start_files_skipped_total": "StateFrame 0.9.4 START",
}


def _assert_default_off_result_bytes_parity(released: bytes, candidate: bytes) -> None:
    released_value = json.loads(released)
    candidate_value = json.loads(candidate)
    _assert_default_off_delivery_manifest_digest(released_value)
    _assert_default_off_delivery_manifest_digest(candidate_value)
    released_structure = _default_off_result_parity_view(released_value)
    candidate_structure = _default_off_result_parity_view(candidate_value)
    for structure in (released_structure, candidate_structure):
        if isinstance(structure, dict) and "seconds" in structure:
            structure["seconds"] = "<seconds>"
    assert canonical_json(released_structure) == canonical_json(candidate_structure), (
        "default-off result structure differs at "
        f"{_first_wire_component(released_structure, candidate_structure)}")

    seconds_value = rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    seconds_field = re.compile(rb'("seconds"\s*:\s*)' + seconds_value)
    released_without_seconds, released_seconds_count = seconds_field.subn(
        rb"\1<seconds>", released)
    candidate_without_seconds, candidate_seconds_count = seconds_field.subn(
        rb"\1<seconds>", candidate)
    assert released_seconds_count == candidate_seconds_count == 1

    released_without_manifest_ids = _omit_manifest_id_fields_from_bytes_after_validation(
        released_without_seconds)
    candidate_without_manifest_ids = _omit_manifest_id_fields_from_bytes_after_validation(
        candidate_without_seconds)
    event_id_field = re.compile(
        rb'("(?:(?:[A-Za-z0-9_]+_)?event_id)"\s*:\s*)"(?:\\.|[^"\\])*"')
    released_normalized, released_event_id_count = event_id_field.subn(
        rb'\1"<run-specific-id>"', released_without_manifest_ids)
    candidate_normalized, candidate_event_id_count = event_id_field.subn(
        rb'\1"<run-specific-id>"', candidate_without_manifest_ids)
    assert released_event_id_count == candidate_event_id_count
    assert released_normalized == candidate_normalized


def _normalize_default_off_run_specific_ids(value: object, *,
                                            is_delivery_manifest: bool = False) -> object:
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            is_run_specific_event_id = key == "event_id" or key.endswith("_event_id")
            is_run_specific_manifest_id = key == "manifest_id" and not is_delivery_manifest
            if is_run_specific_event_id or is_run_specific_manifest_id:
                assert item is None or type(item) is str, f"{key} must be a string or null"
                normalized[key] = "<run-specific-id>" if item is not None else None
            else:
                normalized[key] = _normalize_default_off_run_specific_ids(
                    item, is_delivery_manifest=key == "delivery_manifest")
        return normalized
    if isinstance(value, list):
        return [_normalize_default_off_run_specific_ids(item) for item in value]
    return value


def _default_off_result_parity_view(value: object) -> object:
    normalized = _normalize_default_off_run_specific_ids(value)
    if isinstance(normalized, dict):
        result = normalized.get("result")
        if isinstance(result, dict):
            manifest = result.get("delivery_manifest")
            if isinstance(manifest, dict):
                # The digest is validated against each unnormalized manifest below. It is
                # derived from run-scoped IDs, so it does not participate in cross-run parity.
                manifest.pop("manifest_id", None)
    return normalized


def _assert_default_off_delivery_manifest_digest(result_value: object) -> None:
    from kernel_delivery_manifest import _digest

    if not isinstance(result_value, dict):
        return
    result = result_value.get("result")
    if not isinstance(result, dict):
        return
    manifest = result.get("delivery_manifest")
    if manifest is None:
        return
    assert isinstance(manifest, dict), "delivery manifest must be an object"
    body = dict(manifest)
    assert body.pop("manifest_id") == _digest(body), "invalid delivery manifest digest"


def _omit_manifest_id_fields_from_bytes_after_validation(raw: bytes) -> bytes:
    # Structured parity checks ID-field shape; callers validate delivery digests before this.
    return re.sub(rb'"manifest_id"\s*:\s*"(?:\\.|[^"\\])*"', b"", raw)


def _assert_default_off_result_parity(released: dict, candidate: dict,
                                     _released_run: Path, _released_endpoint: str,
                                     _candidate_run: Path, _candidate_endpoint: str) -> None:
    _assert_default_off_delivery_manifest_digest(released)
    _assert_default_off_delivery_manifest_digest(candidate)
    released_structure = _default_off_result_parity_view(released)
    candidate_structure = _default_off_result_parity_view(candidate)
    _assert_default_off_timed_mapping_parity(
        released_structure, candidate_structure, "seconds", "engine results")
    assert type(released.get("exit_status")) is type(candidate.get("exit_status")) is int


def test_default_off_result_parity_checks_the_published_exit_status(tmp_path):
    released = {"result": {"result": "SUBMITTED"}, "seconds": 0.0, "exit_status": 0}
    arguments = (tmp_path / "released", "http://127.0.0.1:30001/v1",
                 tmp_path / "candidate", "http://127.0.0.1:30002/v1")
    _assert_default_off_result_parity(released, released, *arguments)
    _assert_default_off_result_parity(released, {**released, "seconds": 1.25}, *arguments)

    for changed in ({"result": released["result"]}, {**released, "exit_status": 1},
                    {**released, "exit_status": False}, {**released, "result": {"result": "DRAFT"}},
                    {**released, "unexpected": "field"}, {**released, "seconds": "0.0"},
                    {**released, "seconds": -0.1}, {**released, "seconds": float("nan")},
                    {**released, "seconds": float("inf")}):
        with pytest.raises(AssertionError):
            _assert_default_off_result_parity(released, changed, *arguments)

    _assert_default_off_result_event_id_values(tmp_path)


def test_default_off_summary_parity_allows_only_released_candidate_keys():
    released = {"result": "SUBMITTED", "wall_seconds": 0.0, "exit_code": 0}
    candidate = {
        **released,
        "wall_seconds": 1.25,
        "no_change": False,
        "start_files_skipped": [],
        "start_files_skipped_total": 0,
    }
    _assert_default_off_timed_mapping_parity(
        released, candidate, "wall_seconds", "summaries",
        candidate_only_keys=_DEFAULT_OFF_CANDIDATE_ONLY_SUMMARY_KEYS)

    with pytest.raises(AssertionError, match="added unexpected keys"):
        _assert_default_off_timed_mapping_parity(
            released, {**candidate, "planted_unexpected_key": True}, "wall_seconds", "summaries",
            candidate_only_keys=_DEFAULT_OFF_CANDIDATE_ONLY_SUMMARY_KEYS)

    for invalid_wall_seconds in (0, "0.0", -0.1, float("nan"), float("inf")):
        with pytest.raises(AssertionError):
            _assert_default_off_timed_mapping_parity(
                released, {**released, "wall_seconds": invalid_wall_seconds},
                "wall_seconds", "summaries")


def test_default_off_result_bytes_parity_ignores_only_seconds_value():
    released = b'{"result":"SUBMITTED","seconds":0.0,"exit_status":0}'
    candidate = b'{"result":"SUBMITTED","seconds":1.25,"exit_status":0}'
    _assert_default_off_result_bytes_parity(released, candidate)

    with pytest.raises(AssertionError):
        _assert_default_off_result_bytes_parity(
            released, b'{"result":"DRAFT","seconds":1.25,"exit_status":0}')


def _assert_default_off_result_event_id_values(tmp_path):
    from kernel_delivery_manifest import _digest

    released_manifest = {"payload": {"prepared_event_id": "release-run:12"}}
    released_manifest["manifest_id"] = _digest(released_manifest)
    candidate_manifest = {"payload": {"prepared_event_id": "candidate-run:17"}}
    candidate_manifest["manifest_id"] = _digest(candidate_manifest)
    released = {
        "result": {"receipt": {
            "prepared_event_id": "release-run:12",
            "exposure": {"records": [{"event_id": "release-run:12"}],
                         "response_event_id": "release-run:13"},
            "status": "READY",
        }, "delivery_manifest": released_manifest},
        "seconds": 0.0,
        "exit_status": 0,
    }
    candidate = {
        "result": {"receipt": {
            "prepared_event_id": "candidate-run:17",
            "exposure": {"records": [{"event_id": "candidate-run:17"}],
                         "response_event_id": "candidate-run:18"},
            "status": "READY",
        }, "delivery_manifest": candidate_manifest},
        "seconds": 1.25,
        "exit_status": 0,
    }
    arguments = (tmp_path / "released", "http://127.0.0.1:30001/v1",
                 tmp_path / "candidate", "http://127.0.0.1:30002/v1")

    _assert_default_off_result_parity(released, candidate, *arguments)
    _assert_default_off_result_bytes_parity(
        json.dumps(released, separators=(",", ":")).encode(),
        json.dumps(candidate, separators=(",", ":")).encode(),
    )

    for changed in (
        {**candidate, "result": {**candidate["result"], "receipt": {"status": "READY"}}},
        {**candidate, "result": {**candidate["result"], "receipt": {
            **candidate["result"]["receipt"],
            "exposure": {**candidate["result"]["receipt"]["exposure"], "records": [{}]},
        }}},
        {**candidate, "result": {**candidate["result"], "receipt": {
            "prepared_event_id": 17,
            "exposure": {"records": [{"event_id": "candidate-run:17"}],
                         "response_event_id": "candidate-run:18"},
            "status": "READY",
        }}},
        {**candidate, "result": {**candidate["result"], "receipt": {
            **candidate["result"]["receipt"], "status": "CHANGED",
        }}},
    ):
        with pytest.raises(AssertionError):
            _assert_default_off_result_parity(released, changed, *arguments)
        with pytest.raises(AssertionError):
            _assert_default_off_result_bytes_parity(
                json.dumps(released, separators=(",", ":")).encode(),
                json.dumps(changed, separators=(",", ":")).encode(),
            )


def test_run_specific_result_ids_normalize_without_changing_shape():
    from kernel_delivery_manifest import _digest

    delivery_manifest = {"payload": {"prepared_event_id": "release-run:12"}}
    delivery_manifest["manifest_id"] = _digest(delivery_manifest)
    delivery_manifest_id = delivery_manifest["manifest_id"]
    value = {
        "result": {"receipt": {
            "prepared_event_id": "release-run:12",
            "exposure": {"records": [{"event_id": "release-run:12"}],
                         "response_event_id": "release-run:13"},
            "manifest_id": "release-manifest:1",
        }, "delivery_manifest": {
            **delivery_manifest,
        }},
        "seconds": 0.0,
        "exit_status": 0,
    }
    assert _normalize_default_off_run_specific_ids(value) == {
        "result": {"receipt": {
            "prepared_event_id": "<run-specific-id>",
            "exposure": {"records": [{"event_id": "<run-specific-id>"}],
                         "response_event_id": "<run-specific-id>"},
            "manifest_id": "<run-specific-id>",
        }, "delivery_manifest": {
            "payload": {"prepared_event_id": "<run-specific-id>"},
            "manifest_id": delivery_manifest_id,
        }},
        "seconds": 0.0,
        "exit_status": 0,
    }

    for invalid_id in (17, False):
        with pytest.raises(AssertionError, match="prepared_event_id must be a string or null"):
            _normalize_default_off_run_specific_ids({"prepared_event_id": invalid_id})


def test_off_cache_profile_uses_omitted_defaults_without_explicit_cache_switches(tmp_path):
    profile_path = _write_off_cache_profile(tmp_path)
    switches = profiles.load(str(profile_path))

    assert not {
        "STATEFRAME_WORKING_SET_STABLE_FIRST",
        "STATEFRAME_CONSTANT_TOOL_MENU",
        "STATEFRAME_SHARED_CHECK_CACHE",
        "STATEFRAME_PROMPT_CACHE_UNPINNED",
    }.intersection(switches)


@pytest.mark.parametrize("setting, expected", [("off", None), ("on", "on")])
def test_historical_off_profile_does_not_send_the_retired_read_lockout_default(tmp_path, setting, expected):
    source = tmp_path / "released.json"
    source.write_text(json.dumps({
        "schema": "stateframe-product-profile/1",
        "switches": {"STATEFRAME_READ_LOCKOUT": setting},
    }))

    profile = _write_off_cache_profile(tmp_path, released_profile=source)

    assert profiles.load(str(profile)).get("STATEFRAME_READ_LOCKOUT") == expected


def _install_release_package(python: Path, package: Path, requirements: Path) -> None:
    assert requirements.is_file(), f"release dependency lock is missing: {requirements}"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    install = [str(python), "-I", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
               "--no-build-isolation", "--no-deps", "--no-input"]
    completed = subprocess.run([*install, "--force-reinstall", str(package)], env=environment,
                               capture_output=True, text=True, timeout=_scaled_test_timeout(180), check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    dependencies = subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
         "--require-hashes", "--no-input", "-r", str(requirements)],
        env=environment, capture_output=True, text=True, timeout=_scaled_test_timeout(300), check=False)
    assert dependencies.returncode == 0, dependencies.stdout + dependencies.stderr


@pytest.mark.parametrize("helper", ["parallel-e2e", "release-parity"])
def test_engine_venv_installs_bypass_shared_pip_cache(tmp_path, monkeypatch, helper):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "fixtures"))
    from fixtures.parallel.e2e_support import create_engine_venv

    monkeypatch.delenv("PIP_NO_CACHE_DIR", raising=False)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    if helper == "parallel-e2e":
        create_engine_venv(tmp_path / "engine-venv")
    else:
        root = Path(__file__).resolve().parents[1]
        _install_release_package(tmp_path / "engine-venv/bin/python", root, root / "requirements.txt")
    installs = [command for command in commands if "pip" in command and "install" in command]
    assert len(installs) == 2
    assert all("--no-cache-dir" in command for command in installs), installs


def _run_scripted_engine(python: Path, script: Path, repo: Path, run_dir: Path,
                         *, parallel_off: bool, endpoint: str, server,
                         allow_incomplete: bool = False, profile: str = "r2hc3") -> tuple[Path, str]:
    from stateframe.scripted_endpoint import Script, make_handler

    server.RequestHandlerClass = make_handler(Script(json.loads(script.read_bytes())))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_PROMPT_CACHE_UNPINNED", None)
    active_run = run_dir.with_name("active-run")
    assert not active_run.exists()
    launcher = Path(__file__).resolve().parent / "fixtures/parallel/off_release_launch.py"
    command = [str(python), "-I", str(launcher), "run", "--repo", str(repo),
               "--task", "Fix the pricing code so the tests pass.", "--endpoint", endpoint,
               "--model", "scripted", "--profile", profile, "--run-dir", str(active_run),
               "--max-turns", "16", "--test-python", str(python), "--shell", "standard"]
    if parallel_off:
        command.extend(("--parallel", "off"))
    completed = run_monitored(command, cwd=repo, env=environment, run_dir=active_run,
                              expected_seconds=_scaled_test_timeout(_DEFAULT_ENGINE_TIMEOUT),
                              step="parallel-off release parity", check=False)
    if active_run.exists():
        active_run.rename(run_dir)
    if not allow_incomplete:
        assert completed.returncode == 0, completed.stderr + completed.stdout
    return run_dir, endpoint


def _read_child_environment(python: Path, helper: Path, repo: Path, endpoint: str,
                            *, profile: str = "r2hc3") -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([str(python), "-I", str(helper), str(repo), endpoint, profile],
                               env=environment, capture_output=True, text=True,
                               timeout=_scaled_test_timeout(30), check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(completed.stdout)
    assert isinstance(value, dict) and all(isinstance(key, str) and isinstance(item, str)
                                            for key, item in value.items())
    return value


def _directory_effect_record(run_dir: Path) -> dict[str, object]:
    expected = "files changed by this command: created: generated"
    effects: list[dict[str, object]] = []
    for line in (run_dir / "tool-tape.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert isinstance(row, dict), "tool tape contains a non-object record"
        observation = row.get("observation")
        if (row.get("kind") == "complete" and row.get("exit_code") == 0
                and isinstance(observation, str) and expected in observation):
            effects.append(row)
    assert len(effects) == 1, f"expected one recorded directory effect, found {len(effects)}"
    return effects[0]


def _assert_directory_effect_values_match(left: dict[str, object], right: dict[str, object]) -> None:
    expected = "files changed by this command: created: generated"
    left_observation = left.get("observation")
    right_observation = right.get("observation")
    assert left.get("exit_code") == right.get("exit_code") == 0, "directory effect command did not succeed"
    assert isinstance(left_observation, str) and expected in left_observation, "left directory effect differs"
    assert isinstance(right_observation, str) and expected in right_observation, "right directory effect differs"
    assert left.get("post_manifest_digest") == right.get("post_manifest_digest"), (
        "directory effect changed the recorded workspace manifest")

    assert left_observation == right_observation, "directory effect observations differ"


def test_directory_effect_comparison_rejects_observation_changes():
    observation = "$ mkdir generated\nexit 0 | 0.25s | standard\nfiles changed by this command: created: generated"
    base = {"exit_code": 0, "post_manifest_digest": "stable", "observation": observation}
    _assert_directory_effect_values_match(base, base)

    for old, new in (("exit 0", "exit 9"), ("0.25s", "0.64s")):
        changed = {**base, "observation": observation.replace(old, new)}
        with pytest.raises(AssertionError, match="directory effect observations differ"):
            _assert_directory_effect_values_match(base, changed)


@pytest.mark.parametrize("parent_value", ["on", "off", None])
def test_prepare_worker_inherits_working_set_allocator_for_subframes(
        tmp_path, parent_value):
    snapshot, run_dir, spec_path, spec, parent_spec, _parent_run = _unit_fixture(tmp_path)
    key = "STATEFRAME_WORKING_SET_ALLOCATOR"
    if parent_value is not None:
        parent_spec["environment"][key] = parent_value
        parent_spec.setdefault("environment_sources", {})[key] = "parent:profile"

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    prepared = json.loads((run_dir / "spec.json").read_text())
    # The worker mirrors the parent exactly: same value, same provenance, and no pin.
    assert prepared["environment"].get(key) == parent_value
    if parent_value is None:
        assert key not in prepared["environment_sources"]
    else:
        assert prepared["environment_sources"][key] == "parent:profile"


def test_prepare_worker_binds_workspace_store_when_enabled(tmp_path, monkeypatch):
    snapshot, run_dir, spec_path, spec, parent_spec, _parent_run = _unit_fixture(tmp_path)
    # A 0002 umask makes a default 0777 directory group-writable; binding must secure it first.
    run_dir.chmod(0o775)
    stateframe_home = tmp_path / "stateframe-home"
    stateframe_home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(stateframe_home))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    parent_spec["environment"]["STATEFRAME_HOME"] = str(stateframe_home)
    parent_spec["environment"]["STATEFRAME_WORKSPACE_OBJECTS"] = "on"
    from stateframe.parallel_worker import _child_environment
    assert _child_environment({"environment": parent_spec["environment"]})["STATEFRAME_HOME"] == str(
        stateframe_home)

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    from stateframe.runner import _workspace_engine_imports
    with _workspace_engine_imports():
        from kernel_workspace_refs import open_run_store

        store = open_run_store(run_dir)
    assert store._path.is_dir()
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE((run_dir / ".workspace-objects.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(store._path.stat().st_mode) == 0o700
    binding = (run_dir / ".workspace-objects.json").read_bytes()
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    assert (run_dir / ".workspace-objects.json").read_bytes() == binding


def test_prepare_worker_builds_an_ordinary_isolated_run_from_verified_snapshot(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, parent_run = _unit_fixture(tmp_path)
    parent_spec["run_env"] = {"root": str(parent_run / "env"), "kind": "venv", "env": {}}
    parent_spec["environment"]["STATEFRAME_TEST_PYTHON"] = sys.executable
    parent_spec["tool_settings"]["tools_files"] = [{"path": str(tmp_path / "trusted-tools.json"),
                                                       "digest": "a" * 64}]
    parent_spec["custom_tools_digest"] = "b" * 64
    parent_spec["custom_executables"] = {"parent_only": "c" * 64}

    result = prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    assert result == run_dir
    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["schema"] == "stateframe-run/1"
    assert prepared["public_row"]["allowed_product_paths"] == ["pkg/a.py"]
    assert prepared["public_row"]["public_check_paths"] == ["tests/test_a.py"]
    assert prepared["public_row"]["msg"].startswith("Subframe a: change value")
    assert prepared["run_identity_seed"] == spec.run_seed
    assert prepared["unit_spec"] == spec.to_json()
    assert prepared["parallel_engine_python"] == parent_spec["capabilities"]["python"]
    assert prepared["parallel_mode"] == "off"
    assert "parallel_slots" not in prepared
    assert prepared["parallel_permit_root"] == str((parent_run / "parallel" / "check-permits").resolve())
    assert prepared["parallel_parent_endpoint"] == parent_spec["chat_url"]
    assert prepared["routes"] == parent_spec["routes"]
    assert prepared["routes_digest"] == parent_spec["routes_digest"]
    assert prepared["environment"]["PROFILE_VALUE"] == "retained"
    assert prepared["environment"]["STATEFRAME_TEST_PYTHON"] == sys.executable
    assert prepared["environment"]["STATEFRAME_ROUTE_SCOPE"] == "unit"
    assert prepared["environment"]["STATEFRAME_ROUTE_UNIT_CLASS"] == "default"
    assert prepared["environment_sources"]["STATEFRAME_ROUTE_SCOPE"] == "runner:unit-route"
    settings = ToolSettings.from_dict(prepared["tool_settings"])
    assert settings.shell == "off"
    assert settings.web is False and settings.git_tool is False and settings.run_env is False
    assert settings.shell_allow == () and settings.shell_read == () and settings.web_allow == ()
    assert prepared["test_cmd"] is None
    assert "run_env" not in prepared
    assert "resume" not in prepared and "parallel_recovery" not in prepared
    assert prepared["parallel_workspace_clone"] in {"apfs-clone", "reflink-auto", "copytree"}
    assert prepared["custom_tools_digest"] == load_tool_files((), builtin_names=frozenset(),
                                                               shell_network="none").digest
    assert prepared["custom_executables"] == {}
    assert not settings.tools_files
    parent = run_dir / "parent"
    assert parent.is_dir() and not parent.is_symlink()
    assert (parent / "pkg" / "a.py").read_bytes() == (snapshot / "pkg" / "a.py").read_bytes()
    assert scan(parent, exclude=lambda _: False)["pkg/a.py"].digest == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    workspace = run_dir / "workspace"
    assert set(DirectorySnapshot(workspace, spec.start_generation).list_files()) == set(
        DirectorySnapshot(snapshot, spec.start_generation).list_files()) - {"snapshot-manifest.json"}
    assert (workspace / "pkg" / "a.py").read_bytes() == (snapshot / "pkg" / "a.py").read_bytes()
    assert stat.S_IMODE((workspace / "pkg" / "a.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((parent / "pkg" / "a.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((workspace / "pkg" / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((parent / "pkg" / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((workspace / "pkg").stat().st_mode) == 0o700
    assert stat.S_IMODE((parent / "pkg").stat().st_mode) == 0o700
    assert sorted(path.name for path in run_dir.glob(".workspace-*")) == []


def test_prepare_worker_projects_parent_task_file_scope(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["public_row"]["task_file_scope"] = ["pkg/a.py"]

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["public_row"]["task_file_scope"] == ["pkg/a.py"]


def test_legacy_parent_without_recorded_task_scope_keeps_base_projection(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["public_row"] = {"msg": "Only edit pkg/a.py."}
    start_files = {path: (snapshot / path).read_bytes()
                   for path in ("pkg/a.py", "tests/test_a.py")}
    expected = build_public_row(spec, start_files)

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["public_row"] == expected
    assert "task_file_scope" not in prepared["public_row"]


def test_unit_task_builder_omits_empty_legacy_test_fields_when_no_checks(tmp_path):
    _snapshot, _run_dir, _spec_path, spec, _parent_spec, _parent_run = _unit_fixture(tmp_path)
    no_checks = replace(spec, contract=replace(
        spec.contract, check=CheckRecipe((), (), "none")))

    row = build_public_row(no_checks, {"pkg/a.py": b"VALUE = 1\n"})

    assert row["public_tests"] == {}
    assert "public_check_paths" not in row
    assert "test_path" not in row
    assert "test_src" not in row


def test_unit_task_builder_types_invalid_check_source_path(tmp_path):
    _snapshot, _run_dir, _spec_path, spec, _parent_spec, _parent_run = _unit_fixture(tmp_path)
    invalid_check = replace(spec, contract=replace(
        spec.contract, check=CheckRecipe(("",), ("tests/test_a.py::test_value",), "pytest")))

    with pytest.raises(ValueError, match="INVALID_PUBLIC_TEST_PATH"):
        build_public_row(invalid_check, {"pkg/a.py": b"VALUE = 1\n"})


def _parent_hooks(root: Path, parent_run: Path, parent_spec: dict) -> Path:
    source = root / "hook-source"
    source.mkdir()
    script = source / "guard.py"
    script.write_text("print('inherited guard')\n")
    config = source / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [{
        "name": "guard", "event": "pre_tool", "tools": ["str_replace"],
        "argv": [sys.executable, "guard.py"], "files": ["guard.py"],
    }]}))
    hooks = load_hook_files([config], shell_network="none", repo=root)
    manifest = create_bundle(hooks, parent_run / "hook-bundle")
    parent_spec["hooks"] = {**hooks.as_spec(manifest), "start_context": []}
    return script


def test_prepare_worker_copies_authenticated_parent_hook_bundle(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, parent_run = _unit_fixture(tmp_path)
    source = _parent_hooks(tmp_path, parent_run, parent_spec)
    # Provision from the authenticated run artifact, never reload mutable source files.
    source.write_text("print('changed source')\n")

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["hooks"] == parent_spec["hooks"]
    verify_bundle(prepared["hooks"]["bundle_manifest"], run_dir / "hook-bundle")
    copied = run_dir / "hook-bundle/0/guard.py"
    assert copied.read_text() == "print('inherited guard')\n"
    assert copied.stat().st_ino != (parent_run / "hook-bundle/0/guard.py").stat().st_ino
    assert prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec) == run_dir


@pytest.mark.parametrize("location", ["parent", "worker"])
def test_prepare_worker_refuses_changed_hook_bundle(tmp_path, location):
    snapshot, run_dir, spec_path, spec, parent_spec, parent_run = _unit_fixture(tmp_path)
    _parent_hooks(tmp_path, parent_run, parent_spec)
    if location == "worker":
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    bundle = (parent_run if location == "parent" else run_dir) / "hook-bundle"
    script = bundle / "0/guard.py"
    script.chmod(0o600)
    script.write_text("print('tampered bundle')\n")
    script.chmod(0o400)

    with pytest.raises(WorkerPreparationError, match="hook bundle"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    if location == "parent":
        assert not (run_dir / "spec.json").exists()


def test_prepare_worker_verifies_copied_hook_bundle_before_spec_publication(tmp_path, monkeypatch):
    snapshot, run_dir, spec_path, spec, parent_spec, parent_run = _unit_fixture(tmp_path)
    _parent_hooks(tmp_path, parent_run, parent_spec)
    copytree = shutil.copytree

    def corrupt_copy(source, destination, *args, **kwargs):
        result = copytree(source, destination, *args, **kwargs)
        if Path(source) == parent_run / "hook-bundle":
            copied = Path(destination) / "0/guard.py"
            copied.chmod(0o600)
            copied.write_text("print('corrupted copy')\n")
            copied.chmod(0o400)
        return result

    monkeypatch.setattr(shutil, "copytree", corrupt_copy)
    with pytest.raises(WorkerPreparationError, match="hook bundle"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    assert not (run_dir / "spec.json").exists()


def test_prepare_worker_retry_refuses_a_changed_hook_contract(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, parent_run = _unit_fixture(tmp_path)
    _parent_hooks(tmp_path, parent_run, parent_spec)
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    parent_spec.pop("hooks")

    with pytest.raises(WorkerPreparationError, match="frozen subframe spec"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)


def test_prepare_worker_rejects_normalized_non_test_check_path_before_launch(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _parent_run = _unit_fixture(tmp_path)
    invalid_check = replace(spec, contract=replace(
        spec.contract, check=CheckRecipe(("pkg/a.py",), ("pkg/a.py::test_value",), "pytest")))
    spec_path.write_bytes(canonical_json(invalid_check.to_json()) + b"\n")

    with pytest.raises(WorkerPreparationError, match="INVALID_PUBLIC_TEST_PATH.*pytest test file"):
        prepare_worker(invalid_check, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    assert not (run_dir / "spec.json").exists()
    assert not (run_dir / "workspace").exists()


def test_repo_context_reuses_readonly_content_addressed_snapshot(tmp_path, monkeypatch):
    import stateframe._child as child_module

    monkeypatch.setitem(child_module.HOST_STATE, "repo_context", None)
    snapshot, _run_dir, _spec_path, spec, _parent_spec, parent_run = _unit_fixture(
        tmp_path, executable_helper=False)
    snapshot_digest = hashlib.sha256(canonical_json(
        json.loads((snapshot / "snapshot-manifest.json").read_bytes()))).hexdigest()
    public_source = "def test_value(): assert True\n"

    roots = [tmp_path / "engine-a", tmp_path / "engine-b"]
    installed_contexts = []
    for root in roots:
        (root / "pkg").mkdir(parents=True)
        (root / "pkg/a.py").write_text("candidate value\n")
        class Engine:
            def __init__(self):
                self._materialize_public_tests = self.materialize

            @staticmethod
            def materialize(public_root, _fileset, _row):
                path = public_root / "tests/test_a.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(public_source)

        engine = Engine()
        install_repo_context(
            engine, parent_run / "parent", shared_snapshot=snapshot,
            snapshot_manifest_digest=snapshot_digest,
            source_manifest_digest=spec.start_manifest_digest,
            excluded_paths=("pkg/a.py", "tests/test_a.py"),
        )
        engine._materialize_public_tests(root, {}, {})
        installed_contexts.append(dict(child_module.HOST_STATE["repo_context"]))

    first_shared = roots[0] / "pkg/run.sh"
    second_shared = roots[1] / "pkg/run.sh"
    assert first_shared.stat().st_ino != second_shared.stat().st_ino
    assert first_shared.stat().st_ino != (snapshot / "pkg/run.sh").stat().st_ino
    assert first_shared.stat().st_nlink == second_shared.stat().st_nlink == 1
    assert stat.S_IMODE(first_shared.stat().st_mode) == 0o644
    assert (snapshot / "pkg/run.sh").stat().st_mode & 0o222 == 0
    assert all(context["shared_snapshot"] == snapshot
               and context["shared_snapshot_digest"] == snapshot_digest
               for context in installed_contexts)
    first_shared.write_text("unit a private copy\n")
    assert second_shared.read_text() == "#!/bin/sh\nexit 0\n"
    assert (snapshot / "pkg/run.sh").read_text() == "#!/bin/sh\nexit 0\n"
    assert (roots[0] / "pkg/a.py").stat().st_ino != (snapshot / "pkg/a.py").stat().st_ino
    assert (roots[0] / "tests/test_a.py").stat().st_ino != (roots[1] / "tests/test_a.py").stat().st_ino
    (roots[0] / "tests/test_a.py").write_text("unit a\n")
    assert (roots[1] / "tests/test_a.py").read_text() == public_source

    executable = parent_run / "parent/pkg/run.sh"
    executable.chmod(0o644)
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o755)
    changed = _snapshot(parent_run / "parent", tmp_path / "changed" / "parallel" / "call", 5)
    changed_root = tmp_path / "engine-changed"
    (changed_root / "pkg").mkdir(parents=True)
    (changed_root / "pkg/a.py").write_text("candidate value\n")
    class ChangedEngine:
        def __init__(self):
            self._materialize_public_tests = self.materialize

        @staticmethod
        def materialize(public_root, _fileset, _row):
            path = public_root / "tests/test_a.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(public_source)

    changed_engine = ChangedEngine()
    install_repo_context(
        changed_engine, parent_run / "parent", shared_snapshot=changed.root,
        snapshot_manifest_digest=changed.snapshot_manifest_digest,
        source_manifest_digest=changed.manifest_digest,
        excluded_paths=("pkg/a.py", "tests/test_a.py"),
    )
    changed_engine._materialize_public_tests(changed_root, {}, {})
    assert (changed_root / "pkg/run.sh").read_text() == "#!/bin/sh\nexit 1\n"
    assert (changed_root / "pkg/run.sh").stat().st_ino != first_shared.stat().st_ino
    assert stat.S_IMODE((changed_root / "pkg/run.sh").stat().st_mode) == 0o755


def test_prepare_worker_carries_the_recorded_slot_pool_in_operator_order(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["parallel_slots"] = [5, 2]

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    worker_spec = json.loads((run_dir / "spec.json").read_text())
    assert worker_spec["parallel_slots"] == [5, 2]


def test_prepare_worker_leases_the_resolved_subframe_endpoint(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["prompt_cache_slot"] = {"slot": 7, "selection": "explicit", "reason": "parent"}
    default = {"endpoint": "http://127.0.0.1:8080/v1", "model": "local-model",
               "api_key_env": None, "temperature": 1.0, "max_tokens": 24000,
               "timeout_s": 1800.0, "slot": None, "context_tokens": 65536,
               "allow_remote": False}
    parent_spec["routes"] = routes.resolve(
        default, cli=["unit.endpoint=http://127.0.0.1:9090/v1"], environ={}).public()

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    worker_spec = json.loads((run_dir / "spec.json").read_text())
    assert worker_spec["parallel_parent_endpoint"] == "http://127.0.0.1:9090/v1/chat/completions"
    assert worker_spec["prompt_cache_slot"]["slot"] is None


def test_unscoped_worker_projection_is_byte_identical_to_base(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["public_row"] = {"msg": "parent task"}
    start_files = {path: (snapshot / path).read_bytes()
                   for path in ("pkg/a.py", "tests/test_a.py")}
    expected = build_public_row(spec, start_files)

    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    prepared = json.loads((run_dir / "spec.json").read_text())
    assert canonical_json(prepared["public_row"]) == canonical_json(expected)


@pytest.mark.host_sandbox
def test_routed_worker_sends_unit_class_provider_credential(monkeypatch, tmp_path, tmp_path_factory):
    from stateframe.scripted_endpoint import Script, serve

    script = Script({"chat": {"require_key": "fake-unit-key"},
                     "slots": [65536, 65536],
                     "tokenize": {"mode": "ok"},
                     "steps": [{"tool": "done", "arguments": {"summary": "Checked."}, "repeat": True}]})
    server = serve(script)
    handle = None
    try:
        snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
        monkeypatch.setenv("SF_TEST_PARENT_KEY", "fake-parent-key")
        monkeypatch.setenv("SF_TEST_UNIT_KEY", "fake-unit-key")
        parent_spec["api_key_env"] = "SF_TEST_PARENT_KEY"
        parent_spec["chat_url"] = endpoint + "/chat/completions"
        parent_spec["environment"].update(_pin_off_cache_switches(profiles.load("r2hc3")))
        # prepare() consumes this runner setting before recording the engine environment.
        parent_spec["environment"].pop("STATEFRAME_PROTECT_TESTS", None)
        parent_spec["environment"].update(DS_URL=parent_spec["chat_url"], DS_MODEL="scripted")
        # Slot leases use HOME independently of the session's configuration directory.
        child_home = tmp_path / "child-home"
        child_home.mkdir()
        parent_spec["environment"]["HOME"] = str(child_home)
        # The runner records the installation-identity pin and home in a real parent's spec.
        from stateframe.engine import kernel_request_auth_identity
        # The pinned home must lie outside the run directory and the repository (tmp_path).
        stateframe_home = tmp_path_factory.mktemp("stateframe-home")
        stateframe_home.chmod(0o700)
        parent_spec["environment"]["STATEFRAME_HOME"] = str(stateframe_home)
        parent_spec["environment"]["STATEFRAME_RUNNER_HOME_IDENTITY"] = json.dumps(
            kernel_request_auth_identity.protected_home_identity(stateframe_home, initialize=True),
            sort_keys=True, separators=(",", ":"))
        parent_spec["routes"] = routes.resolve(
            {"endpoint": endpoint, "model": "scripted", "api_key_env": "SF_TEST_PARENT_KEY"},
            cli=[f"unit.{spec.contract.model_class}.api_key_env=SF_TEST_UNIT_KEY"],
            environ={"SF_TEST_PARENT_KEY": "fake-parent-key", "SF_TEST_UNIT_KEY": "fake-unit-key"}).public()
        parent_spec["routes_digest"] = parent_spec["routes"]["digest"]
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

        handle = launch_worker(run_dir, 1)
        assert handle.reap(timeout=_scaled_test_timeout(60)) is not None

        assert script.chat_authorized, "worker did not reach the provider"
        assert all(script.chat_authorized), "worker sent an incorrect provider credential"
        assert all(script.chat_authorization_present)
        assert "fake-parent-key" not in (run_dir / "spec.json").read_text()
        assert "fake-unit-key" not in (run_dir / "spec.json").read_text()
        assert "fake-unit-key" not in handle.log_path.read_text()
    finally:
        if handle is not None and handle.poll() is None:
            handle.cancel("test cleanup")
            handle.escalate(grace=0)
            handle.reap(timeout=10)
        server.shutdown()
        server.server_close()


def test_prepare_worker_refuses_routes_spanning_multiple_endpoints(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    default = {"endpoint": "http://127.0.0.1:8080/v1", "model": "local-model",
               "api_key_env": None, "temperature": 1.0, "max_tokens": 24000,
               "timeout_s": 1800.0, "slot": None, "context_tokens": 65536,
               "allow_remote": False}
    parent_spec["routes"] = routes.resolve(
        default, cli=["read.endpoint=http://127.0.0.1:9090/v1"], environ={}).public()

    with pytest.raises(WorkerPreparationError, match="multiple model endpoints"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    assert not (run_dir / "spec.json").exists()


def test_prepare_worker_retry_refuses_a_changed_slot_pool(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["parallel_slots"] = [5, 2]
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    parent_spec["parallel_slots"] = [2, 5]

    with pytest.raises(WorkerPreparationError, match="frozen subframe spec"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)


def test_prepare_worker_refuses_a_snapshot_changed_after_projection(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    (snapshot / "pkg" / "a.py").chmod(0o644)
    (snapshot / "pkg" / "a.py").write_bytes(b"VALUE = 99\n")

    with pytest.raises(WorkerPreparationError, match="snapshot file differs"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    assert not (run_dir / "spec.json").exists()
    assert not (run_dir / "workspace").exists()


def test_prepare_worker_refuses_unmanifested_snapshot_symlinks(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path, include_symlink=True)

    with pytest.raises(WorkerPreparationError, match="inventory differs"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    assert not (run_dir / "workspace").exists()


def test_prepare_worker_keeps_changed_workspace_for_same_execution(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    workspace = run_dir / "workspace" / "pkg" / "a.py"
    workspace.chmod(0o644)
    workspace.write_bytes(b"VALUE = 2\n")
    workspace.chmod(0o444)
    original_spec = (run_dir / "spec.json").read_bytes()

    assert prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec) == run_dir
    assert workspace.read_bytes() == b"VALUE = 2\n"
    assert (run_dir / "spec.json").read_bytes() == original_spec


def test_prepare_worker_refuses_changed_parent_snapshot_on_retry(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    parent_file = run_dir / "parent" / "pkg" / "a.py"
    parent_file.chmod(0o644)
    parent_file.write_bytes(b"VALUE = 99\n")

    with pytest.raises(WorkerPreparationError, match="parent snapshot differs"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)


def test_prepare_worker_requires_recorded_parent_engine_python(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["capabilities"].pop("python")

    with pytest.raises(WorkerPreparationError, match="recorded engine interpreter"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    assert not (run_dir / "spec.json").exists()


def test_prepare_worker_requires_replayable_run_seed(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path, run_seed="not-a-seed")

    with pytest.raises(WorkerPreparationError, match="32 lowercase hex"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    assert not (run_dir / "spec.json").exists()


def test_launch_worker_refuses_missing_recorded_parent_interpreter(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    prepared_path = run_dir / "spec.json"
    prepared = json.loads(prepared_path.read_bytes())
    prepared.pop("parallel_engine_python")
    prepared_path.write_text(json.dumps(prepared))

    with pytest.raises(WorkerLaunchError, match="recorded parent engine interpreter"):
        launch_worker(run_dir, 1)
    assert not (run_dir / "proc").exists()


def test_launch_intent_is_durable_before_spawn(tmp_path, monkeypatch):
    import stateframe.parallel_worker as worker_module

    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    observed = []

    def refuse_spawn(command, **_kwargs):
        intent = json.loads((run_dir / "proc" / "1" / "launch-intent.json").read_text())
        observed.append(intent)
        raise OSError("injected spawn refusal")

    monkeypatch.setattr(worker_module.subprocess, "Popen", refuse_spawn)
    with pytest.raises(WorkerLaunchError, match="injected spawn refusal"):
        launch_worker(run_dir, 1)
    assert observed == [{
        "schema": "stateframe-subframe-launch-intent/1", "incarnation": 1,
        "run_dir": str(run_dir),
        "command": [parent_spec["capabilities"]["python"], "-B",
                    str(Path(parent_spec["engine_dir"]).parent / "_child.py"), str(run_dir)],
    }]


def test_subframe_resume_refuses_unmanifested_snapshot_symlinks(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path, include_symlink=True)
    with pytest.raises(WorkerPreparationError, match="inventory differs"):
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)


def test_launch_probe_failure_sweeps_group_before_wait(tmp_path, monkeypatch):
    import stateframe.parallel_worker as worker_module

    package = tmp_path / "package"
    engine_dir = package / "engine"
    engine_dir.mkdir(parents=True)
    (package / "_child.py").write_text("pass\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1", "public_row": {}, "engine_dir": str(engine_dir),
        "parallel_engine_python": sys.executable, "environment": {},
    }))
    calls = []

    class Process:
        pid = 4321
        stdout = None

        def kill(self):
            calls.append("kill-leader")

        def wait(self, timeout=None):
            calls.append("wait")
            return 0

    process = Process()
    identity = LaunchIdentity(process.pid, (100, 1), None)
    monkeypatch.setattr(worker_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(worker_module, "capture_launch_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(worker_module, "kernel_start_identity", lambda pid: None)

    def sweep(child, *, launch_identity):
        assert child is process and launch_identity is identity
        calls.append("sweep-group")
        child.wait(timeout=1)
        return ()

    monkeypatch.setattr(worker_module, "kill_launch_unit", sweep)
    with pytest.raises(WorkerLaunchError, match="kernel identity"):
        launch_worker(run_dir, 1)
    assert calls == ["sweep-group", "wait"]


def test_launch_without_identity_reports_unverified_without_reaping(tmp_path, monkeypatch):
    import stateframe.parallel_worker as worker_module

    package = tmp_path / "package"
    engine_dir = package / "engine"
    engine_dir.mkdir(parents=True)
    (package / "_child.py").write_text("pass\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1", "public_row": {}, "engine_dir": str(engine_dir),
        "parallel_engine_python": sys.executable, "environment": {},
    }))
    calls = []

    class Process:
        pid = 4321
        stdout = None

        def kill(self):
            calls.append("kill-leader")

        def wait(self, timeout=None):
            calls.append("wait")
            return 0

    process = Process()
    monkeypatch.setattr(worker_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(worker_module, "capture_launch_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module.os, "kill",
                        lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(WorkerLaunchError, match="containment unverified.*left unreaped"):
        launch_worker(run_dir, 1)
    assert calls == [(4321, signal.SIGKILL)]


def test_parallel_off_keeps_main_run_spec_keys_and_child_env(tmp_path, monkeypatch):
    from stateframe.skills import skills_digest

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    options = RunOptions(repo=repo, task="inspect sample",
                         endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "local-model", None),
                         run_dir=tmp_path / "run", max_turns=1, parallel_mode="off")

    run, _, spec = prepare(options)

    assert run == tmp_path / "run"
    assert "parallel_mode" not in spec and "parallel_limits" not in spec
    assert "parallel_engine_python" not in spec
    assert "STATEFRAME_ROUTE_SCOPE" not in spec["environment"]
    assert "STATEFRAME_ROUTE_UNIT_CLASS" not in spec["environment"]
    assert spec["skills"] == []
    assert spec["skills_digest"] == skills_digest([])
    assert set(spec) == {
        "schema", "created", "engine_dir", "engine_source", "profile", "capabilities", "chat_url", "model",
        "api_key_env", "max_turns", "parent_snapshot", "public_row", "tool_settings", "custom_tools_digest",
        "custom_executables", "environment", "environment_sources", "prompt_cache_slot", "test_cmd",
        "run_identity_seed", "repo", "in_place", "skills", "skills_digest",
        "check_cmd", "check_source", "runner_kind", "protected_test_globs", "protect_tests",
    }

    for name in ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER", "LOGNAME", "SHELL",
                 "SYSTEMROOT", "TERM", "LLAMA_CPP_TOKENIZE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", "http://127.0.0.1:8080/tokenize")
    monkeypatch.setenv("STATEFRAME_PARALLEL_MODE", "must-not-leak")
    assert _child_env(options) == {
        "PATH": "/usr/bin", "HOME": str(tmp_path),
        "LLAMA_CPP_TOKENIZE_URL": "http://127.0.0.1:8080/tokenize",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        # 0.6.38: every runner child receives the installation-identity pin.
        "STATEFRAME_RUNNER_HOME_IDENTITY": json.dumps(
            options._stateframe_home_identity, sort_keys=True, separators=(",", ":")),
    }
    assert "parallel" not in summarize(run, 0.0)


def test_parallel_off_ignores_ambient_limits_and_slots_before_prepare(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fixtures.parallel.off_release_launch import RunDateTime

    from stateframe import runner

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample(): assert True\n")
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "local-model", None)
    monkeypatch.setattr(runner, "datetime", RunDateTime)
    monkeypatch.setattr(runner, "secrets", SimpleNamespace(token_hex=lambda size: "ab" * size))

    def spec_for(name):
        active, _, spec = prepare(RunOptions(repo=repo, task="inspect sample", endpoint=endpoint,
                                             run_dir=tmp_path / "active", max_turns=1,
                                             parallel_mode="off"))
        published = (active / "spec.json").read_bytes()
        active.rename(tmp_path / name)
        return published, spec

    for key in ("STATEFRAME_PARALLEL_SLOTS", "STATEFRAME_PARALLEL_MAX_CONCURRENCY",
                "STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS",
                "STATEFRAME_PARALLEL_FANOUT_CONTINUATION"):
        monkeypatch.delenv(key, raising=False)
    baseline_disk, baseline = spec_for("baseline")
    monkeypatch.setenv("STATEFRAME_PARALLEL_SLOTS", "2,5")
    monkeypatch.setenv("STATEFRAME_PARALLEL_MAX_CONCURRENCY", "8")
    monkeypatch.setenv("STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS", "9")
    monkeypatch.setenv("STATEFRAME_PARALLEL_FANOUT_CONTINUATION", "on")
    ambient_disk, ambient = spec_for("ambient")
    for recorded in (json.loads(baseline_disk), json.loads(ambient_disk), baseline, ambient):
        assert "parallel_slots" not in recorded
        assert "parallel_limits" not in recorded
        assert not any(key.startswith("STATEFRAME_PARALLEL_")
                       for key in recorded["environment_sources"])
    assert ambient_disk == baseline_disk
    assert ambient["environment_sources"] == baseline["environment_sources"]
    assert canonical_json(ambient) == canonical_json(baseline)


@pytest.mark.host_sandbox
def test_launch_worker_starts_ordinary_child_and_drains_output(tmp_path):
    package = tmp_path / "fake" / "stateframe"
    engine_dir = package / "engine"
    engine_dir.mkdir(parents=True)
    child = package / "_child.py"
    child.write_text("print('worker-output')\n")
    run_dir = tmp_path / "worker"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1", "public_row": {}, "engine_dir": str(engine_dir),
        "parallel_engine_python": sys.executable, "environment": {},
    }))

    handle = launch_worker(run_dir, 1)
    code = handle.reap(timeout=10)

    assert code == 0
    assert handle.process.args[0] == str(sys.executable)
    assert handle.pid == handle.pgid
    assert (run_dir / "proc" / "1" / "process.json").is_file()
    assert handle.log_path.read_text() == "worker-output\n"


@pytest.mark.host_sandbox
def test_spawn_to_process_record_crash_keeps_durable_launch_intent(tmp_path, monkeypatch):
    import stateframe.parallel_worker as worker_module
    from stateframe.parallel_host import ParallelHostError, fence_and_reap
    from stateframe.engine.kernel_parallel_records import ExecutionState, RecoveryDescriptor

    package = tmp_path / "fake" / "stateframe"
    engine_dir = package / "engine"
    engine_dir.mkdir(parents=True)
    (package / "_child.py").write_text("import time\ntime.sleep(30)\n")
    parent_run = tmp_path / "run"
    run_dir = parent_run / "parallel" / "call" / "units" / "a"
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1", "public_row": {}, "engine_dir": str(engine_dir),
        "parallel_engine_python": sys.executable, "environment": {},
    }))
    launched = []
    original_popen = worker_module.subprocess.Popen
    original_write = worker_module._write_json

    def capture_spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    def crash_before_identity(path, value):
        if path.name == "process.json":
            raise SystemExit("injected host crash")
        return original_write(path, value)

    monkeypatch.setattr(worker_module.subprocess, "Popen", capture_spawn)
    monkeypatch.setattr(worker_module, "_write_json", crash_before_identity)
    try:
        with pytest.raises(SystemExit, match="injected host crash"):
            launch_worker(run_dir, 1)
        intent_path = run_dir / "proc" / "1" / "launch-intent.json"
        assert json.loads(intent_path.read_text()) == {
            "schema": "stateframe-subframe-launch-intent/1", "incarnation": 1,
            "run_dir": str(run_dir),
            "command": [sys.executable, "-B", str(package / "_child.py"), str(run_dir)],
        }
        assert not (run_dir / "proc" / "1" / "process.json").exists()
        descriptor = RecoveryDescriptor("call", "a" * 64, (), (),
                                        (ExecutionState("a", "execution", str(run_dir), "b" * 64,
                                                        1, None),))
        with pytest.raises(ParallelHostError, match="complete process identity") as error:
            fence_and_reap(parent_run, descriptor)
        assert str(intent_path.parent) in str(error.value)
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_restart_refuses_missing_prior_incarnation_record(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

    with pytest.raises(WorkerLaunchError, match="prior incarnation process record"):
        launch_worker(run_dir, 2)


@pytest.mark.host_sandbox
def test_restart_archives_same_execution_and_replays_from_immutable_snapshot(tmp_path):
    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["public_row"]["task_file_scope"] = ["pkg/a.py"]
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    changed = run_dir / "workspace" / "pkg" / "a.py"
    changed.chmod(0o644)
    changed.write_bytes(b"VALUE = 2\n")
    changed.chmod(0o444)
    original = json.loads((run_dir / "spec.json").read_text())
    prior = run_dir / "proc" / "1"
    prior.mkdir(parents=True)
    (prior / "process.json").write_text(json.dumps({"incarnation": 1, "pid": 100, "pgid": 100}))

    handle = launch_worker(run_dir, 2)
    handle.reap(timeout=30)

    resumed = json.loads((run_dir / "spec.json").read_text())
    assert resumed["unit_spec"] == original["unit_spec"]
    assert resumed["run_identity_seed"] == original["run_identity_seed"]
    assert resumed["public_row"]["task_file_scope"] == ["pkg/a.py"]
    assert resumed["resume"]["attempt"] == 2
    assert (run_dir / "attempts" / "1" / "workspace" / "pkg" / "a.py").read_bytes() == b"VALUE = 2\n"
    assert changed.read_bytes() == (snapshot / "pkg" / "a.py").read_bytes()
    assert (run_dir / "proc" / "2" / "process.json").is_file()


@pytest.mark.host_sandbox
@pytest.mark.parametrize("record_scope", [True, False], ids=["scoped", "legacy"])
def test_worker_scope_after_live_restart(tmp_path, tmp_path_factory, monkeypatch, record_scope):
    from fixtures.barrier_endpoint import BarrierEndpoint
    from stateframe.engine.kernel_request_auth_identity import protected_home_identity

    snapshot, run_dir, spec_path, spec, parent_spec, _ = _unit_fixture(tmp_path)
    parent_spec["public_row"] = {"msg": "Only edit pkg/a.py."}
    if record_scope:
        parent_spec["public_row"]["task_file_scope"] = ["pkg/a.py"]
    parent_spec.pop("routes")
    parent_spec.pop("routes_digest")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    stateframe_home = tmp_path_factory.mktemp("stateframe-home")
    stateframe_home.chmod(0o700)
    steps = [
        {"tool": "read", "arguments": {"path": "pkg/a.py"}},
        {"tool": "subframe_blocked", "arguments": {
            "reason_code": "OTHER", "detail": "Script completed after replay."}},
    ]

    with BarrierEndpoint([], {"a": steps}, hold_subframe_request={"a": 2},
                         barrier_timeout=_scaled_test_timeout(240)) as endpoint:
        parent_spec["chat_url"] = endpoint.url + "/chat/completions"
        parent_spec["model"] = "scripted"
        environment = parent_spec["environment"]
        environment.update(_pin_off_cache_switches(profiles.load("r2hc3")))
        environment.pop("STATEFRAME_PROTECT_TESTS", None)
        environment.update(DS_URL=parent_spec["chat_url"], DS_MODEL="scripted", HOME=str(home),
                           STATEFRAME_HOME=str(stateframe_home),
                           STATEFRAME_RUNNER_HOME_IDENTITY=json.dumps(
                               protected_home_identity(stateframe_home, initialize=True),
                               sort_keys=True, separators=(",", ":")))
        prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)

        first = launch_worker(run_dir, 1)
        try:
            assert endpoint.wait_for_requests("a", 2, timeout=_scaled_test_timeout(120)), (
                first.log_path.read_text())
        finally:
            first.cancel("test interruption at the second provider request")
            first.escalate(grace=0)
            assert first.reap(timeout=15) is not None

        restarted = launch_worker(run_dir, 2)
        try:
            assert restarted.reap(timeout=_scaled_test_timeout(120)) is not None, restarted.log_path.read_text()
        finally:
            if restarted.exit_code is None:
                restarted.cancel("test cleanup")
                restarted.escalate(grace=0)
                restarted.reap(timeout=15)
        endpoint.assert_consumed()
        assert len(endpoint.requests) == 3

    resumed = json.loads((run_dir / "spec.json").read_text())
    assert resumed["resume"]["attempt"] == 2
    assert resumed["resume"]["tape"], "the completed read must be replayed"
    assert '"kind":"replay_done"' in restarted.log_path.read_text()
    assert json.loads((run_dir / "engine-result.json").read_text())["result"]["result"] == "SUBFRAME_BLOCKED"
    assert not (run_dir / "resume-divergence.json").exists()
    if record_scope:
        assert resumed["public_row"]["task_file_scope"] == ["pkg/a.py"]
    else:
        assert "task_file_scope" not in resumed["public_row"]


def _latest_stable_release_commit(root: Path) -> str:
    """Use the newest reachable stable version tag as the parity baseline.

    Merge commits can put a banner or other side branch at their second parent, so that parent is
    not a reliable release baseline. Select the highest numeric vX.Y.Z tag reachable from HEAD;
    prerelease tags and version tags outside HEAD's history are ignored.
    """
    tags = subprocess.run(["git", "-C", str(root), "tag", "--merged", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    versions = []
    for tag in tags:
        match = re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag)
        if match:
            versions.append((tuple(int(part) for part in match.groups()), tag))
    if not versions:
        raise RuntimeError("No stable version release tag matching vX.Y.Z is reachable from HEAD")

    latest_tag = max(versions, key=lambda item: item[0])[1]
    return subprocess.run(["git", "-C", str(root), "rev-parse", "--verify",
                           f"refs/tags/{latest_tag}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_latest_stable_release_commit_survives_banner_merge_and_rc_commit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "StateFrame tests")
    git("config", "user.email", "stateframe-tests@example.invalid")
    (root / "version.txt").write_text("0.4.9\n")
    git("add", "version.txt")
    git("commit", "-q", "-m", "release 0.4.9")
    git("tag", "v0.4.9")

    (root / "version.txt").write_text("0.4.38\n")
    git("commit", "-q", "-am", "release 0.4.38")
    git("tag", "-a", "v0.4.38", "-m", "release 0.4.38")
    release_commit = git("rev-parse", "HEAD")

    git("switch", "-q", "-c", "banner")
    (root / "banner.txt").write_text("banner\n")
    git("add", "banner.txt")
    git("commit", "-q", "-m", "add banner")
    side_branch_tip = git("rev-parse", "HEAD")
    git("tag", "v999.0.0-rc1")

    git("switch", "-q", "main")
    git("merge", "--no-ff", "--no-edit", "banner")

    assert _latest_stable_release_commit(root) == release_commit

    def old_helper(repo):
        merge = subprocess.run(["git", "-C", str(repo), "rev-list", "--first-parent", "--merges",
                                "-n", "1", "HEAD"], capture_output=True, text=True,
                               check=True).stdout.strip()
        assert merge, "lane has no main merge to compare against"
        return subprocess.run(["git", "-C", str(repo), "rev-parse", f"{merge}^2"],
                              capture_output=True, text=True, check=True).stdout.strip()

    assert old_helper(root) == side_branch_tip

    git("commit", "--allow-empty", "-m", "RC implementation")
    assert _latest_stable_release_commit(root) == release_commit


def test_latest_stable_release_commit_fails_without_reachable_tag(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "StateFrame tests")
    git("config", "user.email", "stateframe-tests@example.invalid")
    (root / "file.txt").write_text("no release tag\n")
    git("add", "file.txt")
    git("commit", "-q", "-m", "unreleased commit")

    message = "No stable version release tag matching vX.Y.Z is reachable from HEAD"
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        _latest_stable_release_commit(root)


def _expected_build_metadata(source_tree: Path) -> dict[str, object]:
    # A build publishes its own fingerprint. Read the exact input for that build
    # through the product's loader, including when the test changes the profile location.
    with patch.object(profiles, "HERE", source_tree / "src/stateframe/profiles"):
        metadata = profiles.engine_source_summary()
    assert metadata["shipped_hash"]
    return metadata


def _archive_off_release(root: Path, destination: Path) -> None:
    destination.mkdir()
    archive = destination.with_suffix(".tar")
    with archive.open("wb") as output:
        archived = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", _OFF_RELEASE_COMMIT],
                                  stdout=output, stderr=subprocess.PIPE, check=False,
                                  timeout=_scaled_test_timeout(120))
    assert archived.returncode == 0, (
        f"Could not archive release 0.8.5 ({_OFF_RELEASE_COMMIT}) from the checkout's Git object store:\n"
        f"{archived.stderr.decode(errors='replace')}")
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")


@pytest.mark.host_sandbox
def test_default_parallel_off_wire_and_public_artifacts_match_merged_main(tmp_path):
    root = Path(__file__).resolve().parents[1]
    old = tmp_path / "main-base"
    _archive_off_release(root, old)
    off_profile = str(_write_off_cache_profile(
        tmp_path, released_profile=old / "src/stateframe/profiles/r2hc3.json"))
    repo = tmp_path / "repo"
    shutil.copytree(old / "examples" / "discount_bug", repo)
    script = old / "examples" / "discount_bug.script.json"
    directory_effect_script = tmp_path / "directory-effect-script.json"
    directory_effect_data = json.loads(script.read_text())
    directory_effect_data["steps"].insert(1, {
        "tool": "shell", "required": True, "arguments": {"command": "mkdir generated"},
    })
    directory_effect_script.write_text(json.dumps(directory_effect_data))
    child_environment_helper = tmp_path / "read_child_environment.py"
    child_environment_helper.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from stateframe import profiles\n"
        "from stateframe.endpoint import Endpoint\n"
        "from stateframe.runner import RunOptions, _child_env\n"
        "options = RunOptions(Path(sys.argv[1]), 'Fix the pricing code so the tests pass.',\n"
        "    Endpoint.from_args(sys.argv[2], 'scripted', None), max_turns=16, profile=sys.argv[3],\n"
        ")\n"
        "child_env = _child_env(options)\n"
        "seed = profiles.load(options.profile).get('PYTHONHASHSEED')\n"
        "if seed is not None: child_env['PYTHONHASHSEED'] = str(seed)\n"
        "print(json.dumps(child_env, sort_keys=True))\n"
    )
    release_venv = tmp_path / "main-base-venv"
    created = subprocess.run([sys.executable, "-m", "venv", str(release_venv)],
                             capture_output=True, text=True, timeout=_scaled_test_timeout(60), check=False)
    assert created.returncode == 0, created.stdout + created.stderr
    python = release_venv / "bin" / "python"
    requirements = old / "requirements.txt"
    _install_release_package(python, old, requirements)

    from stateframe.scripted_endpoint import Script, serve

    server = serve(Script({}))
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
    script_options = {"endpoint": endpoint, "server": server, "profile": off_profile}
    try:
        old_a, endpoint_a = _run_scripted_engine(
            python, script, repo, tmp_path / "old-run-a", parallel_off=False, **script_options)
        old_b, endpoint_b = _run_scripted_engine(
            python, script, repo, tmp_path / "old-run-b", parallel_off=False, **script_options)
        old_effect_run, _ = _run_scripted_engine(
            python, directory_effect_script, repo, tmp_path / "old-directory-effect",
            parallel_off=False, allow_incomplete=True, **script_options)
        old_effect = _directory_effect_record(old_effect_run)
        old_child_env_a = _read_child_environment(python, child_environment_helper, repo, endpoint_a,
                                                  profile=off_profile)
        old_child_env_b = _read_child_environment(python, child_environment_helper, repo, endpoint_b,
                                                  profile=off_profile)

        _install_release_package(python, root, requirements)
        # Each build supplies its own shipped settings. The shared path is the
        # actual CLI input, so the recorded profile identity must also match.
        assert str(_write_off_cache_profile(tmp_path)) == off_profile
        new_run, new_endpoint = _run_scripted_engine(
            python, script, repo, tmp_path / "new-run", parallel_off=True, **script_options)
        _assert_wire_parity(old_a, endpoint_a, old_b, endpoint_b)
        _assert_wire_parity(old_a, endpoint_a, new_run, new_endpoint)
        _assert_wire_parity(old_b, endpoint_b, new_run, new_endpoint)
        new_effect_run, _ = _run_scripted_engine(
            python, directory_effect_script, repo, tmp_path / "new-directory-effect",
            parallel_off=True, allow_incomplete=True, **script_options)
        new_effect = _directory_effect_record(new_effect_run)
        _assert_directory_effect_values_match(old_effect, new_effect)
        planted_effect = {**new_effect, "observation": str(new_effect["observation"]).replace(
            "created: generated", "created: planted")}
        with pytest.raises(AssertionError, match="directory effect"):
            _assert_directory_effect_values_match(old_effect, planted_effect)
        new_child_env = _read_child_environment(python, child_environment_helper, repo, new_endpoint,
                                                profile=off_profile)
    finally:
        server.shutdown()
        server.server_close()

    old_spec_a = json.loads((old_a / "spec.json").read_bytes())
    old_spec_b = json.loads((old_b / "spec.json").read_bytes())
    new_spec = json.loads((new_run / "spec.json").read_bytes())
    assert old_spec_a["profile"] == old_spec_b["profile"] == new_spec["profile"] == off_profile
    _assert_default_off_spec_parity(old_spec_a, old_spec_b, old_a, endpoint_a, old_b, endpoint_b)
    assert old_spec_a["environment_sources"] == new_spec["environment_sources"], (
        "default-off run spec values differ at environment_sources")
    released_build = _expected_build_metadata(old)
    candidate_build = _expected_build_metadata(root)
    assert old_spec_a["engine_source"] == released_build
    expected_spec = {**old_spec_a, "engine_source": candidate_build}
    _assert_default_off_spec_parity(expected_spec, new_spec, old_a, endpoint_a, new_run, new_endpoint)

    planted_extra = {**new_spec, "planted_unexpected_spec_key": "must fail parity"}
    with pytest.raises(AssertionError, match="candidate added unexpected run-spec keys"):
        _assert_default_off_spec_parity(old_spec_a, planted_extra, old_a, endpoint_a, new_run, new_endpoint)

    assert old_spec_a["environment"] == old_spec_b["environment"] == new_spec["environment"]
    assert old_child_env_a == old_child_env_b == new_child_env
    assert old_child_env_a["PYTHONHASHSEED"] == "0"

    old_summary_a = json.loads((old_a / "summary.json").read_bytes())
    old_summary_b = json.loads((old_b / "summary.json").read_bytes())
    new_summary = json.loads((new_run / "summary.json").read_bytes())
    assert "parallel" not in old_summary_a and "parallel" not in old_summary_b
    assert "parallel" not in new_summary
    _assert_default_off_timed_mapping_parity(
        old_summary_a, old_summary_b, "wall_seconds", "reference summaries")
    assert old_summary_a["engine_source_hash"] == released_build["source_hash"]
    assert old_summary_a["engine_shipped_hash"] == released_build["shipped_hash"]
    expected_summary = {**old_summary_a, "engine_source_hash": candidate_build["source_hash"],
                        "engine_shipped_hash": candidate_build["shipped_hash"]}
    _assert_default_off_timed_mapping_parity(
        expected_summary, new_summary, "wall_seconds", "summaries",
        candidate_only_keys=_DEFAULT_OFF_CANDIDATE_ONLY_SUMMARY_KEYS)

    for released_run, released_endpoint in ((old_a, endpoint_a), (old_b, endpoint_b)):
        released_result_bytes = (released_run / "engine-result.json").read_bytes()
        candidate_result_bytes = (new_run / "engine-result.json").read_bytes()
        released_result = json.loads(released_result_bytes)
        candidate_result = json.loads(candidate_result_bytes)
        _assert_default_off_result_parity(released_result, candidate_result, released_run, released_endpoint,
                                          new_run, new_endpoint)
        _assert_default_off_result_bytes_parity(released_result_bytes, candidate_result_bytes)
