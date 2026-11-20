"""The internal forced manifest enters through the parent's parallel tool route."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from barrier_endpoint import BarrierEndpoint
from kernel_parallel_records import default_subframe_token_cap
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET
from e2e_support import (
    copy_example,
    parallel_call_roots,
    parent_wire_files,
    read_json,
    run_cli,
    subframe_directories,
    subframe_wire_files,
    tool_step,
)


pytestmark = pytest.mark.host_sandbox


def _unit(unit_id: str, *, owns: list[str], creates: list[str] | None = None) -> dict:
    return {
        "id": unit_id,
        "goal": f"Update {unit_id}",
        "owns": owns,
        "creates": creates or [],
        "reads": [],
        "exports": [],
        "consumes": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _invalid_cases() -> tuple[tuple[str, dict, str], ...]:
    return (
        ("overlap", {"subframes": [
            _unit("first", owns=["a.py"]),
            _unit("second", owns=["a.py"]),
        ]}, "OWNERSHIP_OVERLAP"),
        ("missing_owned", {"subframes": [
            _unit("first", owns=["missing.py"]),
            _unit("second", owns=["b.py"]),
        ]}, "UNSAFE_PATH"),
        ("existing_create", {"subframes": [
            _unit("first", owns=[], creates=["a.py"]),
            _unit("second", owns=["b.py"]),
        ]}, "CREATE_EXISTS"),
    )


def _repo(root: Path) -> Path:
    root.mkdir()
    (root / "a.py").write_text("VALUE = 1\n")
    (root / "b.py").write_text("VALUE = 2\n")
    (root / "tests").mkdir()
    (root / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    return root


def _admission_reason(run_dir: Path) -> str:
    roots = parallel_call_roots(run_dir)
    assert len(roots) == 1
    return _admission_reason_from_root(roots[0])


def _admission_reason_from_root(root: Path) -> str:
    result = read_json(root / "result.json")
    assert result["admission"]["status"] == "REFUSED"
    return result["admission"]["refusal"]["reason"]


def test_forced_manifest_is_first_action_and_matches_model_admission(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    cases = _invalid_cases()
    model_repo = _repo(tmp_path / "model-repo")
    model_run = tmp_path / "model-run"
    model_steps = [tool_step("parallel", args) for _case, args, _reason in cases]
    model_steps.append(tool_step("done", {"summary": "The invalid plans were refused."}))
    with BarrierEndpoint(model_steps, {}) as endpoint:
        model_result = run_cli(
            isolated_engine_python, endpoint.url, model_repo, model_run,
            "Record why the subframe plan is invalid.", parallel="2")
        assert model_result.returncode == 0, model_result.stdout + model_result.stderr
        endpoint.assert_consumed()

    model_roots = parallel_call_roots(model_run)
    assert len(model_roots) == len(cases)
    model_reasons = [_admission_reason_from_root(root) for root in model_roots]
    expected_reasons = {expected_reason for _case, _args, expected_reason in cases}
    assert set(model_reasons) == expected_reasons

    for case, args, expected_reason in cases:
        forced_repo = _repo(tmp_path / f"{case}-forced-repo")
        manifest_path = tmp_path / f"{case}.json"
        manifest_bytes = json.dumps(args, separators=(",", ":"))
        manifest_path.write_text(manifest_bytes)
        forced_run = tmp_path / f"{case}-forced-run"
        forced_steps = [tool_step("done", {"summary": "The forced plan was refused."})]
        with BarrierEndpoint(forced_steps, {}) as endpoint:
            forced_result = run_cli(
                isolated_engine_python, endpoint.url, forced_repo, forced_run,
                "Record why the subframe plan is invalid.", parallel="2",
                extra_environment={"STATEFRAME_PARALLEL_FORCE_MANIFEST": str(manifest_path)})
            assert forced_result.returncode == 0, forced_result.stdout + forced_result.stderr
            endpoint.assert_consumed()

            # The first actual provider request sees the synthetic call and its disposition.
            assert endpoint.requests
            first_wire = read_json(parent_wire_files(forced_run)[0])
            messages = first_wire["messages"]
            calls = [call for message in messages if message.get("role") == "assistant"
                     for call in message.get("tool_calls", [])
                     if call.get("function", {}).get("name") == "parallel"]
            assert len(calls) == 1
            assert json.loads(calls[0]["function"]["arguments"]) == args
            refusal = next(message["content"] for message in messages
                           if message.get("role") == "tool"
                           and message.get("tool_call_id") == calls[0]["id"])
            assert expected_reason in refusal
            assert manifest_bytes

        assert _admission_reason(forced_run) == expected_reason
        assert expected_reason in model_reasons


def test_force_manifest_switch_is_ignored_without_parallel(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = _repo(tmp_path / "off-repo")
    missing_manifest = tmp_path / "does-not-exist.json"
    run_dir = tmp_path / "off-run"
    with BarrierEndpoint([tool_step("done", {"summary": "No fan-out ran."})], {}) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Finish without subframes.", parallel="off",
            extra_environment={"STATEFRAME_PARALLEL_FORCE_MANIFEST": str(missing_manifest)})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    wire = read_json(parent_wire_files(run_dir)[0])
    assert all(tool.get("function", {}).get("name") != "parallel"
               for tool in wire["tools"])
    assert not (run_dir / "parallel").exists()
    assert "STATEFRAME_PARALLEL_FORCE_MANIFEST" not in read_json(run_dir / "spec.json")["environment"]


def _compact_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def _first_stateframe(request_path: Path) -> tuple[dict[str, object], str]:
    request = read_json(request_path)
    messages = request.get("messages")
    assert isinstance(messages, list), f"request has no messages: {request_path}"
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.startswith("[StateFrame]\n"):
            raw = content[len("[StateFrame]\n"):].split("\n[", 1)[0]
            frame = json.loads(raw)
            assert isinstance(frame, dict), f"StateFrame is not an object: {request_path}"
            return frame, raw
    raise AssertionError(f"request has no StateFrame: {request_path}")


def _largest_frame_branches(frame: dict[str, object]) -> list[tuple[str, int]]:
    branches: list[tuple[str, int]] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}"
                branches.append((child_path, _compact_bytes(item)))
                visit(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    child_path = f"{path}[{index}]"
                    branches.append((child_path, _compact_bytes(item)))
                    visit(item, child_path)

    visit(frame, "frame")
    return sorted(branches, key=lambda row: (-row[1], row[0]))[:5]


def test_forced_manifest_is_parent_only_and_child_state_is_unit_scoped(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = json.loads((root / "examples" / "parallel_fixes.script.json").read_text())
    parent_plan = next(step["arguments"] for step in script["parent"]
                       if step.get("tool") == "parallel")
    repo = copy_example(tmp_path)
    run_dir = tmp_path / "run"
    manifest = tmp_path / "forced-plan.json"
    manifest.write_text(json.dumps(parent_plan, separators=(",", ":")))
    assert json.loads(manifest.read_text()) == parent_plan

    with BarrierEndpoint(script["parent"][1:], script["subframes"]) as endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Complete the independent adapter changes and verify each unit.",
            parallel="2", timeout=600,
            extra_environment={"STATEFRAME_PARALLEL_FORCE_MANIFEST": str(manifest)},
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        endpoint.assert_consumed()

    parent_requests = parent_wire_files(run_dir)
    assert parent_requests
    parent_request = read_json(parent_requests[0])
    parent_messages = parent_request.get("messages")
    assert isinstance(parent_messages, list)
    parent_calls = [call for message in parent_messages if isinstance(message, dict)
                    for call in message.get("tool_calls", [])
                    if isinstance(call, dict)
                    and call.get("function", {}).get("name") == "parallel"]
    assert len(parent_calls) == 1
    parent_call_id = parent_calls[0].get("id")

    children = subframe_directories(run_dir)
    assert len(children) >= 2
    owned_by_unit: dict[str, set[str]] = {}
    for unit_id, child_dir in children.items():
        spec = read_json(child_dir / "unit-spec.json")
        contract = spec.get("contract")
        assert isinstance(contract, dict)
        owned = contract.get("owns")
        assert isinstance(owned, list) and all(isinstance(path, str) for path in owned)
        owned_by_unit[unit_id] = set(owned)

    measurements: list[dict[str, object]] = []
    checks: list[tuple[str, dict[str, object], dict[str, object], str]] = []
    for unit_id, child_dir in sorted(children.items()):
        request_path = subframe_wire_files(child_dir)[0]
        request = read_json(request_path)
        spec = read_json(child_dir / "unit-spec.json")
        contract = spec["contract"]
        frame, frame_text = _first_stateframe(request_path)
        frame_bytes = len(frame_text.encode("utf-8"))
        calibration = spec.get("subframe_cap_calibration")
        ratio = (float(calibration["tokens_per_byte"])
                 if isinstance(calibration, dict) else 0.3193)
        residual = int(calibration.get("residual_tokens", 0)) if isinstance(calibration, dict) else 0
        request_bytes = _compact_bytes(request)
        request_tokens = math.ceil(request_bytes * ratio + residual)
        top_key_bytes = {key: _compact_bytes({key: value}) for key, value in frame.items()}
        metrics = {
            "unit": unit_id,
            "frame_bytes": frame_bytes,
            "frame_estimated_tokens": math.ceil(frame_bytes * ratio + residual),
            "request_bytes": request_bytes,
            "request_estimated_tokens": request_tokens,
            "top_key_bytes": top_key_bytes,
            "largest_nested": _largest_frame_branches(frame),
        }
        measurements.append(metrics)
        messages = request.get("messages")
        assert isinstance(messages, list)
        parallel_calls = [call for message in messages if isinstance(message, dict)
                          for call in message.get("tool_calls", [])
                          if isinstance(call, dict)
                          and call.get("function", {}).get("name") == "parallel"]
        parallel_rejections = [str(message.get("content")) for message in messages
                               if isinstance(message, dict)
                               and "REJECTED" in str(message.get("content", ""))
                               and "parallel" in str(message.get("content", ""))]
        state = frame.get("state")
        assert isinstance(state, dict)
        framefit5 = state.get("framefit5")
        assert isinstance(framefit5, dict)
        checks.append((unit_id, spec, framefit5, frame_text))

    print("CHILD_FRAME_METRICS " + json.dumps(measurements, sort_keys=True))

    for unit_id, spec, framefit5, frame_text in checks:
        messages = read_json(subframe_wire_files(children[unit_id])[0]).get("messages")
        assert isinstance(messages, list)
        parallel_calls = [call for message in messages if isinstance(message, dict)
                          for call in message.get("tool_calls", [])
                          if isinstance(call, dict)
                          and call.get("function", {}).get("name") == "parallel"]
        parallel_rejections = [str(message.get("content")) for message in messages
                               if isinstance(message, dict)
                               and "REJECTED" in str(message.get("content", ""))
                               and "parallel" in str(message.get("content", ""))]
        assert not parallel_calls, (unit_id, parallel_calls)
        assert not parallel_rejections, (unit_id, parallel_rejections)

        own_paths = owned_by_unit[unit_id]
        other_paths = set().union(*(paths for key, paths in owned_by_unit.items()
                                    if key != unit_id))
        assert all(path in frame_text for path in own_paths), (unit_id, own_paths)
        assert not any(path in frame_text for path in other_paths), (unit_id, other_paths)
        assert not isinstance(parent_call_id, str) or parent_call_id not in frame_text
        scope = framefit5.get("subframe_scope")
        assert isinstance(scope, dict), (unit_id, framefit5)
        assert scope.get("unit_id") == spec.get("unit_id")
        assert scope.get("requirements") == spec.get("requirement_clauses")
        contract = spec.get("contract")
        scope_contract = scope.get("contract")
        assert isinstance(contract, dict) and isinstance(scope_contract, dict)
        assert scope_contract.get("owns") == contract.get("owns")
        check = contract.get("check")
        scoped_check = scope_contract.get("check")
        assert isinstance(check, dict) and isinstance(scoped_check, dict)
        assert scoped_check.get("selectors") == check.get("selectors")
        assert "latest_parallel_disposition" not in framefit5.get("action_ledger", {})
        metric = next(row for row in measurements if row["unit"] == unit_id)
        assert metric["request_estimated_tokens"] <= default_subframe_token_cap(
            ENDPOINT_PROMPT_TOKEN_BUDGET), (unit_id, metric)
