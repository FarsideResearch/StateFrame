"""The latest subframe disposition stays in the next parent request under pressure."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import kernel_parallel
from kernel_parallel_records import (
    AdmissionResult,
    CheckRecipe,
    FileDelta,
    ParallelLimits,
    ParallelMeasured,
    ParallelRequest,
    ParallelResult,
    Refusal,
    UnitContract,
    UnitOutcome,
    UnitUsage,
)
from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_capacity import fit_protected_frame, protected_bytes
from kernel_stateframe_framefit5 import (
    FrameFit5Error,
    STATE_HEADER,
    _serialize_state_message,
    _state_payload,
    _validated_parallel_disposition,
    assemble_framefit5_request,
    build_typed_state,
)


UNIT_IDS = ("invoicing", "pricing", "shipping")
OLD_ACTIONS = 8


def _units(*, max_units: int = 3, creates_first: bool = False) -> tuple[UnitContract, ...]:
    names = (UNIT_IDS if max_units <= len(UNIT_IDS) else
             tuple(f"u{index:04d}" for index in range(max_units)))
    return tuple(UnitContract(
        unit_id,
        f"Repair {unit_id}",
        (f"shop/{unit_id}.py",),
        (f"shop/{unit_id}.py",) if creates_first and index == 0 else (),
        (), (), (), (),
        CheckRecipe((f"tests/test_{unit_id}.py",),
                    (f"tests/test_{unit_id}.py",), "pytest"),
    ) for index, unit_id in enumerate(names))


def _parallel_record(tmp_path, scenario: str, *, max_units: int = 3,
                     timing_scale: float = 1.0):
    call_id = "call_0"
    units = _units(max_units=max_units, creates_first=scenario == "admission_refused")
    limits = ParallelLimits(max_units=max(6, len(units)))
    refusal = (Refusal(
        "CREATE_EXISTS", "shop/invoicing.py already exists", ("shop/invoicing.py",), (), (),
        "declare the existing file under owns",
    ) if scenario == "admission_refused" else None)
    admission = AdmissionResult(
        "REFUSED" if refusal else "ADMITTED", refusal, (), (),
        tuple(unit.id for unit in units), 0, "manifest", limits, units=units,
    )

    statuses = {
        "partial": ("COMMITTED", "COMMITTED", "REFUSED"),
        "verified": ("COMMITTED", "COMMITTED", "COMMITTED"),
        "engine_error": ("COMMITTED", "ENGINE_ERROR", "COMMITTED"),
        "max_units": ("COMMITTED",) * len(units),
    }.get(scenario, ())
    outcomes = []
    for index, (unit, status) in enumerate(zip(units, statuses), 1):
        failed = status not in {"COMMITTED", "COMMITTED_UNVERIFIED"}
        reason = (
            "SUBFRAME_CONTEXT_TOO_LARGE" if scenario == "partial" and unit.id == "shipping"
            else "ENGINE_ERROR" if status == "ENGINE_ERROR" else None
        )
        detail = (
            "first_frame_tokens=35136; cap=33344"
            if reason == "SUBFRAME_CONTEXT_TOO_LARGE" else
            "child process exited before returning a terminal result"
            if reason == "ENGINE_ERROR" else None
        )
        changed_path = (f"shop/{unit.id}/" + "x" * 100 + ".py"
                        if scenario == "max_units" else f"shop/{unit.id}.py")
        files = (() if failed else
                 (FileDelta(changed_path, 2, 1, "modify"),))
        usage = UnitUsage(2 if reason == "ENGINE_ERROR" else 1, 60, 8, 1, 100)
        outcomes.append(UnitOutcome(
            unit.id, status, reason, detail,
            index if status == "COMMITTED" else None,
            files, None if failed else True,
            "NOT_RUN" if failed else "PASS", usage, 0.25 * timing_scale, 0.03 * timing_scale,
        ))
    outcomes = tuple(outcomes)
    final_check = {
        "partial": "FAIL",
        "verified": "PASS",
        "engine_error": "FAIL",
        "admission_refused": "NOT_RUN",
        "max_units": "PASS",
    }[scenario]
    integration = {
        "partial": "FAILED",
        "verified": "VERIFIED",
        "engine_error": "FAILED",
        "admission_refused": "UNVERIFIED",
        "max_units": "VERIFIED",
    }[scenario]
    measured = ParallelMeasured(1.25 * timing_scale, 3.5 * timing_scale,
                                2.8 / timing_scale, 3 if timing_scale == 1.0 else 1,
                                0.1 * timing_scale)
    request = ParallelRequest(call_id, units, str(min(len(units), 3)))
    observation = kernel_parallel._observation(
        admission, outcomes, final_check, integration, measured, request,
    )
    result = ParallelResult(
        call_id, admission, outcomes,
        tuple(item.txn_id for item in outcomes if item.txn_id is not None),
        final_check, (), integration, observation, measured,
    )
    call_root = tmp_path / "parallel" / call_id
    call_root.mkdir(parents=True)
    (call_root / "result.json").write_text(json.dumps(result.to_json()))
    (call_root / "intent.json").write_text(json.dumps({
        "schema": "stateframe-parallel-intent/1",
        "call_id": call_id,
        "request": request.to_json(),
        "admission": admission.to_json(),
    }))
    arguments = {
        "subframes": [{
            "id": unit.id,
            "owns": list(unit.owns),
            "creates": list(unit.creates),
            "check": {"tests": list(unit.check.selectors)},
        } for unit in units],
    }
    assistant = {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id,
        "type": "function",
        "function": {"name": "parallel", "arguments": json.dumps(arguments)},
    }]}
    reply = {"role": "tool", "tool_call_id": call_id, "content": observation}
    return result, request, (assistant, reply)


def _old_tool_turn(index: int) -> tuple[dict[str, object], dict[str, object]]:
    call_id = f"old_{index}"
    assistant = {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"index": index})},
    }]}
    reply = {"role": "tool", "tool_call_id": call_id,
             "content": f"historical lookup result {index}"}
    return assistant, reply


def _build_next_request(tmp_path, monkeypatch, scenario: str, *, timing_scale: float = 1.0):
    task = {"role": "user", "content": "Task: repair the subframe changes."}
    messages: list[dict[str, object]] = [task]
    for index in range(OLD_ACTIONS):
        messages.extend(_old_tool_turn(index))

    if scenario == "no_call":
        call_id = "shell_0"
        assistant = {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "shell", "arguments": json.dumps({"command": "true"})},
        }]}
        latest = (assistant, {"role": "tool", "tool_call_id": call_id, "content": "exit 0"})
    elif scenario == "parallel_disabled":
        call_id = "parallel_disabled_0"
        assistant = {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "parallel", "arguments": json.dumps({"subframes": []})},
        }]}
        latest = (assistant, {
            "role": "tool", "tool_call_id": call_id,
            "content": "REFUSED RESOURCES_UNAVAILABLE: subframe execution is disabled for this run",
        })
        monkeypatch.setattr(kernel_parallel, "_service", None)
    else:
        _result, _request, latest = _parallel_record(tmp_path, scenario, timing_scale=timing_scale)
        monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "3"}))
    messages.extend(latest)

    context = build_action_context(messages, workspace_root=tmp_path)
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={}),
        binding=lambda **_kwargs: {"product_revision": "sha256:" + "a" * 64},
        edit_contract={"edit_authority": {"editable_paths": []}},
        protected_tests={},
        frame_ceiling=100_000,
        micrograph_wire=None,
    )
    frame = build_typed_state(messages, context, runtime)
    prepared = [
        {"role": "system", "content": "StateFrame test rules."},
        task,
        {"role": "user", "content": _serialize_state_message(
            {"state": {"framefit5": frame}})},
    ]
    request_messages = assemble_framefit5_request(
        prepared, messages, live_frame=frame, runtime=runtime,
    )
    body: dict[str, object] = {"messages": request_messages, "tools": []}
    protected_messages = [task, *latest]
    before = protected_bytes(body, protected_messages)
    audit = fit_protected_frame(
        body,
        semantic_messages=messages,
        action_context=context,
        protected_messages=protected_messages,
        max_request_bytes=max(1, before - 1),
    )
    return body, context, audit, messages


def _frame_from_request(body: dict[str, object]) -> dict[str, object]:
    for message in body["messages"]:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.startswith(STATE_HEADER + "\n"):
            return _state_payload(message)[1]["framefit5"]
    raise AssertionError("next request has no FrameFit5 state")


@pytest.mark.parametrize("scenario", ["partial", "verified", "engine_error"])
def test_fresh_next_request_bytes_ignore_elapsed_timing(tmp_path, monkeypatch, scenario):
    bodies = []
    records = []
    for name, scale in (("fast", 1.0), ("slow", 100.0)):
        run = tmp_path / name
        body, _context, _audit, _messages = _build_next_request(
            run, monkeypatch, scenario, timing_scale=scale,
        )
        bodies.append(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
        records.append(json.loads((run / "parallel" / "call_0" / "result.json").read_text()))
    assert records[0]["measured"] != records[1]["measured"]
    assert records[0]["outcomes"][0]["wall_seconds"] != records[1]["outcomes"][0]["wall_seconds"]
    assert records[0]["outcomes"][0]["queue_wait_seconds"] != records[1]["outcomes"][0]["queue_wait_seconds"]
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("scenario", ["partial", "verified", "engine_error"])
def test_parallel_observation_ignores_elapsed_timing(tmp_path, scenario):
    fast, _request, _turns = _parallel_record(tmp_path / "fast", scenario)
    slow, _request, _turns = _parallel_record(tmp_path / "slow", scenario, timing_scale=100.0)
    assert fast.measured != slow.measured
    assert fast.observation == slow.observation
    assert "Measured:" not in fast.observation


@pytest.mark.parametrize("mutation", ["fields", "row"])
def test_disposition_rejects_wire_timing_columns(tmp_path, mutation):
    result, request, _turns = _parallel_record(tmp_path, "verified")
    item = kernel_parallel.render_parallel_disposition(result, request)
    section = item["sections"]["disposition"]
    assert section["unit_details_fields"] == [
        "id", "reason", "changed_files", "contract_ok", "check_status", "requests",
    ]
    _validated_parallel_disposition(item)
    if mutation == "fields":
        section["unit_details_fields"].extend(["wall_seconds", "queue_wait_seconds"])
    else:
        section["unit_details"][0].extend([0.25, 0.03])
    with pytest.raises(FrameFit5Error, match="parallel disposition"):
        _validated_parallel_disposition(item)


@pytest.mark.parametrize("scenario", ["partial", "verified", "admission_refused", "engine_error"])
def test_parallel_disposition_survives_capacity_pressure(tmp_path, monkeypatch, scenario):
    body, _context, audit, _messages = _build_next_request(tmp_path, monkeypatch, scenario)
    assert audit["before"] > audit["limit"]
    assert audit["after"] <= audit["limit"]
    ledger = _frame_from_request(body)["action_ledger"]
    disposition = ledger.get("latest_parallel_disposition")
    assert isinstance(disposition, dict), "latest fan-out result was folded out of the next request"
    assert disposition["kind"] == "parallel_disposition"
    summary = disposition["sections"]["disposition"]
    assert summary["result_summary"]
    assert summary["final_check"]
    assert summary["integration"]
    rows = {row[0]: row[1] for row in summary["units"]}
    details = {row[0]: row[1:] for row in summary["unit_details"]}
    if scenario == "partial":
        assert {unit_id: rows[unit_id] for unit_id in UNIT_IDS} == {
            "invoicing": "COMMITTED", "pricing": "COMMITTED", "shipping": "REFUSED",
        }
        assert "SUBFRAME_CONTEXT_TOO_LARGE" in details["shipping"][0]
        assert details["invoicing"][1] == [["shop/invoicing.py", "modify", 2, 1]]
        assert details["invoicing"][2:5] == [True, "PASS", 1]
        assert summary["final_check"] == "FAIL"
        assert summary["integration"] == "FAILED"
    elif scenario == "verified":
        assert all(rows[unit_id] == "COMMITTED" for unit_id in UNIT_IDS)
        assert all(details[unit_id][3] == "PASS" for unit_id in UNIT_IDS)
        assert summary["final_check"] == "PASS"
        assert summary["integration"] == "VERIFIED"
    elif scenario == "admission_refused":
        assert "CREATE_EXISTS" in summary["result_summary"]
        assert summary["admission"]["reason"] == "CREATE_EXISTS"
        assert all(rows[unit_id] == "NOT_STARTED" for unit_id in UNIT_IDS)
    else:
        assert rows["pricing"] == "ENGINE_ERROR"
        assert details["pricing"][0].startswith("ENGINE_ERROR:")
        assert "child process exited" in details["pricing"][0]


def test_no_parallel_call_keeps_pressure_fitted_wire_bytes_unchanged(tmp_path, monkeypatch):
    body, _context, audit, _messages = _build_next_request(tmp_path, monkeypatch, "no_call")
    assert audit["before"] > audit["limit"]
    assert audit["after"] <= audit["limit"]
    assert "latest_parallel_disposition" not in _frame_from_request(body)["action_ledger"]
    actual = hashlib.sha256(json.dumps(
        body["messages"], ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert actual == "408c49be0080a620580f431de945079e16f9b86745d3511f5277bc495f746f05"


def test_disabled_parallel_sentinel_is_not_classified_as_a_fanout_under_pressure(
        tmp_path, monkeypatch):
    body, _context, audit, _messages = _build_next_request(
        tmp_path, monkeypatch, "parallel_disabled",
    )
    assert audit["before"] > audit["limit"]
    assert audit["after"] <= audit["limit"]
    ledger = _frame_from_request(body)["action_ledger"]
    assert "latest_parallel_disposition" not in ledger
    assert not any(
        row.get("kind") == "parallel_disposition"
        for row in ledger.get("unclassified_actions", []) if isinstance(row, dict)
    )


def test_max_units_disposition_is_bounded_and_truncation_is_deterministic(tmp_path, monkeypatch):
    units = tuple(f"u{index:04d}" for index in range(1024))
    result, request, _turns = _parallel_record(tmp_path, "max_units", max_units=1024)
    monkeypatch.setattr(kernel_parallel, "_service", (tmp_path, {"parallel_mode": "3"}))
    item = kernel_parallel.render_parallel_disposition(result, request)
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= kernel_parallel.MAX_PARALLEL_DISPOSITION_BYTES
    assert item["truncation"]["truncated"] is True
    assert item["truncation"]["omitted_units"] == 0
    assert item["truncation"]["omitted_details"] > 0
    assert item == kernel_parallel.render_parallel_disposition(result, request)
    visible = item["sections"]["disposition"]["units"]
    assert len(visible) == 1024
    assert visible[0][0] == units[0]


def test_package_disposition_does_not_require_bare_wire_identity(tmp_path, monkeypatch):
    from stateframe.engine import kernel_parallel as package_parallel

    result, request, _turns = _parallel_record(tmp_path, "verified")
    expected = kernel_parallel.render_parallel_disposition(result, request)
    monkeypatch.setitem(sys.modules, "kernel_wire_identity", None)
    assert package_parallel.render_parallel_disposition(result, request) == expected


def test_disposition_sections_accept_a_future_continuation_section(tmp_path):
    result, request, _turns = _parallel_record(tmp_path, "verified")
    item = kernel_parallel.render_parallel_disposition(result, request)
    item["sections"]["fanout_continuation"] = {
        "schema": "stateframe-fanout-continuation/1",
        "summary": "follow-up work remains",
    }
    validated = _validated_parallel_disposition(item)
    assert validated["sections"]["fanout_continuation"] == item["sections"]["fanout_continuation"]
