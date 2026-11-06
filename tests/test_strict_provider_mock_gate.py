"""A DeepInfra-shaped provider must admit a multi-send source-window run."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from kernel_completion_owner import (
    RequestActionContractError, assert_request_action_contract,
    establish_request_action_contract,
)
from kernel_measurement import RequestCapture
from kernel_read_fixes import ContinuationLimits
from kernel_request_reprepare import SourceRevisionReplanRefused, source_replan_preparation
from tests.test_request_seal_wire_form import _run_scripted, assert_sealed_wire


@pytest.mark.host_sandbox
def test_strict_provider_six_sends_seal_provider_bytes(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": path}}
        for path in ("app.py", "tests/test_value.py", "app.py", "tests/test_value.py")
    ] + [{"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
         {"tool": "done", "arguments": {"summary": "VALUE is seven and tests pass."}}]
    process, run, script = _run_scripted(tmp_path, strict=True, steps=steps)
    output = process.stdout + process.stderr
    assert len(script.requests) >= 6, output
    assert (run / "tagged-source-window").is_file(), output
    assert "provider dialect: openai-strict" in output
    assert "measurement body/projection/provider identity changed" not in output
    assert process.returncode == 0, output
    script.assert_consumed()
    assert not script.strict_openai_errors
    assert_sealed_wire(run, minimum=6)


@pytest.mark.host_sandbox
def test_strict_provider_charged_source_replan_sends_replacement(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "done", "arguments": {"summary": "VALUE is seven."}},
    ]
    process, run, script = _run_scripted(
        tmp_path, strict=True, steps=steps, inject_allocation_replan=True,
    )
    output = process.stdout + process.stderr
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += traceback.read_text(encoding="utf-8")
    assert (run / "injected-replan").read_text() == "1", output
    assert process.returncode == 0, output
    events = measurement_events(run)
    started = [row for row in events if row.get("kind") == "source_revision_replan_started"]
    adopted = [row for row in events if row.get("kind") == "source_revision_reprepared"]
    assert len(started) == len(adopted) == 1, output
    original_id = started[0]["request_id"]
    replacement_id = adopted[0]["replacement_request_id"]
    assert replacement_id != original_id
    submitted = [row for row in events if row.get("kind") == "request_submitted"]
    assert sum(row.get("request_id") == replacement_id for row in submitted) == 1
    assert all(row.get("request_id") != original_id for row in submitted)
    assert sum(row.get("kind") == "budget_unit_charged" and row.get("step") == 0
               for row in events) == 1
    assert not any(row.get("kind") == "source_revision_replan_refused" for row in events)
    assert len(script.requests) >= len(steps), output
    script.assert_consumed()
    assert_sealed_wire(run, minimum=len(steps))


def test_charged_replan_contract_requires_distinct_request_identity() -> None:
    class Journal:
        task_id = "charged-replan-contract"

        def emit(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request_records(self, _request_id: str) -> list[object]:
            return []

    journal = Journal()
    runtime = SimpleNamespace(journal=journal, owner=SimpleNamespace(root="."),
                              system_outcome=None)
    budget = ContinuationLimits(32, mode="response", output_reservation_tokens=24000)
    runtime.continuation_limits = budget
    tools: list[dict[str, object]] = [{"type": "function", "function": {
        "name": "read", "description": "read",
        "parameters": {"type": "object", "properties": {}},
    }}]
    original = RequestCapture(journal, 1, "snapshot")
    assert original.request_id == RequestCapture(
        journal, 1, "snapshot",
        identity_content={"task_id": journal.task_id, "purpose": ""},
    ).request_id
    original_messages = [{"role": "user", "content": budget.render_continuation_limits()}]
    original_contract = establish_request_action_contract(
        runtime, original_messages, tools, request_id=original.request_id,
        preserve_input_surface=True, framefit5_preassembly=True,
    )
    assert budget.charge_response(step=1)
    assert budget.continuation_model_decisions_used == 1
    replacement_messages = [{"role": "user", "content": budget.render_continuation_limits()}]
    stale = establish_request_action_contract(
        runtime, replacement_messages, tools, request_id=original.request_id,
        preserve_input_surface=True, framefit5_preassembly=True,
    )
    assert stale["budget_wire_digest"] == original_contract["budget_wire_digest"]
    with pytest.raises(RequestActionContractError, match="serialized continuation limits were removed or changed"):
        assert_request_action_contract(
            None, replacement_messages, stale["admitted_tool_schemas"],
            request_id=original.request_id, frozen_contract=stale,
            framefit5_preassembly=True,
        )
    replacement = RequestCapture(journal, 1, "snapshot", identity_content={
        "task_id": journal.task_id, "purpose": "",
        "source_replan_of": original.request_id,
    })
    assert replacement.request_id != original.request_id
    original._source_replan_attempted = True
    with pytest.raises(SourceRevisionReplanRefused, match="source_revision_replan_exhausted"):
        original._recover_source_revision_replan(
            RuntimeError("allocation ledger authority changed"),
            SimpleNamespace(runtime=runtime),
        )
    fresh = establish_request_action_contract(
        runtime, replacement_messages, tools, request_id=replacement.request_id,
        preserve_input_surface=True, framefit5_preassembly=True,
    )
    assert fresh["budget_wire_digest"] != original_contract["budget_wire_digest"]
    assert_request_action_contract(
        None, replacement_messages, fresh["admitted_tool_schemas"],
        request_id=replacement.request_id, frozen_contract=fresh,
        framefit5_preassembly=True,
    )
    for changed in ([], [{"role": "user", "content": budget.render_continuation_limits() + " changed"}]):
        with pytest.raises(RequestActionContractError, match="serialized continuation limits were removed or changed"):
            assert_request_action_contract(
                None, changed, fresh["admitted_tool_schemas"],
                request_id=replacement.request_id, frozen_contract=fresh,
                framefit5_preassembly=True,
            )


def test_source_replan_origin_context_restores_prior_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kernel_request_reprepare.refresh_source_replan_inputs",
                        lambda _runtime, _capture: 0)
    runtime = SimpleNamespace(_source_replan_in_progress=False)
    with source_replan_preparation(runtime, SimpleNamespace(request_id="original")):
        assert runtime._source_replan_origin_request_id == "original"
        with source_replan_preparation(runtime, SimpleNamespace(request_id="nested")):
            assert runtime._source_replan_origin_request_id == "nested"
        assert runtime._source_replan_origin_request_id == "original"
    assert runtime._source_replan_in_progress is False
    assert not hasattr(runtime, "_source_replan_origin_request_id")
