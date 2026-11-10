"""A refused host action must be returned to the model and must not strand continuation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_nav_basics
import kernel_public_evidence
from kernel_continuation_evidence import (
    binding_versions,
    classify_run_test_exit,
)
from kernel_continuation_executive import ContinuationExecutive
from kernel_continuation_journal import ContinuationJournal
from kernel_continuation_runtime import LiveContinuation
from kernel_continuation_state import ContinuationState
from kernel_measurement import digest
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_step_ledger import StepLedger
from probes import react_repair


def _binding() -> dict:
    return {
        "product_files": {"pkg/value.py": "sha256:source"},
        "harness_files": {},
    }


def _continue_decision(engine) -> tuple[str, StateFrameRuntime]:
    """Run the production final-menu gate; None means the provider request may proceed."""
    live = LiveContinuation.__new__(LiveContinuation)
    live.engine = engine
    live.journal = getattr(engine, "journal", None)
    live.guarded_effects = None
    live.completed = False
    live.certified = False
    live.terminal_blocked = False
    live._stranded_blocker = None
    live._stranded_revision = None
    live.reconcile_external_effects = lambda: set()
    live.completion_finding = lambda: {"code": "CONTINUATION_BLOCKED"}

    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.continuation = live
    runtime.system_outcome = None
    live.runtime = runtime

    prepared = runtime._finalize_continuation_menu(
        SimpleNamespace(request_id="next-request"),
        ({"function": {"name": "read"}}, {"function": {"name": "done"}}),
    )
    decision = "continue" if prepared is None else str(prepared.overflow["kind"])
    return decision, runtime


def _record_selector_refusal(
    tmp_path: Path,
    selector: str,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    resolution_error: bool = False,
) -> dict[str, object]:
    """Drive the real selector refusal, receipt, closure, settlement, and next-request gate."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "value.py").write_text("VALUE = 1\n")

    if resolution_error:
        assert monkeypatch is not None
        candidate = root / selector

        # Isolate _run_tests_impl's resolution-exception refusal after the public
        # selector parser has admitted the host-side failure case.
        monkeypatch.setattr(
            kernel_public_evidence,
            "requested_check_targets",
            lambda _root, _explicit: [selector],
        )
        original_resolve = Path.resolve

        def fail_candidate(path, *args, **kwargs):
            if path == candidate:
                raise RuntimeError("synthetic symlink resolution loop")
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", fail_candidate)
        monkeypatch.setattr(
            kernel_nav_basics,
            "missing_path_message",
            lambda _root, requested, label="read": (
                f"[selector] MISSING: {requested} (host could not resolve selector)."
                if label == "selector" else f"MISSING: {requested}."
            ),
        )

    args = {"target": selector}
    observation, _codes = react_repair.run_tests_for_selectors(
        root, "value", (selector,), row={})

    # Match the live run_tests producer: the refusal prefix must reach the top
    # of the observation so the receipt retains its pre-dispatch result class.
    if observation.startswith(react_repair.HOST_REFUSED_PREFIX):
        result_class = "HARNESS_ERROR"
    elif observation.startswith("[selector] REFUSED_PREDISPATCH"):
        result_class = "REFUSED_PREDISPATCH"
    else:
        result_class = classify_run_test_exit(react_repair._selector_exit_code(observation))

    journal = ContinuationJournal(tmp_path, "predispatch-refusal", {"origin": "test"})
    try:
        engine = ContinuationExecutive(journal)
        binding = _binding()
        dependencies = binding_versions(binding)
        engine.apply(
            "create", new_id="root", objective="run requested check",
            pending_decision="run check", dependencies=dependencies, turns=3,
            acceptance={"kind": "no_public_checks", "selectors": []},
        )
        assert engine.choose_next(binding) == "root"

        step = 1
        request_id = "request-1"
        effect_id = "effect-1"
        arguments_hash = digest(args)
        step_ledger = StepLedger()
        step_ledger.start("run_tests", args, step, binding, tool_call_id="call-1")
        dispatch = journal.emit(
            "step_dispatch_binding", producer="stateframe_runtime",
            snapshot_id="snapshot", accepted=True, expected=binding, current=binding,
            continuation_id="root", continuation_effect_id=effect_id,
            request_id=request_id, step=step, tool="run_tests",
            arguments_hash=arguments_hash,
        )
        engine.apply(
            "begin_effect", intention_id="root", effect_id=effect_id,
            action_hash=digest({"tool": "run_tests", "arguments_hash": arguments_hash}),
            arguments_hash=arguments_hash, tool="run_tests", read_only=False,
            step=step, request_id=request_id, snapshot_id="snapshot",
            dispatch_ref=dispatch["event_id"], current_binding=binding,
            current_dependencies=dependencies,
        )
        step_ledger.pending.update({
            "continuation_id": "root",
            "continuation_effect_id": effect_id,
            "continuation_dispatch_ref": dispatch["event_id"],
        })

        receipt = {
            "receipt_id": "receipt-1", "run_id": journal.run_id,
            "request_id": request_id, "turn": step, "kind": "test",
            "intent": selector, "result_class": result_class,
            "continuation_id": "root", "continuation_effect_id": effect_id,
            "continuation_dispatch_ref": dispatch["event_id"],
        }
        journal.emit("action_receipt", receipt=receipt)
        step_ledger.receipt(receipt)

        messages = [{
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "run_tests", "arguments": json.dumps(args),
            }}],
        }]
        tool_result = react_repair._tool_reply_for_call(messages, "call-1", observation)
        messages.append(tool_result)
        closed_record = step_ledger.close(messages, binding)
        assert closed_record is not None
        closure = journal.emit("action_outcome_closed", record=closed_record)

        settled = engine.settle_host_effect(
            effect_id=effect_id, receipt_ref=closure["event_id"],
            current_binding=binding,
        )
        assert settled is not None
        decision, runtime = _continue_decision(engine)
        return {
            "observation": observation,
            "tool_result": tool_result,
            "result_class": result_class,
            "effect_status": engine.state.effects[effect_id].status,
            "settlement_status": settled["data"]["status"],
            "decision": decision,
            "system_outcome": runtime.system_outcome,
        }
    finally:
        journal.close()


def test_missing_selector_refusal_settles_and_allows_next_request(tmp_path):
    trace = _record_selector_refusal(tmp_path, "nope/")

    assert trace["tool_result"]["role"] == "tool"
    assert "REFUSED_PREDISPATCH" in trace["tool_result"]["content"]
    assert "MISSING: nope/" in trace["tool_result"]["content"]
    assert (trace["effect_status"], trace["decision"]) == ("not_applied", "continue"), (
        f"receipt={trace['result_class']}; settlement={trace['settlement_status']}; "
        f"next_request={trace['decision']}"
    )
    assert trace["settlement_status"] == "not_applied"
    assert trace["result_class"] == "REFUSED_PREDISPATCH"
    assert trace["system_outcome"] is None


def test_refusal_after_another_selector_stays_typed(monkeypatch):
    def fake_run_tests(_root, _fn, selector, _rel_hint=None, row=None):
        if selector == "passing.py":
            return "pytest exit 0 — 1 passed"
        return "[selector] REFUSED_PREDISPATCH: missing test"

    monkeypatch.setattr(react_repair, "run_tests", fake_run_tests)
    observation, codes = react_repair.run_tests_for_selectors(
        Path("/unused"), "value", ("passing.py", "missing.py"))

    assert codes == (0, -1)
    assert observation.startswith("[selector] REFUSED_PREDISPATCH")
    assert "pytest exit 0" in observation


@pytest.mark.parametrize(
    ("selector", "resolution_error"),
    [
        pytest.param("missing_test.py", False, id="missing-file"),
        pytest.param("broken.py", True, id="resolution-exception"),
    ],
)
def test_other_selector_refusal_sites_return_a_result_and_continue(
    tmp_path, monkeypatch, selector, resolution_error
):
    trace = _record_selector_refusal(
        tmp_path, selector, monkeypatch=monkeypatch,
        resolution_error=resolution_error,
    )

    assert trace["tool_result"]["role"] == "tool"
    assert "REFUSED_PREDISPATCH" in trace["tool_result"]["content"]
    assert "MISSING:" in trace["tool_result"]["content"]
    assert (trace["effect_status"], trace["decision"]) == ("not_applied", "continue"), (
        f"selector={selector}; receipt={trace['result_class']}; "
        f"settlement={trace['settlement_status']}; next_request={trace['decision']}"
    )
    assert trace["settlement_status"] == "not_applied"
    assert trace["result_class"] == "REFUSED_PREDISPATCH"
    assert trace["system_outcome"] is None

