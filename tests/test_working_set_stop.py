"""Typed suspension survives durable journal, worker, terminal and product boundaries."""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_working_set_state as ws
import kernel_working_set_stop as stop
from kernel_continuation_journal import ContinuationJournal
from kernel_completion_owner import terminal_result
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_pilot import ACTIVE_RUNTIME
from stateframe import outcomes, resume, runner

from test_working_set_state import capture

ROOT = Path(__file__).resolve().parents[1]
FLOOR_REASONS = {stop.ReasonCode.MANDATORY_ENVELOPE_OVER_CAP, stop.ReasonCode.FLOOR_INFEASIBLE}
NON_FLOOR_REASONS = tuple(reason for reason in stop.ReasonCode if reason not in FLOOR_REASONS)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def suspension():
    proof = stop.MandatoryEnvelopeProof(b'{"messages":[]}', 3,
                                       (("fixed", 15), ("protected", 0), ("pins", 0)))
    return stop.Suspension(
        reason_code=stop.ReasonCode.MANDATORY_ENVELOPE_OVER_CAP,
        logical_request_id="unsent-request", due_ordinal=2,
        snapshot_digest="a" * 64, active_obligation_digest="b" * 64,
        violated_limits=("serialized_bytes",), proof=proof,
        minimum_policy_candidate_ref="candidate-ref", trials_ref="trials-ref",
        recovery_condition="Increase the mandatory envelope byte cap.")


def test_typed_verified_overflow_only():
    value = suspension()
    value.validate()
    assert stop.BLOCKING_REASONS == set(stop.ReasonCode)
    for reason in stop.ReasonCode:
        replaced = replace(value, reason_code=reason)
        if reason in FLOOR_REASONS:
            assert stop.Suspension.from_document(replaced.document()).reason_code == reason
        else:
            with pytest.raises(ValueError, match="mandatory"):
                replaced.validate()
    with pytest.raises(ValueError, match="exceed"):
        stop.MandatoryEnvelopeProof(b"x", 10, (("fixed", 1),)).validate()
    with pytest.raises(ValueError, match="breakdown"):
        stop.MandatoryEnvelopeProof(b"xx", 1, (("fixed", 1),)).validate()


@pytest.mark.parametrize("reason", NON_FLOOR_REASONS)
def test_non_floor_validation_and_roundtrip(reason):
    value = replace(suspension(), reason_code=reason, proof=None,
                    minimum_policy_candidate_ref=None, violated_limits=())
    document = value.document()
    assert document["reason_code"] == reason.value
    assert document["proof"] is None
    assert stop.Suspension.from_document(json.loads(json.dumps(document))) == value
    with pytest.raises(ValueError, match="typed"):
        replace(value, reason_code=reason.value).validate()
    with pytest.raises(ValueError, match="mandatory"):
        replace(value, proof=suspension().proof).validate()
    with pytest.raises(ValueError, match="limits"):
        replace(value, violated_limits=("unknown_limit",)).validate()
    with pytest.raises(ValueError):
        stop.Suspension.from_document({**document, "reason_code": "UNKNOWN"})


@pytest.mark.parametrize("reason", (None, *NON_FLOOR_REASONS))
def test_order_and_failed_durable_emit(tmp_path, monkeypatch, reason):
    value = suspension() if reason is None else replace(suspension(), reason_code=reason,
        proof=None, minimum_policy_candidate_ref=None, violated_limits=())
    journal = ContinuationJournal(tmp_path, "suspended", {})
    rt = SimpleNamespace(journal=journal)
    ws.record_capture(rt, capture())
    order = []
    original = journal.emit
    def emit(kind, **fields):
        if kind == "working_set_blocked":
            assert ws.state_for(rt).suspension is None
            order.append("durable")
        return original(kind, **fields)
    monkeypatch.setattr(journal, "emit", emit)
    def record_problem(runtime, problem):
        assert runtime.working_set_state.suspension is not None
        assert problem["reason_code"] == "WORKING_SET_ALLOCATION_BLOCKED"
        order.append("problem")
    monkeypatch.setattr(stop, "record_problem", record_problem)
    try:
        with pytest.raises(stop.WorkingSetSuspended) as caught:
            stop.suspend(rt, value)
        assert order == ["durable", "problem"]
        result = caught.value.outcome()
        assert result["result"] == "BLOCKED" and result["problems"]
        assert result["resumable"] and not result["accepted"] and not result["certified"]
        saved = rt.working_set_state.checkpoint()
        restored = ws.WorkingSetState.from_checkpoint(json.loads(json.dumps(saved)))
        assert restored.suspension.document() == value.document()
        def fail(kind, **fields):
            raise OSError("fsync failed")
        monkeypatch.setattr(journal, "emit", fail)
        rt.working_set_state.suspension = None
        with pytest.raises(OSError, match="fsync"):
            stop.suspend(rt, value)
        assert rt.working_set_state.suspension is None
    finally:
        journal.close()


def _runtime(journal):
    rt = StateFrameRuntime.__new__(StateFrameRuntime)
    rt.journal = journal
    rt.owner = SimpleNamespace(current_step=1)
    rt.snapshot_id = "snapshot"
    rt.endpoint_profile = None
    rt.system_outcome = None
    rt._finished = False
    rt._active_capture = None
    rt._record_passive_retention_capture = lambda _: None
    rt._record_passive_retention_delivery_metric = lambda: None
    rt.stateframe_completion_owner_mode = "on"
    rt.stateframe_host_done_facts_mode = "on"
    rt._finalize_host_done_facts = lambda *a, **k: pytest.fail("cannot certify a suspension")
    rt.stateframe_completion_v2_mode = "on"
    rt.stateframe_cache_prefill_economics_mode = "off"
    rt.delivery_contract = None
    rt.information = None
    rt.continuation = None
    rt.lifecycle = lambda _: None
    rt.step_limit = 10
    rt.parent = SimpleNamespace(snapshot=lambda: {}, completed_attempts=[])
    rt.actions = SimpleNamespace(actions=[], diagnose=lambda: {})
    rt.binding = lambda: {"product_revision": "sha256:fixture"}
    rt._emit_tool_call_cap_audit = lambda: None
    rt._runtime_token = ACTIVE_RUNTIME.set(rt)
    rt._parent_token = None
    return rt


@pytest.mark.parametrize("reason", (None, *NON_FLOOR_REASONS))
def test_injected_worker_finish_and_engine_end(tmp_path, reason):
    journal = ContinuationJournal(tmp_path, "blocked-boundary", {})
    rt = _runtime(journal)
    try:
        ws.record_capture(rt, capture())
        with pytest.raises(stop.WorkingSetSuspended) as caught:
            if reason is None:
                stop.inject_suspension_for_test(rt, suspension())
                stop.check_test_injection(rt)
            else:
                from kernel_working_set_live import _blocked
                from kernel_working_set_prepare import Blocked, Measurement, Trial
                snapshot = SimpleNamespace(request_id="unsent-request", due_ordinal=1,
                                           snapshot_digest="a" * 64)
                trials = ()
                if reason == stop.ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED:
                    trials = (Trial("c" * 64, "d" * 64, 80,
                        Measurement(101, 50, "tokenize"), ("prompt_tokens",)),)
                _blocked(rt, snapshot, Blocked(reason, 0, 0, 100, (), trials=trials))
        document = caught.value.suspension.document()
        assert document["reason_code"] == (reason or suspension().reason_code).value
        saved = ws.state_for(rt).checkpoint()
        assert ws.WorkingSetState.from_checkpoint(json.loads(json.dumps(saved))).checkpoint() == saved
        rows = [json.loads(line) for line in journal.path.read_bytes().splitlines()]
        assert ws.reconstruct(rows).checkpoint() == saved
        blocked, = [row for row in rows if row["kind"] == "working_set_blocked"]
        assert blocked["suspension"] == document
        if reason is not None:
            assert document["proof"] is None
            assert document["minimum_policy_candidate_ref"] is None
            assert "mandatory_envelope_blob_ref" not in blocked
            assert document["violated_limits"] == (
                ["prompt_tokens"] if reason == stop.ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED else [])
        restored = ws.WorkingSetState.from_checkpoint(saved)
        ws.retry_unsent_boundary(restored)
        assert restored.suspension is None
        assert restored.active_obligations() == ws.state_for(rt).active_obligations()
        # Execute the actual outer worker handler, without starting an engine.
        tree = ast.parse((ROOT / "src/stateframe/engine/probes/react_repair.py").read_text())
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                    and isinstance(n.type, ast.Name) and n.type.id == "WorkingSetSuspended"]
        assert len(handlers) >= 2, "both request and outer catch must be explicit"
        outer = next(n for n in handlers if any(isinstance(s, ast.Return) for s in n.body))
        fn = ast.FunctionDef(name="handle", args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg="exc")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=copy.deepcopy(outer.body), decorator_list=[])
        env = dict(measurement=rt, msgs=[], fn="fixture", step=1,
                   finish_native_batch_transcript=lambda _: None,
                   _attach_env_receipt=lambda value, _: value)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     "worker_suspension_handler", "exec"), env)
        result = env["handle"](caught.value)
        assert result["result"] == rt._last_finish_result["result"] == "BLOCKED"
        assert terminal_result(rt, result)["result"] == "BLOCKED"
        child = ast.parse((ROOT / "src/stateframe/_child.py").read_text())
        emit = next(n for n in ast.walk(child) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "emit" and n.value.args
                    and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == "engine_end")
        ends = []
        exec(compile(ast.Module(body=[emit], type_ignores=[]), "engine_end", "exec"),
             dict(result=result, elapsed=0, emit=lambda kind, **f: ends.append((kind, f))))
        assert ends[0][1]["result"] == "BLOCKED"
        _assert_summary_outcome_and_resume_contract(tmp_path, result)
    finally:
        journal.close()


def test_summary_outcome_and_resume_contract(tmp_path):
    value = stop.WorkingSetSuspended(suspension()).outcome()
    _assert_summary_outcome_and_resume_contract(tmp_path, value)


def _assert_summary_outcome_and_resume_contract(tmp_path, value):
    (tmp_path / "engine-result.json").write_text(json.dumps({"result": value}))
    # Exercise the summary terminal projection itself, not a copied allowlist.
    tree = ast.parse((ROOT / "src/stateframe/runner.py").read_text())
    assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                          and t.value.id == "summary" and isinstance(t.slice, ast.Constant)
                          and t.slice.value == "engine_terminal" for t in n.targets))
    summary = {"engine_result": "BLOCKED", "child_exit_code": 0,
               "candidate_captured": True, "problems": value["problems"]}
    exec(compile(ast.Module(body=[assign], type_ignores=[]), "summary_projection", "exec"),
         dict(summary=summary, result=value))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["engine_terminal"]["suspension"] == value["suspension"]
    assert saved["engine_terminal"]["problems"] == value["problems"]
    assert saved["engine_terminal"]["problems"][0]["suspension"] == value["suspension"]
    assert stop.is_suspended_result(value)
    assert saved["engine_terminal"]["resumable"] and saved["problems"]
    saved.update(outcomes.project(saved))
    assert saved["runtime_outcome"] == "suspended" and saved["artifact"] == "partial"
    assert outcomes.exit_code(saved) == 1
    assert not runner._session_end_is_final(saved)
    assert not resume._terminal_resume_refusal("BLOCKED", saved)
    assert not resume.run_finished_for_good(tmp_path)


def test_injection_off_and_resume_unsent_boundary(tmp_path, monkeypatch):
    rt = SimpleNamespace(journal=SimpleNamespace(emit=lambda *a, **k: None))
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    stop.inject_suspension_for_test(rt, suspension())
    stop.check_test_injection(rt)
    assert not hasattr(rt, "working_set_state")
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    ws.record_capture(rt, capture())
    state = ws.state_for(rt)
    state.suspension = suspension()
    saved = state.checkpoint()
    # The replay path reconstructs events; a valid earlier checkpoint carries
    # the typed state too. A suspension is never a prepared or submitted wire.
    fresh = ws.WorkingSetState.from_checkpoint(saved)
    ws.retry_unsent_boundary(fresh)
    assert fresh.logical_ordinal == state.logical_ordinal
    assert fresh.active_obligations() == state.active_obligations()
    assert fresh.suspension is None
