"""A subframe's live host frame projects the owner protocol its sealed request carries.

Live P0 (--parallel 2, 0.7.9): after a subframe edited an owned file and its check
failed, the completion owner reordered the callable menu (stall rule). The prepare-time
attach bound that menu order into the live host frame, while final qualification
re-rendered the sealed frame from the rebound contract on a detached runtime. Every
later validation of the sealed request on the live runtime then failed with
``owner_protocol_keys=['admitted_actions']`` and the replan reproduced it forever.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_unit_policy
from kernel_frame_allocation_validation import _authority_mismatch_message, _authority_view
from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
from kernel_stateframe_envelope import STATE_HEADER
from kernel_stateframe_framefit5 import (
    PRODUCT_MODE, _state_payload, attach_owner_protocol, scope_typed_state_for_unit,
)

_TOOLS = ("done", "file_create", "grep", "insert_before", "read", "run_tests",
          "str_replace", "subframe_blocked")
# The stall rule's surface order (work actions first), as recorded in the live run.
_STALL_MENU = ("str_replace", "read", "grep", "run_tests", "done", "file_create",
               "insert_before", "subframe_blocked")


def _schemas(names):
    return [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in names]


def _contract(*, callable_actions=None):
    contract = {"route": "ACTION", "advertised_actions": list(_TOOLS),
                "admitted_tool_schemas": _schemas(_TOOLS)}
    if callable_actions is not None:
        contract["callable_actions"] = list(callable_actions)
    return contract


def _spec():
    recipe = CheckRecipe(("tests/test_pricing.py",), ("tests/test_pricing.py",), "pytest")
    contract = UnitContract("pricing", "Fix pricing.", ("pricing.py",), (), (), (), (), (), recipe)
    return UnitSpec("call", "pricing", "execution", "seed", contract, (), (),
                    ("pricing.py", "tests/test_pricing.py"), (), "", 0, 0, "manifest", "policy",
                    ParallelLimits())


def _host_frame():
    return {"mode": PRODUCT_MODE, "open_sites": [], "read_facts": [], "applied_edits": [],
            "action_ledger": {"schema": "stateframe-framefit3-action-ledger/1", "reads": [],
                              "edits": [], "unclassified_actions": [],
                              "superseded_test_count": 0, "last_test": None}}


def _state_message(frame):
    from kernel_stateframe_framefit5 import _serialize_state_message
    return {"role": "user", "content": _serialize_state_message({"state": {"framefit5": frame}})}


def _wire_frame(messages):
    return next(_state_payload(row)[1]["framefit5"] for row in messages
                if str(row.get("content") or "").startswith(STATE_HEADER + "\n"))


def _assert_live_authority(runtime, messages, spec):
    """The validator's scoped owner comparison, on the given (live) runtime."""
    frame = _wire_frame(messages)
    projection = scope_typed_state_for_unit(
        runtime._framefit_action_context["framefit5"], spec)
    frame_view = {"owner_protocol": frame.get("owner_protocol")}
    projection_view = {"owner_protocol": projection.get("owner_protocol")}
    if _authority_view(frame_view) != _authority_view(projection_view):
        raise FrameAllocationReplanNeeded(_authority_mismatch_message(
            ["owner_protocol"], frame_view, projection_view))


@pytest.fixture
def sealed_subframe(tmp_path, monkeypatch):
    """Real qualification builder; unrelated owner/transport/allocation services isolated."""
    import kernel_request_candidate as candidates
    import kernel_request_budget as budget
    import kernel_wire_truth_gate as truth
    from kernel_request_diet import RequestTokenMeter
    from kernel_provider_dialect import ProviderCapabilities
    from kernel_final_wire_allocation import build_qualified_request
    from kernel_measurement import AuditJournal
    from kernel_request_reprepare import PreparationInput
    from kernel_request_seal import QualifiedRequest

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    spec = _spec()
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: SimpleNamespace(spec=spec))
    journal = AuditJournal(tmp_path, "sealed-owner", {})
    plan = {"paths": [], "units": [], "source_contents": {}, "revisions": {},
            "workspace_authorization": {"scope": "test", "generation": "test"}}
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path), journal=journal,
        _request_source_closure=plan, _empirical_frame_sources={}, frame_ceiling=None,
        _framefit_action_context={"framefit5": _host_frame()},
        binding=lambda: {"product_revision": "rev"})
    final_contract = _contract()
    capture = SimpleNamespace(request_id="sealed", step=1, journal=journal,
        request_action_contract=final_contract, _request_source_closure_override=plan,
        _ensure_final_wire_ticket=lambda body: body,
        _request_capture_pre_wire_extensions=lambda *a, **k: {},
        _assert_wire_string_hygiene=lambda wire: None,
        final_tool_surface=None, release_precomposed_host_text=None,
        _request_measurement=None, _frame_candidate_counter=None,
        _frame_candidate_fallback_fit=None, _frame_candidate_inputs={},
        _tokenize_endpoint=None,
        _framefit5_task_message={"role": "user", "content": "Task: fix pricing"})
    body = {"model": "local", "messages": [
        {"role": "system", "content": "rules"}, capture._framefit5_task_message,
        _state_message(_host_frame()),
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}],
        "tools": _schemas(_TOOLS), "max_tokens": 100}
    # Prepare-time attach on the LIVE runtime: the stall rule's callable menu order.
    body["messages"] = attach_owner_protocol(
        body["messages"], body["messages"], runtime, _contract(callable_actions=_STALL_MENU))
    assert (runtime._framefit_action_context["framefit5"]["owner_protocol"]["admitted_actions"]
            == list(_STALL_MENU))

    meter = RequestTokenMeter(disabled=True)
    meter.bind_request("sealed", "sealed")
    runtime._request_token_meter = meter
    monkeypatch.setattr(budget, "_request_token_meter", lambda *a, **k: meter)
    monkeypatch.setattr("kernel_request_kind.classify", lambda *a: True)
    monkeypatch.setattr(truth, "capture_context", lambda *a, **k: {})
    monkeypatch.setattr(truth, "check_wire",
                        lambda wire, *a, **k: SimpleNamespace(wire=wire, verdict="DISABLED"))
    monkeypatch.setattr("kernel_wire_truth_facts.require_fact_context", lambda value: value)
    monkeypatch.setattr(truth, "enabled", lambda: False)
    monkeypatch.setattr("kernel_read_once.append_final_message", lambda rt, rows: rows)

    def render(cap, rt, rendered):
        # Final assembly binds the sealed (rebound) contract into the frame on the
        # detached candidate runtime, exactly as assemble_framefit5_request does.
        return {**rendered, "messages": attach_owner_protocol(
            rendered["messages"], rendered["messages"], rt, cap.request_action_contract,
            owner_presentation=cap.owner_presentation)}

    monkeypatch.setattr(candidates, "render_candidate", render)
    monkeypatch.setattr("kernel_provider_dialect.capabilities_for",
                        lambda *a, **k: ProviderCapabilities(dialect="llama.cpp"))
    monkeypatch.setattr("kernel_request_tool_surface.author_final_tool_surface", lambda *a, **k: None)
    monkeypatch.setattr("kernel_cache_layout_candidate.compose_stable_guidance", lambda cap, b: b)
    monkeypatch.setattr("kernel_greenfact.maybe_append_greenfact", lambda *a, **k: None)
    monkeypatch.setattr("kernel_stateframe_framefit5.validate_final_product_boundary",
                        lambda *a, **k: {})
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *a, **k: {"status": "COMPLETE", "proof_status": "VERIFIED"})
    monkeypatch.setattr("kernel_frame_source_snapshot.verify_source_coverage_claims",
                        lambda *a, **k: None)
    monkeypatch.setattr("kernel_frame_optional_allocation.selection_from_frame",
                        lambda *a, **k: ({}, {}))
    monkeypatch.setattr("kernel_final_wire_budget.reconcile_frozen_budget",
                        lambda *a, **k: {"max_request_bytes": 100_000, "max_prompt_tokens": 50_000,
                                         "measure_method": "test"})
    monkeypatch.setattr("kernel_final_wire_budget.snapshot_revalidation_inputs", lambda *a, **k: {})
    monkeypatch.setattr("kernel_working_set_live.certified_visibility",
                        lambda allocated: ({"pins_added": 0}, {
                            "result": True, "missing_ranges": [], "body_sha256": "x"}))
    allocation = SimpleNamespace(identity={"request_lineage": "sealed", "generation": 0},
                                 digest="allocation")
    monkeypatch.setattr("kernel_frame_allocation_capture.capture_candidate",
                        lambda **kwargs: allocation)
    monkeypatch.setattr(PreparationInput, "capture", classmethod(lambda cls, *a, **k: None))
    validated = []

    def validate(record, cap, rt, sealed_body=None):
        # Stand-in for the seal's allocation validation: the scoped owner comparison.
        validated.append(rt)
        _assert_live_authority(rt, json.loads(record.wire)["messages"], spec)

    monkeypatch.setattr(QualifiedRequest, "validate", validate)

    def run():
        return build_qualified_request(capture, runtime, copy.deepcopy(body),
                                       endpoint="http://local/v1", key="")
    return run, runtime, capture, spec, validated


def test_live_host_frame_carries_the_sealed_owner_protocol(sealed_subframe):
    """RED on 0.7.9: the post-publication validation on the live runtime fails with
    owner_protocol_keys=['admitted_actions']."""
    run, runtime, _, spec, validated = sealed_subframe
    record = run()
    # The seal validated the detached candidate, then the live runtime after publication.
    assert validated[-1] is runtime
    frame = _wire_frame(json.loads(record.wire)["messages"])
    assert frame["owner_protocol"]["admitted_actions"] == list(_TOOLS)
    live = runtime._framefit_action_context["framefit5"]["owner_protocol"]
    assert live == frame["owner_protocol"]
    _assert_live_authority(runtime, json.loads(record.wire)["messages"], spec)
