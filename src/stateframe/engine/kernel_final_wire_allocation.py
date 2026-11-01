"""The common complete-envelope builder and atomic request seal."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping

import kernel_frame_routes as _frame_routes

from kernel_request_seal import QualifiedRequest, OwnerPresentation, RequestSealError, freeze


def publishes_runtime_token_meter(candidate) -> bool:
    """Whether sealing copies the context meter back onto the runtime.

    A routed request's meter lives on its capture only (kernel_request_budget._request_token_meter):
    publishing it would hand the next request this route's tokenizer, and the routed context has none.
    """
    return not (_frame_routes.ROUTES_ACTIVE and getattr(candidate, "frame_route", None) is not None)


def routed_tokenize_endpoint(capture) -> str | None:
    """The tokenizer a routed capture was bound to at preparation (its route's tokenizer or endpoint).

    Sealing must keep it: the chat endpoint differs whenever the route names its own tokenizer.
    """
    if not (_frame_routes.ROUTES_ACTIVE and getattr(capture, "frame_route", None) is not None):
        return None
    bound = getattr(capture, "_tokenize_endpoint", None)
    return bound if isinstance(bound, str) and bound else None


def build_qualified_request(capture, runtime, body, *, endpoint, key, deadline=None):
    """Complete, fit and prove detached state, then publish one coherent record."""
    from kernel_request_kind import classify, gate_required, validate_adapter, RequestKindError
    if not classify(capture, runtime, body):
        raise RequestKindError("builder_requires_primary_request")
    validate_adapter(capture)
    from kernel_context import token_estimate
    from kernel_context_capacity import enforce_context_capacity
    from kernel_frame_allocation_capture import finalize_frame_candidate, capture_candidate
    from kernel_final_wire_budget import reconcile_frozen_budget, snapshot_revalidation_inputs
    from kernel_launch_identity_cost import enforce_provider_body, reasoning_profile
    from kernel_read_reuse_visibility import prepare_final_visibility
    from kernel_request_candidate import (
        detached_context, render_candidate, stabilize_host_fields,
        render_visibility_sources,
    )
    from kernel_request_visibility import qualify_visibility
    from kernel_request_reprepare import PreparationInput
    from kernel_request_tool_surface import author_final_tool_surface
    from kernel_stateframe_framefit5 import validate_final_product_boundary
    from kernel_tool_schema_lint import lint_tools, ToolSchemaContractError
    from kernel_wire_truth_gate import (
        WireTruthBlocked, capture_context, add_fact_messages, check_wire, enabled,
    )

    input_digest = hashlib.sha256(freeze({"body": body, "endpoint": endpoint})).hexdigest()
    previous = getattr(capture, "qualified_request", None)
    if previous is not None:
        if input_digest != previous.input_digest:
            raise RequestSealError("retry input differs from sealed request")
        previous.validate(capture, runtime)
        if gate_required(capture) and not capture._qualified_request_gate_passed:
            raise RequestSealError("sealed request did not pass its bound host gate")
        return previous
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("request deadline expired before qualification")
    # Availability is a logical-request service, not detached candidate
    # authority. Even a failed first candidate must consume the same retries.
    from kernel_request_budget import _request_token_meter
    _request_token_meter(runtime, capture, tokenizer_endpoint=endpoint)
    candidate, context = detached_context(capture, runtime)
    candidate._tokenize_deadline = deadline
    candidate._tokenize_endpoint = routed_tokenize_endpoint(capture) or endpoint
    body = copy.deepcopy(body)
    truth = capture_context(candidate, context, body, side_call=False)
    if enabled():
        body = add_fact_messages(body, truth, append_only=False)
        from kernel_release_wire_claims import READ_PREFIX, read_contract_capsule
        capsule, fields = read_contract_capsule(context)
        candidate.read_contract_wire_capsule = capsule
        candidate.read_contract_wire_fields = fields
        if capsule is not None:
            body["messages"] = [message for message in body["messages"]
                                if not str(message.get("content", "")).startswith(READ_PREFIX)]
            body["messages"].append({"role": "user", "content": capsule})
    from kernel_read_once import append_final_message
    body["messages"] = append_final_message(context, body["messages"])
    body, visibility = prepare_final_visibility(candidate, context, body, preserve_deferred=True)
    body = render_visibility_sources(candidate, context, body, visibility)
    body = candidate._ensure_final_wire_ticket(body)
    body = enforce_provider_body(body, profile=reasoning_profile(manifest=capture.journal.manifest))
    from kernel_stateframe_tool_stable import tool_mode
    author_final_tool_surface(candidate, context, body, request_id=candidate.request_id,
                              stabilize_cache_tools=True,
                              bind_current_surface=tool_mode() in {"on", "shadow"})
    if not isinstance(candidate.request_action_contract, Mapping):
        raise RequestSealError("qualified request lacks owner authority")
    # Freeze from admitted owner input before any semantic filtering. A final
    # menu rebind may change contract identity, but may not change its limits.
    candidate.owner_presentation = OwnerPresentation.capture(context, candidate.request_action_contract, body["messages"])
    host_preparer = getattr(candidate, "_qualified_host_text_preparer", None)
    if callable(host_preparer) and not getattr(candidate, "release_precomposed_host_text", None):
        from capability_request_preflight import PRECOMPOSED_HOST_TEXT_SCHEMA
        from kernel_frame_source_snapshot import frame_source_snapshot
        from kernel_request_candidate import candidate_source_plan
        plan = candidate_source_plan(candidate, context)
        body["messages"], host_receipt = host_preparer(
            body["messages"], runtime=context, tools=body.get("tools", []),
            frame_request_allocation=getattr(candidate, "_replacement_allocation", None),
            framefit5_inputs={
                "source_messages": getattr(candidate, "_framefit5_semantic_messages", body["messages"]),
                "task_message": candidate._framefit5_task_message,
                "source_plan": plan, "sources": frame_source_snapshot(context, plan),
                "owner_contract": candidate.request_action_contract,
                "owner_presentation": candidate.owner_presentation,
            },
        )
        candidate.release_precomposed_host_text = {
            "schema": PRECOMPOSED_HOST_TEXT_SCHEMA, "request_id": candidate.request_id,
            "receipt": copy.deepcopy(host_receipt),
        }
    from kernel_greenfact import maybe_append_greenfact
    # Canonicalize the tail first: host-fact producers may have appended a
    # user carrier after an existing GREENFACT. Appending before normalization
    # would produce two generated tails.
    stable = stabilize_host_fields(body, semantic_messages=getattr(candidate, "_framefit5_semantic_messages", body["messages"]))
    candidate.journal.emit("wire_stability_applied", producer="kernel_request_candidate",
                           request_id=candidate.request_id, step=candidate.step,
                           changed=stable != body, scope="host_fields", dispatch=False)
    body = render_candidate(candidate, context, stable)
    from kernel_cache_layout_candidate import compose_stable_guidance
    body = compose_stable_guidance(candidate, body)
    maybe_append_greenfact(body["messages"], context, journal=candidate.journal,
                           turn=candidate.step, request_id=candidate.request_id)
    before_fit_bytes = len(json.dumps(body).encode())
    body = finalize_frame_candidate(context, candidate, body, frame_counter=token_estimate)
    from kernel_cache_layout_candidate import render_cache_candidate, pending_publication
    from kernel_cache_layout_view import semantic_messages
    from kernel_request_budget import _request_token_meter
    body = render_cache_candidate(
        candidate, context, body, meter=_request_token_meter(context, candidate),
        inputs=candidate._frame_candidate_inputs)
    candidate.journal.emit("final_wire_rebudgeted", producer="kernel_final_wire_allocation",
                           request_id=candidate.request_id, step=candidate.step,
                           before=before_fit_bytes, after=len(json.dumps(body).encode()),
                           phase="complete_envelope_fit", dispatch=False)
    problems = lint_tools(body.get("tools"))
    if problems:
        raise ToolSchemaContractError(problems)
    extensions = candidate._request_capture_pre_wire_extensions(
        body, endpoint, key, deadline=deadline, runtime=context, read_only=True,
    )
    from kernel_frame_optional_allocation import selection_from_frame
    from kernel_request_seal import FrameSelectionProposal
    from kernel_stateframe_framefit5 import _state_payload
    frame = next(_state_payload(message)[1]["framefit5"] for message in semantic_messages(body["messages"])
                 if str(message.get("content", "")).startswith("[StateFrame]\n"))
    optional, inventory = selection_from_frame(
        *candidate._frame_candidate_inputs["optional_inputs"], frame)
    selection = FrameSelectionProposal(freeze(optional), freeze(inventory), b"{}")
    product_validation = validate_final_product_boundary(
        context, body["messages"], body.get("tools"), refresh_sources=False,
        refresh_headcarry=False, frame_selection=selection)
    # Output normalization can change body identity and prompt reserves. It
    # precedes the final receipt, whose measurement is carried into the seal.
    enforce_context_capacity(candidate, body, endpoint=endpoint, key=key)
    budget = dict(getattr(candidate, "request_budget_final_wire", None) or {})
    budget["max_request_bytes"] = candidate._frame_candidate_inputs["byte_ceiling"]
    candidate.request_budget_final_wire = budget
    budget = reconcile_frozen_budget(candidate, context, body,
                                    frozen_inputs=snapshot_revalidation_inputs(candidate, context, body))
    if budget.get("final_validation_failures"):
        raise RequestSealError("candidate failed final budget/source/menu proof: " +
                               ",".join(budget["final_validation_failures"]))
    candidate._frame_candidate_inputs.update(
        byte_ceiling=budget["max_request_bytes"], token_ceiling=budget["max_prompt_tokens"],
        measurement_identity=budget["measure_method"],
    )
    wire = json.dumps(body).encode()
    candidate._assert_wire_string_hygiene(wire)
    visibility, visibility_receipt = qualify_visibility(
        candidate, context, wire, visibility, candidate.request_source_final_delivery,
        candidate._frame_candidate_inputs["sources"],
    )
    from kernel_continuation_journal import journal_safe_fields
    candidate.journal.emit("request_source_final_wire", request_id=candidate.request_id,
                           original_request_id=getattr(capture, "_replacement_lineage", candidate.request_id),
                           body_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                           source_delivery=journal_safe_fields(candidate.request_source_final_delivery),
                           dispatch=False)
    candidate.journal.emit("read_reuse_visibility_checked", producer="kernel_read_reuse_visibility",
                           request_id=candidate.request_id, step=candidate.step,
                           result=visibility_receipt["result"],
                           missing_range_count=len(visibility_receipt["missing_ranges"]),
                           pins_added=visibility.get("pins_added", 0),
                           body_sha256=visibility_receipt["body_sha256"], dispatch=False)
    if visibility_receipt["result"] is not True:
        from kernel_request_visibility import SourceVisibilityRecoveryError

        raise SourceVisibilityRecoveryError(visibility, visibility_receipt)
    candidate.journal.emit("boundary_recovery_final_source_menu_checked",
                           producer="kernel_final_wire_allocation", request_id=candidate.request_id,
                           step=candidate.step, phase="candidate_qualification",
                           read_available=bool(candidate.request_source_final_delivery.get("read_tool_available")),
                           deferred_path_count=len(candidate.request_source_final_delivery.get("deferred_paths", ())),
                           deferred_notice_present=bool(candidate.request_source_final_delivery.get("deferred_notice_verified")),
                           dispatch=False)
    candidate._wire_truth_runtime = context
    candidate._wire_truth_side_call = False
    candidate._wire_truth_context = capture_context(candidate, context, body, side_call=False)
    try:
        check = check_wire(wire, candidate._wire_truth_context, emit=candidate.journal.emit,
                           attempt=1, allow_done_repair=False)
    except WireTruthBlocked as blocked:
        # Preserve source placement from the rejected, unsent candidate. The
        # repair path must reassemble and seal a new request before dispatch.
        blocked.source_layout_proposal = capture_candidate(
            runtime=context, capture=candidate, body=body,
            previous_allocation=getattr(capture, "_replacement_allocation", None),
            request_lineage=getattr(capture, "_replacement_lineage", None),
            **candidate._frame_candidate_inputs,
        ).to_bytes()
        raise
    if check.wire != wire or check.verdict not in {"PASS", "DISABLED"}:
        raise RequestSealError("candidate wire truth changed or rejected the proven bytes")
    # Every body producer and proof above precedes the sole allocation capture.
    allocation = capture_candidate(runtime=context, capture=candidate, body=body,
                                   previous_allocation=getattr(capture, "_replacement_allocation", None),
                                   request_lineage=getattr(capture, "_replacement_lineage", None),
                                   **candidate._frame_candidate_inputs)
    metadata = {
        "request_kind": getattr(capture, "request_kind", None),
        "adapter": getattr(capture, "_qualified_adapter", None),
        "request_action_contract": candidate.request_action_contract,
        "_request_source_closure_override": candidate._request_source_closure_override,
        "request_source_final_delivery": candidate.request_source_final_delivery,
        "request_budget_final_wire": candidate.request_budget_final_wire,
        "final_tool_surface": candidate.final_tool_surface,
        "framefit5_product_validation": product_validation,
        "runtime_authority": {name: copy.deepcopy(getattr(context, name, None))
                              for name in ("_completion_owner_request_contract",
                                           "_completion_owner_dispatch_tools",
                                           "completion_menu", "offered_tools")},
        "request_extensions": extensions, "visibility_check": visibility,
        "visibility_receipt": visibility_receipt,
        "host_contribution": getattr(candidate, "release_precomposed_host_text", None),
    }
    record = QualifiedRequest(
        request_id=capture.request_id, wire=wire,
        body_digest=hashlib.sha256(wire).hexdigest(), allocation=allocation,
        metadata=freeze(metadata), binding=freeze(runtime.binding()),
        input_digest=input_digest, owner_presentation=candidate.owner_presentation,
        deadline=deadline, lineage=allocation.identity["request_lineage"],
        measurement=candidate._request_measurement,
    )
    candidate.frame_request_allocation = allocation
    candidate.framefit5_product_validation = product_validation
    record.validate(candidate, context, body)
    # Snapshot can fail on identity drift, so construct it before publication.
    candidate.qualified_request = record
    preparation = PreparationInput.capture(candidate, context, body)
    from kernel_request_candidate import runtime_authority_fields
    runtime_updates = {name: copy.deepcopy(getattr(context, name))
                       for name in runtime_authority_fields(context) if hasattr(context, name)}
    if publishes_runtime_token_meter(candidate):
        runtime_updates["_request_token_meter"] = context._request_token_meter
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("request deadline expired during qualification")
    capture_updates = {
        **pending_publication(candidate, runtime),
        **{name: copy.deepcopy(metadata[name]) for name in (
            "request_action_contract", "_request_source_closure_override",
            "request_source_final_delivery", "request_budget_final_wire", "final_tool_surface",
            "framefit5_product_validation")},
        "frame_request_allocation": allocation, "qualified_request": record,
        "owner_presentation": candidate.owner_presentation,
        "_frame_request_allocation_candidate": None,
        "_framefit5_pending_qualification": False, "request_budget_exceeded": False,
        "_wire_truth_context": candidate._wire_truth_context, "_wire_truth_check": check,
        "_wire_truth_runtime": runtime, "_wire_truth_side_call": False,
        "read_contract_wire_capsule": getattr(candidate, "read_contract_wire_capsule", None),
        "read_contract_wire_fields": getattr(candidate, "read_contract_wire_fields", None),
        "release_precomposed_host_text": copy.deepcopy(metadata["host_contribution"]),
        "_request_preparation_input": preparation,
        "_qualified_request_gate_passed": False,
        "final_submitted_bytes_hash": "sha256:" + record.body_digest,
        "_request_measurement": record.measurement,
        "_tokenize_endpoint": candidate._tokenize_endpoint,
        "_tokenize_deadline": candidate._tokenize_deadline,
    }
    # Finish all fallible copies before publishing. The record is the sole
    # request authority; allocation and its body/proof aliases move together.
    runtime.__dict__.update(runtime_updates)
    capture.__dict__.update(capture_updates)
    capture.journal.emit("qualified_request_sealed", request_id=capture.request_id,
                         body_sha256="sha256:" + record.body_digest,
                         allocation_digest=allocation.digest,
                         generation=allocation.identity["generation"], dispatch=False)
    gate = getattr(capture, "_qualified_request_gate", None)
    if callable(gate):
        gate(record, capture, runtime)
    record.validate(capture, runtime, body)
    # Internal candidate construction may have no enclosing adapter. It does
    # not grant transport admission; physical send requires a bound gate.
    capture._qualified_request_gate_passed = callable(gate)
    return record


def qualify_final_wire_allocation(capture, runtime, body, preparation, *, deadline=None):
    """Compatibility boundary: validate the existing seal, never reconstruct it."""
    from kernel_request_seal import RequestSealError
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("request deadline expired before final wire binding")
    record = getattr(capture, "qualified_request", None)
    if record is None:
        raise RequestSealError("final wire has no qualified request")
    record.validate(capture, runtime, body)
    return body
