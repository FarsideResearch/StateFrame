"""Request-only recovery; never replay a StateFrame or acquire another lease."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
import time

from kernel_frame_request_allocation import (
    FrameRequestAllocation,
    FrameRequestAllocationError,
)


class RequestPreparationContractError(RuntimeError):
    """A preparation invariant failed; retrying a caller body cannot repair it."""

    harness_contract = True
    outcome = "HARNESS_CONTRACT"
    failure_origin = "harness"


class SourceRevisionReplanRefused(RequestPreparationContractError):
    """An unsent request could not obtain stable, fresh source authority."""

    def __init__(self, code: str) -> None:
        self.stop_code = code
        super().__init__(code)


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _serialized_frame_request_allocation(capture: object) -> bytes | None:
    allocation = getattr(capture, "frame_request_allocation", None)
    if allocation is None:
        return None
    if not isinstance(allocation, FrameRequestAllocation):
        raise RequestPreparationContractError(
            "request preparation has an invalid frame request allocation",
        )
    try:
        return allocation.to_bytes()
    except FrameRequestAllocationError as exc:
        raise RequestPreparationContractError(
            "request preparation frame allocation failed integrity validation",
        ) from exc


def _captured_replacement_source_layout(
        preparation: PreparationInput,
) -> FrameRequestAllocation | None:
    """Read the prior source placement without granting old wire authority."""
    encoded = preparation.frame_request_allocation or preparation.source_layout_proposal
    if encoded is None:
        return None
    try:
        return FrameRequestAllocation.from_bytes(encoded)
    except FrameRequestAllocationError as exc:
        raise RequestPreparationContractError(
            "captured frame allocation failed integrity validation",
        ) from exc


def _current_binding(runtime: object) -> object:
    # A cached run-boundary binding cannot authenticate an external file write.
    # Refresh the existing workspace view/cache, without preparing another step.
    current_view = getattr(runtime, "current_view", None)
    if callable(current_view):
        current_view("external")
    invalidate = getattr(runtime, "_invalidate_binding_cache", None)
    if callable(invalidate):
        invalidate()
    return runtime.binding()


def refresh_source_replan_inputs(runtime: object, capture: object | None = None) -> int:
    """Discard stale optional source and read pins before a fresh preparation."""
    _current_binding(runtime)
    # Empirical source bytes are an optimization, never the new request's
    # authority. A changed file can leave its old bytes in this cache even
    # after the workspace view has advanced.
    setattr(runtime, "_empirical_frame_sources", {})
    from kernel_read_reuse_visibility import retire_changed_revision_candidates
    return retire_changed_revision_candidates(capture or runtime, runtime)


@contextmanager
def source_replan_preparation(runtime: object, capture: object | None = None):
    """Mark the sole fresh build so regenerated stale read caches stay optional."""
    retired = refresh_source_replan_inputs(runtime, capture)
    previous = getattr(runtime, "_source_replan_in_progress", False)
    setattr(runtime, "_source_replan_in_progress", True)
    try:
        yield retired
    finally:
        # Preparation may repopulate the optional cache from old read facts.
        # Do not carry those bytes into final-wire validation after this scope.
        setattr(runtime, "_empirical_frame_sources", {})
        setattr(runtime, "_source_replan_in_progress", previous)


def _source_plan_matches_after_deferred_discharge(
        original_bytes: bytes, current: object,
) -> bool:
    """Allow only ordered removal of already-deferred paths during recovery.

    A source read can resolve a prior deferral between request capture and its
    bounded wire repair. All other source-plan authority stays byte-for-byte
    fixed, and final wire qualification still has to prove the discharged path.
    """
    try:
        original = json.loads(original_bytes)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(original, Mapping) or not isinstance(current, Mapping):
        return False
    original_paths = original.get("deferred_paths")
    current_paths = current.get("deferred_paths")
    if (not isinstance(original_paths, list) or not isinstance(current_paths, list)
            or any(not isinstance(path, str) or not path for path in (*original_paths, *current_paths))
            or len(set(original_paths)) != len(original_paths)
            or len(set(current_paths)) != len(current_paths)):
        return False
    cursor = 0
    for path in current_paths:
        try:
            cursor = original_paths.index(path, cursor) + 1
        except ValueError:
            return False
    original_without_deferred = dict(original)
    current_without_deferred = dict(current)
    original_without_deferred["deferred_paths"] = []
    current_without_deferred["deferred_paths"] = []
    return _bytes(original_without_deferred) == _bytes(current_without_deferred)


@dataclass(frozen=True, slots=True)
class PreparationInput:
    request_id: str
    step: int
    body: bytes
    binding: bytes
    source_plan: bytes
    request_source_plan: bytes | None
    sources: bytes
    frame_request_allocation: bytes | None = None
    qualified_request: object | None = None
    source_layout_proposal: bytes | None = None

    @classmethod
    def capture(
            cls, capture: object, runtime: object, body: Mapping[str, object],
    ):
        for owner in (runtime, getattr(runtime, "owner", None)):
            if getattr(owner, "current_request_id", None) != capture.request_id:
                raise RequestPreparationContractError("request preparation identity changed before send")
        binding = _current_binding(runtime)
        if getattr(runtime, "prepared_binding", None) != binding:
            raise RequestPreparationContractError("request preparation binding changed before send")
        sources = {str(path): row["content"] for path, row in
                   getattr(runtime, "_empirical_frame_sources", {}).items()
                   if isinstance(row, Mapping) and isinstance(row.get("content"), str)}
        source_plan = getattr(runtime, "_request_source_closure", None)
        request_source_plan = getattr(
            capture, "_request_source_closure_override", None,
        )
        return cls(
            capture.request_id, capture.step, _bytes(body), _bytes(binding),
            _bytes(source_plan),
            _bytes(request_source_plan)
            if isinstance(request_source_plan, Mapping) else None,
            _bytes(sources), _serialized_frame_request_allocation(capture),
            getattr(capture, "qualified_request", None),
        )

    def validate(self, runtime: object, capture: object | None = None) -> None:
        if self.qualified_request is not None and capture is not None:
            if capture.qualified_request is not self.qualified_request:
                raise RequestPreparationContractError("qualified request changed during recovery")
            self.qualified_request.validate(capture, runtime)
        if (_bytes(_current_binding(runtime)) != self.binding
                or _bytes(getattr(runtime, "prepared_binding", None)) != self.binding):
            raise RequestPreparationContractError("request preparation binding changed during recovery")
        current_source_plan = getattr(runtime, "_request_source_closure", None)
        if (_bytes(current_source_plan) != self.source_plan
                and not _source_plan_matches_after_deferred_discharge(
                    self.source_plan, current_source_plan,
                )):
            raise RequestPreparationContractError("original source closure changed during recovery")
        if self.frame_request_allocation is not None:
            try:
                FrameRequestAllocation.from_bytes(self.frame_request_allocation)
            except FrameRequestAllocationError as exc:
                raise RequestPreparationContractError(
                    "captured frame allocation failed integrity validation",
                ) from exc
        if self.source_layout_proposal is not None:
            if self.frame_request_allocation is not None:
                raise RequestPreparationContractError(
                    "source layout proposal conflicts with a sealed allocation",
                )
            try:
                proposed = FrameRequestAllocation.from_bytes(self.source_layout_proposal)
                binding = json.loads(self.binding)
                identity = proposed.identity
                if (identity["request_lineage"] != self.request_id
                        or identity["product_revision"] != str(
                            binding.get("product_revision") or binding.get("artifact")
                        )):
                    raise RequestPreparationContractError(
                        "source layout proposal belongs to another request or revision",
                    )
            except (FrameRequestAllocationError, TypeError, ValueError) as exc:
                raise RequestPreparationContractError(
                    "source layout proposal failed integrity validation",
                ) from exc
        if (capture is not None
                and _serialized_frame_request_allocation(capture)
                != self.frame_request_allocation):
            raise RequestPreparationContractError(
                "request frame allocation changed during recovery",
            )
        # Runtime source entries are a read cache and may be refreshed by the
        # exact-range reads that triggered bounded recovery. Authority remains
        # the freshly checked workspace binding plus unchanged source plan; wire
        # qualification below uses this immutable source snapshot, never cache.


# These fields describe already-completed preparation or the same logical lease.
# Request contracts, receipts, hashes, transport and response state are excluded.
_STABLE = """
_source_replan_inputs _source_replan_attempted
_tokenize_logical_request_id _tokenize_deadline _tokenize_endpoint
lean_request_shape lean_shape_event_id
ticket_identity stateframe_f4_mode stateframe_think_prefill_mode
_qualified_request_gate _qualified_host_text_preparer
request_kind _request_protocol _qualified_adapter
_framefit5_task_message _framefit5_semantic_messages _framefit5_newest_user_suffix
stateframe_think_prefill_text stateframe_think_prefill_tokens
self_plan_wants_plan self_plan_frame self_plan_trigger self_plan_plan_text
self_plan_plan_tokens self_plan_plan_request_id self_plan_plan_accounting self_plan_pair_cache
prompt_cache_arm long_horizon_request_role long_horizon_depth_tracker
long_horizon_logical_depth_tokens long_horizon_cumulative_logical_depth_tokens
long_horizon_cumulative_logical_depth_source long_horizon_context_limit_tokens
history_summary_ordinal frame_epoch_ordinal hybrid_crossed hybrid_crossing_request_id
hybrid_crossing phase presentation cumulative_logical_depth_tokens cumulative_logical_depth_source
long_horizon_history_summary_ordinal long_horizon_frame_epoch_ordinal
long_horizon_hybrid_crossed long_horizon_hybrid_crossing_request_id long_horizon_hybrid_crossing
long_horizon_phase long_horizon_presentation long_horizon_compaction_ordinal
advice_candidates playbook_card_id playbook_card_version playbook_offer_event_id
playbook_adoption_journaled passive_retention_context passive_retention_recorded
model_router model_router_decision model_router_frame model_router_assignment_id
model_router_parent_assignment_id model_router_call_purpose model_router_child_purpose
model_router_host_thinking_off model_router_handoff model_router_prepared
model_router_effective_endpoint model_router_effective_model_id model_router_selected_endpoint
model_router_applied thinking_router thinking_router_classification
completion_owner_runtime presentation_runtime preedit_coverage_runtime
provider_adapter_cold_resume cancel_requested postgreen_done_stripped_by unified_cap_noted
template_conformance_mode template_conformance_runtime template_conformance_renderer
template_conformance_adapter template_conformance_arm
""".split()
_OPAQUE = frozenset("""_qualified_request_gate _qualified_host_text_preparer model_router model_router_decision model_router_frame
thinking_router thinking_router_classification long_horizon_depth_tracker
completion_owner_runtime presentation_runtime preedit_coverage_runtime
template_conformance_runtime template_conformance_renderer""".split())
_REGENERATED = frozenset("""provider_adapter_contract request_action_contract _lean_required_source_paths
_request_measurement _frame_candidate_measurement
_frame_request_allocation_candidate _frame_candidate_inputs _replacement_allocation _replacement_lineage
_visibility_required_paths
release_precomposed_host_text _release_prefix_boundary _release_prefix_replan_used _release_prefix_qualified
_framefit5_owner_presentation_contract
framefit5_product_validation
request_source_final_delivery final_evidence_pin_count final_submitted_bytes_hash
lean_append_only_receipt lean_capacity_outcome
request_budget_final_wire request_budget_boundary_receipt request_owner_first_allocation_plan
request_budget_prepared request_budget_protected_surface request_budget_stage request_budget_exits
request_budget_completion request_budget_final_tools request_budget_final_schema
request_budget_contract_status request_budget_exceeded request_budget_hard_max
request_budget_host_limit request_budget_reservation_order request_budget_reserved_tool_names
request_budget_owner_reexpanded request_source_model_projection_count
model_router_input_capacity model_router_request_estimate tokenlab_layout_tools
long_horizon_rendered_prompt_tokens long_horizon_rendered_prompt_tokens_source
decode_depth_per_arm_rendered_context_depth decode_depth_per_arm_rendered_context_depth_source
decode_depth_per_arm_request_metadata decode_depth_rendered_context_depth
decode_depth_rendered_context_depth_source decode_depth_slots_busy decode_depth_slots_busy_source
decode_depth_per_arm_slots_busy decode_depth_per_arm_slots_busy_source
template_conformance_contract template_conformance_messages template_conformance_tools
preedit_coverage_claim preedit_coverage_status callsite_census_wire_claim
restate_requirement_claim restate_requirement_status
read_contract_wire_capsule read_contract_wire_fields _wire_truth_runtime _wire_truth_side_call
_wire_truth_side_call_contract _request_preparation_input _request_source_closure_override
_stateframe_tool_menu_memo_frame_digest _stateframe_tool_menu_memo_frame_snapshot
_stateframe_tool_menu_memo_key _stateframe_tool_menu_memo_surface_bytes
_stateframe_tool_menu_memo_shadow_match
wire_truth_context _wire_truth_context _wire_truth_check _wire_truth_rebuild_used _transport_submission_started
wire_truth_fact_store wire_truth_done_disposition
""".split())


def _has_deferred_source_paths(source_plan: object) -> bool:
    if not isinstance(source_plan, Mapping):
        return False
    raw_paths = source_plan.get("deferred_paths", ())
    return bool(
        isinstance(raw_paths, Sequence)
        and not isinstance(raw_paths, (str, bytes, bytearray))
        and any(isinstance(path, str) and path.strip() for path in raw_paths)
    )


def _boundary_recovery_menu(
        runtime: object, capture: object, preparation: PreparationInput,
) -> tuple[list[dict[str, object]], str]:
    del capture, preparation
    from kernel_boundary_recovery import recovery_request_menu

    # Preserve the active frozen menu. The source qualifier drops unverifiable
    # claims when this menu has no read route; it must not turn that condition
    # into a failed recovery.
    return recovery_request_menu(runtime, require_read=False)


def _final_boundary_recovery_messages(
        runtime: object, messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Restore the owner pin while keeping deferred-source notice exact."""
    from kernel_refusal_compact import ensure_final_evidence_pin

    prepared, _pin_changed = ensure_final_evidence_pin(runtime, messages)
    if not isinstance(prepared, list):
        raise RequestPreparationContractError(
            "boundary recovery could not render owner evidence pin")
    deferred_paths = getattr(runtime, "_request_deferred_source_paths", ())
    if (isinstance(deferred_paths, Sequence)
            and not isinstance(deferred_paths, (str, bytes, bytearray))
            and deferred_paths):
        from kernel_request_source_window import upsert_deferred_source_notice
        from kernel_workspace_truth import enabled as workspace_truth_enabled
        if workspace_truth_enabled():
            from kernel_read_reuse_visibility import _runtime_workspace_context
            workspace_root, revision, journal = _runtime_workspace_context(runtime)
            upsert_deferred_source_notice(
                prepared, deferred_paths, workspace_root=workspace_root,
                revision=revision, journal=journal,
            )
        else:
            upsert_deferred_source_notice(prepared, deferred_paths)
    return prepared


def prepare_replacement(runtime: object, original: object, preparation: PreparationInput,
                        transport_body: Mapping[str, object], *, deadline: float | None,
                        boundary_recovery: bool = False,
                        boundary_recovery_body: Mapping[str, object] | None = None,
                        endpoint: str | None = None, key: str | None = None):
    """Prove a replacement on detached state, then adopt the entire generation."""
    from kernel_request_candidate import detached_context, runtime_authority_fields
    from kernel_stateframe_runtime import PreparedRequest
    from kernel_request_kind import classify

    preparation.validate(runtime, original)
    if deadline is not None and not math.isfinite(deadline):
        raise RequestPreparationContractError("replacement deadline must be finite")
    prior_deadline = getattr(getattr(original, "qualified_request", None), "deadline", None)
    if prior_deadline is not None:
        # A new generation inherits the original time authority. Callers may
        # tighten it, but cannot remove or extend it by requesting replacement.
        deadline = prior_deadline if deadline is None else min(prior_deadline, deadline)
    from kernel_request_budget import _request_token_meter
    _request_token_meter(runtime, original)
    candidate, context = detached_context(original, runtime)
    candidate._tokenize_logical_request_id = (
        getattr(original, "_tokenize_logical_request_id", None)
        or getattr(original, "_replacement_lineage", None) or getattr(original, "request_id", None)
    )
    candidate._tokenize_deadline = deadline
    # Preserve the original authorization only for the input validation. The
    # private preparer creates a new draft and never inherits wire authority.
    candidate.frame_request_allocation = original.frame_request_allocation
    candidate.qualified_request = original.qualified_request
    context._active_capture = candidate
    prepared = _prepare_replacement_candidate(
        context, candidate, preparation, transport_body, deadline=deadline,
        boundary_recovery=boundary_recovery, boundary_recovery_body=boundary_recovery_body,
    )
    body = {**copy.deepcopy(dict(transport_body)), "messages": list(prepared.messages),
            "tools": list(prepared.tools or ())}
    if classify(candidate, context, body):
        if endpoint is None or key is None:
            raise RequestPreparationContractError("replacement qualification requires the provider endpoint")
        from kernel_final_wire_allocation import build_qualified_request
        candidate._framefit5_pending_qualification = True
        candidate._replacement_allocation = _captured_replacement_source_layout(preparation)
        candidate._replacement_lineage = candidate._tokenize_logical_request_id
        candidate._request_source_closure_override = copy.deepcopy(
            getattr(candidate, "_request_source_closure_override", None)
            or json.loads(preparation.request_source_plan or preparation.source_plan))
        candidate._frame_request_allocation_candidate = copy.deepcopy(
            getattr(original, "_frame_request_allocation_candidate", None))
        if candidate._replacement_allocation is not None:
            from kernel_request_seal import FrameSelectionProposal, freeze
            allocation = candidate._replacement_allocation
            candidate._frame_request_allocation_candidate = FrameSelectionProposal(
                freeze(allocation.optional_heads), freeze(allocation.site_inventory),
                freeze(allocation.capacity))
        record = build_qualified_request(candidate, context, body,
                                         endpoint=endpoint, key=key, deadline=deadline)
        body = record.body
    # Any candidate producer failure above leaves every live alias untouched.
    preparation.validate(runtime, original)
    updates = {name: copy.deepcopy(getattr(context, name))
               for name in (*runtime_authority_fields(context), "_request_deferred_source_paths",
                            "_boundary_recovery_read_admitted") if hasattr(context, name)}
    if hasattr(context, "_request_token_meter"):
        # A detached replacement can create the first meter. Publish the same
        # service that supplied its sealed evidence, without cloning its lease.
        updates["_request_token_meter"] = context._request_token_meter
    candidate.completion_owner_runtime = runtime
    for name in ("presentation_runtime", "preedit_coverage_runtime", "template_conformance_runtime",
                 "_wire_truth_runtime"):
        if getattr(candidate, name, None) is context:
            setattr(candidate, name, runtime)
    original._adopt_reprepared(candidate, runtime, deadline=deadline,
                               expected_owner_request_id=preparation.request_id,
                               boundary_recovery=boundary_recovery)
    runtime.__dict__.update(updates)
    runtime.owner.current_prompt_hash = context.owner.current_prompt_hash
    runtime._active_capture = original
    # The candidate already holds the verified immutable preparation snapshot.
    # Publication performs no further fallible proof or snapshot construction.
    return PreparedRequest(tuple(body["messages"]), original, tools=tuple(body["tools"]))


def _prepare_replacement_candidate(runtime: object, original: object, preparation: PreparationInput,
                        transport_body: Mapping[str, object], *, deadline: float | None,
                        boundary_recovery: bool = False,
                        boundary_recovery_body: Mapping[str, object] | None = None):
    from kernel_measurement import RequestCapture
    from kernel_stateframe_runtime import PreparedRequest
    from kernel_completion_owner import (
        assert_request_action_contract, establish_request_action_contract, _tool_name,
    )
    preparation.validate(runtime, original)
    if (original.request_id != preparation.request_id or original.step != preparation.step
            or getattr(runtime, "_active_capture", None) is not original
            or (getattr(original, "_wire_truth_rebuild_used", False)
                and (not boundary_recovery
                     or getattr(original, "_boundary_recovery_attempted", False)))
            or getattr(original, "_transport_submission_started", False)
            or original.response_event_id is not None or original.transport_attempt_id is not None):
        raise RequestPreparationContractError("replacement requires the original active unsent preparation")
    if deadline is not None and (not math.isfinite(deadline) or deadline <= time.monotonic()):
        raise TimeoutError("request deadline expired before wire recovery")
    for owner in (runtime, runtime.owner):
        if owner.current_request_id != preparation.request_id:
            raise RequestPreparationContractError("replacement owner identity changed")
    prepared_body = json.loads(preparation.body)
    if boundary_recovery:
        # The original contract's integrity and source binding are authority;
        # its serialized menu and message transforms are explicitly not.
        from kernel_completion_owner import _verify_contract_integrity
        if not isinstance(original.request_action_contract, Mapping):
            raise RequestPreparationContractError("boundary recovery has no frozen owner contract")
        _verify_contract_integrity(original.request_action_contract)
        if original.request_action_contract.get("request_id") != original.request_id:
            raise RequestPreparationContractError("boundary recovery owner contract identity changed")
    else:
        # An already-corrupted original contract cannot become authority for a new ID.
        assert_request_action_contract(runtime, prepared_body.get("messages"),
            prepared_body.get("tools"), request_id=original.request_id, boundary="recovery_input",
            frozen_contract=original.request_action_contract)
    original_authority = {name: copy.deepcopy(original.request_action_contract.get(name))
        for name in (("route", "budget_generation", "revision") if boundary_recovery else
                     ("admitted_tool_schemas", "route", "budget_generation", "revision"))}
    original_contract_kind = (
        original.request_action_contract.get("contract_kind")
        if isinstance(original.request_action_contract, Mapping) else None
    )
    fresh = RequestCapture(original.journal, original.step, original.snapshot_id,
        endpoint_profile=original.endpoint_profile, parent_meter=original.parent_meter)
    unknown = set(vars(original)) - set(vars(fresh)) - set(_STABLE) - _REGENERATED
    if unknown:
        raise RequestPreparationContractError("unsupported replacement ownership fields: " + ",".join(sorted(unknown)))
    for name in _STABLE:
        if hasattr(original, name):
            value = getattr(original, name)
            setattr(fresh, name, value if name in _OPAQUE else copy.deepcopy(value))
    fresh._framefit5_pending_qualification = getattr(original, "_framefit5_pending_qualification", False)
    fresh._frame_request_allocation_candidate = copy.deepcopy(
        getattr(original, "_frame_request_allocation_candidate", None))
    # Adoption retires the old capture's allocation. Keep its validated,
    # immutable source layout only for assembly; the replacement must still
    # capture and seal a new allocation before it can be sent.
    replacement_source_layout = _captured_replacement_source_layout(preparation)
    if replacement_source_layout is not None:
        from kernel_request_seal import FrameSelectionProposal, freeze
        prior = replacement_source_layout
        fresh._frame_request_allocation_candidate = FrameSelectionProposal(
            freeze(prior.optional_heads), freeze(prior.site_inventory), freeze(prior.capacity))
    # Use the immutable pre-budget preparation snapshot. In boundary recovery
    # this deliberately discards every late failed-body mutation.
    body = copy.deepcopy(
        dict(boundary_recovery_body) if boundary_recovery and boundary_recovery_body is not None
        else prepared_body if boundary_recovery else dict(transport_body)
    )
    messages = body.get("messages")
    tools = body.get("tools", [])
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise RequestPreparationContractError("replacement input lacks prepared messages/tools")
    budget = getattr(runtime, "continuation_limits", None)
    if budget is None:
        budget = getattr(runtime, "decision_budget", None)
    renderer = getattr(budget, "render_continuation_limits", None)
    if callable(renderer):
        from kernel_stateframe_framefit5 import has_framefit5_state, attach_owner_protocol
        rendered = renderer()
        if has_framefit5_state(messages):
            messages = attach_owner_protocol(
                messages, [{"role": "user", "content": rendered}], runtime,
                original.request_action_contract)
            body["messages"] = messages
        else:
            for row in messages:
                content = row.get("content")
                if isinstance(content, str):
                    row["content"] = "\n".join(rendered if line.startswith(
                        "[CONTINUATION LIMITS |") else line for line in content.split("\n"))
    prepared_claims = {name: copy.deepcopy(getattr(original, name)) for name in (
        "callsite_census_wire_claim", "preedit_coverage_claim", "preedit_coverage_status",
        "restate_requirement_claim", "restate_requirement_status",
        "template_conformance_contract") if hasattr(original, name)}
    menu_authority: str | None = None
    if boundary_recovery:
        tools, menu_authority = _boundary_recovery_menu(
            runtime, original, preparation,
        )
        body["tools"] = copy.deepcopy(tools)
        tools = body["tools"]
        runtime._boundary_recovery_read_admitted = any(
            isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and tool["function"].get("name") == "read"
            for tool in tools
        )

    # These aliases belong solely to the detached transaction. Live adoption
    # occurs in prepare_replacement only after every candidate proof succeeds.
    original._adopt_reprepared(fresh, runtime, deadline=deadline,
                              expected_owner_request_id=preparation.request_id,
                              boundary_recovery=boundary_recovery, publish_event=False)
    # Adoption supersedes the old request's authorization. A producer failure
    # before its store must leave the fresh request unfinalized, not old-authorized.
    runtime._completion_owner_request_contract = None
    if boundary_recovery:
        original._boundary_recovery_original_request_id = preparation.request_id
        original._boundary_recovery_attempted = True
        from kernel_read_reuse_visibility import defer_unproven_source_claims
        deferred = defer_unproven_source_claims(
            original, runtime, preparation, body, boundary_recovery=True,
            menu_authority=menu_authority)
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise RequestPreparationContractError(
                "boundary source deferral produced invalid messages")
        plan = getattr(original, "_request_source_closure_override", None)
        if not isinstance(plan, Mapping):
            try:
                plan = json.loads(preparation.source_plan)
            except (TypeError, ValueError):
                plan = None
        current_deferred = plan.get("deferred_paths", ()) if isinstance(plan, Mapping) else ()
        read_available = bool(getattr(
            runtime, "_boundary_recovery_read_admitted", False,
        )) and any(
            isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and tool["function"].get("name") == "read"
            for tool in tools
        )
        # Only expose a deferred-source claim when this exact replacement menu
        # can dispatch its resolver. The no-read recovery path above removes
        # unverified claims and must not reintroduce them from the plan snapshot.
        runtime._request_deferred_source_paths = tuple(
            str(path) for path in current_deferred
            if isinstance(path, str) and path.strip()
        ) if read_available and isinstance(current_deferred, (list, tuple)) else ()
        if deferred and runtime._request_deferred_source_paths:
            paths = plan.get("deferred_paths", ()) if isinstance(plan, Mapping) else ()
            if (not isinstance(paths, Sequence)
                    or isinstance(paths, (str, bytes, bytearray)) or not paths):
                raise RequestPreparationContractError(
                    "boundary source deferral did not retain its verified paths")
            runtime._request_deferred_source_paths = tuple(str(path) for path in paths)
            if not any(
                    isinstance(tool, Mapping)
                    and isinstance(tool.get("function"), Mapping)
                    and tool["function"].get("name") == "read"
                    for tool in body.get("tools", ())):
                raise RequestPreparationContractError(
                    "boundary source deferral lost its dispatchable read route")
        messages = _final_boundary_recovery_messages(runtime, messages)
        body["messages"] = messages
        if not isinstance(body.get("tools"), list):
            raise RequestPreparationContractError("boundary capability floor is not a tool list")
        floor_names = tuple(
            str(tool.get("function", {}).get("name", ""))
            for tool in body["tools"] if isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
        )
        runtime.completion_menu = floor_names
        runtime.offered_tools = floor_names
        runtime._completion_owner_dispatch_tools = copy.deepcopy(body["tools"])
        runtime._completion_owner_actionable = True
    framefit5_source_messages = copy.deepcopy(messages)
    framefit5_source_plan = getattr(original, "_request_source_closure_override", None)
    if not isinstance(framefit5_source_plan, Mapping):
        raw_source_plan = preparation.request_source_plan or preparation.source_plan
        try:
            framefit5_source_plan = json.loads(raw_source_plan)
        except (TypeError, ValueError):
            framefit5_source_plan = None
    try:
        framefit5_sources = json.loads(preparation.sources)
    except (TypeError, ValueError):
        framefit5_sources = None
    if not isinstance(framefit5_sources, Mapping):
        framefit5_sources = None
    if original_contract_kind == "lean_stable_surface":
        from kernel_completion_owner import establish_lean_request_action_contract
        contract = establish_lean_request_action_contract(
            runtime, messages, tools, request_id=fresh.request_id)
        from kernel_lean import message_digest
        fresh.lean_prepared_history = copy.deepcopy(messages)
        fresh.lean_prepared_history_sha256 = message_digest(fresh.lean_prepared_history)
    else:
        contract = establish_request_action_contract(runtime, messages, tools,
            request_id=fresh.request_id, provider_request=True,
            preserve_input_surface=boundary_recovery)
    if boundary_recovery:
        from kernel_greenlock_menu_authority import (
            applies as greenlock_menu_applies,
            finalize_wire_menu_authority,
        )
        final_names = (
            finalize_wire_menu_authority(
                runtime, tools, preserve_input_surface=boundary_recovery,
            )
            if greenlock_menu_applies(getattr(runtime, "green_lock", None))
            else tuple(_tool_name(tool) for tool in tools if isinstance(tool, Mapping))
        )
        runtime.completion_menu = final_names
        runtime.offered_tools = final_names
        runtime._completion_owner_dispatch_tools = copy.deepcopy(tools)
        runtime._completion_owner_actionable = True
    from kernel_stateframe_framefit5 import (
        assemble_framefit5_request,
        has_framefit5_state,
        validate_product_request,
    )
    framefit5_validation = None
    if has_framefit5_state(messages):
        from kernel_request_seal import OwnerPresentation
        fresh.owner_presentation = OwnerPresentation.capture(runtime, contract, messages)
        messages = assemble_framefit5_request(
            messages,
            framefit5_source_messages,
            task_message=getattr(original, "_framefit5_task_message", None),
            source_plan=(
                framefit5_source_plan
                if isinstance(framefit5_source_plan, Mapping) else None
            ),
            sources=framefit5_sources,
            runtime=runtime,
            owner_contract=contract,
            owner_source_messages=framefit5_source_messages,
            owner_presentation=fresh.owner_presentation,
            frame_request_allocation=replacement_source_layout,
        )
        body["messages"] = messages
        if boundary_recovery:
            messages = _final_boundary_recovery_messages(runtime, messages)
            # The final pin and deferred-source notice are late renderers. Rebuild
            # from the immutable preparation snapshot so their output cannot
            # bypass FrameFit5 source qualification.
            messages = assemble_framefit5_request(
                messages,
                framefit5_source_messages,
                task_message=getattr(original, "_framefit5_task_message", None),
                source_plan=(
                    framefit5_source_plan
                    if isinstance(framefit5_source_plan, Mapping) else None
                ),
                sources=framefit5_sources,
                runtime=runtime,
                owner_contract=contract,
                owner_source_messages=framefit5_source_messages,
                owner_presentation=fresh.owner_presentation,
                frame_request_allocation=replacement_source_layout,
            )
            body["messages"] = messages
        framefit5_validation = validate_product_request(
            messages, runtime, tools=tools,
            frame_request_allocation=fresh.frame_request_allocation,
            refresh_headcarry=False,
            frame_selection=getattr(fresh, "_frame_request_allocation_candidate", None),
        )
        if isinstance(framefit5_source_plan, Mapping):
            original._request_source_closure_override = copy.deepcopy(
                dict(framefit5_source_plan),
            )
    changed_authorities = [
        name for name, value in original_authority.items()
        if _bytes(contract.get(name)) != _bytes(value)
    ]
    if changed_authorities:
        raise RequestPreparationContractError(
            "replacement changed the original admitted route or generation: "
            + ", ".join(changed_authorities)
        )
    deferred_paths = getattr(runtime, "_request_deferred_source_paths", ())
    if (isinstance(deferred_paths, Sequence)
            and not isinstance(deferred_paths, (str, bytes, bytearray))
            and deferred_paths and not any(
                isinstance(tool, Mapping)
                and isinstance(tool.get("function"), Mapping)
                and tool["function"].get("name") == "read"
                for tool in tools
            )):
        raise RequestPreparationContractError(
            "fresh owner contract removed the read route for deferred sources")
    fresh.request_action_contract = copy.deepcopy(contract)
    if isinstance(framefit5_validation, Mapping):
        fresh.framefit5_product_validation = copy.deepcopy(dict(framefit5_validation))
    assert_request_action_contract(runtime, messages, tools, request_id=fresh.request_id,
        boundary="replacement_prepared", frozen_contract=contract)
    if getattr(runtime, "stateframe_provider_adapter_mode", "off") != "off":
        from kernel_provider_adapter import validate_prepared_request
        fresh.provider_adapter_contract = validate_prepared_request(messages, tools,
            template_identity=runtime.journal.manifest.get("chat_template"),
            continuation_enabled=runtime.continuation is not None,
            native_tool_batches=bool(runtime.native_tool_batches),
            native_prefix_authority=bool(runtime.native_prefix_authority),
            cold_resume=bool(getattr(original, "provider_adapter_cold_resume", False)),
            restart_enabled=bool(runtime.restart_enabled))
    # Census payload and coverage exposure are revision-bound prepared inputs,
    # not submission receipts. Their existing submitted-wire verifiers recheck
    # actual bytes and produce the new capture's durable evidence.
    for name in ("callsite_census_wire_claim", "preedit_coverage_claim", "preedit_coverage_status",
                 "restate_requirement_claim", "restate_requirement_status"):
        if name in prepared_claims:
            setattr(fresh, name, prepared_claims[name])
    template = prepared_claims.get("template_conformance_contract")
    if isinstance(template, Mapping):
        from kernel_template_conformance import build_contract
        fresh.template_conformance_contract = build_contract(messages, tools,
            arm=template["arm"], endpoint=template["endpoint"], model_id=template["model_id"],
            adapter=template["adapter"], request_id=fresh.request_id, step=fresh.step,
            hop=bool(template.get("hop")), resumed=bool(template.get("resumed")))
        fresh.template_conformance_messages = copy.deepcopy(messages)
        fresh.template_conformance_tools = copy.deepcopy(tools)
    from kernel_measurement import digest
    runtime.owner.current_prompt_hash = digest({"messages": messages, "tools": tools})
    runtime.journal.emit("request_prepared", request_id=fresh.request_id,
        step=fresh.step, replacement_of=preparation.request_id,
        context_hash=runtime.owner.current_prompt_hash,
        original_source_closure_sha256="sha256:" + hashlib.sha256(preparation.source_plan).hexdigest(),
        provider_request=True, dispatch=False)
    return PreparedRequest(tuple(messages), original, tools=tuple(tools))


def _source_delivery_only_budget_blocker(
        budget: Mapping[str, object], prior_delivery: object, request_bytes: int,
) -> bool:
    from kernel_request_source_window import source_delivery_satisfied

    if (budget.get("status") != "HOST_BLOCKED"
            or not isinstance(prior_delivery, Mapping)
            or source_delivery_satisfied(prior_delivery)
            or budget.get("messages") is not True
            or budget.get("tools") is not True):
        return False
    failures = budget.get("final_validation_failures")
    if (not isinstance(failures, Sequence)
            or isinstance(failures, (str, bytes, bytearray))
            or "source_delivery_unqualified" not in failures
            or any(item != "source_delivery_unqualified" for item in failures)):
        return False
    if "pass_gates" in budget:
        gates = budget.get("pass_gates")
        if not isinstance(gates, Mapping) or gates.get("status") != "VALID":
            return False
    if "max_request_bytes" in budget:
        maximum = budget.get("max_request_bytes")
        if type(maximum) is not int or maximum < 1 or request_bytes > maximum:
            return False
    return True


def _record_final_budget_source_delivery(
        capture: object, delivery: Mapping[str, object], body: Mapping[str, object],
        wire: bytes,
) -> None:
    """Rebind the provisional budget receipt to exact serialized provider bytes."""
    from kernel_request_source_window import source_delivery_satisfied

    budget = getattr(capture, "request_budget_final_wire", None)
    messages = body.get("messages")
    if (not isinstance(budget, Mapping) or not isinstance(budget.get("source_delivery"), Mapping)
            or not isinstance(messages, list) or not source_delivery_satisfied(delivery)):
        return
    wire_digest = "sha256:" + hashlib.sha256(wire).hexdigest()
    context_bytes = json.dumps(
        {"messages": messages}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    context_digest = "sha256:" + hashlib.sha256(context_bytes).hexdigest()
    if (delivery.get("transport_identity") != "VERIFIED"
            or delivery.get("submitted_http_bytes_digest") != wire_digest
            or delivery.get("context_digest") != context_digest
            or delivery.get("submitted_payload_hash") != context_digest):
        return

    updated = copy.deepcopy(dict(budget))
    updated.update({
        "phase": "final_transport_revalidation",
        "request_bytes": len(wire),
        "context_digest": context_digest,
        "submitted_http_bytes_digest": wire_digest,
        "source_delivery": copy.deepcopy(dict(delivery)),
    })
    failures = updated.get("final_validation_failures")
    if (isinstance(failures, Sequence)
            and not isinstance(failures, (str, bytes, bytearray))):
        updated["final_validation_failures"] = [
            item for item in failures if item != "source_delivery_unqualified"
        ]
    if _source_delivery_only_budget_blocker(budget, budget.get("source_delivery"), len(wire)):
        updated["status"] = "READY"
    if updated == dict(budget):
        return

    capture.request_budget_final_wire = updated
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if callable(emit):
        from kernel_continuation_journal import journal_safe_fields
        emit(
            "request_budget_final_wire", producer="kernel_request_reprepare",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None), dispatch=False,
            **journal_safe_fields(updated),
        )


def _normalized_plan_deferred_paths(plan: Mapping[str, object]) -> tuple[str, ...]:
    raw_paths = plan.get("deferred_paths", ())
    if not isinstance(raw_paths, Sequence) or isinstance(
            raw_paths, (str, bytes, bytearray)):
        return ()
    return tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in raw_paths if isinstance(path, str) and path.strip()
    ))


def _plan_with_capture_deferrals(
        capture: object, plan: Mapping[str, object],
        request_plan_snapshot: Mapping[str, object] | None = None,
) -> dict[str, object]:
    merged = copy.deepcopy(dict(plan))
    captured = getattr(capture, "_request_source_closure_override", None)
    deferred_paths = list(
        _normalized_plan_deferred_paths(captured)
        if isinstance(captured, Mapping) else ()
    )
    deferred_paths.extend(_normalized_plan_deferred_paths(merged))
    if isinstance(request_plan_snapshot, Mapping):
        deferred_paths.extend(_normalized_plan_deferred_paths(request_plan_snapshot))
    if deferred_paths:
        merged["deferred_paths"] = list(dict.fromkeys(deferred_paths))
    return merged


def _verified_deferred_paths(
        delivery: Mapping[str, object], deferred_paths: Sequence[str],
) -> tuple[str, ...] | None:
    raw_verified = delivery.get("deferred_paths")
    if (not isinstance(raw_verified, Sequence)
            or isinstance(raw_verified, (str, bytes, bytearray))):
        return None
    verified_paths = tuple(
        str(path).replace("\\", "/").strip()
        for path in raw_verified if isinstance(path, str) and path.strip()
    )
    verified_set = set(verified_paths)
    deferred_set = set(deferred_paths)
    path_results = delivery.get("paths")
    path_results = path_results if isinstance(path_results, Mapping) else {}
    if (
        delivery.get("proof_status") != "VERIFIED"
        or str(delivery.get("status", "")).upper() not in {"COMPLETE", "DEFERRED"}
        or len(verified_paths) != len(raw_verified)
        or len(verified_set) != len(verified_paths)
        or not verified_set.issubset(deferred_set)
    ):
        return None
    for path in verified_set:
        result = path_results.get(path)
        if (not isinstance(result, Mapping)
                or str(result.get("status", "")).upper() != "DEFERRED"
                or str(result.get("proof_status", "")).upper() != "VERIFIED"):
            return None
    for path in deferred_set - verified_set:
        result = path_results.get(path)
        if (not isinstance(result, Mapping)
                or str(result.get("status", "")).upper() != "COMPLETE"
                or str(result.get("proof_status", "")).upper() != "VERIFIED"):
            return None
    return verified_paths


def _persist_verified_request_source_plan(
        capture: object, plan: Mapping[str, object],
        delivery: Mapping[str, object],
) -> None:
    deferred_paths = _normalized_plan_deferred_paths(plan)
    if not deferred_paths:
        from kernel_request_source_window import source_delivery_satisfied
        if not source_delivery_satisfied(delivery):
            raise RequestPreparationContractError("cannot persist unverified complete source plan")
        capture._request_source_closure_override = copy.deepcopy(dict(plan))
        return
    raw_verified = delivery.get("deferred_paths")
    if (not isinstance(raw_verified, Sequence)
            or isinstance(raw_verified, (str, bytes, bytearray))):
        error = RequestPreparationContractError(
            "verified source delivery omitted its deferred path set")
        error.failures = ("source_deferred_path_set_omitted",)
        raise error
    verified_paths = _verified_deferred_paths(delivery, deferred_paths)
    if verified_paths is None:
        error = RequestPreparationContractError(
            "verified source delivery changed the request's deferred path set")
        error.failures = ("source_deferred_path_set_mismatch",)
        raise error
    persisted = copy.deepcopy(dict(plan))
    # A previously deferred path leaves this set only after the exact final
    # receipt proves COMPLETE; still-deferred paths must match it exactly.
    persisted["deferred_paths"] = list(verified_paths)
    # Frozen budget revalidation reads this capture override (or the runtime
    # closure), not PreparationInput.source_plan. Persist only after the exact
    # wire has proved every remaining deferral.
    setattr(capture, "_request_source_closure_override", persisted)


def _source_messages_from_preparation(
        preparation: PreparationInput | None,
        fallback: Sequence[Mapping[str, object]],
) -> Sequence[Mapping[str, object]]:
    if preparation is None:
        return fallback
    try:
        body = json.loads(preparation.body)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RequestPreparationContractError(
            "immutable request preparation body is unavailable"
        ) from error
    messages = body.get("messages") if isinstance(body, Mapping) else None
    if (not isinstance(messages, list)
            or any(not isinstance(message, Mapping) for message in messages)):
        raise RequestPreparationContractError(
            "immutable request preparation has no message list"
        )
    return messages


def _source_delivery_inputs(
        capture: object, runtime: object, preparation: PreparationInput | None,
) -> tuple[Mapping[str, object] | None, Mapping[str, str] | None]:
    if preparation is not None:
        preparation.validate(runtime)
        prepared_plan = json.loads(preparation.source_plan)
        raw_request_plan = getattr(preparation, "request_source_plan", None)
        try:
            request_plan = (
                json.loads(raw_request_plan)
                if isinstance(raw_request_plan, (bytes, str)) else None
            )
        except (TypeError, ValueError):
            request_plan = None
        plan = getattr(capture, "_request_source_closure_override", None)
        has_capture_override = isinstance(plan, Mapping)
        if not has_capture_override:
            # The immutable preparation plan validates recovery; current runtime
            # authority supplies the plan when this capture has no own override.
            plan = getattr(runtime, "_request_source_closure", None)
        if not isinstance(plan, Mapping) and isinstance(prepared_plan, Mapping):
            plan = prepared_plan
        if isinstance(plan, Mapping):
            # Only an immutable plan explicitly captured from this request's
            # override can supplement its current authority; the runtime-global
            # PreparationInput plan is validation evidence, never a supplement.
            same_request_snapshot = (
                getattr(preparation, "request_id", None)
                == getattr(capture, "request_id", None)
            )
            supplement = (
                request_plan if has_capture_override and same_request_snapshot
                and isinstance(request_plan, Mapping) else None
            )
            plan = _plan_with_capture_deferrals(capture, plan, supplement)
        sources = json.loads(preparation.sources)
        return plan, sources if isinstance(sources, Mapping) else None

    plan = getattr(capture, "_request_source_closure_override", None)
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        return None, None
    plan = _plan_with_capture_deferrals(capture, plan)
    raw_sources = getattr(runtime, "_empirical_frame_sources", {})
    sources = {
        str(path): row["content"] for path, row in raw_sources.items()
        if isinstance(row, Mapping) and isinstance(row.get("content"), str)
    } if isinstance(raw_sources, Mapping) else {}
    return plan, sources or None


SOURCE_REQUALIFY_MODE_ENV = "STATEFRAME_SOURCE_REQUALIFY"
_SOURCE_COVERAGE_LINE = re.compile(
    r"(?m)^Coverage: lines (\d+)-(\d+), columns 0-0; "
    r"current bytes are delivered for this range\.$"
)


def source_requalify_enabled(value: object = None) -> bool:
    raw = os.environ.get(SOURCE_REQUALIFY_MODE_ENV, "off") if value is None else value
    if isinstance(raw, Mapping):
        raw = raw.get(SOURCE_REQUALIFY_MODE_ENV, "off")
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{SOURCE_REQUALIFY_MODE_ENV} must be off or on")
    return raw.strip().lower() == "on"


def _source_requalification_rows(
        value: object, *, allow_empty: bool = False,
) -> list[dict[str, object]] | None:
    if value is None and allow_empty:
        return []
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or (not value and not allow_empty)
            or any(not isinstance(row, Mapping) for row in value)):
        return None
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _source_mismatch_candidates(
        omissions: Sequence[Mapping[str, object]],
        failures: Sequence[Mapping[str, object]],
) -> list[dict[str, object]] | None:
    mismatches = [dict(row) for row in omissions
                  if row.get("kind") == "source_range_bytes_mismatch"]
    if not mismatches or len(mismatches) != len(omissions):
        return None
    paths = {row.get("path") for row in mismatches}
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        return None
    mismatch_failures = [row for row in failures
                         if row.get("kind") == "source_range_bytes_mismatch"]
    final_failures = [row for row in failures
                      if row.get("kind") in {
                          "final_wire_proof_missing", "final_wire_proof_incomplete",
                      }]
    if (mismatch_failures != mismatches
            or any(row.get("path") not in paths for row in final_failures)
            or len(mismatch_failures) + len(final_failures) != len(failures)):
        return None
    for row in final_failures:
        expected = row.get("expected")
        actual = row.get("actual")
        if (type(expected) is not int or expected < 1
                or type(actual) is not int or actual < 0 or actual >= expected):
            return None
    return mismatches


def _missing_source_candidates(
        omissions: Sequence[Mapping[str, object]],
        failures: Sequence[Mapping[str, object]],
) -> list[dict[str, object]] | None:
    if omissions:
        return None
    missing = [dict(row) for row in failures
               if row.get("kind") == "final_wire_proof_missing"]
    if not missing or len(missing) != len(failures):
        return None
    paths = [row.get("path") for row in missing]
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        return None
    if len(set(paths)) != len(paths):
        return None
    for row in missing:
        expected = row.get("expected")
        actual = row.get("actual")
        if (type(expected) is not int or expected < 1
                or type(actual) is not int or actual != 0):
            return None
    return missing


def _source_requalification_candidates(
        delivery: Mapping[str, object],
) -> list[dict[str, object]] | None:
    omissions = _source_requalification_rows(
        delivery.get("integrity_omissions"), allow_empty=True,
    )
    failures = _source_requalification_rows(delivery.get("proof_failures"))
    if omissions is None or failures is None:
        return None
    return (
        _source_mismatch_candidates(omissions, failures)
        or _missing_source_candidates(omissions, failures)
    )


def _requalification_source_block(
        content: str, path: str, revision: str, root: Path,
        known_source: str,
) -> tuple[str, tuple[int, int]] | None:
    from kernel_f4_source_ranges import exact_current_source_span
    from kernel_request_source_window import source_revision

    header = re.match(
        rf"^\[(?:Current|Working) source:\s*{re.escape(path)}\]\n",
        content,
    )
    coverage = _SOURCE_COVERAGE_LINE.search(content)
    if header is None or coverage is None or coverage.end() != len(content):
        return None
    start, end = int(coverage.group(1)), int(coverage.group(2))
    if start < 1 or end < start or source_revision(known_source) != revision:
        return None
    exact = exact_current_source_span(root, path, revision, start, end)
    if exact is None:
        return None
    known_lines = known_source.splitlines(keepends=True)
    if exact != "".join(known_lines[start - 1:end]):
        return None
    if not exact.endswith(("\n", "\r")):
        exact += "\n"
    replacement = content[:header.end()] + exact + content[coverage.start():]
    return replacement, (start, end)


def _source_requalification_attempt(
        capture: object, runtime: object, plan: Mapping[str, object],
        sources: Mapping[str, str] | None, body: Mapping[str, object], wire: bytes,
        delivery: Mapping[str, object],
) -> tuple[dict[str, object], bytes, dict[str, object], Mapping[str, object], list[tuple[str, str]]] | None:
    from pathlib import Path
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES
    from kernel_request_source_window import (
        record_deferred_source_paths, source_delivery_satisfied,
        verify_final_source_delivery,
    )

    if len(wire) > int(HARD_MAX_REQUEST_BYTES):
        return None
    candidates = _source_requalification_candidates(delivery)
    if candidates and all(
            row.get("kind") == "final_wire_proof_missing" for row in candidates):
        from kernel_request_source_requalify import attempt_missing_source_requalification
        return attempt_missing_source_requalification(
            runtime, plan, sources, body, wire, candidates,
        )
    messages = body.get("messages")
    revisions = plan.get("revisions")
    units = plan.get("units")
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if (not candidates or not isinstance(messages, list)
            or not isinstance(revisions, Mapping)
            or not isinstance(units, Sequence)
            or isinstance(units, (str, bytes, bytearray))
            or root_value is None):
        return None
    selected_paths = {
        str(unit.get("path", "")).replace("\\", "/")
        for unit in units if isinstance(unit, Mapping) and str(unit.get("path", "")).strip()
    }
    try:
        root = Path(root_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    source_map = sources if isinstance(sources, Mapping) else {}
    repair_rows: list[tuple[str, int, int, int]] = []
    seen: set[tuple[str, int, int, int]] = set()
    for row in candidates:
        path = str(row.get("path", "")).replace("\\", "/")
        message_index = row.get("message")
        revision = revisions.get(path)
        path_parts = path.split("/")
        if (not path or path.startswith("/")
                or any(part in {"", ".", ".."} or part.casefold() == "backups"
                       for part in path_parts)
                or "\x00" in path or "\n" in path or "\r" in path
                or path not in selected_paths or type(message_index) is not int
                or not 0 <= message_index < len(messages)
                or not isinstance(revision, str) or not revision):
            return None
        message = messages[message_index]
        content = message.get("content") if isinstance(message, Mapping) else None
        known_source = source_map.get(path)
        if not isinstance(content, str) or not isinstance(known_source, str):
            return None
        rendered = _requalification_source_block(content, path, revision, root, known_source)
        if rendered is None:
            match = _SOURCE_COVERAGE_LINE.search(content)
            if match is None:
                return None
            span = (int(match.group(1)), int(match.group(2)))
        else:
            span = rendered[1]
        key = (path, message_index, span[0], span[1])
        if key not in seen:
            seen.add(key)
            repair_rows.append(key)

    candidate_body = copy.deepcopy(dict(body))
    candidate_messages = candidate_body.get("messages")
    if not isinstance(candidate_messages, list):
        return None
    repairs: list[tuple[str, int, int, int]] = []
    for path, message_index, start, end in repair_rows:
        content = candidate_messages[message_index].get("content")
        known_source = source_map.get(path)
        revision = revisions[path]
        if not isinstance(content, str) or not isinstance(known_source, str):
            return None
        rendered = _requalification_source_block(content, path, revision, root, known_source)
        if rendered is None:
            continue
        candidate_messages[message_index] = {
            **candidate_messages[message_index], "content": rendered[0],
        }
        repairs.append((path, message_index, start, end))

    candidate_wire = json.dumps(candidate_body).encode()
    if len(candidate_wire) <= int(HARD_MAX_REQUEST_BYTES) and repairs:
        working_plan = copy.deepcopy(dict(plan))
        qualified = verify_final_source_delivery(
            working_plan, candidate_messages, payload=candidate_wire, sources=source_map,
        )
        if source_delivery_satisfied(qualified):
            return (
                candidate_body, candidate_wire, working_plan, qualified,
                [(path, "rerendered_live_range") for path, _index, _start, _end in repairs],
            )

    # A verified read route is the only safe fallback when the live revision cannot
    # reproduce a clipped range. Remove the source assertion and state what was omitted.
    fallback_body = copy.deepcopy(dict(body))
    fallback_messages = fallback_body.get("messages")
    tools = fallback_body.get("tools")
    if not isinstance(fallback_messages, list):
        return None
    read_available = False
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        for tool in tools:
            function = tool.get("function") if isinstance(tool, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else (
                tool.get("name") if isinstance(tool, Mapping) else None
            )
            if name == "read":
                read_available = True
                break
    if not read_available:
        return None
    fallback_plan = copy.deepcopy(dict(plan))
    host_facts: list[str] = []
    deferred_paths = tuple(dict.fromkeys(path for path, _i, _a, _b in repair_rows))
    for path, message_index, start, end in repair_rows:
        message = fallback_messages[message_index]
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or message.get("role") not in {"user", "tool"}:
            return None
        route = (
            f"read(path={json.dumps(path, ensure_ascii=False)}, "
            f"start_line={start}, end_line={end})"
        )
        host_facts.append(f"[Host fact] {path} lines {start}-{end} omitted; {route} loads them.")
        header = re.match(
            rf"^\[(?:Current|Working) source:\s*{re.escape(path)}\]\n", content,
        )
        coverage = _SOURCE_COVERAGE_LINE.search(content)
        if header is None or coverage is None:
            return None
        sanitized_content = content[:header.start()] + content[coverage.end():]
        if message.get("role") == "tool" and not sanitized_content:
            sanitized_content = "Source range omitted from this tool payload; see the host fact read route."
        fallback_messages[message_index] = {**message, "content": sanitized_content}
    record_deferred_source_paths(
        fallback_plan, fallback_messages, deferred_paths,
    )
    if host_facts:
        fallback_messages.append({
            "role": "user",
            "content": "\n".join(host_facts),
            "_request_budget_section": "stage_evidence",
        })
    fallback_wire = json.dumps(fallback_body).encode()
    if len(fallback_wire) > int(HARD_MAX_REQUEST_BYTES):
        return None
    qualified = verify_final_source_delivery(
        fallback_plan, fallback_messages, payload=fallback_wire, sources=source_map,
    )
    if not source_delivery_satisfied(qualified):
        return None
    return (
        fallback_body, fallback_wire, fallback_plan, qualified,
        [(path, "deferred_to_verified_read") for path in deferred_paths],
    )


def qualify_source(capture: object, runtime: object, preparation: PreparationInput | None,
                   body: Mapping[str, object], wire: bytes, *,
                   allow_requalification: bool = False,
                   framefit5_preassembled: bool = False,
                   framefit5_final_assembly_pending: bool = False) -> bytes:
    if runtime is None:
        return wire
    if (isinstance(body, dict)
            and not framefit5_preassembled
            and not framefit5_final_assembly_pending):
        messages = body.get("messages")
        if isinstance(messages, list):
            from kernel_stateframe_framefit5 import (
                has_framefit5_state,
                refresh_final_product_inputs,
            )
            if has_framefit5_state(messages):
                refresh_final_product_inputs(
                    runtime, messages,
                    frame_request_allocation=getattr(capture, "frame_request_allocation", None),
                )
                wire = json.dumps(body).encode()
    plan, sources = _source_delivery_inputs(capture, runtime, preparation)
    if preparation is None and not isinstance(plan, Mapping):
        return wire
    if isinstance(plan, Mapping):
        from kernel_request_source_window import (
            source_delivery_satisfied, verify_final_source_delivery,
        )
        delivery = verify_final_source_delivery(
            plan, body.get("messages", []), payload=wire, sources=sources,
        )
        if not source_delivery_satisfied(delivery) and allow_requalification:
            try:
                requalification_on = source_requalify_enabled()
            except ValueError:
                requalification_on = False
            budget = getattr(capture, "request_budget_final_wire", None)
            budget_delivery = (
                budget.get("source_delivery") if isinstance(budget, Mapping) else None
            )
            sole_source_blocker = (
                isinstance(budget, Mapping)
                and _source_delivery_only_budget_blocker(
                    budget, budget_delivery, len(wire),
                )
            )
            if requalification_on and sole_source_blocker and isinstance(body, dict):
                attempt = _source_requalification_attempt(
                    capture, runtime, plan, sources, body, wire, delivery,
                )
                if attempt is not None:
                    repaired_body, repaired_wire, repaired_plan, delivery, actions = attempt
                    body.clear()
                    body.update(repaired_body)
                    wire = repaired_wire
                    plan = repaired_plan
                    for _path, action in actions:
                        capture.journal.emit(
                            "source_requalified",
                            producer="kernel_request_reprepare",
                            request_id=getattr(capture, "request_id", None),
                            action=action, dispatch=False,
                        )
                    if any(action in {
                            "deferred_to_verified_read",
                            "requirement_dropped_with_read_route",
                    } for _path, action in actions):
                        setattr(
                            capture, "_request_source_closure_override",
                            copy.deepcopy(repaired_plan),
                        )
                if (not source_delivery_satisfied(delivery)
                        and not framefit5_final_assembly_pending):
                    capture.journal.emit(
                        "fit_false",
                        producer="kernel_request_reprepare",
                        request_id=getattr(capture, "request_id", None),
                        reason="source_requalification_unverified",
                        vital=True, dispatch=False,
                    )
        original_request_id = getattr(capture, "request_id", None)
        if preparation is not None:
            boundary_origin = getattr(
                capture, "_boundary_recovery_original_request_id", None,
            )
            original_request_id = (
                boundary_origin if isinstance(boundary_origin, str)
                else preparation.request_id
            )
        if (not source_delivery_satisfied(delivery)
                and not framefit5_final_assembly_pending):
            raise RequestPreparationContractError(
                "final request source delivery is not wire-verified")
        if isinstance(body, dict):
            messages = body.get("messages")
            if isinstance(messages, list):
                from kernel_stateframe_framefit5 import (
                    assemble_framefit5_request,
                    has_framefit5_state,
                    validate_final_product_boundary,
                )
                if has_framefit5_state(messages):
                    if framefit5_preassembled:
                        validate_final_product_boundary(
                            runtime, messages, body.get("tools"),
                            refresh_sources=False,
                            refresh_headcarry=False,
                            frame_request_allocation=getattr(capture, "frame_request_allocation", None),
                        )
                    elif not framefit5_final_assembly_pending:
                        normalized_messages = assemble_framefit5_request(
                            messages,
                            messages,
                            task_message=getattr(capture, "_framefit5_task_message", None),
                            source_plan=plan if isinstance(plan, Mapping) else None,
                            sources=sources,
                            runtime=runtime,
                            owner_contract=getattr(capture, "request_action_contract", None),
                            frame_request_allocation=getattr(
                                capture, "frame_request_allocation", None,
                            ),
                        )
                        body["messages"] = normalized_messages
                        validate_final_product_boundary(
                            runtime, normalized_messages, body.get("tools"),
                            refresh_sources=False,
                            refresh_headcarry=False,
                            frame_request_allocation=getattr(capture, "frame_request_allocation", None),
                        )
                        wire = json.dumps(body).encode()
                        delivery = verify_final_source_delivery(
                            plan, normalized_messages, payload=wire, sources=sources,
                        )
                        if not source_delivery_satisfied(delivery):
                            capture.journal.emit(
                                "fit_false",
                                producer="kernel_request_reprepare",
                                request_id=getattr(capture, "request_id", None),
                                reason="source_requalification_unverified_after_framefit5_wire_order",
                                vital=True, dispatch=False,
                            )
                            raise RequestPreparationContractError(
                                "final source delivery failed after frame-only wire ordering"
                            )
        if not framefit5_final_assembly_pending:
            capture.request_source_final_delivery = delivery
            capture.journal.emit("request_source_final_wire", request_id=capture.request_id,
                original_request_id=original_request_id,
                body_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
                source_delivery=delivery, dispatch=False)
            _persist_verified_request_source_plan(capture, plan, delivery)
            _record_final_budget_source_delivery(capture, delivery, body, wire)
    if not framefit5_final_assembly_pending:
        capture.final_submitted_bytes_hash = "sha256:" + hashlib.sha256(wire).hexdigest()
        from kernel_stateframe_runtime import token_estimate
        estimate = token_estimate(body)
        capture.long_horizon_rendered_prompt_tokens = estimate
        capture.long_horizon_rendered_prompt_tokens_source = "final_wire_token_estimate"
        if getattr(capture, "model_router", None) is not None:
            capture.model_router_request_estimate = estimate
        capture.final_evidence_pin_count = sum(str(row.get("content", "")).count("[EVIDENCEPIN]")
            for row in body.get("messages", []) if isinstance(row, Mapping))
    return wire
