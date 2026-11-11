"""Explicit host restart at durable, public decision boundaries.

This profile is deliberately fail-closed. A checkpoint is not permission to retry
an ambiguous effect or reset an activated holder whose replay is unsupported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kernel_history_spool import HistorySpool, history
from kernel_history_receipts import receipts as receipt_history

from kernel_continuation_workspace import ContinuationWorkspace
from kernel_measurement import canonical_bytes, digest
from kernel_evictfix_mode import enabled as evictfix_enabled
from kernel_stateframe_resume import (
    ResumeError,
    append_follow_up,
    capture_owner_restore,
    checkpoint_conversation,
    latest_checkpoint,
    require_snapshot,
    restore_conversation,
    restore_owner_state,
    restore_tail_follow_up,
    user_request_event,
    validate_owner_restore,
    workspace_revision_matches,
)

SCHEMA = "continuation-worker-checkpoint/4"
SLIM_LEDGER_SCHEMA = "continuation-worker-slim-ledger/2"
_SLIM_LEDGER_SCHEMAS = {SLIM_LEDGER_SCHEMA, "continuation-worker-slim-ledger/1"}
FIRST_GREEN_CHECKPOINT_KEY = "first_green_disclosure"
FIRST_GREEN_CHECKPOINT_SCHEMA = "stateframe-first-green-disclosure-checkpoint/1"
_FGD_CHECKPOINT_FIELDS = (
    "_fgd_challenge_revision", "_fgd_progress", "_fgd_read_paths",
    "_fgd_probe_fingerprints", "_fgd_identical_reruns", "_fgd_disclosed_revisions",
)
_DONE_CHALLENGE_CHECKPOINT_FIELDS = (
    "_done_challenge_revision", "_done_challenge_count",
    "_done_challenge_overrides", "_done_submission_record",
)
WORKER_FIELDS = (
    "edits", "tests", "unparsed_streak", "harness_suspect", "last_line",
    "same_streak", "run_temp", "edits_since_test", "baseline_tests",
    "created_this_run", "last_surgery_step", "tool_calls_seen", "decoy_call_names",
    "schema_violation_details", "unknown_tool_names", "first_edit_step",
    "wrong_tool_before_first_edit",
)
_HOST_PROGRESS_WORKER_FIELD = "host_progress_facts"
_HOST_PROGRESS_ADVISORY_FIELD = "zero_progress_advised_checkpoints"


def _valid_host_progress_facts(facts: object) -> bool:
    if not isinstance(facts, Mapping):
        return False
    if facts.get("schema") != "kernel-host-progress-facts/1" or facts.get("source") != "host":
        return False
    for name in ("edits", "checks", "product_edits", "edits_since_check"):
        if type(facts.get(name)) is not int or facts[name] < 0:
            return False
    for name in ("changed_files", "edit_receipt_ids", "check_receipt_ids"):
        rows = facts.get(name)
        if not isinstance(rows, (tuple, list)) or any(not isinstance(row, str) for row in rows):
            return False
    first_edit_step = facts.get("first_edit_step")
    return (first_edit_step is None
            or type(first_edit_step) is int and first_edit_step >= 0)


def _valid_advised_checkpoints(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) <= 3
        and all(type(step) is int and step in {50, 100, 150} for step in value)
        and len(set(value)) == len(value)
    )


def _valid_worker_state_fields(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if "unbounded_progress_watch" in value:
        from kernel_tool_call_loop_breaker import UnboundedProgressWatch
        try:
            watch = UnboundedProgressWatch.from_snapshot(value["unbounded_progress_watch"])
        except ValueError:
            return False
        if watch.threshold != 32:
            return False
        value = {key: child for key, child in value.items() if key != "unbounded_progress_watch"}
    base = set(WORKER_FIELDS)
    extra = set(value) - base
    if not extra:
        return set(value) == base
    allowed_extra = {
        frozenset({_HOST_PROGRESS_WORKER_FIELD}),
        frozenset({_HOST_PROGRESS_WORKER_FIELD, _HOST_PROGRESS_ADVISORY_FIELD}),
    }
    if frozenset(extra) not in allowed_extra or set(value) != base | extra:
        return False
    from kernel_progress_facts import host_truth_progress_enabled
    if not host_truth_progress_enabled():
        return False
    return (
        _valid_host_progress_facts(value[_HOST_PROGRESS_WORKER_FIELD])
        and (_HOST_PROGRESS_ADVISORY_FIELD not in extra
             or _valid_advised_checkpoints(value[_HOST_PROGRESS_ADVISORY_FIELD]))
    )


OWNER_FIELDS = (
    "_receipts", "trace", "edits", "last_test_exit", "product_edits",
    "workspace_revision", "_last_relevant_match", "_self_oracle_tainted",
    "f2p_ids", "f2p_names", "f2p_inworld", "oracle_file_ids",
    "oracle_selector_counts", "_selector_error_fallback_files", "_futile_answered",
)
RUNTIME_FIELDS = (
    "observations", "test_receipts", "fault_receipts", "integrity_receipts",
    "last_completed_receipt", "artifact_hashes", "snapshot_id",
)
_EVICTFIX_NOTICE_FIELDS = (
    "_evictfix_environment_notice_pending", "_evictfix_environment_notice_announced",
)


def runtime_fields(runtime=None) -> tuple[str, ...]:
    """Return checkpoint fields without changing the legacy OFF shape."""
    fields = list(RUNTIME_FIELDS)
    from kernel_stable_wire import enabled_for_replay
    if enabled_for_replay():
        fields.append("_stable_wire_state")
    from kernel_cache_layout_candidate import FINAL_SOURCE_ORDER_STATE_ATTR
    if runtime is not None and hasattr(runtime, FINAL_SOURCE_ORDER_STATE_ATTR):
        fields.append(FINAL_SOURCE_ORDER_STATE_ATTR)
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        fields.append("_challenge_budget_state")
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        fields.append("_preedit_coverage_state")
    if getattr(runtime, "micrograph_wire", None) is not None:
        fields.append("micrograph_wire")
    if (evictfix_enabled()
            and all(hasattr(runtime, field) for field in _EVICTFIX_NOTICE_FIELDS)):
        fields.extend(_EVICTFIX_NOTICE_FIELDS)
    return tuple(fields)


def checkpoint_runtime_fields_match(runtime, saved_runtime) -> bool:
    """An older checkpoint can omit COV_A; its advisory then stays NOOP."""
    if not isinstance(saved_runtime, dict):
        return False
    actual, expected = set(saved_runtime), set(runtime_fields(runtime))
    if "_stable_wire_state" in expected:
        try:
            from kernel_stable_wire import validate_checkpoint_state
            validate_checkpoint_state(saved_runtime.get("_stable_wire_state"))
        except (TypeError, ValueError):
            return False
    from kernel_cache_layout_candidate import (
        FINAL_SOURCE_ORDER_STATE_ATTR, validate_final_source_order_state,
    )
    if FINAL_SOURCE_ORDER_STATE_ATTR in actual:
        try:
            validate_final_source_order_state(saved_runtime[FINAL_SOURCE_ORDER_STATE_ATTR])
        except (TypeError, ValueError):
            return False
    notice_fields = set(_EVICTFIX_NOTICE_FIELDS)
    present_notice_fields = actual & notice_fields
    if present_notice_fields:
        if present_notice_fields != notice_fields:
            return False
        pending = saved_runtime["_evictfix_environment_notice_pending"]
        announced = saved_runtime["_evictfix_environment_notice_announced"]
        if (pending is not None and (not isinstance(pending, str)
                                    or not 0 < len(pending) <= 160)):
            return False
        if type(announced) is not bool or (announced and pending is not None):
            return False
    if actual == expected:
        return True
    # Source-order state was added after existing checkpoints. Accept its
    # presence or absence across a restart while retaining strict checks for
    # every other field.
    if (actual - {FINAL_SOURCE_ORDER_STATE_ATTR}
            == expected - {FINAL_SOURCE_ORDER_STATE_ATTR}):
        return True
    optional = []
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        optional.append({"_preedit_coverage_state"})
    if set(_EVICTFIX_NOTICE_FIELDS) <= expected:
        optional.append(set(_EVICTFIX_NOTICE_FIELDS))
    return any(actual == expected - missing for missing in (
        *optional,
        (set().union(*optional) if len(optional) > 1 else set()),
    ))


def restore_runtime_field_values(runtime, saved_runtime: Mapping[str, object]) -> None:
    """Restore only checkpoint-owned fields, including pending notice delivery."""
    if not checkpoint_runtime_fields_match(runtime, saved_runtime):
        raise WorkerResumeError("invalid worker checkpoint")
    for key, value in saved_runtime.items():
        if key == "micrograph_wire":
            from kernel_micrograph_wire import MicroGraphWire
            try:
                runtime.micrograph_wire = MicroGraphWire.from_checkpoint(
                    value, str(getattr(runtime.owner, "task_id", "task")))
            except (KeyError, TypeError, ValueError) as error:
                raise WorkerResumeError("invalid MicroGraph checkpoint") from error
        elif key == "_stable_wire_state":
            from kernel_stable_wire import validate_checkpoint_state
            runtime._stable_wire_state = validate_checkpoint_state(value)
        elif key == "_final_cache_source_order_state":
            from kernel_cache_layout_candidate import validate_final_source_order_state
            runtime._final_cache_source_order_state = validate_final_source_order_state(value)
        else:
            setattr(runtime, key, copy.deepcopy(value))


def reconcile_environment_notice_tail(
        runtime: object, tail: list[dict], checkpoint_sequence: int,
) -> bool:
    """Acknowledge a notice only from a received response to its durable wire."""
    if not getattr(runtime, "_evictfix_environment_notice_pending", None):
        return False
    from kernel_step_contract import commit_environment_notice_delivery

    prepared_by_request: dict[str, dict] = {}
    submitted: dict[str, tuple[str, dict, dict]] = {}
    bodies: dict[str, dict] = {}
    responses: dict[str, tuple[str, str]] = {}
    for event in tail:
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence <= checkpoint_sequence:
            continue
        kind = event.get("kind")
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            continue
        if kind == "request_final_wire_prepared":
            prepared_by_request[request_id] = event
        elif kind == "request_submitted":
            sent_id = event.get("event_id")
            prepared = prepared_by_request.get(request_id)
            if isinstance(sent_id, str) and prepared is not None:
                submitted[sent_id] = (request_id, prepared, event)
        elif kind == "request_body":
            sent_id = event.get("submitted_event_id")
            if isinstance(sent_id, str):
                bodies[sent_id] = event
        elif kind == "response_received":
            response_id = event.get("event_id")
            sent_id = event.get("transport_attempt_id")
            if (isinstance(response_id, str) and isinstance(sent_id, str)
                    and sent_id in submitted and submitted[sent_id][0] == request_id):
                responses[response_id] = (request_id, sent_id)
        elif kind == "native_batch_received":
            response_id = event.get("origin_response_event_id")
            matched = responses.get(response_id) if isinstance(response_id, str) else None
            if matched is None or matched[0] != request_id:
                continue
            sent_id = matched[1]
            body = bodies.get(sent_id)
            _, prepared, sent = submitted[sent_id]
            wire_text = sent.get("payload_utf8")
            if not isinstance(wire_text, str):
                delta = sent.get("payload_delta")
                base = submitted.get(sent.get("payload_ref"))
                if (not isinstance(delta, Mapping)
                        or delta.get("identical_wire") is not True
                        or delta.get("base_attempt") != 1
                        or delta.get("changed_keys") != []
                        or base is None or base[0] != request_id
                        or sent.get("payload_hash") != base[2].get("payload_hash")):
                    continue
                wire_text = base[2].get("payload_utf8")
            if not isinstance(wire_text, str):
                continue
            wire = wire_text.encode("utf-8")
            sha = "sha256:" + hashlib.sha256(wire).hexdigest()
            if (sent.get("payload_hash") != sha
                    or prepared.get("payload_sha256") != sha
                    or (body is not None and (
                        body.get("request_id") != request_id
                        or body.get("payload_sha256") != sha
                        or body.get("payload_utf8") != wire_text))):
                continue
            try:
                payload = json.loads(wire)
            except (TypeError, ValueError, RecursionError):
                continue
            if (sent.get("payload") is not None
                    and payload != sent.get("payload")):
                continue
            messages = payload.get("messages") if isinstance(payload, Mapping) else None
            if (not isinstance(messages, list)
                    or messages != prepared.get("messages")):
                continue
            return commit_environment_notice_delivery(runtime, messages)
    return False
PROJECTION_FIELDS = (
    "frame", "last_step", "seen_trace_events", "model_claims",
    "initial_harness_revision", "initial_harness_files",
)
CAPACITY_EVENT_CLASSIFICATION: dict[str, tuple[str, str]] = {
    "capacity_context_limit_ignored": (
        "SAFE_TAIL",
        "A below-floor limit is ignored with dispatch=False and changes no request or retry state.",
    ),
    "capacity_context_limit_learned": (
        "SAFE_TAIL",
        "A resumed journal has no limit cache, so its next lookup folds this scoped limit from durable records.",
    ),
    "capacity_preflight": (
        "REJECT",
        "Live slot and tokenization results control dispatch, and checkpoint restore does not replay them.",
    ),
    "capacity_reroute": (
        "REJECT",
        "A capacity reroute changes the active request path and must be re-evaluated from restored state.",
    ),
    "capacity_retry": (
        "REJECT",
        "This records a newly sealed retry after a provider rejection, which tail restore cannot reconstruct.",
    ),
    "capacity_retry_refit_failed": (
        "REJECT",
        "This records a failed request refit; resume must reconcile the transport path instead of trusting it.",
    ),
}
_CAPACITY_SAFE_TAIL = frozenset(
    kind for kind, (classification, _reason) in CAPACITY_EVENT_CLASSIFICATION.items()
    if classification == "SAFE_TAIL"
)
SAFE_TAIL = frozenset({
    "continuation_writer_resumed", "rich_state_committed", "snapshot_committed",
    "history_decision", "history_exchange_removed", "selection_decision",
    "exposure_composition", "latest_exchange_decision", "request_prepared", "request_submitted",
    "request_body",
    "request_final_wire_prepared", "provider_request_not_dispatched",
    "provider_boundary_host_stop",
    "response_received", "transport_error", "parent_budget",
    "parent_attached", "continuation_worker_restored",
    "continuation_resume_authorized", "continuation_owner_restore", "user_request",
    "perpetual_soak_prepare", "post_green_menu_restored",
    # Audit-only events emitted while PREPARING a request (no state change; replay re-emits them).
    # Missing from this set made every resume refuse once prompt v2 / router / cards were on (09-25 gate).
    # State-changing ops (e.g. decision_cache_op_accepted) must NOT be listed here.
    "system_prompt_variant", "thinking_router_classified", "pattern_cards_rendered",
    "decision_cache_render", "tokens_row6_delivery_audit", "tool_sidecar_stripped",
    # Census emissions record request preparation and the serialized wire; none
    # changes continuation state or performs an effect. Packet/digest facts are
    # derived from the checkpoint-bound workspace and request. Timing/cache fields
    # are diagnostic only and are never used to authorize replay or dispatch.
    "callsite_census_outcome", "callsite_census_prepared",
    "callsite_census_wire_outcome", "callsite_census_fired",
    "callsite_census_shadow", "callsite_census_incremental_shadow",
    # STATEFRAME_PERPETUAL_SOAK=on prepare audit event (no state change).  Found by the
    # real-child soak: without it every soak resume refused (fix round, perpetual_soak).
    "perpetual_soak_prepare",
    # Restore-time audit of the post-green checkpoint section (earlier run fix DONE, item 9).
    "post_green_menu_restored",
}) | _CAPACITY_SAFE_TAIL
CHALLENGE_BUDGET_SAFE_TAIL = frozenset({
    "challenge_budget_state", "done_challenge_once", "done_challenge_overridden",
    "done_submission_record", "hstate_plus_done_challenge", "hstate_plus_done_override",
    "hstate_plus_submission_record",
})
PREPARE_TRANSITIONS = frozenset({"create", "select", "expire", "exhaust"})


def safe_tail_for(runtime=None) -> frozenset[str]:
    """Return journal events admissible after a checkpoint for this runtime.

    Router events stay out of the baseline tail.  They are admitted only when
    the runtime explicitly enabled and constructed the durable replay gate.
    """
    allowed = SAFE_TAIL
    if getattr(runtime, "stateframe_model_router_replay_mode", "off") == "on":
        from kernel_stateframe_model_router_replay import MODEL_ROUTER_SAFE_TAIL
        allowed |= MODEL_ROUTER_SAFE_TAIL
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        allowed |= CHALLENGE_BUDGET_SAFE_TAIL
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        # Derived audit only; durable state is joined from request_submitted.
        allowed |= frozenset({"preedit_coverage_fired"})
    if getattr(runtime, "micrograph_wire", None) is not None:
        allowed |= frozenset({"micrograph_fact_ingested", "micrograph_input_skipped"})
    return allowed


# The AST guard enumerates journal writer literals/constants from src/stateframe.
# Unknown kinds fail closed until they have a replay-safe policy or reconciliation.
_DEFAULT_REJECT_REASON = (
    "no deterministic replay or event-specific reconciliation contract is registered"
)
DOCUMENTED_REJECT: dict[str, str] = {
    kind: f"{kind}: {_DEFAULT_REJECT_REASON}"
    for kind in frozenset("""
        FIRED_ON_WIRE action_envelope_metrics action_envelope_rejected action_group_dropped
        action_menu_guard_decision action_outcome_closed action_receipt action_receipt_amended
        adversary_attached adversary_attack_admitted adversary_attack_dispatched adversary_attack_due
        adversary_attack_receipt adversary_challenge_schema_evaluated adversary_decision_effect adversary_error
        adversary_pressure_reused anchor_recovery_fired attempt_terminated authored_check_context_error
        authored_check_receipt authored_check_replay_scheduled authored_witness_recorded
        authored_witness_rejected authored_witness_retired auto_verify_completed
        auto_verify_deferred_not_settled auto_verify_delivered auto_verify_failed
        auto_verify_revalidation_deferred auto_verify_skipped auto_verify_started automatic_check_cap_hit
        behavior_witness_admitted behavior_witness_bound behavior_witness_done_gate
        behavior_witness_done_gate_blocked behavior_witness_done_gate_exhausted behavior_witness_refusal_settled
        behavior_witness_rerun_after_edit behavior_witness_unrunnable bestofn_candidate_receipt
        bestofn_candidate_rejected bestofn_candidate_requested bestofn_no_eligible_candidate
        bestofn_winner_applied bind_quantity_execution bind_quantity_receipt bind_quantity_request
        boot_read_offered boundary_check_bound boundary_check_dispute boundary_compiler_error
        boundary_contract_cell_bound boundary_contract_recovery_finished boundary_contract_recovery_reprepared
        boundary_contract_recovery_started boundary_field_rejected boundary_nonjson_value_tagged
        boundary_recovery_final_source_menu_checked boundary_recovery_menu_conflict boundary_recovery_terminal
        bounded_locate_fired branch_failed branch_finished branch_started budget_configured budget_failure
        budget_unit_charged cache_bypass cache_layout_final_wire cache_mismatch cache_prefill_accounting
        cache_prefill_response cache_request_summary cache_reuse_alarm candidate_preserved candidate_proposed
        candidate_rejected candidate_restart_intent candidate_restart_portfolio_linked
        candidate_restart_reconciled capability_floor_kept capability_floor_restored capability_manifest_status
        capacity_preflight capacity_reroute catalog_assumption_projection_failed catalog_projection_failed
        catalog_unknowns_completion_blocked causal_localization_decision challenge_budget_state
        check_generator_coverage_blocked check_generator_rule_fired check_receipt
        check_truth_vacuity_refresh_failed claim_check claim_grounding_reconciliation_failed
        claim_locator_unparsed claim_rule_rendered clause_authored_check_dispatched clause_discriminator_receipt
        clause_review_assessed clause_review_case_unrunnable clause_review_episode_started
        clause_review_exhausted clause_review_not_funded clause_review_packet_executed
        clause_review_repair_admitted close_step_hoist_fired closure_menu_narrowed
        code_bookmark_activation_decision code_bookmark_dropped common_sense_governor_error
        common_sense_governor_fired common_sense_governor_released completed_action
        completion_authority_decision completion_authority_finish_error completion_authority_menu_preserved
        completion_authority_resumed completion_authority_started completion_authority_state completion_bounded
        completion_certification completion_diagnosed completion_diagnosis_request completion_mutation_rejected
        completion_open_counterexample_blocked completion_owner_authored_receipt_recomputed
        completion_owner_churn_decision completion_owner_clause_coverage_menu
        completion_owner_common_sense_control completion_owner_common_sense_host_block
        completion_owner_declared_repair_menu completion_owner_declared_repairs_detected
        completion_owner_evidence_witness_menu completion_owner_exhausted completion_owner_finalization_deferred
        completion_owner_green_binding_refreshed completion_owner_green_frame_rendered
        completion_owner_host_gate_held completion_owner_new_behavior_unwitnessed
        completion_owner_next_action_menu completion_owner_next_action_rendered
        completion_owner_no_host_checks_menu completion_owner_no_progress_edit_cap
        completion_owner_no_progress_edit_cap_terminal completion_owner_non_done_response
        completion_owner_open_requirement_route_preserved completion_owner_premature_rule_frame
        completion_owner_premature_rule_menu completion_owner_premature_rule_uncertified
        completion_owner_progress_reset completion_owner_reserved_response_reconciled
        completion_owner_reserved_response_released completion_owner_reserved_response_submitted
        completion_owner_reused_verification completion_owner_route_decision_rendered
        completion_owner_stall_rule_attempted completion_owner_stall_rule_notice
        completion_owner_stall_rule_rendered completion_owner_terminal_choice_offered
        completion_owner_terminal_choice_restatement completion_owner_terminal_choice_unavailable
        completion_owner_terminal_done_recovered completion_owner_terminal_menu_narrowed
        completion_owner_terminal_menu_preserved completion_owner_terminal_text_reconciled
        completion_owner_terminal_text_withdrawn completion_owner_tools_registered
        completion_owner_typed_action_menu completion_owner_typed_host_blocker
        completion_owner_uncertified_submit completion_owner_uncertified_submit_menu
        completion_owner_uncertified_submit_started completion_owner_unclearable_refusal
        completion_owner_unsent_reserved_response completion_owner_witness_limitation_disclosed
        completion_owner_witness_route_exhausted completion_reducer_decision completion_route_menu_aligned
        completion_route_recovery_refused completion_route_surface_mismatch completion_unresolved_questions
        conclusion_cache_reuse context_capacity_shrunk context_scope_archive_rejected context_scope_bound
        context_scope_dispatch context_scope_rejected continuation_authored_check_settled continuation_control
        continuation_deadline_exhausted continuation_grading_isolated continuation_green_revalidated
        continuation_green_revalidation_rejected continuation_host_done_dispatch continuation_menu_projection
        continuation_native_restored continuation_policy continuation_public_certification
        continuation_reobserve_attempt continuation_retention_gc continuation_return_cue
        continuation_return_ready continuation_return_reconciled continuation_source_refresh_refused
        continuation_stranded continuation_transition continuation_transition_rejected continuation_unit_charged
        continuation_worker_checkpoint continuation_worker_stopped continuation_workspace_bound contract_orphan
        convergence_observer_menu_projected convergence_observer_reset convergence_observer_summary
        coverage_reminders_evidence decide2_advisory_completed decide2_advisory_episode
        decide2_checkpoint_skipped decide2_completed decide_notice_suppressed decision_cache_seeded
        decision_cache_state decision_continuity_delta_rejected decision_continuity_organ_projection_failed
        decision_continuity_seeded decision_continuity_state decision_state_rejected decision_state_updated
        decisive_factor_advisory decisive_factor_recheck decisive_factor_recorded declared_path_done_gate_fired
        decode_depth_accounting decode_depth_per_arm_accounting deferred_finalization_classified
        degeneration_detected delivery_audit_error delivery_audit_omitted_for_budget delivery_audit_rendered
        delivery_manifest_error delivery_manifest_final delivery_manifest_omitted_for_budget
        delivery_manifest_prepared dependency_fact_fired diagnosis_fired done_advisory_downgraded
        done_auxiliary_capture_failed done_capture_error done_challenge_cap_blocked done_challenge_cap_exhausted
        done_challenge_capped done_challenge_deferred done_challenge_once done_challenge_overridden
        done_challenge_repair_hold done_challenge_waiting_for_action done_decision done_decision_unavailable
        done_fact_rendered done_facts_computed done_facts_finalized done_is_submit_final_tools
        done_label_readiness_host_debt done_labels_unavailable done_offer_after_green_prepared
        done_readiness_menu_projected done_reply_filled done_restoration_prepared done_sealed
        done_submission_finish_error done_submission_record done_submission_recovered_after_exception
        done_submission_stored done_submission_transcript_error done_submit_dispatch_admitted
        done_submit_ownerless_gate_unavailable done_submit_requested done_submit_source_reused
        done_witnessed_failure_counted done_witnessed_failure_refused done_witnessed_failure_stop
        early_done_gate_skipped early_done_refusal_bound_reached early_done_refused edit_nturn_rendered
        edit_nturn_scope_turns_on_wire edit_nturn_scope_turns_rendered edit_nturn_sites_resolved
        edit_nturn_skipped_budget editready_fact_prepared effect_application_receipt effect_application_rendered
        effect_decision effect_decision_started effect_liveness_assessment effect_liveness_assessment_error
        effect_liveness_experiment effect_liveness_obligation_registered effect_liveness_probe_fired
        effect_liveness_receipt effect_liveness_receipt_ingested effect_liveness_receipt_invalid
        engine_conclusion_carried entry_symbol_verification env_preflight_fired environment_cleanup_error
        environment_discovery_fired environment_panel_delivered environment_process_cleanup_escalated
        environment_process_cleanup_finished environment_process_cleanup_requested
        environment_process_launch_failed environment_process_launch_intent environment_process_reaped
        environment_process_started environment_state_attached environment_state_closed escalation_fired
        evictfix_mode evictfix_mode_warning evidence_pin_final_wire_restored evidence_pin_rendered
        evidence_pin_request_delta existence_claim_downgraded experiment_protocol_rendered
        expert_gap_sidecars_failed external_effect_dispatch external_effect_intent external_effect_outcome
        external_effect_reconciliation external_grading_finished external_grading_started f4_budget_fit
        f4_byte_fit f4_frame_epoch f4_real_exchange_ring f4_settled_state_render_failed f4_state_projected
        false_completion_evidence false_completion_rendered false_completion_started figure_it_out_receipt
        figure_it_out_rendered final_advice_sanitized final_product_manifest_failed final_product_no_candidate
        final_tool_surface_authoring_failed final_wire_rebudgeted first_green_disclosure
        first_green_disclosure_progress first_green_disclosure_rerun_refused fit_false
        fl_checkpoint_capture_disabled fl_checkpoint_capture_error fl_checkpoint_captured fl_cutoff_rule_menu
        fork_a_attached fork_finished fork_not_triggered frame_budget_applied
        frame_continuity_evidence_pin_failed frame_continuity_projected frame_continuity_projection_failed
        frame_diet_long_run_signal frame_files_listed frame_journal_gate_checked frame_route
        frame_route_token_drift frame_series_dispatch frame_series_folded frame_series_restrictions
        frame_series_target_bound framefit5_action_context_degraded framefit5_call_coverage
        framefit5_capacity_compacted framefit5_headcarry_rendered framefit5_provider_boundary_checked
        framefit5_request_preflight_failed framefit_fail_closed framefit_impossible_minimum front_door_fired
        frozen_gate_surface_degraded gear_entry gear_promotion gear_token_accounting generated_artifact_recalled
        get_sure_done_refused get_sure_hint_dropped get_sure_prediction green_evidence_owner_acted
        green_lock_dispatch_idempotent_replay green_lock_dispatch_refused green_lock_menu_authority
        green_lock_menu_authority_empty green_lock_repair_done_refused green_lock_repair_receipt
        green_lock_terminal_menu green_lock_terminal_menu_reopened green_lock_terminal_menu_skipped
        green_then_exhaust_boundary_omitted green_then_exhaust_boundary_rendered green_then_exhaust_evaluated
        green_then_exhaust_host_submit_suppressed green_then_exhaust_not_submitted green_then_exhaust_submission
        greenfact_error greenfact_green_now greenfact_skipped greenlock_first_edit_applied grounded_intake_fired
        guarded_effect_acceptance guarded_effect_dispatch guarded_effect_intent guarded_effect_receipt
        harness_contract_error harness_fault harness_traceback_line_mismatch history_compaction
        history_compaction_accounting history_compaction_failed history_findings_dropped
        history_findings_projection_failed history_findings_rejected history_findings_restore_rejected
        history_findings_seeded history_findings_state history_state_tail hook_blocked host_check_bundle_source
        host_checks_after_public_green host_closure_render_degraded host_closure_run_tests_refused
        host_failure_replayed host_obligation_check host_repair_check host_repair_episode host_span_profile
        host_span_profile_failed hstate_plus_applied_receipt hstate_plus_done_challenge
        hstate_plus_done_override hstate_plus_full_menu hstate_plus_submission_record hy_step_fix_armed
        hybrid_compaction idealpath_action idealpath_action_rejected idealpath_action_replanned
        idealpath_check_receipt idealpath_completion_consulted idealpath_decision_graph_adapter_error
        idealpath_disabled idealpath_frozen_repro_result idealpath_host_draft_error
        idealpath_legacy_repro_rejected idealpath_menu_projected idealpath_navigation_evidence_error
        idealpath_owner_attached idealpath_reproduction_result idealpath_s3force_result
        idealpath_spine_binding_refused idealpath_spine_bound idealpath_spine_stage_consumed
        idealpath_spine_stage_refused idealpath_stage_exit_forced idealpath_stage_transition
        idealpath_stage_transition_boundary idealpath_state idealpath_symbol_packet_built
        idealpath_ticket_failure_admission idealpath_typed_repro_result idealpath_typed_testwriter_result
        idealpath_verification_recorded identical_frame_guard import_smoke import_smoke_baseline
        import_smoke_baseline_stale independent_roots_finished inflight_fix_applied information_exposure
        information_policy_final infra_work_classifier_fired infra_work_pair_excluded
        instrument_validation_nonce intent_cleared intent_delivered intent_extracted intent_no_progress
        intent_rendered intent_suppressed jit_clearing_line_rendered job_completed job_concurrency_cap_hit
        job_kill_requested job_launch_failed job_launch_refused job_operation_completed job_operation_rejected
        job_snapshot_committed job_started job_wall_cap_hit job_wall_cap_rejected jobqueue_contract_audit
        jobqueue_contract_reaudit kernel_error knowledge_packet_bound kv_chunk_reuse_prepared
        lean_action_contract_finalized lean_fired lean_history_append_only_preserved
        lean_history_append_only_refused lean_history_capacity_outcome lean_history_compaction_suppressed
        lean_request_shaped lean_source_refresh_skipped lean_source_refreshed lean_tool_results_reconciled
        lifecycle_state_table literal_site_inventory live_counterexample_done_check
        live_counterexample_done_gate localization_search_completed manifest_ledger_mismatch
        memory_isolation_carry memory_isolation_carry_filtered menu_authority_wire_mismatch
        micrograph_fact_ingested micrograph_input_skipped model_decision_cap_hit model_profile_selected
        model_request model_response model_retry model_router_assignment model_router_call
        model_router_provider_request_completed model_router_provider_request_failed
        model_router_provider_request_started model_router_replay_state model_shape_guard_applied
        multi_definition_ast_retired multi_definition_open_work multi_definition_open_work_closed
        named_layer_coverage_evidence named_layer_coverage_rendered named_layer_coverage_started
        named_layer_repair_applied native_batch_action_closed native_batch_action_force_closed
        native_batch_action_selected native_batch_action_unresolved native_batch_call_receipt
        native_batch_discarded native_batch_menu_frozen native_batch_receipts_delivered native_batch_received
        native_batch_rejected native_batch_unexecuted native_response_rejected nitpicker_bound_green
        no_done_only_collapse_applied no_done_only_collapse_fired noread_applied noread_fired
        obligation_snapshot obligation_transaction obligation_transition observation_committed
        observation_coverage observation_probe_completed organ_wiring_failure organ_wiring_state
        other_end_contract_error other_end_contract_fired owner_green_open_repair_kept
        owner_menu_dispatch_refused parent_product_identity_captured partial_obs_rejected
        pass_gates_shared_uncertified_decision pass_gates_vector_failed passive_retention_dedup
        passive_retention_delivery_metric passive_retention_delivery_metric_error passive_retention_omitted
        passive_retention_rejected passive_retention_state passive_withheld_dropped phase_release_refused
        phase_released planned_artifact_event playbook_adoption playbook_card_offered portfolio_attached
        portfolio_budget_exhausted portfolio_candidate_admitted portfolio_fork_completed
        portfolio_preservation_failed portfolio_selection_committed portfolio_started
        post_edit_static_check_counts post_edit_static_check_error post_green_edit_cycle
        post_green_edit_verified post_green_reopen post_green_stall_completion post_green_stall_terminal
        post_green_verification_cleared post_green_work_menu postgreen_budget_evaluated
        postgreen_budget_submitted postgreen_duplicate_evidence postgreen_host_settled postgreen_new_evidence
        postgreen_no_progress_advisory postgreen_no_progress_reset postgreen_no_progress_signal
        postgreen_no_progress_submitted postgreen_no_progress_terminal postgreen_no_veto_admitted
        postgreen_no_veto_done_refused postgreen_non_success_evidence postgreen_quiet_applied
        postgreen_suppressed_unchanged_parent postgreen_terminal_action_admitted postgreen_typed_refusal
        predispatch_decision preedit_coverage_fired preflight_source_refreshed premortem_fired
        prepublish_diff_audit_baseline prepublish_diff_audit_blocked prepublish_diff_audit_fired
        prepublish_diff_restore_applied prepublish_git_command_denied probe_gate_decision probe_pytest_refused
        probe_render_applied probe_result progress_guard_decision prompt_cache_accounting prompt_compact_applied
        prompt_diet_applied provability_completion_record provider_attempt_outcome_unknown
        provider_boundary_contract_asserted provider_boundary_contract_decided provider_boundary_contract_error
        provider_capability_refused provider_dialect provider_feature_degraded provider_malformed_response
        provider_queue_wait provider_request_timeout provider_transport_retry provider_transport_retry_aborted
        ps_counterexamples_fired ps_invariants_fired ps_minimize_fired ps_minimize_rejected ps_minimize_skipped
        ps_router_fired public_probe_executed public_probe_quarantined public_probe_rerun_unavailable
        public_probe_reviewed public_probe_validity public_review_admission public_review_advice_selected
        public_review_child public_review_unexposed public_state_final public_state_final_unavailable
        public_test_verdict_fact_bypass public_test_verdict_fact_normalized
        public_test_verdict_fact_shadow_checked public_test_verdict_shadow_mismatch qgate_answered
        qgate_belief_falsified qgate_belief_opened qgate_check_observed qgate_check_opened qgate_check_settled
        qgate_cost_report qgate_dodged qgate_drift_reanchored qgate_error qgate_transition
        qualified_request_sealed question_answered r44_blocks_rendered raising_h_packet_rendered
        read_budget_admitted read_contract_fired read_lockout_decision read_once_final_fired read_once_fired
        read_once_note_queued read_once_wire_note_reconciled read_pointer_edit_menu_armed
        read_pointer_edit_menu_fired read_pointer_fired read_pointer_inline_fired read_repeat_assessed
        read_reuse_source_snapshot_reconciled read_reuse_visibility_checked readiness_decision
        readiness_done_refused readiness_menu_projected readiness_submit_committed readiness_submit_deferred
        readiness_submit_refused reasoning_ledger_state recall_requested red_terminal_suppressed
        refusal_compact_rendered registry_build_failed registry_built registry_lookup_completed
        registry_refresh_failed registry_reused rejected_candidate_artifact rejected_candidate_closed
        rejected_candidate_delivery rejection_template_fired rejection_template_notice_fired
        rejection_template_restrict_armed rejection_template_restrict_fired render_degraded
        repair_blocks_done_advisory repair_blocks_done_non_progress repair_blocks_done_stopped_incomplete
        repair_packet_request_audit repair_packet_unrenderable repeated_zero repeated_zero_suppressed
        repo_intel_query repo_intel_query_failed repo_map_modes repo_outline repro_first_gate_fired
        request_action_pin_repaired request_budget_exceeded request_budget_final_wire
        request_budget_revalidation_consistency request_contract_revision_rebind_shadow_mismatch
        request_final_validation_blocked request_not_admitted request_prepare_timing request_reprepared
        request_source_claim_deferred request_source_claim_dropped request_source_closure
        request_source_final_wire requirement_alignment_done_refused requirement_recheck_refused
        requirements_dispute requirements_ledger_compiled requirements_ledger_error
        requirements_ledger_update_error requirements_ticket_read response_budget_exhausted
        response_truncation_checked restate_requirement_fired resume_work_state_restored retained_context_error
        retained_edit_focus_unavailable retained_edit_focused retained_external_read_unbound
        retained_read_focus_unavailable retained_read_focused retained_recall retained_recall_rejected
        retained_refresh_raced retained_selection_proposed revalidation_disagreement
        ripple_inline_safety_checked ripple_preview_disposition ripple_preview_disposition_update
        ripple_preview_error ripple_preview_fired ripple_preview_inline ripple_preview_reconciled
        ripple_preview_shadow_recall ripple_preview_staged ripple_preview_unavailable
        route_budget_menu_projected route_escalation_discarded row_role run_exception run_terminal_advisory
        run_tests_host_refused s2_force_menu_rerender same_file_multi_edit_applied same_shape_sweep_failed
        same_shape_sweep_fired same_shape_sweep_obligation_seeded same_shape_sweep_pattern_validated
        same_shape_sweep_query_compiled same_shape_sweep_scan_completed same_shape_sweep_signature_rejected
        same_shape_sweep_signature_requested same_shape_sweep_skipped scope_lock_authorization_admitted
        scope_lock_held scope_widen selection_committed self_plan self_plan_pair_metrics semantic_error
        send_stall settled_findings_projection_failed settled_findings_render_failed settled_findings_rendered
        shadow_admission_receipt shadow_consumption_receipt shadow_response_receipt shell_environment_changed
        ship_tree_smoke_decision_accepted sidekick_finding site_facts_failed
        skeptic_cleared_by_reused_verification skeptic_fired source_requalified
        source_revision_read_pins_retired source_revision_replan_refused source_revision_replan_started
        source_revision_reprepared source_store_error source_store_state stable_wire_epoch
        stable_wire_failed_open stage_transition_action_refused stagnation_cap_blocked stagnation_cap_observed
        stall_observation_applied stall_observation_fired stall_submit_strict_evaluated
        starter_conflict_boundary_bundle_error starter_conflict_refresh_error starter_conflict_setup_error
        state_owner_inventory stateframe_frame_append_committed stateframe_frame_append_failed_open
        stateframe_frame_append_prepared stateframe_frame_append_rebased stateframe_frame_append_shadow_mismatch
        stateframe_framefit_m3_advisory stateframe_framefit_source_windows stateframe_m3_tool_pairing_rejected
        stateframe_m3_tool_results_reconciled stateframe_m3_working_context_advisory
        stateframe_m3_working_context_fit stateframe_menu_freeze_committed stateframe_menu_freeze_failed_open
        stateframe_menu_freeze_rebase_evaluated stateframe_menu_freeze_rebase_failed_open
        stateframe_menu_freeze_rebase_shadow_mismatch stateframe_menu_freeze_rebased stateframe_page_log_budget
        stateframe_page_log_committed stateframe_page_log_exchange_evaluated
        stateframe_page_log_exchange_failed_open stateframe_page_log_exchange_shadow_mismatch
        stateframe_page_log_exchange_text_refused stateframe_page_log_failed_open
        stateframe_page_log_inventory_failed_open stateframe_page_log_inventory_prepared
        stateframe_page_log_inventory_shadow_mismatch stateframe_page_log_prepared stateframe_page_log_rebased
        stateframe_page_log_retirement_evaluated stateframe_page_log_retirement_failed_open
        stateframe_page_log_retirement_shadow_mismatch stateframe_page_log_shadow_mismatch
        stateframe_page_log_src_supersede_dependency_warning stateframe_page_log_src_supersede_evaluated
        stateframe_page_log_src_supersede_failed_open stateframe_page_log_src_supersede_shadow_mismatch
        stateframe_page_log_window_key_evaluated stateframe_page_log_window_key_failed_open
        stateframe_page_log_window_key_shadow_mismatch stateframe_switch_preflight_refused
        stateframe_tool_stable_committed stateframe_tool_stable_failed_open step_contract_rejected
        step_dispatch_binding stop_reason_classified stop_reason_taxonomy_error structural_fact_emitted
        structural_fact_error stub_sweep_dispatch_refused stub_sweep_evaluated stub_sweep_read_dropped
        submission_idempotency sufficiency_fired supervised_command_completed supervision_stop
        system_one_receipt system_one_triggered system_trim_applied target_bound task_id
        template_conformance_prepared template_conformance_submitted terminal_candidate_capture_failed
        terminal_disposition_minted terminal_lower_refused terminal_outcome_recorded
        terminal_outcome_recording_failed terminal_reserve_exhausted terminal_result test_integrity_baseline
        test_integrity_baseline_restored test_integrity_checked test_integrity_dispute
        test_integrity_mutation_check test_integrity_obligation test_integrity_obligation_closed
        test_reported_wrong test_subprocess_completed think_prefill think_prefill_metrics thinking_3d_computed
        thinking_4d_computed thinking_5d_computed thinking_rendered ticket_final_wire_receipt
        ticket_final_wire_repaired ticket_model_brief_receipt ticket_retrieval_receipt timeline_frame_projected
        timeline_projection_error tokenlab_read_plain tokenlab_stallwire_detected tokenlab_stallwire_error
        tokenlab_stallwire_menu_projected tokenlab_stallwire_probe_suppressed
        tokenlab_stallwire_source_delivery_confirmed tokenlab_v15_cache_prefill_protected
        tokenlab_v15_postgreen_identical_rerun tokenlab_v16_postgreen_no_evidence tokenlab_v17_edit_hardening
        tokenlab_v17_host_result_guard tokenlab_v17_no_progress_stall tokenlab_v17_sampler_sent
        tokenlab_v17_stall_menu_projection tokenlab_v17_tool_list_projected tokenlab_v19_truthful_exit2
        tokenlab_v20_completion_guard_blocked tokenlab_v20_no_progress_stall tokenlab_v20_stall_breaker_restored
        tokenlab_v20_stall_menu_projection tokenlab_v20_ticket_clause_surface tokenlab_v22_change_set
        tokenlab_v22_compatibility_check tokenlab_v22_stall_progress tokenlab_v22_unified_ledger
        tokens_row6_observed tokens_row6_observer_error tokens_row6_repeat_detected tokens_row6_state
        tool_applicability_decision tool_attempt_cap_hit tool_call_cap_audit_summary tool_call_cap_hit
        tool_call_counted tool_call_loop_breaker_rendered tool_call_loop_breaker_tripped tool_menu_composed
        tool_menu_memo_shadow_mismatch tool_reply_composition_error tool_reply_duplicate_removed
        tool_reply_filled tool_reply_receipt tool_schema_diet_applied tool_surface_applied tool_surface_signal
        trailing_panels_history_fold_pointer trailing_panels_history_prefix trailing_panels_prepared
        trailing_panels_tool_menu_break trailing_panels_wire_layout turn0_cache turn_timing turtle_accounting
        turtle_request turtle_review_result turtle_trigger turtle_verdict unbounded_no_progress_terminal
        uncertified_submit_available uncertified_submit_outcome uncertified_submit_rendered
        unified_stop_decision unknown_opened unknown_resolved unknowns_producer_failed unknowns_render_failed
        unknowns_rendered unverified_checks_done_gate unverified_checks_menu_window user_hook_config
        user_hook_run v2_done_decision v2_host_review v2_terminal v46_mechanism_notice_rendered
        v46_mechanism_wiring_error vacuity_verdict verdict_identity_fired verification_reuse_skipped
        where_used_closure_error where_used_closure_fired where_used_site_closed wire_host_guard_applied
        wire_invariant_violated wire_stability_applied wire_string_hygiene_violation wire_string_sanitized
        work_intent_claimed work_intent_rejected work_record_rejected work_record_seeded work_record_updated
        worker_crashed worker_exception worker_finished worker_response worker_resumed worker_started
        WIRE_TRUTH_BLOCKED WIRE_TRUTH_DEGRADED candidate_admitted done_offered_after_green
        done_restored_on_wire fork_a_budget_denied fork_a_finished fork_a_not_triggered
        fork_a_resumed fork_a_run_reused fork_a_started frame_journal_mismatch
        frame_journal_panel_dropped journal_observer_failed jq_class_edit_applied
        outline_parser_capability read_working_set_remembered read_working_set_removed
        request_budget_applied request_diet_applied request_diet_cap_infeasible
        request_source_delivery_candidate request_source_delivery_candidate_failed
        request_visibility_recovery run_started runtime_mode_snapshot source_store_evicted
        source_store_page_remembered wire_truth_checked wire_truth_gate_disabled
        wire_truth_rebuild_failed worker_spawned working_set_evicted working_set_updated
        working_set_virtual_created working_set_virtual_updated
        working_notebook_rendered working_notebook_retrieved world_knowledge_event world_model_fired
        world_model_scored world_observation_error
    """.split())
}
DOCUMENTED_REJECT.update({
    "model_router_assignment": (
        "Model-router assignment state is admitted only with model-router replay enabled."
    ),
    "model_router_call": (
        "Model-router call state is admitted only with model-router replay enabled."
    ),
    "model_router_replay_state": (
        "Model-router replay state requires its durable replay gate and validated state record."
    ),
    "preedit_coverage_fired": (
        "This derived coverage audit is admitted only when its treatment or placebo arm is active."
    ),
    "micrograph_fact_ingested": (
        "Micrograph facts are admitted only when the runtime has the matching micrograph wire."
    ),
    "micrograph_input_skipped": (
        "Micrograph input audits are admitted only when the runtime has the matching micrograph wire."
    ),
    "step_dispatch_binding": (
        "A dispatched action can be retried only under keyed done reconciliation."
    ),
    "action_receipt": (
        "An action receipt can be replayed only as part of keyed done reconciliation."
    ),
    "completion_certification": (
        "Completion certification can be replayed only as part of keyed done reconciliation."
    ),
    "continuation_public_certification": (
        "Public completion certification can be replayed only as part of keyed done reconciliation."
    ),
    "submission_idempotency": (
        "Submission state can be replayed only with the keyed idempotency protocol enabled."
    ),
    "challenge_budget_state": (
        "Challenge-budget state requires a configured budget owner and validated restore state."
    ),
    "completion_reducer_decision": (
        "Reducer decisions can be replayed only as part of keyed done reconciliation."
    ),
    "continuation_transition": (
        "Only prepare, select, expire, and exhaust transitions are admitted without action reconciliation."
    ),
    "native_batch_received": (
        "Native batches are admitted only when the journal declares durable native-batch handling."
    ),
    "host_check_bundle_source": (
        "Host-check provenance changed after the checkpoint; restore reloads only the "
        "checkpointed value, so this tail needs host-check reconciliation."
    ),
    "passive_retention_state": (
        "This event mutates retained state after the checkpoint; passive-retention state "
        "must be reconciled before resume."
    ),
    "WIRE_TRUTH_BLOCKED": (
        "Wire validation blocked dispatch, and its retry/counter state is not restored from the checkpoint."
    ),
    "WIRE_TRUTH_DEGRADED": (
        "Bounded wire recovery changed the request result; resume must rebuild and revalidate it."
    ),
    "candidate_admitted": (
        "Fork-A admitted a candidate in its separate portfolio journal; worker resume does not restore it."
    ),
    "done_offered_after_green": (
        "The final completion-tool surface was derived after the checkpoint and has no restore contract."
    ),
    "done_restored_on_wire": (
        "The final wire was changed after the checkpoint; resume must revalidate the serialized request."
    ),
    "fork_a_budget_denied": (
        "Fork-A parent budget state is stored in its separate portfolio journal, not this checkpoint."
    ),
    "fork_a_finished": (
        "Fork-A worker status is stored in its separate portfolio journal, not this checkpoint."
    ),
    "fork_a_not_triggered": (
        "Fork-A admission state is stored in its separate portfolio journal, not this checkpoint."
    ),
    "fork_a_resumed": (
        "Fork-A child replay state requires portfolio-journal reconciliation before worker resume."
    ),
    "fork_a_run_reused": (
        "Fork-A reuse depends on separate portfolio-journal state not restored from this checkpoint."
    ),
    "fork_a_started": (
        "Fork-A creates separate worker checkpoints and requires its own resume reconciliation."
    ),
    "frame_journal_mismatch": (
        "The runtime frame disagreed with journal state after checkpoint; fail closed pending reconciliation."
    ),
    "frame_journal_panel_dropped": (
        "A stateframe panel was dropped after journal reconciliation failed; request state must be rebuilt."
    ),
    "journal_observer_failed": (
        "A journal observer failed after checkpoint and has no deterministic replay contract."
    ),
    "jq_class_edit_applied": (
        "A repository edit was applied after checkpoint; resume requires effect reconciliation."
    ),
    "outline_parser_capability": (
        "Parser capability reflects runtime setup and has no checkpointed reconciliation record."
    ),
    "read_working_set_remembered": (
        "Working-set spans changed after checkpoint; restore only reloads checkpointed source state."
    ),
    "read_working_set_removed": (
        "Working-set spans changed after checkpoint; restore only reloads checkpointed source state."
    ),
    "request_budget_applied": (
        "The request tool surface changed after checkpoint; replay must rebuild and revalidate it."
    ),
    "request_diet_applied": (
        "Request content changed after checkpoint; replay must rebuild and revalidate the final request."
    ),
    "request_diet_cap_infeasible": (
        "Request preparation failed its size contract after checkpoint and must be retried from restored state."
    ),
    "request_source_delivery_candidate": (
        "Source delivery selection changed after checkpoint and requires request-level revalidation."
    ),
    "request_source_delivery_candidate_failed": (
        "Source delivery proof failed after checkpoint and must be rebuilt from restored state."
    ),
    "request_visibility_recovery": (
        "Source visibility recovery changed the candidate request and must be repeated from restored state."
    ),
    "run_started": (
        "The journal run-start envelope is not an admissible post-checkpoint transition."
    ),
    "runtime_mode_snapshot": (
        "Runtime mode is setup state and is not restored or reconciled from a checkpoint tail."
    ),
    "source_store_evicted": (
        "Source evidence was evicted after checkpoint; restore does not reload that index state."
    ),
    "source_store_page_remembered": (
        "Source evidence was inserted after checkpoint; restore does not reload that index state."
    ),
    "wire_truth_checked": (
        "Wire validation is bound to regenerated request bytes and must be rerun after restore."
    ),
    "wire_truth_gate_disabled": (
        "Wire-validation policy was disabled at request time; this checkpoint does not reconcile policy drift."
    ),
    "wire_truth_rebuild_failed": (
        "Wire recovery failed after checkpoint and requires request-level reconciliation."
    ),
    "worker_spawned": (
        "Fork-A spawned a worker outside this continuation checkpoint; host reconciliation is required."
    ),
    "working_set_evicted": (
        "Working-set state was evicted after checkpoint; restore only reloads checkpointed working-set state."
    ),
    "working_set_updated": (
        "Working-set state changed after checkpoint; restore only reloads checkpointed working-set state."
    ),
    "working_set_virtual_created": (
        "Virtual working-set state was created after checkpoint and has no restore reconciliation contract."
    ),
    "working_set_virtual_updated": (
        "Virtual working-set state changed after checkpoint and has no restore reconciliation contract."
    ),
})
DOCUMENTED_REJECT.update({
    kind: f"{kind}: {reason}"
    for kind, (classification, reason) in CAPACITY_EVENT_CLASSIFICATION.items()
    if classification == "REJECT"
})
DOCUMENTED_REJECT.update({
    "intake_orientation_failed": (
        "The failure omits orientation from the first request, so replay can rebuild different request bytes."
    ),
    "repo_map_commit_sha_unavailable": (
        "A transient HEAD lookup failure is not guaranteed to recur during checkpoint replay."
    ),
    "shell_memory_enforcement": (
        "Memory enforcement can kill a command and alter its observation or effects; "
        "restore does not replay that supervisor result."
    ),
    "completion_owner_unavailable": (
        "Completion-owner fallbacks can retain stale request-contract or menu authority after checkpoint."
    ),
    "host_obligation_evidence_failure": (
        "Failed host obligation reads can change evidence assessment or abort check construction during replay."
    ),
    "unified_stop_evidence_failure": (
        "Evidence failure aborts completion-snapshot construction and can change the shared stop decision."
    ),
    "wire_truth_fact_evidence_failure": (
        "Unavailable wire-truth evidence changes request facts or aborts dispatch preparation."
    ),
})


def resume_event_rejection_reason(
        kind: str, runtime=None, *, allowed_tail: frozenset[str] | None = None) -> str | None:
    """Return the fail-closed reason for a kind outside this runtime's safe tail."""
    allowed = safe_tail_for(runtime) if allowed_tail is None else allowed_tail
    if kind in allowed:
        return None
    reason = DOCUMENTED_REJECT.get(kind)
    if reason is None:
        return f"unclassified post-checkpoint journal event kind: {kind}"
    return f"post-checkpoint state requires reconciliation: {kind}: {reason}"


def resume_tail_rejection_reason(
        tail: list[Mapping[str, object]], runtime=None, *,
        allowed_tail: frozenset[str] | None = None,
        native_tool_batches: bool = False) -> str | None:
    """Return the first event-kind refusal that checkpoint restore would raise."""
    allowed = safe_tail_for(runtime) if allowed_tail is None else allowed_tail
    for entry in tail:
        kind = entry["kind"]
        if (kind == "continuation_transition"
                and entry.get("operation") in PREPARE_TRANSITIONS):
            continue
        if kind == "native_batch_received" and native_tool_batches:
            continue
        if kind not in allowed:
            reason = resume_event_rejection_reason(kind, runtime, allowed_tail=allowed)
            if reason is not None:
                return reason
            return f"unclassified post-checkpoint journal event kind: {kind}"
    return None


def restore_router_replay_tail(runtime: object, records: list[Mapping[str, object]],
                               checkpoint_sequence: int) -> None:
    """Restore one committed post-checkpoint router state before the next route."""
    if getattr(runtime, "stateframe_model_router_replay_mode", "off") != "on":
        return
    from kernel_stateframe_model_router_replay import latest_state_record, validate_tail
    tail = [record for record in records
            if type(record.get("sequence")) is int
            and record["sequence"] > checkpoint_sequence]
    validate_tail(tail)
    router = getattr(runtime, "model_router", None)
    latest_router_state = latest_state_record(runtime.journal)
    if (router is None or latest_router_state is None
            or type(latest_router_state.get("sequence")) is not int
            or latest_router_state["sequence"] <= checkpoint_sequence):
        return
    state_checkpoint = latest_router_state.get("checkpoint_sequence")
    if state_checkpoint is not None and state_checkpoint != checkpoint_sequence:
        raise WorkerResumeError("post-checkpoint router state has the wrong checkpoint identity")
    router.restore_replay_state(latest_router_state["state"])


class WorkerResumeError(RuntimeError):
    pass


def _register_checkpoint(runtime, event: dict) -> None:
    """Preserve the retention engine's checkpoint index at the journal seam."""
    continuation = getattr(runtime, "continuation", None)
    engine = getattr(continuation, "engine", None)
    register = getattr(engine, "register_checkpoint", None)
    if callable(register):
        register(event)


def _reconcile_external_effects(runtime) -> set[str]:
    """Reconcile host-owned outcomes before applying the unresolved-effect gate."""
    continuation = getattr(runtime, "continuation", None)
    reconcile = getattr(continuation, "reconcile_external_effects", None)
    if not callable(reconcile):
        return set()
    events = reconcile()
    if events is None:
        return set()
    if not isinstance(events, (set, list, tuple)) or any(
            not isinstance(event_id, str) or not event_id for event_id in events):
        raise WorkerResumeError("external reconciliation returned invalid event references")
    return set(events)


def enabled() -> bool:
    value = os.environ.get("KERNEL_CONTINUATION_RESTART", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_CONTINUATION_RESTART must be 0 or 1")
    return value == "1"


def validate_profile(information) -> None:
    from kernel_continuation_runtime import enabled as executive_enabled
    from kernel_intervention_policy import interventions_enabled
    if not enabled() or not executive_enabled():
        raise WorkerResumeError("worker restart requires both explicit continuation switches")
    if not information.strict or not information.off or information.feedback != "FINAL_ONLY":
        raise WorkerResumeError("worker restart requires strict unassisted FINAL_ONLY information policy")
    try:
        from kernel_stateframe_model_router import (
            model_router_mode, validate_restart_compatibility,
        )
        validate_restart_compatibility(True, mode=model_router_mode())
    except ValueError as error:
        raise WorkerResumeError(str(error)) from error
    if interventions_enabled():
        raise WorkerResumeError("legacy organ replay is not implemented for worker restart")
    from kernel_decision_state import notes_enabled
    if notes_enabled() and os.environ.get("KERNEL_DECISION_STATE", "0") != "0":
        raise WorkerResumeError("working notes cannot restore full decision state")
    from kernel_native_tools import validate_configuration
    native = validate_configuration()
    if notes_enabled() and not native:
        raise WorkerResumeError("working-note restart requires native provenance")
    from kernel_frame_series_runtime import configuration
    series, _ = configuration()
    if series and not native:
        raise WorkerResumeError("frame-series restart requires native dispatch")
    required = {
        "REACT_TOOLCALL": "1", "REACT_STEP_CONTRACT": "0" if native else "1",
        "STATEFRAME_PROJECTOR": "rich",
        "STATEFRAME_VIEW": "operational",
        "KERNEL_DECISION_STATE": "0",
        "KERNEL_REJECTED_CANDIDATE": "0",
        "FACT_COMPILER": "0", "CROSSRUN_MEMORY": "0",
        # A capsule bank is an external immutable input, but this earlier run restart
        # checkpoint has no bank binding/replay section.  Refuse it rather than
        # silently loading a different bank after worker restart.
        "STATEFRAME_MEMORY_CAPSULE": "off",
        "FAIR_BRIEF": "0", "GROUNDED_BRIEF": "0",
    }
    defaults = {"REACT_TOOLCALL": "1", "STATEFRAME_PROJECTOR": "legacy",
                "STATEFRAME_VIEW": "existing", "STATEFRAME_LATEST_EXCHANGE": "off",
                "FACT_COMPILER": "1", "CROSSRUN_MEMORY": "1",
                "STATEFRAME_MEMORY_CAPSULE": "off",
                "FAIR_BRIEF": "1", "GROUNDED_BRIEF": "1"}
    bad = [key for key, value in required.items()
           if os.environ.get(key, defaults.get(key, "0")) != value]
    exchange = os.environ.get("STATEFRAME_LATEST_EXCHANGE", "off")
    if exchange not in {"off", "quoted", "roles"} or (exchange != "off" and not native):
        bad.append("STATEFRAME_LATEST_EXCHANGE")
    if bad:
        raise WorkerResumeError("unsupported restart state holders/profile: " + ", ".join(bad))


@dataclass(frozen=True)
class WorkerResume:
    workspace: Path
    expected_journal_head: str
    follow_up: str | None = None
    request_id: str | None = None

    def open_workspace(self, parent: Path, *, task_digest: str) -> ContinuationWorkspace:
        path = Path(self.workspace).absolute()
        if (path.is_symlink() or path.parent.resolve() != parent.resolve()
                or len(path.name) != 32 or any(c not in "0123456789abcdef" for c in path.name)):
            raise WorkerResumeError("resume workspace must be an original direct session")
        if (not isinstance(self.expected_journal_head, str)
                or len(self.expected_journal_head) != 71
                or not self.expected_journal_head.startswith("sha256:")
                or any(c not in "0123456789abcdef" for c in self.expected_journal_head[7:])):
            raise WorkerResumeError("explicit trusted journal head required")
        if self.follow_up is not None and (not isinstance(self.follow_up, str) or not self.follow_up):
            raise WorkerResumeError("follow-up user message must be nonempty text")
        session = ContinuationWorkspace.open_existing(path, task_digest=task_digest)
        try:
            if (path / "grading.json").exists():
                raise WorkerResumeError("terminal grading session cannot resume execution")
            execution_paths(session)
            return session
        except BaseException:
            session.close()
            raise


def execution_paths(workspace):
    initial = workspace.read("initialized.json")
    paths = []
    for parent, key in ((workspace.directory, "root"), (None, "target")):
        parent = paths[0] if parent is None else parent
        raw = initial.get(key)
        if (not isinstance(raw, str) or not raw or Path(raw).is_absolute()
                or any(part in {".", ".."} for part in raw.split("/"))):
            raise WorkerResumeError("invalid retained execution path")
        path = parent / raw
        if path.is_symlink() or path.resolve() != path or not path.is_relative_to(parent):
            raise WorkerResumeError("retained execution path escaped or was aliased")
        paths.append(path)
    if not paths[0].is_dir() or not paths[1].is_file():
        raise WorkerResumeError("retained execution files are missing")
    return tuple(paths)


def resume_journal(runtime, output, manifest, descriptor):
    from kernel_continuation_journal import ContinuationJournal
    initial = runtime.workspace.read("initialized.json")
    root, target = execution_paths(runtime.workspace)
    if root != Path(runtime.owner.root) or target != runtime.target:
        raise WorkerResumeError("worker paths differ from original execution binding")
    journal_path = Path(initial["journal_path"])
    if (journal_path.name != "events.jsonl"
            or journal_path.parent.parent.resolve() != Path(output).resolve()
            or journal_path.parent.name != initial["run_id"]):
        raise WorkerResumeError("journal is outside the original measurement allocation")
    journal = ContinuationJournal.resume(journal_path, task_id=runtime.owner.task_id,
        manifest=manifest, expected_head=descriptor.expected_journal_head)
    try:
        bindings = [e for e in journal.records() if e["kind"] == "continuation_workspace_bound"]
        if (len(bindings) != 1 or bindings[0].get("initialization_hash") != digest(initial)
                or bindings[0].get("allocation_hash") != digest(runtime.workspace.read("allocation.json"))
                or bindings[0].get("directory") != str(runtime.workspace.directory)
                or bindings[0].get("root") != str(root)
                or initial["config_hash"] != journal.config_hash
                or initial["task_id"] != journal.task_id):
            raise WorkerResumeError("workspace metadata is not bound to this journal")
    except BaseException:
        journal.close()
        raise
    runtime.resume_follow_up = getattr(descriptor, "follow_up", None)
    runtime.resume_request_id = getattr(descriptor, "request_id", None)
    return journal


def _plain(value):
    from kernel_memfix_history import SpoolSequence
    if isinstance(value, (tuple, set, HistorySpool, SpoolSequence)):
        return [_plain(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _checkpoint_ledger(ledger):
    """Journal full actions once; repeat only compact identities in each checkpoint."""
    from kernel_memfix_history import legacy
    if legacy():
        return _plain(ledger.records)
    return {
        "schema": SLIM_LEDGER_SCHEMA,
        "record_ids": [record["step_record_id"] for record in ledger.records],
        "slim": _plain(ledger.slim_state()),
    }


def _journal_ledger(saved, records, boundary):
    """Authenticate compact checkpoint identities against the closed journal prefix."""
    closed = [event["record"] for event in records
              if event["kind"] == "action_outcome_closed"
              and event["sequence"] <= boundary["sequence"]]
    stored = saved.get("ledger")
    if type(stored) is list:
        if canonical_bytes(stored) != canonical_bytes(closed):
            raise WorkerResumeError("checkpoint ledger differs from closed journal actions")
        return closed, saved, None
    if (not isinstance(stored, dict) or stored.get("schema") not in _SLIM_LEDGER_SCHEMAS
            or type(stored.get("slim")) is not dict
            or type(stored.get("record_ids")) is not list
            or len(stored["record_ids"]) != len(closed)
            or any(type(ref) is not str or ref != record.get("step_record_id")
                   for ref, record in zip(stored["record_ids"], closed))):
        raise WorkerResumeError("checkpoint ledger differs from closed journal actions")
    # Existing replay validators need the full records. The authenticated
    # journal already owns them, so this transient view adds only references.
    replay_saved = dict(saved)
    replay_saved["ledger"] = closed
    return closed, replay_saved, stored.get("slim")


def _restore_ledger(ledger, closed, slim, directory):
    try:
        if slim is None:  # Only the explicitly authenticated legacy full-list envelope.
            ledger.restore_records(closed, directory=directory)
        else:
            ledger.restore_slim(slim, full_records=closed, directory=directory)
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("checkpoint slim ledger differs from closed journal actions") from error
    if len(ledger.records) != len(closed):
        raise WorkerResumeError("restored slim ledger lost closed actions")


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (set, frozenset, tuple, list)):
        return []
    return sorted({str(item) for item in value if str(item).strip()})


def _rerun_rows(value) -> list[dict[str, object]]:
    if not isinstance(value, Mapping):
        return []
    rows = []
    for key, count in value.items():
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            continue
        try:
            number = int(count)
        except (TypeError, ValueError):
            continue
        rows.append({"selector": str(key[0]), "revision": str(key[1]), "count": number})
    return sorted(rows, key=lambda row: (row["selector"], row["revision"]))


def first_green_checkpoint_section(runtime) -> dict[str, object] | None:
    """Encode the opt-in first-green/done policy state without JSON key tuples."""
    if getattr(runtime, "stateframe_first_green_disclosure_mode", "off") != "on":
        return None
    state = {
        "_fgd_challenge_revision": _plain(
            getattr(runtime, "_fgd_challenge_revision", None)),
        "_fgd_progress": bool(getattr(runtime, "_fgd_progress", False)),
        "_fgd_read_paths": _string_list(getattr(runtime, "_fgd_read_paths", None)),
        "_fgd_probe_fingerprints": _string_list(
            getattr(runtime, "_fgd_probe_fingerprints", None)),
        "_fgd_identical_reruns": _rerun_rows(
            getattr(runtime, "_fgd_identical_reruns", None)),
        "_fgd_disclosed_revisions": _string_list(
            getattr(runtime, "_fgd_disclosed_revisions", None)),
        "_done_challenge_revision": _plain(
            getattr(runtime, "_done_challenge_revision", None)),
        "_done_challenge_count": int(
            getattr(runtime, "_done_challenge_count", 0) or 0),
        "_done_challenge_overrides": int(
            getattr(runtime, "_done_challenge_overrides", 0) or 0),
        "_done_submission_record": _plain(
            getattr(runtime, "_done_submission_record", None)),
    }
    return {"schema": FIRST_GREEN_CHECKPOINT_SCHEMA, "state": state}


def restore_first_green_checkpoint(runtime, section) -> None:
    """Restore the opt-in policy section into a fresh worker runtime."""
    if section is None:
        return
    if getattr(runtime, "stateframe_first_green_disclosure_mode", "off") != "on":
        raise WorkerResumeError("first-green checkpoint in a disabled profile")
    if (not isinstance(section, Mapping)
            or section.get("schema") != FIRST_GREEN_CHECKPOINT_SCHEMA
            or not isinstance(section.get("state"), Mapping)):
        raise WorkerResumeError("invalid first-green policy checkpoint")
    state = section["state"]
    expected = set(_FGD_CHECKPOINT_FIELDS) | set(_DONE_CHALLENGE_CHECKPOINT_FIELDS)
    if set(state) != expected:
        raise WorkerResumeError("incomplete first-green policy checkpoint")
    rows = state["_fgd_identical_reruns"]
    if not isinstance(rows, list):
        raise WorkerResumeError("invalid first-green rerun checkpoint")
    reruns = {}
    for row in rows:
        if (not isinstance(row, Mapping)
                or set(row) != {"selector", "revision", "count"}
                or not isinstance(row["selector"], str)
                or not isinstance(row["revision"], str)
                or type(row["count"]) is not int
                or row["count"] < 0):
            raise WorkerResumeError("invalid first-green rerun row")
        reruns[(row["selector"], row["revision"])] = row["count"]
    for name in ("_fgd_read_paths", "_fgd_probe_fingerprints", "_fgd_disclosed_revisions"):
        if not isinstance(state[name], list) or any(not isinstance(item, str) for item in state[name]):
            raise WorkerResumeError("invalid first-green set checkpoint")
    if type(state["_fgd_progress"]) is not bool:
        raise WorkerResumeError("invalid first-green progress checkpoint")
    for name in ("_done_challenge_count", "_done_challenge_overrides"):
        if type(state[name]) is not int or state[name] < 0:
            raise WorkerResumeError("invalid done challenge checkpoint")
    submission = state["_done_submission_record"]
    if submission is not None and not isinstance(submission, Mapping):
        raise WorkerResumeError("invalid done submission checkpoint")

    setattr(runtime, "_fgd_challenge_revision",
            copy.deepcopy(state["_fgd_challenge_revision"]))
    setattr(runtime, "_fgd_progress", state["_fgd_progress"])
    setattr(runtime, "_fgd_read_paths", set(state["_fgd_read_paths"]))
    setattr(runtime, "_fgd_probe_fingerprints", set(state["_fgd_probe_fingerprints"]))
    setattr(runtime, "_fgd_identical_reruns", reruns)
    setattr(runtime, "_fgd_disclosed_revisions", set(state["_fgd_disclosed_revisions"]))
    setattr(runtime, "_done_challenge_revision",
            copy.deepcopy(state["_done_challenge_revision"]))
    setattr(runtime, "_done_challenge_count", state["_done_challenge_count"])
    setattr(runtime, "_done_challenge_overrides", state["_done_challenge_overrides"])
    setattr(runtime, "_done_submission_record", copy.deepcopy(submission))


# earlier run fix DONE (consult item 9): STATEFRAME_POST_GREEN_WORK_MENU state (edit cycles, pending
# verification, reopen sets) survives a worker restart; without it a restart replenished the
# post-green edit budget and dropped the pending re-verification.
POST_GREEN_CHECKPOINT_KEY = "post_green_work_menu"
POST_GREEN_CHECKPOINT_SCHEMA = "post-green-work-menu-checkpoint/1"
RESUME_WORK_CHECKPOINT_KEY = "resume_work_state"
RESUME_WORK_CHECKPOINT_SCHEMA = "stateframe-resume-work-state/2"
LEGACY_RESUME_WORK_CHECKPOINT_SCHEMA = "stateframe-resume-work-state/1"
_RESUME_WORK_HOST_FIELDS = (
    "assessments", "payload", "snapshot", "check_file", "repair_episode",
    "repair_packet_audit", "check_bundle_source", "retired_check_ids",
)
_RESUME_WORK_AUTO_VERIFY_FIELDS = (
    "pending_edit", "deferred_edit", "delivery", "pending_edit_id",
)


def post_green_checkpoint_section(runtime) -> dict[str, object] | None:
    if getattr(runtime, "stateframe_post_green_work_menu_mode", "off") != "on":
        return None
    from kernel_post_green_menu import state_snapshot
    return {"schema": POST_GREEN_CHECKPOINT_SCHEMA, "state": state_snapshot(runtime)}


def restore_post_green_checkpoint(runtime, section) -> None:
    if getattr(runtime, "stateframe_post_green_work_menu_mode", "off") != "on":
        raise WorkerResumeError("post-green checkpoint in a disabled profile")
    if (not isinstance(section, Mapping)
            or section.get("schema") != POST_GREEN_CHECKPOINT_SCHEMA
            or not isinstance(section.get("state"), Mapping)
            or not isinstance(section["state"].get("cycles", {}), Mapping)):
        raise WorkerResumeError("invalid post-green policy checkpoint")
    runtime._post_green_menu_state = copy.deepcopy(dict(section["state"]))
    runtime.journal.emit("post_green_menu_restored", producer="kernel_continuation_restart",
                         dispatch=False, state=copy.deepcopy(dict(section["state"])))


def _resume_work_assessment(value: object):
    if not isinstance(value, Mapping):
        raise WorkerResumeError("invalid resume-work obligation assessment")
    from kernel_obligation_assess import CheckReceipt, ObligationAssessment
    raw_receipts = value.get("receipts", ())
    if not isinstance(raw_receipts, (list, tuple)):
        raise WorkerResumeError("invalid resume-work obligation receipts")
    try:
        receipts = tuple(
            CheckReceipt.from_mapping(item) for item in raw_receipts
            if isinstance(item, Mapping))
        return ObligationAssessment(
            obligation_id=str(value.get("obligation_id", "")),
            verdict=str(value.get("verdict", "")),
            reason=str(value.get("reason", "")),
            check_ids=tuple(value.get("check_ids", ())), receipts=receipts,
            observed=copy.deepcopy(value.get("observed")),
            expected=copy.deepcopy(value.get("expected")),
            evidence_refs=tuple(value.get("evidence_refs", ())),
            clause=str(value.get("clause", "")), location_hint=value.get("location_hint"),
            coupled_obligations=tuple(value.get("coupled_obligations", ())),
            localization=copy.deepcopy(value.get("localization")),
            coverage_required=bool(value.get("coverage_required", False)),
            coverage_disposition=str(value.get("coverage_disposition", "REQUIRED")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerResumeError("invalid resume-work obligation assessment") from error


def resume_work_checkpoint_section(runtime) -> dict[str, object] | None:
    """Persist host adjudication, witness and deferred verification state."""
    host_state = {
        "assessments": [
            _plain(item.to_dict() if hasattr(item, "to_dict") else item)
            for item in getattr(runtime, "host_closure_assessments", ())
        ],
        "payload": _plain(getattr(runtime, "host_closure_payload", None)),
        "snapshot": _plain(getattr(runtime, "host_closure_snapshot", None)),
        "check_file": _plain(getattr(runtime, "host_closure_check_file", None)),
        "repair_episode": _plain(getattr(runtime, "host_repair_episode", None)),
        "repair_packet_audit": _plain(getattr(runtime, "host_repair_packet_audit", None)),
        "check_bundle_source": _plain(getattr(runtime, "host_check_bundle_source", None)),
        "retired_check_ids": sorted(
            str(item) for item in getattr(runtime, "host_retired_check_ids", ())
        ),
    }
    auto_verify = {
        "pending_edit": _plain(getattr(runtime, "_auto_verify_pending_edit", None)),
        "deferred_edit": _plain(getattr(runtime, "_auto_verify_deferred_edit", None)),
        "delivery": _plain(getattr(runtime, "_auto_verify_delivery", None)),
        "pending_edit_id": _plain(getattr(runtime, "_auto_verify_pending_edit_id", None)),
    }
    behavior_records = _plain(getattr(runtime, "behavior_witness_records", []))
    from kernel_completion_owner import snapshot as completion_owner_snapshot
    completion_owner = completion_owner_snapshot(runtime)
    host_values = tuple(host_state[key] for key in _RESUME_WORK_HOST_FIELDS)
    if not (any(value not in (None, [], {}, ()) for value in auto_verify.values())
            or any(value not in (None, [], {}, ()) for value in host_values)
            or behavior_records or completion_owner is not None):
        return None
    return {
        "schema": RESUME_WORK_CHECKPOINT_SCHEMA,
        "host_closure": host_state,
        "behavior_witness_records": behavior_records,
        "auto_verify": auto_verify,
        "completion_owner": completion_owner,
    }


def restore_resume_work_checkpoint(runtime, section) -> None:
    if section is None:
        return
    if not isinstance(section, Mapping):
        raise WorkerResumeError("invalid resume-work checkpoint")
    schema = section.get("schema")
    if schema not in {LEGACY_RESUME_WORK_CHECKPOINT_SCHEMA, RESUME_WORK_CHECKPOINT_SCHEMA}:
        raise WorkerResumeError("invalid resume-work checkpoint")
    expected_keys = {"schema", "host_closure", "behavior_witness_records", "auto_verify"}
    if schema == RESUME_WORK_CHECKPOINT_SCHEMA:
        expected_keys.add("completion_owner")
    if set(section) != expected_keys:
        raise WorkerResumeError("invalid resume-work checkpoint")
    host = section["host_closure"]
    auto_verify = section["auto_verify"]
    records = section["behavior_witness_records"]
    if (not isinstance(host, Mapping) or set(host) != set(_RESUME_WORK_HOST_FIELDS)
            or not isinstance(auto_verify, Mapping)
            or set(auto_verify) != set(_RESUME_WORK_AUTO_VERIFY_FIELDS)
            or not isinstance(records, list)
            or not isinstance(host["assessments"], list)
            or not isinstance(host["retired_check_ids"], list)):
        raise WorkerResumeError("incomplete resume-work checkpoint")
    assessments = tuple(_resume_work_assessment(item) for item in host["assessments"])
    runtime.host_closure_assessments = assessments
    runtime.host_closure_payload = copy.deepcopy(host["payload"])
    runtime.host_closure_snapshot = copy.deepcopy(host["snapshot"])
    runtime.host_closure_check_file = copy.deepcopy(host["check_file"])
    runtime.host_repair_episode = copy.deepcopy(host["repair_episode"])
    runtime.host_repair_packet_audit = copy.deepcopy(host["repair_packet_audit"])
    runtime.host_check_bundle_source = copy.deepcopy(host["check_bundle_source"])
    runtime.host_retired_check_ids = {str(item) for item in host["retired_check_ids"]}
    runtime.behavior_witness_records = copy.deepcopy(records)
    runtime._auto_verify_pending_edit = copy.deepcopy(auto_verify["pending_edit"])
    deferred_edit = copy.deepcopy(auto_verify["deferred_edit"])
    if isinstance(deferred_edit, Mapping):
        deferred_edit["selectors"] = tuple(deferred_edit.get("selectors", ()))
    runtime._auto_verify_deferred_edit = deferred_edit
    runtime._auto_verify_delivery = copy.deepcopy(auto_verify["delivery"])
    runtime._auto_verify_pending_edit_id = copy.deepcopy(auto_verify["pending_edit_id"])
    if schema == RESUME_WORK_CHECKPOINT_SCHEMA:
        try:
            from kernel_completion_owner import restore as restore_completion_owner
            restore_completion_owner(runtime, section["completion_owner"])
        except (TypeError, ValueError) as error:
            raise WorkerResumeError("invalid completion-owner checkpoint") from error
    runtime.journal.emit(
        "resume_work_state_restored", producer="kernel_continuation_restart",
        dispatch=False, assessment_count=len(assessments),
        behavior_witness_count=len(records),
        deferred_revalidation=auto_verify["deferred_edit"] is not None,
        completion_owner_restored=schema == RESUME_WORK_CHECKPOINT_SCHEMA,
    )


def checkpoint(runtime, messages, worker_state):
    if not enabled():
        return
    if not _valid_worker_state_fields(worker_state):
        raise WorkerResumeError("worker checkpoint fields are incomplete")
    if runtime.step_ledger.pending is not None:
        raise WorkerResumeError("cannot checkpoint an open action")
    projection = runtime.live_projection
    if projection is not None:
        projection.update(list(messages), runtime.owner.trace, runtime.owner.current_step,
                          runtime.binding(), runtime.test_receipts)
    from kernel_native_restart import snapshot
    from kernel_frame_series_replay import snapshot_series
    from kernel_working_notes_restart import snapshot as note_snapshot
    state = {
        "frame_series": snapshot_series(runtime.frame_series),
        "working_notes": note_snapshot(runtime.decision_state),
        "native": snapshot(getattr(runtime, "native_batch", None)),
        "generated": runtime.generated.checkpoint_snapshot() if runtime.generated is not None else None,
        "retained": runtime.retained.checkpoint_snapshot() if runtime.retained is not None else None,
        "schema": SCHEMA, "step": runtime.owner.current_step,
        "binding": runtime.binding(), "worker": _plain(worker_state),
        "owner": {key: _plain(getattr(runtime.owner, key))
                  for key in OWNER_FIELDS if hasattr(runtime.owner, key)},
        "runtime": {key: (runtime.micrograph_wire.checkpoint() if key == "micrograph_wire"
                          else _plain(getattr(runtime, key, None)
                                      if key == "_stable_wire_state" else getattr(runtime, key)))
                    for key in runtime_fields(runtime)},
        "ledger": _checkpoint_ledger(runtime.step_ledger),
        "actions": _plain(list(runtime.actions.actions)),
        "projection": None if projection is None else {
            key: _plain(getattr(projection, key)) for key in PROJECTION_FIELDS},
        "trimmed": sorted(runtime._trimmed),
        "conversation": checkpoint_conversation(messages),
        "owner_restore": capture_owner_restore(runtime),
    }
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    from kernel_read_lockout import enabled as lockout_enabled
    if bookmarks_enabled() or lockout_enabled():
        # This clock advances even when no read/bookmark state exists. Only the mechanisms that number
        # repeated reads need it; with both switches off the checkpoint stays byte-identical to 0.2.1.
        state["provider_request_count"] = getattr(runtime, "_frame_budget_request_count", 0)
    if bookmarks_enabled() and (
             getattr(runtime, "_code_bookmarks_active", False) is True
             or getattr(runtime, "_reread_result_truth", None)
             or getattr(runtime, "_read_contract_source_cache", None)
             or getattr(runtime, "_read_page_delivery_replay", None)
             or getattr(runtime, "_read_lockout_state", None)):
        from kernel_reread_result_truth import checkpoint_replay
        state["code_bookmark_replay"] = checkpoint_replay(runtime)
    policy = first_green_checkpoint_section(runtime)
    if policy is not None:
        state[FIRST_GREEN_CHECKPOINT_KEY] = policy
    post_green = post_green_checkpoint_section(runtime)
    if post_green is not None:
        state[POST_GREEN_CHECKPOINT_KEY] = post_green
    resume_work = resume_work_checkpoint_section(runtime)
    if resume_work is not None:
        state[RESUME_WORK_CHECKPOINT_KEY] = resume_work
    checkpoint_event = runtime.journal.emit(
        "continuation_worker_checkpoint", checkpoint=state, checkpoint_hash=digest(state))
    _register_checkpoint(runtime, checkpoint_event)


KEYED_DONE_TAIL = frozenset({
    "step_dispatch_binding", "action_receipt", "completion_certification",
    "continuation_public_certification", "submission_idempotency", "challenge_budget_state",
    "completion_reducer_decision",
})


def _keyed_done_tail(runtime, tail):
    """STATEFRAME_SUBMIT_IDEMPOTENT only: a crash anywhere inside a done step
    before its terminal record is safe to retry.  Before a durable intent no
    effect can have happened; after it, the retried done reconciles against the
    keyed destination.  Authorization rests on durable state, never on an audit
    event: the only effect a done tail may contain is a ``complete`` that carries
    a submission key.  Returns the ids of the covered entries, or ``None``."""
    if getattr(runtime, "stateframe_submit_idempotent_mode", "off") != "on":
        return None
    covered = set()
    started = False
    for entry in tail:
        kind = entry["kind"]
        if kind == "step_dispatch_binding":
            tool = entry.get("tool")
            if tool not in (None, "done"):
                return None
            started = started or tool == "done"
            covered.add(id(entry))
            continue
        if not started:
            continue
        if kind in KEYED_DONE_TAIL:
            covered.add(id(entry))
        elif kind == "parent_budget" and entry.get("category") == "tool_work":
            covered.add(id(entry))
        elif (kind == "continuation_transition" and entry.get("operation") == "complete"
              and isinstance(entry.get("data"), dict) and entry["data"].get("submission_key")):
            covered.add(id(entry))
        elif kind in SAFE_TAIL:
            continue        # ordinary safe entries keep their normal check
        else:
            return None
    return covered if started else None


def restore_provider_request_count(runtime: object, saved: Mapping[str, object]) -> int | None:
    """Restore the provider clock even when no bookmark replay was recorded."""
    request_count = saved.get("provider_request_count")
    if "provider_request_count" in saved:
        if type(request_count) is not int or not 0 <= request_count <= 1_000_000_000:
            raise WorkerResumeError("invalid provider request count")
        runtime._frame_budget_request_count = request_count
    return request_count


def restore_runtime(runtime):
    from kernel_true_resume import require_enabled
    try:
        require_enabled()
    except RuntimeError as error:
        raise WorkerResumeError(str(error)) from error
    from kernel_continuation_state import unresolved
    from kernel_live_projection import LiveProjection
    reconciled_events = _reconcile_external_effects(runtime)
    from kernel_candidate_restart_reconcile import enabled as candidate_reconcile_enabled
    if candidate_reconcile_enabled():
        from kernel_candidate_restart_reconcile import reconcile
        reconciled_events.update(reconcile(runtime))
    state = runtime.continuation.engine.state
    if any(unresolved(state, identity) for identity in state.intentions):
        raise WorkerResumeError("unresolved effect requires host reconciliation before worker resume")
    records = list(runtime.journal.records())
    try:
        saved = latest_checkpoint(records)
        event = next(record for record in reversed(records)
                     if record.get("kind") == "continuation_worker_checkpoint"
                     and record.get("checkpoint") == saved)
        require_snapshot(records, saved)
        conversation = restore_conversation(saved.get("conversation"))
        owner_restore = validate_owner_restore(saved.get("owner_restore"))
        restore_owner_state(runtime, owner_restore)
    except (ResumeError, StopIteration) as error:
        raise WorkerResumeError(str(error)) from error
    try:
        actual_binding = runtime.binding()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise WorkerResumeError("cannot verify retained workspace revision") from error
    if not workspace_revision_matches(saved.get("binding"), actual_binding):
        raise WorkerResumeError("retained workspace revision differs from checkpoint")
    if (type(saved.get("step")) is not int or saved["step"] < 0
            or (runtime.step_limit is not None and saved["step"] >= runtime.step_limit)
            or not _valid_worker_state_fields(saved.get("worker"))
            or not checkpoint_runtime_fields_match(runtime, saved.get("runtime", {}))
            or set(saved.get("owner", {})) - set(OWNER_FIELDS)):
        raise WorkerResumeError("invalid worker checkpoint")
    closed_ledger, replay_saved, slim_ledger = _journal_ledger(saved, records, event)
    runtime.resume_messages = conversation
    runtime.resume_owner_restore = owner_restore
    follow_up = getattr(runtime, "resume_follow_up", None)
    try:
        runtime.resume_messages, tail_follow_up_event = restore_tail_follow_up(
            runtime.resume_messages, records, event["sequence"], follow_up)
    except ResumeError as error:
        raise WorkerResumeError(str(error)) from error
    if tail_follow_up_event is not None:
        follow_up = None
    elif follow_up is not None:
        try:
            runtime.resume_messages = append_follow_up(runtime.resume_messages, follow_up)
        except ResumeError as error:
            raise WorkerResumeError(str(error)) from error
    tail = [e for e in records if e["sequence"] > event["sequence"]]
    if getattr(runtime, "stateframe_model_router_replay_mode", "off") == "on":
        try:
            restore_router_replay_tail(runtime, records, event["sequence"])
        except ValueError as error:
            raise WorkerResumeError(str(error)) from error
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        from kernel_challenge_budget import durable_state
        durable_budget = durable_state(
            runtime, records=records,
            checkpoint_state=saved.get("runtime", {}).get("_challenge_budget_state"),
        )
        if durable_budget is not None and durable_budget.get("terminal"):
            raise WorkerResumeError("durable terminal submission cannot resume worker")
    allowed_tail = safe_tail_for(runtime)
    keyed_done = _keyed_done_tail(runtime, tail)
    for entry in tail:
        if entry.get("event_id") in reconciled_events:
            continue
        if keyed_done is not None and id(entry) in keyed_done:
            continue
        rejection_reason = resume_tail_rejection_reason(
            [entry], runtime, allowed_tail=allowed_tail,
            native_tool_batches=runtime.journal.manifest["native_tool_batches"])
        if rejection_reason is not None:
            raise WorkerResumeError(rejection_reason)
        if entry["kind"] == "parent_budget" and entry["category"] in {"tool_work", "attempts", "challenges"}:
            raise WorkerResumeError("post-checkpoint dispatch cannot be retried")
    # A begun request consumes its decision even if no response was received.
    begun = any(e["kind"] == "request_submitted" or (
        e["kind"] == "parent_budget" and e["category"] == "model_calls" and not e["blocked"]) for e in tail)
    from kernel_native_restart import prepare_restore
    retained_receive = prepare_restore(runtime, saved, records, event)
    begun = begun and not retained_receive
    runtime.resume_step = saved["step"] + int(begun)
    if runtime.step_limit is not None and runtime.resume_step >= runtime.step_limit:
        raise WorkerResumeError("original worker step allocation is exhausted")
    if runtime.generated is not None:
        try:
            runtime.generated.restore_snapshot(saved.get("generated"), records, event["sequence"])
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerResumeError("invalid generated proposal restoration") from error
    elif saved.get("generated") is not None:
        raise WorkerResumeError("generated state in a disabled profile")
    if runtime.retained is not None:
        try:
            runtime.retained.restore_snapshot(saved.get("retained"), records, event["sequence"])
            if runtime.retained.checkpoint_snapshot() != saved.get("retained"):
                raise ValueError("retained restoration postcondition mismatch")
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerResumeError("invalid retained source restoration") from error
        runtime._retained_binding = copy.deepcopy(saved["binding"])
    elif saved.get("retained") is not None:
        raise WorkerResumeError("retained state in a disabled profile")
    from kernel_frame_series_replay import restore_series
    restore_series(runtime, replay_saved, records, event)
    from kernel_working_notes_restart import restore as restore_notes
    restore_notes(runtime, replay_saved, records, event)
    runtime.resume_worker_state = copy.deepcopy(saved["worker"])
    restore_runtime_field_values(runtime, saved["runtime"])
    request_count = restore_provider_request_count(runtime, saved)
    if "code_bookmark_replay" in saved:
        from kernel_code_bookmarks import enabled as bookmarks_enabled
        from kernel_reread_result_truth import restore_replay
        if bookmarks_enabled():
            try:
                replay = saved["code_bookmark_replay"]
                if (request_count is not None and isinstance(replay, Mapping)
                        and replay.get("request_number") != request_count):
                    raise ValueError("provider request count differs from bookmark replay")
                restore_replay(runtime, replay)
            except ValueError as error:
                raise WorkerResumeError("invalid bookmark read replay") from error
    reconcile_environment_notice_tail(runtime, tail, event["sequence"])
    if getattr(runtime, "stateframe_challenge_budget", None) is not None:
        from kernel_challenge_budget import restore_state
        restore_state(
            runtime,
            records=records,
            checkpoint_state=saved["runtime"].get("_challenge_budget_state"),
        )
    for key, value in saved["owner"].items():
        if key in {"_selector_error_fallback_files", "_futile_answered"}:
            value = set(value)
        elif key in {"f2p_ids", "f2p_names", "f2p_inworld", "oracle_file_ids"}:
            value = tuple(value)
        if key in {"trace", "_receipts"}:
            restored_history = (history(directory=runtime.journal.path.parent) if key == "trace"
                                else receipt_history(directory=runtime.journal.path.parent))
            restored_history.extend(copy.deepcopy(value))
            setattr(runtime.owner, key, restored_history)
        else:
            setattr(runtime.owner, key, copy.deepcopy(value))
    if getattr(runtime, "stateframe_preedit_coverage_arm", None) in {"treatment", "placebo"}:
        from kernel_preedit_coverage_runtime import restore as restore_preedit_coverage
        restore_preedit_coverage(runtime, records, saved["runtime"].get("_preedit_coverage_state"))
    if FIRST_GREEN_CHECKPOINT_KEY in saved:
        restore_first_green_checkpoint(runtime, saved[FIRST_GREEN_CHECKPOINT_KEY])
    if POST_GREEN_CHECKPOINT_KEY in saved:
        restore_post_green_checkpoint(runtime, saved[POST_GREEN_CHECKPOINT_KEY])
    if RESUME_WORK_CHECKPOINT_KEY in saved:
        restore_resume_work_checkpoint(runtime, saved[RESUME_WORK_CHECKPOINT_KEY])
    _restore_ledger(runtime.step_ledger, closed_ledger, slim_ledger,
                    runtime.journal.path.parent)
    if runtime.latest_exchange_mode != "off":
        from kernel_native_restart import latest_closed_exchange
        prefix = [entry for entry in records if entry["sequence"] <= event["sequence"]]
        identity, pair = latest_closed_exchange(prefix, closed_ledger)
        runtime.resume_latest_exchange = (identity, pair)
    runtime.actions.actions.extend(copy.deepcopy(saved["actions"]))
    runtime._trimmed = set(saved["trimmed"])
    initial = runtime.workspace.read("initialized.json")
    root = Path(runtime.owner.root)
    runtime.protected_tests = {root / name: sha for name, sha in initial["protected_tests"].items()}
    runtime.live_projection = LiveProjection(task_id=runtime.owner.task_id,
        symbol=runtime.owner.fn, path=runtime.target.relative_to(root).as_posix(),
        brief=runtime.information.ordinary_brief, selectors=(
            runtime.required_public_selectors()
            if getattr(runtime.owner, "task_row", {}).get("runner_kind") == "generic"
            else list(runtime.owner.f2p_ids)),
        binding=initial["binding"])
    projection = saved["projection"]
    if projection is not None:
        if (set(projection) != set(PROJECTION_FIELDS)
                or projection["initial_harness_revision"] != initial["binding"]["test_harness_revision"]
                or projection["initial_harness_files"] != initial["binding"]["harness_files"]):
            raise WorkerResumeError("checkpoint replaced original public acceptance identity")
        for key, value in projection.items():
            setattr(runtime.live_projection, key,
                    set(value) if key == "seen_trace_events" else copy.deepcopy(value))
    follow_up_event = tail_follow_up_event
    if follow_up is not None:
        try:
            follow_up_event = user_request_event(
                runtime.journal, follow_up,
                request_id=getattr(runtime, "resume_request_id", None))
        except ResumeError as error:
            raise WorkerResumeError(str(error)) from error
    if follow_up_event is not None:
        runtime.resume_follow_up_event = copy.deepcopy(follow_up_event)
    runtime.journal.emit(
        "continuation_resume_authorized",
        checkpoint_ref=event["event_id"],
        execution_authorized=True,
        authorization="explicit_resume_command",
        follow_up_event_id=(follow_up_event["event_id"] if follow_up_event is not None else None),
    )
    runtime.journal.emit(
        "continuation_owner_restore",
        checkpoint_ref=event["event_id"],
        owner_restore=owner_restore,
    )
    runtime.journal.emit("continuation_worker_restored", checkpoint_ref=event["event_id"],
        original_step=saved["step"], next_step=runtime.resume_step,
        discarded_unexecuted_request=begun, parent_budget=runtime.parent.snapshot(),
        previous_binding=saved["binding"], current_binding=runtime.binding(),
        restored_without_chat_history=False)
