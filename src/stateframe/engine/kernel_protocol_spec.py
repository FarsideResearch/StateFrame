"""Source-derived public contract for the StateFrame kernel protocol.

This module is an additive documentation surface.  It is not imported by the
request-preparation path, so an unset switch cannot change a normal request.
The release worker consumes the explicit ``publish_public_spec`` helper only
after its output directory is created.  ``build_spec`` reads the contract values
owned by the existing protocol modules; ``render_snapshot`` is used to produce
the checked-in, Python-only public snapshot.  The snapshot test is deliberately
exact: a changed event, switch, schema, or replay boundary must be regenerated
rather than silently leaving stale documentation behind.

The optional publication switch is ``STATEFRAME_KERNEL_PROTOCOL_SPEC``.  It
controls access to ``public_spec`` only.  It does not enable any of the
documented runtime switches and it never grants the model authority.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import pprint
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args


MODE_ENV = "STATEFRAME_KERNEL_PROTOCOL_SPEC"
SPEC_VERSION = "stateframe-kernel-protocol/1"

Authority = Literal["MODEL_PROPOSAL", "HOST_AUTHORITY", "HOST_OBSERVED"]


@dataclass(frozen=True)
class SwitchRef:
    """A code-owned environment control included in the public contract."""

    name: str
    module: str
    reader: str
    values: tuple[str, ...]
    default: str
    purpose: str
    env_symbol: str | None = None


@dataclass(frozen=True)
class EventRef:
    """A typed journal event family and its host-owned source."""

    kind: str
    authority: Authority
    role: str
    source: str


_SWITCH_REFS = (
    SwitchRef(
        "STATEFRAME_PAGE_LOG", "kernel_stateframe_page_log", "mode",
        ("off", "shadow", "on"), "off",
        "render typed cache pages and frame deltas inside C4c before sealing", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TOOL_STABLE", "kernel_stateframe_tool_stable", "tool_mode",
        ("off", "shadow", "on"), "off",
        "stabilize admitted tool order and descriptions; current guidance stays volatile", "TOOL_MODE_ENV",
    ),
    SwitchRef(
        MODE_ENV, __name__, "enabled", ("off", "on"), "off",
        "publish this generated documentation surface", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_ACTION_SCHEMA", "kernel_step_contract", "action_schema_enabled",
        ("off", "on"), "off", "strict visible JSON action envelope", "ACTION_SCHEMA_ENV",
    ),
    SwitchRef(
        "REACT_STEP_CONTRACT", "kernel_step_contract", "enabled", ("0", "1"), "0",
        "legacy state-id/tool/arguments step envelope",
    ),
    SwitchRef(
        "STATEFRAME_SYSTEM_PROMPT", "kernel_step_contract", "system_prompt_variant",
        ("v1", "v2", "v3"), "v1", "selected host-authored instruction variant", "SYSTEM_PROMPT_ENV",
    ),
    SwitchRef(
        "KERNEL_CONTINUATION_EXECUTIVE", "kernel_continuation_runtime", "enabled",
        ("0", "1"), "0", "live host continuation executive",
    ),
    SwitchRef(
        "KERNEL_CONTINUATION_RESTART", "kernel_continuation_restart", "enabled",
        ("0", "1"), "0", "trusted checkpoint restart and replay",
    ),
    SwitchRef(
        "KERNEL_INTERVENTIONS", "kernel_intervention_policy", "interventions_enabled",
        ("0", "1"), "1", "legacy experimental intervention consumers",
    ),
    SwitchRef(
        "STATEFRAME_SUBMIT_IDEMPOTENT", "kernel_submission_idempotency", "mode",
        ("off", "on"), "off", "keyed effect submission and reconciliation", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_DONE_IS_SUBMIT", "kernel_done_is_submit_authority", "enabled",
        ("off", "on"), "off", "seal and submit a captured candidate when done is invoked", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_DONE_ADVISORY", "kernel_done_is_submit_authority", "advisory_mode",
        ("off", "on"), "off",
        "report clause-coverage and unresolved-gating blockers as advisories on accepted done",
        "ADVISORY_ENV",
    ),
    SwitchRef(
        "STATEFRAME_REQUEST_TOOL_FLOOR", "kernel_request_tool_surface", "request_tool_floor_enabled",
        ("off", "on"), "off", "restore stable base tool schemas at the final request boundary",
        "REQUEST_TOOL_FLOOR_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TOOL_MENU_MEMO", "kernel_request_tool_surface", "_tool_menu_memo_mode",
        ("off", "shadow", "on"), "off",
        "reuse exact request-final tool surfaces; shadow recomputes and journals mismatches",
        "TOOL_MENU_MEMO_ENV",
    ),
    SwitchRef(
        "STATEFRAME_REQUEST_CONTRACT_REVISION_REBIND", "kernel_completion_owner",
        "_request_contract_revision_rebind_mode", ("off", "shadow", "on"), "off",
        "bind final request contracts to the current revision; shadow compares",
        "REQUEST_CONTRACT_REVISION_REBIND_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TERMINAL_LATTICE", "kernel_terminal_lattice", "enabled",
        ("off", "on"), "off",
        "monotone accepted, delivery, stop-reason, and artifact-digest terminal labels",
        "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TRUTHFUL_READ_PANELS", "kernel_request_diet", "truthful_read_panels_enabled",
        ("off", "on"), "off", "truthful dropped and deferred source-range panels",
        "TRUTHFUL_READ_PANELS_ENV",
    ),
    SwitchRef(
        "STATEFRAME_DIET_EVICT_DUPLICATED_ONLY", "kernel_request_diet",
        "duplicated_source_eviction_enabled", ("off", "on"), "off",
        "evict source tool results only when same-revision bytes remain on the request wire",
        "DUPLICATED_SOURCE_EVICTION_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TOOL_REPLY_CONSTRUCTOR", "kernel_tool_reply", "enabled",
        ("off", "on"), "off", "typed non-empty tool replies and fail-open call/result repair",
        "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_REMAINING_SITES", "kernel_remaining_sites", "enabled",
        ("off", "on"), "off", "live executable ticket-literal site panel", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_GREEN_NAG_SITES", "kernel_remaining_sites", "green_nag_enabled",
        ("off", "on"), "off",
        "gate post-green edit nags on a complete current-revision ticket-site inventory",
        "GREEN_NAG_MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_FRAMEFIT5_JOURNAL_ORIGIN", "kernel_continuation_journal",
        "framefit5_journal_origin_mode", ("off", "shadow", "on"), "off",
        "preserve pinned source-journal receipt identities in replay projections",
        "FRAMEFIT5_JOURNAL_ORIGIN_ENV",
    ),
    SwitchRef(
        "STATEFRAME_FRAMEFIT5_ACTION_CONTEXT", "kernel_stateframe_action_ledger",
        "framefit5_action_context_mode", ("off", "shadow", "on"), "off",
        "degrade an unresolvable latest file-action site as explicitly unverifiable",
        "FRAMEFIT5_ACTION_CONTEXT_ENV",
    ),
    SwitchRef(
        "STATEFRAME_FRAME_CONTINUITY", "kernel_stateframe_frame_continuity",
        "mode", ("off", "shadow", "on"), "off",
        "carry bounded live public state through the F4/FRAMEFIT5 projection",
        "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_HEADCARRY", "kernel_stateframe_headcarry", "framefit5_headcarry_mode",
        ("off", "shadow", "on"), "off",
        "carry current, bounded module heads for open and applied product files",
        "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_PROMPT_CACHE", "kernel_prompt_cache", "enabled",
        ("off", "on"), "off", "pinned prompt-cache configuration", "MODE_ENV",
    ),
    SwitchRef(
        "KERNEL_MODE", "kernel_effect_boundary", "resolve_mode",
        ("OFF", "SHADOW", "ADVISE", "ENFORCE", "BOTH"), "BOTH",
        "effect-boundary evaluation/advice/application axes",
    ),
    SwitchRef(
        "KERNEL_WORKING_VIEW", "kernel_step_contract", "working_view_mode",
        ("off", "facts", "resolved", "actions", "decision", "findings"), "off",
        "host-selected operational working-view projection",
    ),
    SwitchRef(
        "STATEFRAME_PROVIDER_ADAPTER", "kernel_provider_adapter", "adapter_mode",
        ("off", "on"), "off", "provider/native message contract", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_MODEL_ROUTER_REPLAY", "kernel_stateframe_model_router_replay", "replay_mode",
        ("off", "on"), "off", "durable model-router lease replay", "REPLAY_ENV",
    ),
    SwitchRef(
        "STATEFRAME_MODEL_HOPPING_COMPARISON", "kernel_model_hopping_comparison", "comparison_mode",
        ("off", "on"), "off", "opt-in model-hopping comparison specification", "COMPARISON_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TEMPLATE_CONFORMANCE", "kernel_template_conformance", "mode",
        ("off", "on"), "off", "endpoint-rendered template conformance evidence", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_CHECK_EXECUTION_BINDING", "kernel_check_receipts",
        "check_execution_binding_mode", ("off", "shadow", "on"), "off",
        "pin check receipts to one detached pre-launch product binding and selector scope",
    ),
    SwitchRef(
        "STATEFRAME_CHECK_RECEIPT_ENRICHMENT_DEDUP", "kernel_check_receipts",
        "check_receipt_enrichment_dedup_mode", ("off", "shadow", "on"), "off",
        "suppress only fully equal enrichment re-emits for one check execution",
    ),
    SwitchRef(
        "KERNEL_STAGNATION_CAP", "kernel_stagnation_cap", "mode",
        ("off", "on"), "off", "bounded host-observed stagnation stop control", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_SUPERVISED_EXECUTOR", "kernel_supervised_executor", "mode",
        ("off", "on"), "off", "bounded supervised command execution", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_FORK_A", "kernel_stateframe_fork_a", "mode",
        ("off", "on"), "off", "checkpoint-derived parent-admitted workers", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_COMPARISON_ADMISSION", "kernel_comparison_admission", "enabled",
        ("off", "on"), "off", "information-balanced comparison admission", "SWITCH",
    ),
    SwitchRef(
        "CAPABILITY_HOST_CAPACITY", "capability_host_capacity", "mode",
        ("off", "on"), "off", "opt-in host-capacity worker admission", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_LEAN", "kernel_lean", "enabled",
        ("off", "on"), "off", "plain-loop request shaping with truth gates only", "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_RESTATE_REQUIREMENT", "kernel_restate_requirement", "mode",
        ("off", "on", "auto"), "auto",
        "LEAN default restatement for declared public scopes with at least two files",
        "ENV",
    ),
    SwitchRef(
        "STATEFRAME_LEAN_STACK_RESTATE_REQUIREMENT", "kernel_restate_requirement",
        "legacy_mode", ("off", "on"), "off",
        "legacy force-on alias for STATEFRAME_RESTATE_REQUIREMENT", "LEGACY_ENV",
    ),
    SwitchRef(
        "STATEFRAME_LEAN_STACK_WIDE_TARGET_WINDOW", "kernel_lean",
        "wide_target_window_stack_enabled", ("off", "on"), "on",
        "C5 current edit-target source window under LEAN",
    ),
    SwitchRef(
        "STATEFRAME_LEAN_STACK_SIBLING_SITE_FACTS", "kernel_lean",
        "sibling_site_facts_stack_enabled", ("off", "on"), "off",
        "experimental current call-site facts from task-declared files",
    ),
    SwitchRef(
        "STATEFRAME_STALL_OBSERVATION", "kernel_lean", "stall_observation_enabled",
        ("off", "on"), "off", "one bounded observation after three unproductive source queries",
        "STALL_ENV",
    ),
    SwitchRef(
        "STATEFRAME_TEST_RUN_SPANS", "kernel_test_run_spans", "mode",
        ("off", "shadow", "on"), "off",
        "journal monotonic pytest preparation, child, receipt, and enrichment spans without changing test results",
        "MODE_ENV",
    ),
    SwitchRef(
        "STATEFRAME_PUBLIC_TEST_VERDICT_FACT", "kernel_stateframe_runtime",
        "public_test_verdict_fact_mode", ("off", "shadow", "on"), "off",
        "pure public-test binding normalization with compiled matching, content reuse, and legacy parity shadow",
        "PUBLIC_TEST_VERDICT_FACT_ENV",
    ),
    SwitchRef(
        "STATEFRAME_CENSUS_VALIDATION_CAP_MIB", "kernel_census_index_reuse",
        "validation_source_cap_mib", tuple(str(value) for value in range(257)), "32",
        "source-byte validation cap in MiB, with a 256 MiB hard ceiling",
        "VALIDATION_CAP_MIB_ENV",
    ),
    SwitchRef(
        "STATEFRAME_SOURCE_PAGE_HEADER_V2", "kernel_stateframe_f4", "source_page_header_v2_mode",
        ("off", "on"), "off",
        "parse requested/effective source-page headers while keeping host trailers separate",
        "SOURCE_PAGE_HEADER_V2_ENV",
    ),
)


_UNSUPPORTED_COMBINATIONS = (
    {
        "when": {
            "KERNEL_CONTINUATION_RESTART": "1",
            "KERNEL_CONTINUATION_EXECUTIVE": "0",
        },
        "result": "rejected",
        "reason": "worker restart requires both explicit continuation switches",
        "source": "kernel_continuation_restart.validate_profile",
    },
    {
        "when": {
            "KERNEL_CONTINUATION_RESTART": "1",
            "continuation_effect": "unresolved",
        },
        "result": "rejected",
        "reason": "unresolved effect requires host reconciliation before worker resume",
        "source": "kernel_continuation_restart.restore_runtime",
    },
    {
        "when": {
            "STATEFRAME_SUBMIT_IDEMPOTENT": "on",
            "challenge_budget": "absent",
        },
        "result": "rejected",
        "reason": "exactly-once submission requires the challenge-budget journal",
        "source": "kernel_submission_idempotency.validate_configuration",
    },
    {
        "when": {
            "KERNEL_CONTINUATION_RESTART": "1",
            "KERNEL_INTERVENTIONS": "1",
        },
        "result": "rejected",
        "reason": "legacy organ replay is not implemented for worker restart",
        "source": "kernel_continuation_restart.validate_profile",
    },
    {
        "when": {
            "STATEFRAME_ACTION_SCHEMA": "on",
            "transport": "native tool calls",
        },
        "result": "unsupported",
        "reason": "the visible action envelope is content-only and removes native tools",
        "source": "kernel_step_contract.configure_action_envelope",
    },
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict documentation-publication mode."""
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _module(name: str) -> object:
    return importlib.import_module(name)


def _reader(module: object, path: str) -> object:
    value = module
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _reference(path: str) -> object:
    """Resolve a public source reference and fail closed if it disappeared."""
    parts = path.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise RuntimeError(f"invalid protocol source reference: {path!r}")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        attribute_path = ".".join(parts[split:])
        try:
            module = _module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name or module_name.startswith(f"{error.name}."):
                continue
            raise RuntimeError(f"protocol source reference is unavailable: {path}") from error
        try:
            return _reader(module, attribute_path)
        except AttributeError as error:
            raise RuntimeError(f"protocol source reference is unavailable: {path}") from error
    raise RuntimeError(f"protocol source reference is unavailable: {path}")


def _normalise_default(value: object, values: tuple[str, ...], *, name: str) -> str:
    """Convert a reader's native default into its documented wire spelling."""
    if isinstance(value, bool):
        candidates = ("on", "off") if set(values) == {"on", "off"} else ("1", "0")
        value = candidates[0] if value else candidates[1]
    elif isinstance(value, int) and not hasattr(value, "value"):
        value = str(value)
    elif hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        value = getattr(value, "value")
    if not isinstance(value, str) or value not in values:
        raise RuntimeError(f"protocol switch {name} returned undocumented default {value!r}")
    return value


def _reader_default(ref: SwitchRef, module: object, reader: object) -> str:
    """Read a switch default without allowing the caller's environment to leak in."""
    if ref.name == "KERNEL_MODE":
        default = getattr(module, "DEFAULT_MODE", None)
        return _normalise_default(default, ref.values, name=ref.name)
    if not callable(reader):
        raise RuntimeError(f"protocol switch reader is not callable: {ref.module}.{ref.reader}")
    try:
        parameters = inspect.signature(reader).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"protocol switch reader has no inspectable signature: {ref.name}") from error
    required = [
        parameter for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and parameter.name != "environ"
    ]
    if required:
        raise RuntimeError(
            f"protocol switch reader requires unsupported arguments: {ref.name}"
        )
    if "environ" in parameters:
        default_environ = (
            {"STATEFRAME_LEAN": "on"}
            if ref.name.startswith("STATEFRAME_LEAN_STACK_") else {}
        )
        value = reader(default_environ)
    else:
        previous = os.environ.pop(ref.name, None)
        try:
            value = reader()
        finally:
            if previous is not None:
                os.environ[ref.name] = previous
    return _normalise_default(value, ref.values, name=ref.name)


def _switch_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ref in _SWITCH_REFS:
        try:
            module = _module(ref.module)
        except ModuleNotFoundError as error:
            if error.name != ref.module:
                raise
            rows.append({
                "name": ref.name,
                "default": ref.default,
                "values": list(ref.values),
                "reader": f"{ref.module}.{ref.reader}",
                "purpose": ref.purpose,
                "available": False,
            })
            continue
        reader = _reader(module, ref.reader)
        if not callable(reader):
            raise RuntimeError(f"protocol switch reader is not callable: {ref.module}.{ref.reader}")
        actual_name = getattr(module, ref.env_symbol) if ref.env_symbol else ref.name
        if actual_name != ref.name:
            raise RuntimeError(
                f"protocol switch name drift: {ref.module}.{ref.env_symbol}={actual_name!r} "
                f"but spec expects {ref.name!r}"
            )
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            source = ""
        if source and ref.name not in source:
            raise RuntimeError(f"protocol switch disappeared from source: {ref.name}")
        actual_default = _reader_default(ref, module, reader)
        if actual_default != ref.default:
            raise RuntimeError(
                f"protocol switch default drift: {ref.name}={actual_default!r}; "
                f"spec registry expects {ref.default!r}"
            )
        rows.append({
            "name": ref.name,
            "default": actual_default,
            "values": list(ref.values),
            "reader": f"{ref.module}.{ref.reader}",
            "purpose": ref.purpose,
            "available": True,
        })
    return rows


def _event_rows() -> list[dict[str, object]]:
    evidence = _module("kernel_continuation_evidence")
    restart = _module("kernel_continuation_restart")
    roles: dict[str, set[str]] = {}
    for kind in evidence.EVIDENCE_KINDS:
        roles.setdefault(kind, set()).add("evidence")
    for kind in restart.SAFE_TAIL:
        roles.setdefault(kind, set()).add("replay_safe_tail")
    for kind in restart.CHALLENGE_BUDGET_SAFE_TAIL:
        roles.setdefault(kind, set()).add("challenge_budget_safe_tail")
    for kind in restart.KEYED_DONE_TAIL:
        roles.setdefault(kind, set()).add("keyed_done_tail")
    explicit = {
        "done_advisory_downgraded": EventRef(
            "done_advisory_downgraded", "HOST_OBSERVED", "done_gate_downgraded_to_advisory",
            "kernel_done_is_submit_authority.capture_advisory_downgrades",
        ),
        "terminal_lower_refused": EventRef(
            "terminal_lower_refused", "HOST_AUTHORITY", "terminal_label_lowering_refused",
            "kernel_terminal_lattice.join_terminal",
        ),
        "tool_reply_filled": EventRef(
            "tool_reply_filled", "HOST_OBSERVED", "nonempty_tool_call_pair",
            "kernel_tool_reply.compose_tool_replies",
        ),
        "tool_reply_receipt": EventRef(
            "tool_reply_receipt", "HOST_OBSERVED", "typed_tool_outcome",
            "kernel_tool_reply.record_and_strip_receipts",
        ),
        "tool_reply_duplicate_removed": EventRef(
            "tool_reply_duplicate_removed", "HOST_OBSERVED", "unique_reply_per_call_id",
            "kernel_tool_reply.compose_tool_replies",
        ),
        "tool_reply_composition_error": EventRef(
            "tool_reply_composition_error", "HOST_OBSERVED", "fail_open_composition_fault",
            "kernel_tool_reply.compose_tool_replies",
        ),
        "tool_menu_memo_shadow_mismatch": EventRef(
            "tool_menu_memo_shadow_mismatch", "HOST_OBSERVED", "tool_menu_memo_shadow_parity_mismatch",
            "kernel_request_tool_surface.author_final_tool_surface",
        ),
        "request_contract_revision_rebind_shadow_mismatch": EventRef(
            "request_contract_revision_rebind_shadow_mismatch", "HOST_OBSERVED",
            "request_contract_revision_shadow_parity_mismatch",
            "kernel_completion_owner.bind_final_tool_surface",
        ),
        "provider_boundary_host_stop": EventRef(
            "provider_boundary_host_stop", "HOST_AUTHORITY",
            "provider_boundary_host_stop_route", "probes.react_repair.work",
        ),
        "provider_request_not_dispatched": EventRef(
            "provider_request_not_dispatched", "HOST_AUTHORITY",
            "provider_request_dispatch_disposition",
            "kernel_completion_owner.record_provider_request_not_dispatched",
        ),
        "request_final_wire_prepared": EventRef(
            "request_final_wire_prepared", "HOST_AUTHORITY",
            "provider_request_final_wire_shape",
            "kernel_measurement.RequestCapture._send_transport_with_retries",
        ),
        "stateframe_page_log_failed_open": EventRef(
            "stateframe_page_log_failed_open", "HOST_OBSERVED", "cache_layout_fail_open_vital",
            "kernel_cache_layout_candidate.render_cache_candidate",
        ),
        "stateframe_page_log_prepared": EventRef(
            "stateframe_page_log_prepared", "HOST_OBSERVED", "cache_layout_candidate_projection",
            "kernel_cache_layout_candidate.render_cache_candidate",
        ),
        "request_body": EventRef(
            "request_body", "HOST_AUTHORITY", "provider_request_body",
            "kernel_measurement.RequestCapture._send_transport_with_retries",
        ),
        "run_started": EventRef(
            "run_started", "HOST_AUTHORITY", "journal_identity",
            "kernel_continuation_journal.ContinuationJournal.__init__",
        ),
        "runtime_mode_snapshot": EventRef(
            "runtime_mode_snapshot", "HOST_AUTHORITY", "effective_runtime_configuration",
            "kernel_measurement.AuditJournal.__init__",
        ),
        "outline_parser_capability": EventRef(
            "outline_parser_capability", "HOST_AUTHORITY", "effective_outline_parser_capability",
            "kernel_continuation_journal.ContinuationJournal._record_outline_parser_capability",
        ),
        "read_budget_admitted": EventRef(
            "read_budget_admitted", "HOST_OBSERVED", "governed_read_admission",
            "kernel_idle_governor.IdleGovernor.should_allow",
        ),
        "read_reuse_visibility_checked": EventRef(
            "read_reuse_visibility_checked", "HOST_OBSERVED", "serialized_source_visibility",
            "kernel_measurement.RequestCapture.send",
        ),
        "read_reuse_source_snapshot_reconciled": EventRef(
            "read_reuse_source_snapshot_reconciled", "HOST_OBSERVED",
            "exact_current_source_snapshot", "kernel_read_reuse_visibility.prepare_final_visibility",
        ),
        "read_repeat_assessed": EventRef(
            "read_repeat_assessed", "HOST_OBSERVED", "exact_read_repeat_detection",
            "kernel_read_once.note_for",
        ),
        "read_once_note_queued": EventRef(
            "read_once_note_queued", "HOST_OBSERVED", "read_once_tool_result_notice",
            "kernel_read_once.note_for",
        ),
        "read_once_wire_note_reconciled": EventRef(
            "read_once_wire_note_reconciled", "HOST_OBSERVED", "serialized_note_presence",
            "kernel_read_once.ensure_wire_note",
        ),
        "request_source_claim_deferred": EventRef(
            "request_source_claim_deferred", "HOST_AUTHORITY", "verified_source_deferral",
            "kernel_read_reuse_visibility.defer_unproven_source_claims",
        ),
        "continuation_writer_resumed": EventRef(
            "continuation_writer_resumed", "HOST_AUTHORITY", "journal_identity",
            "kernel_continuation_journal.ContinuationJournal.resume",
        ),
        "continuation_worker_checkpoint": EventRef(
            "continuation_worker_checkpoint", "HOST_AUTHORITY", "checkpoint",
            "kernel_continuation_restart.checkpoint",
        ),
        "continuation_transition": EventRef(
            "continuation_transition", "HOST_AUTHORITY", "state_transition",
            "kernel_continuation_executive.ContinuationExecutive.apply",
        ),
        "continuation_stranded": EventRef(
            "continuation_stranded", "HOST_AUTHORITY", "unresolved_effect",
            "kernel_continuation_runtime.LiveContinuation._record_stranded",
        ),
        "continuation_worker_restored": EventRef(
            "continuation_worker_restored", "HOST_AUTHORITY", "checkpoint",
            "kernel_continuation_restart.restore_runtime",
        ),
        "framefit5_journal_origin_shadow_mismatch_event_id": EventRef(
            "framefit5_journal_origin_shadow_mismatch_event_id", "HOST_OBSERVED",
            "replay_receipt_identity_comparison",
            "kernel_continuation_journal.ContinuationJournal.framefit_receipt_records",
        ),
        "framefit5_journal_origin_shadow_mismatch_sequence": EventRef(
            "framefit5_journal_origin_shadow_mismatch_sequence", "HOST_OBSERVED",
            "replay_receipt_identity_comparison",
            "kernel_continuation_journal.ContinuationJournal.framefit_receipt_records",
        ),
        "framefit5_journal_origin_shadow_mismatch_source_identity": EventRef(
            "framefit5_journal_origin_shadow_mismatch_source_identity", "HOST_OBSERVED",
            "replay_receipt_identity_comparison",
            "kernel_continuation_journal.ContinuationJournal.framefit_receipt_records",
        ),
        "framefit5_action_context_degraded": EventRef(
            "framefit5_action_context_degraded", "HOST_OBSERVED",
            "unverifiable_file_action_context",
            "kernel_stateframe_runtime.StateFrameRuntime.prepare",
        ),
        "framefit5_action_context_shadow_mismatch_open_edit_site": EventRef(
            "framefit5_action_context_shadow_mismatch_open_edit_site", "HOST_OBSERVED",
            "action_context_shadow_comparison",
            "kernel_stateframe_action_ledger.build_framefit_action_context",
        ),
        "framefit5_action_context_shadow_mismatch_site_context": EventRef(
            "framefit5_action_context_shadow_mismatch_site_context", "HOST_OBSERVED",
            "action_context_shadow_comparison",
            "kernel_stateframe_action_ledger.build_framefit_action_context",
        ),
        "framefit5_action_context_shadow_mismatch_build_result": EventRef(
            "framefit5_action_context_shadow_mismatch_build_result", "HOST_OBSERVED",
            "action_context_shadow_comparison",
            "kernel_stateframe_action_ledger.build_framefit_action_context",
        ),
        "framefit5_headcarry_rendered": EventRef(
            "framefit5_headcarry_rendered", "HOST_OBSERVED",
            "current_module_head_projection", "kernel_stateframe_headcarry.render_headcarry",
        ),
        "framefit5_headcarry_shadow_mismatch_heads": EventRef(
            "framefit5_headcarry_shadow_mismatch_heads", "HOST_OBSERVED",
            "headcarry_shadow_comparison", "kernel_stateframe_headcarry.render_headcarry",
        ),
        "framefit5_headcarry_shadow_mismatch_omitted_count": EventRef(
            "framefit5_headcarry_shadow_mismatch_omitted_count", "HOST_OBSERVED",
            "headcarry_shadow_comparison", "kernel_stateframe_headcarry.render_headcarry",
        ),
        "framefit5_headcarry_shadow_mismatch_deduplicated_count": EventRef(
            "framefit5_headcarry_shadow_mismatch_deduplicated_count", "HOST_OBSERVED",
            "headcarry_shadow_comparison", "kernel_stateframe_headcarry.render_headcarry",
        ),
        "framefit5_headcarry_shadow_mismatch_byte_count": EventRef(
            "framefit5_headcarry_shadow_mismatch_byte_count", "HOST_OBSERVED",
            "headcarry_shadow_comparison", "kernel_stateframe_headcarry.render_headcarry",
        ),
        "check_execution_binding_shadow_mismatch": EventRef(
            "check_execution_binding_shadow_mismatch", "HOST_OBSERVED",
            "pre_launch_check_binding_shadow_difference",
            "kernel_check_receipts.record_check_output",
        ),
        "check_receipt_enrichment_dedup_shadow_mismatch": EventRef(
            "check_receipt_enrichment_dedup_shadow_mismatch", "HOST_OBSERVED",
            "fully_equal_enrichment_would_be_suppressed",
            "kernel_check_receipts.record_check_output",
        ),
        "test_run_profile": EventRef(
            "test_run_profile", "HOST_OBSERVED", "monotonic_public_pytest_timing",
            "kernel_test_run_spans.TestRunProfile.finalize",
        ),
        "test_run_spans_shadow_mismatch": EventRef(
            "test_run_spans_shadow_mismatch", "HOST_OBSERVED",
            "legacy_and_profiled_pytest_observation_agreement",
            "kernel_test_run_spans.TestRunProfile._compare_shadow",
        ),
        "public_test_verdict_fact_normalized": EventRef(
            "public_test_verdict_fact_normalized", "HOST_OBSERVED", "optimized_binding_normalization",
            "kernel_stateframe_runtime.StateFrameRuntime._public_test_verdict_uncached",
        ),
        "public_test_verdict_fact_shadow_checked": EventRef(
            "public_test_verdict_fact_shadow_checked", "HOST_OBSERVED", "pure_normalization_parity",
            "kernel_stateframe_runtime.StateFrameRuntime._public_test_verdict_uncached",
        ),
        "public_test_verdict_shadow_mismatch": EventRef(
            "public_test_verdict_shadow_mismatch", "HOST_OBSERVED", "normalized_binding_field_mismatch",
            "kernel_stateframe_runtime.StateFrameRuntime._public_test_verdict_uncached",
        ),
        "public_test_verdict_fact_bypass": EventRef(
            "public_test_verdict_fact_bypass", "HOST_OBSERVED", "optimized_normalization_fail_open",
            "kernel_stateframe_runtime.StateFrameRuntime._public_test_verdict_uncached",
        ),
    }
    rows: list[dict[str, object]] = []
    for kind in sorted(set(roles) | set(explicit)):
        if kind in explicit:
            event = explicit[kind]
            role, source, authority = event.role, event.source, event.authority
        elif "evidence" in roles[kind]:
            role, source = "evidence", "kernel_continuation_evidence.EVIDENCE_KINDS"
            authority = "HOST_OBSERVED"
        elif "challenge_budget_safe_tail" in roles[kind]:
            role, source = (
                "challenge_budget_safe_tail",
                "kernel_continuation_restart.CHALLENGE_BUDGET_SAFE_TAIL",
            )
            authority = "HOST_AUTHORITY"
        elif "keyed_done_tail" in roles[kind]:
            role, source = "keyed_done_tail", "kernel_continuation_restart.KEYED_DONE_TAIL"
            authority = "HOST_AUTHORITY"
        else:
            role, source = "replay_safe_tail", "kernel_continuation_restart.SAFE_TAIL"
            authority = "HOST_AUTHORITY"
        _reference(source)
        rows.append({
            "kind": kind,
            "authority": authority,
            "roles": sorted(roles.get(kind, set()) | {role}),
            "source": source,
        })
    return rows


def _enum_values(enum_type: object) -> list[str]:
    return [item.value for item in enum_type]


def _dataclass_fields(cls: object) -> list[str]:
    from dataclasses import fields

    return [item.name for item in fields(cls)]


def _model_proposal() -> dict[str, object]:
    step = _module("kernel_step_contract")
    return {
        "authority": {
            "model": [
                "propose one action and its arguments",
                "supply done.claim from the host-owned claim vocabulary",
                "supply done.evidence_refs as references, never as proof",
            ],
            "host": [
                "construct the live tool menu",
                "validate the action schema and known selectors",
                "bind the proposal to the current revision",
                "dispatch, observe, receipt, and completion decisions",
            ],
            "rule": "syntactic validity is not execution authority",
        },
        "switch": step.ACTION_SCHEMA_ENV,
        "schema_name": step.ACTION_SCHEMA_NAME,
        "envelope": {
            "decoder": "kernel_step_contract.decode_action_envelope",
            "schema_builder": "kernel_step_contract.action_schema",
            "required": ["action", "args"],
            "additional_properties": False,
            "action_args_are_correlated": True,
        },
        "done_claims": list(step.ACTION_CLAIMS),
    }


def _action_receipts() -> dict[str, object]:
    evidence = _module("kernel_continuation_evidence")
    legacy = _module("detreason.receipt_contract")
    for source in (
        "kernel_continuation_evidence.HostEvidence.observe",
        "kernel_continuation_evidence.HostEvidence._closed",
        "kernel_bolt.KernelBolt.receipt",
        "kernel_stateframe_runtime.StateFrameRuntime.close_step",
    ):
        _reference(source)
    return {
        "bindings": [
            "step_dispatch_binding",
            "action_receipt",
            "action_outcome_closed",
        ],
        "live_event": "action_receipt",
        "live_record_schema": "stateframe-audit/1",
        "live_receipt_fields": [
            "receipt_id", "kind", "result_class", "detail", "turn",
            "workspace_revision", "run_id", "request_id",
        ],
        "outcome_event": "action_outcome_closed",
        "outcome_record_fields": [
            "tool", "step", "result", "receipt_ids", "before_binding",
            "after_binding", "observation", "changes", "step_record_id",
        ],
        "evidence_kinds": sorted(evidence.EVIDENCE_KINDS),
        "legacy_schema": legacy.PROCEDURE_RECEIPT_SCHEMA_VERSION,
        "legacy_mapping": {
            "run_id": "live receipt.run_id / journal.run_id",
            "task_id": "journal.task_id",
            "attempts": "action_outcome_closed.record and step ledger history",
            "observations": "action_outcome_closed.record.observation",
            "evidence": "action_outcome_closed.record.receipt_ids",
        },
        "authority_note": (
            "A receipt records host-observed execution. It does not promote model prose, "
            "a claim, or a legacy procedure receipt into authority."
        ),
    }


def _journal_replay() -> dict[str, object]:
    journal = _module("kernel_continuation_journal")
    restart = _module("kernel_continuation_restart")
    _reference("kernel_continuation_journal.ContinuationJournal.records")
    _reference("kernel_continuation_journal.ContinuationJournal.__init__")
    _reference("kernel_continuation_journal.ContinuationJournal.resume")
    return {
        "record_schema": "stateframe-audit/1",
        "journal_format": journal.FORMAT,
        "max_record_bytes": journal.MAX_RECORD_BYTES,
        "reserved_fields": sorted(journal.RESERVED),
        "first_event": "run_started",
        "event_id": "{run_id}:{sequence}",
        "lineage_fields": [
            "schema", "kind", "run_id", "task_id", "event_id", "sequence",
            "timestamp_ns", "journal_format", "previous_hash", "record_hash",
        ],
        "replay_call": "kernel_continuation_journal.ContinuationJournal.records",
        "replay_guards": [
            "canonical JSON object per line",
            "contiguous sequence and run/task identity",
            "previous_hash and record_hash chain",
            "trusted writer ownership and no torn record",
        ],
        "safe_tail": sorted(restart.SAFE_TAIL),
        "challenge_budget_safe_tail": sorted(restart.CHALLENGE_BUDGET_SAFE_TAIL),
        "prepare_transitions": sorted(restart.PREPARE_TRANSITIONS),
        "keyed_done_tail": sorted(restart.KEYED_DONE_TAIL),
    }


def _continuation() -> dict[str, object]:
    state = _module("kernel_continuation_state")
    restart = _module("kernel_continuation_restart")
    for source in (
        "kernel_continuation_state.transition",
        "kernel_continuation_state.unresolved",
        "kernel_continuation_restart.restore_runtime",
    ):
        _reference(source)
    return {
        "phase_type": "kernel_continuation_state.Phase",
        "phases": sorted(get_args(state.Phase)),
        "terminal_phases": sorted(state.TERMINAL),
        "effect_status_type": "kernel_continuation_state.EffectStatus",
        "effect_statuses": sorted(get_args(state.EffectStatus)),
        "typed_state_fields": {
            "Wake": _dataclass_fields(state.Wake),
            "Intention": _dataclass_fields(state.Intention),
            "Effect": _dataclass_fields(state.Effect),
            "ContinuationState": _dataclass_fields(state.ContinuationState),
        },
        "transition_call": "kernel_continuation_state.transition",
        "unresolved_query": "kernel_continuation_state.unresolved",
        "restart_call": "kernel_continuation_restart.restore_runtime",
        "checkpoint_schema": restart.SCHEMA,
        "checkpoint_fields": {
            "worker": list(restart.WORKER_FIELDS),
            "owner": list(restart.OWNER_FIELDS),
            "runtime": list(restart.RUNTIME_FIELDS),
        },
        "unresolved_effect_policy": (
            "restore refuses resume until the host reconciles every unresolved effect"
        ),
    }


def _host_authority() -> dict[str, object]:
    boundary = _module("kernel_effect_boundary")
    _reference("kernel_effect_boundary.authorize")
    return {
        "modes": _enum_values(boundary.Mode),
        "default_mode": boundary.DEFAULT_MODE.value,
        "axes": {
            mode.value: list(boundary.AXES[mode]) for mode in boundary.Mode
        },
        "consequence_types": _enum_values(boundary.ConsequenceType),
        "component_kinds": _enum_values(boundary.ComponentKind),
        "decision_source": "kernel_effect_boundary.Decision",
        "rule": "evaluators propose; authorize decides whether effects are applied",
    }


def _executable_example() -> dict[str, object]:
    for source in (
        "kernel_step_contract.decode_action_envelope",
        "kernel_continuation_runtime.LiveContinuation",
        "kernel_continuation_journal.ContinuationJournal.records",
        "kernel_continuation_restart.restore_runtime",
    ):
        _reference(source)
    return {
        "tool": "run_tests",
        "proposal": {
            "action": "run_tests",
            "args": {"selector": "probes/test_journal_reserved_fields.py"},
        },
        "steps": [
            {"role": "host dispatch binding", "event_kind": "step_dispatch_binding"},
            {"role": "host action receipt", "event_kind": "action_receipt"},
            {"role": "host closed outcome", "event_kind": "action_outcome_closed"},
        ],
        "walkthrough": [
            "decode_action_envelope validates the visible proposal against the live menu",
            "the host binds and dispatches the action; the model cannot author the binding",
            "the host writes a receipt and closes the action with observed before/after bindings",
            "ContinuationJournal.records replays the canonical chain before restart can proceed",
            "a receipt-to-decision consumer may settle a continuation effect; explanation text has no authority",
        ],
        "source_path": [
            "kernel_step_contract.decode_action_envelope",
            "kernel_continuation_runtime.LiveContinuation",
            "kernel_continuation_journal.ContinuationJournal.records",
            "kernel_continuation_restart.restore_runtime",
        ],
    }


def build_spec() -> dict[str, object]:
    """Build the versioned public spec exclusively from live code contracts."""
    spec = {
        "spec_version": SPEC_VERSION,
        "publication_switch": MODE_ENV,
        "authority_model": "model proposals are advisory; host observations and transitions are authoritative",
        "switches": _switch_rows(),
        "model_proposal": _model_proposal(),
        "host_authority": _host_authority(),
        "typed_events": _event_rows(),
        "action_receipts": _action_receipts(),
        "journal_replay": _journal_replay(),
        "continuation": _continuation(),
        "unsupported_combinations": list(copy.deepcopy(_UNSUPPORTED_COMBINATIONS)),
        "executable_example": _executable_example(),
    }
    return copy.deepcopy(spec)


def public_spec() -> dict[str, object] | None:
    """Return a fresh public projection only when this documentation switch is on."""
    if not enabled():
        return None
    return build_spec()


def publish_public_spec(destination: str | Path) -> Path | None:
    """Write the opt-in public contract for a worker's release evidence."""
    spec = public_spec()
    if spec is None:
        return None
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def render_snapshot() -> str:
    """Render the Python-only artifact checked by the drift test."""
    return (
        '"""GENERATED by kernel_protocol_spec.render_snapshot(); do not edit by hand."""\n'
        "from __future__ import annotations\n\n"
        "SPEC = " + pprint.pformat(build_spec(), sort_dicts=True, width=100) + "\n"
    )


__all__ = [
    "MODE_ENV", "SPEC_VERSION", "build_spec", "enabled", "mode", "public_spec",
    "publish_public_spec", "render_snapshot",
]
