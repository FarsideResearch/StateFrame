"""Pure, fail-closed receipts for explicitly planned public files.

This module does not discover or authenticate host state. A host adapter must
obtain records from the authoritative journal, verify their origin, invoke a
pinned public-response plan parser, preserve its exact file spans, and project
file hashes from rich bindings before constructing these DTOs. Text search
alone does not establish that a path was planned. The renderer checks bounded
canonical digests and joins; digests alone do not authenticate caller claims.
No source files, workspace paths, model state, or private reasoning are read.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from typing import Literal, TypeAlias


ReceiptStatus: TypeAlias = Literal["CHANGED_FROM_PLAN", "UNCHANGED_FROM_PLAN", "UNKNOWN"]
PlanStatus: TypeAlias = Literal["READY", "NOOP"]

SCHEMA = "kernel-scope-receipt/1"
MAX_PLAN_TEXT_BYTES = 65_536
MAX_FACTS_BYTES = 262_144
MAX_FILES = 32
MAX_CLOSED_STEPS = 64
MAX_ACTION_RECEIPTS = 256
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 20_000
MAX_JSON_VALUE_BYTES = 262_144
MAX_ID_BYTES = 512
MAX_PATH_BYTES = 1_024
MAX_SYMLINK_TARGET_BYTES = 4_096
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BINDING_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UNKNOWN_IDENTIFIERS = frozenset({
    "unknown",
    "unknown_scope",
    "unknown_policy",
    "unknown_tenant",
    "unknown_environment",
})


@dataclass(frozen=True)
class AuthorityContext:
    run_id: str
    attempt_id: str
    scope_id: str
    policy_id: str
    tenant_id: str
    environment_id: str


@dataclass(frozen=True)
class PublicPlanFile:
    """A planned path and byte span emitted by the adapter's pinned parser."""

    path: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class FileHash:
    path: str
    sha256: str | None


@dataclass(frozen=True)
class BindingSnapshot:
    """A full binding plus its adapter-projected hashes for named files.

    ``file_hashes`` must be derived by the trusted adapter from ``binding``;
    omitted entries mean UNKNOWN, never deletion. Hash projections are included
    in evidence digests so a changed projection cannot be silently reused.
    """

    binding: object
    file_hashes: tuple[FileHash, ...]


@dataclass(frozen=True)
class PublicResponseOrigin:
    run_id: str
    attempt_id: str
    request_id: str
    candidate_revision: str
    record_id: str
    sequence: int
    source_id: str
    text: str
    source_sha256: str
    plan_parser_ref: str
    plan_parser_revision: str
    plan_parser_sha256: str
    planned_files: tuple[PublicPlanFile, ...]
    authority: AuthorityContext
    projection_sha256: str  # DTO projection digest, not the raw host event digest


@dataclass(frozen=True)
class OwnerContract:
    run_id: str
    attempt_id: str
    request_id: str
    candidate_revision: str
    record_id: str
    sequence: int
    revision: str
    before: BindingSnapshot
    authority: AuthorityContext
    projection_sha256: str  # DTO projection digest, not the raw host event digest


@dataclass(frozen=True)
class PreparedRequest:
    run_id: str
    attempt_id: str
    request_id: str
    owner_contract_request_id: str
    candidate_revision: str
    record_id: str
    sequence: int
    contract_revision: str
    authority: AuthorityContext
    projection_sha256: str


@dataclass(frozen=True)
class ActionReceipt:
    run_id: str
    attempt_id: str
    request_id: str
    candidate_revision: str
    record_id: str
    receipt_id: str
    sequence: int
    authority: AuthorityContext
    projection_sha256: str


@dataclass(frozen=True)
class ClosedActionStep:
    run_id: str
    attempt_id: str
    request_id: str
    candidate_revision: str
    record_id: str
    sequence: int
    chain_index: int
    action_receipt_ids: tuple[str, ...]
    before: BindingSnapshot
    after: BindingSnapshot
    authority: AuthorityContext
    projection_sha256: str


@dataclass(frozen=True)
class ClosedActionHistory:
    """Authenticated adapter export bounded by exact baseline/current rows."""

    candidate_revision: str
    start_record_id: str
    start_sequence: int
    end_record_id: str
    end_sequence: int
    complete: bool
    closed_steps: tuple[ClosedActionStep, ...]
    action_receipts: tuple[ActionReceipt, ...]
    authority: AuthorityContext
    history_projection_sha256: str
    unmatched_action_count: int = 0
    omitted_record_count: int = 0
    partial: bool = False


@dataclass(frozen=True)
class CurrentBinding:
    run_id: str
    attempt_id: str
    candidate_revision: str
    record_id: str
    sequence: int
    snapshot: BindingSnapshot
    authority: AuthorityContext
    projection_sha256: str


@dataclass(frozen=True)
class ScopeReceiptInput:
    current_candidate_revision: str
    authority: AuthorityContext
    public_response: PublicResponseOrigin
    owner_contracts: tuple[OwnerContract, ...]
    prepared_requests: tuple[PreparedRequest, ...]
    current_binding: CurrentBinding
    history: ClosedActionHistory


@dataclass(frozen=True)
class HistoryPoint:
    boundary: Literal["baseline", "before_action", "after_action", "current"]
    sequence: int
    record_id: str
    sha256: str | None
    action_receipt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileObservation:
    path: str
    plan_quote: str
    start_byte: int
    end_byte: int
    status: ReceiptStatus
    baseline_sha256: str | None
    current_sha256: str | None
    history: tuple[HistoryPoint, ...]


@dataclass(frozen=True)
class ReceiptReference:
    kind: str
    record_id: str
    projection_sha256: str
    request_id: str | None
    sequence: int
    receipt_id: str | None = None


@dataclass(frozen=True)
class ScopeReceiptPlan:
    status: PlanStatus
    reason: str
    candidate_revision: str | None
    observations: tuple[FileObservation, ...] = ()
    receipt_references: tuple[ReceiptReference, ...] = ()
    facts_json: str | None = None
    facts_sha256: str | None = None
    utf8_byte_count: int = 0


def canonical_json_bytes(value: object) -> bytes:
    """Return bounded canonical UTF-8 JSON; reject non-JSON and non-finite data."""

    budget = [0, 0]
    _validate_json_value(value, depth=0, budget=budget)
    if budget[1] > MAX_JSON_VALUE_BYTES:
        raise ValueError("json_value_too_large")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid_json_value") from exc
    if len(encoded) > MAX_JSON_VALUE_BYTES:
        raise ValueError("json_value_too_large")
    return encoded


def binding_sha256(binding: object) -> str:
    """Return the source format's prefixed canonical binding revision."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(binding)).hexdigest()


def evidence_record_sha256(record: object) -> str:
    """Return a prefixed DTO projection digest, never raw host-journal authority."""

    payload = _record_payload(record)
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def history_slice_sha256(history: ClosedActionHistory) -> str:
    """Return a prefixed digest of DTO row projections and explicit boundaries."""

    payload = _history_payload(history)
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def render_scope_receipt(
    value: ScopeReceiptInput | None,
    *,
    enabled: bool = False,
    max_facts_bytes: int = 65_536,
) -> ScopeReceiptPlan:
    """Render hash-only observations for paths named in one public response.

    This is an informational source receipt, not proof of causality, semantic
    correctness, successful validation, or whole-task/repository coverage.
    History records closed-action boundary snapshots; it cannot observe an edit
    that is made and fully restored inside one action.
    """

    if type(enabled) is not bool:
        return _noop("invalid_enabled", None)
    if not enabled:
        return _noop("disabled", None)
    candidate = _candidate_from(value)
    if type(max_facts_bytes) is not int or not 1 <= max_facts_bytes <= MAX_FACTS_BYTES:
        return _noop("invalid_facts_budget", candidate)
    if type(value) is not ScopeReceiptInput:
        return _noop("invalid_input", candidate)

    try:
        observations, references, owner, prepared = _validate_and_project(value)
        facts = _facts_payload(value, owner, prepared, observations, references)
        facts_bytes = canonical_json_bytes(facts)
        if len(facts_bytes) > max_facts_bytes:
            return _noop("rendered_facts_too_large", candidate)
        facts_text = facts_bytes.decode("utf-8", "strict")
    except _EvidenceError as exc:
        return _noop(exc.reason, candidate)
    except (AttributeError, TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        return _noop("invalid_evidence", candidate)

    return ScopeReceiptPlan(
        status="READY",
        reason="joined_public_file_hash_receipt",
        candidate_revision=value.current_candidate_revision,
        observations=observations,
        receipt_references=references,
        facts_json=facts_text,
        facts_sha256=hashlib.sha256(facts_bytes).hexdigest(),
        utf8_byte_count=len(facts_bytes),
    )


class _EvidenceError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _validate_and_project(
    value: ScopeReceiptInput,
) -> tuple[
    tuple[FileObservation, ...],
    tuple[ReceiptReference, ...],
    OwnerContract,
    PreparedRequest,
]:
    _identifier(value.current_candidate_revision, "candidate_revision")
    _validate_authority(value.authority)
    response = value.public_response
    if type(response) is not PublicResponseOrigin:
        raise _EvidenceError("missing_public_response_origin")
    _validate_public_response(response, value)
    if (
        type(value.owner_contracts) is not tuple
        or len(value.owner_contracts) > MAX_ACTION_RECEIPTS
        or type(value.prepared_requests) is not tuple
        or len(value.prepared_requests) > MAX_ACTION_RECEIPTS
    ):
        raise _EvidenceError("invalid_request_join_rows")
    if any(type(row) is not OwnerContract for row in value.owner_contracts) or any(
        type(row) is not PreparedRequest for row in value.prepared_requests
    ):
        raise _EvidenceError("invalid_request_join_rows")

    matching_owners = tuple(
        row
        for row in value.owner_contracts
        if type(row) is OwnerContract and row.request_id == response.request_id
    )
    matching_prepared = tuple(
        row
        for row in value.prepared_requests
        if type(row) is PreparedRequest and row.request_id == response.request_id
    )
    if len(matching_owners) != 1:
        raise _EvidenceError("missing_or_conflicting_owner_contract")
    if len(matching_prepared) != 1:
        raise _EvidenceError("missing_or_conflicting_prepared_request")
    owner = matching_owners[0]
    prepared = matching_prepared[0]
    _validate_owner_contract(owner, value)
    _validate_prepared_request(prepared, value)
    baseline_revision = _binding_product_revision(owner.before.binding)
    if (
        response.candidate_revision != baseline_revision
        or owner.candidate_revision != baseline_revision
        or prepared.candidate_revision != baseline_revision
        or prepared.owner_contract_request_id != owner.request_id
        or prepared.contract_revision != owner.revision
        or owner.revision != binding_sha256(owner.before.binding)
    ):
        raise _EvidenceError("request_candidate_or_contract_revision_mismatch")
    if prepared.sequence <= owner.sequence or response.sequence <= prepared.sequence:
        raise _EvidenceError("request_origin_order_mismatch")

    current = value.current_binding
    _validate_current_binding(current, value)
    fixed_record_ids = (
        response.record_id,
        owner.record_id,
        prepared.record_id,
        current.record_id,
    )
    if len(set(fixed_record_ids)) != len(fixed_record_ids):
        raise _EvidenceError("duplicate_evidence_record_id")
    history = value.history
    _validate_history(history, value, owner, current, response)
    history_record_ids = tuple(
        [step.record_id for step in history.closed_steps]
        + [receipt.record_id for receipt in history.action_receipts]
    )
    if len(set((*fixed_record_ids, *history_record_ids))) != len(
        fixed_record_ids + history_record_ids
    ):
        raise _EvidenceError("duplicate_evidence_record_id")

    matching_steps = tuple(
        step for step in history.closed_steps if step.request_id == response.request_id
    )
    if not matching_steps:
        raise _EvidenceError("no_closed_action_for_public_request")
    first_request_step = matching_steps[0]
    if binding_sha256(first_request_step.before.binding) != owner.revision:
        raise _EvidenceError("first_action_before_binding_mismatch")
    if canonical_json_bytes(first_request_step.before.binding) != canonical_json_bytes(
        owner.before.binding
    ):
        raise _EvidenceError("first_action_before_binding_mismatch")

    first_ids = set(first_request_step.action_receipt_ids)
    for receipt in history.action_receipts:
        if (
            receipt.request_id == response.request_id
            and receipt.sequence < first_request_step.sequence
            and receipt.receipt_id not in first_ids
        ):
            raise _EvidenceError("preceding_unmatched_action_receipt")

    observations = tuple(
        _observe_file(
            planned,
            owner.before,
            history.closed_steps,
            current,
            owner,
        )
        for planned in response.planned_files
    )
    references = _receipt_references(response, owner, prepared, current, history)
    return observations, references, owner, prepared


def _validate_public_response(response: PublicResponseOrigin, value: ScopeReceiptInput) -> None:
    _validate_common_record(
        response.run_id,
        response.attempt_id,
        response.candidate_revision,
        response.record_id,
        response.sequence,
        response.authority,
        value,
        "public_response",
    )
    _identifier(response.request_id, "request_id")
    _identifier(response.candidate_revision, "invalid_public_response_candidate_revision")
    _identifier(response.source_id, "public_source_id")
    _safe_relative_path(response.plan_parser_ref)
    _identifier(response.plan_parser_revision, "invalid_plan_parser_revision")
    _valid_sha(response.plan_parser_sha256, "invalid_plan_parser_sha256")
    if type(response.text) is not str:
        raise _EvidenceError("invalid_public_response_text")
    source_bytes = _utf8(response.text, MAX_PLAN_TEXT_BYTES, "invalid_public_response_text")
    _valid_sha(response.source_sha256, "invalid_public_response_sha256")
    if hashlib.sha256(source_bytes).hexdigest() != response.source_sha256:
        raise _EvidenceError("public_response_sha256_mismatch")
    if type(response.planned_files) is not tuple or not 1 <= len(response.planned_files) <= MAX_FILES:
        raise _EvidenceError("invalid_planned_file_count")
    seen_paths: set[str] = set()
    for planned in response.planned_files:
        if type(planned) is not PublicPlanFile:
            raise _EvidenceError("invalid_planned_file")
        path = _safe_relative_path(planned.path)
        if path in seen_paths:
            raise _EvidenceError("duplicate_planned_file")
        seen_paths.add(path)
        start = _strict_int(planned.start_byte, "invalid_plan_span")
        end = _strict_int(planned.end_byte, "invalid_plan_span")
        if start < 0 or end <= start or end > len(source_bytes):
            raise _EvidenceError("invalid_plan_span")
        try:
            quote = source_bytes[start:end].decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise _EvidenceError("plan_span_splits_utf8") from exc
        if quote != path:
            raise _EvidenceError("plan_span_path_mismatch")
    _verify_record_digest(response, "public_response_projection_sha256_mismatch")


def _validate_owner_contract(owner: OwnerContract, value: ScopeReceiptInput) -> None:
    _validate_common_record(
        owner.run_id,
        owner.attempt_id,
        owner.candidate_revision,
        owner.record_id,
        owner.sequence,
        owner.authority,
        value,
        "owner_contract",
    )
    _identifier(owner.request_id, "owner_contract_request_id")
    _validate_snapshot(owner.before, value.authority)
    _valid_projection_sha(owner.revision, "invalid_owner_contract_revision")
    if owner.revision != binding_sha256(owner.before.binding):
        raise _EvidenceError("owner_contract_revision_mismatch")
    if owner.candidate_revision != _binding_product_revision(owner.before.binding):
        raise _EvidenceError("owner_contract_candidate_revision_mismatch")
    _verify_record_digest(owner, "owner_contract_projection_sha256_mismatch")


def _validate_prepared_request(prepared: PreparedRequest, value: ScopeReceiptInput) -> None:
    _validate_common_record(
        prepared.run_id,
        prepared.attempt_id,
        prepared.candidate_revision,
        prepared.record_id,
        prepared.sequence,
        prepared.authority,
        value,
        "prepared_request",
    )
    _identifier(prepared.request_id, "prepared_request_id")
    _identifier(prepared.candidate_revision, "invalid_prepared_candidate_revision")
    _identifier(prepared.owner_contract_request_id, "owner_contract_request_id")
    _valid_projection_sha(prepared.contract_revision, "invalid_prepared_contract_revision")
    _verify_record_digest(prepared, "prepared_request_projection_sha256_mismatch")


def _validate_current_binding(current: CurrentBinding, value: ScopeReceiptInput) -> None:
    if type(current) is not CurrentBinding:
        raise _EvidenceError("missing_current_binding")
    _validate_common_record(
        current.run_id,
        current.attempt_id,
        current.candidate_revision,
        current.record_id,
        current.sequence,
        current.authority,
        value,
        "current_binding",
    )
    _validate_snapshot(current.snapshot, value.authority)
    current_product_revision = _binding_product_revision(current.snapshot.binding)
    if (
        current.candidate_revision != current_product_revision
        or value.current_candidate_revision != current_product_revision
    ):
        raise _EvidenceError("current_candidate_product_revision_mismatch")
    _verify_record_digest(current, "current_binding_projection_sha256_mismatch")


def _validate_history(
    history: ClosedActionHistory,
    value: ScopeReceiptInput,
    owner: OwnerContract,
    current: CurrentBinding,
    response: PublicResponseOrigin,
) -> None:
    if type(history) is not ClosedActionHistory or type(history.complete) is not bool:
        raise _EvidenceError("incomplete_action_history")
    if (
        not history.complete
        or type(history.partial) is not bool
        or history.partial
        or _strict_int(history.unmatched_action_count, "invalid_history_unmatched_count") != 0
        or _strict_int(history.omitted_record_count, "invalid_history_omitted_count") != 0
    ):
        raise _EvidenceError("incomplete_action_history")
    if history.candidate_revision != value.current_candidate_revision:
        raise _EvidenceError("history_candidate_revision_mismatch")
    if history.authority != value.authority:
        raise _EvidenceError("history_authority_mismatch")
    _identifier(history.start_record_id, "invalid_history_start_record")
    _identifier(history.end_record_id, "invalid_history_end_record")
    start = _strict_int(history.start_sequence, "invalid_history_boundary")
    end = _strict_int(history.end_sequence, "invalid_history_boundary")
    if (
        start < 0
        or history.start_record_id != owner.record_id
        or start != owner.sequence
        or history.end_record_id != current.record_id
        or end != current.sequence
        or end <= start
    ):
        raise _EvidenceError("history_boundary_mismatch")
    if type(history.closed_steps) is not tuple or not 1 <= len(history.closed_steps) <= MAX_CLOSED_STEPS:
        raise _EvidenceError("invalid_closed_step_count")
    if type(history.action_receipts) is not tuple or len(history.action_receipts) > MAX_ACTION_RECEIPTS:
        raise _EvidenceError("invalid_action_receipt_count")

    for step in history.closed_steps:
        if type(step) is not ClosedActionStep:
            raise _EvidenceError("invalid_closed_action_step")
        if _strict_int(step.sequence, "invalid_closed_action_step_sequence") <= response.sequence:
            raise _EvidenceError("action_before_public_response")
    for receipt in history.action_receipts:
        if type(receipt) is not ActionReceipt:
            raise _EvidenceError("invalid_action_receipt")
        if _strict_int(receipt.sequence, "invalid_action_receipt_sequence") <= response.sequence:
            raise _EvidenceError("action_before_public_response")

    total_receipt_refs = 0
    for step in history.closed_steps:
        if type(step) is not ClosedActionStep:
            raise _EvidenceError("invalid_closed_action_step")
        if type(step.action_receipt_ids) is not tuple or not step.action_receipt_ids:
            raise _EvidenceError("closed_action_receipt_ids_missing")
        if len(step.action_receipt_ids) > MAX_ACTION_RECEIPTS:
            raise _EvidenceError("closed_action_receipt_ids_too_large")
        total_receipt_refs += len(step.action_receipt_ids)
        if total_receipt_refs > MAX_ACTION_RECEIPTS:
            raise _EvidenceError("history_action_receipt_ids_too_large")

    previous_sequence = start
    previous_binding = owner.before.binding
    previous_index = -1
    referenced_ids: list[str] = []
    closed_ids: set[str] = set()
    for step in history.closed_steps:
        if type(step) is not ClosedActionStep:
            raise _EvidenceError("invalid_closed_action_step")
        _validate_common_record(
            step.run_id,
            step.attempt_id,
            step.candidate_revision,
            step.record_id,
            step.sequence,
            step.authority,
            value,
            "closed_action_step",
        )
        chain_index = _strict_int(step.chain_index, "invalid_history_chain_index")
        if chain_index < 0 or step.sequence <= previous_sequence or step.sequence >= end:
            raise _EvidenceError("history_sequence_order_mismatch")
        if chain_index != previous_index + 1:
            raise _EvidenceError("history_chain_index_gap")
        if step.record_id in closed_ids:
            raise _EvidenceError("duplicate_closed_action_record")
        closed_ids.add(step.record_id)
        _identifier(step.request_id, "closed_action_request_id")
        _validate_snapshot(step.before, value.authority)
        _validate_snapshot(step.after, value.authority)
        if step.candidate_revision != _binding_product_revision(step.before.binding):
            raise _EvidenceError("closed_action_before_product_revision_mismatch")
        if canonical_json_bytes(step.before.binding) != canonical_json_bytes(previous_binding):
            raise _EvidenceError("closed_action_binding_chain_gap")
        for receipt_id in step.action_receipt_ids:
            _uuid_identifier(receipt_id, "invalid_action_receipt_id")
            referenced_ids.append(receipt_id)
        _verify_record_digest(step, "closed_action_projection_sha256_mismatch")
        previous_sequence = step.sequence
        previous_index = chain_index
        previous_binding = step.after.binding

    if canonical_json_bytes(previous_binding) != canonical_json_bytes(current.snapshot.binding):
        raise _EvidenceError("current_binding_history_mismatch")
    if len(set(referenced_ids)) != len(referenced_ids):
        raise _EvidenceError("duplicate_action_receipt_binding")

    receipts_by_id: dict[str, ActionReceipt] = {}
    last_receipt_sequence = start
    for receipt in history.action_receipts:
        if type(receipt) is not ActionReceipt:
            raise _EvidenceError("invalid_action_receipt")
        _validate_common_record(
            receipt.run_id,
            receipt.attempt_id,
            receipt.candidate_revision,
            receipt.record_id,
            receipt.sequence,
            receipt.authority,
            value,
            "action_receipt",
        )
        _identifier(receipt.request_id, "action_receipt_request_id")
        _uuid_identifier(receipt.receipt_id, "invalid_action_receipt_id")
        if receipt.sequence <= last_receipt_sequence or receipt.sequence >= end:
            raise _EvidenceError("action_receipt_sequence_order_mismatch")
        last_receipt_sequence = receipt.sequence
        if receipt.receipt_id in receipts_by_id:
            raise _EvidenceError("duplicate_action_receipt_id")
        receipts_by_id[receipt.receipt_id] = receipt
        _verify_record_digest(receipt, "action_receipt_projection_sha256_mismatch")

    if set(referenced_ids) != set(receipts_by_id):
        raise _EvidenceError("action_receipt_history_join_mismatch")
    all_event_sequences = [
        *(step.sequence for step in history.closed_steps),
        *(receipt.sequence for receipt in history.action_receipts),
    ]
    if len(set(all_event_sequences)) != len(all_event_sequences):
        raise _EvidenceError("duplicate_history_event_sequence")
    previous_closed_sequence = start
    for step in history.closed_steps:
        after_revision = _binding_product_revision(step.after.binding)
        for receipt_id in step.action_receipt_ids:
            receipt = receipts_by_id[receipt_id]
            if receipt.sequence <= previous_closed_sequence:
                raise _EvidenceError("action_receipt_before_previous_closed_step")
            if (
                receipt.request_id != step.request_id
                or receipt.sequence >= step.sequence
                or receipt.candidate_revision != after_revision
            ):
                raise _EvidenceError("action_receipt_closed_revision_or_record_mismatch")
        previous_closed_sequence = step.sequence
    _verify_history_digest(history)


def _validate_common_record(
    run_id: str,
    attempt_id: str,
    candidate_revision: str,
    record_id: str,
    sequence: int,
    authority: AuthorityContext,
    value: ScopeReceiptInput,
    label: str,
) -> None:
    _identifier(run_id, f"invalid_{label}_run")
    _identifier(attempt_id, f"invalid_{label}_attempt")
    _identifier(candidate_revision, f"invalid_{label}_candidate_revision")
    _identifier(record_id, f"invalid_{label}_record_id")
    if _strict_int(sequence, f"invalid_{label}_sequence") < 0:
        raise _EvidenceError(f"invalid_{label}_sequence")
    _validate_authority(authority)
    if (
        run_id != authority.run_id
        or attempt_id != authority.attempt_id
        or run_id != value.public_response.run_id
        or attempt_id != value.public_response.attempt_id
        or authority != value.authority
    ):
        raise _EvidenceError(f"{label}_authority_or_candidate_mismatch")


def _validate_snapshot(snapshot: BindingSnapshot, authority: AuthorityContext) -> None:
    if type(snapshot) is not BindingSnapshot:
        raise _EvidenceError("invalid_binding_snapshot")
    if type(snapshot.binding) is not dict:
        raise _EvidenceError("invalid_binding_object")
    try:
        canonical_json_bytes(snapshot.binding)
    except ValueError as exc:
        raise _EvidenceError("invalid_binding_object") from exc
    _validate_binding_authority(snapshot.binding, authority)
    _binding_product_revision(snapshot.binding)
    product_files = snapshot.binding.get("product_files")
    if type(product_files) is not dict:
        raise _EvidenceError("missing_binding_product_files")
    for path, revision in product_files.items():
        _safe_relative_path(path)
        if _is_symlink_revision(revision):
            target = revision[len("symlink:") :]
            if not target.strip():
                raise _EvidenceError("invalid_binding_symlink_target")
            _utf8(target, MAX_SYMLINK_TARGET_BYTES, "invalid_binding_symlink_target")
        else:
            _valid_projection_sha(revision, "invalid_binding_product_file_revision")
    if type(snapshot.file_hashes) is not tuple or len(snapshot.file_hashes) > MAX_FILES:
        raise _EvidenceError("invalid_binding_file_hashes")
    seen: set[str] = set()
    for item in snapshot.file_hashes:
        if type(item) is not FileHash:
            raise _EvidenceError("invalid_binding_file_hash")
        path = _safe_relative_path(item.path)
        if path in seen:
            raise _EvidenceError("duplicate_binding_file_hash")
        seen.add(path)
        source_revision = product_files.get(path)
        if item.sha256 is None:
            if source_revision is not None and not _is_symlink_revision(source_revision):
                raise _EvidenceError("known_product_file_marked_unknown")
        else:
            _valid_sha(item.sha256, "invalid_file_sha256")
            if source_revision != "sha256:" + item.sha256:
                raise _EvidenceError("product_file_projection_mismatch")


def _binding_product_revision(binding: object) -> str:
    if type(binding) is not dict:
        raise _EvidenceError("invalid_binding_object")
    revision = _identifier(binding.get("product_revision"), "invalid_product_revision")
    product_files = binding.get("product_files")
    if type(product_files) is not dict:
        raise _EvidenceError("missing_binding_product_files")
    if revision != binding_sha256(product_files):
        raise _EvidenceError("product_revision_binding_mismatch")
    return revision


def _is_symlink_revision(value: object) -> bool:
    return type(value) is str and value.startswith("symlink:")


def _validate_binding_authority(
    binding: dict[str, object], authority: AuthorityContext
) -> None:
    expected_fields = (
        ("policy_identity", authority.policy_id, "binding_policy_mismatch"),
        ("tenant_identity", authority.tenant_id, "binding_tenant_mismatch"),
        ("environment_revision", authority.environment_id, "binding_environment_mismatch"),
        ("authorization_scope", authority.scope_id, "binding_scope_mismatch"),
    )
    for field, expected, reason in expected_fields:
        actual = _identifier(binding.get(field), reason)
        if actual != expected:
            raise _EvidenceError(reason)


def _observe_file(
    planned: PublicPlanFile,
    baseline: BindingSnapshot,
    steps: tuple[ClosedActionStep, ...],
    current: CurrentBinding,
    owner: OwnerContract,
) -> FileObservation:
    points: list[HistoryPoint] = [
        HistoryPoint(
            boundary="baseline",
            sequence=owner.sequence,
            record_id=owner.record_id,
            sha256=_lookup_hash(baseline, planned.path),
        )
    ]
    for step in steps:
        points.extend(
            (
                HistoryPoint(
                    boundary="before_action",
                    sequence=step.sequence,
                    record_id=step.record_id,
                    sha256=_lookup_hash(step.before, planned.path),
                    action_receipt_ids=step.action_receipt_ids,
                ),
                HistoryPoint(
                    boundary="after_action",
                    sequence=step.sequence,
                    record_id=step.record_id,
                    sha256=_lookup_hash(step.after, planned.path),
                    action_receipt_ids=step.action_receipt_ids,
                ),
            )
        )
    points.append(
        HistoryPoint(
            boundary="current",
            sequence=current.sequence,
            record_id=current.record_id,
            sha256=_lookup_hash(current.snapshot, planned.path),
        )
    )
    baseline_sha = points[0].sha256
    current_sha = points[-1].sha256
    if any(point.sha256 is None for point in points):
        status: ReceiptStatus = "UNKNOWN"
    elif baseline_sha == current_sha:
        status = "UNCHANGED_FROM_PLAN"
    else:
        status = "CHANGED_FROM_PLAN"
    return FileObservation(
        path=planned.path,
        plan_quote=planned.path,
        start_byte=planned.start_byte,
        end_byte=planned.end_byte,
        status=status,
        baseline_sha256=baseline_sha,
        current_sha256=current_sha,
        history=tuple(points),
    )


def _lookup_hash(snapshot: BindingSnapshot, path: str) -> str | None:
    for item in snapshot.file_hashes:
        if item.path == path:
            return item.sha256
    return None


def _receipt_references(
    response: PublicResponseOrigin,
    owner: OwnerContract,
    prepared: PreparedRequest,
    current: CurrentBinding,
    history: ClosedActionHistory,
) -> tuple[ReceiptReference, ...]:
    refs = [
        ReceiptReference(
            "public_response",
            response.record_id,
            response.projection_sha256,
            response.request_id,
            response.sequence,
        ),
        ReceiptReference(
            "owner_contract",
            owner.record_id,
            owner.projection_sha256,
            owner.request_id,
            owner.sequence,
        ),
        ReceiptReference(
            "request_prepared",
            prepared.record_id,
            prepared.projection_sha256,
            prepared.request_id,
            prepared.sequence,
        ),
    ]
    for step in history.closed_steps:
        refs.append(
            ReceiptReference(
                "action_outcome_closed",
                step.record_id,
                step.projection_sha256,
                step.request_id,
                step.sequence,
            )
        )
    for receipt in history.action_receipts:
        refs.append(
            ReceiptReference(
                "action_receipt",
                receipt.record_id,
                receipt.projection_sha256,
                receipt.request_id,
                receipt.sequence,
                receipt.receipt_id,
            )
        )
    refs.append(
        ReceiptReference(
            "current_binding",
            current.record_id,
            current.projection_sha256,
            None,
            current.sequence,
        )
    )
    return tuple(refs)


def _facts_payload(
    value: ScopeReceiptInput,
    owner: OwnerContract,
    prepared: PreparedRequest,
    observations: tuple[FileObservation, ...],
    references: tuple[ReceiptReference, ...],
) -> dict[str, object]:
    response = value.public_response
    history = value.history
    return {
        "schema": SCHEMA,
        "candidate_revision": value.current_candidate_revision,
        "public_plan": {
            "source_id": response.source_id,
            "source_sha256": response.source_sha256,
            "plan_parser_ref": response.plan_parser_ref,
            "plan_parser_revision": response.plan_parser_revision,
            "plan_parser_sha256": response.plan_parser_sha256,
            "request_id": response.request_id,
            "response_record_id": response.record_id,
            "response_projection_sha256": response.projection_sha256,
        },
        "owner_contract": {
            "record_id": owner.record_id,
            "projection_sha256": owner.projection_sha256,
            "request_id": owner.request_id,
            "revision": owner.revision,
        },
        "request_prepared": {
            "record_id": prepared.record_id,
            "projection_sha256": prepared.projection_sha256,
            "request_id": prepared.request_id,
            "contract_revision": prepared.contract_revision,
        },
        "history": {
            "start_record_id": history.start_record_id,
            "start_sequence": history.start_sequence,
            "end_record_id": history.end_record_id,
            "end_sequence": history.end_sequence,
            "complete": history.complete,
            "unmatched_action_count": history.unmatched_action_count,
            "omitted_record_count": history.omitted_record_count,
            "partial": history.partial,
            "history_projection_sha256": history.history_projection_sha256,
        },
        "files": [
            {
                "path": item.path,
                "public_plan_quote": item.plan_quote,
                "source_byte_span": [item.start_byte, item.end_byte],
                "status": item.status,
                "baseline_sha256": item.baseline_sha256,
                "current_sha256": item.current_sha256,
                "history": [
                    {
                        "boundary": point.boundary,
                        "sequence": point.sequence,
                        "record_id": point.record_id,
                        "sha256": point.sha256,
                        "action_receipt_ids": list(point.action_receipt_ids),
                    }
                    for point in item.history
                ],
            }
            for item in observations
        ],
        "receipt_references": [
            {
                "kind": ref.kind,
                "record_id": ref.record_id,
                "projection_sha256": ref.projection_sha256,
                "request_id": ref.request_id,
                "sequence": ref.sequence,
                "receipt_id": ref.receipt_id,
            }
            for ref in references
        ],
    }


def _record_payload(record: object) -> dict[str, object]:
    if type(record) is PublicResponseOrigin:
        payload = _common_record_payload("public_response_origin", record)
        payload.update({
            "request_id": record.request_id,
            "source_id": record.source_id,
            "text": record.text,
            "source_sha256": record.source_sha256,
            "plan_parser_ref": record.plan_parser_ref,
            "plan_parser_revision": record.plan_parser_revision,
            "plan_parser_sha256": record.plan_parser_sha256,
            "planned_files": [
                {"path": item.path, "start_byte": item.start_byte, "end_byte": item.end_byte}
                for item in record.planned_files
            ],
        })
        return payload
    if type(record) is OwnerContract:
        payload = _common_record_payload("owner_contract", record)
        payload.update({
            "request_id": record.request_id,
            "revision": record.revision,
            "before": _snapshot_payload(record.before),
        })
        return payload
    if type(record) is PreparedRequest:
        payload = _common_record_payload("request_prepared", record)
        payload.update({
            "request_id": record.request_id,
            "owner_contract_request_id": record.owner_contract_request_id,
            "contract_revision": record.contract_revision,
        })
        return payload
    if type(record) is ActionReceipt:
        payload = _common_record_payload("action_receipt", record)
        payload.update({
            "request_id": record.request_id,
            "receipt_id": record.receipt_id,
        })
        return payload
    if type(record) is ClosedActionStep:
        payload = _common_record_payload("action_outcome_closed", record)
        payload.update({
            "request_id": record.request_id,
            "chain_index": record.chain_index,
            "action_receipt_ids": list(record.action_receipt_ids),
            "before": _snapshot_payload(record.before),
            "after": _snapshot_payload(record.after),
        })
        return payload
    if type(record) is CurrentBinding:
        payload = _common_record_payload("current_binding", record)
        payload["snapshot"] = _snapshot_payload(record.snapshot)
        return payload
    raise ValueError("unsupported_evidence_record")


def _common_record_payload(
    kind: str,
    record: (
        PublicResponseOrigin
        | OwnerContract
        | PreparedRequest
        | ActionReceipt
        | ClosedActionStep
        | CurrentBinding
    ),
) -> dict[str, object]:
    return {
        "kind": kind,
        "run_id": record.run_id,
        "attempt_id": record.attempt_id,
        "candidate_revision": record.candidate_revision,
        "record_id": record.record_id,
        "sequence": record.sequence,
        "authority": _authority_payload(record.authority),
    }


def _history_payload(history: ClosedActionHistory) -> dict[str, object]:
    return {
        "kind": "closed_action_history",
        "candidate_revision": history.candidate_revision,
        "start_record_id": history.start_record_id,
        "start_sequence": history.start_sequence,
        "end_record_id": history.end_record_id,
        "end_sequence": history.end_sequence,
        "complete": history.complete,
        "unmatched_action_count": history.unmatched_action_count,
        "omitted_record_count": history.omitted_record_count,
        "partial": history.partial,
        "closed_step_projection_digests": [
            step.projection_sha256 for step in history.closed_steps
        ],
        "action_receipt_projection_digests": [
            receipt.projection_sha256 for receipt in history.action_receipts
        ],
        "authority": _authority_payload(history.authority),
    }


def _snapshot_payload(snapshot: BindingSnapshot) -> dict[str, object]:
    return {
        "binding": snapshot.binding,
        "file_hashes": [
            {"path": item.path, "sha256": item.sha256} for item in snapshot.file_hashes
        ],
    }


def _authority_payload(authority: AuthorityContext) -> dict[str, object]:
    return {
        "run_id": authority.run_id,
        "attempt_id": authority.attempt_id,
        "scope_id": authority.scope_id,
        "policy_id": authority.policy_id,
        "tenant_id": authority.tenant_id,
        "environment_id": authority.environment_id,
    }


def _verify_record_digest(record: object, reason: str) -> None:
    digest = getattr(record, "projection_sha256", None)
    _valid_projection_sha(digest, reason)
    if evidence_record_sha256(record) != digest:
        raise _EvidenceError(reason)


def _verify_history_digest(history: ClosedActionHistory) -> None:
    _valid_projection_sha(history.history_projection_sha256, "invalid_history_projection_sha256")
    if history_slice_sha256(history) != history.history_projection_sha256:
        raise _EvidenceError("history_projection_sha256_mismatch")


def _validate_authority(authority: AuthorityContext) -> None:
    if type(authority) is not AuthorityContext:
        raise _EvidenceError("invalid_authority_context")
    for field in (
        authority.run_id,
        authority.attempt_id,
        authority.scope_id,
        authority.policy_id,
        authority.tenant_id,
        authority.environment_id,
    ):
        _identifier(field, "invalid_authority_context")


def _valid_sha(value: object, reason: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _EvidenceError(reason)
    return value


def _valid_projection_sha(value: object, reason: str) -> str:
    if type(value) is not str or _BINDING_SHA256_RE.fullmatch(value) is None:
        raise _EvidenceError(reason)
    return value


def _identifier(value: object, reason: str) -> str:
    if type(value) is not str:
        raise _EvidenceError(reason)
    encoded = _utf8(value, MAX_ID_BYTES, reason)
    if not encoded or not value.strip() or value.strip().casefold() in _UNKNOWN_IDENTIFIERS:
        raise _EvidenceError(reason)
    return value


def _uuid_identifier(value: object, reason: str) -> str:
    text = _identifier(value, reason)
    # New provider-visible action receipts are 80-bit deterministic IDs. Old
    # journal exports still contain UUIDs and remain valid for crash replay.
    if re.fullmatch(r"[0-9a-f]{20}", text):
        return text
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as exc:
        raise _EvidenceError(reason) from exc
    if text not in {parsed.hex, str(parsed)}:
        raise _EvidenceError(reason)
    return text


def _strict_int(value: object, reason: str) -> int:
    if type(value) is not int or value.bit_length() > 63:
        raise _EvidenceError(reason)
    return value


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or len(value) > MAX_PATH_BYTES:
        raise _EvidenceError("invalid_relative_path")
    encoded = _utf8(value, MAX_PATH_BYTES, "invalid_relative_path")
    if (
        not encoded
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
        or ":" in value.split("/", 1)[0]
    ):
        raise _EvidenceError("invalid_relative_path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise _EvidenceError("invalid_relative_path")
    return value


def _utf8(value: str, max_bytes: int, reason: str) -> bytes:
    if len(value) > max_bytes:
        raise _EvidenceError(reason)
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise _EvidenceError(reason) from exc
    if len(encoded) > max_bytes:
        raise _EvidenceError(reason)
    return encoded


def _validate_json_value(value: object, *, depth: int, budget: list[int]) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError("json_depth_exceeded")
    budget[0] += 1
    if budget[0] > MAX_JSON_NODES:
        raise ValueError("json_node_limit_exceeded")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if value.bit_length() > 4096:
            raise ValueError("json_integer_too_large")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non_finite_json_number")
        return
    if type(value) is str:
        if len(value) > MAX_JSON_VALUE_BYTES:
            raise ValueError("json_string_too_large")
        encoded = value.encode("utf-8", "strict")
        budget[1] += len(encoded)
        if len(encoded) > MAX_JSON_VALUE_BYTES or budget[1] > MAX_JSON_VALUE_BYTES:
            raise ValueError("json_string_too_large")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item, depth=depth + 1, budget=budget)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("json_object_key_not_string")
            if len(key) > MAX_JSON_VALUE_BYTES:
                raise ValueError("json_string_too_large")
            key_bytes = key.encode("utf-8", "strict")
            budget[1] += len(key_bytes)
            if budget[1] > MAX_JSON_VALUE_BYTES:
                raise ValueError("json_string_too_large")
            _validate_json_value(item, depth=depth + 1, budget=budget)
        return
    raise ValueError("unsupported_json_value")


def _candidate_from(value: object) -> str | None:
    if type(value) is ScopeReceiptInput and type(value.current_candidate_revision) is str:
        revision = value.current_candidate_revision
        if revision and revision.strip() and revision.strip().casefold() not in _UNKNOWN_IDENTIFIERS:
            try:
                _utf8(revision, MAX_ID_BYTES, "invalid_candidate_revision")
            except _EvidenceError:
                return None
            return revision
    return None


def _noop(reason: str, candidate_revision: str | None) -> ScopeReceiptPlan:
    return ScopeReceiptPlan(
        status="NOOP",
        reason=reason,
        candidate_revision=candidate_revision,
    )
