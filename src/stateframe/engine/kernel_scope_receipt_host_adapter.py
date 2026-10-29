"""Bounded adapter for host-shaped scope receipts; caller origin stays unverified.

The durable host exporter and trusted task-attempt mapping are not implemented in
this generation. This module accepts an explicit journal-shaped slice to create
candidate facts only. Caller booleans and projection hashes never authenticate
the journal or transport. The only filesystem read is the bounded, explicit pin
check for the maintained public plan parser.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import kernel_scope_receipt as scope_receipt
import kernel_step_contract as step_contract

EXPORT_SCHEMA = "stateframe-scope-receipt-export/1"
JOURNAL_SCHEMA = "stateframe-audit/1"
PARSER_REF = "kernel_step_contract.py"
PARSER_REVISION = "kernel_step_contract@fc46ed7021a489cd0ce779a7bd047ea0ac27a9af3006d2f9e9238928731d9ce4"
PARSER_SHA256 = "fc46ed7021a489cd0ce779a7bd047ea0ac27a9af3006d2f9e9238928731d9ce4"
MAX_EVENTS = 4096
MAX_EVENT_BYTES = 262_144
MAX_EXPORT_BYTES = 4 * 1024 * 1024
MAX_WIRE_BYTES = 4 * 1024 * 1024
_PATH_TOKEN = re.compile(r"[A-Za-z0-9_./\\-]")

AdapterStatus = Literal["UNVERIFIED", "NOOP"]


@dataclass(frozen=True)
class ScopeHostExport:
    """Explicit event slice; completeness fields are untrusted shape claims."""

    schema: str
    records: tuple[object, ...]
    start_sequence: int
    end_sequence: int
    complete: bool
    omitted_record_count: int
    pending_action_count: int
    request_id: str
    current_boundary_event_id: str


@dataclass(frozen=True)
class ScopeHostAdapterPlan:
    """Candidate source facts. UNVERIFIED is the strongest available status."""

    status: AdapterStatus
    reason: str
    run_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    request_id: str | None = None
    serving_request_id: str | None = None
    task_attempt_id: str | None = None
    scope_id: str | None = None
    policy_id: str | None = None
    tenant_id: str | None = None
    environment_id: str | None = None
    current_boundary_event_id: str | None = None
    boundary_identity: str | None = None
    candidate_revision: str | None = None
    candidate_facts_json: str | None = None
    candidate_facts_sha256: str | None = None
    utf8_byte_count: int = 0
    producer_reason: str | None = None


@dataclass(frozen=True)
class WireMessageLocation:
    """Expected exact role, index, and complete message content."""

    index: int
    role: str
    content: str


@dataclass(frozen=True)
class ScopeServingRequestContext:
    """Caller-supplied source-shaped binding for the later serving request.

    The adapter validates joins but cannot authenticate this export. The
    boundary's serving_request.request_id must match the prepared event's
    top-level and nested request IDs.
    """

    boundary_event: object
    provider_contract_event: object
    request_prepared_event: object


@dataclass(frozen=True)
class ScopeWireCheck:
    """Mechanical raw-body check; it is not proof of transport delivery."""

    status: AdapterStatus
    reason: str
    request_id: str | None = None
    boundary_identity: str | None = None
    raw_http_sha256: str | None = None
    message_sha256: str | None = None
    placement_matches: bool | None = None
    utf8_byte_count: int = 0


class _AdapterError(ValueError):
    pass


def project_scope_receipt_export(
    export: ScopeHostExport | None,
    *,
    enabled: bool = False,
    max_facts_bytes: int = 65_536,
) -> ScopeHostAdapterPlan:
    """Translate bounded journal-shaped events into UNVERIFIED candidate facts."""
    if type(enabled) is not bool:
        return _noop("invalid_enabled")
    if not enabled:
        return _noop("disabled")
    if (type(max_facts_bytes) is not int
            or not 1 <= max_facts_bytes <= scope_receipt.MAX_FACTS_BYTES):
        return _noop("invalid_facts_budget")

    try:
        _require(type(export) is ScopeHostExport, "invalid_export")
        if not _parser_pin_matches():
            raise _AdapterError("maintained_parser_pin_unavailable_or_changed")
        rows, run_id, task_id = _validate_export_slice(export)
        by_kind: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            kind = row["kind"]
            if type(kind) is str:
                by_kind.setdefault(kind, []).append(row)

        boundary = _one(by_kind, "scope_receipt_binding_boundary")
        boundary_id = _event_id(boundary)
        _require(boundary_id == export.current_boundary_event_id,
                 "current_boundary_id_mismatch")
        _require(_strict_int(boundary["sequence"]) == export.end_sequence,
                 "current_boundary_not_slice_end")
        task_attempt = _mapping(boundary.get("task_attempt"),
                                "missing_task_attempt_mapping")
        attempt_id = _identifier(task_attempt.get("attempt_id"),
                                 "invalid_task_attempt")
        task_attempt_id = _identifier(task_attempt.get("task_attempt_id"),
                                      "invalid_task_attempt")
        _require(task_attempt.get("run_id") == run_id
                 and task_attempt.get("task_id") == task_id,
                 "task_attempt_run_join_mismatch")
        authority = scope_receipt.AuthorityContext(
            run_id=run_id,
            attempt_id=attempt_id,
            scope_id=_identifier(task_attempt.get("scope_id"),
                                 "invalid_authority_scope"),
            policy_id=_identifier(task_attempt.get("policy_id"),
                                  "invalid_authority_policy"),
            tenant_id=_identifier(task_attempt.get("tenant_id"),
                                  "invalid_authority_tenant"),
            environment_id=_identifier(task_attempt.get("environment_id"),
                                       "invalid_authority_environment"),
        )
        binding = _mapping(boundary.get("binding"), "missing_current_binding")
        current_revision = _product_revision(binding)
        request_id = _identifier(export.request_id, "invalid_request_id")
        serving_map = _mapping(boundary.get("serving_request"),
                               "missing_serving_request_mapping")
        serving_request_id = _identifier(serving_map.get("request_id"),
                                         "invalid_serving_request_id")
        _require(serving_request_id != request_id,
                 "historical_and_serving_request_must_be_distinct")

        owners = [row for row in by_kind.get("provider_boundary_contract_decided", [])
                  if row.get("request_id") == request_id]
        prepared_events = by_kind.get("request_prepared", [])
        prepared_row = _matching_prepared_event(prepared_events, request_id)
        responses = [row for row in by_kind.get("response_received", [])
                     if row.get("request_id") == request_id]
        _require(len(owners) == len(responses) == 1,
                 "missing_or_conflicting_request_chain")
        owner_row, response_row = owners[0], responses[0]
        owner_sequence = _strict_int(owner_row["sequence"])
        prepared_sequence = _strict_int(prepared_row["sequence"])
        response_sequence = _strict_int(response_row["sequence"])
        _require(owner_sequence < prepared_sequence < response_sequence
                 < export.end_sequence, "request_chain_order_mismatch")
        owner_revision = _identifier(owner_row.get("revision"),
                                     "invalid_owner_revision")
        decision_digest = _identifier(owner_row.get("decision_digest"),
                                      "invalid_owner_decision")
        prepared_contract = _prepared_contract(prepared_row)
        _require(
            prepared_row.get("request_id") == request_id
            and
            prepared_contract.get("request_id") == request_id
            and prepared_contract.get("revision") == owner_revision
            and prepared_contract.get("decision_digest") == decision_digest,
            "prepared_owner_contract_mismatch",
        )
        public_text = _public_response_text(response_row.get("response"))
        planned_files = _planned_files(public_text)
        source_sha = hashlib.sha256(public_text.encode("utf-8", "strict")).hexdigest()
        submissions = [
            row for row in by_kind.get("request_submitted", [])
            if row.get("request_id") == request_id
        ]
        _require(len(submissions) == 1, "unsupported_or_ambiguous_transport_retry")
        submitted_row = submissions[0]
        submitted_payload = _mapping(
            submitted_row.get("payload"), "missing_submitted_payload"
        )
        payload_utf8 = _bounded_text(
            submitted_row.get("payload_utf8"), MAX_EVENT_BYTES
        )
        payload_bytes = payload_utf8.encode("utf-8", "strict")
        decoded_payload = _mapping(
            _strict_json_loads(payload_bytes), "invalid_submitted_payload_json"
        )
        _require(
            scope_receipt.canonical_json_bytes(decoded_payload)
            == scope_receipt.canonical_json_bytes(submitted_payload)
            and submitted_row.get("payload_hash")
            == "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
            and type(submitted_row.get("transport_attempt")) is int
            and submitted_row.get("transport_attempt") == 1
            and type(submitted_row.get("retry_index")) is int
            and submitted_row.get("retry_index") == 0,
            "submitted_payload_or_attempt_metadata_mismatch",
        )
        submitted_sequence = _strict_int(submitted_row["sequence"])
        _require(
            _event_id(submitted_row) == response_row.get("transport_attempt_id")
            and prepared_sequence < submitted_sequence < response_sequence,
            "response_transport_attempt_join_mismatch",
        )
        _require(
            "payload_ref" not in submitted_row and "payload_delta" not in submitted_row,
            "unsupported_transport_retry_payload_reference",
        )
        receipt_events = [
            row for row in by_kind.get("action_receipt", [])
            if owner_sequence < _strict_int(row["sequence"]) < export.end_sequence
        ]
        step_events = [
            row for row in by_kind.get("action_outcome_closed", [])
            if owner_sequence < _strict_int(row["sequence"]) < export.end_sequence
        ]
        raw_receipts = _receipt_rows(receipt_events, run_id)
        raw_steps = _step_rows(step_events)
        steps, receipts, baseline_binding = _make_history_rows(
            raw_steps, raw_receipts, authority, planned_files
        )
        _require(owner_revision == scope_receipt.binding_sha256(baseline_binding),
                 "first_action_before_owner_revision_mismatch")
        _require(any(step.request_id == request_id for step in steps),
                 "public_request_has_no_closed_action")
        last_history_sequence = max(
            [step.sequence for step in steps]
            + [receipt.sequence for receipt in receipts]
        )
        _require(last_history_sequence < export.end_sequence,
                 "history_not_before_current_boundary")
        baseline_revision = _product_revision(baseline_binding)

        response = scope_receipt.PublicResponseOrigin(
            run_id=run_id,
            attempt_id=attempt_id,
            request_id=request_id,
            candidate_revision=baseline_revision,
            record_id=_event_id(response_row),
            sequence=response_sequence,
            source_id=_event_id(response_row),
            text=public_text,
            source_sha256=source_sha,
            plan_parser_ref=PARSER_REF,
            plan_parser_revision=PARSER_REVISION,
            plan_parser_sha256=PARSER_SHA256,
            planned_files=planned_files,
            authority=authority,
            projection_sha256=_ZERO_PROJECTION,
        )
        response = _seal_record(response)
        owner = scope_receipt.OwnerContract(
            run_id=run_id,
            attempt_id=attempt_id,
            request_id=request_id,
            candidate_revision=baseline_revision,
            record_id=_event_id(owner_row),
            sequence=owner_sequence,
            revision=owner_revision,
            before=_snapshot(baseline_binding, planned_files),
            authority=authority,
            projection_sha256=_ZERO_PROJECTION,
        )
        owner = _seal_record(owner)
        prepared = scope_receipt.PreparedRequest(
            run_id=run_id,
            attempt_id=attempt_id,
            request_id=request_id,
            owner_contract_request_id=request_id,
            candidate_revision=baseline_revision,
            record_id=_event_id(prepared_row),
            sequence=prepared_sequence,
            contract_revision=owner_revision,
            authority=authority,
            projection_sha256=_ZERO_PROJECTION,
        )
        prepared = _seal_record(prepared)
        current = scope_receipt.CurrentBinding(
            run_id=run_id,
            attempt_id=attempt_id,
            candidate_revision=current_revision,
            record_id=boundary_id,
            sequence=export.end_sequence,
            snapshot=_snapshot(binding, planned_files),
            authority=authority,
            projection_sha256=_ZERO_PROJECTION,
        )
        current = _seal_record(current)
        history = scope_receipt.ClosedActionHistory(
            candidate_revision=current_revision,
            start_record_id=owner.record_id,
            start_sequence=owner.sequence,
            end_record_id=current.record_id,
            end_sequence=current.sequence,
            complete=True,
            closed_steps=steps,
            action_receipts=receipts,
            authority=authority,
            history_projection_sha256=_ZERO_PROJECTION,
        )
        history = replace(
            history,
            history_projection_sha256=scope_receipt.history_slice_sha256(history),
        )
        producer_input = scope_receipt.ScopeReceiptInput(
            current_candidate_revision=current_revision,
            authority=authority,
            public_response=response,
            owner_contracts=(owner,),
            prepared_requests=(prepared,),
            current_binding=current,
            history=history,
        )
        candidate = scope_receipt.render_scope_receipt(
            producer_input, enabled=True, max_facts_bytes=max_facts_bytes
        )
        if candidate.status != "READY" or candidate.facts_json is None:
            return _noop(
                candidate.reason,
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                request_id=request_id,
                boundary_identity=_boundary_identity(boundary_id, task_attempt_id),
                candidate_revision=current_revision,
            )

        facts_bytes = scope_receipt.canonical_json_bytes({
            "schema": "kernel-scope-receipt-host-candidate/1",
            "status": "UNVERIFIED",
            "candidate_revision": current_revision,
            "limitation": (
                "Caller-supplied export shape and projection digests do not "
                "authenticate journal origin, task-attempt mapping, or delivery."
            ),
            "host_slice": {
                "task_attempt_id": task_attempt_id,
                "origin_request_id": request_id,
                "origin_transport_attempt_id": _event_id(submitted_row),
                "serving_request_id": serving_request_id,
                "start_sequence": export.start_sequence,
                "end_sequence": export.end_sequence,
                "current_boundary_event_id": boundary_id,
            },
            "scope_receipt_candidate": json.loads(candidate.facts_json),
        })
        _require(len(facts_bytes) <= max_facts_bytes, "rendered_facts_too_large")
        return ScopeHostAdapterPlan(
            status="UNVERIFIED",
            reason="candidate_facts_only_host_export_unverified",
            run_id=run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            request_id=request_id,
            serving_request_id=serving_request_id,
            task_attempt_id=task_attempt_id,
            scope_id=authority.scope_id,
            policy_id=authority.policy_id,
            tenant_id=authority.tenant_id,
            environment_id=authority.environment_id,
            current_boundary_event_id=boundary_id,
            boundary_identity=_boundary_identity(boundary_id, task_attempt_id),
            candidate_revision=current_revision,
            candidate_facts_json=facts_bytes.decode("utf-8", "strict"),
            candidate_facts_sha256=hashlib.sha256(facts_bytes).hexdigest(),
            utf8_byte_count=len(facts_bytes),
            producer_reason=candidate.reason,
        )
    except _AdapterError as exc:
        return _noop(str(exc))
    except (
        ValueError, TypeError, KeyError, AttributeError,
        UnicodeError, OverflowError, RecursionError, json.JSONDecodeError,
    ):
        return _noop("invalid_or_incomplete_host_export")


def verify_scope_receipt_wire(
    plan: ScopeHostAdapterPlan | None,
    *,
    serving_context: ScopeServingRequestContext | None,
    wire: bytes,
    expected_request_id: str,
    boundary_identity: str,
    expected_message_location: WireMessageLocation,
) -> ScopeWireCheck:
    """Compare exact message placement in raw JSON; transport remains unverified."""
    try:
        _require(type(plan) is ScopeHostAdapterPlan,
                 "candidate_plan_not_available")
        _require(type(plan.status) is str and plan.status == "UNVERIFIED",
                 "candidate_plan_not_available")
        _validate_plan_identity_fields(plan)
        serving_request_id, serving_boundary_identity = _validate_serving_context(
            serving_context, plan
        )
        request_id = _identifier(expected_request_id, "invalid_request_id")
        boundary = _identifier(boundary_identity, "invalid_boundary_identity")
        _require(
            request_id == serving_request_id
            and request_id == plan.serving_request_id
            and boundary == serving_boundary_identity
            and boundary == plan.boundary_identity,
                 "wire_identity_does_not_match_candidate")
        _require(type(expected_message_location) is WireMessageLocation,
                 "invalid_expected_message_location")
        _require(type(wire) is bytes and 0 < len(wire) <= MAX_WIRE_BYTES,
                 "invalid_wire_bytes")
        location = expected_message_location
        _require(type(location.index) is int and 0 <= location.index < 256,
                 "invalid_message_index")
        _require(type(location.role) is str and location.role in {
            "system", "developer", "user"
        }, "invalid_message_role")
        expected_content = _bounded_text(location.content, MAX_WIRE_BYTES)
        facts = plan.candidate_facts_json
        facts_bytes = _bounded_text(facts, scope_receipt.MAX_FACTS_BYTES).encode(
            "utf-8", "strict"
        )
        facts_value = _mapping(_strict_json_loads(facts_bytes),
                               "invalid_candidate_facts_json")
        _require(
            bool(facts_bytes)
            and type(plan.utf8_byte_count) is int
            and plan.utf8_byte_count == len(facts_bytes)
            and type(plan.candidate_facts_sha256) is str
            and plan.candidate_facts_sha256 == hashlib.sha256(facts_bytes).hexdigest()
            and facts_value.get("schema") == "kernel-scope-receipt-host-candidate/1"
            and facts_value.get("status") == "UNVERIFIED"
            and facts_value.get("candidate_revision") == plan.candidate_revision
            and _mapping(facts_value.get("host_slice"),
                         "invalid_candidate_host_slice").get(
                             "current_boundary_event_id"
                         ) == plan.current_boundary_event_id
            and facts in expected_content
            and expected_content.count(facts) == 1,
                 "expected_message_does_not_bind_candidate_facts")
        body = _mapping(_strict_json_loads(wire), "invalid_wire_json_object")
        messages = body.get("messages")
        _require(type(messages) is list and len(messages) <= 256,
                 "invalid_wire_messages")
        _require(location.index < len(messages), "wire_message_location_missing")
        message = _mapping(messages[location.index], "invalid_wire_message")
        content = message.get("content")
        _require(
            message.get("role") == location.role
            and type(content) is str
            and content == expected_content,
            "wire_message_location_mismatch",
        )
        message_bytes = content.encode("utf-8", "strict")
        return ScopeWireCheck(
            status="UNVERIFIED",
            reason="raw_body_placement_matches_caller_expectation_transport_unverified",
            request_id=request_id,
            boundary_identity=boundary,
            raw_http_sha256=hashlib.sha256(wire).hexdigest(),
            message_sha256=hashlib.sha256(message_bytes).hexdigest(),
            placement_matches=True,
            utf8_byte_count=len(wire),
        )
    except _AdapterError as exc:
        return ScopeWireCheck(status="NOOP", reason=str(exc))
    except (
        ValueError, TypeError, KeyError, AttributeError,
        UnicodeError, OverflowError, RecursionError, json.JSONDecodeError,
    ):
        return ScopeWireCheck(status="NOOP", reason="wire_identity_or_placement_unverified")


_ZERO_PROJECTION = "sha256:" + "0" * 64


def _validate_export_slice(
    export: ScopeHostExport,
) -> tuple[list[dict[str, object]], str, str]:
    _require(type(export.schema) is str and export.schema == EXPORT_SCHEMA,
             "unsupported_export_schema")
    _require(type(export.complete) is bool and export.complete, "incomplete_host_export")
    _require(_strict_int(export.omitted_record_count) == 0
             and _strict_int(export.pending_action_count) == 0,
             "omitted_or_pending_host_evidence")
    _require(type(export.records) is tuple and 1 <= len(export.records) <= MAX_EVENTS,
             "invalid_host_record_count")
    start = _strict_int(export.start_sequence)
    end = _strict_int(export.end_sequence)
    _require(start > 0 and end >= start and end - start + 1 == len(export.records),
             "host_sequence_slice_gap_or_size")
    rows: list[dict[str, object]] = []
    total_bytes = 0
    run_id: str | None = None
    task_id: str | None = None
    event_ids: set[str] = set()
    for offset, value in enumerate(export.records):
        row = _mapping(value, "invalid_host_event")
        encoded = scope_receipt.canonical_json_bytes(row)
        _require(len(encoded) <= MAX_EVENT_BYTES, "host_event_too_large")
        total_bytes += len(encoded)
        _require(total_bytes <= MAX_EXPORT_BYTES, "host_export_too_large")
        _require(row.get("schema") == JOURNAL_SCHEMA, "invalid_journal_schema")
        current_run = _identifier(row.get("run_id"), "invalid_journal_run")
        current_task = _identifier(row.get("task_id"), "invalid_journal_task")
        sequence = _strict_int(row.get("sequence"))
        event_id = _identifier(row.get("event_id"), "invalid_journal_event_id")
        _require(sequence == start + offset
                 and event_id == f"{current_run}:{sequence}",
                 "journal_event_identity_or_sequence_mismatch")
        _require(run_id is None or current_run == run_id, "journal_run_changed")
        _require(task_id is None or current_task == task_id, "journal_task_changed")
        _require(event_id not in event_ids, "duplicate_journal_event_id")
        run_id, task_id = current_run, current_task
        event_ids.add(event_id)
        rows.append(row)
    _require(run_id is not None and task_id is not None, "empty_host_export")
    _identifier(export.request_id, "invalid_request_id")
    _identifier(export.current_boundary_event_id, "invalid_current_boundary_id")
    return rows, run_id, task_id


def _receipt_rows(
    events: list[dict[str, object]], run_id: str
) -> dict[str, tuple[dict[str, object], int, str]]:
    _require(len(events) <= scope_receipt.MAX_ACTION_RECEIPTS, "too_many_action_receipts")
    result: dict[str, tuple[dict[str, object], int, str]] = {}
    for event in events:
        receipt = _mapping(event.get("receipt"), "missing_action_receipt_payload")
        receipt_id = _identifier(receipt.get("receipt_id"), "invalid_action_receipt_id")
        _require(re.fullmatch(r"[0-9a-f]{20}|[0-9a-f]{32}", receipt_id) is not None,
                 "invalid_action_receipt_id")
        _identifier(receipt.get("request_id"), "invalid_receipt_request_id")
        _require(receipt.get("run_id") == run_id, "receipt_run_mismatch")
        _require(receipt_id not in result, "duplicate_action_receipt_id")
        sequence = _strict_int(event["sequence"])
        result[receipt_id] = (receipt, sequence, _event_id(event))
    return result


def _step_rows(
    events: list[dict[str, object]],
) -> list[tuple[dict[str, object], int, str]]:
    _require(len(events) <= scope_receipt.MAX_CLOSED_STEPS, "too_many_closed_steps")
    result: list[tuple[dict[str, object], int, str]] = []
    for event in events:
        record = _mapping(event.get("record"), "missing_closed_action_record")
        record_id = _identifier(record.get("step_record_id"), "missing_step_record_id")
        without_digest = dict(record)
        del without_digest["step_record_id"]
        _require(record_id == scope_receipt.binding_sha256(without_digest),
                 "closed_action_host_digest_mismatch")
        _mapping(record.get("before_binding"), "closed_action_before_binding_missing")
        _mapping(record.get("after_binding"), "closed_action_after_binding_missing")
        result.append((record, _strict_int(event["sequence"]), _event_id(event)))
    result.sort(key=lambda item: item[1])
    return result


def _make_history_rows(
    raw_steps: list[tuple[dict[str, object], int, str]],
    raw_receipts: dict[str, tuple[dict[str, object], int, str]],
    authority: scope_receipt.AuthorityContext,
    planned_files: tuple[scope_receipt.PublicPlanFile, ...],
) -> tuple[
    tuple[scope_receipt.ClosedActionStep, ...],
    tuple[scope_receipt.ActionReceipt, ...],
    dict[str, object],
]:
    referenced: list[str] = []
    requests_by_receipt: dict[str, str] = {}
    revisions_by_receipt: dict[str, str] = {}
    step_rows: list[
        tuple[dict[str, object], int, str, tuple[str, ...], str]
    ] = []
    previous_binding: dict[str, object] | None = None
    for index, (record, sequence, event_id) in enumerate(raw_steps):
        receipt_values = record.get("receipt_ids")
        _require(type(receipt_values) is list
                 and 1 <= len(receipt_values) <= scope_receipt.MAX_ACTION_RECEIPTS,
                 "closed_action_receipts_missing_or_invalid")
        receipt_ids = tuple(_identifier(item, "invalid_closed_action_receipt_id")
                            for item in receipt_values)
        _require(len(set(receipt_ids)) == len(receipt_ids),
                 "duplicate_closed_action_receipt")
        before = _mapping(record.get("before_binding"), "invalid_before_binding")
        after = _mapping(record.get("after_binding"), "invalid_after_binding")
        if previous_binding is not None:
            _require(scope_receipt.canonical_json_bytes(before)
                     == scope_receipt.canonical_json_bytes(previous_binding),
                     "closed_action_binding_chain_gap")
        previous_binding = after
        requests: set[str] = set()
        after_revision = _product_revision(after)
        for receipt_id in receipt_ids:
            _require(receipt_id in raw_receipts, "closed_action_receipt_missing")
            receipt, receipt_sequence, _ = raw_receipts[receipt_id]
            request_id = _identifier(receipt.get("request_id"),
                                     "invalid_receipt_request_id")
            _require(receipt_sequence < sequence, "receipt_after_closed_action")
            requests.add(request_id)
            requests_by_receipt[receipt_id] = request_id
            revisions_by_receipt[receipt_id] = after_revision
            referenced.append(receipt_id)
        _require(len(requests) == 1, "mixed_request_or_bookkeeping_closure")
        step_rows.append((record, sequence, event_id, receipt_ids, next(iter(requests))))
    _require(set(referenced) == set(raw_receipts),
             "unmatched_action_receipt_or_closure")
    _require(len(referenced) <= scope_receipt.MAX_ACTION_RECEIPTS,
             "too_many_receipt_references")

    steps: list[scope_receipt.ClosedActionStep] = []
    for index, (record, sequence, event_id, receipt_ids, request_id) in enumerate(step_rows):
        before = _mapping(record["before_binding"], "invalid_before_binding")
        after = _mapping(record["after_binding"], "invalid_after_binding")
        step = scope_receipt.ClosedActionStep(
            run_id=authority.run_id,
            attempt_id=authority.attempt_id,
            request_id=request_id,
            candidate_revision=_product_revision(before),
            record_id=event_id,
            sequence=sequence,
            chain_index=index,
            action_receipt_ids=receipt_ids,
            before=_snapshot(before, planned_files),
            after=_snapshot(after, planned_files),
            authority=authority,
            projection_sha256=_ZERO_PROJECTION,
        )
        steps.append(_seal_record(step))

    receipts: list[scope_receipt.ActionReceipt] = []
    for receipt_id, (receipt, sequence, event_id) in sorted(
        raw_receipts.items(), key=lambda item: item[1][1]
    ):
        row = scope_receipt.ActionReceipt(
            run_id=authority.run_id,
            attempt_id=authority.attempt_id,
            request_id=requests_by_receipt[receipt_id],
            candidate_revision=revisions_by_receipt[receipt_id],
            record_id=event_id,
            receipt_id=receipt_id,
            sequence=sequence,
            authority=authority,
            projection_sha256=_ZERO_PROJECTION,
        )
        receipts.append(_seal_record(row))
    _require(bool(step_rows), "closed_action_history_missing")
    baseline = _mapping(step_rows[0][0]["before_binding"], "invalid_history_baseline")
    return tuple(steps), tuple(receipts), baseline


def _planned_files(text: str) -> tuple[scope_receipt.PublicPlanFile, ...]:
    source = _bounded_text(text, scope_receipt.MAX_PLAN_TEXT_BYTES)
    encoded = source.encode("utf-8", "strict")
    _require(bool(encoded), "empty_public_response")
    sites = step_contract.parse_model_plan_site_records(source)
    _require(type(sites) is tuple and 1 <= len(sites) <= scope_receipt.MAX_FILES,
             "no_bounded_public_plan_sites")
    paths: list[str] = []
    for site in sites:
        _require(type(site.path) is str and type(site.owner) is str,
                 "invalid_maintained_parser_site")
        scope_receipt._safe_relative_path(site.path)
        if site.path not in paths:
            paths.append(site.path)
    result: list[scope_receipt.PublicPlanFile] = []
    for path in paths:
        needle = path.encode("utf-8", "strict")
        starts: list[int] = []
        offset = 0
        while True:
            found = encoded.find(needle, offset)
            if found < 0:
                break
            end = found + len(needle)
            left_ok = found == 0 or _path_boundary(encoded[found - 1])
            right_ok = _path_right_boundary(encoded, end)
            if left_ok and right_ok:
                starts.append(found)
            offset = found + 1
        _require(len(starts) == 1, "planned_path_missing_or_ambiguous_literal_span")
        start = starts[0]
        result.append(scope_receipt.PublicPlanFile(path, start, start + len(needle)))
    _require(1 <= len(result) <= scope_receipt.MAX_FILES, "invalid_public_plan_file_count")
    return tuple(result)


def _path_boundary(byte: int) -> bool:
    return byte < 0x80 and _PATH_TOKEN.fullmatch(chr(byte)) is None


def _path_right_boundary(encoded: bytes, end: int) -> bool:
    if not 0 <= end <= len(encoded):
        return False
    if end == len(encoded):
        return True
    if encoded[end] != ord("."):
        return _path_boundary(encoded[end])
    after_period = end + 1
    return (after_period == len(encoded)
            or _path_boundary(encoded[after_period]))


def _snapshot(
    binding: dict[str, object], planned: tuple[scope_receipt.PublicPlanFile, ...]
) -> scope_receipt.BindingSnapshot:
    _product_revision(binding)
    product_files = _mapping(binding.get("product_files"), "missing_product_files")
    hashes: list[scope_receipt.FileHash] = []
    for item in planned:
        value = product_files.get(item.path)
        if value is None or (type(value) is str and value.startswith("symlink:")):
            digest: str | None = None
        else:
            _require(type(value) is str and value.startswith("sha256:"),
                     "invalid_product_file_hash")
            digest = value.removeprefix("sha256:")
            _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                     "invalid_product_file_hash")
        hashes.append(scope_receipt.FileHash(item.path, digest))
    return scope_receipt.BindingSnapshot(binding=binding, file_hashes=tuple(hashes))


def _product_revision(binding: Mapping[str, object]) -> str:
    _require(type(binding) is dict, "invalid_binding")
    product_files = _mapping(binding.get("product_files"), "missing_product_files")
    revision = _identifier(binding.get("product_revision"),
                           "invalid_product_revision")
    _require(revision == scope_receipt.binding_sha256(product_files),
             "product_revision_binding_mismatch")
    return revision


def _prepared_contract(event: Mapping[str, object]) -> dict[str, object]:
    return _mapping(event.get("request_action_contract"),
                    "missing_prepared_action_contract")


def _matching_prepared_event(
    events: list[dict[str, object]], request_id: str
) -> dict[str, object]:
    matching: list[dict[str, object]] = []
    for event in events:
        contract = _prepared_contract(event)
        top_request = event.get("request_id")
        nested_request = contract.get("request_id")
        if top_request == request_id or nested_request == request_id:
            _require(top_request == nested_request == request_id,
                     "prepared_request_id_projection_mismatch")
            matching.append(event)
    _require(len(matching) == 1, "missing_or_conflicting_prepared_request")
    return matching[0]


def _validate_serving_context(
    value: ScopeServingRequestContext | None,
    plan: ScopeHostAdapterPlan,
) -> tuple[str, str]:
    _require(type(value) is ScopeServingRequestContext,
             "missing_serving_request_context")
    boundary = _bounded_host_event(
        value.boundary_event, "invalid_serving_boundary_event"
    )
    owner = _bounded_host_event(
        value.provider_contract_event,
        "invalid_serving_provider_contract_event",
    )
    prepared = _bounded_host_event(
        value.request_prepared_event, "invalid_serving_prepared_event"
    )
    _require(boundary.get("schema") == JOURNAL_SCHEMA
             and boundary.get("kind") == "scope_receipt_binding_boundary"
             and owner.get("schema") == JOURNAL_SCHEMA
             and owner.get("kind") == "provider_boundary_contract_decided"
             and prepared.get("schema") == JOURNAL_SCHEMA
             and prepared.get("kind") == "request_prepared",
             "invalid_serving_event_kinds")
    boundary_id = _event_id(boundary)
    owner_id = _event_id(owner)
    prepared_id = _event_id(prepared)
    boundary_sequence = _strict_int(boundary.get("sequence"))
    owner_sequence = _strict_int(owner.get("sequence"))
    prepared_sequence = _strict_int(prepared.get("sequence"))
    _require(
        boundary_id == plan.current_boundary_event_id
        and boundary_id == f"{plan.run_id}:{boundary_sequence}"
        and owner_id == f"{plan.run_id}:{owner_sequence}"
        and prepared_id == f"{plan.run_id}:{prepared_sequence}"
        and boundary.get("run_id") == plan.run_id
        and boundary.get("task_id") == plan.task_id
        and owner.get("run_id") == plan.run_id
        and owner.get("task_id") == plan.task_id
        and prepared.get("run_id") == plan.run_id
        and prepared.get("task_id") == plan.task_id
        and boundary_sequence < owner_sequence < prepared_sequence,
        "serving_event_identity_or_order_mismatch",
    )
    task_attempt = _mapping(boundary.get("task_attempt"),
                            "missing_serving_task_attempt")
    _require(
        task_attempt.get("run_id") == plan.run_id
        and task_attempt.get("task_id") == plan.task_id
        and task_attempt.get("attempt_id") == plan.attempt_id
        and task_attempt.get("task_attempt_id") == plan.task_attempt_id
        and task_attempt.get("scope_id") == plan.scope_id
        and task_attempt.get("policy_id") == plan.policy_id
        and task_attempt.get("tenant_id") == plan.tenant_id
        and task_attempt.get("environment_id") == plan.environment_id,
        "serving_task_attempt_authority_mismatch",
    )
    binding = _mapping(boundary.get("binding"), "missing_serving_binding")
    _require(_product_revision(binding) == plan.candidate_revision,
             "serving_binding_revision_mismatch")
    _require(
        binding.get("authorization_scope") == plan.scope_id
        and binding.get("policy_identity") == plan.policy_id
        and binding.get("tenant_identity") == plan.tenant_id
        and binding.get("environment_revision") == plan.environment_id,
        "serving_binding_authority_mismatch",
    )
    serving_map = _mapping(boundary.get("serving_request"),
                           "missing_serving_request_map")
    request_id = _identifier(serving_map.get("request_id"),
                             "invalid_serving_request_id")
    contract = _prepared_contract(prepared)
    owner_revision = _identifier(owner.get("revision"),
                                 "invalid_serving_owner_revision")
    _require(
        owner_revision == scope_receipt.binding_sha256(binding),
        "serving_owner_binding_revision_mismatch",
    )
    decision_digest = _identifier(owner.get("decision_digest"),
                                  "invalid_serving_decision_digest")
    _require(
        owner.get("request_id") == request_id
        and prepared.get("request_id") == request_id
        and request_id == plan.serving_request_id
        and contract.get("request_id") == request_id
        and contract.get("revision") == owner_revision
        and contract.get("decision_digest") == decision_digest,
        "serving_prepared_boundary_join_mismatch",
    )
    _require(len({boundary_id, owner_id, prepared_id}) == 3,
             "duplicate_serving_event_id")
    identity = _boundary_identity(boundary_id, plan.task_attempt_id or "")
    return request_id, identity


def _public_response_text(response: object) -> str:
    payload = _mapping(response, "missing_public_provider_response")
    choices = payload.get("choices")
    _require(type(choices) is list and bool(choices),
             "missing_public_response_choice")
    first = _mapping(choices[0], "invalid_public_response_choice")
    _require(first.get("finish_reason") == "stop",
             "public_response_not_complete_stop")
    message = _mapping(first.get("message"), "missing_public_response_message")
    content = message.get("content")
    return _bounded_text(content, scope_receipt.MAX_PLAN_TEXT_BYTES)


def _one(by_kind: dict[str, list[dict[str, object]]], kind: str) -> dict[str, object]:
    rows = by_kind.get(kind, [])
    _require(len(rows) == 1, f"missing_or_conflicting_{kind}")
    return rows[0]


def _seal_record(record: object) -> object:
    return replace(record, projection_sha256=scope_receipt.evidence_record_sha256(record))


def _event_id(event: Mapping[str, object]) -> str:
    return _identifier(event.get("event_id"), "invalid_event_id")


def _boundary_identity(event_id: str, task_attempt_id: str) -> str:
    return _bounded_text(
        f"{event_id}|{task_attempt_id}", scope_receipt.MAX_ID_BYTES
    )


def _strict_json_loads(raw: bytes) -> object:
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_WIRE_BYTES,
             "invalid_wire_bytes")

    def reject_constant(value: str) -> object:
        raise ValueError("nonfinite_wire_json")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            _require(key not in result, "duplicate_wire_json_key")
            result[key] = value
        return result

    parsed = json.loads(raw.decode("utf-8", "strict"),
                        object_pairs_hook=reject_duplicates,
                        parse_constant=reject_constant)
    _json_bounds(parsed)
    return parsed


def _json_bounds(value: object, *, max_text_bytes: int = MAX_WIRE_BYTES) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    text_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _require(nodes <= 20_000 and depth <= 32, "wire_json_depth_or_node_limit")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            _require(item.bit_length() <= 63, "wire_json_integer_too_large")
            continue
        if type(item) is float:
            _require(math.isfinite(item), "wire_json_nonfinite_number")
            continue
        if type(item) is str:
            text_bytes += len(_bounded_text(item, MAX_EVENT_BYTES).encode("utf-8"))
            _require(text_bytes <= max_text_bytes, "wire_json_text_too_large")
            continue
        if type(item) not in {dict, list}:
            raise _AdapterError("wire_json_nonjson_value")
        identity = id(item)
        _require(identity not in seen, "wire_json_cycle")
        seen.add(identity)
        if type(item) is dict:
            for key, nested in item.items():
                _require(type(key) is str, "wire_json_nonstring_key")
                stack.append((key, depth + 1))
                stack.append((nested, depth + 1))
        else:
            stack.extend((nested, depth + 1) for nested in item)


def _parser_pin_matches() -> bool:
    """Read only the explicitly pinned parser source, with a strict byte cap."""
    source_path = getattr(step_contract, "__file__", None)
    if type(source_path) is not str or not source_path.endswith("/" + PARSER_REF):
        return False
    try:
        with open(source_path, "rb") as handle:
            source = handle.read(256 * 1024 + 1)
    except OSError:
        return False
    return len(source) <= 256 * 1024 and hashlib.sha256(source).hexdigest() == PARSER_SHA256


def _mapping(value: object, reason: str) -> dict[str, object]:
    _require(type(value) is dict and all(type(key) is str for key in value), reason)
    return value


def _validate_plan_identity_fields(plan: ScopeHostAdapterPlan) -> None:
    for field in (
        "run_id",
        "task_id",
        "attempt_id",
        "task_attempt_id",
        "request_id",
        "serving_request_id",
        "current_boundary_event_id",
        "boundary_identity",
        "candidate_revision",
        "scope_id",
        "policy_id",
        "tenant_id",
        "environment_id",
    ):
        _identifier(getattr(plan, field), f"invalid_plan_{field}")


def _bounded_host_event(value: object, reason: str) -> dict[str, object]:
    event = _mapping(value, reason)
    _json_bounds(event, max_text_bytes=MAX_EVENT_BYTES)
    encoded = scope_receipt.canonical_json_bytes(event)
    _require(len(encoded) <= MAX_EVENT_BYTES, "serving_event_too_large")
    return event


def _identifier(value: object, reason: str) -> str:
    text = _bounded_text(value, scope_receipt.MAX_ID_BYTES)
    _require(bool(text.strip()) and text.strip().casefold() not in {
        "unknown", "unknown_scope", "unknown_policy", "unknown_tenant",
        "unknown_environment",
    }, reason)
    return text


def _bounded_text(value: object, max_bytes: int) -> str:
    _require(type(value) is str and len(value) <= max_bytes, "invalid_or_oversized_text")
    encoded = value.encode("utf-8", "strict")
    _require(len(encoded) <= max_bytes, "invalid_or_oversized_text")
    return value


def _strict_int(value: object) -> int:
    _require(type(value) is int and value.bit_length() <= 63, "invalid_integer")
    return value


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _AdapterError(reason)


def _noop(
    reason: str,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
    request_id: str | None = None,
    boundary_identity: str | None = None,
    candidate_revision: str | None = None,
) -> ScopeHostAdapterPlan:
    return ScopeHostAdapterPlan(
        status="NOOP",
        reason=reason,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        request_id=request_id,
        boundary_identity=boundary_identity,
        candidate_revision=candidate_revision,
    )
