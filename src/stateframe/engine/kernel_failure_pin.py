"""Bounded, informational pins for canonical check failures.

This module consumes an adapter-produced, host-authenticated complete export. Its
projection digests detect inconsistent DTO contents; they do not authenticate
the host journal or prove that an export is complete. The adapter owns that
provenance boundary and must provide the whole ordered slice and current
prepared-injection boundary. No filesystem, environment, project, or execution
access is performed here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal


MAX_HISTORY_VIEWS = 128
MAX_REQUEST_ORIGINS = 256
MAX_ID_UTF8_BYTES = 256
MAX_SELECTOR_UTF8_BYTES = 512
MAX_SCOPE_UTF8_BYTES = 1024
MAX_PREVIEW_UTF8_BYTES = 1200
MAX_LINE_UTF8_BYTES = 1200
MAX_HISTORY_JSON_BYTES = 512 * 1024
DEFAULT_MAX_FACTS_UTF8_BYTES = 16 * 1024
MAX_FACTS_UTF8_BYTES = 32 * 1024
MAX_SEQUENCE = (1 << 63) - 1

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COUNT_NAMES = frozenset({"collected", "passed", "failed", "errored", "skipped", "deselected"})
_CANONICAL_OUTCOMES = frozenset({
    "PASS", "BEHAVIOR_FAIL", "INVALID_CHECK", "ENVIRONMENT",
    "RESOURCE_LIMIT", "NOT_RUN", "UNKNOWN",
})
_UNKNOWN_IDENTIFIERS = frozenset({"unknown", "unknown_id", "unknown_revision", "<unknown>"})


@dataclass(frozen=True, slots=True)
class PreparedInjection:
    """Adapter-bound current request boundary, after the complete history slice."""

    record_id: str
    event_sequence: int
    run_id: str
    attempt_id: str
    request_id: str
    candidate_revision: str
    context_sha256: str
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class FailureAuthority:
    run_id: str
    attempt_id: str
    current_request_id: str
    current_revision: str
    selector: str
    selector_scope: tuple[str, ...]
    prepared_injection: PreparedInjection


@dataclass(frozen=True, slots=True)
class SliceBoundary:
    boundary_id: str
    kind: Literal["start", "end"]
    event_sequence: int
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class RequestOriginEvent:
    record_id: str
    event_sequence: int
    run_id: str
    attempt_id: str
    request_id: str
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalCheckReceipt:
    """Public receipt projection; receipt_sequence is a step ordinal, not time."""

    receipt_id: str
    execution_id: str
    schema: str
    selector: str
    selector_scope: tuple[str, ...]
    receipt_sequence: int
    status: str
    outcome: str
    failure_phase: str
    revision: str
    execution_complete: bool
    termination_reason: str | None
    capture_complete: bool | None
    capture_status: str
    counts: tuple[tuple[str, int | None], ...]
    collected_tests: int | None
    assertion_context_preview: str
    assertion_context_omitted_bytes: int
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class CheckReceiptJournalEvent:
    record_id: str
    event_sequence: int
    event_type: str
    producer: str
    dispatch: bool
    execution_id: str
    receipt_id: str
    selector: str
    selector_scope: tuple[str, ...]
    status: str
    outcome: str
    failure_phase: str
    revision: str
    execution_complete: bool
    termination_reason: str | None
    counts: tuple[tuple[str, int | None], ...]
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class SubprocessCompletionEvent:
    """One host event, potentially grouping several unordered check rows.

    ``canonical_revision_from_binding`` is adapter-derived from this event's
    filtered binding with the pinned receipt-revision rule; it is not a direct
    field on ``test_subprocess_completed``.
    """

    record_id: str
    event_sequence: int
    event_type: str
    run_id: str
    attempt_id: str
    request_id: str
    execution_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    selectors: tuple[str, ...]
    selector_scope: tuple[str, ...]
    canonical_revision_from_binding: str
    binding_stable: bool
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalExecutionView:
    receipt: CanonicalCheckReceipt
    receipt_event: CheckReceiptJournalEvent
    subprocess_event: SubprocessCompletionEvent
    subprocess_item_index: int


@dataclass(frozen=True, slots=True)
class CompleteHistoryExport:
    export_id: str
    run_id: str
    attempt_id: str
    start_boundary: SliceBoundary
    end_boundary: SliceBoundary
    complete: bool
    omitted_event_count: int
    pending_execution_count: int
    conflicting_record_count: int
    request_origins: tuple[RequestOriginEvent, ...]
    views: tuple[CanonicalExecutionView, ...]
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class PublicFailureLine:
    """Caller-selected half-open UTF-8 byte span in one public preview line."""

    receipt_id: str
    start_byte: int
    end_byte: int
    text: str


@dataclass(frozen=True, slots=True)
class FailurePinInput:
    authority: FailureAuthority
    history: CompleteHistoryExport
    failure_line: PublicFailureLine


@dataclass(frozen=True, slots=True)
class FailurePinPlan:
    status: Literal["READY", "NOOP"]
    reason: str
    facts_json: str | None = None
    facts_sha256: str | None = None
    utf8_byte_count: int = 0


def receipt_handle(execution_id: str) -> str:
    """Return the canonical receipt handle derived by kernel_check_receipts."""

    if type(execution_id) is not str or not _bounded_text(execution_id, MAX_ID_UTF8_BYTES):
        raise ValueError("invalid_execution_id")
    return "check-receipt:" + hashlib.sha256(execution_id.encode("utf-8")).hexdigest()[:24]


def projection_sha256_for_prepared_injection(value: PreparedInjection) -> str:
    return _projection_sha256("prepared-injection", _prepared_injection_payload(value))


def projection_sha256_for_boundary(value: SliceBoundary) -> str:
    return _projection_sha256("slice-boundary", _boundary_payload(value))


def projection_sha256_for_request_origin(value: RequestOriginEvent) -> str:
    return _projection_sha256("request-origin", _request_origin_payload(value))


def projection_sha256_for_receipt(value: CanonicalCheckReceipt) -> str:
    return _projection_sha256("canonical-check-receipt", _receipt_payload(value))


def projection_sha256_for_receipt_event(value: CheckReceiptJournalEvent) -> str:
    return _projection_sha256("check-receipt-journal-event", _receipt_event_payload(value))


def projection_sha256_for_subprocess_event(value: SubprocessCompletionEvent) -> str:
    return _projection_sha256("subprocess-completion-event", _subprocess_payload(value))


def projection_sha256_for_history(value: CompleteHistoryExport) -> str:
    return _projection_sha256("complete-history-export", _history_payload(value))


def render_failure_pin(
    value: FailurePinInput | None,
    *,
    enabled: bool = False,
    max_facts_utf8_bytes: int = DEFAULT_MAX_FACTS_UTF8_BYTES,
) -> FailurePinPlan:
    """Return a deterministic public evidence pin; never executes or grades work."""

    if type(enabled) is not bool:
        return _noop("invalid_enabled")
    if not enabled:
        return _noop("disabled")
    if type(max_facts_utf8_bytes) is not int or not 1 <= max_facts_utf8_bytes <= MAX_FACTS_UTF8_BYTES:
        return _noop("invalid_output_budget")
    if type(value) is not FailurePinInput:
        return _noop("missing_input")

    try:
        invalid = _validate_input(value)
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        return _noop("malformed_or_oversized_source_projection")
    if invalid is not None:
        return _noop(invalid)

    authority = value.authority
    history = value.history
    target_views = tuple(sorted(
        (view for view in history.views
        if view.receipt.selector == authority.selector
        and view.receipt.selector_scope == authority.selector_scope),
        key=lambda view: view.subprocess_event.event_sequence,
    ))
    failures = tuple(view for view in target_views if view.receipt.outcome == "BEHAVIOR_FAIL")
    if not failures:
        return _noop("no_canonical_behavior_failure")

    failure_view = failures[-1]
    receipt = failure_view.receipt
    if not _completed_execution(receipt):
        return _noop("latest_failure_not_complete")
    line = value.failure_line
    if type(line) is not PublicFailureLine:
        return _noop("failure_line_not_bound_to_complete_public_line")
    if line.receipt_id != receipt.receipt_id or not _line_is_complete(receipt, line):
        return _noop("failure_line_not_bound_to_complete_public_line")

    later = tuple(
        view for view in target_views
        if view.subprocess_event.event_sequence > failure_view.subprocess_event.event_sequence
    )
    latest_later = later[-1] if later else None
    qualifying_passes = tuple(view for view in later if _qualifying_pass(view.receipt))
    clearance = qualifying_passes[-1] if qualifying_passes else None

    if clearance is None:
        disposition = (
            "PINNED_CURRENT_REVISION"
            if receipt.revision == authority.current_revision
            else "PINNED_OTHER_REVISION"
        )
    else:
        disposition = (
            "CLEARED_CURRENT_REVISION"
            if clearance.receipt.revision == authority.current_revision
            else "CLEARED_OTHER_REVISION"
        )

    try:
        facts = _facts_payload(
            authority=authority,
            history=history,
            failure_view=failure_view,
            failure_line=line,
            disposition=disposition,
            latest_later=latest_later,
            clearance=clearance,
        )
        facts_bytes = _canonical_json_bytes(facts, max_bytes=max_facts_utf8_bytes)
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        return _noop("public_facts_exceed_budget")
    return FailurePinPlan(
        status="READY",
        reason="canonical_failure_pinned",
        facts_json=facts_bytes.decode("utf-8"),
        facts_sha256="sha256:" + hashlib.sha256(facts_bytes).hexdigest(),
        utf8_byte_count=len(facts_bytes),
    )


def _validate_input(value: FailurePinInput) -> str | None:
    authority, history = value.authority, value.history
    if type(authority) is not FailureAuthority or type(history) is not CompleteHistoryExport:
        return "invalid_input_shape"
    if not _valid_identifier(authority.run_id) or not _valid_identifier(authority.attempt_id):
        return "invalid_authority_identity"
    if not _valid_identifier(authority.current_request_id):
        return "missing_current_request_identity"
    if not _valid_revision(authority.current_revision):
        return "unknown_current_revision"
    if not _bounded_text(authority.selector, MAX_SELECTOR_UTF8_BYTES):
        return "invalid_selector"
    if not _valid_selector_scope(authority.selector_scope):
        return "invalid_selector_scope"
    injection_error = _validate_injection(authority.prepared_injection, authority)
    if injection_error is not None:
        return injection_error

    if not _valid_identifier(history.export_id):
        return "invalid_history_export_id"
    if history.run_id != authority.run_id or history.attempt_id != authority.attempt_id:
        return "history_authority_mismatch"
    if type(history.complete) is not bool or not history.complete:
        return "history_not_complete"
    for count in (history.omitted_event_count, history.pending_execution_count, history.conflicting_record_count):
        if not _nonnegative_int(count):
            return "invalid_history_count"
        if count != 0:
            return "history_has_omitted_pending_or_conflicting_records"
    if type(history.request_origins) is not tuple or type(history.views) is not tuple:
        return "history_rows_not_immutable_tuples"
    if not history.views or len(history.views) > MAX_HISTORY_VIEWS:
        return "history_view_count_out_of_bounds"
    if not history.request_origins or len(history.request_origins) > MAX_REQUEST_ORIGINS:
        return "request_origin_count_out_of_bounds"

    boundary_error = _validate_boundaries(history, authority.prepared_injection.event_sequence)
    if boundary_error is not None:
        return boundary_error

    request_ids: set[str] = set()
    record_ids: set[str] = {history.start_boundary.boundary_id, history.end_boundary.boundary_id}
    if authority.prepared_injection.record_id in record_ids:
        return "duplicate_or_invalid_history_record_id"
    record_ids.add(authority.prepared_injection.record_id)
    event_sequences: set[int] = set()
    last_request_sequence = -1
    current_injection_sequence = authority.prepared_injection.event_sequence
    for request in history.request_origins:
        if type(request) is not RequestOriginEvent:
            return "invalid_request_origin_record"
        if not _valid_identifier(request.record_id) or request.record_id in record_ids:
            return "duplicate_or_invalid_history_record_id"
        record_ids.add(request.record_id)
        if not _valid_identifier(request.request_id) or request.request_id in request_ids:
            return "duplicate_or_invalid_request_id"
        request_ids.add(request.request_id)
        if request.run_id != history.run_id or request.attempt_id != history.attempt_id:
            return "request_origin_context_mismatch"
        if not _valid_event_sequence(request.event_sequence, history, current_injection_sequence):
            return "request_origin_sequence_out_of_bounds"
        if request.event_sequence <= last_request_sequence:
            return "request_origins_not_in_host_order"
        last_request_sequence = request.event_sequence
        if request.event_sequence in event_sequences:
            return "duplicate_host_event_sequence"
        event_sequences.add(request.event_sequence)
        if not _digest_matches("request-origin", _request_origin_payload(request), request.projection_sha256):
            return "request_origin_projection_mismatch"
    if authority.current_request_id in request_ids:
        return "current_request_must_follow_historical_slice"

    previous_completion_sequence = -1
    process_groups_by_id: dict[str, SubprocessCompletionEvent] = {}
    process_groups_by_sequence: dict[int, str] = {}
    execution_ids: set[str] = set()
    receipt_ids: set[str] = set()
    target_group_counts: dict[int, int] = {}
    group_indices: dict[str, set[int]] = {}
    for view in history.views:
        if type(view) is not CanonicalExecutionView:
            return "invalid_execution_view"
        receipt, receipt_event, process = view.receipt, view.receipt_event, view.subprocess_event
        receipt_error = _validate_receipt(receipt)
        if receipt_error is not None:
            return receipt_error
        event_error = _validate_receipt_event(receipt_event)
        if event_error is not None:
            return event_error
        if not _valid_event_sequence(receipt_event.event_sequence, history, authority.prepared_injection.event_sequence):
            return "history_event_sequence_out_of_bounds"
        process_error = _validate_subprocess_event(
            process, history, authority.prepared_injection.event_sequence
        )
        if process_error is not None:
            return process_error
        if not _valid_event_sequence_number(view.subprocess_item_index):
            return "invalid_subprocess_item_index"
        if not _join_receipt(receipt, receipt_event, process, view.subprocess_item_index):
            return "receipt_process_join_mismatch"
        group_indices.setdefault(process.record_id, set()).add(view.subprocess_item_index)
        if receipt.execution_id in execution_ids or receipt.receipt_id in receipt_ids:
            return "duplicate_canonical_execution_or_receipt_id"
        execution_ids.add(receipt.execution_id)
        receipt_ids.add(receipt.receipt_id)
        if process.request_id not in request_ids:
            return "subprocess_missing_request_origin"
        origin = next(row for row in history.request_origins if row.request_id == process.request_id)
        if not origin.event_sequence < min(receipt_event.event_sequence, process.event_sequence):
            return "request_receipt_subprocess_order_mismatch"
        previous_group = process_groups_by_id.get(process.record_id)
        if receipt_event.record_id in record_ids:
            return "duplicate_or_invalid_history_record_id"
        record_ids.add(receipt_event.record_id)
        if process.record_id in record_ids and previous_group is None:
            return "duplicate_or_invalid_history_record_id"
        if not _valid_event_sequence(receipt_event.event_sequence, history, current_injection_sequence):
            return "history_event_sequence_out_of_bounds"
        if receipt_event.event_sequence in event_sequences:
            return "duplicate_host_event_sequence"
        event_sequences.add(receipt_event.event_sequence)
        if process.event_sequence < previous_completion_sequence:
            return "execution_views_not_in_host_completion_order"
        previous_completion_sequence = process.event_sequence
        if previous_group is None:
            if process.event_sequence in event_sequences:
                return "duplicate_host_event_sequence"
            process_groups_by_id[process.record_id] = process
            process_groups_by_sequence[process.event_sequence] = process.record_id
            record_ids.add(process.record_id)
            event_sequences.add(process.event_sequence)
        elif previous_group != process:
            return "conflicting_subprocess_group_projection"
        if process_groups_by_sequence.get(process.event_sequence) != process.record_id:
            return "duplicate_subprocess_event_sequence"
        if receipt.selector == authority.selector and receipt.selector_scope == authority.selector_scope:
            target_group_counts[process.event_sequence] = target_group_counts.get(process.event_sequence, 0) + 1
            if target_group_counts[process.event_sequence] > 1:
                return "ambiguous_same_event_target_chronology"
        if not _digest_matches("canonical-check-receipt", _receipt_payload(receipt), receipt.projection_sha256):
            return "receipt_projection_mismatch"
        if not _digest_matches("check-receipt-journal-event", _receipt_event_payload(receipt_event), receipt_event.projection_sha256):
            return "receipt_event_projection_mismatch"
        if previous_group is None and not _digest_matches(
            "subprocess-completion-event", _subprocess_payload(process), process.projection_sha256
        ):
            return "subprocess_projection_mismatch"

    for process_id, process in process_groups_by_id.items():
        expected_indices = set(range(len(process.execution_ids)))
        if group_indices.get(process_id) != expected_indices:
            return "subprocess_group_rows_incomplete"

    try:
        history_digest = _projection_sha256("complete-history-export", _history_payload(history))
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        return "history_projection_exceeds_budget"
    if not _digest_field(history.projection_sha256) or history.projection_sha256 != history_digest:
        return "history_projection_mismatch"
    return None


def _validate_injection(value: PreparedInjection, authority: FailureAuthority) -> str | None:
    if type(value) is not PreparedInjection:
        return "missing_prepared_injection_boundary"
    if not _valid_identifier(value.record_id):
        return "invalid_prepared_injection_record"
    if not _valid_event_sequence_number(value.event_sequence):
        return "invalid_prepared_injection_sequence"
    if value.run_id != authority.run_id or value.attempt_id != authority.attempt_id:
        return "prepared_injection_context_mismatch"
    if value.request_id != authority.current_request_id:
        return "prepared_injection_request_mismatch"
    if value.candidate_revision != authority.current_revision or not _valid_revision(value.candidate_revision):
        return "prepared_injection_revision_mismatch"
    if not _digest_field(value.context_sha256):
        return "missing_prepared_context_digest"
    if not _digest_matches("prepared-injection", _prepared_injection_payload(value), value.projection_sha256):
        return "prepared_injection_projection_mismatch"
    return None


def _validate_boundaries(history: CompleteHistoryExport, injection_sequence: int) -> str | None:
    start, end = history.start_boundary, history.end_boundary
    if type(start) is not SliceBoundary or type(end) is not SliceBoundary:
        return "missing_history_slice_boundaries"
    if not _valid_identifier(start.boundary_id) or not _valid_identifier(end.boundary_id):
        return "invalid_history_slice_boundary_id"
    if start.boundary_id == end.boundary_id or start.kind != "start" or end.kind != "end":
        return "invalid_history_slice_boundary_pair"
    if not _valid_event_sequence_number(start.event_sequence) or not _valid_event_sequence_number(end.event_sequence):
        return "invalid_history_slice_boundary_sequence"
    if start.event_sequence >= end.event_sequence or end.event_sequence >= injection_sequence:
        return "history_slice_not_before_current_injection"
    if not _digest_matches("slice-boundary", _boundary_payload(start), start.projection_sha256):
        return "history_start_boundary_projection_mismatch"
    if not _digest_matches("slice-boundary", _boundary_payload(end), end.projection_sha256):
        return "history_end_boundary_projection_mismatch"
    return None


def _validate_receipt(value: CanonicalCheckReceipt) -> str | None:
    if type(value) is not CanonicalCheckReceipt:
        return "invalid_canonical_receipt"
    if value.schema != "stateframe-check-output-receipt/1":
        return "unexpected_canonical_receipt_schema"
    if not _valid_identifier(value.execution_id) or not _valid_identifier(value.receipt_id):
        return "invalid_receipt_identity"
    if value.receipt_id != receipt_handle(value.execution_id):
        return "receipt_handle_does_not_match_execution"
    for text, limit, reason in (
        (value.selector, MAX_SELECTOR_UTF8_BYTES, "invalid_receipt_selector"),
        (value.status, 128, "invalid_receipt_status"),
        (value.outcome, 128, "invalid_receipt_outcome"),
        (value.failure_phase, 256, "invalid_receipt_phase"),
        (value.capture_status, 128, "invalid_capture_status"),
    ):
        if not _bounded_text(text, limit):
            return reason
    if not _bounded_text(value.assertion_context_preview, MAX_PREVIEW_UTF8_BYTES, allow_empty=True):
        return "invalid_public_preview"
    preview_bytes = len(value.assertion_context_preview.encode("utf-8"))
    if not _valid_selector_scope(value.selector_scope):
        return "invalid_receipt_scope"
    if value.termination_reason is not None and not _bounded_text(value.termination_reason, 128):
        return "invalid_termination_reason"
    if not _valid_revision(value.revision):
        return "unknown_receipt_revision"
    if type(value.execution_complete) is not bool or (
        value.capture_complete is not None and type(value.capture_complete) is not bool
    ):
        return "invalid_receipt_completeness_flag"
    expected_capture_status = {True: "COMPLETE", False: "PARTIAL", None: "UNKNOWN"}[
        value.capture_complete
    ]
    if value.capture_status != expected_capture_status:
        return "capture_status_completeness_mismatch"
    if value.outcome not in _CANONICAL_OUTCOMES:
        return "unknown_canonical_outcome"
    if value.outcome == "PASS" and value.status != "PASS":
        return "canonical_pass_status_mismatch"
    if value.outcome == "BEHAVIOR_FAIL" and value.status != "FAIL":
        return "canonical_behavior_failure_status_mismatch"
    if not _valid_event_sequence_number(value.receipt_sequence):
        return "invalid_receipt_step_ordinal"
    if not _nonnegative_int(value.assertion_context_omitted_bytes):
        return "invalid_preview_omitted_byte_count"
    if value.assertion_context_omitted_bytes and not (
        MAX_PREVIEW_UTF8_BYTES - 3 <= preview_bytes <= MAX_PREVIEW_UTF8_BYTES
    ):
        return "truncated_preview_length_not_source_shaped"
    if value.collected_tests is not None and not _nonnegative_int(value.collected_tests):
        return "invalid_collected_test_count"
    count_error = _validate_counts(value.counts, value.collected_tests)
    if count_error is not None:
        return count_error
    return None


def _validate_receipt_event(value: CheckReceiptJournalEvent) -> str | None:
    if type(value) is not CheckReceiptJournalEvent:
        return "invalid_check_receipt_journal_event"
    if not _valid_identifier(value.record_id):
        return "invalid_check_receipt_event_id"
    if not _valid_event_sequence_number(value.event_sequence):
        return "invalid_check_receipt_event_sequence"
    if value.event_type != "check_receipt" or value.producer != "kernel_check_receipts":
        return "unexpected_check_receipt_event_origin"
    if type(value.dispatch) is not bool or value.dispatch:
        return "check_receipt_event_must_be_non_dispatch"
    for text, limit in (
        (value.execution_id, MAX_ID_UTF8_BYTES),
        (value.receipt_id, MAX_ID_UTF8_BYTES),
        (value.selector, MAX_SELECTOR_UTF8_BYTES),
        (value.status, 128),
        (value.outcome, 128),
        (value.failure_phase, 256),
    ):
        if not _bounded_text(text, limit):
            return "invalid_check_receipt_event_field"
    if not _valid_selector_scope(value.selector_scope):
        return "invalid_check_receipt_event_scope"
    if value.termination_reason is not None and not _bounded_text(value.termination_reason, 128):
        return "invalid_check_receipt_event_termination"
    if not _valid_revision(value.revision):
        return "unknown_receipt_event_revision"
    if type(value.execution_complete) is not bool:
        return "invalid_receipt_event_completeness"
    return _validate_counts(value.counts, None)


def _validate_subprocess_event(
    value: SubprocessCompletionEvent,
    history: CompleteHistoryExport,
    injection_sequence: int,
) -> str | None:
    if type(value) is not SubprocessCompletionEvent:
        return "invalid_subprocess_completion_event"
    for identifier in (value.record_id, value.run_id, value.attempt_id, value.request_id,
                       ):
        if not _valid_identifier(identifier):
            return "invalid_subprocess_completion_identity"
    if value.event_type != "test_subprocess_completed":
        return "unexpected_subprocess_completion_event_type"
    if value.run_id != history.run_id or value.attempt_id != history.attempt_id:
        return "subprocess_context_mismatch"
    if not _valid_event_sequence(value.event_sequence, history, injection_sequence):
        return "subprocess_completion_sequence_out_of_bounds"
    if type(value.execution_ids) is not tuple or type(value.receipt_ids) is not tuple:
        return "invalid_subprocess_canonical_id_arrays"
    if not 1 <= len(value.execution_ids) <= MAX_HISTORY_VIEWS or len(value.execution_ids) != len(value.receipt_ids):
        return "subprocess_canonical_id_array_length_mismatch"
    if any(not _valid_identifier(item) for item in (*value.execution_ids, *value.receipt_ids)):
        return "invalid_subprocess_canonical_id"
    if len(set(value.execution_ids)) != len(value.execution_ids) or len(set(value.receipt_ids)) != len(value.receipt_ids):
        return "duplicate_subprocess_canonical_id"
    if type(value.selectors) is not tuple or not 1 <= len(value.selectors) <= MAX_HISTORY_VIEWS:
        return "invalid_subprocess_selectors"
    if any(not _bounded_text(item, MAX_SELECTOR_UTF8_BYTES) for item in value.selectors):
        return "invalid_subprocess_selector"
    normalized_selectors = tuple(item.strip() for item in value.selectors if item.strip())
    if len(normalized_selectors) != len(value.selectors):
        return "invalid_subprocess_selector"
    if len(set(normalized_selectors)) != len(normalized_selectors):
        return "duplicate_subprocess_selector"
    if not _valid_selector_scope(value.selector_scope):
        return "invalid_subprocess_scope"
    if value.selector_scope != normalized_selectors:
        return "subprocess_scope_selectors_mismatch"
    if not _valid_revision(value.canonical_revision_from_binding):
        return "unknown_subprocess_binding_revision"
    if type(value.binding_stable) is not bool or not value.binding_stable:
        return "subprocess_binding_not_stable"
    return None


def _join_receipt(
    receipt: CanonicalCheckReceipt,
    event: CheckReceiptJournalEvent,
    process: SubprocessCompletionEvent,
    item_index: int,
) -> bool:
    normalized_selectors = tuple(selector.strip() for selector in process.selectors if selector.strip())
    selector_matches = (
        item_index < len(normalized_selectors)
        and normalized_selectors[item_index] == receipt.selector
    ) or (
        normalized_selectors == process.selector_scope
        and receipt.selector == ", ".join(process.selector_scope)
    )
    same_receipt = (
        type(item_index) is int
        and 0 <= item_index < len(process.execution_ids)
        and receipt.execution_id == event.execution_id == process.execution_ids[item_index]
        and receipt.receipt_id == event.receipt_id == process.receipt_ids[item_index]
        and receipt.selector == event.selector
        and selector_matches
        and receipt.selector_scope == event.selector_scope == process.selector_scope
        and receipt.status == event.status
        and receipt.outcome == event.outcome
        and receipt.failure_phase == event.failure_phase
        and receipt.revision == event.revision == process.canonical_revision_from_binding
        and receipt.execution_complete == event.execution_complete
        and receipt.termination_reason == event.termination_reason
        and receipt.counts == event.counts
    )
    return same_receipt


def _validate_counts(counts: tuple[tuple[str, int | None], ...], collected_tests: int | None) -> str | None:
    if type(counts) is not tuple or len(counts) > len(_COUNT_NAMES):
        return "invalid_receipt_counts_shape"
    names: set[str] = set()
    for row in counts:
        if type(row) is not tuple or len(row) != 2:
            return "invalid_receipt_count_entry"
        name, count = row
        if type(name) is not str or name not in _COUNT_NAMES or name in names:
            return "invalid_or_duplicate_receipt_count_name"
        names.add(name)
        if count is not None and not _nonnegative_int(count):
            return "invalid_receipt_count_value"
    passed = dict(counts).get("passed")
    if collected_tests is not None and passed is not None and passed > collected_tests:
        return "passed_count_exceeds_collected_tests"
    return None


def _completed_execution(receipt: CanonicalCheckReceipt) -> bool:
    return (
        receipt.execution_complete is True
        and receipt.termination_reason == "EXITED"
    )


def _qualifying_pass(receipt: CanonicalCheckReceipt) -> bool:
    if (
        receipt.outcome != "PASS"
        or not _completed_execution(receipt)
        or receipt.capture_complete is not True
    ):
        return False
    counts = dict(receipt.counts)
    passed = counts.get("passed")
    return type(passed) is int and passed > 0


def _line_is_complete(receipt: CanonicalCheckReceipt, line: PublicFailureLine) -> bool:
    if type(line) is not PublicFailureLine or type(line.start_byte) is not int or type(line.end_byte) is not int:
        return False
    if not _bounded_text(line.text, MAX_LINE_UTF8_BYTES):
        return False
    try:
        preview = receipt.assertion_context_preview.encode("utf-8")
        expected = line.text.encode("utf-8")
    except UnicodeError:
        return False
    start, end = line.start_byte, line.end_byte
    if start < 0 or end <= start or end > len(preview) or end - start > MAX_LINE_UTF8_BYTES:
        return False
    if start != 0 and preview[start - 1] != 0x0A:
        return False
    if b"\n" in preview[start:end]:
        return False
    if end < len(preview):
        if preview[end] != 0x0A:
            return False
    elif receipt.assertion_context_omitted_bytes != 0:
        return False
    try:
        actual = preview[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return actual == line.text and expected == preview[start:end]


def _facts_payload(
    *,
    authority: FailureAuthority,
    history: CompleteHistoryExport,
    failure_view: CanonicalExecutionView,
    failure_line: PublicFailureLine,
    disposition: str,
    latest_later: CanonicalExecutionView | None,
    clearance: CanonicalExecutionView | None,
) -> dict[str, object]:
    failure = failure_view.receipt
    failure_process = failure_view.subprocess_event
    failure_origin = next(
        row for row in history.request_origins if row.request_id == failure_process.request_id
    )
    later_receipt = latest_later.receipt if latest_later is not None else None
    later_process = latest_later.subprocess_event if latest_later is not None else None
    clearance_receipt = clearance.receipt if clearance is not None else None
    clearance_process = clearance.subprocess_event if clearance is not None else None
    return {
        "schema": "kernel-failure-pin/1",
        "disposition": disposition,
        "run_id": authority.run_id,
        "attempt_id": authority.attempt_id,
        "current_revision": authority.current_revision,
        "current_request_id": authority.current_request_id,
        "selector": authority.selector,
        "selector_scope": authority.selector_scope,
        "current_prepared_context": {
            "record_id": authority.prepared_injection.record_id,
            "event_sequence": authority.prepared_injection.event_sequence,
            "candidate_revision": authority.prepared_injection.candidate_revision,
            "context_sha256": authority.prepared_injection.context_sha256,
            "adapter_projection_sha256": authority.prepared_injection.projection_sha256,
        },
        "history_source": {
            "export_id": history.export_id,
            "adapter_projection_sha256": history.projection_sha256,
            "start_boundary_id": history.start_boundary.boundary_id,
            "start_event_sequence": history.start_boundary.event_sequence,
            "end_boundary_id": history.end_boundary.boundary_id,
            "end_event_sequence": history.end_boundary.event_sequence,
            "complete": history.complete,
            "omitted_event_count": history.omitted_event_count,
            "pending_execution_count": history.pending_execution_count,
            "conflicting_record_count": history.conflicting_record_count,
        },
        "failure": {
            "receipt_id": failure.receipt_id,
            "execution_id": failure.execution_id,
            "run_id": failure_process.run_id,
            "attempt_id": failure_process.attempt_id,
            "request_id": failure_process.request_id,
            "request_origin_record_id": failure_origin.record_id,
            "request_origin_adapter_projection_sha256": failure_origin.projection_sha256,
            "receipt_schema": failure.schema,
            "receipt_adapter_projection_sha256": failure.projection_sha256,
            "preview_utf8_sha256": "sha256:" + hashlib.sha256(
                failure.assertion_context_preview.encode("utf-8")
            ).hexdigest(),
            "journal_record_id": failure_view.receipt_event.record_id,
            "journal_event_type": failure_view.receipt_event.event_type,
            "journal_producer": failure_view.receipt_event.producer,
            "journal_adapter_projection_sha256": failure_view.receipt_event.projection_sha256,
            "host_completion_record_id": failure_process.record_id,
            "host_completion_event_sequence": failure_process.event_sequence,
            "host_completion_adapter_projection_sha256": failure_process.projection_sha256,
            "canonical_revision_from_binding": failure_process.canonical_revision_from_binding,
            "host_binding_stable": failure_process.binding_stable,
            "subprocess_request_id": failure_process.request_id,
            "subprocess_group_item_index": failure_view.subprocess_item_index,
            "receipt_sequence_step_ordinal": failure.receipt_sequence,
            "revision": failure.revision,
            "status": failure.status,
            "outcome": failure.outcome,
            "phase": failure.failure_phase,
            "execution_complete": failure.execution_complete,
            "termination_reason": failure.termination_reason,
            "capture_complete": failure.capture_complete,
            "capture_status": failure.capture_status,
            "line": failure_line.text,
            "line_byte_span": [failure_line.start_byte, failure_line.end_byte],
            "preview_omitted_bytes": failure.assertion_context_omitted_bytes,
        },
        "latest_later_exact_scope_observation": (
            {
                "receipt_id": later_receipt.receipt_id,
                "execution_id": later_receipt.execution_id,
                "request_id": later_process.request_id,
                "host_completion_event_sequence": later_process.event_sequence,
                "receipt_adapter_projection_sha256": later_receipt.projection_sha256,
                "host_completion_adapter_projection_sha256": later_process.projection_sha256,
                "revision": later_receipt.revision,
                "status": later_receipt.status,
                "outcome": later_receipt.outcome,
            }
            if later_receipt is not None and later_process is not None
            else None
        ),
        "clearance": (
            {
                "receipt_id": clearance_receipt.receipt_id,
                "execution_id": clearance_receipt.execution_id,
                "request_id": clearance_process.request_id,
                "host_completion_event_sequence": clearance_process.event_sequence,
                "receipt_adapter_projection_sha256": clearance_receipt.projection_sha256,
                "host_completion_adapter_projection_sha256": clearance_process.projection_sha256,
                "revision": clearance_receipt.revision,
                "status": clearance_receipt.status,
                "outcome": clearance_receipt.outcome,
                "passed_count": dict(clearance_receipt.counts).get("passed"),
                "revision_relation": disposition.removeprefix("CLEARED_"),
            }
            if clearance_receipt is not None and clearance_process is not None
            else None
        ),
        "projection_digest_note": "adapter_projection_sha256 values bind these DTO projections only; they are not raw host-journal hashes and do not authenticate host origin.",
        "interpretation": "Canonical public outcomes and the exact public quote are preserved; this pin does not infer cause, verify current source, or execute a check.",
    }


def _prepared_injection_payload(value: PreparedInjection) -> dict[str, object]:
    return {
        "record_id": value.record_id,
        "event_sequence": value.event_sequence,
        "run_id": value.run_id,
        "attempt_id": value.attempt_id,
        "request_id": value.request_id,
        "candidate_revision": value.candidate_revision,
        "context_sha256": value.context_sha256,
    }


def _boundary_payload(value: SliceBoundary) -> dict[str, object]:
    return {"boundary_id": value.boundary_id, "kind": value.kind, "event_sequence": value.event_sequence}


def _request_origin_payload(value: RequestOriginEvent) -> dict[str, object]:
    return {
        "record_id": value.record_id,
        "event_sequence": value.event_sequence,
        "run_id": value.run_id,
        "attempt_id": value.attempt_id,
        "request_id": value.request_id,
    }


def _receipt_payload(value: CanonicalCheckReceipt) -> dict[str, object]:
    return {
        "receipt_id": value.receipt_id,
        "execution_id": value.execution_id,
        "schema": value.schema,
        "selector": value.selector,
        "selector_scope": value.selector_scope,
        "receipt_sequence": value.receipt_sequence,
        "status": value.status,
        "outcome": value.outcome,
        "failure_phase": value.failure_phase,
        "revision": value.revision,
        "execution_complete": value.execution_complete,
        "termination_reason": value.termination_reason,
        "capture_complete": value.capture_complete,
        "capture_status": value.capture_status,
        "counts": [list(row) for row in value.counts],
        "collected_tests": value.collected_tests,
        "assertion_context_preview": value.assertion_context_preview,
        "assertion_context_omitted_bytes": value.assertion_context_omitted_bytes,
    }


def _receipt_event_payload(value: CheckReceiptJournalEvent) -> dict[str, object]:
    return {
        "record_id": value.record_id,
        "event_sequence": value.event_sequence,
        "event_type": value.event_type,
        "producer": value.producer,
        "dispatch": value.dispatch,
        "execution_id": value.execution_id,
        "receipt_id": value.receipt_id,
        "selector": value.selector,
        "selector_scope": value.selector_scope,
        "status": value.status,
        "outcome": value.outcome,
        "failure_phase": value.failure_phase,
        "revision": value.revision,
        "execution_complete": value.execution_complete,
        "termination_reason": value.termination_reason,
        "counts": [list(row) for row in value.counts],
    }


def _subprocess_payload(value: SubprocessCompletionEvent) -> dict[str, object]:
    return {
        "record_id": value.record_id,
        "event_sequence": value.event_sequence,
        "event_type": value.event_type,
        "run_id": value.run_id,
        "attempt_id": value.attempt_id,
        "request_id": value.request_id,
        "execution_ids": value.execution_ids,
        "receipt_ids": value.receipt_ids,
        "selectors": value.selectors,
        "selector_scope": value.selector_scope,
        "canonical_revision_from_binding": value.canonical_revision_from_binding,
        "binding_stable": value.binding_stable,
    }


def _history_payload(value: CompleteHistoryExport) -> dict[str, object]:
    return {
        "export_id": value.export_id,
        "run_id": value.run_id,
        "attempt_id": value.attempt_id,
        "start_boundary": _boundary_payload(value.start_boundary),
        "end_boundary": _boundary_payload(value.end_boundary),
        "complete": value.complete,
        "omitted_event_count": value.omitted_event_count,
        "pending_execution_count": value.pending_execution_count,
        "conflicting_record_count": value.conflicting_record_count,
        "request_origins": [
            {**_request_origin_payload(row), "projection_sha256": row.projection_sha256}
            for row in value.request_origins
        ],
        "views": [
            {
                "receipt": {**_receipt_payload(view.receipt), "projection_sha256": view.receipt.projection_sha256},
                "receipt_event": {**_receipt_event_payload(view.receipt_event), "projection_sha256": view.receipt_event.projection_sha256},
                "subprocess_event": {**_subprocess_payload(view.subprocess_event), "projection_sha256": view.subprocess_event.projection_sha256},
                "subprocess_item_index": view.subprocess_item_index,
            }
            for view in value.views
        ],
    }


def _projection_sha256(kind: str, payload: dict[str, object]) -> str:
    encoded = _canonical_json_bytes({"projection_kind": kind, "payload": payload}, max_bytes=MAX_HISTORY_JSON_BYTES)
    return _sha256(encoded)


def _digest_matches(kind: str, payload: dict[str, object], observed: str) -> bool:
    if not _digest_field(observed):
        return False
    try:
        return _projection_sha256(kind, payload) == observed
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return False


def _canonical_json_bytes(value: object, *, max_bytes: int) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("json_payload_too_large")
    return encoded


def _sha256(encoded: bytes) -> str:
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _valid_event_sequence(value: object, history: CompleteHistoryExport, injection_sequence: int) -> bool:
    return (
        _valid_event_sequence_number(value)
        and history.start_boundary.event_sequence < value <= history.end_boundary.event_sequence
        and value < injection_sequence
    )


def _valid_event_sequence_number(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_SEQUENCE


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_SEQUENCE


def _valid_selector_scope(value: object) -> bool:
    if type(value) is not tuple or not 1 <= len(value) <= MAX_HISTORY_VIEWS:
        return False
    if any(not _bounded_text(item, MAX_SELECTOR_UTF8_BYTES) for item in value):
        return False
    if len(set(value)) != len(value):
        return False
    joined_bytes = sum(len(item.encode("utf-8")) for item in value) + 2 * (len(value) - 1)
    return joined_bytes <= MAX_SCOPE_UTF8_BYTES


def _valid_identifier(value: object) -> bool:
    if not _bounded_text(value, MAX_ID_UTF8_BYTES):
        return False
    assert isinstance(value, str)
    return value == value.strip() and value.casefold() not in _UNKNOWN_IDENTIFIERS


def _valid_revision(value: object) -> bool:
    if not _bounded_text(value, MAX_ID_UTF8_BYTES):
        return False
    assert isinstance(value, str)
    return value == value.strip() and value.casefold() not in _UNKNOWN_IDENTIFIERS


def _digest_field(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _bounded_text(value: object, max_bytes: int, *, allow_empty: bool = False) -> bool:
    if type(value) is not str or (not value and not allow_empty) or len(value) > max_bytes:
        return False
    try:
        return len(value.encode("utf-8")) <= max_bytes
    except UnicodeError:
        return False


def _noop(reason: str) -> FailurePinPlan:
    return FailurePinPlan(status="NOOP", reason=reason)
