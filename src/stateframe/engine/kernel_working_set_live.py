"""Atomic allocator preparation and read-only proof of its exact provider wire."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace

from kernel_reason_log import emit_refusal
from kernel_request_seal import RequestSealError
from kernel_working_set_cache import transaction


def assert_certified_body(body, prepared) -> None:
    """Reject even equal-length validator rewrites before publishing authority."""
    if json.dumps(body).encode() != prepared.wire:
        emit_refusal("working_set_live", what="certified working-set body refused",
                     why="validator changed certified body", journal=False)
        raise RequestSealError("validator changed certified working-set body")
    from kernel_working_set_proof import verify_certificate
    verify_certificate(prepared.wire, prepared.rendered.certificate)


def certified_visibility(prepared):
    """Receipt-scoped visibility, without path forgiveness or a repair pass."""
    from kernel_working_set_proof import verify_certificate
    certificate = prepared.rendered.certificate
    verify_certificate(prepared.wire, certificate)
    selected = dict(certificate.assignment)
    ranges = [{"path": slot.identity.path, "revision": slot.identity.revision,
               "start_line": slot.identity.first_line, "end_line": slot.identity.last_line}
              for slot in certificate.snapshot.sources if selected[slot.key] == "FULL"]
    check = {"candidates": ranges, "missing_ranges": [], "pins_added": 0}
    receipt = {"result": True, "missing_ranges": [], "delivered_sources": ranges,
               "body_sha256": "sha256:" + certificate.body_sha256}
    return check, receipt


class ReadAdmissionUnbounded(RequestSealError):
    """A pending producer has no host-enforced result bound; end admission."""


class ReadAdmissionUnavailable(RequestSealError):
    """Read admission could not freeze its allocator snapshot; ``reason`` is typed.

    Callers refuse fresh exposure for this batch with this reason. A new
    request retries the freeze; refusal cursors preserve the requested ranges.
    """
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__("read admission lacks a complete allocator snapshot: " + reason)


def _snapshot_failure_reason(failure):
    from kernel_working_set_stop import ReasonCode
    # A failed freeze has no measured floor. Unknown or misplaced solver
    # reasons cannot grant floor authority or bypass the non-floor contract.
    reasons = {reason.value: reason for reason in (
        ReasonCode.ACTIVE_OBLIGATION_LIMIT, ReasonCode.SOURCE_SNAPSHOT_UNSTABLE,
        ReasonCode.CAPTURE_BLOB_UNAVAILABLE, ReasonCode.PREPARATION_CONTRACT_FAILURE)}
    return reasons.get(getattr(failure, "reason", None), ReasonCode.PREPARATION_CONTRACT_FAILURE)


def _recovery_conditions():
    from kernel_working_set_stop import ReasonCode
    # All refusals stop durably. Search exhaustion is not infeasibility and
    # has neither a mandatory proof nor a fabricated minimum candidate.
    return {
        ReasonCode.BYTE_POLICY_DOMAIN_EXHAUSTED: "retry with a wider byte-policy search or more capacity",
        ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED: "retry token admission with more search budget or capacity",
        ReasonCode.RECOVERY_ROUTE_UNAVAILABLE: "restore an authorized recovery route and retry",
        ReasonCode.ACTIVE_OBLIGATION_LIMIT: "reduce active obligations or increase the active limit",
        ReasonCode.SOURCE_SNAPSHOT_UNSTABLE: "stabilize source revisions and retry preparation",
        ReasonCode.CAPTURE_BLOB_UNAVAILABLE: "restore the captured source blobs and retry preparation",
        ReasonCode.PREPARATION_CONTRACT_FAILURE: "repair the preparation contract and retry",
    }


def allocator_context_floor(snapshot, result, *, size=None):
    """Return the normal small-context terminal only for a proven allocator floor.

    The allocator's measured complete minimum is already represented by
    ``floor_bytes``. Convert that wire floor with the same conservative byte
    calibration used by the legacy context-floor path. Other allocator
    refusals, explicit prompt budgets, and normal-sized slots remain untouched.
    """
    from kernel_working_set_stop import ReasonCode
    if getattr(result, "reason", None) is not ReasonCode.FLOOR_INFEASIBLE:
        return None
    from kernel_endpoint_size import ENDPOINT_SIZE, DEFAULT_CONTEXT_TOKENS
    endpoint_size = ENDPOINT_SIZE if size is None else size
    if (snapshot.context_limit is None
            or snapshot.context_limit != endpoint_size.context_tokens
            or endpoint_size.context_tokens >= DEFAULT_CONTEXT_TOKENS):
        return None
    from kernel_context_floor import context_floor
    return context_floor({"request_bytes": result.floor_bytes,
                          "measure_method": "calibrated_bytes"}, size=endpoint_size)


def _snapshot_failed(runtime, failure, *, request_id: str):
    """Snapshot construction failed: a durable, typed, non-floor suspension.

    No snapshot exists, so there is no measured floor, no proof and no trials.
    The digest binds the typed diagnostic only; nothing fabricates infeasibility.
    """
    from kernel_read_receipts import _blob
    from kernel_working_set_state import state_for
    from kernel_working_set_stop import Suspension, suspend
    reason = _snapshot_failure_reason(failure)
    error_type = getattr(failure, "error_type", "missing")
    state = state_for(runtime)
    diagnostic = {"request_id": request_id, "reason": reason.value, "error_type": error_type}
    suspend(runtime, Suspension(reason, request_id, state.next_due_ordinal,
        hashlib.sha256(json.dumps(diagnostic, sort_keys=True).encode()).hexdigest(),
        hashlib.sha256(json.dumps(state.checkpoint(), sort_keys=True).encode()).hexdigest(),
        (), None, None, _blob(runtime, "[]"), _recovery_conditions()[reason],
        detail=f"snapshot construction failed: {error_type}: {reason.value}"))


def _blocked(runtime, snapshot, result):
    capacity_stop = allocator_context_floor(snapshot, result)
    if capacity_stop is not None:
        from kernel_frame_allocation_capture import FrameFit5ImpossibleMinimum
        raise FrameFit5ImpossibleMinimum({
            "request_bytes": result.floor_bytes,
            "prompt_tokens": None,
            "measure_method": "calibrated_bytes",
            "allocator_floor": True,
        })
    from kernel_read_receipts import _blob
    from kernel_working_set_state import state_for
    from kernel_working_set_stop import MANDATORY_ENVELOPE_REASONS, Suspension, suspend
    recovery = _recovery_conditions()
    state = state_for(runtime)
    trials = _blob(runtime, json.dumps([asdict(t) for t in result.trials]))
    minimum = (_blob(runtime, result.proof.mandatory_wire.decode())
               if result.proof is not None else None)
    floor = result.reason in MANDATORY_ENVELOPE_REASONS
    limits = (("serialized_bytes",) if floor else
              result.trials[-1].violated_limits if result.trials else ())
    suspend(runtime, Suspension(result.reason, snapshot.request_id, snapshot.due_ordinal,
        snapshot.snapshot_digest, hashlib.sha256(json.dumps(state.checkpoint(),
            sort_keys=True).encode()).hexdigest(), limits, result.proof,
        minimum, trials, ("reduce the mandatory envelope or increase capacity" if floor
                         else recovery[result.reason]), floor_bytes=result.floor_bytes,
        cap=result.cap, section_breakdown=result.section_breakdown, detail=result.detail))


def live_measure(meter):
    """The one token authority: preparation and read admission measure alike."""
    from kernel_context import token_estimate
    from kernel_frame_allocation_capture import calibrated_estimate, _remember_exact_pair
    from kernel_working_set_prepare import Measurement

    def measure(wire):
        value = json.loads(wire)
        tokens, method = meter.measure(value)
        if method == "calibrated_bytes":
            tokens = calibrated_estimate(value, meter)
        elif method == "tokenize":
            _remember_exact_pair(value, meter, tokens)
        return Measurement(tokens, token_estimate(value), method,
                           meter.tokenizer_identity(value))

    return measure


def read_admission_measure(runtime):
    """Admission's measurer, built like preparation's; None keeps the frozen offline ratio."""
    from kernel_request_budget import _request_token_meter

    capture = getattr(runtime, "_active_capture", None)
    try:
        meter = _request_token_meter(runtime, capture)
    except Exception:
        return None
    return live_measure(meter) if meter is not None else None


# Admitted pages replace a pending read's row; the placeholder only reserves protocol bytes.
PENDING_READ_PLACEHOLDER = " " * 192


def pending_result_reserve(call):
    """Wire bytes a not-yet-executed non-read call may still add to the next frame.

    Returns None when no host bound exists for that result.  Only LEAN mode
    bounds tool replies at the append boundary (kernel_lean.BoundedToolMessages),
    so the enforceable ceiling is the largest bounded reply, escaped on the wire.
    """
    from kernel_lean import MAX_TOOL_RESULT_CHARS, bound_tool_result, enabled as lean_enabled

    if not lean_enabled():
        return None
    # Two-byte UTF-8 characters satisfy the byte bound and escape at six wire bytes each.
    return bound_tool_result("é" * MAX_TOOL_RESULT_CHARS)


def _source_plan_with_deferred_omissions(plan):
    """Add bounded read evidence for deferred paths with no selected units."""
    if not isinstance(plan, Mapping):
        return plan
    units = plan.get("units")
    deferred = plan.get("deferred_paths")
    omissions = plan.get("omissions")
    if not (
        isinstance(units, list)
        and isinstance(deferred, Sequence) and not isinstance(deferred, (str, bytes, bytearray))
        and isinstance(omissions, Sequence) and not isinstance(omissions, (str, bytes, bytearray))
    ):
        return plan

    from kernel_request_source_window import _normalized_deferred_paths

    selected = {unit["path"] for unit in units if isinstance(unit, Mapping)
                and isinstance(unit.get("path"), str)}
    unselected = set(_normalized_deferred_paths(deferred)) - selected
    if not unselected:
        return plan
    relevant = [row for row in omissions if isinstance(row, Mapping)
                and str(row.get("path", "")).replace("\\", "/").strip() in unselected]
    if not relevant:
        return plan

    from kernel_request_budget import _materialize_bounded_omission_units

    # Only the detached units list changes. Keep all original omission rows for
    # the final verifier and spend the recovery cap solely on unselected paths.
    recovered = units.copy()
    _materialize_bounded_omission_units(dict(plan, units=recovered, omissions=relevant))
    return dict(plan, units=recovered) if len(recovered) > len(units) else plan


def prepare_live(capture, runtime, body, *, dialect, reasoning_mode):
    """Prepare a certified wire, retrying plain if its cache-layout proof fails."""
    from kernel_frame_allocation_capture import FrozenRequestCounter
    from kernel_frame_optional_allocation import snapshot_optional_inputs
    from kernel_frame_source_snapshot import frame_source_snapshot
    from kernel_request_budget import _request_token_meter
    from kernel_request_candidate import candidate_source_plan
    from kernel_request_source_window import verify_final_source_delivery, source_delivery_satisfied
    from kernel_stateframe_framefit5 import _state_payload, _workspace_root
    from kernel_working_set_prepare import Blocked, prepare_candidate
    from kernel_working_set_shadow import freeze_shadow, SnapshotFailure

    plan = _source_plan_with_deferred_omissions(copy.deepcopy(candidate_source_plan(capture, runtime)))
    runtime._request_source_closure = plan
    snapshot = freeze_shadow(body, runtime, request_id=capture.request_id,
                             dialect=dialect, reasoning_mode=reasoning_mode)
    if isinstance(snapshot, SnapshotFailure) or snapshot is None:
        # Persist an orderly, resumable stop before publishing any request.
        _snapshot_failed(runtime, snapshot, request_id=capture.request_id)
        emit_refusal("working_set_live", what="working-set snapshot refused",
                     why="snapshot unavailable before publish", journal=False)
        raise RequestSealError("working-set snapshot: " + getattr(snapshot, "reason", "missing"))
    ceiling = getattr(runtime, "frame_ceiling", None)
    from kernel_cache_layout_candidate import final_layout_mode
    snapshot = replace(snapshot, frame_limit=ceiling if type(ceiling) is int and ceiling > 0 else None,
                       layouts=("plain", "final_cache") if final_layout_mode() == "on" else ("plain",))
    meter = _request_token_meter(runtime, capture)
    limits = getattr(capture, "_capacity_retry_limits", None)
    if limits is not None:
        snapshot = replace(snapshot, byte_cap=min(snapshot.byte_cap, limits.max_request_bytes),
                           prompt_limit=min(snapshot.prompt_limit, limits.max_prompt_tokens))

    measure = live_measure(meter)

    def qualify(prepared):
        """Bind the closure proof to this certified body and its assignment."""
        body = json.loads(prepared.wire)
        assert_certified_body(body, prepared)
        certificate = prepared.rendered.certificate
        capture._working_set_certificate = certificate
        selected = dict(certificate.assignment)
        closure = copy.deepcopy(plan)
        sources = frame_source_snapshot(runtime, closure)
        # CARRY-NOTE: bind closure proof to this certified assignment, never a legacy selector.
        from kernel_working_set_facts import _covered
        full = [slot for slot in certificate.snapshot.sources if selected[slot.key] == "FULL"]
        full_capture_ids = {slot.key for slot in full if slot.capture_ids}
        verified_capture_witnesses = tuple(w for w in certificate.witnesses
                                           if w.slot_id in full_capture_ids)
        # Every plan unit not covered FULL by whole lines is deferred. The
        # renderer declares the same set via uncovered_plan_unit_paths, so
        # late-deferral reconciliation finds each notice on re-verification.
        deferred = set(closure.get("deferred_paths", ()))
        revisions = dict(certificate.snapshot.current_revisions)
        for unit in closure.get("units", ()):
            path = unit["path"]
            if not _covered(full, path, revisions.get(path), unit["start_line"], unit["end_line"]):
                deferred.add(path)
        closure["deferred_paths"] = sorted(deferred)
        closure["source_contents"] = sources
        capture._request_source_closure_override = closure
        receipt = verify_final_source_delivery(closure, body["messages"], sources=sources,
                                               payload=prepared.wire,
                                               verified_capture_witnesses=verified_capture_witnesses)
        return body, closure, sources, receipt

    def proof_failures(receipt):
        return sorted({str(row.get("kind", "unknown")) for row in
                       receipt.get("proof_failures", ()) if isinstance(row, dict)})

    prepared = prepare_candidate(snapshot, measure=measure)
    if isinstance(prepared, Blocked):
        _blocked(runtime, snapshot, prepared)
    body, closure, sources, receipt = qualify(prepared)
    retried_plain = False
    if (not source_delivery_satisfied(receipt)
            and prepared.rendered.certificate.layout == "final_cache"
            and "plain" in snapshot.layouts):
        # Match the legacy cache-layout fail-open without invoking its selectors.
        # Narrow the search so the failed cache layout cannot win again; the plain
        # wire needs its own budget measurement, certificate and closure proof.
        retried_plain = True
        capture.journal.emit("final_cache_layout_failed_open", request_id=capture.request_id,
            reason="closure_proof_failed", vital=True, status=str(receipt.get("status")),
            failures=proof_failures(receipt))
        snapshot = replace(snapshot, layouts=("plain",))
        prepared = prepare_candidate(snapshot, measure=measure)
        if isinstance(prepared, Blocked):
            _blocked(runtime, snapshot, prepared)
        body, closure, sources, receipt = qualify(prepared)
    if not source_delivery_satisfied(receipt):
        emit_refusal("working_set_live", what="working-set closure proof refused",
                     why="closure proof failed: " + str(receipt.get("status")), journal=False)
        raise RequestSealError(("cache_layout_refused: " if retried_plain else "") +
                               "certified working-set failed closure proof: " +
                               str(receipt.get("status")) + ":" + ",".join(proof_failures(receipt)))
    certificate = prepared.rendered.certificate
    capture.request_source_final_delivery = receipt
    from kernel_cache_layout_view import semantic_messages
    frame = next(_state_payload(m)[1]["framefit5"] for m in semantic_messages(body["messages"])
                 if str(m.get("content", "")).startswith("[StateFrame]\n"))
    optional = snapshot_optional_inputs(frame, _workspace_root(runtime))
    measured = prepared.measurement
    capture._frame_candidate_counter = FrozenRequestCounter.capture(
        body, meter, measured.prompt_tokens, measured.method, str(capture.request_id))
    capture._frame_candidate_fallback_fit = measured.method == "calibrated_bytes"
    capture._frame_candidate_inputs = {
        "source_plan": closure, "sources": sources, "receipt": copy.deepcopy(receipt),
        "optional_inputs": optional, "byte_ceiling": certificate.snapshot.byte_cap,
        "token_ceiling": certificate.snapshot.prompt_limit, "measurement_identity": measured.method,
    }
    capture.journal.emit("working_set_prepared", request_id=capture.request_id,
        body_sha256=certificate.body_sha256, assignment_digest=prepared.assignment_digest,
        proof_passed=True, floor=prepared.floor_bytes, total=prepared.total_bytes,
        cap=prepared.cap, fits_without_degrade=prepared.fits_without_degrade,
        measurement_method=measured.method, dispatch=False)
    return body, prepared


def read_admission_snapshot(runtime, visible_messages, *, byte_cap):
    """Freeze the same renderer inputs as preparation, before fresh capture.

    Owner-visible history is often missing schemas and the typed frame. Prefer
    the last sealed provider envelope in that case. Fresh batch pages are priced
    separately against its exact minimum completion, never raw payload lengths.
    """
    from kernel_cache_layout_candidate import final_layout_mode
    from kernel_working_set_shadow import SnapshotFailure, freeze_shadow

    body = getattr(runtime, "_read_admission_provider_body", None)
    capture = getattr(runtime, "_active_capture", None)
    qualified = getattr(capture, "qualified_request", None)
    if qualified is not None:
        body = qualified.body
    elif visible_messages and any(str(m.get("content", "")).startswith("[StateFrame]\n")
                                for m in visible_messages):
        body = {**(body or {}), "messages": list(visible_messages)}
    if not isinstance(body, dict) or not any(
            str(row.get("content", "")).startswith("[StateFrame]\n")
            for row in body.get("messages", ())):
        return None
    body = copy.deepcopy(body)
    if qualified is not None:
        # Qualification already owns the full fixed layout. Re-composition
        # from owner-visible history can discard mandatory menus/facts and
        # manufacture fresh allowance. Decode overlays losslessly before
        # replacing the exchange, while their physical references still bind.
        from kernel_working_set_snapshot_compat import snapshot_semantic_messages
        body["messages"] = snapshot_semantic_messages(body["messages"])
    elif capture is not None and getattr(capture, "owner_presentation", None) is not None:
        from kernel_request_candidate import detached_context, render_candidate
        candidate, context = detached_context(capture, runtime)
        body = render_candidate(candidate, context, body)
    # The next exchange replaces the previous one. Retained capture obligations
    # are independently hydrated by freeze_shadow and cannot disappear here.
    body["messages"] = [m for m in body["messages"] if m.get("role") not in {"assistant", "tool"}]
    batch = getattr(runtime, "native_batch", None)
    pending = list(getattr(batch, "pending", ()))
    inflight = getattr(batch, "inflight", None)
    if isinstance(inflight, dict):
        pending.insert(0, inflight)
    calls = [{"id": call["id"], "type": call.get("type", "function"),
              "function": copy.deepcopy(call["function"])}
             for call in pending if isinstance(call, dict)]
    if calls:
        # Refused reads still occupy protocol bytes. Reserve their cursor replies
        # before spending any fresh allowance; admitted pages replace these rows.
        # Every other pending call reserves its enforceable result bound: an
        # admitted page cannot be dropped once a later shell/check result lands.
        exchange = []
        for call in calls:
            name = call["function"]["name"]
            if name == "read":
                content = PENDING_READ_PLACEHOLDER
            else:
                content = pending_result_reserve(call)
                if content is None:
                    raise ReadAdmissionUnbounded(
                        f"read admission cannot bound the pending {name} result")
            # NativeBatch normalizes execution as individual assistant/tool
            # pairs; one grouped assistant would underprice repeated shells.
            exchange.extend([
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}])
        end = len(body["messages"]) - 1
        body["messages"][end:end] = exchange
    certificate = getattr(capture, "_working_set_certificate", None)
    authority = getattr(certificate, "snapshot", None) or getattr(
        runtime, "_read_admission_provider_authority", None)
    plan = getattr(runtime, "_request_source_closure", None)
    admission_plan = _source_plan_with_deferred_omissions(plan)
    snapshot_runtime = runtime
    if admission_plan is not plan:
        snapshot_runtime = copy.copy(runtime)
        snapshot_runtime._request_source_closure = admission_plan
    snapshot = freeze_shadow(body, snapshot_runtime, request_id="read-admission",
                             dialect=getattr(authority, "dialect", "llama.cpp"),
                             reasoning_mode=getattr(authority, "reasoning_mode", "on"))
    if snapshot is None or isinstance(snapshot, SnapshotFailure):
        # A known envelope that cannot be frozen is not permission to expose
        # unaccounted bytes. The caller closes this batch under the typed reason
        # rather than treating the previous envelope as a future wire bound.
        raise ReadAdmissionUnavailable(_snapshot_failure_reason(snapshot).value)
    limits = getattr(capture, "_capacity_retry_limits", None)
    ceiling = getattr(runtime, "frame_ceiling", None)
    # Multiple pending calls reserve separate assistant/tool shells, including
    # replies that canonical assembly may later fold out of the newest exchange.
    # The cache projector accepts only that canonical exchange. Price this
    # conservative reservation as plain, a layout final preparation also admits,
    # instead of letting an inapplicable cache projection destroy floor authority.
    layouts = (("plain", "final_cache") if final_layout_mode() == "on" and len(calls) <= 1
               else ("plain",))
    return replace(snapshot,
        byte_cap=min(byte_cap, limits.max_request_bytes) if limits is not None else byte_cap,
        prompt_limit=min(snapshot.prompt_limit, limits.max_prompt_tokens) if limits is not None
                     else snapshot.prompt_limit,
        frame_limit=ceiling if type(ceiling) is int and ceiling > 0 else None,
        layouts=layouts)


@transaction
def read_admission_floor(snapshot, pages=(), *, require_feasible=False, measure=None,
                         violations_out=None):
    """Render the allocator floor with every proposed fresh page FULL-only.

    ``measure`` is preparation's live meter (see ``live_measure``); without it
    feasibility falls back to the frozen offline ratio. ``violations_out``, when a
    list is given, receives the violated limit names for this floor; the return
    value is unchanged for every caller.
    """
    from kernel_working_set_prepare import (
        _minimum_completion,
        _offline_measure,
        _violations,
    )
    from kernel_working_set_proof import verify_certificate
    from kernel_working_set_render import propose_slots

    body = json.loads(snapshot.fixed_body)
    sources = list(snapshot.sources)
    protected = set(snapshot.protected_paths)
    protected_ranges = set(snapshot.protected_ranges)
    revisions = dict(snapshot.current_revisions)
    for slot, call_id in pages:
        sources.append(slot)
        identity = slot.identity
        if identity.path not in protected:
            protected_ranges.add((identity.path, identity.first_line, identity.last_line))
        # The proposed page is a positive current-revision observation. Include
        # its host fact even for paths absent from the previous source plan.
        revisions[slot.identity.path] = slot.identity.revision
        carrier = next((row for row in body["messages"]
                        if row.get("role") == "tool" and row.get("tool_call_id") == call_id), None)
        if carrier is not None:
            carrier["content"] = slot.page
    # Price all promised ranges together. Pending edits retain the path-level
    # protection already present on the snapshot.
    trial = replace(snapshot, fixed_body=json.dumps(body).encode(), sources=tuple(sources),
                    protected_ranges=tuple(sorted(protected_ranges)),
                    current_revisions=tuple(sorted(revisions.items())))
    _, floor = _minimum_completion(trial, tuple(propose_slots(trial)))
    verify_certificate(floor.wire, floor.certificate)
    if require_feasible or violations_out is not None:
        from kernel_working_set_prepare import Measurement
        raw = measure(floor.wire) if measure is not None else _offline_measure(trial, floor.wire)
        measured = raw if isinstance(raw, Measurement) else Measurement(*raw)
        violations = _violations(trial, floor, measured)
        if violations_out is not None:
            violations_out.extend(violations)
        if require_feasible and violations:
            return None
    return floor


def read_admission_slot(source, *, path, arguments, ordinal):
    """Create a detached proposal with capture-identical byte/line authority."""
    from kernel_read_receipts import (
        DecodedSpan,
        ReceiptIdentity,
        line_digests_with_numbers,
    )
    from kernel_working_set_snapshot import SourceSlot

    metadata = source.metadata
    effective = metadata["effective_range"]
    lossy = bool(metadata.get("decoded_with_replacement"))
    span = (DecodedSpan(effective["start_column"], effective["end_column"],
                        str(metadata["presentation_version"])) if lossy else None)
    identity = ReceiptIdentity(path, str(metadata["revision"]), effective["start_line"],
        effective["end_line"], None if lossy else metadata.get("byte_start"),
        None if lossy else metadata.get("byte_end"),
        hashlib.sha256(source.content.encode()).hexdigest(), span)
    digests, numbers = line_digests_with_numbers(source.content, identity.first_line)
    key = f"admission-{ordinal}"
    return SourceSlot(key, (key,), identity, source.content, source.text, digests, numbers,
                      json.dumps(dict(arguments)).encode(), ordinal, ordinal, "DUE",
                      payload_start=source.text.encode().find(source.content.encode()))
