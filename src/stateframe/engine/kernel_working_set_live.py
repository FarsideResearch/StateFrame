"""Atomic allocator preparation and read-only proof of its exact provider wire."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json

from kernel_request_seal import RequestSealError
from kernel_working_set_cache import transaction


def assert_certified_body(body, prepared) -> None:
    """Reject even equal-length validator rewrites before publishing authority."""
    if json.dumps(body).encode() != prepared.wire:
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


def _blocked(runtime, snapshot, result):
    from kernel_read_receipts import _blob
    from kernel_working_set_state import state_for
    from kernel_working_set_stop import MANDATORY_ENVELOPE_REASONS, ReasonCode, Suspension, suspend
    # All refusals stop durably. Search exhaustion is not infeasibility and
    # has neither a mandatory proof nor a fabricated minimum candidate.
    recovery = {
        ReasonCode.BYTE_POLICY_DOMAIN_EXHAUSTED: "retry with a wider byte-policy search or more capacity",
        ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED: "retry token admission with more search budget or capacity",
        ReasonCode.RECOVERY_ROUTE_UNAVAILABLE: "restore an authorized recovery route and retry",
        ReasonCode.ACTIVE_OBLIGATION_LIMIT: "reduce active obligations or increase the active limit",
        ReasonCode.SOURCE_SNAPSHOT_UNSTABLE: "stabilize source revisions and retry preparation",
        ReasonCode.CAPTURE_BLOB_UNAVAILABLE: "restore the captured source blobs and retry preparation",
        ReasonCode.PREPARATION_CONTRACT_FAILURE: "repair the preparation contract and retry",
    }
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


def prepare_live(capture, runtime, body, *, dialect, reasoning_mode):
    """Prepare once; legacy fitters and cache selectors never enter this path."""
    from kernel_context import token_estimate
    from kernel_frame_allocation_capture import (
        FrozenRequestCounter, calibrated_estimate, _remember_exact_pair,
    )
    from kernel_frame_optional_allocation import snapshot_optional_inputs
    from kernel_frame_source_snapshot import frame_source_snapshot
    from kernel_request_budget import _request_token_meter
    from kernel_request_candidate import candidate_source_plan
    from kernel_request_source_window import verify_final_source_delivery, source_delivery_satisfied
    from kernel_stateframe_framefit5 import _state_payload, _workspace_root
    from kernel_working_set_prepare import Blocked, Measurement, prepare_candidate
    from kernel_working_set_shadow import freeze_shadow, SnapshotFailure

    plan = copy.deepcopy(candidate_source_plan(capture, runtime))
    runtime._request_source_closure = plan
    snapshot = freeze_shadow(body, runtime, request_id=capture.request_id,
                             dialect=dialect, reasoning_mode=reasoning_mode)
    if isinstance(snapshot, SnapshotFailure) or snapshot is None:
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

    def measure(wire):
        value = json.loads(wire)
        tokens, method = meter.measure(value)
        if method == "calibrated_bytes":
            tokens = calibrated_estimate(value, meter)
        elif method == "tokenize":
            _remember_exact_pair(value, meter, tokens)
        return Measurement(tokens, token_estimate(value), method,
                           meter.tokenizer_identity(value))

    prepared = prepare_candidate(snapshot, measure=measure)
    if isinstance(prepared, Blocked):
        _blocked(runtime, snapshot, prepared)
    body = json.loads(prepared.wire)
    assert_certified_body(body, prepared)
    certificate = prepared.rendered.certificate
    capture._working_set_certificate = certificate
    selected = dict(certificate.assignment)
    plan = copy.deepcopy(plan)
    sources = frame_source_snapshot(runtime, plan)
    # Bind closure proof to this assignment, never to a legacy path selector.
    from kernel_working_set_facts import _covered
    full = [slot for slot in certificate.snapshot.sources if selected[slot.key] == "FULL"]
    # CARRY-NOTE: every plan unit the assignment does not cover FULL is marked
    # deferred here even when no allocator slot (and so no deferred notice) is on
    # the wire. Mainline late-deferral reconciliation (reconcile_owner_source_delivery)
    # judges deferrals by notice presence; a re-verify on retry/resume may flag these.
    # Left as-is so the allocator stays authoritative under ON; fixer item.
    deferred = set(plan.get("deferred_paths", ()))
    revisions = dict(certificate.snapshot.current_revisions)
    for unit in plan.get("units", ()):
        path = unit["path"]
        if not _covered(full, path, revisions.get(path), unit["start_line"], unit["end_line"]):
            deferred.add(path)
    plan["deferred_paths"] = sorted(deferred)
    plan["source_contents"] = sources
    capture._request_source_closure_override = plan
    receipt = verify_final_source_delivery(plan, body["messages"], sources=sources,
                                           payload=prepared.wire)
    if not source_delivery_satisfied(receipt):
        failures = sorted({str(row.get("kind", "unknown")) for row in
                           receipt.get("proof_failures", ()) if isinstance(row, dict)})
        raise RequestSealError("certified working-set failed closure proof: " +
                               str(receipt.get("status")) + ":" + ",".join(failures))
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
        "source_plan": plan, "sources": sources, "receipt": copy.deepcopy(receipt),
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
    if visible_messages and any(str(m.get("content", "")).startswith("[StateFrame]\n")
                                for m in visible_messages):
        body = {**(body or {}), "messages": list(visible_messages)}
    if not isinstance(body, dict) or not any(
            str(row.get("content", "")).startswith("[StateFrame]\n")
            for row in body.get("messages", ())):
        return None
    body = copy.deepcopy(body)
    if capture is not None and getattr(capture, "owner_presentation", None) is not None:
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
        # Refused calls still occupy protocol bytes. Reserve their cursor replies
        # before spending any fresh allowance; admitted pages replace these rows.
        exchange = []
        for call in calls:
            # NativeBatch normalizes execution as individual assistant/tool
            # pairs; one grouped assistant would underprice repeated shells.
            exchange.extend([
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                 "content": " " * 192}])
        end = len(body["messages"]) - 1
        body["messages"][end:end] = exchange
    certificate = getattr(capture, "_working_set_certificate", None)
    authority = getattr(certificate, "snapshot", None) or getattr(
        runtime, "_read_admission_provider_authority", None)
    snapshot = freeze_shadow(body, runtime, request_id="read-admission",
                             dialect=getattr(authority, "dialect", "llama.cpp"),
                             reasoning_mode=getattr(authority, "reasoning_mode", "on"))
    if snapshot is None or isinstance(snapshot, SnapshotFailure):
        # A known envelope that cannot be frozen is not permission to expose
        # unaccounted bytes. The caller refuses with an executable cursor.
        raise RequestSealError("read admission lacks a complete allocator snapshot")
    limits = getattr(capture, "_capacity_retry_limits", None)
    ceiling = getattr(runtime, "frame_ceiling", None)
    return replace(snapshot,
        byte_cap=min(byte_cap, limits.max_request_bytes) if limits is not None else byte_cap,
        prompt_limit=min(snapshot.prompt_limit, limits.max_prompt_tokens) if limits is not None
                     else snapshot.prompt_limit,
        frame_limit=ceiling if type(ceiling) is int and ceiling > 0 else None,
        layouts=("plain", "final_cache") if final_layout_mode() == "on" else ("plain",))


@transaction
def read_admission_floor(snapshot, pages=(), *, require_feasible=False):
    """Render the allocator floor with every proposed fresh page FULL-only."""
    from kernel_working_set_prepare import (
        _minimum_completion,
        _offline_measure,
        _violations,
    )
    from kernel_working_set_proof import verify_certificate
    from kernel_working_set_render import propose_slots

    body = json.loads(snapshot.fixed_body)
    sources = list(snapshot.sources)
    for slot, call_id in pages:
        sources.append(slot)
        carrier = next((row for row in body["messages"]
                        if row.get("role") == "tool" and row.get("tool_call_id") == call_id), None)
        if carrier is not None:
            carrier["content"] = slot.page
    trial = replace(snapshot, fixed_body=json.dumps(body).encode(), sources=tuple(sources))
    _, floor = _minimum_completion(trial, tuple(propose_slots(trial)))
    verify_certificate(floor.wire, floor.certificate)
    if require_feasible and _violations(trial, floor, _offline_measure(trial, floor.wire)):
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
