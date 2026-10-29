"""Join native batch authority to immutable final wire and bound host source."""
from __future__ import annotations

import hashlib
import json

from kernel_continuation_state import ContinuationError
from kernel_native_prefix import one, require


def submitted_payload(evidence, submitted: dict) -> dict:
    """Resolve frozen retry bytes without crossing request/snapshot lineage."""
    payload_event = submitted
    payload_ref = submitted.get("payload_ref")
    if not isinstance(submitted.get("payload_utf8"), str) and isinstance(payload_ref, str):
        payload_event = evidence.event(payload_ref, "request_submitted")
    if payload_event is not submitted:
        require(payload_event.get("request_id") == submitted.get("request_id")
                and payload_event.get("snapshot_id") == submitted.get("snapshot_id")
                and payload_event["sequence"] < submitted["sequence"],
                "retry payload reference crosses request identity")
    wire = payload_event.get("payload_utf8")
    require(isinstance(wire, str), "missing submitted bytes")
    require("sha256:" + hashlib.sha256(wire.encode()).hexdigest()
            == submitted.get("payload_hash") == payload_event.get("payload_hash"),
            "submitted bytes changed")
    try:
        payload = json.loads(wire)
    except ValueError as error:
        raise ContinuationError("native prefix: invalid submitted bytes") from error
    require(isinstance(payload, dict) and payload == payload_event.get("payload"),
            "submitted payload mismatch")
    return payload


def validate_prepared_delivery(events: list[dict], prepared: dict,
                               submitted: dict, payload: dict) -> bool:
    """Drafts prove request identity; only the final attempt proves delivery."""
    require(prepared["sequence"] < submitted["sequence"]
            and prepared.get("snapshot_id") == submitted.get("snapshot_id")
            and prepared.get("overflow") is None, "prepared request identity differs")
    finals = [e for e in events if e["kind"] == "request_final_wire_prepared"
              and e.get("request_id") == submitted.get("request_id")]
    if not finals:
        # Old journals predate the explicit prewire/final-wire distinction.
        require(prepared.get("body_artifact") != "prewire_draft"
                and prepared.get("messages") == payload.get("messages"),
                "prepared delivery differs")
        return False
    final = one([e for e in finals
                 if e.get("transport_attempt") == submitted.get("transport_attempt")],
                "missing or duplicate final delivery")
    require(prepared["sequence"] < final["sequence"] < submitted["sequence"]
            and type(final.get("transport_attempt")) is int
            and type(submitted.get("transport_attempt")) is int
            and submitted["transport_attempt"] > 0
            and final.get("producer") == "kernel_measurement"
            and final.get("event_schema") == "stateframe-request-final-wire-prepared/1"
            and final.get("body_artifact") == "final_wire_pre_dispatch"
            and final.get("provider_request") is True and final.get("dispatch") is False
            and final.get("payload_sha256") == submitted.get("payload_hash")
            and final.get("messages") == payload.get("messages")
            and final.get("tools") == payload.get("tools"), "final delivery differs")
    return True


def _acquired_source(events: list[dict], path: str, revision: str, before: int) -> str:
    """Acquisition is reconstruction input, never a claim of model delivery."""
    for event in reversed(events):
        if (event["kind"] != "request_source_closure" or event["sequence"] >= before
                or event.get("producer") != "kernel_request_source_window"
                or event.get("dispatch") is not False):
            continue
        plan = event.get("closure")
        if not isinstance(plan, dict):
            continue
        revisions, sources = plan.get("revisions"), plan.get("source_contents")
        if not isinstance(revisions, dict) or not isinstance(sources, dict):
            continue
        source = sources.get(path)
        if (revisions.get(path) == revision and isinstance(source, str)
                and "sha256:" + hashlib.sha256(source.encode()).hexdigest() == revision):
            return source
    raise ContinuationError("native prefix: missing bound source acquisition")


def _anchor_delivered(messages: list[dict], path: str, source: str,
                      revision: str, anchor: object) -> bool:
    from kernel_request_source_window import (
        _model_window_segments, parse_model_source_window, verify_model_source_window,
    )

    if not isinstance(anchor, str) or not anchor:
        return False
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        metadata = parse_model_source_window(content)
        if (metadata is None or metadata.get("revision") != revision
                or not verify_model_source_window(content, path=path, source=source)):
            continue
        segments = _model_window_segments(content, metadata)
        declared = metadata.get("ranges", [])
        # The model-window contract is whole-line. Its display separators and
        # declared ranges must agree; a conflicting column/range claim is not
        # historical source authority even if the body happens to contain text.
        if any(original.get("start_line") != row.get("start_line")
               or original.get("end_line") != row.get("end_line")
               or "start_column" in original or "end_column" in original
               for original, (row, _text) in zip(declared, segments)):
            continue
        for row, text in segments:
            if not row["trailing_newline"] and text.endswith("\n"):
                text = text[:-1]
            if anchor in text:
                return True
    return False
