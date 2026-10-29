"""Optional source-observation refresh from an actual model request, never proof."""
from __future__ import annotations

import hashlib
import os

from kernel_continuation_state import ContinuationError
from kernel_measurement import digest
from kernel_paged_source import PAGED_SOURCE_PREFIX, parse_descriptor, validate_descriptor
from kernel_read_fixes import source_header
from kernel_source_presentation import strip_callsite_census, working_source_block


def enabled() -> bool:
    value = os.environ.get("KERNEL_CONTINUATION_SOURCE_REFRESH", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_CONTINUATION_SOURCE_REFRESH must be 0 or 1")
    return value == "1"


def source_metadata(messages: list[dict], *, source: str, path: str, binding: dict,
                    intention_id: str, step: int) -> dict | None:
    """Recognize a complete host-extracted file, not a label or arbitrary excerpt."""
    revision = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    revisions = {**binding.get("product_files", {}), **binding.get("harness_files", {})}
    if not source or revisions.get(path) != revision:
        return None
    paged_matches = [
        (i, m) for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(PAGED_SOURCE_PREFIX)
    ]
    if len(paged_matches) == 1:
        index, message = paged_matches[0]
        try:
            validate_descriptor(message["content"], path=path, source=source,
                                revision=revision)
        except (TypeError, ValueError):
            return None
        return {"schema": "continuation-source-delivery/1", "binding": binding,
                "intention_id": intention_id, "step": step, "path": path,
                "revision": revision, "message_index": index,
                "message_hash": digest(message), "paged": True,
                "content_offset": 0, "content_chars": len(message["content"])}
    forms = {f"[Working source: {path}]\n{source}", f"{source_header(path)}\n{source}",
             working_source_block(path, source, exists=True)}
    matches = [(i, m) for i, m in enumerate(messages)
               if m.get("role") == "user" and isinstance(m.get("content"), str)
               and strip_callsite_census(m["content"]) in forms]
    if len(matches) != 1:
        return None
    index, message = matches[0]
    return {"schema": "continuation-source-delivery/1", "binding": binding,
            "intention_id": intention_id, "step": step, "path": path,
            "revision": revision, "message_index": index, "message_hash": digest(message),
            "content_offset": len(message["content"]) - len(source),
            "content_chars": len(source)}


def validate_delivery(evidence, data: dict) -> None:
    """Join prepared bytes to the response's actual submitted request on replay too."""
    prepared = evidence.event(data.get("source_prepared_ref"), "request_prepared")
    received = evidence.event(data.get("receipt_ref"), "response_received")
    submitted = evidence.event(received.get("transport_attempt_id"), "request_submitted")
    meta = prepared.get("continuation_source")
    current = data.get("current_binding")
    if (not isinstance(data.get("request_id"), str) or not data["request_id"]
            or not isinstance(data.get("snapshot_id"), str) or not data["snapshot_id"]
            or not isinstance(meta, dict) or meta.get("schema") != "continuation-source-delivery/1"
            or prepared.get("overflow") is not None
            or meta.get("binding") != current
            or meta.get("intention_id") != data.get("intention_id")
            or not prepared["sequence"] < submitted["sequence"] < received["sequence"]
            or any(e.get("request_id") != data.get("request_id")
                   for e in (prepared, submitted, received))
            or any(e.get("snapshot_id") != data.get("snapshot_id")
                   for e in (prepared, submitted))
            or any(e.get("step") != data.get("step") for e in (meta, submitted, received))):
        raise ContinuationError("source refresh lacks current request/response lineage")
    from kernel_native_prefix_delivery import submitted_payload, validate_prepared_delivery
    payload = submitted_payload(evidence, submitted)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ContinuationError("submitted source messages are missing")
    final_delivery = validate_prepared_delivery(
        evidence.final_delivery_events(submitted),
        prepared, submitted, payload)
    if final_delivery:
        # Draft offsets are not final-wire coordinates. First prove the original
        # source descriptor, then locate that exact bound source in final bytes.
        original = _validate_source_message(prepared.get("messages"), meta, current)
        if meta.get("paged") is True:
            matches = [i for i, message in enumerate(messages) if digest(message) == meta["message_hash"]]
            if len(matches) != 1:
                raise ContinuationError("final paged source delivery is missing or ambiguous")
            meta = {**meta, "message_index": matches[0]}
        else:
            meta = source_metadata(messages, source=original, path=meta["path"], binding=current,
                                   intention_id=data["intention_id"], step=data["step"])
            if meta is None:
                raise ContinuationError("final source delivery is stale, incomplete or missing")
    _validate_source_message(messages, meta, current)


def _validate_source_message(messages: object, meta: dict, current: dict) -> str:
    if not isinstance(messages, list):
        raise ContinuationError("source delivery messages missing")
    index = meta.get("message_index")
    if type(index) is not int or index < 0 or index >= len(messages):
        raise ContinuationError("source delivery message missing")
    message = messages[index]
    content = message.get("content") if isinstance(message, dict) else None
    start, length = meta.get("content_offset"), meta.get("content_chars")
    if (not isinstance(content, str) or message.get("role") != "user"
            or digest(message) != meta.get("message_hash")
            or type(start) is not int or start < 0 or type(length) is not int or length < 1
            or start + length != len(content)):
        raise ContinuationError("source delivery is empty, altered or truncated")
    if meta.get("paged") is True:
        descriptor = parse_descriptor(content)
        if (descriptor is None or descriptor.get("path") != meta.get("path")
                or descriptor.get("revision") != meta.get("revision")
                or descriptor.get("complete") is not True
                or descriptor.get("content_in_frame") is not False):
            raise ContinuationError("paged source descriptor is stale or incomplete")
        return content
    source = content[start:]
    revision = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    revisions = {**current.get("product_files", {}), **current.get("harness_files", {})}
    path = meta.get("path")
    if (not isinstance(path, str) or revisions.get(path) != revision
            or meta.get("revision") != revision
            or strip_callsite_census(content) not in {f"[Working source: {path}]\n{source}",
                                                      f"{source_header(path)}\n{source}",
                                                      working_source_block(path, source, exists=True)}):
        raise ContinuationError("source delivery revision is stale or incomplete")
    return source
