"""Read-only request composition measured from final provider wire bytes."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping

EXPOSURE_TRUTH_ENV = "STATEFRAME_EXPOSURE_TRUTH"
EXPOSURE_COMPOSITION_SCHEMA = "stateframe-exposure-composition/1"
_ON_VALUES = frozenset({"on", "1", "true", "yes"})
_PART_NAMES = (
    "system", "task", "frame", "newest_turn", "prior_turns", "host_panels", "tools",
)


def exposure_truth_enabled() -> bool:
    """Return whether wire-composition telemetry is explicitly enabled."""
    return os.environ.get(EXPOSURE_TRUTH_ENV, "off").strip().lower() in _ON_VALUES


def _is_frame_message(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    return isinstance(content, str) and content.startswith("[StateFrame]")


def _is_host_panel_message(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    if not isinstance(content, str) or content.startswith("[StateFrame]"):
        return False
    from kernel_lean import advisory_kind, is_truth_content

    return advisory_kind(content) is not None or is_truth_content(content)


def _serialized_size(value: object) -> int:
    return len(json.dumps(value).encode("utf-8"))


def exposure_composition_from_wire(wire: bytes) -> dict[str, object]:
    """Measure request parts and history counts from the immutable serialized body."""
    body = json.loads(wire)
    if not isinstance(body, Mapping):
        raise ValueError("serialized provider request must be a JSON object")

    raw_messages = body.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    message_rows = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, Mapping)
    ]
    assistant_indices = [
        index for index, message in message_rows if message.get("role") == "assistant"
    ]
    newest_assistant_index = assistant_indices[-1] if assistant_indices else None
    newest_indices: set[int] = set()
    prior_indices: set[int] = set()
    prior_turns = 0
    if newest_assistant_index is not None:
        newest_indices.add(newest_assistant_index)
        prior_turns = sum(index < newest_assistant_index for index in assistant_indices)
        prior_indices = {
            index for index, message in message_rows
            if index < newest_assistant_index
            and message.get("role") in {"assistant", "tool"}
        }
        next_index = newest_assistant_index + 1
        while next_index < len(messages):
            message = messages[next_index]
            if not isinstance(message, Mapping) or message.get("role") != "tool":
                break
            newest_indices.add(next_index)
            next_index += 1
    else:
        prior_indices = {
            index for index, message in message_rows if message.get("role") == "tool"
        }

    frame_indices = {
        index for index, message in message_rows if _is_frame_message(message)
    }
    user_rows = [
        (index, message) for index, message in message_rows
        if message.get("role") == "user" and index not in frame_indices
    ]
    task_candidates = [
        index for index, message in user_rows if not _is_host_panel_message(message)
    ]
    task_index = task_candidates[0] if task_candidates else (
        user_rows[0][0] if user_rows else None
    )

    bytes_per_part = dict.fromkeys(_PART_NAMES, 0)
    for index, message in message_rows:
        size = _serialized_size(message)
        if index in frame_indices:
            part = "frame"
        elif message.get("role") in {"system", "developer"}:
            part = "system"
        elif index in newest_indices:
            part = "newest_turn"
        elif index in prior_indices:
            part = "prior_turns"
        elif message.get("role") == "user":
            part = "task" if index == task_index else "host_panels"
        else:
            continue
        bytes_per_part[part] += size
    if "tools" in body:
        bytes_per_part["tools"] = _serialized_size(body["tools"])

    return {
        "frame_block_present": bool(frame_indices),
        "prior_turn_messages": len(prior_indices),
        "prior_turns": prior_turns,
        "bytes_per_part": bytes_per_part,
    }


def emit_exposure_composition(
    journal: object,
    *,
    wire: bytes,
    request_id: str,
    step: int,
    submitted_event_id: str,
    transport_attempt: int,
) -> None:
    """Publish counts only; telemetry failures never gate provider submission."""
    if not exposure_truth_enabled():
        return
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        composition = exposure_composition_from_wire(wire)
        emit(
            "exposure_composition",
            event_schema=EXPOSURE_COMPOSITION_SCHEMA,
            producer="kernel_exposure_truth",
            request_id=request_id,
            step=step,
            submitted_event_id=submitted_event_id,
            transport_attempt=transport_attempt,
            body_sha256="sha256:" + hashlib.sha256(wire).hexdigest(),
            dispatch=False,
            **composition,
        )
    except Exception:
        # Observation must not change whether the already-qualified bytes are sent.
        return
