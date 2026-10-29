"""Canonical StateFrame envelope construction shared by live and replay paths."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence


STATE_HEADER = "[StateFrame]"
_F4_SCHEMA = "stateframe-f4/1"
_FRAMEFIT5_MODE = "FRAME_ONLY"
_ENVELOPE_KEYS = frozenset({"schema", "revision", "rev", "frame_hygiene"})


def build_stateframe_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a copied payload with any flat F4 projection restored to its envelope."""
    result = copy.deepcopy(dict(payload))
    if "state" in result:
        return result

    if result.get("schema") != _F4_SCHEMA:
        return result

    envelope = {key: value for key, value in result.items() if key in _ENVELOPE_KEYS}
    envelope["state"] = {
        key: value for key, value in result.items() if key not in _ENVELOPE_KEYS
    }
    return envelope


def serialize_stateframe_payload(payload: Mapping[str, object]) -> str:
    """Serialize one StateFrame payload through the canonical envelope builder."""
    normalized = build_stateframe_payload(payload)
    return STATE_HEADER + "\n" + json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def normalize_stateframe_messages(
        messages: Sequence[Mapping[str, object]],
        *,
        framefit5_state: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Restore flat F4 envelopes and attach trusted typed state when a live F5 view lost it."""
    normalized_messages: list[dict[str, object]] = []
    prefix = STATE_HEADER + "\n"
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if not isinstance(content, str) or not content.startswith(prefix):
            normalized_messages.append(item)
            continue
        boundary = content.find("\n[", len(prefix))
        body = content[len(prefix):] if boundary < 0 else content[len(prefix):boundary]
        suffix = "" if boundary < 0 else content[boundary:]
        try:
            payload = json.loads(body)
        except (TypeError, ValueError, RecursionError):
            normalized_messages.append(item)
            continue
        if not isinstance(payload, dict):
            normalized_messages.append(item)
            continue
        normalized = build_stateframe_payload(payload)
        state = normalized.get("state")
        current_framefit5 = state.get("framefit5") if isinstance(state, dict) else None
        if (
                normalized.get("schema") == _F4_SCHEMA
                and isinstance(state, dict)
                and isinstance(framefit5_state, Mapping)
                and framefit5_state.get("mode") == _FRAMEFIT5_MODE
                and not (
                    isinstance(current_framefit5, Mapping)
                    and current_framefit5.get("mode") == _FRAMEFIT5_MODE
                )
        ):
            state = copy.deepcopy(state)
            state["framefit5"] = copy.deepcopy(dict(framefit5_state))
            normalized["state"] = state
        if normalized != payload:
            item["content"] = serialize_stateframe_payload(normalized) + suffix
        normalized_messages.append(item)
    return normalized_messages
