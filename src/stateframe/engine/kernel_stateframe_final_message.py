"""Shared volatile user-message composition and Stage 6 inventory projection."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence


INVENTORY_MODE_ENV = "STATEFRAME_PAGE_LOG_INVENTORY"
INVENTORY_EVENT_PREFIX = "stateframe_page_log_inventory"
INVENTORY_HEAD = "[Stage 6 workspace inventory]"
AUTHORITY_OVERLAY_HEADER = "[StateFrame authority]"
PAGE_LOG_FINAL_MESSAGE_NAME = "stateframe_page_log_final"
_VALID_MODES = frozenset({"off", "shadow", "on"})


class FinalMessageError(ValueError):
    """A volatile message cannot be composed without losing request meaning."""


def inventory_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(INVENTORY_MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def _emit(journal: object | None, kind: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, producer="kernel_stateframe_final_message", dispatch=False, **fields)
        except Exception:
            return


def prepare_inventory(
        body: dict[str, object], *, selected_mode: str | None = None,
        journal: object | None = None, request_id: object = None,
        step: object = None,
) -> dict[str, object]:
    """Move one exact Stage 6 system message to the current volatile user tail."""
    selected = inventory_mode() if selected_mode is None else selected_mode
    if selected not in _VALID_MODES:
        selected = "off"
    if selected == "off":
        return body
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(message, Mapping) for message in messages):
        _emit(journal, "stateframe_page_log_inventory_failed_open",
              request_id=request_id, step=step, mode=selected, reason="malformed_messages")
        return body
    matches = [
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and message["content"].startswith(INVENTORY_HEAD)
    ]
    if len(matches) > 1:
        _emit(journal, "stateframe_page_log_inventory_failed_open",
              request_id=request_id, step=step, mode=selected,
              reason="multiple_inventory_messages", count=len(matches))
        return body
    candidate = copy.deepcopy(body)
    moved = bool(matches)
    if matches:
        index, message = matches[0]
        content = message.get("content")
        if not isinstance(content, str):
            _emit(journal, "stateframe_page_log_inventory_failed_open",
                  request_id=request_id, step=step, mode=selected, reason="inventory_not_text")
            return body
        copied_messages = [copy.deepcopy(dict(item)) for item in messages]
        del copied_messages[index]
        copied_messages.append({"role": "user", "content": content})
        candidate["messages"] = copied_messages
    _emit(journal, "stateframe_page_log_inventory_prepared",
          request_id=request_id, step=step, mode=selected, inventory_found=moved, moved=moved)
    if selected == "shadow":
        try:
            original_bytes = json.dumps(
                body.get("messages"), ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")
            candidate_bytes = json.dumps(
                candidate.get("messages"), ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            _emit(journal, "stateframe_page_log_inventory_failed_open",
                  request_id=request_id, step=step, mode=selected,
                  reason="messages_not_json_serializable")
            return body
        if original_bytes != candidate_bytes:
            _emit(journal, "stateframe_page_log_inventory_shadow_mismatch",
                  request_id=request_id, step=step, field="messages",
                  original_sha256=hashlib.sha256(original_bytes).hexdigest(),
                  projected_sha256=hashlib.sha256(candidate_bytes).hexdigest())
        return body
    return candidate


def tail_start_after_stateframe(messages: Sequence[object]) -> int:
    """Locate volatile user messages after the typed base and contiguous deltas."""
    indexes = [
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith("[StateFrame]\n")
    ]
    if len(indexes) != 1:
        raise FinalMessageError("final-message composition requires exactly one StateFrame")
    cursor = indexes[0] + 1
    while cursor < len(messages):
        message = messages[cursor]
        if (not isinstance(message, Mapping)
                or message.get("role") != "user"
                or message.get("name") != "stateframe_frame_delta"
                or not isinstance(message.get("content"), str)
                or not message["content"].startswith("[StateFrame delta]\n")):
            break
        cursor += 1
    return cursor


def compose_final_user_message(
        body: dict[str, object], *, tail_start: int,
        additions: Sequence[str] = (), greenfact_text: str = "",
        final_message_name: str | None = None,
) -> dict[str, object]:
    """Merge the current volatile tail/additions and place one exact GREENFACT suffix last."""
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(message, Mapping) for message in messages):
        raise FinalMessageError("final-message composition requires a valid message array")
    if type(tail_start) is not int or not 0 <= tail_start <= len(messages):
        raise FinalMessageError("final-message tail boundary is invalid")
    contents: list[str] = []
    for message in messages[tail_start:]:
        if message.get("role") != "user" or set(message) - {"role", "content"}:
            raise FinalMessageError("volatile tail contains a non-user message or opaque metadata")
        content = message.get("content")
        if not isinstance(content, str):
            raise FinalMessageError("volatile tail contains non-text content")
        contents.append(content)
    extra = list(additions)
    if any(not isinstance(content, str) for content in extra):
        raise FinalMessageError("volatile additions must be text")
    if greenfact_text and any(greenfact_text in content for content in extra):
        raise FinalMessageError("GREENFACT cannot be duplicated by a volatile addition")
    green_occurrences = sum(content.count(greenfact_text) for content in contents) if greenfact_text else 0
    greenfact = ""
    if green_occurrences:
        suffix_indexes = [index for index, content in enumerate(contents) if content.endswith(greenfact_text)]
        if green_occurrences != 1 or len(suffix_indexes) != 1:
            raise FinalMessageError("GREENFACT must be a unique exact suffix of one current V message")
        index = suffix_indexes[0]
        contents[index] = contents[index][:-len(greenfact_text)]
        greenfact = greenfact_text
    final_content = "\n\n".join([*contents, *extra]) + greenfact
    output = copy.deepcopy(body)
    final_message: dict[str, object] = {"role": "user", "content": final_content}
    if final_message_name is not None:
        if final_message_name != PAGE_LOG_FINAL_MESSAGE_NAME:
            raise FinalMessageError("final-message name is not an owned protocol marker")
        final_message["name"] = final_message_name
    output["messages"] = [*copy.deepcopy(messages[:tail_start]), final_message]
    return output


__all__ = [
    "AUTHORITY_OVERLAY_HEADER", "FinalMessageError", "INVENTORY_EVENT_PREFIX",
    "INVENTORY_HEAD", "INVENTORY_MODE_ENV", "compose_final_user_message",
    "PAGE_LOG_FINAL_MESSAGE_NAME", "inventory_mode", "prepare_inventory",
    "tail_start_after_stateframe",
]
