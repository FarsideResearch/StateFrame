"""Normalize unsafe Unicode before model request preparation and validate wire JSON."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re


_REPLACEMENT = "\ufffd"
_MAX_PATH_CHARS = 192
_UNSAFE_TEXT_RE = re.compile(r"[\x00\ud800-\udfff]")
_SAFE_PATH_FIELDS = frozenset({
    "messages", "model", "role", "content", "name", "tool_calls",
    "tool_call_id", "function", "arguments", "text", "type", "tools",
})


@dataclass(frozen=True, slots=True)
class WireStringReplacement:
    message_index: int | None
    kind: str
    count: int


@dataclass(frozen=True, slots=True)
class WireStringViolation:
    path: str
    kind: str


def _sanitize_string(
        value: str, message_index: int | None
) -> tuple[str, tuple[WireStringReplacement, ...]]:
    if _UNSAFE_TEXT_RE.search(value) is None:
        return value, ()

    nul_count = 0
    surrogate_count = 0
    characters: list[str] = []
    for char in value:
        codepoint = ord(char)
        if codepoint == 0:
            nul_count += 1
            characters.append(_REPLACEMENT)
        elif 0xD800 <= codepoint <= 0xDFFF:
            surrogate_count += 1
            characters.append(_REPLACEMENT)
        else:
            characters.append(char)

    replacements = []
    if surrogate_count:
        replacements.append(WireStringReplacement(
            message_index, "lone_surrogate", surrogate_count))
    if nul_count:
        replacements.append(WireStringReplacement(message_index, "nul", nul_count))
    return "".join(characters), tuple(replacements)


def _sanitize_value(
        value: object, message_index: int | None
) -> tuple[object, tuple[WireStringReplacement, ...]]:
    if isinstance(value, str):
        return _sanitize_string(value, message_index)
    if isinstance(value, Mapping):
        cleaned: dict[object, object] = {}
        replacements: list[WireStringReplacement] = []
        changed = False
        for key, item in value.items():
            clean_item, item_replacements = _sanitize_value(item, message_index)
            cleaned[key] = clean_item
            replacements.extend(item_replacements)
            changed = changed or clean_item is not item
        return (cleaned if changed else value), tuple(replacements)
    if isinstance(value, list):
        cleaned_items: list[object] = []
        replacements = []
        changed = False
        for item in value:
            clean_item, item_replacements = _sanitize_value(item, message_index)
            cleaned_items.append(clean_item)
            replacements.extend(item_replacements)
            changed = changed or clean_item is not item
        return (cleaned_items if changed else value), tuple(replacements)
    if isinstance(value, tuple):
        cleaned_items = []
        replacements = []
        changed = False
        for item in value:
            clean_item, item_replacements = _sanitize_value(item, message_index)
            cleaned_items.append(clean_item)
            replacements.extend(item_replacements)
            changed = changed or clean_item is not item
        return (tuple(cleaned_items) if changed else value), tuple(replacements)
    return value, ()


def sanitize_text(
        value: str, *, message_index: int | None = None
) -> tuple[str, tuple[WireStringReplacement, ...]]:
    """Replace NUL and surrogate code points while preserving clean strings exactly."""
    return _sanitize_string(value, message_index)


def sanitize_value(
        value: object, *, message_index: int | None = None
) -> tuple[object, tuple[WireStringReplacement, ...]]:
    """Sanitize string values in arbitrary JSON-compatible request structures."""
    return _sanitize_value(value, message_index)


def sanitize_message(
        message: object, *, message_index: int
) -> tuple[object, tuple[WireStringReplacement, ...]]:
    """Recursively sanitize string values within one model message."""
    return _sanitize_value(message, message_index)


def sanitize_messages(
        messages: object,
) -> tuple[object, tuple[WireStringReplacement, ...]]:
    """Sanitize all message values, keeping clean sequences and strings untouched."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return messages, ()

    cleaned_messages: list[object] = []
    replacements: list[WireStringReplacement] = []
    changed = False
    for message_index, message in enumerate(messages):
        clean_message, message_replacements = _sanitize_value(message, message_index)
        cleaned_messages.append(clean_message)
        replacements.extend(message_replacements)
        changed = changed or clean_message is not message

    if not changed:
        return messages, ()
    if isinstance(messages, list):
        messages[:] = cleaned_messages
        return messages, tuple(replacements)
    return cleaned_messages, tuple(replacements)


def sanitize_request_body(
        body: Mapping[str, object],
) -> tuple[dict[str, object], tuple[WireStringReplacement, ...]]:
    """Sanitize messages with indexes and all other request fields before preparation."""
    cleaned: dict[str, object] = {}
    replacements: list[WireStringReplacement] = []
    for key, value in body.items():
        if key == "messages":
            clean_value, found = sanitize_messages(value)
        else:
            clean_value, found = _sanitize_value(value, None)
        cleaned[key] = clean_value
        replacements.extend(found)
    return cleaned, tuple(replacements)


def record_replacements(
        journal: object | None,
        replacements: Sequence[WireStringReplacement],
        *,
        step: int | None = None,
        request_id: str | None = None,
) -> None:
    """Journal counts and message indexes without retaining unsafe source text."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    for replacement in replacements:
        fields: dict[str, object] = {
            "producer": "kernel_wire_string_hygiene",
            "count": replacement.count,
            "message_index": replacement.message_index,
            "replacement_kind": replacement.kind,
            "dispatch": False,
        }
        if step is not None:
            fields["step"] = step
        if request_id is not None:
            fields["request_id"] = request_id
        emit("wire_string_sanitized", **fields)


def _has_unpaired_surrogate(value: str) -> bool:
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 >= len(value) or not 0xDC00 <= ord(value[index + 1]) <= 0xDFFF:
                return True
            index += 2
            continue
        if 0xDC00 <= codepoint <= 0xDFFF:
            return True
        index += 1
    return False


def _string_violation(value: str) -> str | None:
    if "\x00" in value:
        return "nul"
    if _has_unpaired_surrogate(value):
        return "lone_surrogate"
    return None


def _child_path(path: str, key: object, index: int) -> str:
    if isinstance(key, str) and key in _SAFE_PATH_FIELDS:
        return f"{path}.{key}"[:_MAX_PATH_CHARS]
    return f"{path}.field[{index}]"[:_MAX_PATH_CHARS]


def _first_violation(value: object, path: str) -> WireStringViolation | None:
    if isinstance(value, str):
        kind = _string_violation(value)
        return WireStringViolation(path[:_MAX_PATH_CHARS], kind) if kind else None
    if isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items()):
            if isinstance(key, str):
                key_kind = _string_violation(key)
                if key_kind:
                    return WireStringViolation(
                        f"{path}.key[{index}]"[:_MAX_PATH_CHARS], key_kind)
            violation = _first_violation(item, _child_path(path, key, index))
            if violation is not None:
                return violation
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            violation = _first_violation(item, f"{path}[{index}]"[:_MAX_PATH_CHARS])
            if violation is not None:
                return violation
    return None


def first_wire_string_violation(wire: bytes) -> WireStringViolation | None:
    """Parse serialized JSON and return its first NUL or unpaired surrogate path."""
    try:
        body = json.loads(wire)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return WireStringViolation("body", "invalid_json")
    return _first_violation(body, "body")
