"""Default-off final-wire projection for stable request prefixes."""
from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_tool_reply import copy_tool_reply_message


WIRE_STABLE_ENV = "STATEFRAME_WIRE_STABLE"
VOLATILE_TAIL_MARKER = "[WIRE VOLATILE TAIL]"

_VOLATILE_TOKENS = (
    VOLATILE_TAIL_MARKER,
    "[LEAN TRUTH TAIL]",
    "[ENVIRONMENT STATE]",
    "[CONTINUATION LIMITS",
    "[History fold pointer]",
    "assembled_at=",
    "sandbox_seconds_remaining=",
    "seconds_remaining=",
    "run_id=",
    "run_id:",
    "request_id=",
    "check-execution:",
    "check-receipt:",
    "packet_id",
    "packet_fingerprint",
    "execution_id=",
    "receipt_id=",
    "tool_call_id",
    "native_",
    "[Host fact: StateFrame unavailable]",
)
_MESSAGE_START_VOLATILE_TOKENS = frozenset({
    "[Host fact: StateFrame unavailable]",
})
_REQUEST_WINDOW_MESSAGE_PREFIX = "[Request budget source window]"
_DISK_FACT_PREFIX = re.compile(r"^(?:fact|resource)\s+disk:workspace(?:/|\s|$)", re.IGNORECASE)


@dataclass(frozen=True)
class WireStabilityProjection:
    body: dict[str, object]
    changed: bool
    volatile_marker_added: bool
    disk_capacity_lines_removed: int


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict switch mode; OFF remains the compatibility default."""
    source = os.environ if environ is None else environ
    value = source.get(WIRE_STABLE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{WIRE_STABLE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _is_disk_capacity_line(line: str) -> bool:
    stripped = line.lstrip()
    if not _DISK_FACT_PREFIX.match(stripped):
        return False
    lowered = stripped.lower()
    return "free_bytes" in lowered or "used_bytes" in lowered


def _clean_text(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines(keepends=True):
        if _is_disk_capacity_line(line):
            removed += 1
        else:
            kept.append(line)
    return "".join(kept), removed


def _clean_content(value: object) -> tuple[object, int]:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, Mapping):
        copied = copy.deepcopy(dict(value))
        text = copied.get("text")
        removed = 0
        if isinstance(text, str):
            copied["text"], removed = _clean_text(text)
        return copied, removed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        cleaned: list[object] = []
        removed = 0
        for part in value:
            projected, count = _clean_content(part)
            cleaned.append(projected)
            removed += count
        if isinstance(value, tuple):
            return tuple(cleaned), removed
        return cleaned, removed
    return copy.deepcopy(value), 0


def _first_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for part in value:
            text = _first_text(part)
            if text is not None:
                return text
    return None


def _is_marker_message(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("role") != "user":
        return False
    text = _first_text(value.get("content"))
    return text is not None and text.startswith(VOLATILE_TAIL_MARKER)


def _is_request_window_message(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("role") != "user":
        return False
    text = _first_text(value.get("content"))
    return text is not None and text.startswith(_REQUEST_WINDOW_MESSAGE_PREFIX)


def _last_assistant_or_tool_index(messages: Sequence[object]) -> int:
    return max(
        (index for index, message in enumerate(messages)
         if isinstance(message, Mapping) and message.get("role") in {"assistant", "tool"}),
        default=-1,
    )


def _first_volatile_offset(text: str) -> int | None:
    lowered = text.lower()
    present: list[int] = []
    for token in _VOLATILE_TOKENS:
        offset = 0
        while (found := lowered.find(token.lower(), offset)) >= 0:
            # Ordinary volatile values are recognized at line starts; this host block moves only
            # when it begins the user message, so matching source lines remain untouched.
            if found == 0 or (
                token not in _MESSAGE_START_VOLATILE_TOKENS
                and text[found - 1] == "\n"
            ):
                present.append(found)
                break
            offset = found + 1
    return min(present) if present else None


def _volatile_suffix(text: str, offset: int) -> tuple[str, str]:
    stable = text[:offset]
    volatile = text[offset:]
    if volatile.startswith(VOLATILE_TAIL_MARKER):
        volatile = volatile[len(VOLATILE_TAIL_MARKER):].lstrip("\r\n")
    return stable, volatile


def _split_volatile_content(value: object) -> tuple[object, object | None, bool]:
    """Split a user content value at its first request-volatile token."""
    if isinstance(value, str):
        offset = _first_volatile_offset(value)
        if offset is None:
            return value, None, False
        stable, volatile = _volatile_suffix(value, offset)
        return stable, volatile, True
    if isinstance(value, Mapping):
        text = value.get("text")
        if not isinstance(text, str):
            return value, None, False
        offset = _first_volatile_offset(text)
        if offset is None:
            return value, None, False
        stable_text, volatile_text = _volatile_suffix(text, offset)
        stable = copy.deepcopy(dict(value))
        tail = copy.deepcopy(dict(value))
        stable["text"] = stable_text
        tail["text"] = volatile_text
        return stable, [tail], True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        stable: list[object] = []
        tail: list[object] = []
        split = False
        for part in value:
            if split:
                tail.append(copy.deepcopy(part))
                continue
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                text = part["text"]
                offset = _first_volatile_offset(text)
                if offset is not None:
                    # Whitespace before a volatile-only text block belongs with its tail.
                    if not text[:offset].strip():
                        offset = 0
                    stable_text, volatile_text = _volatile_suffix(text, offset)
                    if stable_text:
                        before = copy.deepcopy(dict(part))
                        before["text"] = stable_text
                        stable.append(before)
                    after = copy.deepcopy(dict(part))
                    after["text"] = volatile_text
                    tail.append(after)
                    split = True
                    continue
            if isinstance(part, str):
                offset = _first_volatile_offset(part)
                if offset is not None:
                    if offset:
                        stable_text, volatile_text = _volatile_suffix(part, offset)
                        if stable_text:
                            stable.append(stable_text)
                    else:
                        volatile_text = _volatile_suffix(part, offset)[1]
                    tail.append(volatile_text)
                    split = True
                    continue
            stable.append(copy.deepcopy(part))
        if split:
            if isinstance(value, tuple):
                return tuple(stable), tail, True
            return stable, tail, True
    return value, None, False


_VOLATILE_SEGMENT_MARKERS = (
    ("[WIRE_TRUTH_FACT ", "wire_facts"),
    ("[LEAN TRUTH TAIL]", "truth"),
    ("[CONTINUATION LIMITS", "limits"),
    ("[ENVIRONMENT STATE]", "environment"),
    ("assembled_at=", "time"),
    ("sandbox_seconds_remaining=", "time"),
    ("seconds_remaining=", "time"),
    ("run_id=", "identifiers"),
    ("run_id:", "identifiers"),
    ("request_id=", "identifiers"),
    ("check-execution:", "identifiers"),
    ("check-receipt:", "identifiers"),
    ("packet_id", "identifiers"),
    ("packet_fingerprint", "identifiers"),
    ("execution_id=", "identifiers"),
    ("receipt_id=", "identifiers"),
    ("tool_call_id", "identifiers"),
    ("native_", "identifiers"),
)
_TAIL_CHUNK_ORDER = (
    "truth", "limits", "environment", "time", "identifiers", "other", "wire_facts",
)


def _strip_marker_mapping(value: Mapping[str, object]) -> object | None:
    copied = copy.deepcopy(dict(value))
    text = copied.get("text")
    if not isinstance(text, str) or not text.startswith(VOLATILE_TAIL_MARKER):
        return copied

    stripped = text[len(VOLATILE_TAIL_MARKER):].lstrip("\r\n")
    if not stripped.strip() and set(copied).issubset({"type", "text"}):
        return None
    copied["text"] = stripped
    return copied


def _strip_marker_sequence(value: Sequence[object]) -> object | None:
    result: list[object] = []
    removed = False
    for part in value:
        text = _first_text(part)
        if removed or not isinstance(text, str) or not text.startswith(VOLATILE_TAIL_MARKER):
            result.append(copy.deepcopy(part))
            continue
        stripped = _strip_volatile_marker(part)
        if stripped is not None:
            result.append(stripped)
        removed = True
    return result or None


def _strip_volatile_marker(value: object) -> object | None:
    if isinstance(value, str):
        if not value.startswith(VOLATILE_TAIL_MARKER):
            return copy.deepcopy(value)
        stripped = value[len(VOLATILE_TAIL_MARKER):].lstrip("\r\n")
        return stripped or None
    if isinstance(value, Mapping):
        return _strip_marker_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _strip_marker_sequence(value)
    return copy.deepcopy(value)


def _tail_category(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in (
        "run_id", "request_id", "check-execution:", "check-receipt:",
        "packet_id", "packet_fingerprint", "execution_id=", "receipt_id=",
        "tool_call_id", "native_",
    )):
        return "identifiers"
    if any(token in lowered for token in (
        "assembled_at=", "sandbox_seconds_remaining=", "seconds_remaining=",
        "minutes left", "minutes_left", "minutes_remaining",
        "turns left", "turns_left", "turns_remaining",
    )):
        return "time"
    return "other"


def _volatile_segments(text: str) -> list[tuple[str, str]]:
    markers: list[tuple[int, str, str]] = []
    lowered = text.lower()
    for marker, category in _VOLATILE_SEGMENT_MARKERS:
        offset = 0
        while (found := lowered.find(marker.lower(), offset)) >= 0:
            # a marker counts only at the START of a line (where host panels put it). Substrings like
            # "request_id=" / "native_" inside delivered source code were being moved to the tail, which broke the
            # source-delivery proof (full-ALL smoke: WIRE_TRUTH_BLOCKED source_delivery_incomplete, run stopped).
            if found == 0 or text[found - 1] == "\n":
                markers.append((found, marker, category))
            offset = found + len(marker)
    if not markers:
        stripped = text.strip("\r\n")
        return [(_tail_category(stripped), stripped)] if stripped else []
    markers.sort(key=lambda item: item[0])
    # the "[CONTINUATION LIMITS | ...]" LINE is one atomic segment. The owner contract digests that exact
    # line (kernel_completion_owner._wire_continuation_limits); splitting it on inner time markers raised
    # RequestActionContractError "serialized continuation limits were removed or changed" before request 0.
    atomic_spans: list[tuple[int, int]] = []
    search = 0
    while (start := text.find("[CONTINUATION LIMITS |", search)) >= 0:
        end = text.find("\n", start)
        end = len(text) if end < 0 else end
        atomic_spans.append((start, end))
        search = end
    if atomic_spans:
        markers = [m for m in markers
                   if not any(span_start < m[0] < span_end for span_start, span_end in atomic_spans)]
    non_overlapping: list[tuple[int, str, str]] = []
    covered_until = -1
    for marker in markers:
        if marker[0] < covered_until:
            continue
        non_overlapping.append(marker)
        covered_until = marker[0] + len(marker[1])
    segments: list[tuple[str, str]] = []
    first = non_overlapping[0][0]
    prefix = text[:first].strip("\r\n")
    if prefix:
        segments.append((_tail_category(prefix), prefix))
    for index, (start, _marker, category) in enumerate(non_overlapping):
        end = (
            non_overlapping[index + 1][0]
            if index + 1 < len(non_overlapping) else len(text)
        )
        segment = text[start:end].strip("\r\n")
        if segment:
            segments.append((category, segment))
    return segments


def _store_volatile_chunk(
    chunks: dict[str, list[object]], category: str, value: object,
) -> None:
    detached = copy.deepcopy(value)
    if category in {"identifiers", "other", "wire_facts"}:
        # FrameFit5 packs independently verified facts into its final tail.
        # They must not become part of a replaceable limits/environment chunk.
        values = chunks.setdefault(category, [])
        if detached not in values:
            values.append(detached)
    elif category == "time":
        # one slot per time KEY (assembled_at, sandbox_seconds_remaining, ...), latest wins. A single
        # shared slot let assembled_at overwrite sandbox seconds / minutes / turns-left, dropping TRUE useful facts.
        values = chunks.setdefault(category, [])
        key = _time_key(detached)
        values[:] = [item for item in values if _time_key(item) != key]
        values.append(detached)
    else:
        chunks[category] = [detached]


def _time_key(value: object) -> str:
    text = value if isinstance(value, str) else (
        value.get("text") if isinstance(value, Mapping) else None)
    if not isinstance(text, str):
        return ""
    head = text.strip().split("=", 1)[0] if "=" in text else text.strip()[:24]
    return head.strip().lower()


def _record_volatile_chunks(chunks: dict[str, list[object]], value: object) -> None:
    if isinstance(value, str):
        for category, segment in _volatile_segments(value):
            _store_volatile_chunk(chunks, category, segment)
        return
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            for category, segment in _volatile_segments(text):
                part = copy.deepcopy(dict(value))
                part["text"] = segment
                _store_volatile_chunk(chunks, category, part)
        else:
            _store_volatile_chunk(chunks, "other", value)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for part in value:
            _record_volatile_chunks(chunks, part)
        return
    _store_volatile_chunk(chunks, "other", value)


def _tail_content(chunks: Mapping[str, Sequence[object]]) -> object | None:
    values = [
        copy.deepcopy(value)
        for category in _TAIL_CHUNK_ORDER
        for value in chunks.get(category, ())
    ]
    if not values:
        return None
    if all(isinstance(value, str) for value in values):
        return "\n\n".join(value for value in values if isinstance(value, str))
    structured: list[object] = []
    for value in values:
        if isinstance(value, str):
            structured.append({"type": "text", "text": value})
        else:
            structured.append(value)
    return structured[0] if len(structured) == 1 else structured


def _tail_message(content: object | None) -> dict[str, object]:
    if content is None:
        tail_content: object = VOLATILE_TAIL_MARKER
    elif isinstance(content, str):
        tail_content = VOLATILE_TAIL_MARKER + "\n\n" + content
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        tail_content = [
            {"type": "text", "text": VOLATILE_TAIL_MARKER + "\n\n"},
            *copy.deepcopy(list(content)),
        ]
    else:
        tail_content = [
            {"type": "text", "text": VOLATILE_TAIL_MARKER + "\n\n"},
            copy.deepcopy(content),
        ]
    return {"role": "user", "content": tail_content}


def _content_is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not any(
            _content_is_empty(item)
            for key, item in value.items()
            if key != "type"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return not any(_content_is_empty(item) for item in value)
    return False


def _split_user_message(
    message: dict[str, object],
    chunks: dict[str, list[object]],
) -> bool:
    if "content" not in message:
        return False
    stable_content, volatile_content, split = _split_volatile_content(message["content"])
    if not split:
        return False

    message["content"] = stable_content
    if volatile_content is not None:
        _record_volatile_chunks(chunks, volatile_content)
    return (
        _content_is_empty(stable_content)
        and set(message).issubset({"role", "content"})
    )


def _project_message(
    message: object,
    prior_chunks: dict[str, list[object]],
    current_chunks: dict[str, list[object]],
) -> tuple[object, bool, int]:
    if not isinstance(message, Mapping):
        return copy.deepcopy(message), False, 0

    if (isinstance(message, dict) and message.get("role") == "tool"
            and isinstance(getattr(message, "_stateframe_tool_reply_receipt", None), Mapping)):
        copied = copy_tool_reply_message(message)
    else:
        copied = copy.deepcopy(dict(message))
    disk_lines_removed = 0
    if "content" in copied:
        copied["content"], disk_lines_removed = _clean_content(copied["content"])

    if copied.get("role") != "user":
        return copied, False, disk_lines_removed
    if _is_marker_message(copied):
        previous_tail = _strip_volatile_marker(copied.get("content"))
        if previous_tail is not None:
            _record_volatile_chunks(prior_chunks, previous_tail)
        return copied, True, disk_lines_removed
    if _split_user_message(copied, current_chunks):
        return copied, True, disk_lines_removed
    return copied, False, disk_lines_removed


def _add_volatile_marker(messages: list[object], content: object | None) -> None:
    """Append one rebuilt volatile tail after all stable history."""
    messages.append(_tail_message(content))


def project(
    body: dict[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> WireStabilityProjection:
    """Keep request volatility in one rebuilt user tail after all history."""
    if not enabled(environ):
        return WireStabilityProjection(body, False, False, 0)

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        return WireStabilityProjection(body, False, False, 0)

    messages: list[object] = []
    request_window_messages: list[object] = []
    prior_chunks: dict[str, list[object]] = {}
    current_chunks: dict[str, list[object]] = {}
    disk_lines_removed = 0
    had_user_message = any(
        isinstance(message, Mapping) and message.get("role") == "user"
        for message in raw_messages
    )
    last_action_index = _last_assistant_or_tool_index(raw_messages)

    for index, message in enumerate(raw_messages):
        if _is_request_window_message(message):
            # Keep historical proof-bound windows in place. Move only current trailing windows
            # whole, preserving their exact header, source bytes, and expansion route.
            copied_window = copy.deepcopy(dict(message))
            if index > last_action_index:
                request_window_messages.append(copied_window)
            else:
                messages.append(copied_window)
            continue
        projected_message, drop_message, removed = _project_message(
            message, prior_chunks, current_chunks,
        )
        disk_lines_removed += removed
        if not drop_message:
            messages.append(projected_message)

    marker_was_present = any(_is_marker_message(message) for message in raw_messages)
    marker_added = had_user_message and not marker_was_present
    if had_user_message:
        chunks = {category: list(values) for category, values in prior_chunks.items()}
        chunks.update({
            category: list(values) for category, values in current_chunks.items()
        })
        _add_volatile_marker(messages, _tail_content(chunks))
        # Source windows are indivisible proof-bound messages; place them after
        # the marker so their per-turn expansion handles cannot shorten the prefix.
        messages.extend(request_window_messages)

    changed = messages != raw_messages or disk_lines_removed > 0
    if not changed:
        return WireStabilityProjection(body, False, False, 0)

    projected = copy.deepcopy(dict(body))
    projected["messages"] = messages
    return WireStabilityProjection(
        projected,
        True,
        marker_added,
        disk_lines_removed,
    )
