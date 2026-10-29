"""Lossless PAGE_LOG projection for frame-only StateFrame requests.

Each request is rebuilt as stable system/task/tools, a host-verified StateFrame
base plus typed deltas and versioned keyed pages, then one volatile user message.
The run state advances only after response_received is durably journaled.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from kernel_stateframe_final_message import (
    AUTHORITY_OVERLAY_HEADER,
    PAGE_LOG_FINAL_MESSAGE_NAME,
    compose_final_user_message,
)
from kernel_stateframe_message_layout import (
    StateFrameMessageLayoutError,
    stateframe_message_index,
    validate_stateframe_message_order,
)


MODE_ENV = "STATEFRAME_PAGE_LOG"
CAP_MODE_ENV = "STATEFRAME_PAGE_LOG_CAP"
SOURCE_SUPERSESSION_MODE_ENV = "STATEFRAME_PAGE_LOG_SRC_SUPERSEDE"
RETIREMENT_MODE_ENV = "STATEFRAME_PAGE_LOG_RETIREMENT"
EXCHANGE_MODE_ENV = "STATEFRAME_PAGE_LOG_EXCHANGE"
WINDOW_KEY_MODE_ENV = "STATEFRAME_PAGE_LOG_WINDOW_KEY"
EVENT_PREFIX = "stateframe_page_log"
_RUNTIME_STATE_ATTR = "_stateframe_page_log_state"
_PENDING_ATTR = "_stateframe_page_log_pending"
_SOURCE_SUPERSESSION_WARNING_ATTR = "_stateframe_page_log_src_supersede_warning_run_id"
_EXCHANGE_TEXT_REFUSAL_ATTR = "_stateframe_page_log_exchange_text_refused_run_id"
_VALID_MODES = frozenset({"off", "shadow", "on"})
_VALID_EXCHANGE_LAYOUTS = frozenset({"native"})
_MAX_JOURNALED_EXCHANGE_IDS = 64
_MAX_FRAME_INTERVAL = 2**31 - 1
_MAX_DELTA_BYTES = 2**63 - 1
_CONTEXT_POLICY_ERROR = "rebased request still exceeds the 85 percent context policy"
_SRC_HEADER = re.compile(r"^\[Current source:\s*([^\]\r\n]+)\]")
_HEAD_HEADER = re.compile(r"^\[MODULE HEAD \| ([^\]\r\n]+)\]")
_STATIC_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("literal_site_inventory", ("[LITERAL SITE INVENTORY",)),
    ("navbasics", ("[NAVBASICS",)),
    ("public_authored_check_context", ("[Public authored-check context",)),
    ("passive_conclusions", ("[Passive conclusions",)),
    ("files_in_frame", ("[Files in frame",)),
    ("public_behavior_witness", ("[PUBLIC BEHAVIOR WITNESS",)),
    ("import_graph_fact", ("[IMPORT GRAPH FACT",)),
    ("initialization_fact", ("[INITIALIZATION FACT",)),
)
_GREEN_CLAIM_MARKERS = (
    "[host] All tests pass on the current candidate.",
    "[host] Current candidate passes all tests.",
)


class PageLogError(ValueError):
    """A request cannot be projected without weakening current-frame meaning."""


class PageLogBudgetError(PageLogError):
    """The rebased request still exceeds the host admission prompt budget."""

    def __init__(self, prompt_tokens: int, cap: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.cap = cap
        super().__init__("rebased request still exceeds the host prompt budget")


@dataclass(frozen=True)
class PageLogState:
    run_id: str
    frame_state: object
    cache_frame_state: object
    log: tuple[dict[str, object], ...]
    pages: tuple[tuple[str, int, str], ...]
    versions: tuple[tuple[str, int], ...]
    stale_contents: tuple[str, ...]
    stale_tokens: int
    greenfact_revisions: tuple[str, ...]
    generation: int
    turn_index: int
    epoch: int
    epoch_turns: int


@dataclass(frozen=True)
class PageLogProjection:
    body: dict[str, object]
    state: PageLogState
    previous_state: PageLogState | None
    action: str
    rebase_reason: str | None
    prefix_preserved: bool
    prompt_tokens: int | None
    live_tokens: int | None
    stale_tokens: int | None
    source_shadow_mismatches: tuple[str, ...] = ()
    source_supersession_error_type: str | None = None
    retirement_shadow_mismatches: tuple[str, ...] = ()
    retirement_error_type: str | None = None
    exchange_lifted: bool = False
    exchange_tool_call_ids: tuple[str, ...] = ()
    exchange_shadow_mismatches: tuple[str, ...] = ()
    exchange_error_type: str | None = None
    exchange_shadow_error_type: str | None = None
    window_key_shadow_mismatches: tuple[str, ...] = ()
    window_key_error_type: str | None = None
    budget_action: str | None = None
    budget_cap: int | None = None


@dataclass(frozen=True)
class PageLogPending:
    runtime: object
    previous_state: PageLogState | None
    candidate_state: PageLogState
    mode: str
    request_id: str | None
    step: int | None


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    selected = str(source.get(MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def source_supersession_mode(environ: Mapping[str, str] | None = None) -> str:
    """Read the independent source-page supersession control; invalid values fail closed to legacy behavior."""
    source = os.environ if environ is None else environ
    selected = str(source.get(SOURCE_SUPERSESSION_MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def page_retirement_mode(environ: Mapping[str, str] | None = None) -> str:
    """Read the independent page-retirement rebase policy; invalid values retain legacy behavior."""
    source = os.environ if environ is None else environ
    selected = str(source.get(RETIREMENT_MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def page_log_cap_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the host prompt-cap policy, failing invalid values to legacy behavior."""
    source = os.environ if environ is None else environ
    selected = str(source.get(CAP_MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def page_log_exchange_mode(environ: Mapping[str, str] | None = None) -> str:
    """Select the only supported provider-native exchange placement."""
    source = os.environ if environ is None else environ
    selected = str(source.get(EXCHANGE_MODE_ENV, "native")).strip().lower()
    return selected if selected in _VALID_EXCHANGE_LAYOUTS else "native"


def page_log_window_key_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve per-source-window page identity; invalid values preserve legacy behavior."""
    source = os.environ if environ is None else environ
    selected = str(source.get(WINDOW_KEY_MODE_ENV, "off")).strip().lower()
    return selected if selected in _VALID_MODES else "off"


def _page_log_exchange_text_requested(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(EXCHANGE_MODE_ENV, "")).strip().lower() == "text"


def _safe_journal_error_type(error: BaseException) -> str:
    try:
        from kernel_boundary_recovery import safe_error_class
        return safe_error_class(error)
    except Exception:
        return "ContractError"


def _wire_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _newest_complete_exchange(
        body: Mapping[str, object],
) -> tuple[int, tuple[Mapping[str, object], ...]] | None:
    """Return only the newest complete assistant tool call and its contiguous replies."""
    raw_messages = body.get("messages")
    if (not isinstance(raw_messages, list)
            or any(not isinstance(message, Mapping) for message in raw_messages)):
        return None
    messages = raw_messages
    try:
        state_index = stateframe_message_index(messages)
    except StateFrameMessageLayoutError:
        return None
    assistants = [
        index for index, message in enumerate(messages)
        if index > state_index and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list) and message["tool_calls"]
    ]
    if not assistants:
        return None

    start = assistants[-1]
    assistant = messages[start]
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    call_ids: list[str] = []
    for call in calls:
        call_id = call.get("id") if isinstance(call, Mapping) else None
        if not isinstance(call_id, str) or not call_id or call_id in call_ids:
            return None
        call_ids.append(call_id)

    end = start + 1
    replies: list[Mapping[str, object]] = []
    while end < len(messages) and messages[end].get("role") == "tool":
        replies.append(messages[end])
        end += 1
    reply_ids = [message.get("tool_call_id") for message in replies]
    if (len(replies) != len(call_ids)
            or any(not isinstance(reply_id, str) or not reply_id for reply_id in reply_ids)
            or len(set(reply_ids)) != len(reply_ids)
            or set(reply_ids) != set(call_ids)):
        return None
    if end >= len(messages) or messages[-1].get("role") != "user":
        return None
    if any(message.get("role") != "user" for message in messages[end:]):
        return None
    return start, tuple(messages[start:end])


def _body_without_exchange(body: dict[str, object]) -> dict[str, object]:
    out = copy.deepcopy(body)
    source = out.get("messages")
    if not isinstance(source, list):
        raise PageLogError("exchange request messages are unavailable")
    # StateFrame replaces prior action history. The newest exchange travels
    # separately and is reinserted natively before final [V].
    out["messages"] = [
        message for message in source
        if not isinstance(message, Mapping)
        or message.get("role") not in {"assistant", "tool"}
    ]
    return out


def _insert_exchange_before_final_user(
        body: dict[str, object],
        exchange: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    out = copy.deepcopy(body)
    messages = out.get("messages")
    if (not isinstance(messages, list) or not messages
            or not isinstance(messages[-1], Mapping)
            or messages[-1].get("role") != "user"):
        raise PageLogError("exchange projection has no final volatile user message")
    messages[-1:-1] = copy.deepcopy(list(exchange))
    return out


def page_key(content: str) -> str | None:
    """Return an allowlisted logical key; presentation ranges are never identity."""
    first = content.split("\n", 1)[0]
    source = _SRC_HEADER.match(first)
    if source:
        path_and_range = source.group(1).strip()
        path = re.split(r"\s+lines?\s+\d|\s+\(lines?\s+\d|\s+range\s*[:=]", path_and_range, maxsplit=1)[0]
        if not path or path != path.strip():
            return None
        return "src:" + path
    head = _HEAD_HEADER.match(first)
    if head:
        path_and_range = head.group(1).strip()
        path = re.split(r"\s+lines?\s+\d|\s+\(lines?\s+\d|\s+range\s*[:=]", path_and_range, maxsplit=1)[0]
        if not path:
            return None
        return "head:" + path
    for key, prefixes in _STATIC_SECTIONS:
        if first.startswith(prefixes):
            return "section:" + key
    # Stop, verdict, green, menu, done, and unknown sections remain volatile.
    return None


def _merge_source_line_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise PageLogError("source window has invalid line coverage")
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _source_window_ranges_from_metadata(metadata: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    raw_ranges = metadata.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise PageLogError("source window has no canonical line coverage")
    ranges: list[tuple[int, int]] = []
    for row in raw_ranges:
        if not isinstance(row, Mapping):
            raise PageLogError("source window line coverage is malformed")
        start, end = row.get("start_line"), row.get("end_line")
        if type(start) is not int or type(end) is not int:
            raise PageLogError("source window line coverage is not integral")
        ranges.append((start, end))
    return _merge_source_line_ranges(ranges)


def _legacy_source_window_ranges(content: str) -> tuple[tuple[int, int], ...]:
    if "[Model source window]" in content:
        raise PageLogError("source window model header is malformed")
    explicit: list[tuple[int, int]] = []
    header = content.split("\n", 1)[0]
    header_range = re.search(r"\blines?\s+(\d+)\s*[-–]\s*(\d+)\b", header, re.IGNORECASE)
    if header_range is not None:
        explicit.append((int(header_range.group(1)), int(header_range.group(2))))
    for match in re.finditer(
            r"^(?:--- source lines (\d+)-(\d+) ---|Coverage: lines (\d+)-(\d+))\s*$",
            content, re.IGNORECASE | re.MULTILINE):
        first, second = match.group(1, 2) if match.group(1) is not None else match.group(3, 4)
        explicit.append((int(first), int(second)))
    if explicit:
        return _merge_source_line_ranges(explicit)

    from kernel_request_diet import _source_identity

    identity = _source_identity({"role": "user", "content": content})
    source_range = identity[3] if identity is not None and identity[0] == "source" else None
    if source_range is None:
        raise PageLogError("source window has no stable range or coverage header")
    return _merge_source_line_ranges((source_range,))


def _source_window_identity(
        content: str,
) -> tuple[str, tuple[tuple[int, int], ...], str] | None:
    """Bind a source page to path plus coverage; revision and continuation handles are metadata."""
    legacy_key = page_key(content)
    if legacy_key is None or not legacy_key.startswith("src:"):
        return None
    header_path = legacy_key[len("src:"):].replace("\\", "/")
    from kernel_request_source_window import (
        parse_model_source_window,
        verify_model_source_window,
    )

    metadata = parse_model_source_window(content)
    if metadata is None:
        ranges = _legacy_source_window_ranges(content)
    else:
        metadata_path = str(metadata.get("path") or "").replace("\\", "/").strip()
        if not metadata_path or metadata_path != header_path:
            raise PageLogError("source window path disagrees with its model header")
        if not verify_model_source_window(content, path=header_path):
            raise PageLogError("source window literal does not match its declared coverage")
        ranges = _source_window_ranges_from_metadata(metadata)
    range_id = ",".join(f"L{start}-{end}" for start, end in ranges)
    return header_path, ranges, f"src:{header_path}#{range_id}"


def _source_window_revision(content: str) -> str | None:
    from kernel_request_source_window import parse_model_source_window

    metadata = parse_model_source_window(content)
    if metadata is None:
        return None
    revision = metadata.get("revision")
    if isinstance(revision, str) and revision:
        return revision
    digest = metadata.get("sha256")
    return "sha256:" + digest if isinstance(digest, str) and digest else None


def _source_window_path_revision(content: str) -> tuple[str, str | None] | None:
    identity = _source_window_identity(content)
    if identity is None:
        return None
    return identity[0], _source_window_revision(content)


def _current_source_window_revisions(current: Mapping[str, str]) -> dict[str, str | None]:
    revisions: dict[str, str | None] = {}
    for key, content in current.items():
        row = _source_window_path_revision(content) if key.startswith("src:") and "#" in key else None
        if row is not None:
            revisions[row[0]] = row[1]
    return revisions


def _matches_source_window_revision(
        content: str, revisions: Mapping[str, str | None],
) -> bool:
    row = _source_window_path_revision(content)
    return row is not None and row[0] in revisions and row[1] == revisions[row[0]]


def _retained_prior_source_windows(
        previous: Mapping[str, tuple[int, str]],
        current: Mapping[str, str],
        revisions: Mapping[str, str | None],
) -> dict[str, str]:
    return {
        key: content
        for key, (_version, content) in previous.items()
        if key not in current and key.startswith("src:") and "#" in key
        and _matches_source_window_revision(content, revisions)
    }


def _retain_prior_source_windows(
        current: dict[str, str],
        previous: Mapping[str, tuple[int, str]],
) -> dict[str, str]:
    revisions_by_path = _current_source_window_revisions(current)
    if not revisions_by_path:
        return current
    current = {
        key: content for key, content in current.items()
        if not (key.startswith("src:") and "#" in key)
        or _matches_source_window_revision(content, revisions_by_path)
    }
    merged = _retained_prior_source_windows(previous, current, revisions_by_path)
    for key, content in current.items():
        # Repeated windows in the current request are newest in the supersession order.
        merged.pop(key, None)
        merged[key] = content
    _remove_fully_superseded_source_windows(merged)
    return merged


def _subtract_source_line_ranges(
        ranges: Sequence[tuple[int, int]],
        covered: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    remaining = list(ranges)
    for cover_start, cover_end in covered:
        next_ranges: list[tuple[int, int]] = []
        for start, end in remaining:
            if cover_end < start or cover_start > end:
                next_ranges.append((start, end))
                continue
            if start < cover_start:
                next_ranges.append((start, cover_start - 1))
            if cover_end < end:
                next_ranges.append((cover_end + 1, end))
        remaining = next_ranges
    return tuple(remaining)


def _source_window_index_ranges(
        pages: Mapping[str, tuple[int, str]],
) -> dict[str, tuple[tuple[int, int], ...]]:
    by_path: dict[str, list[tuple[str, tuple[tuple[int, int], ...]]]] = {}
    for key, (_version, content) in pages.items():
        if not key.startswith("src:") or "#" not in key:
            continue
        identity = _source_window_identity(content)
        if identity is None or identity[2] != key:
            raise PageLogError("source window key does not match its literal coverage")
        path, ranges, _stable_key = identity
        by_path.setdefault(path, []).append((key, ranges))

    current: dict[str, tuple[tuple[int, int], ...]] = {}
    for windows in by_path.values():
        later_ranges: tuple[tuple[int, int], ...] = ()
        for key, ranges in reversed(windows):
            effective = _subtract_source_line_ranges(ranges, later_ranges)
            if effective:
                current[key] = effective
            later_ranges = _merge_source_line_ranges((*later_ranges, *ranges))
    return current


def _remove_fully_superseded_source_windows(current: dict[str, str]) -> None:
    page_views = {key: (1, content) for key, content in current.items()}
    current_ranges = _source_window_index_ranges(page_views)
    for key in tuple(current):
        if key.startswith("src:") and "#" in key and key not in current_ranges:
            del current[key]


def _render_source_line_ranges(ranges: Sequence[tuple[int, int]]) -> str:
    return ",".join(f"{start}-{end}" for start, end in ranges)


def _page_message(key: str, version: int, content: str) -> dict[str, object]:
    return {"role": "user", "content": f"[page {key} v{version}]\n{content}"}


def _retired_message(key: str) -> dict[str, object]:
    return {"role": "user", "content": f"[page retired] {key}"}


def _page_map(state: PageLogState | None) -> dict[str, tuple[int, str]]:
    if state is None:
        return {}
    return {key: (version, content) for key, version, content in state.pages}


def _version_map(state: PageLogState | None) -> dict[str, int]:
    if state is None:
        return {}
    return dict(state.versions)


def _page_index_content(pages: Mapping[str, tuple[int, str]]) -> str:
    if not pages:
        return ""
    current_ranges = _source_window_index_ranges(pages)
    entries = []
    for key, (version, _content) in pages.items():
        entry = f"{key}@v{version}"
        ranges = current_ranges.get(key)
        if ranges is not None:
            entry += f" ranges={_render_source_line_ranges(ranges)}"
        entries.append(entry)
    return "[page index]\n" + "\n".join(entries)


def _replay_page_log(log: Sequence[Mapping[str, object]]) -> dict[str, str]:
    current: dict[str, str] = {}
    for message in log:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith("[page retired] "):
            current.pop(content[len("[page retired] "):], None)
            continue
        if not content.startswith("[page "):
            continue
        header, separator, page_body = content.partition("\n")
        if not separator or not header.endswith("]"):
            raise PageLogError("page log contains a malformed version header")
        key_version = header[len("[page "):-1]
        key, marker, raw_version = key_version.rpartition(" v")
        if not marker or not key or not raw_version.isdecimal():
            raise PageLogError("page log contains an invalid typed key/version")
        current[key] = page_body
    return current


_VOLATILE_AUTHORITY_HEADERS = (
    "[unified stop authority]", "[stop authority]", "[verification certificate]",
    "[completion owner]", "[wire_truth_fact", "[done authority]", "[menu authority]",
)
_HOST_MESSAGE_ANNOTATIONS = {
    "_request_budget_section", "section", "authored_check_context", "context_digest",
}


def _host_annotations_are_representable(message: Mapping[str, object]) -> bool:
    """Recognize host-only labels while keeping all message text in the final V block."""
    extra = set(message) - {"role", "content"}
    if not extra <= _HOST_MESSAGE_ANNOTATIONS:
        return False
    for key in ("_request_budget_section", "section", "context_digest"):
        if key in message and (not isinstance(message[key], str) or not message[key]):
            return False
    if "context_digest" in message and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(message["context_digest"])):
        return False
    if "authored_check_context" in message and message["authored_check_context"] is not True:
        return False
    if "authored_check_context" in message and "context_digest" not in message:
        return False
    return True


def _tool_guidance_label(field: str) -> str | None:
    if not field.startswith("/"):
        raise PageLogError("tool guidance field is not a JSON pointer")
    parts = [
        part.replace("~1", "/").replace("~0", "~")
        for part in field.split("/")[1:]
    ]
    if not parts or parts[-1] != "description":
        raise PageLogError("tool guidance field is not a description")
    labels = [
        part for part in parts[:-1]
        if part not in {"function", "parameters", "properties", "items"}
        and not part.isdecimal()
    ]
    return ".".join(labels) or None


def _decode_tool_guidance_row(line: str) -> tuple[str, str, str]:
    try:
        row = json.loads(line[len("[tool guidance] "):])
    except (TypeError, ValueError) as error:
        raise PageLogError("tool guidance row is malformed") from error
    if not isinstance(row, Mapping):
        raise PageLogError("tool guidance row is not an object")
    tool = row.get("tool")
    field = row.get("field")
    if not isinstance(tool, str) or not tool.strip() or not isinstance(field, str):
        raise PageLogError("tool guidance row has no tool or field")
    label = _tool_guidance_label(field)
    if isinstance(row.get("text"), str):
        value = " ".join(str(row["text"]).split())
        slot = "main" if label is None else "field:" + label
    elif row.get("present") is False:
        value = "No current description" if label is None else f"{label}: no current description"
        slot = "notice"
    else:
        raise PageLogError("tool guidance row has no current description")
    return tool.strip(), slot, value


def _parse_compact_tool_guidance_line(line: str) -> tuple[str, str] | None:
    if ": " not in line:
        return None
    tool, value = line.split(": ", 1)
    if not tool.strip() or not value.strip():
        raise PageLogError("compact tool guidance row is malformed")
    return tool.strip(), " ".join(value.split())


def _joined_tool_descriptions(values: Sequence[str]) -> str:
    return "; ".join(dict.fromkeys(value for value in values if value))


def _format_tool_guidance_description(fields: Mapping[str, Sequence[str]]) -> str:
    main = _joined_tool_descriptions(fields.get("main", ()))
    details = [
        f"{slot[len('field:'):]}: {value}"
        for slot in sorted(key for key in fields if key.startswith("field:"))
        if (value := _joined_tool_descriptions(fields[slot]))
    ]
    notice = _joined_tool_descriptions(fields.get("notice", ()))
    if notice and notice != main and notice not in details:
        details.append(notice)
    parts = ([main] if main else []) + details
    return "; ".join(parts) or "No current description is available."


def _render_tool_guidance(grouped: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, str]:
    return {
        tool: _format_tool_guidance_description(fields)
        for tool, fields in sorted(grouped.items())
    }


def _compact_tool_guidance(content: str) -> tuple[dict[str, str], str]:
    grouped: dict[str, dict[str, list[str]]] = {}
    residual: list[str] = []
    in_compact_section = False
    found = False
    for line in content.splitlines():
        if line == "[tool guidance]":
            found = True
            in_compact_section = True
            continue
        if line.startswith("[tool guidance] "):
            found = True
            in_compact_section = False
            tool, slot, value = _decode_tool_guidance_row(line)
            grouped.setdefault(tool, {}).setdefault(slot, []).append(value)
            continue
        compact_row = _parse_compact_tool_guidance_line(line) if in_compact_section else None
        if compact_row is not None:
            tool, value = compact_row
            grouped.setdefault(tool, {}).setdefault("main", []).append(value)
            continue
        in_compact_section = False
        residual.append(line)
    if not found or not grouped:
        return {}, content
    return _render_tool_guidance(grouped), "\n".join(residual)


def _validated_tail_contents(
        tail: Sequence[object], greenfact: str,
) -> tuple[list[str], str]:
    copied = [copy.deepcopy(dict(message)) for message in tail if isinstance(message, Mapping)]
    if len(copied) != len(tail):
        raise PageLogError("tail contains a malformed message")
    contents: list[str] = []
    for message in copied:
        if message.get("role") != "user" or not _host_annotations_are_representable(message):
            raise PageLogError("volatile tail contains unsupported message metadata or role")
        content = message.get("content")
        if not isinstance(content, str):
            raise PageLogError("volatile tail contains non-text user content")
        contents.append(content)
    green_occurrences = sum(content.count(greenfact) for content in contents) if greenfact else 0
    if not green_occurrences:
        return contents, ""
    suffix_indexes = [
        index for index, content in enumerate(contents) if content.endswith(greenfact)
    ]
    if green_occurrences != 1 or len(suffix_indexes) != 1:
        raise PageLogError("GREENFACT is not a unique exact suffix of one current user tail")
    index = suffix_indexes[0]
    contents[index] = contents[index][:-len(greenfact)]
    return contents, greenfact


def _record_logical_page(
        content: str,
        current: dict[str, str],
        volatile: list[str],
        *,
        window_key_sources: bool = False,
) -> None:
    if window_key_sources and (legacy_key := page_key(content)) and legacy_key.startswith("src:"):
        identity = _source_window_identity(content)
        if identity is None:
            raise PageLogError("source window cannot be assigned a stable range key")
        key = identity[2]
    else:
        key = page_key(content)
    header = content.split("\n", 1)[0].strip().lower()
    if any(header.startswith(marker) for marker in _VOLATILE_AUTHORITY_HEADERS):
        key = None
    if key is None:
        volatile.append(content)
    elif key in current and key.startswith("src:") and window_key_sources:
        if current[key] != content:
            raise PageLogError("duplicate source window has conflicting literal bytes")
    elif key in current and key.startswith(("src:", "head:")):
        current[key] += "\n\n" + content
    elif key in current:
        # Repeated static sections remain volatile rather than replacing a page.
        volatile.append(content)
    else:
        current[key] = content


def _logical_page_map(
        tail: Sequence[object], greenfact: str, *,
        window_key_sources: bool = False,
        previous_pages: Mapping[str, tuple[int, str]] | None = None,
) -> tuple[dict[str, str], list[str], str]:
    contents, green = _validated_tail_contents(tail, greenfact)
    current: dict[str, str] = {}
    volatile: list[str] = []
    tool_guidance: dict[str, str] = {}
    for content in contents:
        if any(marker in content for marker in _GREEN_CLAIM_MARKERS):
            raise PageLogError("green claim cannot be routed into the page log")
        compact_guidance, remainder = _compact_tool_guidance(content)
        if compact_guidance:
            tool_guidance.update(compact_guidance)
            if not remainder.strip():
                continue
            content = remainder
        elif any(line.startswith("[tool guidance]") for line in content.splitlines()):
            # Unknown guidance stays on the volatile tail instead of being mis-keyed.
            volatile.append(content)
            continue
        _record_logical_page(
            content, current, volatile, window_key_sources=window_key_sources,
        )
    if tool_guidance:
        volatile.append("\n".join([
            "[tool guidance]",
            *(f"{tool}: {description}" for tool, description in sorted(tool_guidance.items())),
        ]))
    return current, volatile, green






def _frame_projection(body: dict[str, object], previous: object | None, *, require_product: bool):
    from kernel_stateframe_frame_append import project_request
    return project_request(
        body, previous,
        interval=_MAX_FRAME_INTERVAL,
        byte_bound=_MAX_DELTA_BYTES,
        require_product=require_product,
        page_log_layout=True,
    )


_AUTHORITY_FIELDS = frozenset({
    "done_permission", "public_checks", "submission", "completion", "edit_authority",
    "established_facts",
})
_AUTHORITY_TOKENS = frozenset({"stop", "verdict", "menu", "authority"})


def _authority_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized == "owner_protocol":
        # Keep its stable schema identifier in [L]; dynamic owner values live in [V].
        return False
    return normalized in _AUTHORITY_FIELDS or bool(_AUTHORITY_TOKENS.intersection(normalized.split("_")))


def _is_recorded_wire_truth(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("source") == "recorded_wire_truth"


def _contains_recorded_wire_truth(value: object) -> bool:
    if _is_recorded_wire_truth(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_recorded_wire_truth(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_recorded_wire_truth(child) for child in value)
    return False


def _partition_authority(value: object, path: tuple[str | int, ...] = ()) -> tuple[object, list[dict[str, object]]]:
    """Remove authority-shaped keys from cacheable state and retain their exact paths in V."""
    if _is_recorded_wire_truth(value):
        if not path:
            raise PageLogError("recorded wire-truth facts must be nested in the StateFrame payload")
        return {}, [{"path": list(path), "value": copy.deepcopy(dict(value))}]
    if isinstance(value, Mapping):
        cacheable: dict[str, object] = {}
        overlay: list[dict[str, object]] = []
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise PageLogError("StateFrame authority partition requires string object keys")
            if raw_key == "owner_protocol":
                if (isinstance(child, Mapping)
                        and child.get("schema") == "stateframe-framefit5-owner/1"):
                    cacheable[raw_key] = {"schema": child["schema"]}
                    for protocol_key, protocol_value in child.items():
                        if protocol_key == "schema":
                            continue
                        if not isinstance(protocol_key, str):
                            raise PageLogError("owner protocol keys must be strings")
                        overlay.append({
                            "path": [*path, raw_key, protocol_key],
                            "value": copy.deepcopy(protocol_value),
                        })
                else:
                    overlay.append({"path": [*path, raw_key], "value": copy.deepcopy(child)})
                continue
            if _authority_key(raw_key):
                overlay.append({"path": [*path, raw_key], "value": copy.deepcopy(child)})
                continue
            filtered, nested = _partition_authority(child, (*path, raw_key))
            cacheable[raw_key] = filtered
            overlay.extend(nested)
        return cacheable, overlay
    if isinstance(value, list):
        cacheable_list: list[object] = []
        overlay: list[dict[str, object]] = []
        for index, child in enumerate(value):
            filtered, nested = _partition_authority(child, (*path, index))
            cacheable_list.append(filtered)
            overlay.extend(nested)
        return cacheable_list, overlay
    return copy.deepcopy(value), []


def _restore_authority(cacheable: object, overlay: Sequence[Mapping[str, object]]) -> object:
    restored = copy.deepcopy(cacheable)
    for row in overlay:
        path = row.get("path")
        if not isinstance(path, list) or not path:
            raise PageLogError("StateFrame authority overlay has an invalid path")
        current = restored
        for part in path[:-1]:
            if isinstance(current, dict) and isinstance(part, str) and part in current:
                current = current[part]
            elif isinstance(current, list) and type(part) is int and 0 <= part < len(current):
                current = current[part]
            else:
                raise PageLogError("StateFrame authority overlay path cannot be restored")
        final = path[-1]
        if isinstance(current, dict) and isinstance(final, str):
            current[final] = copy.deepcopy(row.get("value"))
        elif isinstance(current, list) and type(final) is int and 0 <= final < len(current):
            current[final] = copy.deepcopy(row.get("value"))
        else:
            raise PageLogError("StateFrame authority overlay target cannot be restored")
    return restored


def _authority_content(fields: Sequence[Mapping[str, object]]) -> str:
    content = AUTHORITY_OVERLAY_HEADER + "\n" + json.dumps(
        {"schema": "stateframe-authority-overlay/1", "fields": list(fields)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return content


def _payload_has_authority_keys(value: object) -> bool:
    if _is_recorded_wire_truth(value):
        return True
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return True
            if key == "owner_protocol":
                if (not isinstance(child, Mapping)
                        or set(child) != {"schema"}
                        or child.get("schema") != "stateframe-framefit5-owner/1"):
                    return True
                continue
            if _authority_key(key) or _payload_has_authority_keys(child):
                return True
        return False
    if isinstance(value, list):
        return any(_payload_has_authority_keys(child) for child in value)
    return False


def _authority_path(parts: Sequence[object], value: object = None) -> bool:
    if _contains_recorded_wire_truth(value):
        return True
    for index, part in enumerate(parts):
        if isinstance(part, str) and _authority_key(part):
            return True
        if part == "owner_protocol" and any(value != "schema" for value in parts[index + 1:]):
            return True
    return False


def _body_with_payload(body: dict[str, object], payload: Mapping[str, object]) -> dict[str, object]:
    from kernel_stateframe_frame_append import STATE_HEADER
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise PageLogError("StateFrame authority partition has no message array")
    indexes = [
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith(STATE_HEADER + "\n")
    ]
    if len(indexes) != 1:
        raise PageLogError("StateFrame authority partition requires exactly one frame")
    output = copy.deepcopy(body)
    output_messages = [copy.deepcopy(dict(message)) for message in messages if isinstance(message, Mapping)]
    if len(output_messages) != len(messages):
        raise PageLogError("StateFrame authority partition found a malformed message")
    output_messages[indexes[0]]["content"] = STATE_HEADER + "\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    output["messages"] = output_messages
    return output


def _tail_after_frame(projection) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    messages = projection.body.get("messages")
    if not isinstance(messages, list):
        raise PageLogError("frame projection has no message array")
    try:
        layout = validate_stateframe_message_order(
            messages,
            require_task_label=False,
            allow_user_context_before_stateframe=True,
        )
    except StateFrameMessageLayoutError as error:
        raise PageLogError(
            "frame projection violates the shared StateFrame message order",
        ) from error
    frame_index = layout.state_index
    deltas = list(projection.state.delta_messages)
    tail_start = frame_index + 1 + len(deltas)
    if tail_start > len(messages):
        raise PageLogError("frame projection delta boundary exceeds messages")
    prefix = [copy.deepcopy(dict(message)) for message in messages[:frame_index]
              if isinstance(message, Mapping)]
    if len(prefix) != frame_index:
        raise PageLogError("stable prefix contains a malformed message")
    tail = [copy.deepcopy(dict(message)) for message in messages[tail_start:]
            if isinstance(message, Mapping)]
    if len(tail) != len(messages) - tail_start:
        raise PageLogError("volatile tail contains a malformed message")
    return prefix, tail


def _frame_log_messages(frame_state: object, previous: PageLogState | None,
                        *, rebase: bool) -> list[dict[str, object]]:
    base = copy.deepcopy(frame_state.base_message)
    all_deltas = [copy.deepcopy(message) for message in frame_state.delta_messages]
    if rebase or previous is None:
        return [base, *all_deltas]
    old_count = len(previous.cache_frame_state.delta_messages)
    if old_count > len(all_deltas):
        raise PageLogError("typed StateFrame delta chain shortened without rebase")
    appended = all_deltas[old_count:]
    return [copy.deepcopy(message) for message in previous.log] + appended


def _sensitive_page_change(
        previous: PageLogState | None,
        current: Mapping[str, str],
        *,
        supersede_source_pages: bool = False,
        no_rebase_page_retirements: bool = False,
) -> bool:
    if previous is None:
        return False
    before = {key: content for key, (_version, content) in _page_map(previous).items()}
    for key in set(before) | set(current):
        changed = before.get(key) != current.get(key)
        existing_sensitive = (
            changed
            and (key.startswith("head:") or key == "section:import_graph_fact")
        )
        source_change = not supersede_source_pages and key.startswith("src:") and changed
        if existing_sensitive or source_change:
            return True
    return False


def _reason_after_sensitive_page_change(
        reason: str | None, sensitive_page_changed: bool,
) -> str | None:
    if not sensitive_page_changed:
        return reason
    if reason == "tool_surface_changed":
        from kernel_stateframe_tool_stable import menu_rebase_mode
        if menu_rebase_mode() == "on":
            return reason
    return "correctness_sensitive_page_changed"


def _exact_text_tokens(meter: object, content: str) -> int:
    counter = getattr(meter, "count_text_exact", None)
    if not callable(counter):
        raise PageLogError("exact tokenizer has no text-count API")
    value = counter(content)
    if type(value) is not int or value < 0:
        raise PageLogError("exact tokenizer text count is unavailable")
    return value


def _message_content(message: Mapping[str, object]) -> str:
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _token_counts(meter: object, frame_state: object,
                  pages: Mapping[str, tuple[int, str]],
                  stale_contents: Sequence[str]) -> tuple[int, int]:
    live_parts = [
        _message_content(message)
        for message in (frame_state.base_message, *frame_state.delta_messages)
    ]
    live_parts.extend(
        _message_content(_page_message(key, version, content))
        for key, (version, content) in pages.items()
    )
    live = _exact_text_tokens(meter, "\n".join(live_parts))
    stale = _exact_text_tokens(meter, "\n".join(stale_contents)) if stale_contents else 0
    return live, stale


def _request_budget(meter: object, body: Mapping[str, object], *, context_tokens: int) -> tuple[int, int]:
    measured = getattr(meter, "measure", None)
    if not callable(measured):
        raise PageLogError("request token meter is unavailable")
    tokens, method = measured(body)
    if method != "tokenize" or type(tokens) is not int or tokens < 0:
        raise PageLogError("exact request tokenizer is unavailable")
    from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET
    reserve = max(0, context_tokens - ENDPOINT_PROMPT_TOKEN_BUDGET)
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if type(value) is int and value > 0:
            reserve = max(reserve, value)
    reasoning = body.get("reasoning_budget_tokens")
    if type(reasoning) is int and reasoning > 0:
        declared_output = max(
            (body.get(key, 0) for key in ("max_tokens", "max_completion_tokens")
             if type(body.get(key, 0)) is int and body.get(key, 0) > 0),
            default=0,
        )
        reserve = max(reserve, reasoning + declared_output)
    return tokens, reserve


def _prefix_parts(prefix: Sequence[Mapping[str, object]]) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    system_end = next((index for index, message in enumerate(prefix)
                       if message.get("role") != "system"), len(prefix))
    return list(prefix[:system_end]), list(prefix[system_end:])


def _build_output(
        body: dict[str, object],
        prefix: Sequence[Mapping[str, object]],
        log: Sequence[Mapping[str, object]],
        pages: Mapping[str, tuple[int, str]],
        volatile: Sequence[str],
        authority: str,
        greenfact: str,
) -> dict[str, object]:
    out = copy.deepcopy(body)
    out["messages"] = [*copy.deepcopy(list(prefix)), *copy.deepcopy(list(log))]
    tail_start = len(out["messages"])
    if greenfact:
        # Recreate the validated suffix in the composer input; PAGE_LOG has
        # already removed it from the volatile content while routing the tail.
        out["messages"].append({"role": "user", "content": greenfact})
    additions = [*volatile]
    page_index = _page_index_content(pages)
    if authority:
        additions.append(authority)
    if page_index:
        additions.append(page_index)
    # The exchange starts [V]; the shared composer keeps GREENFACT as its exact suffix.
    try:
        return compose_final_user_message(
            out, tail_start=tail_start, additions=additions,
            greenfact_text=greenfact,
            final_message_name=PAGE_LOG_FINAL_MESSAGE_NAME if authority else None,
        )
    except ValueError as error:
        raise PageLogError("shared final-message composition failed") from error


def _epoch_log(frame_projection, current_pages: Mapping[str, str], previous: PageLogState | None,
               versions: dict[str, int]) -> tuple[list[dict[str, object]], dict[str, tuple[int, str]]]:
    log = _frame_log_messages(frame_projection.state, previous, rebase=True)
    pages: dict[str, tuple[int, str]] = {}
    old_pages = _page_map(previous)
    for key, content in current_pages.items():
        if old_pages.get(key, (None, None))[1] == content:
            version = versions.get(key, old_pages[key][0])
        else:
            version = versions.get(key, 0) + 1
        versions[key] = version
        pages[key] = (version, content)
        log.append(_page_message(key, version, content))
    return log, pages


def _project_request_once(
        body: dict[str, object],
        previous: PageLogState | None = None,
        *,
        token_meter: object | None,
        selected_mode: str = "on",
        require_product: bool = True,
        context_tokens: int | None = None,
        measure_token_policy: bool = True,
        run_id: str = "",
        runtime: object | None = None,
        request_id: str | None = None,
        supersede_source_pages: bool = False,
        no_rebase_page_retirements: bool = False,
        window_key_sources: bool = False,
        _exchange_messages: Sequence[Mapping[str, object]] = (),
        token_input_body: Mapping[str, object] | None = None,
        budget_mode: str = "off",
        host_prompt_cap: int | None = None,
) -> PageLogProjection:
    """Project one request and assert exact frame/page reconstruction before return.

    `measure_token_policy=False` is only for labeled historical byte-prefix replay;
    the production capture path always requires exact llama.cpp tokenization.
    """
    if selected_mode not in _VALID_MODES - {"off"}:
        raise PageLogError("PAGE_LOG projection requires shadow or on mode")
    if measure_token_policy and token_meter is None:
        raise PageLogError("exact request token meter is required")
    from kernel_stateframe_frame_append import materialize_payload, _same
    from kernel_greenfact import GREENFACT_TEXT

    if measure_token_policy:
        measured = getattr(token_meter, "measure", None)
        if not callable(measured):
            raise PageLogError("request token meter is unavailable")
        _initial_tokens, initial_method = measured(body if token_input_body is None else token_input_body)
        if initial_method != "tokenize":
            raise PageLogError("exact request tokenizer is unavailable")
        if context_tokens is None and budget_mode != "on":
            raise PageLogError("exact per-slot context capacity is unavailable")

    prior = previous if isinstance(previous, PageLogState) and previous.run_id == run_id else None

    def project_pair(full_previous: object | None, cache_previous: object | None):
        full = _frame_projection(body, full_previous, require_product=require_product)
        payload = full.state.current_payload
        cacheable, authority_fields = _partition_authority(payload)
        if not isinstance(cacheable, Mapping):
            raise PageLogError("partitioned StateFrame payload is not an object")
        cache_body = _body_with_payload(body, cacheable)
        cached = _frame_projection(cache_body, cache_previous, require_product=False)
        if not _same(cached.state.current_payload, cacheable):
            raise PageLogError("cacheable StateFrame projection changed its partition")
        authority = _authority_content(authority_fields) if authority_fields else ""
        return full, cached, authority_fields, authority

    frame_projection, cache_projection, authority_fields, authority = project_pair(
        prior.frame_state if prior is not None else None,
        prior.cache_frame_state if prior is not None else None,
    )
    prefix, raw_tail = _tail_after_frame(cache_projection)
    current_pages, volatile, green = _logical_page_map(
        raw_tail, GREENFACT_TEXT, window_key_sources=window_key_sources,
        previous_pages=_page_map(prior),
    )

    greenfact_revisions = list(prior.greenfact_revisions) if prior is not None else []
    if green and measure_token_policy:
        from kernel_greenfact import _green_revision
        revision, unavailable = _green_revision(runtime)
        if unavailable is not None or not isinstance(revision, str) or not revision:
            raise PageLogError("GREENFACT suffix lacks a current host revision splice")
        # The current host owner deduplicates C4c splices by request/revision.
        # Legacy callers without an ID still use its revision-only receipt.
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id:
                raise PageLogError("GREENFACT request identity is invalid")
            splice_key = f"{revision}\0{request_id}"
            rendered = getattr(runtime, "_greenfact_rendered_request_revisions", None)
        else:
            splice_key = revision
            rendered = getattr(runtime, "_greenfact_rendered_revisions", None)
        if not isinstance(rendered, set) or splice_key not in rendered:
            raise PageLogError("GREENFACT suffix lacks a current host revision splice")
        if splice_key in greenfact_revisions:
            raise PageLogError("GREENFACT suffix was already rendered for this request/revision")
        greenfact_revisions.append(splice_key)

    reason = (
        cache_projection.rebase_reason or "cache_state_rebase"
        if prior is not None and cache_projection.action == "rebase"
        else None
    )
    sensitive_page_changed = _sensitive_page_change(
        prior, current_pages,
        supersede_source_pages=supersede_source_pages,
        no_rebase_page_retirements=no_rebase_page_retirements,
    )
    if prior is not None:
        # C4c carries module heads in the typed frame, as well as literal page
        # variants. Do not leave an obsolete head in a base plus correcting delta.
        before_frame = prior.frame_state.current_payload.get("state", {}).get("framefit5", {})
        current_frame = frame_projection.state.current_payload.get("state", {}).get("framefit5", {})
        sensitive_page_changed = sensitive_page_changed or (
            before_frame.get("headcarry") != current_frame.get("headcarry"))
    reason = _reason_after_sensitive_page_change(reason, sensitive_page_changed)
    versions = _version_map(prior)
    if prior is None or reason is not None:
        if reason is not None:
            frame_projection, cache_projection, authority_fields, authority = project_pair(None, None)
            prefix, raw_tail = _tail_after_frame(cache_projection)
            current_pages, volatile, green = _logical_page_map(
        raw_tail, GREENFACT_TEXT, window_key_sources=window_key_sources,
        previous_pages=_page_map(prior),
    )
        log, pages = _epoch_log(cache_projection, current_pages, prior, versions)
        stale_contents: list[str] = []
        stale_tokens = 0
        action = "base" if prior is None and reason is None else "rebase"
        epoch = (prior.epoch + 1) if prior is not None else 1
        epoch_turns = 1
    else:
        log = _frame_log_messages(cache_projection.state, prior, rebase=False)
        pages = _page_map(prior)
        stale_contents = list(prior.stale_contents)
        stale_tokens = prior.stale_tokens
        for key, (_version, content) in list(pages.items()):
            if key in current_pages:
                continue
            old_message = _page_message(key, _version, content)
            retired_message = _retired_message(key)
            if measure_token_policy:
                stale_contents.extend((_message_content(old_message), _message_content(retired_message)))
            log.append(retired_message)
            del pages[key]
        for key, content in current_pages.items():
            current = pages.get(key)
            if current is not None and current[1] == content:
                continue
            if current is not None and measure_token_policy:
                stale_contents.append(_message_content(_page_message(key, current[0], current[1])))
            version = versions.get(key, 0) + 1
            versions[key] = version
            pages[key] = (version, content)
            log.append(_page_message(key, version, content))
        action = "delta"
        epoch = prior.epoch
        epoch_turns = prior.epoch_turns + 1

    live_tokens: int | None = None
    stale_count: int | None = stale_tokens if measure_token_policy else None
    if measure_token_policy:
        live_tokens, stale_count = _token_counts(
            token_meter, cache_projection.state, pages, stale_contents,
        )
        stale_tokens = stale_count
        if prior is not None and action != "rebase" and stale_tokens > 2 * live_tokens:
            reason = "stale_tokens_exceed_2x_live"
    if reason is not None and action != "rebase":
        frame_projection, cache_projection, authority_fields, authority = project_pair(None, None)
        prefix, raw_tail = _tail_after_frame(cache_projection)
        current_pages, volatile, green = _logical_page_map(
        raw_tail, GREENFACT_TEXT, window_key_sources=window_key_sources,
        previous_pages=_page_map(prior),
    )
        log, pages = _epoch_log(cache_projection, current_pages, prior, versions)
        stale_contents = []
        stale_tokens = 0
        action = "rebase"
        epoch = (prior.epoch + 1) if prior is not None else 1
        epoch_turns = 1
        if measure_token_policy:
            live_tokens, stale_count = _token_counts(
                token_meter, cache_projection.state, pages, stale_contents,
            )
            stale_tokens = stale_count

    projected = _build_output(body, prefix, log, pages, volatile, authority, green)
    if _exchange_messages:
        projected = _insert_exchange_before_final_user(projected, _exchange_messages)
    prompt_tokens: int | None = None
    budget_action: str | None = None
    selected_budget_mode = budget_mode if budget_mode in _VALID_MODES else "off"
    selected_host_cap = host_prompt_cap
    if measure_token_policy:
        context = context_tokens
        if selected_budget_mode != "on" and (type(context) is not int or context <= 0):
            raise PageLogError("per-slot context capacity is invalid")
        prompt_tokens, completion_reserve = _request_budget(
            token_meter, projected, context_tokens=context or 0,
        )
        if selected_budget_mode == "on":
            if type(selected_host_cap) is not int or selected_host_cap <= 0:
                raise PageLogError("host prompt-token budget is unavailable")
            should_rebase_for_budget = prompt_tokens > selected_host_cap
        else:
            should_rebase_for_budget = (
                prompt_tokens + completion_reserve > int(context * 0.85)
            )
        if should_rebase_for_budget and action != "rebase":
            reason = (
                "projected_prompt_over_host_budget"
                if selected_budget_mode == "on"
                else "projected_prompt_over_85_percent_context"
            )
            frame_projection, cache_projection, authority_fields, authority = project_pair(None, None)
            prefix, raw_tail = _tail_after_frame(cache_projection)
            current_pages, volatile, green = _logical_page_map(
        raw_tail, GREENFACT_TEXT, window_key_sources=window_key_sources,
        previous_pages=_page_map(prior),
    )
            log, pages = _epoch_log(cache_projection, current_pages, prior, versions)
            stale_contents = []
            stale_tokens = 0
            action = "rebase"
            if selected_budget_mode == "on":
                budget_action = "rebase"
            epoch = (prior.epoch + 1) if prior is not None else 1
            epoch_turns = 1
            projected = _build_output(body, prefix, log, pages, volatile, authority, green)
            if _exchange_messages:
                projected = _insert_exchange_before_final_user(projected, _exchange_messages)
            live_tokens, stale_count = _token_counts(
                token_meter, cache_projection.state, pages, stale_contents,
            )
            stale_tokens = stale_count
            prompt_tokens, completion_reserve = _request_budget(
                token_meter, projected, context_tokens=context or 0,
            )
        if selected_budget_mode == "on":
            if prompt_tokens > selected_host_cap:
                raise PageLogBudgetError(prompt_tokens, selected_host_cap)
        elif prompt_tokens + completion_reserve > int(context * 0.85):
            raise PageLogError(_CONTEXT_POLICY_ERROR)

    payload = frame_projection.state.current_payload
    materialized = materialize_payload(
        frame_projection.state.base_message, frame_projection.state.delta_messages,
    )
    if not _same(materialized, payload):
        raise PageLogError("canonical StateFrame base and deltas do not reconstruct current payload")
    cache_payload = cache_projection.state.current_payload
    cache_materialized = materialize_payload(
        cache_projection.state.base_message, cache_projection.state.delta_messages,
    )
    if not _same(cache_materialized, cache_payload):
        raise PageLogError("cacheable StateFrame base and deltas do not reconstruct current payload")
    if _payload_has_authority_keys(cache_projection.state.base_payload) or _payload_has_authority_keys(cache_payload):
        raise PageLogError("authority-shaped field survived in cacheable StateFrame")
    for delta in cache_projection.state.delta_messages:
        content = delta.get("content")
        try:
            delta_payload = json.loads(str(content).split("\n", 1)[1])
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise PageLogError("cacheable StateFrame delta is malformed") from error
        sections = delta_payload.get("sections") if isinstance(delta_payload, Mapping) else None
        if not isinstance(sections, list) or any(
                isinstance(section, Mapping)
                and isinstance(section.get("path"), list)
                and _authority_path(section["path"], section.get("value"))
                for section in sections):
            raise PageLogError("authority-shaped path survived in cacheable StateFrame delta")
    restored = _restore_authority(cache_materialized, authority_fields)
    if not _same(restored, payload):
        raise PageLogError("StateFrame [L] plus current [V] authority does not reconstruct canonical state")
    replayed = _replay_page_log(log)
    expected_pages = {key: content for key, (_version, content) in pages.items()}
    if replayed != expected_pages:
        raise PageLogError("page log does not reconstruct the current live page map")
    if green:
        final_messages = projected.get("messages")
        final = final_messages[-1] if isinstance(final_messages, list) and final_messages else None
        if not isinstance(final, Mapping) or not str(final.get("content") or "").endswith(GREENFACT_TEXT):
            raise PageLogError("final V message does not end with the exact GREENFACT suffix")
        if sum(str(message.get("content") or "").count(GREENFACT_TEXT)
               for message in final_messages if isinstance(message, Mapping)) != 1:
            raise PageLogError("GREENFACT must appear exactly once in the final request")
    for message in log:
        content = str(message.get("content") or "")
        if GREENFACT_TEXT in content or any(marker in content for marker in _GREEN_CLAIM_MARKERS):
            raise PageLogError("GREENFACT claim survived in the append-only log")
    frame_contents = [
        _message_content(message)
        for message in (cache_projection.state.base_message, *cache_projection.state.delta_messages)
    ]
    if any(GREENFACT_TEXT in content for content in frame_contents):
        raise PageLogError("GREENFACT suffix cannot be stored in StateFrame base or deltas")

    state = PageLogState(
        run_id=run_id,
        frame_state=copy.deepcopy(frame_projection.state),
        cache_frame_state=copy.deepcopy(cache_projection.state),
        log=tuple(copy.deepcopy(message) for message in log),
        pages=tuple((key, version, content) for key, (version, content) in pages.items()),
        versions=tuple(versions.items()),
        stale_contents=tuple(stale_contents),
        stale_tokens=stale_tokens,
        greenfact_revisions=tuple(greenfact_revisions),
        generation=(prior.generation + 1) if prior is not None else 1,
        turn_index=(prior.turn_index + 1) if prior is not None else 0,
        epoch=epoch,
        epoch_turns=epoch_turns,
    )
    prefix_preserved = False
    if prior is not None and action != "rebase":
        old_bytes = prefix_bytes_from_parts(prefix, prior.log, body.get("tools"))
        new_bytes = prefix_bytes_from_parts(prefix, log, body.get("tools"))
        prefix_preserved = new_bytes.startswith(old_bytes)
        if not prefix_preserved:
            raise PageLogError("append-only page prefix changed without a journaled rebase")
    return PageLogProjection(
        body=projected,
        state=state,
        previous_state=prior,
        action=action,
        rebase_reason=reason,
        prefix_preserved=prefix_preserved,
        prompt_tokens=prompt_tokens,
        live_tokens=live_tokens,
        stale_tokens=stale_count,
        budget_action=budget_action,
        budget_cap=selected_host_cap if selected_budget_mode == "on" else None,
    )




def _source_shadow_mismatch_fields(
        legacy: PageLogProjection,
        candidate: PageLogProjection,
) -> tuple[str, ...]:
    comparisons = (
        ("action", legacy.action, candidate.action),
        ("rebase_reason", legacy.rebase_reason, candidate.rebase_reason),
        ("pages", legacy.state.pages, candidate.state.pages),
        ("log", legacy.state.log, candidate.state.log),
    )
    return tuple(field for field, old, new in comparisons if old != new)


def _retirement_shadow_mismatch_fields(
        legacy: PageLogProjection,
        candidate: PageLogProjection,
) -> tuple[str, ...]:
    comparisons = (
        ("action", legacy.action, candidate.action),
        ("rebase_reason", legacy.rebase_reason, candidate.rebase_reason),
        ("body", legacy.body, candidate.body),
        ("pages", legacy.state.pages, candidate.state.pages),
        ("log", legacy.state.log, candidate.state.log),
        ("epoch", legacy.state.epoch, candidate.state.epoch),
        ("epoch_turns", legacy.state.epoch_turns, candidate.state.epoch_turns),
        ("stale_tokens", legacy.state.stale_tokens, candidate.state.stale_tokens),
        ("prefix_preserved", legacy.prefix_preserved, candidate.prefix_preserved),
        ("prompt_tokens", legacy.prompt_tokens, candidate.prompt_tokens),
        ("live_tokens", legacy.live_tokens, candidate.live_tokens),
        ("measured_stale_tokens", legacy.stale_tokens, candidate.stale_tokens),
    )
    return tuple(field for field, old, new in comparisons if old != new)


def _project_request_with_policies(
        body: dict[str, object],
        previous: PageLogState | None = None,
        *,
        token_meter: object | None,
        selected_mode: str = "on",
        require_product: bool = True,
        context_tokens: int | None = None,
        measure_token_policy: bool = True,
        run_id: str = "",
        runtime: object | None = None,
        source_mode: str | None = None,
        retirement_mode: str | None = None,
        window_mode: str | None = None,
        _exchange_messages: Sequence[Mapping[str, object]] = (),
        token_input_body: Mapping[str, object] | None = None,
        budget_mode: str = "off",
        host_prompt_cap: int | None = None,
) -> PageLogProjection:
    """Compose independent source, retirement, and window-key policies with fail-open fallback."""
    selected_source_mode = source_supersession_mode() if source_mode is None else str(source_mode).strip().lower()
    if selected_source_mode not in _VALID_MODES:
        selected_source_mode = "off"
    selected_retirement_mode = (
        page_retirement_mode() if retirement_mode is None
        else str(retirement_mode).strip().lower()
    )
    if selected_retirement_mode not in _VALID_MODES:
        selected_retirement_mode = "off"
    selected_window_mode = page_log_window_key_mode() if window_mode is None else str(window_mode).strip().lower()
    if selected_window_mode not in _VALID_MODES:
        selected_window_mode = "off"

    def project(
            *, supersede_sources: bool, no_rebase_retirements: bool,
            window_key_sources: bool,
    ) -> PageLogProjection:
        return _project_request_once(
            body, previous,
            token_meter=token_meter,
            selected_mode=selected_mode,
            require_product=require_product,
            context_tokens=context_tokens,
            measure_token_policy=measure_token_policy,
            run_id=run_id,
            runtime=runtime,
            supersede_source_pages=supersede_sources,
            no_rebase_page_retirements=no_rebase_retirements,
            window_key_sources=window_key_sources,
            _exchange_messages=_exchange_messages,
            token_input_body=token_input_body,
            budget_mode=budget_mode,
            host_prompt_cap=host_prompt_cap,
        )

    def project_with_source_policy(
            *, no_rebase_retirements: bool, window_key_sources: bool,
    ) -> PageLogProjection:
        if selected_source_mode == "off":
            return project(
                supersede_sources=False, no_rebase_retirements=no_rebase_retirements,
                window_key_sources=window_key_sources,
            )
        if selected_source_mode == "on":
            try:
                return project(
                    supersede_sources=True, no_rebase_retirements=no_rebase_retirements,
                    window_key_sources=window_key_sources,
                )
            except Exception as error:
                legacy = project(
                    supersede_sources=False, no_rebase_retirements=no_rebase_retirements,
                    window_key_sources=window_key_sources,
                )
                return replace(legacy, source_supersession_error_type=_safe_journal_error_type(error))

        legacy = project(
            supersede_sources=False, no_rebase_retirements=no_rebase_retirements,
            window_key_sources=window_key_sources,
        )
        try:
            candidate = project(
                supersede_sources=True, no_rebase_retirements=no_rebase_retirements,
                window_key_sources=window_key_sources,
            )
        except Exception as error:
            return replace(legacy, source_supersession_error_type=_safe_journal_error_type(error))
        return replace(legacy, source_shadow_mismatches=_source_shadow_mismatch_fields(legacy, candidate))

    def project_with_retirement_policy(*, window_key_sources: bool) -> PageLogProjection:
        if selected_retirement_mode == "off":
            return project_with_source_policy(
                no_rebase_retirements=False, window_key_sources=window_key_sources,
            )
        if selected_retirement_mode == "on":
            try:
                return project_with_source_policy(
                    no_rebase_retirements=True, window_key_sources=window_key_sources,
                )
            except Exception as error:
                legacy = project_with_source_policy(
                    no_rebase_retirements=False, window_key_sources=window_key_sources,
                )
                return replace(legacy, retirement_error_type=_safe_journal_error_type(error))

        legacy = project_with_source_policy(
            no_rebase_retirements=False, window_key_sources=window_key_sources,
        )
        try:
            candidate = project_with_source_policy(
                no_rebase_retirements=True, window_key_sources=window_key_sources,
            )
        except Exception as error:
            return replace(legacy, retirement_error_type=_safe_journal_error_type(error))
        return replace(
            legacy,
            retirement_shadow_mismatches=_retirement_shadow_mismatch_fields(legacy, candidate),
        )

    if selected_window_mode == "off":
        return project_with_retirement_policy(window_key_sources=False)
    if selected_window_mode == "on":
        try:
            return project_with_retirement_policy(window_key_sources=True)
        except Exception as error:
            legacy = project_with_retirement_policy(window_key_sources=False)
            return replace(legacy, window_key_error_type=_safe_journal_error_type(error))

    legacy = project_with_retirement_policy(window_key_sources=False)
    try:
        candidate = project_with_retirement_policy(window_key_sources=True)
    except Exception as error:
        return replace(legacy, window_key_error_type=_safe_journal_error_type(error))
    return replace(
        legacy,
        window_key_shadow_mismatches=_retirement_shadow_mismatch_fields(legacy, candidate),
    )


_PageLogRunner = Callable[
    [
        dict[str, object],
        Sequence[Mapping[str, object]],
        Mapping[str, object] | None,
    ],
    PageLogProjection,
]


def _exchange_tool_call_ids(exchange: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    calls = exchange[0].get("tool_calls") if exchange else None
    if not isinstance(calls, list):
        return ()
    return tuple(
        call["id"] for call in calls
        if isinstance(call, Mapping) and isinstance(call.get("id"), str)
    )


def _project_exchange_candidate(
        run: _PageLogRunner,
        candidate_body: dict[str, object],
        original_body: Mapping[str, object],
        exchange_messages: Sequence[Mapping[str, object]],
        call_ids: tuple[str, ...],
) -> PageLogProjection:
    try:
        projection = run(candidate_body, exchange_messages, original_body)
    except Exception as error:
        try:
            fallback = run(dict(original_body), (), None)
        except Exception:
            raise error
        return replace(
            fallback,
            exchange_tool_call_ids=call_ids,
            exchange_error_type=_safe_journal_error_type(error),
        )
    return replace(
        projection,
        exchange_lifted=True,
        exchange_tool_call_ids=call_ids,
    )


def project_request(
        body: dict[str, object],
        previous: PageLogState | None = None,
        *,
        token_meter: object | None,
        selected_mode: str = "on",
        require_product: bool = True,
        context_tokens: int | None = None,
        measure_token_policy: bool = True,
        run_id: str = "",
        runtime: object | None = None,
        source_mode: str | None = None,
        retirement_mode: str | None = None,
        exchange_mode: str | None = None,
        window_mode: str | None = None,
        budget_mode: str = "off",
        host_prompt_cap: int | None = None,
) -> PageLogProjection:
    """Project PageLog and preserve the newest exchange in native provider messages."""
    text_requested = (
        _page_log_exchange_text_requested()
        if exchange_mode is None
        else str(exchange_mode).strip().lower() == "text"
    )
    if text_requested:
        _emit_exchange_text_refused(getattr(runtime, "journal", None), runtime)

    def run(
            source_body: dict[str, object],
            exchange_messages: Sequence[Mapping[str, object]] = (),
            original_body: Mapping[str, object] | None = None,
    ) -> PageLogProjection:
        projection = _project_request_with_policies(
            source_body, previous,
            token_meter=token_meter,
            selected_mode=selected_mode,
            require_product=require_product,
            context_tokens=context_tokens,
            measure_token_policy=measure_token_policy,
            run_id=run_id,
            runtime=runtime,
            source_mode=source_mode,
            retirement_mode=retirement_mode,
            window_mode=window_mode,
            _exchange_messages=exchange_messages,
            token_input_body=original_body,
            budget_mode=budget_mode,
            host_prompt_cap=host_prompt_cap,
        )
        if selected_mode == "on":
            try:
                validate_stateframe_message_order(
                    projection.body.get("messages"),
                    require_task_label=require_product,
                    allow_native_exchange=bool(exchange_messages),
                    allow_user_context_before_stateframe=True,
                )
            except StateFrameMessageLayoutError as error:
                raise PageLogError(
                    "projected request violates the shared StateFrame message order",
                ) from error
        return projection

    extracted = _newest_complete_exchange(body)
    if extracted is None:
        return run(body)
    exchange_messages = extracted[1]
    candidate_body = _body_without_exchange(body)
    call_ids = _exchange_tool_call_ids(exchange_messages)

    return _project_exchange_candidate(
        run, candidate_body, body, exchange_messages, call_ids,
    )


def prefix_bytes_from_parts(prefix: Sequence[Mapping[str, object]],
                            log: Sequence[Mapping[str, object]], tools: object) -> bytes:
    """Canonical open-ended byte stream for [S]+[L], ending at the next [V] slot."""
    systems, task_and_tail = _prefix_parts(prefix)
    parts = [*systems, {"tools": tools}, *task_and_tail, *log]
    return b"".join(_wire_json(part) + b"\n" for part in parts)


def prefix_bytes(body: Mapping[str, object]) -> bytes:
    source_body = copy.deepcopy(dict(body))
    exchange = _newest_complete_exchange(source_body)
    if exchange is not None:
        source_body = _body_without_exchange(source_body)
    messages = source_body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise PageLogError("projected request has no final V message")
    final = messages[-1]
    if not isinstance(final, Mapping) or final.get("role") != "user":
        raise PageLogError("projected request does not end with one user V message")
    state_index = next((index for index, message in enumerate(messages)
                        if isinstance(message, Mapping)
                        and str(message.get("content") or "").startswith("[StateFrame]\n")), None)
    if state_index is None:
        raise PageLogError("projected request has no StateFrame base")
    # The exact assistant/tool exchange is outside [L], immediately before final [V].
    # Removing it here keeps this metric scoped to the stable [S]+[L] byte prefix.
    prefix = [message for message in messages[:state_index] if isinstance(message, Mapping)]
    log = [message for message in messages[state_index:-1] if isinstance(message, Mapping)]
    return prefix_bytes_from_parts(prefix, log, source_body.get("tools"))


def _run_id(runtime: object, capture: object) -> str:
    for owner in (capture, runtime):
        journal = getattr(owner, "journal", None)
        value = getattr(journal, "run_id", None)
        if isinstance(value, str) and value:
            return value
    return ""


def _observe_run_vitals_event(
        journal: object, kind: str, fields: Mapping[str, object],
) -> None:
    if os.environ.get("STATEFRAME_RUN_VITALS", "off") != "on":
        return
    try:
        from kernel_run_vitals import RunVitalsAccumulator
        accumulator = getattr(journal, "_run_vitals_accumulator", None)
        if not isinstance(accumulator, RunVitalsAccumulator):
            accumulator = RunVitalsAccumulator()
            setattr(journal, "_run_vitals_accumulator", accumulator)
            setattr(journal, "_run_vitals_emitted", False)
            setattr(journal, "_run_vitals_observer_failed", False)
        accumulator.observe({"kind": kind, "request_id": fields.get("request_id")})
    except Exception:
        try:
            setattr(journal, "_run_vitals_observer_failed", True)
        except Exception:
            pass


def _emit(journal: object | None, kind: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, producer="kernel_stateframe_page_log", dispatch=False, **fields)
            if kind in {
                "stateframe_page_log_prepared",
                "stateframe_page_log_failed_open",
            }:
                _observe_run_vitals_event(journal, kind, fields)
        except Exception:
            return


def _emit_budget_event(
        journal: object | None, *, request_id: object, step: object,
        prompt_tokens: int, cap: int, action: str, mode_name: str,
        observed_action: str | None = None,
) -> None:
    fields: dict[str, object] = {
        "request_id": request_id,
        "step": step,
        "prompt_tokens": prompt_tokens,
        "cap": cap,
        "action": action,
        "mode": mode_name,
    }
    if observed_action is not None:
        fields["observed_action"] = observed_action
    _emit(journal, "stateframe_page_log_budget", **fields)


def _resolve_host_prompt_cap(
        capture: object, body: Mapping[str, object], host_prompt_cap: int | None,
) -> int:
    if host_prompt_cap is not None:
        return host_prompt_cap
    from kernel_request_budget import effective_prompt_token_budget
    contract = getattr(capture, "request_action_contract", None)
    lean_contract = (
        isinstance(contract, Mapping)
        and contract.get("contract_kind") == "lean_stable_surface"
    )
    stage = getattr(capture, "request_budget_stage", None)
    return effective_prompt_token_budget(
        body,
        stage if isinstance(stage, str) and stage else "NATIVE",
        lean_contract=lean_contract,
    )


def _emit_budget_cap_rebase(
        journal: object | None, capture: object, projection: PageLogProjection,
) -> None:
    if (projection.budget_action != "rebase"
            or projection.prompt_tokens is None or projection.budget_cap is None):
        return
    raw_step = getattr(capture, "step", None)
    _emit_budget_event(
        journal, request_id=getattr(capture, "request_id", None),
        step=raw_step if type(raw_step) is int else None,
        prompt_tokens=projection.prompt_tokens, cap=projection.budget_cap,
        action="rebase", mode_name="on",
    )


def _emit_shadow_budget_mismatch(
        journal: object | None, capture: object, cap: int,
        prompt_tokens: int, observed_action: str,
) -> None:
    raw_step = getattr(capture, "step", None)
    _emit_budget_event(
        journal, request_id=getattr(capture, "request_id", None),
        step=raw_step if type(raw_step) is int else None,
        prompt_tokens=prompt_tokens, cap=cap,
        action="shadow_mismatch", mode_name="shadow",
        observed_action=observed_action,
    )


def _shadow_candidate_after_legacy_failure(
        project_candidate: Callable[[str], PageLogProjection],
        *, legacy_error: Exception, host_prompt_cap: int,
        journal: object | None, capture: object,
) -> None:
    try:
        candidate = project_candidate("on")
    except PageLogBudgetError as candidate_error:
        if type(candidate_error) is not type(legacy_error):
            _emit_shadow_budget_mismatch(
                journal, capture, host_prompt_cap,
                candidate_error.prompt_tokens, "fail_open",
            )
    except Exception:
        return
    else:
        _emit_shadow_budget_mismatch(
            journal, capture, host_prompt_cap,
            candidate.prompt_tokens or 0,
            candidate.budget_action or candidate.action,
        )


def _shadow_candidate_after_legacy_success(
        project_candidate: Callable[[str], PageLogProjection],
        legacy: PageLogProjection, *, host_prompt_cap: int,
        journal: object | None, capture: object,
) -> None:
    try:
        candidate = project_candidate("on")
    except PageLogBudgetError as error:
        _emit_shadow_budget_mismatch(
            journal, capture, host_prompt_cap, error.prompt_tokens, "fail_open",
        )
    except Exception:
        return
    if (
        candidate.action != legacy.action
        or candidate.rebase_reason != legacy.rebase_reason
        or _wire_json(candidate.body) != _wire_json(legacy.body)
    ):
        _emit_shadow_budget_mismatch(
            journal, capture, host_prompt_cap,
            candidate.prompt_tokens or 0,
            candidate.budget_action or candidate.action,
        )


def _project_with_budget_policy(
        project_candidate: Callable[[str], PageLogProjection],
        *, budget_mode: str, host_prompt_cap: int | None,
        journal: object | None, capture: object,
) -> PageLogProjection:
    if budget_mode == "shadow":
        if type(host_prompt_cap) is not int or host_prompt_cap <= 0:
            return project_candidate("off")
        try:
            legacy = project_candidate("off")
        except Exception as legacy_error:
            _shadow_candidate_after_legacy_failure(
                project_candidate, legacy_error=legacy_error,
                host_prompt_cap=host_prompt_cap, journal=journal, capture=capture,
            )
            raise legacy_error
        _shadow_candidate_after_legacy_success(
            project_candidate, legacy, host_prompt_cap=host_prompt_cap,
            journal=journal, capture=capture,
        )
        return legacy

    projection = project_candidate(budget_mode)
    if budget_mode == "on":
        _emit_budget_cap_rebase(journal, capture, projection)
    return projection


def _emit_exchange_text_refused(journal: object | None, runtime: object | None) -> None:
    if journal is None and runtime is not None:
        journal = getattr(runtime, "journal", None)
    if not callable(getattr(journal, "emit", None)):
        return
    run_id = _run_id(runtime, None)
    if not run_id:
        journal_run_id = getattr(journal, "run_id", None)
        run_id = journal_run_id if isinstance(journal_run_id, str) else ""
    owner = runtime if runtime is not None else journal
    if run_id and getattr(owner, _EXCHANGE_TEXT_REFUSAL_ATTR, None) == run_id:
        return
    _emit(
        journal,
        "stateframe_page_log_exchange_text_refused",
        reason="retired_text_layout",
    )
    if run_id:
        try:
            setattr(owner, _EXCHANGE_TEXT_REFUSAL_ATTR, run_id)
        except (AttributeError, TypeError):
            return


def _emit_exchange_failure(
        journal: object | None,
        *,
        exchange_mode: str,
        request_id: str | None,
        step: int | None,
        reason: str,
        error_type: str | None = None,
) -> None:
    if exchange_mode not in _VALID_EXCHANGE_LAYOUTS:
        return
    fields: dict[str, object] = {
        "request_id": request_id,
        "step": step,
        "mode": exchange_mode,
        "reason": reason,
    }
    if error_type is not None:
        fields["error_type"] = error_type
    _emit(journal, "stateframe_page_log_exchange_failed_open", **fields)


def _emit_exchange_result(
        journal: object | None,
        projection: PageLogProjection,
        *,
        exchange_mode: str,
        request_id: str | None,
        step: int | None,
) -> None:
    if exchange_mode not in _VALID_EXCHANGE_LAYOUTS:
        return
    _emit(
        journal, "stateframe_page_log_exchange_evaluated",
        request_id=request_id, step=step, mode=exchange_mode,
        exchange_lifted=projection.exchange_lifted,
        tool_call_ids=list(projection.exchange_tool_call_ids[:_MAX_JOURNALED_EXCHANGE_IDS]),
        tool_call_id_count=len(projection.exchange_tool_call_ids),
        tool_call_ids_truncated=(
            len(projection.exchange_tool_call_ids) > _MAX_JOURNALED_EXCHANGE_IDS
        ),
        turn_index=projection.state.turn_index,
    )
    for field in projection.exchange_shadow_mismatches:
        _emit(
            journal, "stateframe_page_log_exchange_shadow_mismatch",
            request_id=request_id, step=step, mode=exchange_mode,
            field=field, turn_index=projection.state.turn_index,
        )
    if projection.exchange_error_type is not None:
        _emit_exchange_failure(
            journal, exchange_mode=exchange_mode,
            request_id=request_id, step=step,
            reason="candidate_projection_failed",
            error_type=projection.exchange_error_type,
        )
    if projection.exchange_shadow_error_type is not None:
        _emit_exchange_failure(
            journal, exchange_mode=exchange_mode,
            request_id=request_id, step=step,
            reason="shadow_projection_failed",
            error_type=projection.exchange_shadow_error_type,
        )


def _emit_source_supersession_result(
        journal: object | None,
        projection: PageLogProjection,
        *,
        source_mode: str,
        request_id: str | None,
        step: int | None,
) -> None:
    if source_mode == "off":
        return
    _emit(
        journal, "stateframe_page_log_src_supersede_evaluated",
        request_id=request_id, step=step, mode=source_mode,
        action=projection.action, rebase_reason=projection.rebase_reason,
        turn_index=projection.state.turn_index,
    )
    for field in projection.source_shadow_mismatches:
        _emit(
            journal, "stateframe_page_log_src_supersede_shadow_mismatch",
            request_id=request_id, step=step, mode=source_mode,
            field=field, turn_index=projection.state.turn_index,
        )
    if projection.source_supersession_error_type is not None:
        _emit_source_supersession_failure(
            journal, source_mode=source_mode, request_id=request_id, step=step,
            reason="candidate_projection_failed",
            error_type=projection.source_supersession_error_type,
        )


def _emit_source_supersession_failure(
        journal: object | None,
        *,
        source_mode: str,
        request_id: str | None,
        step: int | None,
        reason: str,
        error_type: str | None = None,
) -> None:
    if source_mode == "off":
        return
    fields: dict[str, object] = {
        "request_id": request_id, "step": step, "mode": source_mode, "reason": reason,
    }
    if error_type is not None:
        fields["error_type"] = error_type
    _emit(journal, "stateframe_page_log_src_supersede_failed_open", **fields)


def _emit_source_supersession_dependency_warning(
        journal: object | None,
        runtime: object | None,
        capture: object,
        *,
        selected_mode: str,
        selected_source_mode: str,
) -> None:
    if selected_mode != "on" or selected_source_mode != "off":
        return
    run_id = _run_id(runtime, capture)
    if run_id and getattr(runtime, _SOURCE_SUPERSESSION_WARNING_ATTR, None) == run_id:
        return
    _emit(
        journal, "stateframe_page_log_src_supersede_dependency_warning",
        request_id=(str(getattr(capture, "request_id", "")) or None),
        step=(getattr(capture, "step", None) if type(getattr(capture, "step", None)) is int else None),
        page_log_mode=selected_mode,
        source_supersession_mode=selected_source_mode,
        reason="source_supersession_disabled",
    )
    if run_id:
        try:
            setattr(runtime, _SOURCE_SUPERSESSION_WARNING_ATTR, run_id)
        except (AttributeError, TypeError):
            pass


def _emit_retirement_result(
        journal: object | None,
        projection: PageLogProjection,
        *,
        selected_mode: str,
        request_id: str | None,
        step: int | None,
) -> None:
    if selected_mode == "off":
        return
    _emit(
        journal, "stateframe_page_log_retirement_evaluated",
        request_id=request_id, step=step, mode=selected_mode,
        action=projection.action, rebase_reason=projection.rebase_reason,
        turn_index=projection.state.turn_index,
    )
    for field in projection.retirement_shadow_mismatches:
        _emit(
            journal, "stateframe_page_log_retirement_shadow_mismatch",
            request_id=request_id, step=step, mode=selected_mode,
            field=field, turn_index=projection.state.turn_index,
        )
    if projection.retirement_error_type is not None:
        _emit_retirement_failure(
            journal, selected_mode=selected_mode, request_id=request_id, step=step,
            reason="candidate_projection_failed",
            error_type=projection.retirement_error_type,
        )


def _emit_retirement_failure(
        journal: object | None,
        *,
        selected_mode: str,
        request_id: str | None,
        step: int | None,
        reason: str,
        error_type: str | None = None,
) -> None:
    if selected_mode == "off":
        return
    fields: dict[str, object] = {
        "request_id": request_id, "step": step, "mode": selected_mode, "reason": reason,
    }
    if error_type is not None:
        fields["error_type"] = error_type
    _emit(journal, "stateframe_page_log_retirement_failed_open", **fields)


def _emit_window_key_result(
        journal: object | None,
        projection: PageLogProjection,
        *,
        selected_mode: str,
        request_id: str | None,
        step: int | None,
) -> None:
    if selected_mode == "off":
        return
    _emit(
        journal, "stateframe_page_log_window_key_evaluated",
        request_id=request_id, step=step, mode=selected_mode,
        action=projection.action, rebase_reason=projection.rebase_reason,
        turn_index=projection.state.turn_index,
    )
    for field in projection.window_key_shadow_mismatches:
        _emit(
            journal, "stateframe_page_log_window_key_shadow_mismatch",
            request_id=request_id, step=step, mode=selected_mode,
            field=field, turn_index=projection.state.turn_index,
        )
    if projection.window_key_error_type is not None:
        _emit_window_key_failure(
            journal, selected_mode=selected_mode, request_id=request_id, step=step,
            reason="candidate_projection_failed",
            error_type=projection.window_key_error_type,
        )


def _emit_window_key_failure(
        journal: object | None,
        *,
        selected_mode: str,
        request_id: object,
        step: object,
        reason: str,
        error_type: str | None = None,
) -> None:
    if selected_mode == "off":
        return
    fields: dict[str, object] = {
        "request_id": request_id, "step": step, "mode": selected_mode, "reason": reason,
    }
    if error_type is not None:
        fields["error_type"] = error_type
    _emit(journal, "stateframe_page_log_window_key_failed_open", **fields)


def prepare_for_capture(
        capture: object,
        body: dict[str, object],
        runtime: object | None,
        journal: object | None,
        *,
        selected_mode: str | None = None,
        token_meter: object | None = None,
        context_tokens: int | None = None,
        host_prompt_cap: int | None = None,
) -> tuple[dict[str, object], PageLogPending | None]:
    """Prepare ON bytes or SHADOW facts without advancing run state."""
    if _page_log_exchange_text_requested():
        _emit_exchange_text_refused(journal, runtime)
    selected = mode() if selected_mode is None else selected_mode
    if selected not in _VALID_MODES:
        selected = "off"
    if selected == "off":
        return body, None
    exchange_selected = page_log_exchange_mode()
    source_selected = source_supersession_mode()
    retirement_selected = page_retirement_mode()
    window_selected = page_log_window_key_mode()
    budget_selected = page_log_cap_mode()
    _emit_source_supersession_dependency_warning(
        journal, runtime, capture, selected_mode=selected, selected_source_mode=source_selected,
    )
    if runtime is None:
        request_id = getattr(capture, "request_id", None)
        if exchange_selected in _VALID_EXCHANGE_LAYOUTS and _newest_complete_exchange(body) is not None:
            _emit_exchange_failure(
                journal, exchange_mode=exchange_selected, request_id=request_id, step=None,
                reason="runtime_unavailable",
            )
        _emit(journal, "stateframe_page_log_failed_open", request_id=request_id,
              reason="runtime_unavailable", mode=selected)
        _emit_source_supersession_failure(
            journal, source_mode=source_selected, request_id=request_id, step=None,
            reason="runtime_unavailable",
        )
        _emit_retirement_failure(
            journal, selected_mode=retirement_selected, request_id=request_id, step=None,
            reason="runtime_unavailable",
        )
        _emit_window_key_failure(
            journal, selected_mode=window_selected, request_id=request_id, step=None,
            reason="runtime_unavailable",
        )
        return body, None
    try:
        from kernel_stateframe_framefit5 import has_framefit5_state
        messages = body.get("messages")
        if (not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray))
                or not has_framefit5_state(messages)):
            return body, None
        if token_meter is None:
            from kernel_request_budget import _request_token_meter
            token_meter = _request_token_meter(runtime, capture)
        run_id = _run_id(runtime, capture)
        if not run_id:
            raise PageLogError("request run identity is unavailable")
        previous = getattr(runtime, _RUNTIME_STATE_ATTR, None)
        if not isinstance(previous, PageLogState) or previous.run_id != run_id:
            previous = None
        if budget_selected in {"on", "shadow"}:
            host_prompt_cap = _resolve_host_prompt_cap(
                capture, body, host_prompt_cap,
            )

        def project_budget_candidate(selected_budget_mode: str) -> PageLogProjection:
            return project_request(
                body, previous,
                token_meter=token_meter,
                context_tokens=context_tokens,
                selected_mode=selected,
                require_product=True,
                run_id=run_id,
                runtime=runtime,
                source_mode=source_selected,
                retirement_mode=retirement_selected,
                exchange_mode=exchange_selected,
                window_mode=window_selected,
                budget_mode=selected_budget_mode,
                host_prompt_cap=host_prompt_cap,
            )

        projection = _project_with_budget_policy(
            project_budget_candidate,
            budget_mode=budget_selected,
            host_prompt_cap=host_prompt_cap,
            journal=journal,
            capture=capture,
        )
        request_id = (str(getattr(capture, "request_id", "")) or None)
        step = getattr(capture, "step", None) if type(getattr(capture, "step", None)) is int else None
        _emit_exchange_result(
            journal, projection, exchange_mode=exchange_selected,
            request_id=request_id, step=step,
        )
        pending = PageLogPending(
            runtime=runtime,
            previous_state=projection.previous_state,
            candidate_state=projection.state,
            mode=selected,
            request_id=request_id,
            step=step,
        )
        setattr(capture, _PENDING_ATTR, pending)
        _emit_source_supersession_result(
            journal, projection, source_mode=source_selected,
            request_id=pending.request_id, step=pending.step,
        )
        _emit_retirement_result(
            journal, projection, selected_mode=retirement_selected,
            request_id=pending.request_id, step=pending.step,
        )
        _emit_window_key_result(
            journal, projection, selected_mode=window_selected,
            request_id=pending.request_id, step=pending.step,
        )
        _emit(
            journal, "stateframe_page_log_prepared",
            request_id=pending.request_id, step=pending.step, mode=selected,
            action=projection.action, rebase_reason=projection.rebase_reason,
            turn_index=projection.state.turn_index,
            page_count=len(projection.state.pages), epoch=projection.state.epoch,
            epoch_turns=projection.state.epoch_turns,
            stale_tokens=projection.state.stale_tokens,
            live_tokens=projection.live_tokens, prompt_tokens=projection.prompt_tokens,
            prefix_preserved=projection.prefix_preserved,
        )
        if projection.action == "rebase" and projection.rebase_reason:
            _emit(journal, "stateframe_page_log_rebased", request_id=pending.request_id,
                  step=pending.step, reason=projection.rebase_reason,
                  turn_index=projection.state.turn_index,
                  epoch=projection.state.epoch, epoch_turns=projection.state.epoch_turns)
        if selected == "shadow":
            original_messages = _wire_json(body.get("messages"))
            projected_messages = _wire_json(projection.body.get("messages"))
            if original_messages != projected_messages:
                _emit(
                    journal, "stateframe_page_log_shadow_mismatch",
                    request_id=pending.request_id, step=pending.step,
                    field="messages",
                    original_sha256=hashlib.sha256(original_messages).hexdigest(),
                    projected_sha256=hashlib.sha256(projected_messages).hexdigest(),
                    action=projection.action, rebase_reason=projection.rebase_reason,
                )
            return body, pending
        return projection.body, pending
    except Exception as error:
        if (exchange_selected in _VALID_EXCHANGE_LAYOUTS
                and _newest_complete_exchange(body) is not None):
            _emit_exchange_failure(
                journal, exchange_mode=exchange_selected,
                request_id=getattr(capture, "request_id", None),
                step=getattr(capture, "step", None),
                reason="projection_or_exact_tokenizer_unavailable",
                error_type=_safe_journal_error_type(error),
            )
        _emit_source_supersession_failure(
            journal, source_mode=source_selected,
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            reason="projection_or_exact_tokenizer_unavailable",
            error_type=_safe_journal_error_type(error),
        )
        _emit_retirement_failure(
            journal, selected_mode=retirement_selected,
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            reason="projection_or_exact_tokenizer_unavailable",
            error_type=_safe_journal_error_type(error),
        )
        _emit_window_key_failure(
            journal, selected_mode=window_selected,
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            reason="projection_or_exact_tokenizer_unavailable",
            error_type=_safe_journal_error_type(error),
        )
        if isinstance(error, PageLogBudgetError):
            failed_open_reason = "host_prompt_budget"
        elif isinstance(error, PageLogError) and str(error) == _CONTEXT_POLICY_ERROR:
            failed_open_reason = "context_policy"
        else:
            failed_open_reason = "projection_or_exact_tokenizer_unavailable"
        if budget_selected == "on" and isinstance(error, PageLogBudgetError):
            _emit_budget_event(
                journal,
                request_id=getattr(capture, "request_id", None),
                step=getattr(capture, "step", None),
                prompt_tokens=error.prompt_tokens,
                cap=error.cap,
                action="fail_open",
                mode_name=budget_selected,
            )
        _emit(
            journal, "stateframe_page_log_failed_open",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None), mode=selected,
            reason=failed_open_reason,
            error_type=_safe_journal_error_type(error),
        )
        return body, None


def fail_open_pending(capture: object, journal: object | None, *, reason: str,
                      error_type: str | None = None) -> None:
    pending = getattr(capture, _PENDING_ATTR, None)
    if not isinstance(pending, PageLogPending):
        return
    setattr(capture, _PENDING_ATTR, None)
    _emit(journal, "stateframe_page_log_failed_open", request_id=pending.request_id,
          step=pending.step, mode=pending.mode, reason=reason, error_type=error_type, vital=True)


def commit_after_response(capture: object, journal: object | None) -> None:
    """Commit the candidate epoch only after response_received is durably captured."""
    pending = getattr(capture, _PENDING_ATTR, None)
    if not isinstance(pending, PageLogPending):
        return
    current = getattr(pending.runtime, _RUNTIME_STATE_ATTR, None)
    if pending.previous_state is None:
        if (isinstance(current, PageLogState)
                and current.run_id == pending.candidate_state.run_id):
            fail_open_pending(capture, journal, reason="concurrent_page_log_state_advance")
            return
    elif (not isinstance(current, PageLogState)
          or current.run_id != pending.previous_state.run_id
          or current.generation != pending.previous_state.generation):
        fail_open_pending(capture, journal, reason="concurrent_page_log_state_advance")
        return
    try:
        setattr(pending.runtime, _RUNTIME_STATE_ATTR, copy.deepcopy(pending.candidate_state))
    except (AttributeError, TypeError, ValueError):
        fail_open_pending(capture, journal, reason="page_log_state_commit_failed")
        return
    setattr(capture, _PENDING_ATTR, None)
    _emit(journal, "stateframe_page_log_committed", request_id=pending.request_id,
          step=pending.step, mode=pending.mode,
          generation=pending.candidate_state.generation,
          epoch=pending.candidate_state.epoch,
          epoch_turns=pending.candidate_state.epoch_turns,
          page_count=len(pending.candidate_state.pages),
          stale_tokens=pending.candidate_state.stale_tokens)


__all__ = [
    "EXCHANGE_MODE_ENV", "MODE_ENV", "WINDOW_KEY_MODE_ENV", "PageLogError", "PageLogPending",
    "PageLogProjection", "PageLogState", "commit_after_response", "fail_open_pending", "mode",
    "page_key", "page_log_exchange_mode", "page_log_window_key_mode", "prefix_bytes",
    "prepare_for_capture", "project_request",
]
