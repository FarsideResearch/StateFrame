"""Bounded, truthful observation for repeated source queries without progress."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_context import token_estimate
from kernel_lean import message_digest, stall_observation_enabled


MAX_OBSERVATION_TOKENS = 128
MAX_SEEN_SOURCE_HASHES = 4096
_REQUEST_SOURCE_WINDOW_RE = re.compile(
    r"^\[Request budget source window\]\r?\n"
    r"(?P<metadata>\{[^\r\n]*\})\r?\n"
    r"\[Source window content\]\r?\n"
    r"(?P<body>.*?)(?:\r?\n\[End source window\]|$)",
    re.MULTILINE | re.DOTALL,
)
_CURRENT_SOURCE_HEADER_RE = re.compile(
    r"(?m)^\[(?:Current|Working) source: (?P<path>[^\]\r\n]+)\]$",
)
_CURRENT_SOURCE_FOOTER_RE = re.compile(
    r"(?m)^\[(?:WIRE_TRUTH_FACT [^\]\r\n]+|Files in frame|Host observation)\]",
)


def _tool_name(call: object) -> str:
    if not isinstance(call, Mapping):
        return ""
    function = call.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(call.get("name", ""))


def _tool_arguments(call: object) -> object:
    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    return function.get("arguments") if isinstance(function, Mapping) else call.get("arguments")


def _actions(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    pending: dict[str, dict[str, object]] = {}
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            for call in raw_calls if isinstance(raw_calls, list) else ():
                if not isinstance(call, Mapping):
                    continue
                call_id = str(call.get("id", ""))
                row = {
                    "name": _tool_name(call),
                    "arguments": _tool_arguments(call),
                    "result": None,
                }
                if call_id:
                    pending[call_id] = row
                else:
                    actions.append(row)
        elif role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            action = pending.pop(call_id, None) if call_id else None
            if action is not None:
                action["result"] = str(message.get("content") or "")
                actions.append(action)
    return actions


def _result_source_hash(result: object) -> str:
    value = str(result or "")
    first_line = value.splitlines()[0] if value.splitlines() else ""
    page_match = re.match(
        r"^\[read page\] path=(\S+) revision=(\S+).*?effective=lines:(\d+)-(\d+)@(\d+):(\d+)",
        first_line,
    )
    if page_match is not None:
        return "sha256:" + hashlib.sha256(
            "\0".join(page_match.groups()).encode("utf-8")
        ).hexdigest()
    reuse_match = re.match(
        r"^\[read reuse\] path=(\S+) revision=(\S+) requested=\S+ effective=(\{.*\});",
        first_line,
    )
    if reuse_match is not None:
        try:
            effective = json.loads(reuse_match.group(3))
        except (TypeError, ValueError, json.JSONDecodeError):
            effective = None
        if isinstance(effective, Mapping):
            identity = (
                reuse_match.group(1), reuse_match.group(2),
                str(effective.get("start_line", "")),
                str(effective.get("end_line", "")),
                str(effective.get("start_column", "")),
                str(effective.get("end_column", "")),
            )
            return "sha256:" + hashlib.sha256(
                "\0".join(identity).encode("utf-8")
            ).hexdigest()
    try:
        from kernel_request_source_window import parse_model_source_window, parse_window
        parsed = parse_window(value) or parse_model_source_window(value)
    except (ImportError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping):
        metadata = parsed.get("metadata", parsed)
        if isinstance(metadata, Mapping):
            revision = metadata.get("revision", metadata.get("sha256"))
            ranges = metadata.get("ranges")
            if isinstance(revision, str) and revision and isinstance(ranges, list):
                identity = {
                    "path": metadata.get("path"),
                    "revision": revision,
                    "ranges": ranges,
                }
                return "sha256:" + hashlib.sha256(json.dumps(
                    identity, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
    request_window = _REQUEST_SOURCE_WINDOW_RE.search(value)
    if request_window is not None:
        try:
            request_metadata = json.loads(request_window.group("metadata"))
        except (TypeError, ValueError, json.JSONDecodeError):
            request_metadata = None
        if isinstance(request_metadata, Mapping):
            page = request_metadata.get("page")
            page_metadata = page if isinstance(page, Mapping) else {}
            path = request_metadata.get("path") or page_metadata.get("path")
            revision = (
                request_metadata.get("revision")
                or request_metadata.get("sha256")
                or page_metadata.get("revision")
            )
            ranges = request_metadata.get("ranges")
            if isinstance(path, str) and path:
                source_digest = hashlib.sha256(
                    request_window.group("body").encode("utf-8", "replace")
                ).hexdigest()
                identity = {
                    "path": path,
                    "revision": revision,
                    "ranges": ranges if isinstance(ranges, list) else None,
                    "source_sha256": source_digest,
                }
                return "sha256:" + hashlib.sha256(json.dumps(
                    identity, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
    current_source_header = _CURRENT_SOURCE_HEADER_RE.search(value)
    if current_source_header is not None:
        path = current_source_header.group("path").replace("\\", "/")
        body_start = current_source_header.end()
        if value.startswith("\r\n", body_start):
            body_start += 2
        elif value.startswith("\n", body_start):
            body_start += 1
        source_body = value[body_start:]
        footer = _CURRENT_SOURCE_FOOTER_RE.search(source_body)
        if footer is not None:
            source_body = source_body[:footer.start()]
        identity = {
            "path": path,
            "source_sha256": hashlib.sha256(
                source_body.encode("utf-8", "replace")
            ).hexdigest(),
        }
        return "sha256:" + hashlib.sha256(json.dumps(
            identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
    return "sha256:" + hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def _trailing_no_progress_repeats(actions: Sequence[Mapping[str, object]]) -> int:
    latest_hash = None
    repeats = 0
    for action in reversed(actions):
        if str(action.get("name", "")) not in {"read", "grep", "search"}:
            break
        source_hash = _result_source_hash(action.get("result"))
        if latest_hash is None:
            latest_hash = source_hash
        elif source_hash == latest_hash:
            repeats += 1
        else:
            break
    return repeats


def _diff_hash(runtime: object) -> str | None:
    target = getattr(runtime, "target", None)
    if not isinstance(target, Path) or not target.is_file():
        return None
    try:
        current = target.read_bytes()
    except OSError:
        return None
    baseline = str(getattr(runtime, "_turtle_baseline_source", "")).encode("utf-8")
    return hashlib.sha256(baseline + b"\0" + current).hexdigest()


def _new_state() -> dict[str, object]:
    return {
        "cursor": 0,
        "seen_source_hashes": [],
        "diff_hash": None,
        "consecutive_no_progress": 0,
        "fired": False,
    }


def observe_stall(
    runtime: object,
    messages: Sequence[Mapping[str, object]],
    requirement: str,
    *,
    enabled: bool | None = None,
) -> dict[str, object] | None:
    """Add one 128-token observation after three repeated queries without progress."""
    active = stall_observation_enabled() if enabled is None else bool(enabled)
    if not active:
        return None
    copied = [dict(message) for message in messages]
    latest_tool = next((message for message in reversed(copied)
                        if message.get("role") == "tool"), None)
    if latest_tool is None:
        return None

    state = getattr(runtime, "_stall_observation_state", None)
    if not isinstance(state, dict):
        state = _new_state()
        setattr(runtime, "_stall_observation_state", state)
    actions = _actions(copied)
    previous_diff = state.get("diff_hash")
    current_diff = _diff_hash(runtime)
    diff_changed = previous_diff is not None and current_diff is not None and current_diff != previous_diff
    state["diff_hash"] = current_diff
    if diff_changed:
        state["consecutive_no_progress"] = 0
        state["fired"] = False

    seen = state.get("seen_source_hashes")
    if not isinstance(seen, list):
        seen = []
    seen_set = set(str(item) for item in seen)
    cursor = min(max(int(state.get("cursor", 0)), 0), len(actions))
    progress = diff_changed
    for action in actions[cursor:]:
        name = str(action.get("name", ""))
        if name in {"read", "grep", "search"}:
            source_hash = _result_source_hash(action.get("result"))
            if source_hash not in seen_set:
                seen_set.add(source_hash)
                progress = True
                state["consecutive_no_progress"] = 0
                state["fired"] = False
            else:
                state["consecutive_no_progress"] = int(
                    state.get("consecutive_no_progress", 0)) + 1
        elif name in {"run_tests", "test", "authored_check"}:
            progress = True
            state["consecutive_no_progress"] = 0
            state["fired"] = False
        else:
            state["consecutive_no_progress"] = 0
    state["cursor"] = len(actions)
    state["seen_source_hashes"] = list(seen_set)[-MAX_SEEN_SOURCE_HASHES:]

    count = max(
        int(state.get("consecutive_no_progress", 0)),
        _trailing_no_progress_repeats(actions),
    )
    if diff_changed:
        count = 0
    state["consecutive_no_progress"] = count
    if progress or count < 3 or state.get("fired") is True:
        return None
    requirement = str(requirement)
    text = (
        "[Host observation] "
        f"{count} consecutive read/grep/search calls; no new bytes; diff unchanged.\n"
        "Current requirement (verbatim):\n" + requirement
    )
    if token_estimate(dict(role="tool", content=text)) > MAX_OBSERVATION_TOKENS:
        return None
    content = str(latest_tool.get("content") or "")
    if text in content:
        state["fired"] = True
        return None
    latest_tool["content"] = content.rstrip() + "\n\n" + text
    state["fired"] = True
    return {
        "messages": copied,
        "repeat_count": count,
        "progress": False,
        "observation": text,
        "observation_sha256": message_digest([latest_tool]),
    }


def observe_stall_in_place(
    runtime: object,
    messages: list[dict[str, object]],
    requirement: str,
    *,
    enabled: bool | None = None,
) -> dict[str, object] | None:
    """Attach a fired observation to the live transcript before its next request."""
    result = observe_stall(runtime, messages, requirement, enabled=enabled)
    if result is None:
        return None
    updated_messages = result.get("messages")
    if isinstance(updated_messages, list):
        messages[:] = updated_messages
    return result


__all__ = ["MAX_OBSERVATION_TOKENS", "observe_stall", "observe_stall_in_place"]
