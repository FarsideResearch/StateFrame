"""M3 working-context budgets and exact, callable source deferrals."""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from kernel_context import token_estimate


M3_WORKING_CONTEXT_ENV = "STATEFRAME_M3_WORKING_CONTEXT"
M3_METADATA_BUDGET_TOKENS = 1_500
M3_SOURCE_BUDGET_TOKENS = 6_000
M3_WORKING_CONTEXT_SCHEMA = "stateframe-m3-working-context/1"

_SOURCE_HEADER = re.compile(
    r"^\[(Current|Working|Observed|Retained|Repair) source:\s*([^\]\r\n]+)\]",
    re.IGNORECASE,
)
_READ_PAGE = re.compile(
    r"^\[read page\]\s+path=(.*?)\s+revision=(\S+)\s+"
    r"lines=(\d+)-(\d+)/(\d+)\b",
    re.IGNORECASE,
)
_READ_CURSOR = re.compile(r"page with cursor=(\S+)", re.IGNORECASE)
_COMPLETE_FILE = re.compile(
    r"^Coverage: COMPLETE FILE; lines (\d+)-(\d+) of (\d+);",
    re.MULTILINE,
)
_SOURCE_CONTENT_START = re.compile(r"(?m)^\[Source content: ([^\]\r\n]+)\]\s*$")
_M3_METADATA_PREFIXES = (
    "[StateFrame]", "[Decision handoff]", "[Get-sure state]",
    "[Working sets | host-retained]", "<settled_state>",
    "[Source page evicted]", "[Source page deferred]", "[Read once]",
    "[Read reuse]", "[WIRE_READ_CONTRACT]", "[Host source proof]",
)


@dataclass(frozen=True)
class _SourceUnit:
    message_index: int
    body: str
    metadata: str
    path: str
    revision: str
    start_line: int
    end_line: int
    digest: str
    label: str
    tool_call_id: str | None
    protected: bool
    priority: int
    requested_start_line: int = 0
    requested_end_line: int = 0
    continuation_cursor: str | None = None
    requires_cursor: bool = False

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return self.path, self.revision, self.start_line, self.end_line


def m3_working_context_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the strict M3 switch; it is independent of LEAN and defaults on."""
    source = os.environ if environ is None else environ
    value = str(source.get(M3_WORKING_CONTEXT_ENV, "on")).strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{M3_WORKING_CONTEXT_ENV} must be on or off, got {value!r}")
    return value == "on"


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _safe_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    candidate = PurePosixPath(raw)
    if (not raw or "\x00" in raw or "\n" in raw or "\r" in raw
            or candidate.is_absolute()
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in raw.split("/"))):
        return ""
    return raw


def _positive_int(value: object) -> int | None:
    if type(value) is int and value > 0:
        return value
    return None


def _safe_metadata_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9:._-]", "", str(value or ""))[:128]


def _call_names(messages: Sequence[Mapping[str, object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not _is_sequence(calls):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id")
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(call_id, str) and call_id and isinstance(name, str):
                result[call_id] = name
    return result


def _read_tool_admitted(tools: object, *, require_cursor: bool = False) -> bool:
    if not _is_sequence(tools):
        return False
    for item in tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        name = function.get("name") if isinstance(function, Mapping) else item.get("name")
        if name != "read":
            continue
        parameters = function.get("parameters") if isinstance(function, Mapping) else item.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if not isinstance(properties, Mapping):
            continue
        range_admitted = all(
                isinstance(properties.get(argument), Mapping)
                and properties[argument].get("type") == "integer"
                for argument in ("start_line", "end_line"))
        cursor_admitted = (
            not require_cursor
            or (isinstance(properties.get("cursor"), Mapping)
                and properties["cursor"].get("type") == "string")
        )
        if range_admitted and cursor_admitted:
            return True
    return False


def _line_range(start: object, end: object, body: str) -> tuple[int, int] | None:
    first = _positive_int(start) or 1
    last = _positive_int(end)
    if last is not None and last < first:
        return None
    if last is None:
        last = max(first, first + max(1, len(body.splitlines())) - 1)
    return first, last


def _user_source_unit(index: int, message: Mapping[str, object],
                      protected_keys: set[tuple[str, str, int, int]],
                      protected_paths: set[str]) -> _SourceUnit | None:
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return None
    content = str(message["content"])
    # Final allocation admits these host-rendered, read-receipted pages against
    # the measured complete wire. M3's smaller heuristic source budget must not
    # count or defer them on a later fitting pass.
    from kernel_request_source_window import parse_model_source_window
    projected = parse_model_source_window(content)
    if isinstance(projected, Mapping) and projected.get("owner") in {
            "recent_editable_read", "required_public_check"}:
        projected_path = projected.get("path")
        revision = projected.get("revision")
        if (isinstance(projected_path, str) and _safe_path(projected_path) == projected_path
                and isinstance(revision, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", revision)
                and re.search(r"(?m)^\[read page\] path=" + re.escape(projected_path)
                              + r" revision=" + re.escape(revision)
                              + r" lines=\d+-\d+/\d+ complete=true$", content)):
            return None
    first_line = content.splitlines()[0] if content.splitlines() else ""
    header = _SOURCE_HEADER.match(first_line)
    if header is None:
        return None
    label, header_path = header.group(1).lower(), _safe_path(header.group(2).strip())
    metadata: dict[str, object] = {}
    body = ""
    metadata_text = first_line
    range_start: object = None
    range_end: object = None
    source_match = _SOURCE_CONTENT_START.search(content)
    if source_match is not None:
        metadata_start = content.find("\n[Source metadata]\n")
        if metadata_start >= 0:
            json_start = metadata_start + len("\n[Source metadata]\n")
            json_end = content.find("\n", json_start)
            if json_end >= 0:
                try:
                    parsed = json.loads(content[json_start:json_end])
                except (TypeError, ValueError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, Mapping):
                    metadata = dict(parsed)
        body_start = source_match.end()
        if body_start < len(content) and content[body_start] == "\n":
            body_start += 1
        raw_path = str(metadata.get("path") or source_match.group(1) or header_path)
        path = _safe_path(raw_path)
        end_marker = f"\n[End source content: {source_match.group(1)}]"
        body_end = content.rfind(end_marker)
        if body_end < body_start:
            return None
        body = content[body_start:body_end]
        metadata_text = content[:body_start] + content[body_end:]
        effective = metadata.get("effective_range")
        effective = effective if isinstance(effective, Mapping) else {}
        range_start = metadata.get("start_line", effective.get("start_line"))
        range_end = metadata.get(
            "selected_end_line", metadata.get("end_line", effective.get("end_line")))
    else:
        coverage = _COMPLETE_FILE.search(content)
        if coverage is not None:
            body_start = content.find("\n", coverage.start())
            if body_start < 0:
                return None
            body_start += 1
            body = content[body_start:]
            metadata_text = content[:body_start]
            path = header_path
            range_start, range_end = int(coverage.group(1)), int(coverage.group(2))
        else:
            body_start = len(first_line)
            while body_start < len(content) and content[body_start] in "\r\n":
                body_start += 1
            body = content[body_start:]
            metadata_text = content[:body_start]
            path = header_path
    if not body:
        return None
    if _positive_int(range_start) is not None and _positive_int(range_end) is not None:
        line_range = _line_range(range_start, range_end, body)
        if line_range is None:
            return None
        start_line, end_line = line_range
    else:
        # Keep unbounded source in the allocation, but never invent a callable range.
        start_line, end_line = 0, 0
    revision = _safe_metadata_token(metadata.get("revision") or metadata.get("file_revision"))
    digest = _safe_metadata_token(metadata.get("content_digest"))
    identity = (path, revision, start_line, end_line)
    protected = (label in {"working", "repair"} or path in protected_paths
                 or identity in protected_keys)
    return _SourceUnit(
        index, body, metadata_text, path, revision, start_line, end_line, digest,
        label, None, protected or identity in protected_keys,
        {"current": 0, "working": 0, "repair": 0, "observed": 1, "retained": 2}[label],
        start_line, end_line,
    )


def _tool_source_unit(index: int, message: Mapping[str, object],
                      call_names: Mapping[str, str]) -> _SourceUnit | None:
    if message.get("role") != "tool" or not isinstance(message.get("content"), str):
        return None
    call_id = message.get("tool_call_id")
    if not isinstance(call_id, str) or call_names.get(call_id) != "read":
        return None
    content = str(message["content"])
    first_line = content.splitlines()[0] if content.splitlines() else ""
    match = _READ_PAGE.match(first_line)
    if match is None:
        return None
    path = _safe_path(match.group(1).strip())
    start_line, end_line = int(match.group(3)), int(match.group(4))
    requested_start, requested_end = start_line, end_line
    requested_match = re.search(r"\brequested=lines:(\d+)-(\d+)", first_line)
    if requested_match is not None:
        requested_start, requested_end = int(requested_match.group(1)), int(
            requested_match.group(2))
    body_start = content.find("\n")
    body = content[body_start + 1:] if body_start >= 0 else ""
    if not body:
        return None
    continuation_line = content.splitlines()[1] if len(content.splitlines()) > 1 else ""
    cursor_match = _READ_CURSOR.search(continuation_line)
    truncated_match = re.search(r"\btruncated=(true|false)\b", first_line, re.IGNORECASE)
    truncated = (truncated_match is not None
                 and truncated_match.group(1).lower() == "true")
    cursor = cursor_match.group(1) if cursor_match is not None else None
    if cursor is not None and len(cursor) > 512:
        cursor = None
    requires_cursor = truncated or cursor is not None
    return _SourceUnit(
        index, body, first_line, path, _safe_metadata_token(match.group(2)), start_line, end_line,
        "", "tool_read", call_id, False, 1,
        requested_start, requested_end, cursor, requires_cursor,
    )


def _source_units(messages: Sequence[Mapping[str, object]], tools: object,
                  protected_source_keys: set[tuple[str, str, int, int]],
                  protected_source_paths: set[str]) -> list[_SourceUnit]:
    call_names = _call_names(messages)
    result: list[_SourceUnit] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        source = _user_source_unit(index, message, protected_source_keys, protected_source_paths)
        if source is None:
            source = _tool_source_unit(index, message, call_names)
        if source is not None:
            result.append(source)
    return result


def _metadata_message(message: Mapping[str, object], source_indexes: set[int], index: int) -> bool:
    if index in source_indexes:
        return False
    role = message.get("role")
    if role not in {"user", "tool"}:
        return False
    text = str(message.get("content") or "")
    if text.startswith(_M3_METADATA_PREFIXES):
        return True
    # The F4 fit's notice contract preserves blocker and retrieval lines when compacting.
    from kernel_stateframe_f4_fit import _is_notice
    return role == "user" and _is_notice(text)


def _budget_totals(messages: Sequence[Mapping[str, object]], tools: object,
                   protected_keys: set[tuple[str, str, int, int]],
                   protected_paths: set[str]) -> tuple[int, int, list[_SourceUnit]]:
    units = _source_units(messages, tools, protected_keys, protected_paths)
    by_index = {unit.message_index: unit for unit in units}
    source_tokens = sum(token_estimate(unit.body) for unit in units)
    metadata_tokens = sum(token_estimate(unit.metadata) for unit in units)
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and _metadata_message(message, set(by_index), index):
            metadata_tokens += token_estimate(str(message.get("content") or ""))
    return metadata_tokens, source_tokens, units


def _defer(unit: _SourceUnit) -> str:
    revision = f" revision={unit.revision}" if unit.revision else ""
    digest = f" digest={unit.digest}" if unit.digest else ""
    requested_start = unit.requested_start_line or unit.start_line
    requested_end = unit.requested_end_line or unit.end_line
    read_call = (f"read(path={json.dumps(unit.path, ensure_ascii=False)}, "
                 f"start_line={requested_start}, end_line={requested_end})")
    continuation = ""
    if unit.requires_cursor and unit.continuation_cursor:
        cursor_call = (f"read(path={json.dumps(unit.path, ensure_ascii=False)}, "
                       f"cursor={json.dumps(unit.continuation_cursor, ensure_ascii=False)})")
        continuation = f"\nContinue clipped bytes with: {cursor_call}."
    return (
        "[Source page deferred]\n"
        f"path={unit.path}{revision} lines={unit.start_line}-{unit.end_line}{digest}\n"
        f"Retrieve current bytes with: {read_call}.{continuation}"
    )


def _defer_unit(messages: list[dict[str, object]], unit: _SourceUnit) -> dict[str, object]:
    receipt = _defer(unit)
    message = messages[unit.message_index]
    if unit.tool_call_id is not None:
        # Keep the exact call/result protocol. A read result can be replaced only by a
        # receipt with a current, schema-admitted read route and exact source range.
        message["content"] = receipt
    else:
        message["content"] = receipt
    return {
        "path": unit.path, "revision": unit.revision,
        "start_line": unit.start_line, "end_line": unit.end_line,
        "has_tool_call": unit.tool_call_id is not None, "reason": "m3_source_budget",
    }


def _compact_metadata(messages: list[dict[str, object]], tools: object,
                      protected_keys: set[tuple[str, str, int, int]],
                      protected_paths: set[str]) -> bool:
    """Shed optional StateFrame rows and compact only contract-aware host notices."""
    before = messages.copy()
    units = _source_units(messages, tools, protected_keys, protected_paths)
    source_indexes = {unit.message_index for unit in units}
    source_metadata = sum(token_estimate(unit.metadata) for unit in units)
    metadata_indexes = [
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping) and _metadata_message(message, source_indexes, index)
    ]
    if metadata_indexes:
        from kernel_stateframe_f4_fit import _empty_phase, _shed_state_rows
        subset = [copy.deepcopy(messages[index]) for index in metadata_indexes]
        _shed_state_rows(
            subset, None, max(0, M3_METADATA_BUDGET_TOKENS - source_metadata), _empty_phase())
        for index, value in zip(metadata_indexes, subset):
            messages[index] = value
    from kernel_stateframe_f4_fit import _compact_notice, _is_notice
    for index, message in enumerate(messages):
        if index in source_indexes or not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = str(message.get("content") or "")
        if role != "user" and not (role == "tool" and content.startswith("[Source page deferred]")):
            continue
        if not _is_notice(content):
            continue
        compacted = _compact_notice(content)
        if compacted != content:
            messages[index] = {**dict(message), "content": compacted}
            current, _source, _units = _budget_totals(
                messages, tools, protected_keys, protected_paths)
            if current <= M3_METADATA_BUDGET_TOKENS:
                break
    return messages != before


def apply_m3_working_context_budget(
        messages: Sequence[Mapping[str, object]], tools: object, *,
        protected_source_keys: Sequence[Sequence[object]] = (),
        protected_source_paths: Sequence[str] = (),
        enabled_value: bool | None = None) -> dict[str, object]:
    """Bound current source and metadata while leaving every omission actionable."""
    active = m3_working_context_enabled() if enabled_value is None else bool(enabled_value)
    if not active:
        return {
            "messages": messages, "changed": False, "fits": True,
            "manifest": {
                "schema": M3_WORKING_CONTEXT_SCHEMA, "enabled": False,
                "metadata_tokens": None, "metadata_budget_tokens": M3_METADATA_BUDGET_TOKENS,
                "source_tokens": None, "source_budget_tokens": M3_SOURCE_BUDGET_TOKENS,
                "deferrals": [], "fits": True,
            },
        }
    if not _is_sequence(messages) or any(not isinstance(item, Mapping) for item in messages):
        raise ValueError("M3 working-context messages must be a sequence of objects")
    protected_keys: set[tuple[str, str, int, int]] = set()
    if _is_sequence(protected_source_keys):
        for item in protected_source_keys:
            if not _is_sequence(item) or len(item) != 4:
                continue
            path = _safe_path(item[0])
            revision = str(item[1] or "")[:128]
            start, end = _positive_int(item[2]), _positive_int(item[3])
            if path and start is not None and end is not None and end >= start:
                protected_keys.add((path, revision, start, end))
    protected_paths = {
        path for value in protected_source_paths
        if isinstance(value, str) and (path := _safe_path(value))
    } if _is_sequence(protected_source_paths) else set()

    original = list(messages)
    working = [copy.deepcopy(dict(message)) for message in messages]
    deferrals: list[dict[str, object]] = []

    while True:
        metadata_tokens, source_tokens, units = _budget_totals(
            working, tools, protected_keys, protected_paths)
        if metadata_tokens <= M3_METADATA_BUDGET_TOKENS and source_tokens <= M3_SOURCE_BUDGET_TOKENS:
            break
        _compact_metadata(working, tools, protected_keys, protected_paths)
        metadata_tokens, source_tokens, units = _budget_totals(
            working, tools, protected_keys, protected_paths)
        if metadata_tokens <= M3_METADATA_BUDGET_TOKENS and source_tokens <= M3_SOURCE_BUDGET_TOKENS:
            break
        candidates = [
            unit for unit in units
            if not unit.protected and unit.path and unit.start_line >= 1
            and unit.end_line >= unit.start_line
            and (not unit.requires_cursor or unit.continuation_cursor is not None)
            and _read_tool_admitted(tools, require_cursor=unit.requires_cursor)
            and (source_tokens > M3_SOURCE_BUDGET_TOKENS
                 or token_estimate(unit.metadata) > 0)
        ]
        if not candidates:
            break
        victim = max(candidates, key=lambda unit: (unit.priority, unit.message_index))
        deferrals.append(_defer_unit(working, victim))

    metadata_tokens, source_tokens, _units = _budget_totals(
        working, tools, protected_keys, protected_paths)
    fits = (metadata_tokens <= M3_METADATA_BUDGET_TOKENS
            and source_tokens <= M3_SOURCE_BUDGET_TOKENS)
    changed = working != original
    return {
        "messages": working if changed else messages,
        "changed": changed,
        "fits": fits,
        "manifest": {
            "schema": M3_WORKING_CONTEXT_SCHEMA,
            "enabled": True,
            "metadata_tokens": metadata_tokens,
            "metadata_budget_tokens": M3_METADATA_BUDGET_TOKENS,
            "source_tokens": source_tokens,
            "source_budget_tokens": M3_SOURCE_BUDGET_TOKENS,
            "deferrals": deferrals,
            "fits": fits,
        },
    }


__all__ = [
    "M3_METADATA_BUDGET_TOKENS", "M3_SOURCE_BUDGET_TOKENS",
    "M3_WORKING_CONTEXT_ENV", "M3_WORKING_CONTEXT_SCHEMA",
    "apply_m3_working_context_budget", "m3_working_context_enabled",
]
