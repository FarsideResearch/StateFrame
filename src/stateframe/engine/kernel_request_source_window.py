"""Render a bounded source window around the requested location.

Preserve source coordinates so a displayed range can be matched to subsequent
read and edit operations.
"""
from __future__ import annotations

import ast
import copy
import difflib
import functools
import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_frame_request_allocation import (
    FrameRequestAllocation,
    FrameRequestAllocationError,
)
from kernel_paged_source import locate_symbol, source_revision, symbol_outline
from kernel_request_evidence_closure import (
    build_source_closure,
    closure_plan_for_runtime,
    initialization_fact,
    record_final_source_delivery,
    record_source_delivery,
    render_closure_receipt,
    ticket_identity_for_runtime,
    units_for_path,
)
from kernel_aha11_native_honesty import reserve_source_unit
from kernel_measurement import profile_active_host_span, profiled_function_span
import kernel_workspace_truth as _workspace_truth
from kernel_evictfix_mode import enabled as evictfix_enabled


SCHEMA = "stateframe-request-source-window/1"
WINDOW_MARKER = "[Request budget source window]"
WINDOW_CONTENT_MARKER = "[Source window content]"
WINDOW_END_MARKER = "[End source window]"
MODEL_WINDOW_MARKER = "[Model source window]"
MODEL_WINDOW_SCHEMA = SCHEMA + "-projection/1"
_HEADER = re.compile(r"^\[(?:Current|Working) source: ([^\]\n]+)\]$")
_RANGE = re.compile(r"^--- source lines (\d+)-(\d+) ---$")
_PATH = re.compile(r"(?<![\w./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_MAX_PAGE_LINES = 1000
_MAX_PAGE_CHARS = 64_000
_DEFAULT_CONTEXT_LINES = 12
C5_TARGET_CONTEXT_LINES = 50
_MIN_WINDOW_CHARS = 512
_VERIFIED_SOURCE_HISTORY_LIMIT = 256
_VERIFIED_SOURCE_LOCATION = re.compile(r"^\$(?:\.messages)?\[(\d+)\]\.content$")
_VERIFIED_SOURCE_HISTORY_LOCK = threading.RLock()
_LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".go": "go", ".rs": "rust", ".ts": "typescript",
    ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".php": "php", ".vue": "vue",
}

DEFERRED_SOURCE_NOTICE_HEADER = "[Deferred required source paths]"
_DEFERRED_SOURCE_PATHS_LABEL = (
    "Required source paths are not attached; read one only if your edit needs it:"
)
_LEGACY_DEFERRED_SOURCE_PATHS_LABEL = "Required source paths omitted from this request:"
_READ_PAGE_HEADER = re.compile(
    r"^\[read page\] path=(?P<path>.+?) revision=(?P<revision>sha256:[0-9a-f]{64}) "
    r"lines=(?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+) requested=.+? "
    r"effective=lines:(?P<effective_start>\d+)-(?P<effective_end>\d+)@"
    r"(?P<start_column>\d+):(?P<end_column>\d+) clip=\S+ "
    r"chars=(?P<chars>\d+) bytes=(?P<bytes>\d+) "
    r"truncated=(?P<truncated>true|false) decoded_with_replacement=(?P<replaced>true|false)$"
)
_LIVE_WORKSPACE_DEFERRED_FACT_PATTERNS = (
    re.compile(
        r"^File (?P<path>.+) (?:exists|does not exist) in the live workspace "
        r"\(probe=stat(?:;revision=\S+)?\)\.$"
    ),
    re.compile(
        r"^Lines \d+-\d+ of (?P<path>.+?) were "
        r"(?:last delivered|delivered but clipped) at turn \d+ "
        r"\(read receipt(?: revision=\S+)?\)\.$"
    ),
    re.compile(
        r"^No current read receipt for (?P<path>.+?)(?: lines=\d+-\d+)?; "
        r"fetch current source before relying on it\.$"
    ),
    re.compile(
        r"^Current workspace fact unavailable for (?P<path>.+?) "
        r"\(probe=(?:stat|receipt), reason=.+\); refresh evidence\.$"
    ),
)


def _normalized_deferred_paths(paths: Sequence[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    for path in paths:
        value = str(path).replace("\\", "/").strip().removeprefix("./")
        if value and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _render_truthful_deferred_source_notice(
        paths: Sequence[object], *,
        delivered_source_ranges: Sequence[tuple[str, str, int, int]] = (),
) -> str:
    """Render delivered source spans and excerpts omitted from this request."""
    normalized = _normalized_deferred_paths(paths)
    if not normalized:
        raise ValueError("deferred source notice requires at least one path")
    lines = [
        DEFERRED_SOURCE_NOTICE_HEADER,
        "Source ranges delivered in this request:",
    ]
    delivered = sorted({
        (str(path).replace("\\", "/").strip().removeprefix("./"),
         str(revision or ""), int(start), int(end))
        for path, revision, start, end in delivered_source_ranges
        if str(path).strip() and isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool) and start > 0 and end >= start
    })
    if delivered:
        lines.extend(
            f"- {path}:{start}-{end} revision={revision or 'unknown'}"
            for path, revision, start, end in delivered
        )
    else:
        lines.append("- none")
    lines.append(_DEFERRED_SOURCE_PATHS_LABEL)
    lines.extend(f"- {path}" for path in normalized)
    return "\n".join(lines)


def _upsert_truthful_deferred_source_notice(
        messages: list[dict[str, object]], paths: Sequence[object],
) -> None:
    from kernel_request_diet import (
        _fully_delivered_source_paths,
        _source_delivery_ranges,
        truthful_read_panels_enabled,
    )

    requested = _normalized_deferred_paths(paths)
    if not truthful_read_panels_enabled():
        if not requested:
            return
        messages[:] = [
            message for message in messages
            if not (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].startswith(DEFERRED_SOURCE_NOTICE_HEADER)
            )
        ]
        messages.append({
            "role": "user",
            "content": "\n".join((
                DEFERRED_SOURCE_NOTICE_HEADER,
                _workspace_truth.READ_BEFORE_USE_LABEL,
                *(f"- {path}" for path in requested),
            )),
            "_request_budget_section": "stage_evidence",
        })
        return
    fully_delivered = _fully_delivered_source_paths(messages)
    normalized = tuple(path for path in requested if path not in fully_delivered)
    delivered = [
        (path, revision, start, end)
        for (path, revision), ranges in sorted(_source_delivery_ranges(messages).items())
        for start, end in ranges
    ]
    # These notices are host-generated request panels, including legacy panels
    # that carried a standing read imperative. Rebuild them from the current
    # request contents so stale paths and imperatives cannot survive in history.
    messages[:] = [
        message for message in messages
        if not (
            isinstance(message, Mapping)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].startswith(DEFERRED_SOURCE_NOTICE_HEADER)
        )
    ]
    if not normalized:
        return
    content = _render_truthful_deferred_source_notice(
        normalized, delivered_source_ranges=delivered,
    )
    notice = {
        "role": "user",
        "content": content,
        "_request_budget_section": "stage_evidence",
    }
    messages.append(notice)


def _executable_deferred_read_route(
        path: object, revision: object, route: object,
) -> bool:
    normalized_path = _normalized_deferred_paths((path,))
    if not normalized_path or not isinstance(route, Mapping):
        return False
    path_text = normalized_path[0]
    if not isinstance(revision, str):
        return False
    revision_text = revision.strip()
    route_paths = _normalized_deferred_paths((route.get("path", ""),))
    handles = route.get("unit_handles")
    return bool(
        str(route.get("status", "")).upper() == "EXECUTABLE_BOUNDED_READ"
        and route.get("tool") == "read"
        and route_paths == (path_text,)
        and revision_text.casefold() not in {"", "unknown", "none", "null"}
        and str(route.get("revision", "")).strip() == revision_text
        and isinstance(handles, Sequence)
        and not isinstance(handles, (str, bytes, bytearray))
        and handles
        and all(isinstance(handle, str) and handle.strip() for handle in handles)
    )


def source_delivery_satisfied(receipt: object) -> bool:
    """Accept only exact-wire COMPLETE or a fully verified deferred receipt."""
    if not isinstance(receipt, Mapping):
        return False
    proof_status = str(receipt.get("proof_status", "")).upper()
    status = str(receipt.get("status", "")).upper()
    if proof_status != "VERIFIED":
        return False
    if status == "COMPLETE":
        return True
    deferred_paths = receipt.get("deferred_paths")
    path_results = receipt.get("paths")
    if (status != "DEFERRED"
            or not isinstance(deferred_paths, Sequence)
            or isinstance(deferred_paths, (str, bytes, bytearray))
            or receipt.get("deferred_notice_verified") is not True
            or receipt.get("read_tool_available") is not True
            or not isinstance(path_results, Mapping)):
        return False
    for raw_path in deferred_paths:
        path_values = _normalized_deferred_paths((raw_path,))
        if not path_values:
            return False
        path = path_values[0]
        row = path_results.get(path)
        route = row.get("retrieval_route") if isinstance(row, Mapping) else None
        revision = row.get("revision", "") if isinstance(row, Mapping) else ""
        if (not isinstance(row, Mapping)
                or str(row.get("status", "")).upper() != "DEFERRED"
                or str(row.get("proof_status", "")).upper() != "VERIFIED"
                or not _executable_deferred_read_route(path, revision, route)):
            return False
    return bool(deferred_paths)


def record_verified_source_messages(
        runtime: object, messages: Sequence[Mapping[str, object]], receipt: object,
) -> int:
    """Remember exact source messages only after a verified provider submission."""
    if not source_delivery_satisfied(receipt) or not isinstance(receipt, Mapping):
        return 0
    locations = receipt.get("final_message_locations")
    paths = receipt.get("paths")
    if not isinstance(locations, Mapping) or not isinstance(paths, Mapping):
        return 0
    additions: list[dict[str, str]] = []
    for raw_path, raw_locations in locations.items():
        normalized = _normalized_deferred_paths((raw_path,))
        if not normalized:
            continue
        path = normalized[0]
        path_receipt = paths.get(path)
        if (not isinstance(path_receipt, Mapping)
                or str(path_receipt.get("proof_status", "")).upper() != "VERIFIED"):
            continue
        revision = str(path_receipt.get("revision", "")).strip()
        if not revision or revision.casefold() in {"unknown", "none", "null"}:
            continue
        if not isinstance(raw_locations, Sequence) or isinstance(
                raw_locations, (str, bytes, bytearray)):
            continue
        for location in raw_locations:
            match = _VERIFIED_SOURCE_LOCATION.fullmatch(str(location))
            if match is None:
                continue
            index = int(match.group(1))
            if not 0 <= index < len(messages):
                continue
            message = messages[index]
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                continue
            additions.append({
                "path": path,
                "revision": revision,
                "message_sha256": _digest(content),
            })
    if not additions:
        return 0
    with _VERIFIED_SOURCE_HISTORY_LOCK:
        existing = getattr(runtime, "_verified_source_message_history", ())
        history = [dict(item) for item in existing
                   if isinstance(item, Mapping)
                   and isinstance(item.get("path"), str)
                   and isinstance(item.get("revision"), str)
                   and isinstance(item.get("message_sha256"), str)] \
            if isinstance(existing, Sequence) and not isinstance(
                existing, (str, bytes, bytearray)) else []
        for item in additions:
            history = [previous for previous in history if (
                previous["path"], previous["revision"], previous["message_sha256"]
            ) != (item["path"], item["revision"], item["message_sha256"])]
            history.append(item)
        setattr(runtime, "_verified_source_message_history",
                history[-_VERIFIED_SOURCE_HISTORY_LIMIT:])
    return len(additions)


def _verified_prior_source_message(
        plan: Mapping[str, object], path: str, content: str, current_revision: object,
) -> bool:
    history = plan.get("verified_prior_source_messages", ())
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        return False
    digest = _digest(content)
    expected_revision = str(current_revision)
    return any(
        isinstance(item, Mapping)
        and str(item.get("path", "")).replace("\\", "/") == path
        and str(item.get("revision", "")) != expected_revision
        and str(item.get("message_sha256", "")) == digest
        for item in history
    )


def partial_source_paths_for_deferred_delivery(
        plan: Mapping[str, object], receipt: object,
) -> tuple[str, ...]:
    """Return selected, proven-partial paths with a revision-bound read route."""
    if not isinstance(receipt, Mapping):
        return ()
    authorization = receipt.get("workspace_authorization")
    path_results = receipt.get("paths")
    if (
        str(receipt.get("status", "")).upper() != "PARTIAL"
        or str(receipt.get("proof_status", "")).upper() != "VERIFIED"
        or receipt.get("proof_failures")
        or receipt.get("read_tool_available") is not True
        or not isinstance(authorization, Mapping)
        or str(authorization.get("status", "")).upper() != "VERIFIED"
        or not isinstance(path_results, Mapping)
    ):
        return ()
    revisions = plan.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    units_by_path: dict[str, list[str]] = {}
    for unit in plan.get("units", ()):
        if not isinstance(unit, Mapping):
            continue
        normalized = _normalized_deferred_paths((unit.get("path", ""),))
        handle = str(
            unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle") or ""
        ).strip()
        if normalized:
            units_by_path.setdefault(normalized[0], []).append(handle)
    ordered_paths: list[str] = []
    for raw_path in plan.get("paths", ()):
        normalized = _normalized_deferred_paths((raw_path,))
        if normalized and normalized[0] in units_by_path and normalized[0] not in ordered_paths:
            ordered_paths.append(normalized[0])
    ordered_paths.extend(path for path in units_by_path if path not in ordered_paths)

    candidates: list[str] = []
    for path in ordered_paths:
        row = path_results.get(path)
        if (not isinstance(row, Mapping)
                or str(row.get("status", "")).upper() != "PARTIAL"
                or str(row.get("proof_status", "")).upper() != "VERIFIED"):
            continue
        handles = units_by_path[path]
        revision = str(row.get("revision") or revisions.get(path, "")).strip()
        route = row.get("retrieval_route")
        if (not handles or any(not handle for handle in handles)
                or not _executable_deferred_read_route(path, revision, route)):
            continue
        candidates.append(path)
    return tuple(candidates)


def _wire_deferred_notice_state(
        decoded_wire: object, paths: Sequence[str],
) -> tuple[bool, bool]:
    if isinstance(decoded_wire, Mapping):
        messages = decoded_wire.get("messages")
        tools = decoded_wire.get("tools")
    else:
        messages = decoded_wire if isinstance(decoded_wire, list) else None
        tools = None
    read_available = any(
        isinstance(tool, Mapping)
        and str((tool.get("function", {}) if isinstance(tool.get("function"), Mapping)
                 else tool).get("name", "")) == "read"
        for tool in (tools if isinstance(tools, list) else ())
    )
    required_paths = set(_normalized_deferred_paths(paths))
    notices = deferred_source_notice_paths(messages)
    # Cache projection concatenates host tail messages into V. Parse exact
    # standalone blocks only from that validated region, never source pages,
    # frame JSON, or arbitrary tool text; the canonical notice parser is kept.
    from kernel_cache_layout_view import CacheLayoutError, volatile_blocks
    try:
        blocks = volatile_blocks(messages)
    except CacheLayoutError:
        return False, read_available
    notices += deferred_source_notice_paths([
        {"role": "user", "content": block} for block in blocks
        if block.startswith(DEFERRED_SOURCE_NOTICE_HEADER)
    ])
    # Earlier request notices remain prefix-stable. A canonical historical
    # notice may cover a smaller current deferral set; it can only over-instruct
    # the reader, while every currently deferred path still needs an exact match.
    notice_verified = bool(required_paths and any(
        required_paths.issubset(set(notice)) for notice in notices
    ))
    return notice_verified, read_available


def _read_page_source_range(
        content: object, source_map: Mapping[str, str],
        declared_revisions: Mapping[str, object],
) -> tuple[str, tuple[int, int], str | None] | None:
    """Accept only an exact, revision-bound, whole-line host read page."""
    if not isinstance(content, str):
        return None
    header, first_separator, remainder = content.partition("\n")
    more_line, second_separator, body = remainder.partition("\n")
    if not first_separator or not second_separator:
        return None
    match = _READ_PAGE_HEADER.fullmatch(header.removesuffix("\r"))
    if match is None:
        return None
    fields = match.groupdict()
    path = fields["path"].replace("\\", "/")
    source = source_map.get(path)
    declared_revision = declared_revisions.get(path)
    if (declared_revision is None or fields["revision"] != str(declared_revision)
            or fields["replaced"] != "false"):
        return None
    try:
        start = int(fields["start"])
        end = int(fields["end"])
        total = int(fields["total"])
        effective_start = int(fields["effective_start"])
        effective_end = int(fields["effective_end"])
        start_column = int(fields["start_column"])
        end_column = int(fields["end_column"])
        chars = int(fields["chars"])
        byte_count = int(fields["bytes"])
    except (TypeError, ValueError):
        return None
    # The host can append other captured context after a read result in the
    # same tool message. The page's authenticated extent is its declared char
    # and byte count; qualify that prefix and leave any later context outside it.
    page_body = body[:chars]
    body_bytes = page_body.encode("utf-8")
    if (len(page_body) != chars or len(body_bytes) != byte_count
            or more_line.startswith("READ_CURSOR_STALE")
            or start != effective_start or end != effective_end or start < 1
            or end < start or end > total or start_column != 0):
        return None
    full_source_from_page = None
    if isinstance(source, str) and source_revision(source) == fields["revision"]:
        page_source = source
    else:
        # A single complete read page carries its own full-file revision hash;
        # partial pages still need the host's frozen full source to qualify.
        if (start != 1 or end != total or fields["truncated"] != "false"
                or source_revision(page_body) != fields["revision"]):
            return None
        page_source = page_body
        full_source_from_page = page_body
    source_lines = page_source.splitlines(keepends=True)
    if (total != len(source_lines) or end_column != len(source_lines[end - 1])):
        return None
    expected = "".join(source_lines[start - 1:end])
    if page_body != expected:
        return None
    return path, (start, end), full_source_from_page


def deferred_source_notice_paths(
        messages: object,
) -> tuple[tuple[str, ...], ...]:
    """Return canonical deferred notices in user messages or the latest tool result."""
    if not isinstance(messages, list):
        return ()
    last_tool_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if isinstance(messages[index], Mapping)
         and messages[index].get("role") == "tool"),
        -1,
    )
    notices: list[tuple[str, ...]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            if not content.startswith(DEFERRED_SOURCE_NOTICE_HEADER):
                continue
            workspace_notice = _parse_live_workspace_deferred_notice(content)
            if workspace_notice:
                notices.append(workspace_notice)
                continue
            lines = content.splitlines()
            if len(lines) >= 3 and lines[1] == _workspace_truth.READ_BEFORE_USE_LABEL:
                paths = tuple(line[2:] for line in lines[2:] if line.startswith("- "))
                canonical = "\n".join((
                    DEFERRED_SOURCE_NOTICE_HEADER,
                    _workspace_truth.READ_BEFORE_USE_LABEL,
                    *(f"- {path}" for path in paths),
                ))
                if paths and content == canonical:
                    notices.append(paths)
                continue
            parsed = _parse_deferred_source_notice(content)
            if parsed:
                notices.append(parsed)
            continue
        if role != "tool" or index != last_tool_index:
            continue
        lines = content.splitlines()
        for start, line in enumerate(lines):
            if line != DEFERRED_SOURCE_NOTICE_HEADER:
                continue
            notice_lines = lines[start:]
            notice = "\n".join(notice_lines)
            workspace_notice = _parse_live_workspace_deferred_notice(notice)
            if workspace_notice:
                notices.append(workspace_notice)
                continue
            if len(notice_lines) >= 3 and notice_lines[1] == _workspace_truth.READ_BEFORE_USE_LABEL:
                paths = tuple(item[2:] for item in notice_lines[2:] if item.startswith("- "))
                canonical = "\n".join((
                    DEFERRED_SOURCE_NOTICE_HEADER,
                    _workspace_truth.READ_BEFORE_USE_LABEL,
                    *(f"- {path}" for path in paths),
                ))
                if paths and notice == canonical:
                    notices.append(paths)
                continue
            parsed = _parse_deferred_source_notice(notice)
            if parsed:
                notices.append(parsed)
    return tuple(notices)


def _parse_deferred_source_notice(content: str) -> tuple[str, ...]:
    lines = content.splitlines()
    if (len(lines) < 4 or lines[0] != DEFERRED_SOURCE_NOTICE_HEADER
            or lines[1] != "Source ranges delivered in this request:"):
        return ()
    label = next((line for line in lines
                  if line in {
                      _DEFERRED_SOURCE_PATHS_LABEL,
                      _LEGACY_DEFERRED_SOURCE_PATHS_LABEL,
                  }), None)
    if label is None:
        return ()
    omitted_index = lines.index(label)
    delivered_rows: list[tuple[str, str, int, int]] = []
    for line in lines[2:omitted_index]:
        if line == "- none":
            continue
        match = re.match(r"^- (.+):(\d+)-(\d+) revision=(\S+)$", line)
        if match is None:
            return ()
        path, start, end, revision = match.groups()
        delivered_rows.append((path, "" if revision == "unknown" else revision,
                               int(start), int(end)))
    paths = tuple(line[2:] for line in lines[omitted_index + 1:] if line.startswith("- "))
    canonical = _render_truthful_deferred_source_notice(
        paths, delivered_source_ranges=delivered_rows,
    )
    if label == _LEGACY_DEFERRED_SOURCE_PATHS_LABEL:
        canonical = canonical.replace(
            _DEFERRED_SOURCE_PATHS_LABEL,
            _LEGACY_DEFERRED_SOURCE_PATHS_LABEL,
            1,
        )
    if not paths or content != canonical:
        return ()
    return paths


def _parse_live_workspace_deferred_notice(content: str) -> tuple[str, ...]:
    """Parse only the exact one-line facts emitted for deferred source paths."""
    from kernel_workspace_truth import enabled as workspace_truth_enabled

    if not workspace_truth_enabled():
        return ()
    lines = content.splitlines()
    if (len(lines) < 3 or lines[0] != DEFERRED_SOURCE_NOTICE_HEADER
            or lines[1] != "Current live-workspace source status:"):
        return ()

    paths: list[str] = []
    for line in lines[2:]:
        match = None
        for pattern in _LIVE_WORKSPACE_DEFERRED_FACT_PATTERNS:
            match = pattern.fullmatch(line)
            if match is not None:
                break
        if match is None:
            return ()
        path = match.group("path").replace("\\", "/").strip().removeprefix("./")
        if not path or path in paths:
            return ()
        paths.append(path)
    canonical = "\n".join((
        DEFERRED_SOURCE_NOTICE_HEADER,
        "Current live-workspace source status:",
        *lines[2:],
    ))
    return tuple(paths) if content == canonical else ()


def _final_wire_proof_identity(
        receipt: Mapping[str, object],
) -> tuple[str, str, str, str] | None:
    authorization = receipt.get("workspace_authorization")
    authorization = authorization if isinstance(authorization, Mapping) else {}
    context_digest = str(receipt.get("context_digest", "")).strip()
    submitted_digest = str(receipt.get("submitted_http_bytes_digest", "")).strip()
    scope = str(authorization.get("scope", "")).strip()
    generation = str(authorization.get("generation", "")).strip()
    if (
        str(receipt.get("proof_status", "")).upper() != "VERIFIED"
        or receipt.get("proof_failures")
        or receipt.get("transport_identity") != "VERIFIED"
        or authorization.get("status") != "VERIFIED"
        or not scope or not generation
        or not context_digest or context_digest.upper() in {"UNKNOWN", "UNVERIFIED"}
        or not submitted_digest or submitted_digest.upper() in {"UNKNOWN", "UNVERIFIED"}
    ):
        return None
    return context_digest, submitted_digest, scope, generation


def _complete_source_path_delivery_is_verified(row: object) -> bool:
    if (not isinstance(row, Mapping)
            or str(row.get("status", "")).upper() != "COMPLETE"
            or str(row.get("proof_status", "")).upper() != "VERIFIED"):
        return False
    selected_units = row.get("selected_units")
    delivered_units = row.get("delivered_units")
    omitted_units = row.get("omitted_units")
    selected_handles = row.get("selected_unit_handles")
    delivered_handles = row.get("delivered_unit_handles")
    omitted_handles = row.get("omitted_unit_handles")
    return (
        type(selected_units) is int and selected_units > 0
        and type(delivered_units) is int and delivered_units == selected_units
        and type(omitted_units) is int and omitted_units == 0
        and isinstance(selected_handles, Sequence)
        and not isinstance(selected_handles, (str, bytes, bytearray))
        and bool(selected_handles)
        and isinstance(delivered_handles, Sequence)
        and not isinstance(delivered_handles, (str, bytes, bytearray))
        and isinstance(omitted_handles, Sequence)
        and not isinstance(omitted_handles, (str, bytes, bytearray))
        and not omitted_handles
        and len(selected_handles) == selected_units
        and tuple(selected_handles) == tuple(delivered_handles)
        and all(isinstance(handle, str) and handle.strip() for handle in selected_handles)
    )


def _delivery_proof_matches_final_wire(
        proof: object, path: str, identity: tuple[str, str, str, str],
) -> bool:
    if not isinstance(proof, Mapping):
        return False
    context_digest, submitted_digest, scope, generation = identity
    key = proof.get("key")
    return (
        proof.get("schema") == "stateframe-delivery-proof/1"
        and proof.get("request_id") == context_digest
        and proof.get("serialized_request_digest") == context_digest
        and proof.get("context_digest") == context_digest
        and proof.get("submitted_payload_hash") == context_digest
        and proof.get("submitted_http_bytes_digest") == submitted_digest
        and proof.get("transport_identity") == "VERIFIED"
        and proof.get("wire_contains_exact") is True
        and bool(str(proof.get("final_message_location", "")).strip())
        and isinstance(key, Mapping)
        and key.get("canonical_path") == path
        and key.get("workspace_scope") == scope
        and key.get("workspace_generation") == generation
    )


def _verified_complete_final_wire_paths(
        receipt: Mapping[str, object], selected_paths: set[str],
) -> set[str]:
    identity = _final_wire_proof_identity(receipt)
    path_results = receipt.get("paths")
    proof_rows_by_path = receipt.get("delivery_proofs")
    if (identity is None or not isinstance(path_results, Mapping)
            or not isinstance(proof_rows_by_path, Mapping)):
        return set()
    verified: set[str] = set()
    for path in selected_paths:
        row = path_results.get(path)
        proofs = proof_rows_by_path.get(path)
        if (not _complete_source_path_delivery_is_verified(row)
                or not isinstance(proofs, Sequence)
                or isinstance(proofs, (str, bytes, bytearray)) or not proofs):
            continue
        if all(_delivery_proof_matches_final_wire(proof, path, identity) for proof in proofs):
            verified.add(path)
    return verified


def _plan_omission_rows(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _plan_omission_is_relevant(
        row: object, selected_paths: set[str], covered_paths: set[str],
) -> bool:
    if not isinstance(row, Mapping):
        return True
    path = str(row.get("path", "")).strip()
    if not path:
        return True
    return (
        path not in covered_paths
        and (path in selected_paths or str(row.get("status", "")).upper() != "NOT_SELECTED")
    )


def _plan_has_relevant_omissions(
        plan: Mapping[str, object], selected_paths: set[str],
        verified_deferred_paths: set[str], verified_complete_paths: set[str],
) -> bool:
    covered_paths = verified_deferred_paths | verified_complete_paths
    return any(
        _plan_omission_is_relevant(row, selected_paths, covered_paths)
        for key in ("omissions", "skipped_files")
        for row in _plan_omission_rows(plan.get(key, ()))
    )


def _prune_verified_complete_path_omissions(
        plan: dict[str, object], verified_complete_paths: set[str],
) -> None:
    if not verified_complete_paths:
        return
    for key in ("omissions", "skipped_files"):
        rows = plan.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            continue
        retained = [
            row for row in rows
            if not isinstance(row, Mapping)
            or str(row.get("path", "")).strip() not in verified_complete_paths
        ]
        if len(retained) != len(rows):
            plan[key] = tuple(retained) if isinstance(rows, tuple) else retained


def _selected_delivery_paths_satisfied(
        path_results: Mapping[str, object], selected_paths: set[str],
) -> bool:
    if not selected_paths:
        return False
    for path in selected_paths:
        row = path_results.get(path)
        if (
            not isinstance(row, Mapping)
            or str(row.get("status", "")).upper() not in {"COMPLETE", "DEFERRED"}
            or str(row.get("proof_status", "")).upper() != "VERIFIED"
        ):
            return False
    return True


def _mark_complete_delivery_status(
        plan: Mapping[str, object], receipt: dict[str, object],
) -> None:
    receipt["status"] = "COMPLETE"
    if not isinstance(plan, dict):
        return
    plan["final_delivery_status"] = "COMPLETE"
    coverage = plan.get("coverage")
    delivered_span = coverage.get("delivered_span") if isinstance(coverage, dict) else None
    if isinstance(delivered_span, dict):
        delivered_span.update({
            "status": "COMPLETE", "deferred_paths": [],
            "deferred_units": 0, "proof_status": "VERIFIED",
        })


def _apply_deferred_delivery_status(
        plan: Mapping[str, object], receipt: dict[str, object],
        deferred_paths: Sequence[str], selected_paths: set[str],
        *, notice_verified: bool,
) -> tuple[str, ...]:
    if (receipt.get("proof_status") != "VERIFIED" or receipt.get("proof_failures")):
        return ()
    path_results = receipt.get("paths")
    if not isinstance(path_results, dict):
        return ()
    # A complete, exact-wire proof can discharge historical omission rows even
    # when this candidate has no deferred path and therefore needs no notice.
    verified_complete_set = _verified_complete_final_wire_paths(receipt, selected_paths)
    if isinstance(plan, dict):
        _prune_verified_complete_path_omissions(plan, verified_complete_set)
    if not notice_verified:
        if (
            selected_paths
            and selected_paths.issubset(verified_complete_set)
            and _selected_delivery_paths_satisfied(path_results, selected_paths)
            and not _plan_has_relevant_omissions(
                plan, selected_paths, set(), verified_complete_set,
            )
        ):
            _mark_complete_delivery_status(plan, receipt)
        return ()
    delivery = plan.get("delivery")
    revisions = plan.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    units_by_path: dict[str, list[str]] = {}
    for unit in plan.get("units", ()):
        if isinstance(unit, Mapping):
            normalized_paths = _normalized_deferred_paths((unit.get("path", ""),))
            handle = str(
                unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle") or ""
            ).strip()
            if normalized_paths:
                units_by_path.setdefault(normalized_paths[0], []).append(handle)
    verified: list[str] = []
    for path in deferred_paths:
        row = path_results.get(path)
        if path not in selected_paths or not isinstance(row, dict):
            continue
        if str(row.get("status", "")).upper() == "COMPLETE":
            continue
        delivery_row = delivery.get(path) if isinstance(delivery, Mapping) else None
        route = row.get("retrieval_route")
        if not isinstance(route, Mapping) and isinstance(delivery_row, Mapping):
            route = delivery_row.get("retrieval_route")
        revision = str(revisions.get(path, "")).strip()
        unit_handles = units_by_path.get(path, [])
        if (not _executable_deferred_read_route(path, revision, route)
                or not unit_handles or any(not handle for handle in unit_handles)):
            continue
        retrieval_route = dict(route)
        retrieval_route.update({
            "path": path,
            "revision": revision,
            "unit_handles": unit_handles,
        })
        row.update({
            "status": "DEFERRED",
            "proof_status": "VERIFIED",
            "deferred_notice_verified": True,
            "retrieval_route": retrieval_route,
        })
        if isinstance(delivery_row, dict):
            delivery_row.update({
                "status": "DEFERRED",
                "proof_status": "VERIFIED",
                "deferred_notice_verified": True,
                "retrieval_route": retrieval_route,
            })
        verified.append(path)
    if not verified:
        return ()

    verified_set = set(verified)
    has_omissions = _plan_has_relevant_omissions(
        plan, selected_paths, verified_set, verified_complete_set,
    )
    all_satisfied = _selected_delivery_paths_satisfied(path_results, selected_paths)
    if all_satisfied and not has_omissions:
        receipt["status"] = "DEFERRED"
        coverage = plan.get("coverage")
        if isinstance(coverage, dict):
            delivered_span = coverage.get("delivered_span")
            if isinstance(delivered_span, dict):
                delivered_span.update({
                    "status": "DEFERRED",
                    "deferred_paths": list(verified),
                    "deferred_units": sum(len(units_by_path.get(path, ())) for path in verified),
                    "proof_status": "VERIFIED",
                })
        if isinstance(plan, dict):
            _prune_verified_complete_path_omissions(plan, verified_complete_set)
            plan["final_delivery_status"] = "DEFERRED"
    return tuple(verified)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_stat_token(root: Path, relative: str) -> tuple[object, ...]:
    try:
        stat = (root / relative).stat()
    except OSError:
        return (relative, None, None, None)
    return (relative, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _line_count(source: str) -> int:
    return len(source.splitlines(keepends=True))


def _path_from_runtime(runtime: object) -> str | None:
    target = getattr(runtime, "target", None)
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    if not isinstance(target, Path) or root is None:
        return None
    try:
        return target.resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, TypeError, ValueError):
        return None


def _ticket_text(runtime: object) -> str:
    identity = ticket_identity_for_runtime(runtime)
    if identity is not None:
        return identity.text
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    values: list[str] = []
    if isinstance(row, Mapping):
        for key in (
                "localized_target", "target", "entry", "problem_statement", "ticket_text",
                "public_brief", "brief_text", "symptom", "clause", "msg", "brief"):
            value = row.get(key)
            if isinstance(value, Mapping):
                values.extend(str(item) for item in value.values() if isinstance(item, str))
            elif isinstance(value, str):
                values.append(value)
    information = getattr(owner, "information_policy", None)
    ordinary_brief = getattr(information, "ordinary_brief", None)
    if isinstance(ordinary_brief, str):
        values.append(ordinary_brief)
    return "\n".join(dict.fromkeys(value.strip() for value in values if value.strip()))


def _target_names(runtime: object, ticket: str) -> tuple[str, ...]:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    values: list[str] = []
    for value in (getattr(owner, "fn", None),):
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    if isinstance(row, Mapping):
        for key in ("fn", "symbol", "symbol_name", "target_fn", "localized_symbol"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    # Explicit calls and quoted identifiers are safer anchors than arbitrary
    # ticket words.  The broad identifier fallback is used only when the
    # localized owner did not supply a symbol.
    values.extend(match.group(1) for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(", ticket))
    values.extend(match.group(2) for match in re.finditer(
        r"(['\"])([^'\"\n]+)(?:['\"])", ticket))
    return tuple(dict.fromkeys(value for value in values if value))


@profiled_function_span("source_anchor_resolution")
def _anchor_ranges(source: str, runtime: object, ticket: str, *,
                   path: str = "", is_target: bool = True) -> tuple[tuple[int, int, str], ...]:
    lines = source.splitlines(keepends=True)
    total = len(lines)
    anchors: list[tuple[int, int, str]] = []
    for name in _target_names(runtime, ticket):
        located = locate_symbol(source, name)
        if located is not None:
            anchors.append((located[0], located[1], f"symbol:{name}"))
            continue
        short_name = name.rsplit(".", 1)[-1]
        for index, line in enumerate(lines, 1):
            if re.search(rf"\b{re.escape(short_name)}\b", line):
                anchors.append((index, index, f"text:{short_name}"))
                break
    entry_absent = False
    if site_facts_mode() == "on":
        entry = entry_symbol(runtime)
        # A declared entry symbol that this target never mentions is a phantom
        # (P3): the ticket-word fallback below would then anchor on an
        # arbitrary line, so it is suppressed in favour of the module head.
        entry_absent = bool(is_target and entry and locate_symbol(source, entry) is None
                            and not re.search(rf"\b{re.escape(entry.rsplit('.', 1)[-1])}\b",
                                              source))
        literal_hits = 0
        for token in literal_tokens(ticket, exclude=(entry,)):
            pattern = _literal_pattern(token)
            for index, line in enumerate(lines, 1):
                if literal_hits >= _MAX_LITERAL_ANCHORS:
                    break
                if not _is_comment(line) and pattern.search(line):
                    anchors.append((index, index, f"literal:{token}"))
                    literal_hits += 1
        # The closure planner owns complete statement/body spans through the
        # shared outline contract.  Source-window fitting only projects those
        # already-authorized units; it does not create a second parser/index.
        plan = closure_plan_for_runtime(runtime)
        for unit in units_for_path(plan, path):
            # A unit whose lines are all blank carries no code; anchoring on it
            # produced the live "blank line 22 of 599" window (J857).
            if not "".join(lines[max(0, unit.start_line - 1):unit.end_line]).strip():
                continue
            anchors.append((unit.start_line, unit.end_line, f"closure:{unit.role}"))
        if is_target and (entry_absent or _MODULE_HEAD_TRIGGER.search(ticket)):
            head_end = module_head_end(source, path)
            if head_end >= 1:
                anchors.append((1, head_end, "module-head"))
    if not anchors and not entry_absent:
        words = tuple(dict.fromkeys(
            word for word in _IDENTIFIER.findall(ticket)
            if word.casefold() not in {"the", "and", "for", "with", "must", "should", "return"}
        ))
        scored: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines, 1):
            score = sum(bool(re.search(rf"\b{re.escape(word)}\b", line, re.IGNORECASE))
                        for word in words)
            if score:
                scored.append((score, index, line.strip()[:80]))
        if scored:
            score, index, label = max(scored, key=lambda row: (row[0], -row[1]))
            anchors.append((index, index, f"ticket:{label}"))
    anchors = [
        anchor for anchor in anchors
        if any(
            line.strip()
            for line in lines[max(0, anchor[0] - 1):min(total, anchor[1])]
        )
    ]
    if not anchors and total:
        first_nonblank = next(
            (index for index, line in enumerate(lines, 1) if line.strip()), None,
        )
        if first_nonblank is not None:
            anchors.append((first_nonblank, first_nonblank, "fallback:first-nonblank-line"))
    return tuple(anchors)


def _merge_ranges(
        anchors: Sequence[tuple[int, int, str]], total: int,
        context_lines: int = _DEFAULT_CONTEXT_LINES,
) -> list[dict[str, object]]:
    expanded = sorted((max(1, start - context_lines), min(total, end + context_lines), label)
                      for start, end, label in anchors)
    merged: list[dict[str, object]] = []
    for start, end, label in expanded:
        if merged and start <= int(merged[-1]["end_line"]) + 1:
            merged[-1]["end_line"] = max(int(merged[-1]["end_line"]), end)
            merged[-1]["anchors"].append(label)
            continue
        merged.append({"start_line": start, "end_line": end, "anchors": [label]})
    return merged


def _render_ranges(
        source_lines: Sequence[str], ranges: Sequence[Mapping[str, object]],
        *, path: str, revision: str, total_lines: int,
        page_max_lines: int = _MAX_PAGE_LINES,
        max_bytes: int | None = None,
) -> str:
    normalized: list[dict[str, object]] = []
    chunks: list[str] = []
    for row in ranges:
        start = int(row["start_line"])
        end = int(row["end_line"])
        text = "".join(source_lines[start - 1:end])
        normalized.append({
            "start_line": start,
            "end_line": end,
            "anchors": list(row.get("anchors", ())),
            "trailing_newline": text.endswith("\n"),
        })
        chunks.append(f"--- source lines {start}-{end} ---\n{text}"
                      + ("" if text.endswith("\n") else "\n"))
    covered_lines = sum(int(row["end_line"]) - int(row["start_line"]) + 1
                        for row in normalized)
    omitted_lines = covered_lines < total_lines
    continuation = {
        "status": "CONTINUATION_AVAILABLE" if omitted_lines else "COMPLETE",
        "omitted_lines": max(0, total_lines - covered_lines),
        "expansion_handle": (
            "expand:" + hashlib.sha256(json.dumps({
                "path": path, "revision": revision, "ranges": normalized,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if omitted_lines else None
        ),
    }
    metadata = {
        "schema": SCHEMA,
        "path": path,
        "revision": revision,
        "sha256": revision.removeprefix("sha256:"),
        "total_lines": total_lines,
        "ranges": normalized,
        "omitted_lines": omitted_lines,
        "continuation_receipt": continuation,
        "page": {
            "tool": "read",
            "path": path,
            "revision": revision,
            "max_lines": page_max_lines,
            "max_chars": _MAX_PAGE_CHARS,
            "full_revision_required": True,
        },
    }
    def envelope(fields: Mapping[str, object]) -> str:
        return (WINDOW_MARKER + "\n"
                + json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n" + WINDOW_CONTENT_MARKER + "\n"
                + "".join(chunks) + WINDOW_END_MARKER)

    full = envelope(metadata)
    if max_bytes is None or len(full.encode("utf-8")) <= max_bytes:
        return full
    # A tight request need not repeat the digest, path, and revision in its
    # page descriptor. parse_window restores those deterministic fields before
    # the existing exact-source and provider-edge validators inspect them.
    compact = {key: value for key, value in metadata.items()
               if key not in {"sha256", "page"}}
    compact["encoding"] = "compact-page/1"
    smaller = envelope(compact)
    return smaller if len(smaller.encode("utf-8")) < len(full.encode("utf-8")) else full


def project_source_window(content: object, *, owner: str = "UNKNOWN_OWNER") -> str | None:
    """Project a verified source window without exposing host bookkeeping.

    The returned body is byte-for-byte identical for every displayed source
    range.  Revision/hash/page data stays in the host manifest and is checked
    by :func:`verify_final_source_delivery`; the continuation handle remains a
    real read route, not display advice.
    """
    if not isinstance(content, str):
        return None
    metadata = parse_window(content)
    if metadata is None:
        return None
    path = str(metadata.get("path", "")).replace("\\", "/")
    if not path:
        return None
    continuation = metadata.get("continuation_receipt")
    continuation_map = dict(continuation) if isinstance(continuation, Mapping) else {
        "status": "UNKNOWN", "omitted_lines": None, "expansion_handle": None,
    }
    status = str(continuation_map.get("status", "UNKNOWN"))
    ranges = metadata.get("ranges")
    compact_ranges = [
        {
            key: row[key]
            for key in ("start_line", "end_line", "anchors", "trailing_newline")
            if key in row
        }
        for row in (ranges if isinstance(ranges, list) else ())
        if isinstance(row, Mapping)
    ]
    projection = {
        "schema": MODEL_WINDOW_SCHEMA,
        "path": path,
        "owner": str(owner or "UNKNOWN_OWNER"),
        "revision": metadata.get("revision"),
        "sha256": metadata.get("sha256"),
        "ranges": compact_ranges,
        "total_lines": metadata.get("total_lines"),
        "status": status,
        "continuation": continuation_map,
        "absence": "NONE",
    }
    segments = _window_segments(content, metadata)
    chunks = []
    for row, text in segments:
        start, end = int(row["start_line"]), int(row["end_line"])
        chunks.append(f"--- source lines {start}-{end} ---\n{text}"
                      + ("" if text.endswith("\n") else "\n"))
    # The public source header leads, as for every other source block: the
    # presentation contract, diet protection and refusal compaction classify
    # source messages by this prefix.  parse_model_source_window finds the
    # marker within the first three lines.
    header = f"[Current source: {path}]"
    return (header + "\n" + MODEL_WINDOW_MARKER + "\n"
            + json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n" + WINDOW_CONTENT_MARKER + "\n"
            + "".join(chunks) + WINDOW_END_MARKER)


def project_source_absence(*, path: str, owner: str, reason: str = "NOT_FOUND") -> str:
    """Render scoped absence without pretending a source body was delivered."""
    metadata = {
        "schema": MODEL_WINDOW_SCHEMA,
        "path": path.replace("\\", "/"),
        "owner": owner or "UNKNOWN_OWNER",
        "ranges": [],
        "total_lines": 0,
        "status": "ABSENT",
        "continuation": {"status": "UNAVAILABLE", "omitted_lines": None, "expansion_handle": None},
        "absence": reason,
    }
    return (MODEL_WINDOW_MARKER + "\n"
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n" + WINDOW_END_MARKER)


def parse_model_source_window(content: object) -> dict[str, object] | None:
    if not isinstance(content, str):
        return None
    lines = content.splitlines(keepends=True)
    marker_index = next((index for index, line in enumerate(lines[:3])
                         if line.rstrip("\r\n") == MODEL_WINDOW_MARKER), None)
    if marker_index is None or marker_index + 1 >= len(lines):
        return None
    try:
        value = json.loads(lines[marker_index + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("schema") == MODEL_WINDOW_SCHEMA else None


def _model_window_segments(content: str, metadata: Mapping[str, object]) -> list[tuple[dict[str, object], str]]:
    lines = content.splitlines(keepends=True)
    try:
        start = next(index for index, line in enumerate(lines)
                     if line.rstrip("\r\n") == WINDOW_CONTENT_MARKER) + 1
    except StopIteration:
        return []
    segments: list[tuple[dict[str, object], str]] = []
    current: dict[str, object] | None = None
    buffer: list[str] = []
    for line in lines[start:]:
        match = _RANGE.match(line.rstrip("\r\n"))
        if match:
            if current is not None:
                segments.append((current, "".join(buffer)))
            current = {"start_line": int(match.group(1)), "end_line": int(match.group(2))}
            buffer = []
            continue
        if line.rstrip("\r\n") == WINDOW_END_MARKER:
            break
        if current is not None:
            buffer.append(line)
    if current is not None:
        segments.append((current, "".join(buffer)))
    declared = metadata.get("ranges")
    if not isinstance(declared, list) or len(declared) != len(segments):
        return []
    return [({**dict(declared[index]), **row}, text)
            for index, (row, text) in enumerate(segments)
            if isinstance(declared[index], Mapping)]


def verify_model_source_window(
        content: object, *, path: str, source: str | None = None, owner: str | None = None,
) -> bool:
    """Verify compact projection structure and, when present, exact source bytes."""
    if not isinstance(content, str):
        return False
    metadata = parse_model_source_window(content)
    if metadata is None or str(metadata.get("path", "")).replace("\\", "/") != path.replace("\\", "/"):
        return False
    if owner is not None and metadata.get("owner") != owner:
        return False
    status = str(metadata.get("status", ""))
    ranges = metadata.get("ranges")
    total_lines = metadata.get("total_lines")
    continuation = metadata.get("continuation")
    if (not isinstance(metadata.get("owner"), str) or not metadata["owner"]
            or not isinstance(ranges, list) or type(total_lines) is not int
            or total_lines < 0 or not isinstance(continuation, Mapping)):
        return False
    if status == "PARTIAL_READ":
        partial = metadata.get("partial_read")
        if not isinstance(partial, Mapping) or ranges:
            return False
        start, end = partial.get("start_line"), partial.get("end_line")
        start_column, end_column = partial.get("start_column"), partial.get("end_column")
        length, expected_hash = partial.get("chars"), partial.get("content_sha256")
        if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= total_lines
                or type(start_column) is not int or start_column < 0
                or (end_column is not None and (type(end_column) is not int or end_column < 0))
                or type(length) is not int or length < 1
                or not isinstance(expected_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None):
            return False
        marker = "\n" + WINDOW_CONTENT_MARKER + "\n"
        offset = content.find(marker)
        if offset < 0:
            return False
        offset += len(marker)
        actual = content[offset:offset + length]
        ending = "" if actual.endswith("\n") else "\n"
        if (hashlib.sha256(actual.encode()).hexdigest() != expected_hash
                or content[offset + length:] != ending + WINDOW_END_MARKER):
            return False
        if source is not None:
            if metadata.get("revision") != source_revision(source):
                return False
            lines = source.splitlines(keepends=True)[start - 1:end]
            if start == end:
                expected = lines[0][start_column:end_column]
            else:
                lines[0] = lines[0][start_column:]
                if end_column is not None:
                    lines[-1] = lines[-1][:end_column]
                expected = "".join(lines)
            if actual != expected:
                return False
        return True
    if status == "ABSENT":
        return not ranges and metadata.get("absence") not in (None, "", "NONE")
    if status not in {"COMPLETE", "CONTINUATION_AVAILABLE"} or not ranges:
        return False
    segments = _model_window_segments(content, metadata)
    if len(segments) != len(ranges):
        return False
    previous_end = 0
    covered_lines = 0
    source_lines = source.splitlines(keepends=True) if source is not None else None
    for row, actual in segments:
        start, end = row.get("start_line"), row.get("end_line")
        if (type(start) is not int or type(end) is not int or start < 1 or end < start
                or start <= previous_end or end > total_lines or not actual):
            return False
        trailing_newline = row.get("trailing_newline")
        if type(trailing_newline) is not bool:
            return False
        actual_source = actual
        if not trailing_newline and actual.endswith("\n"):
            # The range separator contributes a framing newline when the
            # selected source line itself has no terminal newline.
            actual_source = actual[:-1]
        elif trailing_newline and not actual.endswith("\n"):
            return False
        if source_lines is not None:
            if end > len(source_lines) or "".join(source_lines[start - 1:end]) != actual_source:
                return False
        covered_lines += end - start + 1
        previous_end = end
    if (covered_lines == total_lines) != (status == "COMPLETE"):
        return False
    handle = continuation.get("expansion_handle")
    if status == "CONTINUATION_AVAILABLE":
        if not isinstance(handle, str) or not handle.startswith("expand:"):
            return False
        if source is not None:
            expected = "expand:" + hashlib.sha256(json.dumps({
                "path": path.replace("\\", "/"), "revision": source_revision(source),
                "ranges": [dict(row) for row, _text in segments],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if handle != expected:
                return False
    return True


def _allocated_prefix_snapshot(
        layout: Mapping[str, object], prefix_start: int, suffix_start: int,
) -> dict[int, Mapping[str, object]]:
    from kernel_stateframe_framefit5 import FrameFit5Error

    facts = layout.get("current_user_facts")
    carriers = layout.get("carriers")
    if not isinstance(facts, list) or not isinstance(carriers, list):
        raise FrameFit5Error("frame-only source allocation has invalid current facts or carriers")
    expected: dict[int, Mapping[str, object]] = {}
    for fact in facts:
        if not isinstance(fact, Mapping):
            raise FrameFit5Error("frame-only source allocation contains an invalid fact")
        expected[fact.get("location")] = fact.get("message")
    for carrier in carriers:
        if not isinstance(carrier, Mapping):
            raise FrameFit5Error("frame-only source allocation contains an invalid carrier")
        if carrier.get("kind") == "prepared_source":
            expected[carrier.get("location")] = carrier.get("message")
    if (suffix_start < prefix_start
            or sorted(expected) != list(range(prefix_start, suffix_start))):
        raise FrameFit5Error("frame-only source allocation prefix locations are not contiguous")
    return expected


def _verify_allocated_carriers(
        messages: Sequence[Mapping[str, object]],
        carriers: Sequence[object],
) -> None:
    from kernel_stateframe_framefit5 import FrameFit5Error

    for carrier in carriers:
        if not isinstance(carrier, Mapping):
            raise FrameFit5Error("frame-only source allocation contains an invalid carrier")
        location = carrier.get("location")
        message = carrier.get("message")
        if (type(location) is not int or not isinstance(message, Mapping)
                or location >= len(messages) or messages[location] != message):
            raise FrameFit5Error("frame-only source carrier moved after allocation")
        matches = [
            index for index, candidate in enumerate(messages) if candidate == message
        ]
        if matches != [location]:
            raise FrameFit5Error("frame-only allocated source carrier is duplicated")


def _validate_frame_request_allocation_placement(
        messages: Sequence[Mapping[str, object]],
        allocation: FrameRequestAllocation,
) -> None:
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_stateframe_framefit5 import FrameFit5Error, _is_volatile_tail_message

    try:
        allocation.validate()
    except FrameRequestAllocationError as exc:
        raise FrameFit5Error("frame-only source allocation failed integrity validation") from exc
    layout = allocation.source_layout
    task = layout.get("task_message")
    suffix = layout.get("newest_user_suffix")
    suffix_start = layout.get("newest_user_suffix_start")
    exchange = layout.get("newest_tool_exchange")
    exchange_start = layout.get("newest_exchange_start")
    carriers = layout.get("carriers")
    if (not isinstance(task, Mapping) or not isinstance(suffix, list)
            or type(suffix_start) is not int or not isinstance(exchange, list)
            or type(exchange_start) is not int or not isinstance(carriers, list)):
        raise FrameFit5Error("frame-only source allocation has an invalid layout")

    first_non_system = next(
        (index for index, message in enumerate(messages)
         if message.get("role") != "system"),
        len(messages),
    )
    state_index = first_non_system + 1
    if (first_non_system >= len(messages)
            or messages[first_non_system] != task
            or state_index >= len(messages)
            or not str(messages[state_index].get("content") or "").startswith(
                STATE_HEADER + "\n",
            )):
        raise FrameFit5Error("frame-only source allocation task or StateFrame moved")

    state_indexes = [
        index for index, message in enumerate(messages)
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    ]
    if state_indexes != [state_index]:
        raise FrameFit5Error("frame-only source allocation requires one StateFrame")
    prefix_start = state_index + 1
    expected_prefix = _allocated_prefix_snapshot(layout, prefix_start, suffix_start)
    for location, message in expected_prefix.items():
        if (type(location) is not int or not isinstance(message, Mapping)
                or location >= len(messages) or messages[location] != message):
            raise FrameFit5Error("frame-only allocated fact/source carrier moved")

    suffix_end = suffix_start + len(suffix)
    if (suffix_end != exchange_start
            or messages[suffix_start:suffix_end] != suffix):
        raise FrameFit5Error("frame-only newest-user suffix moved after allocation")
    exchange_end = exchange_start + len(exchange)
    if messages[exchange_start:exchange_end] != exchange:
        raise FrameFit5Error("frame-only newest tool exchange moved after allocation")
    action_messages = [
        message for message in messages
        if message.get("role") in {"assistant", "tool"}
    ]
    if action_messages != exchange:
        raise FrameFit5Error("frame-only source allocation changed the newest action exchange")
    _verify_allocated_carriers(messages, carriers)
    if (exchange_end != len(messages) - 1
            or not _is_volatile_tail_message(messages[-1])):
        raise FrameFit5Error("frame-only source allocation lost its final volatile tail")


@profiled_function_span("source_window_projection")
def project_model_source_messages(
        messages: list[dict[str, object]], plan: Mapping[str, object] | None = None,
        *, allocation_plan: dict[str, object] | None = None,
        materialize_framefit5_closure_units: bool = False,
        sources: Mapping[str, str] | None = None,
        frame_request_allocation: FrameRequestAllocation | None = None,
        preserve_source_messages: Sequence[Mapping[str, object]] = (),
) -> int:
    """Apply the compact projection at the final provider-facing boundary."""
    if frame_request_allocation is not None:
        _append_framefit5_closure_source_messages(
            messages, plan, sources, frame_request_allocation=frame_request_allocation,
        )
        return 0
    if allocation_plan is None:
        from kernel_owner_first import allocation_enabled, build_priority_plan
        if allocation_enabled():
            from types import SimpleNamespace
            allocation_plan = build_priority_plan(
                {"messages": messages, "tools": []},
                SimpleNamespace(_request_source_closure=plan),
            )
    owners: dict[str, str] = {}
    if isinstance(plan, Mapping):
        for unit in plan.get("units", ()):
            if not isinstance(unit, Mapping):
                continue
            unit_path = str(unit.get("path", "")).replace("\\", "/")
            owner = unit.get("owner", unit.get("symbol_owner", unit.get("enclosing", unit.get("symbol", ""))))
            if unit_path and str(owner or "").strip():
                owners.setdefault(unit_path, str(owner))
    changed = 0
    for message in messages:
        if message in preserve_source_messages:
            continue
        content = message.get("content") if isinstance(message, Mapping) else None
        metadata = parse_window(content)
        if metadata is None or not isinstance(content, str):
            continue
        if (evictfix_enabled()
                and not _standalone_raw_window_envelope(content, metadata)):
            continue
        path = str(metadata.get("path", "")).replace("\\", "/")
        projected = project_source_window(content, owner=owners.get(path, "UNKNOWN_OWNER"))
        if projected is not None and projected != content:
            message["content"] = projected
            changed += 1
    if materialize_framefit5_closure_units:
        changed += _append_framefit5_closure_source_messages(messages, plan, sources)
        changed += _order_framefit5_source_messages(
            messages, plan, preserve_source_messages=preserve_source_messages,
        )
    if evictfix_enabled():
        changed += _coalesce_generated_source_windows(
            messages, plan, sources, preserve_source_messages=preserve_source_messages,
        )
    if allocation_plan is not None:
        from kernel_owner_first import allocation_enabled, finalize_range_plan
        if allocation_enabled():
            finalize_range_plan(allocation_plan, messages)
    return changed


def _standalone_raw_window_envelope(
        content: str, metadata: Mapping[str, object],
) -> bool:
    """Do not project away text outside a raw source window's owned envelope."""
    path = metadata.get("path")
    if not isinstance(path, str) or not path:
        return False
    lines = content.splitlines()
    marker_index = 0 if lines and lines[0] == WINDOW_MARKER else 1
    if marker_index == 1 and (not lines or lines[0] not in {
            f"[{label} source: {path}]" for label in ("Current", "Working")
    }):
        return False
    return (
        len(lines) >= marker_index + 5
        and lines[marker_index] == WINDOW_MARKER
        and lines[marker_index + 2] == WINDOW_CONTENT_MARKER
        and _RANGE.fullmatch(lines[marker_index + 3]) is not None
        and lines[-1] == WINDOW_END_MARKER
        and content.endswith(WINDOW_END_MARKER)
        and sum(line == WINDOW_END_MARKER for line in lines) == 1
    )


def _standalone_window_envelope(content: str, path: str, marker: str,
                                *, labels: tuple[str, ...]) -> bool:
    lines = content.splitlines()
    return (
        len(lines) >= 5
        and lines[0] in {f"[{label} source: {path}]" for label in labels}
        and lines[1] == marker
        and lines[3] == WINDOW_CONTENT_MARKER
        and _RANGE.fullmatch(lines[4]) is not None
        and lines[-1] == WINDOW_END_MARKER
        and content.endswith(WINDOW_END_MARKER)
        and sum(line == WINDOW_END_MARKER for line in lines) == 1
    )


def _generated_window_ranges(
        content: str, *, path: str, revision: str, source: str,
) -> tuple[list[dict[str, object]], str] | None:
    """Accept only current, byte-verified standalone source carriers."""
    model = parse_model_source_window(content)
    if isinstance(model, Mapping):
        if (not _standalone_window_envelope(
                    content, path, MODEL_WINDOW_MARKER, labels=("Current",))
                or model.get("path") != path or model.get("revision") != revision
                or not verify_model_source_window(content, path=path, source=source)):
            return None
        rows = model.get("ranges")
        owner = str(model.get("owner") or "UNKNOWN_OWNER")
    else:
        raw = parse_window(content)
        if isinstance(raw, Mapping):
            if (not _standalone_window_envelope(
                        content, path, WINDOW_MARKER, labels=("Current", "Working"))
                    or raw.get("path") != path or raw.get("revision") != revision
                    or not verify_window(content, path=path, source=source)):
                return None
            rows = raw.get("ranges")
            owner = "UNKNOWN_OWNER"
        else:
            # F4 emits metadata blocks and plain Coverage trailers. Both must
            # match the pinned source bytes before a whole message is replaced.
            header, separator, remainder = content.partition("\n")
            if not separator:
                return None
            if (header not in {f"[{label} source: {path}]" for label in
                               ("Current", "Working", "Observed", "Retained")}):
                return None
            lines = source.splitlines(keepends=True)
            if remainder.startswith("[Source metadata]\n"):
                encoded, separator, remainder = remainder[len("[Source metadata]\n"):].partition("\n")
                if not separator:
                    return None
                try:
                    metadata = json.loads(encoded)
                except (TypeError, ValueError):
                    return None
                if (not isinstance(metadata, Mapping)
                        or metadata.get("path") != path
                        or metadata.get("revision") != revision):
                    return None
                start = metadata.get("start_line")
                end = metadata.get("selected_end_line", metadata.get("end_line"))
                prefix = f"[Source content: {path}]\n"
                suffix = f"\n[End source content: {path}]"
                if not remainder.startswith(prefix) or not remainder.endswith(suffix):
                    return None
                body = remainder[len(prefix):-len(suffix)]
                expected = "".join(lines[start - 1:end]) if (
                    type(start) is int and type(end) is int and 1 <= start <= end <= len(lines)
                ) else None
                trailing = body[len(expected):] if expected is not None and body.startswith(expected) else None
                coverage = (rf"\nCoverage: lines {start}-{end}, columns 0-0; "
                            r"current bytes are delivered for this range\.")
                if trailing is None or not (
                        trailing in {"", "\n"}
                        or re.fullmatch(coverage + r"\n?", trailing)):
                    return None
            else:
                trailer = re.search(
                    r"\nCoverage: lines (\d+)-(\d+), columns 0-0; "
                    r"current bytes are delivered for this range\.\Z", remainder,
                )
                if trailer is None:
                    return None
                start, end = int(trailer.group(1)), int(trailer.group(2))
                if not 1 <= start <= end <= len(lines):
                    return None
                expected = "".join(lines[start - 1:end])
                if remainder[:trailer.start()] not in {expected, expected + "\n"}:
                    return None
            rows = [{"start_line": start, "end_line": end, "anchors": ["observed"]}]
            owner = "UNKNOWN_OWNER"
    if not isinstance(rows, list) or not rows:
        return None
    ranges = [dict(row) for row in rows if isinstance(row, Mapping)]
    if (len(ranges) != len(rows)
            or any(type(row.get("start_line")) is not int
                   or type(row.get("end_line")) is not int for row in ranges)):
        return None
    if model is not None or raw is not None:
        rendered = [match for line in content.splitlines()
                    if (match := _RANGE.fullmatch(line)) is not None]
        if (len(rendered) != len(ranges)
                or any((row["start_line"], row["end_line"]) !=
                       (int(marker.group(1)), int(marker.group(2)))
                       for row, marker in zip(ranges, rendered))):
            return None
    return ranges, owner


def _coalesce_generated_source_windows(
        messages: list[dict[str, object]], plan: Mapping[str, object] | None,
        sources: Mapping[str, str] | None, *,
        preserve_source_messages: Sequence[Mapping[str, object]] = (),
) -> int:
    """Carry each path/revision's union once without changing protected input."""
    if not isinstance(plan, Mapping):
        return 0
    revisions = plan.get("revisions")
    if not isinstance(revisions, Mapping):
        return 0
    # Deferred closures can carry independently retained read pages. Their
    # visibility receipts bind exact ranges, so a larger union is not the same
    # delivery identity. Keep those carriers until capacity explicitly spends
    # them; ordinary closure windows remain eligible for coalescing.
    deferred_paths = set(_normalized_deferred_paths(plan.get("deferred_paths", ())))
    source_map = dict(plan.get("source_contents", {})) if isinstance(
        plan.get("source_contents"), Mapping) else {}
    if isinstance(sources, Mapping):
        source_map.update(sources)
    groups: dict[tuple[str, str], list[tuple[int, list[dict[str, object]], str]]] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "user" or message in preserve_source_messages:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for path, revision in revisions.items():
            if path in deferred_paths:
                continue
            source = source_map.get(path)
            if (not isinstance(path, str) or not isinstance(revision, str)
                    or not isinstance(source, str)
                    or source_revision(source) != revision):
                continue
            candidate = _generated_window_ranges(
                content, path=path, revision=revision, source=source,
            )
            if candidate is not None:
                ranges, owner = candidate
                groups.setdefault((path, revision), []).append((index, ranges, owner))
                break
    replacements: dict[int, str] = {}
    removals: set[int] = set()
    for (path, revision), group in groups.items():
        if len(group) < 2:
            continue
        source = source_map[path]
        source_lines = source.splitlines(keepends=True)
        merged = _merge_closure_source_ranges([
            row for _index, ranges, _owner in group for row in ranges
        ])
        owners = {owner for _index, _ranges, owner in group}
        owner = next(iter(owners)) if len(owners) == 1 else "MULTIPLE_OWNERS"
        projected = project_source_window(_render_ranges(
            source_lines, merged, path=path, revision=revision,
            total_lines=len(source_lines),
        ), owner=owner)
        if projected is None or not verify_model_source_window(
                projected, path=path, source=source):
            continue
        original_bytes = sum(len(str(messages[index]["content"]).encode("utf-8"))
                             for index, _ranges, _owner in group)
        # F4's metadata blocks are cheaper as one exact line span than as a
        # model-window JSON envelope. The final source verifier already proves
        # this canonical Coverage trailer against the same pinned revision.
        if len(projected.encode("utf-8")) >= original_bytes and len(merged) == 1:
            start, end = int(merged[0]["start_line"]), int(merged[0]["end_line"])
            body = "".join(source_lines[start - 1:end])
            projected = (f"[Current source: {path}]\n{body}\n"
                         f"Coverage: lines {start}-{end}, columns 0-0; "
                         "current bytes are delivered for this range.")
        if len(projected.encode("utf-8")) >= original_bytes:
            continue
        if _generated_window_ranges(projected, path=path, revision=revision,
                                    source=source) is None:
            continue
        first = group[0][0]
        replacements[first] = projected
        removals.update(index for index, _ranges, _owner in group[1:])
    for index, projected in replacements.items():
        messages[index]["content"] = projected
    for index in sorted(removals, reverse=True):
        del messages[index]
    return len(replacements) + len(removals)


def _order_framefit5_source_messages(
        messages: list[dict[str, object]], plan: Mapping[str, object] | None,
        *, preserve_source_messages: Sequence[Mapping[str, object]] = (),
) -> int:
    """Order prepared carriers by closure relevance, then exact source identity.

    Reconstructed and surviving windows have equal authority; insertion history
    cannot choose their order. Only source slots before the newest exchange move.
    """
    if not isinstance(plan, Mapping) or not _is_framefit5_closure_projection_input(messages):
        return 0
    ranks = {path: index for index, path in enumerate(dict.fromkeys(plan.get("paths", ())))}
    slots = []
    candidates = []
    for index, message in enumerate(messages):
        if message.get("role") in {"assistant", "tool"}:
            break
        if message.get("role") != "user" or message in preserve_source_messages:
            continue
        metadata = parse_model_source_window(message.get("content"))
        if metadata is None:
            continue
        path, revision = str(metadata.get("path", "")), str(metadata.get("revision", ""))
        slots.append(index)
        candidates.append((
            (ranks.get(path, len(ranks)), path, revision,
             json.dumps(message, ensure_ascii=False, sort_keys=True)),
            message,
        ))
    ordered = [message for _key, message in sorted(candidates, key=lambda row: row[0])]
    changed = sum(messages[index] != message for index, message in zip(slots, ordered))
    for index, message in zip(slots, ordered):
        messages[index] = message
    return changed


def _verified_source_message_ranges(
        content: str, *, path: str, source: str, revision: str,
) -> tuple[tuple[int, int], ...]:
    model_metadata = parse_model_source_window(content)
    if isinstance(model_metadata, Mapping):
        model_path = str(model_metadata.get("path", "")).replace("\\", "/")
        if (model_path != path
                or str(model_metadata.get("revision", "")) != revision
                or not verify_model_source_window(content, path=path, source=source)):
            return ()
        raw_ranges = model_metadata.get("ranges")
        return tuple(
            (int(row["start_line"]), int(row["end_line"]))
            for row in raw_ranges if isinstance(row, Mapping)
            and type(row.get("start_line")) is int
            and type(row.get("end_line")) is int
        ) if isinstance(raw_ranges, list) else ()

    metadata = parse_window(content)
    if not isinstance(metadata, Mapping):
        return ()
    if (str(metadata.get("path", "")).replace("\\", "/") != path
            or str(metadata.get("revision", "")) != revision):
        return ()
    framed = content
    content_lines = content.splitlines()
    if not content_lines or not content_lines[0].startswith(
            (f"[Current source: {path}]", f"[Working source: {path}]")):
        framed = f"[Current source: {path}]\n{content}"
    if not verify_window(framed, path=path, source=source):
        return ()
    raw_ranges = metadata.get("ranges")
    return tuple(
        (int(row["start_line"]), int(row["end_line"]))
        for row in raw_ranges if isinstance(row, Mapping)
        and type(row.get("start_line")) is int
        and type(row.get("end_line")) is int
    ) if isinstance(raw_ranges, list) else ()


def _range_is_covered(
        delivered: Sequence[tuple[int, int]], start: int, end: int,
) -> bool:
    next_line = start
    for first, last in sorted(delivered):
        if last < next_line:
            continue
        if first > next_line:
            return False
        next_line = max(next_line, last + 1)
        if next_line > end:
            return True
    return False


def _merge_closure_source_ranges(
        ranges: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge touching planned spans so each generated source window is valid."""
    merged: list[dict[str, object]] = []
    for row in sorted(
            ranges,
            key=lambda item: (int(item["start_line"]), int(item["end_line"])),
    ):
        start, end = int(row["start_line"]), int(row["end_line"])
        anchors = [str(anchor) for anchor in row.get("anchors", ()) if str(anchor)]
        if merged and start <= int(merged[-1]["end_line"]) + 1:
            merged[-1]["end_line"] = max(int(merged[-1]["end_line"]), end)
            merged[-1]["anchors"] = sorted(set(
                [*merged[-1]["anchors"], *anchors]
            ))
            continue
        merged.append({
            "start_line": start,
            "end_line": end,
            "anchors": sorted(set(anchors)),
        })
    return merged


def _append_framefit5_closure_source_messages(
        messages: list[dict[str, object]], plan: Mapping[str, object] | None,
        sources: Mapping[str, str] | None, *,
        frame_request_allocation: FrameRequestAllocation | None = None,
) -> int:
    """Render retained closure units in the exact format final-wire proof accepts."""
    if frame_request_allocation is not None:
        _validate_frame_request_allocation_placement(messages, frame_request_allocation)
        return 0
    if not isinstance(plan, Mapping) or not isinstance(sources, Mapping):
        return 0
    if not _is_framefit5_closure_projection_input(messages):
        return 0
    raw_units = plan.get("units", ())
    if not isinstance(raw_units, Sequence) or isinstance(raw_units, (str, bytes, bytearray)):
        return 0
    revisions = plan.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    bounded = plan.get("bounded_delivery")
    if not isinstance(bounded, Mapping):
        return 0
    raw_retained = bounded.get("retained_handles")
    raw_omitted = bounded.get("omitted_handles", ())
    if (
        not isinstance(raw_retained, Sequence)
        or isinstance(raw_retained, (str, bytes, bytearray))
        or any(not isinstance(handle, str) or not handle.strip() for handle in raw_retained)
        or not isinstance(raw_omitted, Sequence)
        or isinstance(raw_omitted, (str, bytes, bytearray))
        or any(not isinstance(handle, str) or not handle.strip() for handle in raw_omitted)
    ):
        return 0
    retained_handles = {handle.strip() for handle in raw_retained}
    omitted_handles = {handle.strip() for handle in raw_omitted}
    deferred = set(_normalized_deferred_paths(
        plan.get("deferred_paths", ())
        if isinstance(plan.get("deferred_paths", ()), Sequence)
        and not isinstance(plan.get("deferred_paths", ()), (str, bytes, bytearray))
        else ()
    ))
    from kernel_public_localization import public_path

    planned: dict[str, dict[str, object]] = {}
    source_lines_by_path: dict[str, list[str]] = {}
    for unit in raw_units:
        if not isinstance(unit, Mapping):
            continue
        handle = str(unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle") or "").strip()
        if ((retained_handles is not None and handle not in retained_handles)
                or (handle and handle in omitted_handles)):
            continue
        path = str(unit.get("path", "")).replace("\\", "/").strip()
        if not path or path in deferred or public_path(path) != path:
            continue
        source = sources.get(path)
        revision = str(revisions.get(path, "") or "").strip()
        unit_revision = str(unit.get("revision", "") or "").strip()
        if (not isinstance(source, str) or not revision
                or source_revision(source) != revision
                or (unit_revision and unit_revision != revision)):
            continue
        start, end = unit.get("start_line"), unit.get("end_line")
        source_lines = source_lines_by_path.get(path)
        if source_lines is None:
            source_lines = source.splitlines(keepends=True)
            source_lines_by_path[path] = source_lines
        if (type(start) is not int or type(end) is not int
                or start < 1 or end < start or end > len(source_lines)):
            continue
        row = planned.setdefault(path, {
            "source": source,
            "revision": revision,
            "units": [],
            "owner": str(unit.get(
                "owner", unit.get("symbol_owner", unit.get("enclosing", unit.get("symbol", "UNKNOWN_OWNER")))
            ) or "UNKNOWN_OWNER"),
        })
        ranges = row["units"]
        if isinstance(ranges, list):
            ranges.append((start, end, str(unit.get("role") or "request_source_closure")))
    if not planned:
        return 0

    delivered: dict[str, list[tuple[int, int]]] = {path: [] for path in planned}
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        from kernel_source_presentation import strip_callsite_census
        content = strip_callsite_census(content)
        for path, row in planned.items():
            source = row.get("source")
            revision = row.get("revision")
            if isinstance(source, str) and isinstance(revision, str):
                delivered[path].extend(_verified_source_message_ranges(
                    content, path=path, source=source, revision=revision,
                ))

    additions: list[dict[str, object]] = []
    for path, row in planned.items():
        source = row.get("source")
        revision = row.get("revision")
        units = row.get("units")
        if (not isinstance(source, str) or not isinstance(revision, str)
                or not isinstance(units, list)):
            continue
        pending: dict[tuple[int, int], dict[str, object]] = {}
        for start, end, role in units:
            if _range_is_covered(delivered[path], start, end):
                continue
            pending.setdefault((start, end), {
                "start_line": start,
                "end_line": end,
                "anchors": [role],
            })
        if not pending:
            continue
        source_lines = source_lines_by_path[path]
        window = _render_ranges(
            source_lines,
            _merge_closure_source_ranges([pending[key] for key in sorted(pending)]),
            path=path,
            revision=revision,
            total_lines=len(source_lines),
        )
        projected = project_source_window(
            window, owner=str(row.get("owner") or "UNKNOWN_OWNER"),
        )
        if projected is not None:
            additions.append({"role": "user", "content": projected})
    if not additions:
        return 0

    state_index = next((index for index, message in enumerate(messages)
                        if str(message.get("content") or "").startswith("[StateFrame]\n")), None)
    if state_index is None:
        return 0
    insert_at = state_index + 1
    messages[insert_at:insert_at] = additions
    return len(additions)


def _is_framefit5_closure_projection_input(
        messages: Sequence[Mapping[str, object]],
) -> bool:
    """Accept final products or the typed pre-final frame used before wire assembly."""
    from kernel_stateframe_f4 import HANDOFF_HEADER, STATE_HEADER
    from kernel_stateframe_framefit5 import (
        PRODUCT_MODE,
        FrameFit5Error,
        _is_volatile_tail_message,
        _state_payload,
        has_framefit5_state,
        is_frame_only,
    )

    if is_frame_only(messages):
        return True
    if (not has_framefit5_state(messages)
            or any(message.get("role") not in {"system", "user"} for message in messages)
            or any(_is_volatile_tail_message(message) for message in messages)
            or any(str(message.get("content") or "").startswith(HANDOFF_HEADER)
                   for message in messages)):
        return False
    system_count = next(
        (index for index, message in enumerate(messages)
         if message.get("role") != "system"),
        len(messages),
    )
    if (system_count < 1
            or any(message.get("role") == "system" for message in messages[system_count:])
            or len(messages) < system_count + 2):
        return False
    task = messages[system_count]
    state_message = messages[system_count + 1]
    if (task.get("role") != "user"
            or not str(task.get("content") or "").startswith("Task:")
            or state_message.get("role") != "user"
            or not str(state_message.get("content") or "").startswith(STATE_HEADER + "\n")):
        return False
    state_blocks = [
        message for message in messages
        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")
    ]
    if len(state_blocks) != 1:
        return False
    try:
        _payload, state = _state_payload(state_message)
    except (FrameFit5Error, TypeError, ValueError):
        return False
    framefit5 = state.get("framefit5")
    return isinstance(framefit5, Mapping) and framefit5.get("mode") == PRODUCT_MODE


def _fit_ranges(
        source_lines: Sequence[str], anchors: Sequence[tuple[int, int, str]],
        *, path: str, revision: str, total_lines: int, max_chars: int,
        required: Sequence[tuple[int, int, str]] = (),
        optional_order: Sequence[Sequence[tuple[int, int, str]]] = (),
        context_lines: int | None = None,
) -> str:
    # The caller's value is a serialized UTF-8 byte ceiling despite the legacy
    # parameter name.  Do not silently raise it: a tiny remaining budget must
    # produce a bounded continuation receipt, not an oversized request.
    max_chars = max(1, int(max_chars))
    context_limit = (_DEFAULT_CONTEXT_LINES if context_lines is None
                     else max(0, int(context_lines)))
    initial = _merge_ranges(anchors, total_lines, context_lines=context_limit)
    rendered = _render_ranges(source_lines, initial, path=path, revision=revision,
                              total_lines=total_lines, max_bytes=max_chars)
    if len(rendered.encode("utf-8")) <= max_chars:
        return rendered
    # Reduce context first, then retain each anchor line.  The target symbol
    # is never lost merely because the request is at the S3 ceiling.
    def anchor_rank(label: str) -> tuple[int, str]:
        # Complete closure/body/type units outrank lexical and ticket-word
        # hints.  Stable lexical ordering keeps identical inputs deterministic.
        if label.startswith(("closure:", "symbol:")):
            return (0, label)
        if label.startswith("text:"):
            return (1, label)
        if label.startswith("literal:"):
            return (2, label)
        if label == "module-head":
            return (3, label)
        return (4, label)

    anchors = list(anchors)
    anchors.sort(key=lambda item: (anchor_rank(item[2]), item[0], item[1]))
    context = context_limit
    while context >= 0:
        candidate = _merge_ranges(anchors, total_lines, context_lines=context)
        rendered = _render_ranges(source_lines, candidate, path=path, revision=revision,
                                  total_lines=total_lines, max_bytes=max_chars)
        if len(rendered.encode("utf-8")) <= max_chars:
            return rendered
        context -= 1
    # Optional host anchors (module head, then literal sites) are shed before
    # the target's own anchor ever loses context (``optional_order`` holds the
    # successively smaller anchor sets; ``required`` is the target-only set).
    for kept in optional_order:
        context = context_limit
        while kept and context >= 0:
            candidate = _merge_ranges(kept, total_lines, context_lines=context)
            rendered = _render_ranges(source_lines, candidate, path=path, revision=revision,
                                      total_lines=total_lines, max_bytes=max_chars)
            if len(rendered.encode("utf-8")) <= max_chars:
                return rendered
            context -= 1
    if required:
        anchors = list(required)
    # A very large target body can still exceed the ceiling.  Allocate a
    # deterministic line budget around its first anchor and let the model page
    # the rest.  This branch keeps the source identity, never invents bytes.
    first = anchors[0] if anchors else (1, 1, "fallback:first-line")
    center = max(1, min(total_lines, (first[0] + first[1]) // 2))
    lines_budget = max(1, min(_MAX_PAGE_LINES, max_chars // 80))
    # The window metadata is not free: shrink the page until the serialized
    # window fits instead of jumping straight to a single line.
    while lines_budget >= 2:
        start = max(1, center - lines_budget // 2)
        end = min(total_lines, start + lines_budget - 1)
        start = max(1, end - lines_budget + 1)
        candidate = [{"start_line": start, "end_line": end, "anchors": [first[2]]}]
        rendered = _render_ranges(source_lines, candidate, path=path, revision=revision,
                                  total_lines=total_lines, max_bytes=max_chars)
        if len(rendered.encode("utf-8")) <= max_chars:
            return rendered
        lines_budget = lines_budget * 2 // 3
    # Metadata is small, but preserve a final exact line rather than return a
    # malformed source block if an unusually long single line dominates.  Never
    # spend that line on whitespace when the file has code nearby.
    center = _nearest_code_line(source_lines, center)
    candidate = [{"start_line": center, "end_line": center, "anchors": [first[2]]}]
    # Keep the actual source line number in the final fallback; this branch is
    # only for a single pathological line longer than the request allowance.
    rendered = _render_ranges(source_lines, candidate, path=path, revision=revision,
                              total_lines=total_lines, max_bytes=max_chars)
    return rendered


def _nearest_code_line(source_lines: Sequence[str], center: int) -> int:
    """Return the non-blank line closest to ``center`` (ties go earlier)."""
    total = len(source_lines)
    for distance in range(total):
        for candidate in (center - distance, center + distance):
            if 1 <= candidate <= total and source_lines[candidate - 1].strip():
                return candidate
    return center


def parse_window(content: object) -> dict[str, object] | None:
    if not isinstance(content, str):
        return None
    lines = content.splitlines(keepends=True)
    if len(lines) < 4:
        return None
    # Accept the current framing (marker, metadata) and the legacy framing
    # (preamble, marker, metadata).  Final-delivery accounting must parse the
    # payload that was actually emitted, not silently downgrade valid spans.
    marker_index = next(
        (index for index, line in enumerate(lines[:3]) if line.rstrip("\r\n") == WINDOW_MARKER),
        None,
    )
    if marker_index is None or marker_index + 1 >= len(lines):
        return None
    try:
        value = json.loads(lines[marker_index + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return None
    if value.get("encoding") == "compact-page/1":
        revision = value.get("revision")
        path = value.get("path")
        if (not isinstance(revision, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None
                or not isinstance(path, str) or not path):
            return None
        value = dict(value)
        value["sha256"] = revision.removeprefix("sha256:")
        value["page"] = {
            "tool": "read", "path": path, "revision": revision,
            "max_lines": _MAX_PAGE_LINES, "max_chars": _MAX_PAGE_CHARS,
            "full_revision_required": True,
        }
    return value


def is_source_window(content: object) -> bool:
    return parse_window(content) is not None


def _window_segments(content: str, metadata: Mapping[str, object]) -> list[tuple[dict[str, object], str]]:
    lines = content.splitlines(keepends=True)
    try:
        start = next(index for index, line in enumerate(lines) if line.rstrip("\r\n") == WINDOW_CONTENT_MARKER) + 1
    except StopIteration:
        return []
    segments: list[tuple[dict[str, object], str]] = []
    current: dict[str, object] | None = None
    buffer: list[str] = []
    for line in lines[start:]:
        match = _RANGE.match(line.rstrip("\r\n"))
        if match:
            if current is not None:
                segments.append((current, "".join(buffer)))
            current = {
                "start_line": int(match.group(1)),
                "end_line": int(match.group(2)),
            }
            buffer = []
            continue
        if line.rstrip("\r\n") == WINDOW_END_MARKER:
            break
        if current is not None:
            buffer.append(line)
    if current is not None:
        segments.append((current, "".join(buffer)))
    metadata_ranges = metadata.get("ranges")
    if not isinstance(metadata_ranges, list):
        return []
    for index, (row, text) in enumerate(segments):
        if index >= len(metadata_ranges) or not isinstance(metadata_ranges[index], Mapping):
            continue
        merged = {**dict(metadata_ranges[index]), **row}
        segments[index] = (merged, text)
    return segments


def verify_window(content: object, *, path: str, source: str) -> bool:
    """Verify exact current bytes, ranges, and the revision-bound page contract."""
    if not isinstance(content, str):
        return False
    header = content.splitlines()[0] if content.splitlines() else ""
    header_match = _HEADER.match(header)
    metadata = parse_window(content)
    if header_match is None or metadata is None:
        return False
    if header_match.group(1).replace("\\", "/") != path.replace("\\", "/"):
        return False
    revision = source_revision(source)
    if (metadata.get("path") != path.replace("\\", "/")
            or metadata.get("revision") != revision
            or metadata.get("sha256") != _digest(source)
            or metadata.get("total_lines") != _line_count(source)
            or type(metadata.get("omitted_lines")) is not bool):
        return False
    page = metadata.get("page")
    if (not isinstance(page, Mapping) or page.get("tool") != "read"
            or page.get("path") != metadata.get("path")
            or page.get("revision") != revision
            or page.get("full_revision_required") is not True
            or _page_limit(page.get("max_lines"), _MAX_PAGE_LINES) is False
            or _page_limit(page.get("max_chars"), _MAX_PAGE_CHARS) is False):
        return False
    source_lines = source.splitlines(keepends=True)
    previous_end = 0
    segments = _window_segments(content, metadata)
    declared = metadata.get("ranges")
    if not isinstance(declared, list) or len(segments) != len(declared) or not segments:
        return False
    covered_lines = 0
    for row, actual in segments:
        start = row.get("start_line")
        end = row.get("end_line")
        if (type(start) is not int or type(end) is not int or start < 1 or end < start
                or end > len(source_lines) or start <= previous_end):
            return False
        expected = "".join(source_lines[start - 1:end])
        if not bool(row.get("trailing_newline", expected.endswith("\n"))) and actual.endswith("\n"):
            actual = actual[:-1]
        if actual != expected:
            return False
        covered_lines += end - start + 1
        previous_end = end
    return metadata.get("omitted_lines") is (covered_lines < len(source_lines))


def verify_current_source_message(content: object, *, path: str, source: str) -> bool:
    """Prove a current source block, including one folded into tool output."""
    if not isinstance(content, str) or not path or not source:
        return False
    normalized_path = path.replace("\\", "/")
    if (verify_window(content, path=normalized_path, source=source)
            or verify_model_source_window(content, path=normalized_path, source=source)):
        return True
    header = re.compile(
        rf"(?m)^\[(?:Current|Working) source: {re.escape(normalized_path)}\]$",
    )
    for match in reversed(tuple(header.finditer(content))):
        block = content[match.start():]
        if (verify_window(block, path=normalized_path, source=source)
                or verify_model_source_window(block, path=normalized_path, source=source)):
            return True
        if source in content[match.end():]:
            return True
    return False


def _page_limit(value: object, maximum: int) -> bool:
    try:
        return 0 < int(value) <= maximum
    except (TypeError, ValueError, OverflowError):
        return False


def verify_window_projection(content: object, *, path: str) -> bool:
    """Validate a serialized window when only its visible projection is available.

    The loopback provider reconstructs a disposable workspace from the bytes it
    received.  That workspace contains the window envelope, not the original
    full file, so full-source digest validation is impossible there.  The live
    preflight still uses :func:`verify_window`; this fallback checks only the
    self-describing envelope and exact excerpt structure at the provider edge.
    """
    if not isinstance(content, str):
        return False
    lines = content.splitlines(keepends=True)
    header_match = _HEADER.match(lines[0] if lines else "")
    metadata = parse_window(content)
    if header_match is None or metadata is None:
        return False
    if header_match.group(1).replace("\\", "/") != path.replace("\\", "/"):
        return False
    if (metadata.get("path") != path.replace("\\", "/")
            or not isinstance(metadata.get("revision"), str)
            or not str(metadata["revision"]).startswith("sha256:")
            or not isinstance(metadata.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(metadata["sha256"]))
            or type(metadata.get("total_lines")) is not int
            or metadata["total_lines"] < 1
            or type(metadata.get("omitted_lines")) is not bool):
        return False
    page = metadata.get("page")
    if (not isinstance(page, Mapping) or page.get("tool") != "read"
            or page.get("path") != metadata.get("path")
            or page.get("revision") != metadata.get("revision")
            or page.get("full_revision_required") is not True
            or _page_limit(page.get("max_lines"), _MAX_PAGE_LINES) is False
            or _page_limit(page.get("max_chars"), _MAX_PAGE_CHARS) is False):
        return False
    declared = metadata.get("ranges")
    segments = _window_segments(content, metadata)
    if not isinstance(declared, list) or not declared or len(segments) != len(declared):
        return False
    previous_end = 0
    covered_lines = 0
    total_lines = int(metadata["total_lines"])
    for index, (row, actual) in enumerate(segments):
        declared_row = declared[index]
        if not isinstance(declared_row, Mapping):
            return False
        start = row.get("start_line")
        end = row.get("end_line")
        if (type(start) is not int or type(end) is not int
                or declared_row.get("start_line") != start
                or declared_row.get("end_line") != end
                or start < 1 or end < start or end > total_lines
                or start <= previous_end or not actual):
            return False
        trailing_newline = declared_row.get("trailing_newline")
        if type(trailing_newline) is not bool:
            return False
        if trailing_newline != actual.endswith("\n"):
            return False
        covered_lines += end - start + 1
        previous_end = end
    return metadata.get("omitted_lines") is (covered_lines < total_lines)


@profiled_function_span("source_message_window")
def bound_source_message(
        content: str, *, path: str, source: str, runtime: object,
        max_chars: int, is_target: bool = True,
        context_lines: int | None = None,
) -> str:
    """Return a bounded exact excerpt; full source remains host-owned."""
    reservation = getattr(runtime, "_aha11_source_reservation", None)
    if (isinstance(reservation, Mapping)
            and reservation.get("path") == path
            and reservation.get("revision") == source_revision(source)
            and reservation.get("complete") is True):
        header, separator, body = content.partition("\n")
        if header in {f"[Current source: {path}]", f"[Working source: {path}]"}:
            prefix = header + "\n"
            if body.startswith("Coverage: COMPLETE FILE;"):
                _coverage, newline, body = body.partition("\n")
                prefix += _coverage + (newline if newline else "")
            full_content = prefix + source
            if len(full_content.encode("utf-8")) <= max_chars:
                return full_content
    total_lines = _line_count(source)
    if not total_lines:
        return content
    ticket = _ticket_text(runtime)
    anchors = _anchor_ranges(source, runtime, ticket, path=path, is_target=is_target)
    without_head = [anchor for anchor in anchors if anchor[2] != "module-head"]
    required = [anchor for anchor in without_head if not anchor[2].startswith("literal:")]
    optional_order: list[list[tuple[int, int, str]]] = []
    if required and len(required) < len(anchors):
        if len(without_head) < len(anchors) and len(without_head) > len(required):
            optional_order.append(without_head)
        optional_order.append(required)
    first_line = content.splitlines()[0] if content.splitlines() else ""
    if not anchors:
        # An all-whitespace source has no useful excerpt; final request
        # settlement will replace this header-only message with a read route.
        return first_line + "\n"
    return first_line + "\n" + _fit_ranges(
        source.splitlines(keepends=True), anchors,
        path=path, revision=source_revision(source), total_lines=total_lines,
        max_chars=max_chars,
        required=required if optional_order else (),
        optional_order=optional_order,
        context_lines=context_lines,
    )


def _cached_closure_is_fresh(root: Path, closure: Mapping[str, object]) -> bool:
    """Invalidate a cached closure when any discovered source byte changes."""
    revisions = closure.get("revisions")
    if not isinstance(revisions, Mapping):
        return False
    for raw_path, expected in revisions.items():
        path = str(raw_path)
        current = _read_product_file(root, path)
        if current is None or source_revision(current) != str(expected):
            return False
    dependency_revisions = closure.get("dependency_revisions")
    if isinstance(dependency_revisions, Mapping) and dependency_revisions and not revisions:
        return False
    return True


def _source_unit_binding_key(unit: Mapping[str, object]) -> tuple[str, int, int, str] | None:
    paths = _normalized_deferred_paths((unit.get("path", ""),))
    start = unit.get("start_line")
    end = unit.get("end_line")
    if not paths or type(start) is not int or type(end) is not int:
        return None
    return paths[0], start, end, str(unit.get("role") or "statement")


def _retain_bound_source_unit_handles(
        plan_units: object,
        bounded_units: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Keep exact bounded-read handles on the closure units used by admission."""
    handles: dict[tuple[str, int, int, str], str] = {}
    for unit in bounded_units:
        key = _source_unit_binding_key(unit)
        handle = str(unit.get("unit_handle") or "").strip()
        if key is not None and handle:
            handles[key] = handle
    units = (
        plan_units if isinstance(plan_units, Sequence)
        and not isinstance(plan_units, (str, bytes, bytearray)) else ()
    )
    retained: list[dict[str, object]] = []
    for raw_unit in units:
        if not isinstance(raw_unit, Mapping):
            continue
        unit = dict(raw_unit)
        key = _source_unit_binding_key(unit)
        handle = handles.get(key) if key is not None else None
        if handle:
            unit["unit_handle"] = handle
        retained.append(unit)
    return retained


def _record_required_source_deferral(
        candidate: str,
        *,
        required_paths: set[str],
        deferred_paths: list[str],
        closure_plan: dict[str, object],
        runtime: object,
) -> None:
    if candidate not in required_paths:
        return
    if candidate not in deferred_paths:
        deferred_paths.append(candidate)
    closure_plan["deferred_paths"] = list(deferred_paths)
    setattr(runtime, "_request_deferred_source_paths", tuple(deferred_paths))


@profiled_function_span("source_closure_projection")
def bound_runtime_sources(
        messages: list[dict[str, object]], runtime: object,
        *, max_request_bytes: int, tools: Sequence[object] = (),
) -> dict[str, object]:
    """Window exact target source blocks until the protected set is bounded."""
    target = getattr(runtime, "target", None)
    root = getattr(getattr(runtime, "owner", None), "root", None)
    path = _path_from_runtime(runtime)
    raw_deferred_paths = getattr(runtime, "_request_deferred_source_paths", ())
    deferred_paths = list(_normalized_deferred_paths(
        raw_deferred_paths
        if isinstance(raw_deferred_paths, Sequence)
        and not isinstance(raw_deferred_paths, (str, bytes, bytearray))
        else ()
    ))
    if target is None or root is None or path is None or not Path(target).is_file():
        return {"changed": False, "windowed": 0, "paths": ()}
    try:
        source = Path(target).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {"changed": False, "windowed": 0, "paths": ()}
    source_hash = source_revision(source)
    non_source_bytes = sum(
        len(str(item.get("content", "")).encode("utf-8"))
        for item in messages
        if isinstance(item, Mapping) and not str(item.get("content", "")).startswith(
            (f"[Current source: {path}]", f"[Working source: {path}]")
        )
    )
    non_source_bytes += sum(len(str(tool).encode("utf-8")) for tool in tools)
    non_source_bytes += len(f"[Current source: {path}]\n".encode("utf-8"))
    anchors = _anchor_ranges(source, runtime, _ticket_text(runtime), path=path, is_target=True)
    with profile_active_host_span("request_source_target_reservation"):
        source_reservation = reserve_source_unit(
            source,
            path=path,
            revision=source_hash,
            max_bytes=max_request_bytes,
            non_source_bytes=non_source_bytes,
            required_ranges=tuple(
                (start, end) for start, end, label in anchors if label != "module-head"),
            signature_ranges=tuple(
                (start, end) for start, end, label in anchors if label == "entry-symbol"),
        )
    setattr(runtime, "_aha11_source_reservation", source_reservation.as_dict())
    changed_paths: list[str] = []
    acquired_paths: list[str] = []
    omission_receipts: list[dict[str, object]] = []
    ticket = _ticket_text(runtime)
    identity = ticket_identity_for_runtime(runtime)
    base = Path(root)
    declared_paths = editable_product_paths(runtime)
    parent_sources = _parent_sources(runtime)
    tokens = literal_tokens(ticket, exclude=(entry_symbol(runtime),))
    origins = _token_origins(ticket)
    source_stats = tuple(_source_stat_token(base, item) for item in declared_paths)
    input_key = (str(base.resolve()), str(getattr(runtime, "product_revision", "")),
                 source_hash, tuple(declared_paths), source_stats)
    inventory_key = (*input_key, tuple(tokens), tuple(sorted(origins.items())))
    inventory_cache = getattr(runtime, "_request_literal_site_inventory_cache", None)
    if not isinstance(inventory_cache, dict):
        inventory_cache = {}
        setattr(runtime, "_request_literal_site_inventory_cache", inventory_cache)
    if tokens:
        cached_sites = inventory_cache.get(inventory_key)
        if cached_sites is None:
            cached_sites = tuple(literal_site_inventory(base, declared_paths, tokens, origins=origins))
            inventory_cache[inventory_key] = cached_sites
        sites = copy.deepcopy(cached_sites)
    else:
        sites = ()
    if identity is not None or ticket:
        ticket_key = identity.sha256 if identity is not None else _digest(ticket)
        parent_key = tuple(sorted((name, source_revision(value))
                                  for name, value in parent_sources.items()))
        closure_key = (*input_key, ticket_key, path, parent_key)
        closure_cache = getattr(runtime, "_request_source_closure_cache", None)
        if not isinstance(closure_cache, dict):
            closure_cache = {}
            setattr(runtime, "_request_source_closure_cache", closure_cache)
        cached_closure = closure_cache.get(closure_key)
        if (not isinstance(cached_closure, Mapping)
                or not _cached_closure_is_fresh(base, cached_closure)):
            with profile_active_host_span("request_source_closure_build"):
                closure = build_source_closure(
                    base,
                    declared_paths,
                    identity or ticket,
                    target_path=path,
                    parent_sources=parent_sources,
                )
            cached_closure = closure.as_dict()
            closure_cache[closure_key] = copy.deepcopy(cached_closure)
        closure_plan = copy.deepcopy(cached_closure)
    else:
        # Launch/completion fixtures may intentionally have no public ticket;
        # preserve an explicit unknown closure rather than inventing ticket text.
        closure_plan = {
            "schema": "stateframe-request-evidence-closure/1",
            "ticket_sha256": None,
            "paths": [], "lexical_sites": [], "units": [], "omissions": [],
            "skipped_files": [], "revisions": {}, "dependency_revisions": {},
            "identities": {}, "adapters": {}, "delivery": {},
            "coverage": {
                "completeness": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "skipped_file": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "omission": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "lexical_site": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "acquired_span": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "delivered_span": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "textual_edit": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "resolved_use_definition": {"status": "UNKNOWN", "reason": "no authorized ticket"},
                "observed_behavior": {"status": "UNCOVERED", "reason": "request assembly has no behavior evidence"},
            },
        }
    # The final ledger must carry the same authorization identity as the
    # source-read planner; a path/revision match alone is not cross-workspace
    # evidence.
    closure_plan["workspace_authorization"] = {
        "scope": str(getattr(runtime, "workspace_scope", base)),
        "generation": str(getattr(runtime, "workspace_revision",
                                   getattr(runtime, "product_revision", "0"))),
    }
    with _VERIFIED_SOURCE_HISTORY_LOCK:
        verified_history = getattr(runtime, "_verified_source_message_history", ())
        closure_plan["verified_prior_source_messages"] = copy.deepcopy(
            list(verified_history[-_VERIFIED_SOURCE_HISTORY_LIMIT:])
            if isinstance(verified_history, Sequence)
            and not isinstance(verified_history, (str, bytes, bytearray)) else []
        )
    _ensure_target_unit(closure_plan, path, source_reservation.as_dict())
    closure_plan["deferred_paths"] = list(deferred_paths)
    from kernel_aha10_honest_loop import bounded_source_delivery
    bounded_units: list[dict[str, object]] = []
    with profile_active_host_span("request_source_unit_materialization"):
        for raw_unit in closure_plan.get("units", ()):
            if not isinstance(raw_unit, Mapping):
                continue
            unit = dict(raw_unit)
            unit_path = str(unit.get("path", ""))
            unit_source = _read_product_file(base, unit_path)
            start = int(unit.get("start_line", 0) or 0)
            end = int(unit.get("end_line", 0) or 0)
            source_lines = unit_source.splitlines(keepends=True) if unit_source is not None else []
            complete = bool(source_lines and 1 <= start <= end <= len(source_lines))
            unit["text"] = "".join(source_lines[start - 1:end]) if complete else ""
            unit["complete"] = complete
            unit["revision"] = closure_plan.get("revisions", {}).get(unit_path, "")
            unit["language"] = _LANGUAGE_BY_SUFFIX.get(
                Path(unit_path).suffix.casefold(), "unknown")
            unit["unit_handle"] = "unit:" + _digest(json.dumps({
                "path": unit_path, "start": start, "end": end,
                "revision": unit["revision"], "role": unit.get("role", "statement"),
            }, sort_keys=True))[:32]
            bounded_units.append(unit)
    closure_plan["units"] = _retain_bound_source_unit_handles(
        closure_plan.get("units", ()), bounded_units,
    )
    required_paths = {path}
    required_paths.update(
        str(item.get("path")) for item in closure_plan.get("lexical_sites", ())
        if isinstance(item, Mapping) and item.get("path")
    )

    required_handles = tuple(
        str(item.get("unit_handle")) for item in bounded_units
        if str(item.get("path")) in required_paths and item.get("unit_handle")
    )
    bounded_delivery = bounded_source_delivery(
        bounded_units, max_chars=min(64_000, max(512, max_request_bytes // 2)),
        required_unit_handles=required_handles, required_paths=tuple(required_paths),
    )
    closure_plan["bounded_delivery"] = {
        "schema": bounded_delivery["schema"],
        "complete_units": bounded_delivery["complete_units"],
        "status": bounded_delivery["status"],
        "continuation": bounded_delivery["continuation"],
        "continuation_action": bounded_delivery["continuation_action"],
        "source_paths": bounded_delivery["source_paths"],
        "coverage": bounded_delivery["coverage"],
        "retained_handles": [
            str(item.get("unit_handle", item.get("unit_id", item.get("id", ""))))
            for item in bounded_delivery["retained_units"]
            if isinstance(item, Mapping)
        ],
        "omitted_handles": [
            str(item.get("unit_handle", item.get("unit_id", item.get("id", ""))))
            for item in bounded_delivery["omitted_units"]
            if isinstance(item, Mapping)
        ],
    }
    setattr(runtime, "_request_source_closure", closure_plan)
    connected = tuple(str(item) for item in closure_plan.get("paths", ()) or ())
    connected_for_delivery, filtered_import_windows = _framefit_connected_paths(
        runtime, messages, connected, target_path=path, declared_paths=declared_paths,
    )
    if filtered_import_windows:
        closure_plan["framefit_source_window_filter"] = {
            "omitted_import_windows": filtered_import_windows,
            "retained_requested_windows": len(connected_for_delivery),
        }
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        request_id = getattr(runtime, "current_request_id", None)
        request_key = request_id if request_id is not None else id(messages)
        last_request_id = getattr(runtime, "_framefit_source_filter_journaled_request", None)
        if callable(emit) and request_key != last_request_id:
            emit(
                "stateframe_framefit_source_windows",
                producer="kernel_request_source_window",
                omitted_import_windows=filtered_import_windows,
                retained_requested_windows=len(connected_for_delivery),
                dispatch=False,
            )
            setattr(runtime, "_framefit_source_filter_journaled_request", request_key)
    omission_receipts.extend(
        dict(item) for item in closure_plan.get("omissions", ())
        if isinstance(item, Mapping)
    )

    def has_source_message(candidate: str, candidate_source: str) -> bool:
        from kernel_source_presentation import strip_callsite_census
        prefixes = (f"[Current source: {candidate}]\n", f"[Working source: {candidate}]\n")
        return any(
            isinstance(item, dict) and isinstance(item.get("content"), str)
            and (
                strip_callsite_census(item["content"]).startswith(prefixes)
                or verify_current_source_message(
                    item["content"], path=candidate, source=candidate_source,
                )
            )
            for item in messages
        )

    def ensure_source_message(candidate: str, candidate_source: str, *,
                              ahead_of_optional: bool = False) -> None:
        if candidate in deferred_paths or has_source_message(candidate, candidate_source):
            return
        header = f"[Current source: {candidate}]\n"
        # The required target body is reserved before optional panels, in
        # position as well as in bytes; connected files keep append order. Measure
        # the serialized envelope before inserting source so a floor-sized page
        # cannot push an append-only transcript over its request ceiling.
        index = next((position for position, item in enumerate(messages)
                      if ahead_of_optional and _is_optional_panel(item)), len(messages))
        projected = list(messages)
        projected.insert(index, {"role": "user", "content": header})
        # The fixed reserve is useful on large requests, but consumes nearly
        # the entire ceiling of a small request. The exact envelope is checked
        # below, so small requests can use their actual available bytes.
        limit = int(max_request_bytes) - (768 if max_request_bytes >= 4096 else 0)
        base_bytes = len(json.dumps({"messages": projected, "tools": list(tools)}).encode("utf-8"))
        max_chars = min(12_000, limit - base_bytes)
        if max_chars < _MIN_WINDOW_CHARS:
            _record_required_source_deferral(
                candidate, required_paths=required_paths, deferred_paths=deferred_paths,
                closure_plan=closure_plan, runtime=runtime,
            )
            return

        message = {"role": "user", "content": header}
        for _attempt in range(8):
            message["content"] = bound_source_message(
                header, path=candidate, source=candidate_source, runtime=runtime,
                max_chars=max_chars, is_target=candidate == path,
            )
            window_metadata = parse_window(message["content"])
            if (candidate == path and candidate in required_paths
                    and isinstance(window_metadata, Mapping)
                    and window_metadata.get("omitted_lines") is True):
                # A partial target cannot replace the exact read route.
                _record_required_source_deferral(
                    candidate, required_paths=required_paths,
                    deferred_paths=deferred_paths, closure_plan=closure_plan,
                    runtime=runtime,
                )
                return
            projected[index] = message
            request_bytes = len(json.dumps(
                {"messages": projected, "tools": list(tools)}
            ).encode("utf-8"))
            if request_bytes <= limit:
                messages.insert(index, message)
                acquired_paths.append(candidate)
                if message["content"] != header + candidate_source:
                    changed_paths.append(candidate)
                if (candidate in required_paths
                        and isinstance(window_metadata, Mapping)
                        and window_metadata.get("omitted_lines") is True):
                    # Preserve the verified excerpt while keeping the missing
                    # ranges on the read route; partial is never COMPLETE.
                    _record_required_source_deferral(
                        candidate, required_paths=required_paths,
                        deferred_paths=deferred_paths, closure_plan=closure_plan,
                        runtime=runtime,
                    )
                return
            max_chars -= max(32, request_bytes - limit)
            if max_chars < _MIN_WINDOW_CHARS:
                break

        _record_required_source_deferral(
            candidate, required_paths=required_paths, deferred_paths=deferred_paths,
            closure_plan=closure_plan, runtime=runtime,
        )

    deferred_target = _framefit_deferred_target_present(
        messages, runtime=runtime, path=path, source=source, tools=tools)
    if not has_source_message(path, source) and not deferred_target:
        ensure_source_message(path, source, ahead_of_optional=True)
    if not deferred_target:
        _window_path_messages(messages, runtime, path=path, source=source,
                              max_request_bytes=max_request_bytes, tools=tools,
                              changed_paths=changed_paths, is_target=True)
    if site_facts_mode() == "on":
        # P2: connected files are selected from lexical sites/imports/
        # construction responsibility, never from message history.
        for other in connected_for_delivery:
            if other == path:
                continue
            other_source = _read_product_file(base, other)
            if other_source is None:
                continue
            ensure_source_message(other, other_source)
            _window_path_messages(messages, runtime, path=other, source=other_source,
                                  max_request_bytes=max_request_bytes, tools=tools,
                                  changed_paths=changed_paths, is_target=False)
    for item in messages:
        content = item.get("content") if isinstance(item, dict) else None
        metadata = parse_window(content)
        if metadata is None or not metadata.get("omitted_lines"):
            continue
        receipt = metadata.get("continuation_receipt")
        if isinstance(receipt, Mapping):
            omission_receipts.append({
                "path": metadata.get("path"),
                "revision": metadata.get("revision"),
                **dict(receipt),
            })
    delivered_spans = 0
    delivery_records: list[Mapping[str, object]] = []
    for item in messages:
        metadata = parse_window(item.get("content")) if isinstance(item, dict) else None
        ranges = metadata.get("ranges") if isinstance(metadata, Mapping) else None
        if isinstance(ranges, list):
            delivered_spans += len(ranges)
            path_value = metadata.get("path") if isinstance(metadata, Mapping) else None
            if isinstance(path_value, str):
                delivery = record_source_delivery(
                    closure_plan,
                    path_value,
                    tuple(
                        (int(row["start_line"]), int(row["end_line"]))
                        for row in ranges
                        if isinstance(row, Mapping)
                        and type(row.get("start_line")) is int
                        and type(row.get("end_line")) is int
                    ),
                )
                delivery_records.append(delivery)
    setattr(runtime, "_request_source_closure", closure_plan)
    closure_receipt = render_closure_receipt(closure_plan)
    result = {
        "changed": bool(changed_paths or acquired_paths),
        "windowed": len(changed_paths),
        "paths": tuple(dict.fromkeys(changed_paths)),
        "acquired_paths": tuple(dict.fromkeys(acquired_paths)),
        "omissions": tuple(omission_receipts),
        "lexical_sites": len(sites),
        "acquired_spans": sum(int(item.get("acquired_spans", 0))
                             for item in delivery_records),
        "delivered_spans": delivered_spans,
        "textual_edit_coverage": None,
        "resolved_use_definition": "UNKNOWN",
        "observed_behavior": "UNKNOWN",
        "ticket_sha256": identity.sha256 if identity is not None else None,
        "retrieval_ticket_sha256": identity.sha256 if identity is not None else None,
        "revision": source_hash,
        "source_reservation": source_reservation.as_dict(),
        "deferred_paths": deferred_paths,
        "delivery_phase": "acquisition",
        "final_delivery_status": "UNVERIFIED",
        "closure": closure_plan,
        "closure_receipt": closure_receipt,
        "coverage": closure_plan.get("coverage", {}),
    }
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("request_source_closure", producer="kernel_request_source_window", **result, dispatch=False)
    return result


def _dropped_source_claim_visible(
        messages: Sequence[Mapping[str, object]], paths: Sequence[str],
) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        for path in paths:
            if any(marker in content for marker in (
                f"[Current source: {path}]",
                f"[Working source: {path}]",
                f"[Retained source: {path}]",
                f"[Observed source: {path}]",
                f"[read page] path={path} ",
                f"[read reuse] path={path} ",
                f"CURRENT_SOURCE {path} ",
                f"Observed source {path}",
            )):
                return True
            for parser in (parse_model_source_window, parse_window):
                parsed = parser(content)
                if isinstance(parsed, Mapping) and str(
                        parsed.get("path", "")).replace("\\\\", "/") == path:
                    return True
    return False


def verify_final_source_delivery(
        plan: Mapping[str, object], messages: Sequence[Mapping[str, object]],
        *, payload: object | bytes | None = None,
        sources: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Recompute delivered spans from the post-diet provider payload.

    Metadata is only a locator.  A final source receipt is complete when the
    declared revision, parsed body/ranges, workspace authorization, and exact
    bytes all agree with a ``DeliveryProof`` over the submitted payload.
    """

    from kernel_read_reuse import ReadKey, content_handle, proof_for_serialized_request

    physical_messages = messages
    source_map: dict[str, str] = {}
    if isinstance(plan.get("source_contents"), Mapping):
        source_map.update({str(path): str(value) for path, value in plan["source_contents"].items()
                           if isinstance(value, str)})
    if isinstance(sources, Mapping):
        source_map.update({str(path): str(value) for path, value in sources.items()
                           if isinstance(value, str)})
    declared_revisions = plan.get("revisions", {})
    declared_revisions = declared_revisions if isinstance(declared_revisions, Mapping) else {}
    authorization = plan.get("workspace_authorization", {})
    authorization = authorization if isinstance(authorization, Mapping) else {}
    workspace_scope = str(authorization.get("scope", "")).strip()
    workspace_generation = str(authorization.get("generation", "")).strip()
    integrity_omissions: list[dict[str, object]] = []
    authorization_verified = bool(workspace_scope and workspace_generation)
    if not authorization_verified:
        integrity_omissions.append({
            "kind": "workspace_authorization_unverified",
            "status": "UNVERIFIED",
            "reason": "workspace authorization is absent or incomplete",
        })

    ranges_by_path: dict[str, list[tuple[int, int]]] = {}
    locations_by_path: dict[str, list[str]] = {}
    range_records_by_path: dict[str, list[tuple[tuple[int, int], str | None, str]]] = {}

    def remember_ranges(
            path: str, ranges: Sequence[tuple[int, int]], message_index: int,
            content: str,
    ) -> None:
        location = final_message_location(message_index)
        ranges_by_path.setdefault(path, []).extend(ranges)
        if location is not None:
            locations_by_path.setdefault(path, []).append(location)
        range_records_by_path.setdefault(path, []).extend(
            (item, location, physical_messages[message_index]["content"]) for item in ranges
        )

    payload_is_http = isinstance(payload, bytes)
    if payload_is_http:
        wire_bytes = bytes(payload)
        context_value: object = {"messages": list(messages)}
    elif payload is not None:
        context_value = payload
        wire_bytes = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
    else:
        context_value = list(messages)
        wire_bytes = json.dumps(
            list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
    context_bytes = json.dumps(
        context_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    context_digest = "sha256:" + hashlib.sha256(context_bytes).hexdigest()
    # Bind proofs to the submitted wire, but only current log pages can satisfy
    # source obligations. Preserve indices and literal live-page bytes.
    from kernel_cache_layout_view import source_messages
    messages = source_messages(messages, unwrap_pages=True)
    submitted_http_bytes_digest = (
        "sha256:" + hashlib.sha256(wire_bytes).hexdigest() if payload_is_http else None
    )
    try:
        decoded_wire = json.loads(wire_bytes.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        decoded_wire = None
    if isinstance(decoded_wire, Mapping) and isinstance(decoded_wire.get("messages"), list):
        message_location_prefix = "$.messages"
    elif isinstance(decoded_wire, list):
        message_location_prefix = "$"
    else:
        message_location_prefix = None
    raw_deferred_paths = plan.get("deferred_paths", ())
    deferred_paths = _normalized_deferred_paths(
        raw_deferred_paths
        if isinstance(raw_deferred_paths, Sequence)
        and not isinstance(raw_deferred_paths, (str, bytes, bytearray))
        else ()
    )
    exact_notice, wire_read_tool_available = _wire_deferred_notice_state(
        decoded_wire, deferred_paths,
    )
    deferred_notice_verified = exact_notice and wire_read_tool_available

    def final_message_location(message_index: int) -> str | None:
        if message_location_prefix is None:
            return None
        return f"{message_location_prefix}[{message_index}].content"

    # LEAN can keep the request ending on the latest tool result by placing a
    # newly refreshed source block inside that result. Parse the block from its
    # own header while binding every proof to the containing serialized message.
    for message_index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        from kernel_source_presentation import strip_callsite_census
        source_content = strip_callsite_census(content)
        for raw_path, expected_revision in declared_revisions.items():
            path = str(raw_path).replace("\\", "/")
            source = source_map.get(path)
            if (not isinstance(source, str)
                    or source_revision(source) != str(expected_revision)):
                continue
            header = re.compile(
                rf"(?m)^\[(?:Current|Working) source: {re.escape(path)}\]$",
            )
            matches = tuple(header.finditer(source_content))
            match = next((item for item in reversed(matches) if item.start() > 0), None)
            if match is None:
                continue
            block = source_content[match.start():]
            model_metadata = parse_model_source_window(block)
            if (isinstance(model_metadata, Mapping)
                    and str(model_metadata.get("path", "")).replace("\\", "/") == path
                    and verify_model_source_window(block, path=path, source=source)):
                raw_ranges = model_metadata.get("ranges")
                ranges = tuple(
                    (int(row["start_line"]), int(row["end_line"]))
                    for row in raw_ranges if isinstance(row, Mapping)
                    and type(row.get("start_line")) is int
                    and type(row.get("end_line")) is int
                ) if isinstance(raw_ranges, list) else ()
            else:
                window_metadata = parse_window(block)
                raw_ranges = (
                    window_metadata.get("ranges")
                    if isinstance(window_metadata, Mapping) else None
                )
                if (isinstance(window_metadata, Mapping)
                        and str(window_metadata.get("path", "")).replace("\\", "/") == path
                        and verify_window(block, path=path, source=source)):
                    ranges = tuple(
                        (int(row["start_line"]), int(row["end_line"]))
                        for row in raw_ranges if isinstance(row, Mapping)
                        and type(row.get("start_line")) is int
                        and type(row.get("end_line")) is int
                    ) if isinstance(raw_ranges, list) else ()
                elif source in source_content[match.end():]:
                    line_count = len(source.splitlines(keepends=True))
                    ranges = ((1, line_count),) if line_count else ()
                else:
                    ranges = ()
            if ranges:
                remember_ranges(path, ranges, message_index, content)

    for message_index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, Mapping) else None
        source_content = content
        if isinstance(source_content, str):
            from kernel_source_presentation import strip_callsite_census
            source_content = strip_callsite_census(source_content)
        model_metadata = parse_model_source_window(source_content) if isinstance(source_content, str) else None
        if isinstance(model_metadata, Mapping):
            model_path = str(model_metadata.get("path", "")).replace("\\", "/")
            model_source = source_map.get(model_path)
            expected_model_revision = declared_revisions.get(model_path)
            if (isinstance(content, str)
                    and _verified_prior_source_message(
                        plan, model_path, content, expected_model_revision)):
                continue
            if str(model_metadata.get("status", "")) == "ABSENT":
                integrity_omissions.append({
                    "kind": "source_explicit_absence", "path": model_path,
                    "message": message_index,
                    "reason": model_metadata.get("absence", "UNKNOWN"),
                })
                continue
            expected_model_revision = declared_revisions.get(model_path)
            if (expected_model_revision is None
                    or str(model_metadata.get("revision", "")) != str(expected_model_revision)):
                integrity_omissions.append({
                    "kind": "source_revision_mismatch", "path": model_path,
                    "message": message_index,
                })
                continue
            if model_source is None or not verify_model_source_window(
                    content, path=model_path, source=model_source):
                integrity_omissions.append({
                    "kind": "source_projection_bytes_mismatch" if model_source is not None
                    else "source_bytes_unavailable",
                    "path": model_path, "message": message_index,
                })
                continue
            model_ranges = model_metadata.get("ranges")
            if isinstance(model_ranges, list):
                model_range_values = tuple(
                    (int(row["start_line"]), int(row["end_line"]))
                    for row in model_ranges if isinstance(row, Mapping)
                    and type(row.get("start_line")) is int
                    and type(row.get("end_line")) is int
                )
                remember_ranges(model_path, model_range_values, message_index, content)
            continue
        metadata = parse_window(source_content) if isinstance(source_content, str) else None
        ranges = metadata.get("ranges") if isinstance(metadata, Mapping) else None
        path = metadata.get("path") if isinstance(metadata, Mapping) else None
        if isinstance(source_content, str) and isinstance(declared_revisions, Mapping):
            for candidate_path, revision in declared_revisions.items():
                prefixes = (
                    f"[Current source: {candidate_path}]\n",
                    f"[Working source: {candidate_path}]\n",
                )
                if not source_content.startswith(prefixes):
                    continue
                if _verified_prior_source_message(
                        plan, str(candidate_path), content, revision):
                    continue
                full_source = source_content.split("\n", 1)[1]
                if full_source.startswith("Coverage: COMPLETE FILE;"):
                    _coverage, newline, full_source = full_source.partition("\n")
                    if not newline:
                        full_source = ""
                known_source = source_map.get(str(candidate_path))
                # A bounded source window can follow the public source header.
                # Validate that framed body here: treating an invalid window as
                # absent would let later deferral hide a byte/revision failure.
                model_metadata = parse_model_source_window(full_source)
                if isinstance(model_metadata, Mapping):
                    model_path = str(model_metadata.get("path", "")).replace("\\", "/")
                    model_revision = str(model_metadata.get("revision", ""))
                    if model_path != str(candidate_path):
                        integrity_omissions.append({
                            "kind": "source_projection_path_mismatch",
                            "path": str(candidate_path), "message": message_index,
                        })
                    elif model_revision != str(revision):
                        integrity_omissions.append({
                            "kind": "source_revision_mismatch",
                            "path": str(candidate_path), "message": message_index,
                        })
                    elif not isinstance(known_source, str):
                        integrity_omissions.append({
                            "kind": "source_bytes_unavailable",
                            "path": str(candidate_path), "message": message_index,
                        })
                    elif not verify_model_source_window(
                            full_source, path=str(candidate_path), source=known_source):
                        integrity_omissions.append({
                            "kind": "source_projection_bytes_mismatch",
                            "path": str(candidate_path), "message": message_index,
                        })
                    else:
                        raw_ranges = model_metadata.get("ranges")
                        ranges = tuple(
                            (int(row["start_line"]), int(row["end_line"]))
                            for row in raw_ranges if isinstance(row, Mapping)
                            and type(row.get("start_line")) is int
                            and type(row.get("end_line")) is int
                        ) if isinstance(raw_ranges, list) else ()
                        if ranges:
                            remember_ranges(str(candidate_path), ranges, message_index, content)
                    continue
                # Preserve the checked path header; verify_window validates the framed body.
                source_header = source_content.split("\n", 1)[0]
                framed_source_window = f"{source_header}\n{full_source}"
                window_metadata = parse_window(full_source)
                if isinstance(window_metadata, Mapping):
                    window_path = str(window_metadata.get("path", "")).replace("\\", "/")
                    window_revision = str(window_metadata.get("revision", ""))
                    if window_path != str(candidate_path):
                        integrity_omissions.append({
                            "kind": "source_window_path_mismatch",
                            "path": str(candidate_path), "message": message_index,
                        })
                    elif window_revision != str(revision):
                        integrity_omissions.append({
                            "kind": "source_revision_mismatch",
                            "path": str(candidate_path), "message": message_index,
                        })
                    elif not isinstance(known_source, str):
                        integrity_omissions.append({
                            "kind": "source_bytes_unavailable",
                            "path": str(candidate_path), "message": message_index,
                        })
                    elif not verify_window(
                            framed_source_window, path=str(candidate_path), source=known_source):
                        integrity_omissions.append({
                            "kind": "source_range_bytes_mismatch",
                            "path": str(candidate_path), "message": message_index,
                            "declared_revision": str(revision),
                            "observed_revision": window_revision,
                        })
                    else:
                        raw_ranges = window_metadata.get("ranges")
                        ranges = tuple(
                            (int(row["start_line"]), int(row["end_line"]))
                            for row in raw_ranges if isinstance(row, Mapping)
                            and type(row.get("start_line")) is int
                            and type(row.get("end_line")) is int
                        ) if isinstance(raw_ranges, list) else ()
                        if ranges:
                            remember_ranges(str(candidate_path), ranges, message_index, content)
                    continue
                source_body, separator, range_trailer = full_source.rpartition("\nCoverage: lines ")
                trailer_match = re.fullmatch(
                    r"([0-9]+)-([0-9]+), columns ([0-9]+)-([0-9]+); "
                    r"current bytes are delivered for this range\.", range_trailer)
                if separator and trailer_match is not None and full_source != known_source:
                    start, end, first_column, last_column = map(int, trailer_match.groups())
                    known_lines = known_source.splitlines(keepends=True) if known_source is not None else []
                    # F4's 0-0 columns mean complete selected lines. Validate
                    # the exact revision/range bytes before accepting framing.
                    expected_range = (
                        "".join(known_lines[start - 1:end])
                        if 1 <= start <= end <= len(known_lines) else ""
                    )
                    exact_range = source_body == expected_range
                    if (expected_range and not expected_range.endswith(("\n", "\r"))
                            and source_body == expected_range + "\n"):
                        # The final newline separates exact no-newline source bytes
                        # from the coverage trailer added by the pin renderer.
                        exact_range = True
                    if (expected_range.endswith(("\n", "\r"))
                            and source_body + "\n" == expected_range):
                        # rpartition consumes the LF that separates a newline-
                        # terminated source range from its coverage trailer.
                        exact_range = True
                    if (known_source is not None
                            and source_revision(known_source) == str(revision)
                            and first_column == last_column == 0
                            and 1 <= start <= end <= len(known_lines)
                            and exact_range):
                        remember_ranges(str(candidate_path), ((start, end),), message_index, content)
                    else:
                        integrity_omissions.append({
                            "kind": "source_range_bytes_mismatch", "path": str(candidate_path),
                            "message": message_index,
                            "declared_revision": str(revision),
                            "known_source_revision": (
                                source_revision(known_source)
                                if isinstance(known_source, str) else None
                            ),
                            "expected_range_sha256": (
                                hashlib.sha256(expected_range.encode("utf-8")).hexdigest()
                                if expected_range else None
                            ),
                            "observed_range_sha256": hashlib.sha256(
                                source_body.encode("utf-8")).hexdigest(),
                            "expected_range_bytes": len(expected_range.encode("utf-8")),
                            "observed_range_bytes": len(source_body.encode("utf-8")),
                        })
                    continue
                if known_source is not None and full_source != known_source:
                    integrity_omissions.append({
                        "kind": "source_bytes_mismatch", "path": str(candidate_path),
                        "message": message_index,
                    })
                    continue
                if source_revision(full_source) != str(revision):
                    integrity_omissions.append({
                        "kind": "source_revision_mismatch", "path": str(candidate_path),
                        "message": message_index,
                    })
                    continue
                line_count = len(full_source.splitlines(keepends=True))
                remember_ranges(
                    str(candidate_path),
                    ((1, line_count) if line_count else (1, 0),),
                    message_index,
                    content,
                )
                source_map.setdefault(str(candidate_path), full_source)
                break
        if not isinstance(path, str) or not isinstance(ranges, list):
            continue
        source = source_map.get(path)
        expected_revision = declared_revisions.get(path)
        if source is None:
            integrity_omissions.append({
                "kind": "source_bytes_unavailable", "path": path,
                "message": message_index,
            })
            continue
        if expected_revision is None or source_revision(source) != str(expected_revision):
            integrity_omissions.append({
                "kind": "source_revision_mismatch", "path": path,
                "message": message_index,
            })
            continue
        window_for_validation = content
        if not content.splitlines() or _HEADER.match(content.splitlines()[0]) is None:
            # The standalone source-window renderer is also used as a message
            # body without the optional Current/Working source wrapper.
            window_for_validation = f"[Current source: {path}]\n{content}"
        if not verify_window(window_for_validation, path=path, source=source):
            integrity_omissions.append({
                "kind": "source_window_bytes_mismatch", "path": path,
                "message": message_index,
            })
            continue
        window_range_values = tuple(
            (int(row["start_line"]), int(row["end_line"]))
            for row in ranges
            if isinstance(row, Mapping)
            and type(row.get("start_line")) is int
            and type(row.get("end_line")) is int
        )
        remember_ranges(path, window_range_values, message_index, content)
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping) or str(message.get("role", "")) != "tool":
            continue
        read_page = _read_page_source_range(
            message.get("content"), source_map, declared_revisions,
        )
        if read_page is not None:
            read_path, read_range, source_from_page = read_page
            if source_from_page is not None:
                source_map[read_path] = source_from_page
            remember_ranges(read_path, (read_range,), message_index,
                            str(message.get("content", "")))
    proofs_by_path: dict[str, list[dict[str, object]]] = {}
    if authorization_verified:
        for path, records in range_records_by_path.items():
            source = source_map.get(path)
            if source is None:
                continue
            raw = source.encode("utf-8")
            lines = source.splitlines(keepends=True)
            try:
                handle = content_handle(
                    path, raw, workspace_scope=workspace_scope,
                    workspace_generation=workspace_generation,
                )
            except (TypeError, ValueError):
                integrity_omissions.append({"kind": "delivery_proof_handle", "path": path})
                continue
            for (start_line, end_line), location, message_content in records:
                if not (1 <= start_line <= end_line <= len(lines)):
                    integrity_omissions.append({
                        "kind": "delivery_range_invalid", "path": path,
                        "range": [start_line, end_line],
                    })
                    continue
                if location is None:
                    integrity_omissions.append({
                        "kind": "final_message_location_unavailable", "path": path,
                        "range": [start_line, end_line],
                    })
                    continue
                byte_start = sum(len(item.encode("utf-8")) for item in lines[:start_line - 1])
                exact = b"".join(item.encode("utf-8") for item in lines[start_line - 1:end_line])
                byte_end = byte_start + len(exact)
                key = ReadKey(
                    path, workspace_scope, handle.content_digest,
                    byte_start, byte_end, handle.decoding,
                    handle.presentation_version, workspace_generation,
                )
                try:
                    proof = proof_for_serialized_request(
                        key, exact, wire_bytes,
                        request_id=context_digest,
                        final_message_context=message_content,
                        expected_message_location=location,
                        context_digest=context_digest,
                        submitted_http_bytes=wire_bytes if payload_is_http else None,
                    )
                except (TypeError, UnicodeError, ValueError):
                    integrity_omissions.append({
                        "kind": "delivery_proof_bytes", "path": path,
                        "range": [start_line, end_line],
                    })
                    continue
                proofs_by_path.setdefault(path, []).append(proof.as_dict())
    selected_paths = {
        str(item.get("path"))
        for item in plan.get("units", ())
        if isinstance(item, Mapping) and str(item.get("path", "")).strip()
    }
    proof_failures = [
        omission for omission in integrity_omissions
        if not str(omission.get("path", "")).strip()
        or str(omission.get("path", "")) in selected_paths
    ]
    raw_dropped_paths = plan.get("dropped_source_paths", ())
    dropped_paths = _normalized_deferred_paths(
        raw_dropped_paths
        if isinstance(raw_dropped_paths, Sequence)
        and not isinstance(raw_dropped_paths, (str, bytes, bytearray))
        else ()
    )
    declared_paths = plan.get("paths", ())
    declared_paths = {
        str(path).replace("\\\\", "/") for path in declared_paths
    } if isinstance(declared_paths, Sequence) and not isinstance(
        declared_paths, (str, bytes, bytearray)
    ) else set()
    revisions_value = plan.get("revisions", {})
    revision_paths = {
        str(path).replace("\\\\", "/") for path in revisions_value
    } if isinstance(revisions_value, Mapping) else set()
    if dropped_paths and (
            set(dropped_paths) & (selected_paths | declared_paths | revision_paths)
            or _dropped_source_claim_visible(messages, dropped_paths)):
        proof_failures.append({"kind": "dropped_source_claim_still_visible"})
    selected_deferred_paths = set(deferred_paths) & selected_paths
    if deferred_paths and not deferred_notice_verified:
        proof_failures.append({
            "kind": "deferred_source_notice_unverified",
            "paths": list(deferred_paths),
            "read_tool_available": wire_read_tool_available,
        })
    unselected_deferred_paths = set(deferred_paths) - selected_paths
    if unselected_deferred_paths:
        proof_failures.append({
            "kind": "deferred_source_path_unselected",
            "paths": sorted(unselected_deferred_paths),
        })
    if not authorization_verified:
        proof_failures.append({
            "kind": "workspace_authorization_unverified",
            "status": "UNVERIFIED",
        })
    for path in selected_paths:
        expected_proofs = range_records_by_path.get(path, ())
        actual_proofs = proofs_by_path.get(path, ())
        if path not in selected_deferred_paths and not expected_proofs:
            proof_failures.append({
                "kind": "final_wire_proof_missing",
                "path": path,
                "expected": 1,
                "actual": 0,
            })
        elif path not in selected_deferred_paths and len(actual_proofs) < len(expected_proofs):
            proof_failures.append({
                "kind": "final_wire_proof_incomplete",
                "path": path,
                "expected": len(expected_proofs),
                "actual": len(actual_proofs),
            })
        path_source = source_map.get(path)
        if (path not in selected_deferred_paths and expected_proofs
                and isinstance(path_source, str) and path_source.strip()):
            # Exact bytes are necessary, not sufficient: a window whose every
            # delivered line is whitespace (J857: blank line 22 of 599) carries
            # no code, so it can never complete delivery of a non-empty file.
            path_lines = path_source.splitlines(keepends=True)
            delivered_text = "".join(
                "".join(path_lines[start - 1:end])
                for (start, end), _location, _content in expected_proofs
                if 1 <= start <= end <= len(path_lines)
            )
            if not delivered_text.strip():
                proof_failures.append({
                    "kind": "source_body_blank",
                    "path": path,
                    "ranges": [[start, end] for (start, end), _l, _c in expected_proofs],
                })
    proof_failure = bool(proof_failures)
    receipt = record_final_source_delivery(
        plan,
        ranges_by_path,
        payload_revision=context_digest,
        context_digest=context_digest,
        submitted_http_bytes_digest=submitted_http_bytes_digest,
        wire_proof_verified=not proof_failure,
        wire_failure_reason=(
            "final message location, revision, range, byte, or authorization proof failed"
            if proof_failure else ""
        ),
    )
    receipt["submitted_payload_hash"] = context_digest
    receipt["proof_status"] = "FAILED" if proof_failure else "VERIFIED"
    if dropped_paths and not selected_paths and not proof_failure and receipt.get("status") == "UNKNOWN":
        receipt["status"] = "COMPLETE"
        receipt["dropped_source_claims_verified"] = True
    receipt["context_digest"] = context_digest
    receipt["submitted_http_bytes_digest"] = submitted_http_bytes_digest or "UNVERIFIED"
    receipt["workspace_authorization"] = {
        "scope": workspace_scope or "UNKNOWN",
        "generation": workspace_generation or "UNKNOWN",
        "status": "VERIFIED" if authorization_verified else "UNVERIFIED",
    }
    receipt["final_message_locations"] = {
        path: tuple(locations) for path, locations in locations_by_path.items()
    }
    receipt["delivery_proofs"] = proofs_by_path
    receipt["proof_failures"] = copy.deepcopy(proof_failures)
    if integrity_omissions:
        receipt["integrity_omissions"] = integrity_omissions
    if isinstance(plan, dict):
        plan["delivery_proofs"] = copy.deepcopy(proofs_by_path)
        plan["proof_failures"] = copy.deepcopy(proof_failures)
        plan["integrity_omissions"] = copy.deepcopy(integrity_omissions)
        plan["context_digest"] = context_digest
        plan["submitted_http_bytes_digest"] = submitted_http_bytes_digest or "UNVERIFIED"
        plan["proof_status"] = receipt["proof_status"]
    revisions = plan.get("revisions", {})
    for path, result in receipt.get("paths", {}).items():
        if not isinstance(result, dict):
            continue
        result["revision"] = revisions.get(path, "UNKNOWN") if isinstance(revisions, Mapping) else "UNKNOWN"
        result["ranges"] = [list(item) for item in ranges_by_path.get(path, ())]
        result["workspace_authorization"] = receipt["workspace_authorization"]
        result["final_message_locations"] = tuple(locations_by_path.get(path, ()))
        result["proof_status"] = receipt["proof_status"]
    verified_deferred_paths = _apply_deferred_delivery_status(
        plan, receipt, deferred_paths, selected_paths,
        notice_verified=deferred_notice_verified,
    )
    receipt["deferred_paths"] = list(verified_deferred_paths)
    receipt["deferred_notice_verified"] = deferred_notice_verified
    receipt["read_tool_available"] = wire_read_tool_available
    if isinstance(plan, dict):
        plan["submitted_payload_hash"] = receipt["submitted_payload_hash"]
    if isinstance(plan, dict):
        plan["final_delivery_status"] = receipt["status"]
    return receipt


def _ensure_target_unit(plan: dict[str, object], path: str,
                        reservation: Mapping[str, object]) -> None:
    """Give the target a required unit when the closure planner selected none.

    Without one, a fully delivered and byte-proven target could never be
    COMPLETE (``selected_units == 0``).  The unit is the host's own source
    reservation span, so no second planner or parser is introduced.
    """
    paths = plan.get("paths")
    if not isinstance(paths, (list, tuple)) or path not in [str(item) for item in paths]:
        return
    units = plan.get("units")
    if not isinstance(units, list):
        return
    if any(isinstance(unit, Mapping) and str(unit.get("path")) == path for unit in units):
        return
    lines = [int(line) for line in reservation.get("selected_lines", ()) or ()
             if isinstance(line, int) and line >= 1]
    if reservation.get("path") != path or not lines:
        return
    units.append({
        "path": path, "start_line": min(lines), "end_line": max(lines),
        "role": "target_reservation", "reason": "source_reservation", "priority": 1,
        "handle": f"unit:target-reservation:{min(lines)}-{max(lines)}",
    })


# Panels the request diet sheds before operational evidence.  The target body
# is sized as if they were absent: it is reserved first and the diet then
# trims these panels to fit, never the other way round.
_PANELS_YIELDING_TO_TARGET = frozenset({
    "history_ledger", "passive_conclusions", "coordination_ledger",
    "get_sure_state", "environment_state", "ship_tree",
})


_OPTIONAL_BUDGET_SECTIONS = frozenset({
    "optional", "advice", "playbook", "history", "history_ledger", "diagnostic", "scratch",
})


def _is_optional_panel(message: object) -> bool:
    """An explicitly optional message or a panel the diet sheds first."""
    if not isinstance(message, Mapping) or message.get("role") in {"system", "developer", "tool"} \
            or message.get("tool_calls"):
        return False
    explicit = str(message.get("_request_budget_section") or "").strip().lower()
    if explicit:
        return explicit in _OPTIONAL_BUDGET_SECTIONS
    from kernel_request_diet import section_name
    return section_name(message) in _PANELS_YIELDING_TO_TARGET


def _yields_to_required_source(message: object) -> bool:
    """True for content the target source body is allocated ahead of."""
    if not isinstance(message, Mapping) or message.get("role") == "tool" \
            or message.get("tool_calls"):
        return False
    from kernel_request_diet import section_name, source_window_path
    content = str(message.get("content") or "")
    if source_window_path(content) is not None:
        # Another path's bounded window (the target's own message is excluded
        # by the caller); connected excerpts never starve the target body.
        return True
    return section_name(message) in _PANELS_YIELDING_TO_TARGET


def _window_has_nonblank_source_body(content: object) -> bool:
    """A valid range can still carry no code when every delivered line is blank."""
    if not isinstance(content, str):
        return False
    metadata = parse_window(content)
    if metadata is not None:
        return any(text.strip() for _row, text in _window_segments(content, metadata))
    model_metadata = parse_model_source_window(content)
    if model_metadata is None:
        return False
    return any(text.strip() for _row, text in _model_window_segments(content, model_metadata))


def record_deferred_source_paths(
        plan: dict[str, object], messages: list[dict[str, object]],
        paths: Sequence[object],
) -> tuple[str, ...]:
    """Keep selected-path deferrals and their canonical read notice in sync."""
    requested = _normalized_deferred_paths(paths)
    raw_units = plan.get("units", ())
    units = (
        raw_units if isinstance(raw_units, Sequence)
        and not isinstance(raw_units, (str, bytes, bytearray)) else ()
    )
    selected_paths = {
        normalized[0]
        for unit in units if isinstance(unit, Mapping)
        and (normalized := _normalized_deferred_paths((unit.get("path", ""),)))
    }
    deferred = tuple(path for path in requested if path in selected_paths)
    if not deferred:
        return ()

    current = plan.get("deferred_paths", ())
    existing = (
        current if isinstance(current, Sequence)
        and not isinstance(current, (str, bytes, bytearray)) else ()
    )
    combined = _normalized_deferred_paths((*existing, *deferred))
    plan["deferred_paths"] = list(combined)
    upsert_deferred_source_notice(messages, combined)
    return deferred


def settle_blank_source_windows(
        plan: dict[str, object], messages: list[dict[str, object]],
) -> tuple[str, ...]:
    """Replace blank-only source messages with the existing verified read route."""
    source_indices: dict[str, list[int]] = {}
    blank_indices: dict[str, list[int]] = {}
    nonblank_paths: set[str] = set()
    for index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue

        model_metadata = parse_model_source_window(content)
        window_metadata = parse_window(content)
        if model_metadata is not None and model_metadata.get("status") != "ABSENT":
            path = str(model_metadata.get("path", "")).replace("\\", "/").strip()
            segments = _model_window_segments(content, model_metadata)
            if not path or not segments:
                continue
            body = "".join(text for _row, text in segments)
        elif window_metadata is not None:
            path = str(window_metadata.get("path", "")).replace("\\", "/").strip()
            segments = _window_segments(content, window_metadata)
            if not path or not segments:
                continue
            body = "".join(text for _row, text in segments)
        else:
            lines = content.splitlines()
            header = _HEADER.match(lines[0] if lines else "")
            if header is None:
                continue
            path = header.group(1).replace("\\", "/").strip()
            _header, separator, body = content.partition("\n")
            if not separator:
                body = ""
            if body.startswith("Coverage: COMPLETE FILE;"):
                _coverage, newline, body = body.partition("\n")
                if not newline:
                    body = ""

        if not path:
            continue
        source_indices.setdefault(path, []).append(index)
        if body.strip():
            nonblank_paths.add(path)
        else:
            blank_indices.setdefault(path, []).append(index)

    blank_paths = tuple(
        path for path in source_indices if path not in nonblank_paths
    )
    removal_indices = {
        index for indices in blank_indices.values() for index in indices
    }
    if not removal_indices:
        return ()

    selected_paths = {
        normalized[0]
        for unit in plan.get("units", ())
        if isinstance(unit, Mapping)
        and (normalized := _normalized_deferred_paths((unit.get("path", ""),)))
    }
    messages[:] = [
        message for index, message in enumerate(messages)
        if index not in removal_indices
    ]
    deferred = tuple(path for path in blank_paths if path in selected_paths)
    return record_deferred_source_paths(plan, messages, deferred)


@profiled_function_span("request_source_windowing")
def _window_path_messages(
        messages: list[dict[str, object]], runtime: object, *, path: str, source: str,
        max_request_bytes: int, tools: Sequence[object], changed_paths: list[str],
        is_target: bool,
) -> None:
    lines = source.splitlines(keepends=True)
    expected_prefixes = (
        f"[Current source: {path}]\n", f"[Working source: {path}]\n",
    )
    full_candidates = {
        prefix + source for prefix in expected_prefixes
    }
    scope_lines = len(lines)
    if getattr(runtime, "working_source_scope", False):
        full_candidates.update(
            prefix + f"Coverage: COMPLETE FILE; lines {1 if scope_lines else 0}-{scope_lines} of {scope_lines}; nothing omitted.\n" + source
            for prefix in expected_prefixes
        )
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        is_full_source = isinstance(content, str) and content in full_candidates
        is_valid_window = (
            isinstance(content, str)
            and is_source_window(content)
            and verify_window(content, path=path, source=source)
        )
        is_valid_model_window = (
            isinstance(content, str)
            and parse_model_source_window(content) is not None
            and verify_model_source_window(content, path=path, source=source)
        )
        if not is_full_source and not is_valid_window and not is_valid_model_window:
            continue
        non_source = {
            "messages": [item for item in messages
                         if item is not message
                         and not (is_target and _yields_to_required_source(item))],
            "tools": list(tools),
        }
        non_source_bytes = len(json.dumps(non_source).encode())
        available = max(_MIN_WINDOW_CHARS, int(max_request_bytes) - non_source_bytes - 768)
        # Preserve an exact full carrier when it fits; wrapping it adds bytes
        # and can make repeated connected-source acquisition non-idempotent.
        if is_full_source and len(content.encode("utf-8")) <= min(12_000, available):
            continue
        if (is_valid_window or is_valid_model_window) \
                and len(str(content).encode("utf-8")) <= available:
            metadata = (
                parse_window(content) if is_valid_window
                else parse_model_source_window(content)
            )
            partial = isinstance(metadata, Mapping) and (
                metadata.get("omitted_lines") is True
                or metadata.get("status") == "CONTINUATION_AVAILABLE"
            )
            has_body = _window_has_nonblank_source_body(content)
            if has_body and (not partial or available <= len(str(content).encode("utf-8")) + 128):
                continue
        rendered = bound_source_message(
            str(content), path=path, source=source, runtime=runtime,
            max_chars=min(12_000, available), is_target=is_target,
        )
        if rendered != content:
            message["content"] = rendered
            changed_paths.append(path)


# =============================================================== host site facts (P2 / P3)

SITE_FACTS_ENV = "STATEFRAME_SITE_FACTS"
INVENTORY_HEADER = "[LITERAL SITE INVENTORY"
ENTRY_HEADER = "[ENTRY SYMBOL VERIFICATION"
MODULE_HEAD_HEADER = "[MODULE HEAD"
IMPORT_FACT_HEADER = "[IMPORT GRAPH FACT"
OWNER_COORDS_HEADER = "[SOURCE OWNER COORDINATES"
_MAX_TOKENS = 8
_MAX_SITES = 40
_MAX_RENDERED_SITES = 20
_MAX_LITERAL_ANCHORS = 12
_MAX_FILE_BYTES = 2_000_000
_MAX_EDITABLE_FILES = 16
_MODULE_HEAD_MAX_LINES = 120
_MODULE_HEAD_ROWS = 30
_MODULE_HEAD_TRIGGER = re.compile(
    r"\b(?:import(?:s|ed|ing)?|modules?|defaults?|engines?)\b", re.IGNORECASE)
_COMMENT_PREFIXES = ("#", "//", "/*", "* ", "<!--", "--")
_QUOTED = re.compile(r"(['\"`])([^'\"`\n]{3,80})\1")
_SLASH_DIGIT = re.compile(
    r"(?<![\w/.:-])([A-Za-z][A-Za-z0-9.]*(?:[/-][A-Za-z0-9.]+)+|[A-Za-z][A-Za-z_]*\d[A-Za-z0-9_]*)"
    r"(?![\w/-])")
_CODE_IDENT = re.compile(
    r"(?<![\w.])((?:[A-Za-z_$][A-Za-z0-9_$]*\.)*"
    r"[A-Za-z_$][A-Za-z0-9_$]*)"
    r"(?![\w.])")
_CALL_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")


def _code_identifier(value: str) -> bool:
    terminal = value.rsplit(".", 1)[-1]
    return "." in value or "_" in terminal or bool(re.search(r"[a-z][A-Z]", terminal))
_TEST_PATH = re.compile(
    r"(?:^|/)(?:tests?|__tests__|specs?)/|(?:^|/)test_[^/]*$|_test\.[A-Za-z]+$"
    r"|\.(?:spec|test)\.[A-Za-z]+$|(?:^|/)conftest\.py$")
_FORBIDDEN_PARTS = frozenset({"grader", "graders", "gold", "private", "hidden", "hidden_tests",
                              "backups", "evaluator"})
_SENTINEL_SYMBOLS = frozenset({"__multi_file__", "feature_behavior"})
_WORD_STOPWORDS = frozenset({
    "that", "this", "with", "from", "into", "when", "then", "than", "them", "they", "their",
    "there", "these", "those", "have", "been", "were", "will", "would", "should", "could",
    "must", "only", "each", "every", "also", "after", "before", "about", "other", "which",
    "while", "where", "what", "does", "done", "make", "made", "uses", "used", "using",
    "file", "files", "line", "lines", "code", "test", "tests", "task", "ticket", "change",
    "changes", "import", "imports", "module", "modules", "default", "defaults", "value",
    "values", "first", "still", "true", "false", "none", "self", "return", "returns",
})


def site_facts_mode(environ: Mapping[str, str] | None = None) -> str:
    """``STATEFRAME_SITE_FACTS`` (default ``on``; ``off`` restores the pre-P2 window)."""
    values = os.environ if environ is None else environ
    value = str(values.get(SITE_FACTS_ENV, "on") or "on").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{SITE_FACTS_ENV} must be off or on")
    return value


@dataclass(frozen=True)
class Site:
    path: str
    line: int
    text: str
    enclosing: str
    token: str
    role: str = "executable"
    language: str = "unknown"
    revision: str = ""
    symbol_name: str = ""
    symbol_owner: str = ""
    symbol_start_line: int = 0
    symbol_end_line: int = 0


def _is_comment(line: str) -> bool:
    return line.strip().startswith(_COMMENT_PREFIXES)


def _python_docstring_lines(source: str) -> frozenset[int]:
    """Return docstring lines so prose is not counted as executable use."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()
    lines: set[int] = set()
    scopes = [tree, *[node for node in ast.walk(tree)
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]]
    for scope in scopes:
        body = getattr(scope, "body", ())
        if not body:
            continue
        first = body[0]
        value = getattr(first, "value", None)
        if isinstance(first, ast.Expr) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            start = int(getattr(first, "lineno", 0) or 0)
            end = int(getattr(first, "end_lineno", start) or start)
            lines.update(range(start, end + 1))
    return frozenset(lines)


def _site_role(path: str, source: str, line: int, docstring_lines: frozenset[int]) -> str:
    current = source.splitlines()[line - 1] if line > 0 else ""
    if _is_comment(current):
        return "comment"
    if path.casefold().endswith(".py") and line in docstring_lines:
        return "docstring"
    return "executable"


def _looks_like_path(token: str) -> bool:
    return token.count("/") >= 2 or bool(re.search(r"/[^/]*\.[A-Za-z]{1,5}$", token))


def _literal_candidates(text: str) -> list[tuple[str, str]]:
    """(query, original) pairs in first-appearance order."""
    rows: list[tuple[int, str, str]] = []
    for match in _QUOTED.finditer(text):
        value = match.group(2).strip()
        if re.search(r"[A-Za-z0-9]", value) and not _looks_like_path(value):
            rows.append((match.start(), value, value))
    for match in _SLASH_DIGIT.finditer(text):
        raw = match.group(1).strip(".")
        if _looks_like_path(raw):
            continue
        query = raw.replace("/", "").replace("-", "").lower() if re.search(r"[/-]", raw) else raw
        if len(query) >= 4 and re.search(r"[A-Za-z]", query):
            rows.append((match.start(), query, raw))
    for match in _CODE_IDENT.finditer(text):
        value = match.group(1)
        if len(value.rsplit(".", 1)[-1].strip("_")) >= 4 and _code_identifier(value):
            rows.append((match.start(), value, value))
    rows.sort(key=lambda row: row[0])
    return [(query, original) for _, query, original in rows]


def literal_tokens(ticket: str, focus: str = "", *,
                   exclude: Sequence[str] = ()) -> tuple[str, ...]:
    """Grep queries for the literals in Task/Focus text (deduplicated, bounded).

    Quoted strings, code-shaped identifiers (snake_case, camelCase,
    qualified/dotted names,
    UPPER_CASE) and tokens containing digits or slashes; ``/`` and ``-`` are
    removed and the token lower-cased when present (``HTTP/2`` -> ``http2``).
    Plain English words are never literals.
    """
    excluded = {str(item).casefold() for item in exclude if item}
    excluded.update(str(item).rsplit(".", 1)[-1].casefold() for item in exclude if item)
    seen: set[str] = set()
    result: list[str] = []
    for query, _original in _literal_candidates(f"{ticket}\n{focus}"):
        key = query.casefold()
        if key in seen or key in excluded or key in _WORD_STOPWORDS:
            continue
        seen.add(key)
        result.append(query)
        if len(result) >= _MAX_TOKENS:
            break
    return tuple(result)


def _token_origins(ticket: str, focus: str = "") -> dict[str, str]:
    origins: dict[str, str] = {}
    for query, original in _literal_candidates(f"{ticket}\n{focus}"):
        if query != original:
            origins.setdefault(query, original)
    return origins


@functools.lru_cache(maxsize=256)
def _literal_pattern_cached(token: str, original: str) -> re.Pattern[str]:
    alternatives = [re.escape(token)]
    if original and original != token:
        alternatives.append(re.escape(original))
    terminal = token.rsplit(".", 1)[-1]
    if "." in token and terminal != token:
        # A qualified ticket identity remains the emitted query, while its
        # terminal definition/call is still resolvable in ordinary imports.
        alternatives.append(re.escape(terminal))
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9_])",
                      re.IGNORECASE)


def _literal_pattern(token: str, original: str = "") -> re.Pattern[str]:
    return _literal_pattern_cached(token, original)


def entry_symbol(runtime: object) -> str:
    """The task row's declared entry symbol (``fn``), or "" when none/sentinel."""
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    values: list[object] = [getattr(owner, "fn", None)]
    if isinstance(row, Mapping):
        values.extend(row.get(key) for key in ("fn", "symbol", "symbol_name", "target_fn",
                                               "localized_symbol"))
        entry = row.get("entry")
        if isinstance(entry, Mapping):
            values.append(entry.get("fn"))
    for value in values:
        text = str(value or "").strip()
        if (text and text not in _SENTINEL_SYMBOLS
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", text)):
            return text
    return ""


def _declared_path_values(runtime: object) -> list[object]:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    values: list[object] = []
    target = _path_from_runtime(runtime)
    if target:
        values.append(target)
    contract = getattr(runtime, "edit_contract", None)
    if isinstance(contract, Mapping):
        authority = contract.get("edit_authority")
        for scope in (authority, contract):
            if isinstance(scope, Mapping):
                raw = scope.get("editable_paths", ())
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                    values.extend(raw)
    if isinstance(row, Mapping):
        for key in ("editable_paths", "target_files", "files", "source_paths", "entrypoints"):
            value = row.get(key)
            if isinstance(value, Mapping):
                values.extend(value.keys())
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values.extend(
                    item.get("path") if isinstance(item, Mapping) else item
                    for item in value
                )
        entry = row.get("entry")
        if isinstance(entry, Mapping):
            values.append(entry.get("path"))
    return values


def editable_product_paths(runtime: object) -> tuple[str, ...]:
    """Declared editable PRODUCT files (target first): existing, no tests, no evaluator scope."""
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None:
        return ()
    base = Path(root).resolve()
    result: list[str] = []
    for value in _declared_path_values(runtime):
        relative = str(value or "").strip().replace("\\", "/").removeprefix("./")
        if (not relative or relative.startswith("/") or _TEST_PATH.search(relative)
                or any(part.casefold() in _FORBIDDEN_PARTS for part in relative.split("/"))):
            continue
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        try:
            is_file = candidate.is_file()
        except (OSError, ValueError):
            continue
        if is_file:
            result.append(relative)
    target = _path_from_runtime(runtime)
    return tuple(sorted(set(result), key=lambda path: (path != target, path)))[:_MAX_EDITABLE_FILES]


def _read_product_file(root: Path, relative: str) -> str | None:
    path = (Path(root) / relative)
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _safe_framefit_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip().replace("\\", "/").removeprefix("./")
    parts = path.split("/")
    if (not path or path.startswith("/") or re.match(r"^[A-Za-z]:", path)
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in parts)
            or "\x00" in path or "\n" in path or "\r" in path):
        return None
    return path


def _model_requested_read_paths(messages: Sequence[object]) -> set[str]:
    requested: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping) or function.get("name") != "read":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            if isinstance(arguments, Mapping):
                path = _safe_framefit_path(arguments.get("path"))
                if path:
                    requested.add(path)
    return requested


def _task_requested_source_paths(runtime: object) -> set[str]:
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        return set()
    try:
        root = Path(root_value).resolve()
    except (OSError, TypeError, ValueError):
        return set()
    requested: set[str] = set()
    for match in _PATH.finditer(_ticket_text(runtime)):
        path = _safe_framefit_path(match.group(0).rstrip(".,;:!?)]}'\""))
        if path is None:
            continue
        try:
            candidate = (root / path).resolve()
            candidate.relative_to(root)
            if candidate.is_file():
                requested.add(path)
        except (OSError, ValueError):
            continue
    return requested


def _framefit_connected_paths(
        runtime: object, messages: Sequence[object], connected: Sequence[str], *,
        target_path: str, declared_paths: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    """Keep task-declared/model-read files and drop import-only closure windows."""
    from kernel_stateframe_f4_fit import framefit_mode
    mode = getattr(runtime, "stateframe_framefit_mode", None)
    if mode is None:
        mode = framefit_mode()
    if mode != "on" or not (
            getattr(runtime, "stateframe_f4_mode", "off") == "on"
            or getattr(runtime, "_hybrid_crossed", False)):
        return tuple(connected), 0
    allowed = {target_path} if target_path else set()
    allowed.update(path for value in declared_paths
                   if (path := _safe_framefit_path(value)))
    allowed.update(_task_requested_source_paths(runtime))
    allowed.update(_model_requested_read_paths(messages))
    retained = tuple(path for path in connected
                     if (normalized := _safe_framefit_path(path)) in allowed
                     and normalized is not None)
    return retained, max(0, len(connected) - len(retained))


def _framefit_deferred_target_present(messages: Sequence[object], *,
                                      runtime: object, path: str,
                                      source: str, tools: object) -> bool:
    from kernel_stateframe_f4_fit import (
        framefit_mode, is_valid_deferred_target_source_message,
    )
    mode = getattr(runtime, "stateframe_framefit_mode", None)
    if mode is None:
        mode = framefit_mode()
    if mode != "on" or not (
            getattr(runtime, "stateframe_f4_mode", "off") == "on"
            or getattr(runtime, "_hybrid_crossed", False)):
        return False
    return any(
        isinstance(message, Mapping)
        and is_valid_deferred_target_source_message(
            message.get("content"), path=path, source=source, tools=tools)
        for message in messages
    )


def _bracket_span(lines: Sequence[str], index: int) -> tuple[int, int]:
    """0-based (start, end) of the bracket-balanced statement around ``lines[index]``."""
    depth = 0
    start = index
    for cursor in range(index - 1, max(-1, index - 200), -1):
        for char in reversed(lines[cursor]):
            if char in ")]}":
                depth += 1
            elif char in "([{":
                depth -= 1
        if depth < 0:
            start = cursor
            break
    depth = 0
    end = index
    for cursor in range(start, min(len(lines), start + 400)):
        for char in lines[cursor]:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
        if cursor >= index and depth <= 0:
            end = cursor
            break
    return start, end


def owner_coordinates_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the owner-coordinate switch with strict off/on values."""
    from kernel_owner_first import coordinates_enabled
    return "on" if coordinates_enabled(environ) else "off"


def owner_coordinate_data(path: str, source: str) -> dict[str, object]:
    """Project top-level owner ranges from the shared language-generic outline."""
    normalized = str(path).replace("\\", "/")
    outline = _outline_for(normalized, source)
    lines = source.splitlines()
    owners: list[dict[str, object]] = []
    for collection, kind in (("types", "type"), ("functions", "function")):
        for raw in outline.get(collection, ()) or ():
            if not isinstance(raw, Mapping):
                continue
            parent = str(raw.get("parent") or "").strip()
            qualified = str(raw.get("qualified_name") or raw.get("name") or "").strip()
            if parent and parent.casefold() not in {"module", "<module>", "none"}:
                continue
            if not parent and "." in qualified:
                continue
            name = str(raw.get("name") or qualified.rsplit(".", 1)[-1]).strip()
            start = raw.get("line", raw.get("start_line"))
            end = raw.get("end_line", raw.get("end"))
            if not name or type(start) is not int or start < 1:
                continue
            if type(end) is not int or end < start:
                end = start
            owners.append({"name": name, "kind": kind,
                           "start_line": start, "end_line": end})
    if outline.get("degraded") and lines:
        starts = sorted({int(row["start_line"]) for row in owners})
        for row in owners:
            start, end = int(row["start_line"]), int(row["end_line"])
            if end > start or start > len(lines):
                continue
            next_start = next((line for line in starts if line > start), len(lines) + 1)
            declaration = lines[start - 1]
            if normalized.casefold().endswith(".py"):
                indentation = len(declaration) - len(declaration.lstrip())
                end = len(lines)
                for line_number, line in enumerate(lines[start:], start + 1):
                    if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                        end = line_number - 1
                        break
            else:
                _open, close = _bracket_span(lines, start - 1)
                end = close + 1 if close + 1 > start else next_start - 1
            row["end_line"] = max(start, min(end, len(lines), next_start - 1))
    distinct: dict[tuple[str, int, int, str], dict[str, object]] = {}
    total_lines = _line_count(source)
    for row in owners:
        start, end = int(row["start_line"]), int(row["end_line"])
        if start < 1 or end < start or end > total_lines:
            continue
        distinct.setdefault((str(row["name"]), start, end, str(row["kind"])), row)
    ordered = sorted(distinct.values(), key=lambda row: (
        int(row["start_line"]), int(row["end_line"]), str(row["name"]), str(row["kind"])))
    bounded = ordered[:40]
    revision = source_revision(source)
    for row in bounded:
        row["read"] = {
            "tool": "read", "path": normalized,
            "start_line": row["start_line"], "end_line": row["end_line"],
            "expected_revision": revision,
        }
    return {"schema": "stateframe-source-owner-coordinates/1", "path": normalized,
            "revision": revision, "owners": bounded, "owners_listed": len(bounded),
            "owners_omitted": max(0, len(ordered) - len(bounded))}


def owner_coordinate_block(data: Mapping[str, object]) -> str:
    path = str(data.get("path", ""))
    revision = str(data.get("revision", "UNKNOWN"))
    owners = [row for row in data.get("owners", ()) or () if isinstance(row, Mapping)]
    lines = [f"[SOURCE OWNER COORDINATES | {path} revision={revision} "
             f"owners_listed={len(owners)}]"]
    for row in owners:
        start, end = row.get("start_line"), row.get("end_line")
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            continue
        quoted_path = json.dumps(path, ensure_ascii=False)
        lines.append(f"  owner={row.get('name')} kind={row.get('kind')} lines={start}-{end} "
                     f"read(path={quoted_path}, start_line={start}, end_line={end}, "
                     f"expected_revision={json.dumps(revision)})")
    omitted = int(data.get("owners_omitted", 0) or 0)
    if omitted:
        lines.append(f"  ... {omitted} additional top-level owner(s); use outline(path={path})")
    return "\n".join(lines)


def owner_coordinate_is_current(record: Mapping[str, object], *, path: str,
                                source: str) -> bool:
    """Reject a coordinate when path, revision, or exact line bounds differ."""
    if str(record.get("path", "")).replace("\\", "/") != str(path).replace("\\", "/"):
        return False
    if record.get("revision") != source_revision(source):
        return False
    start, end = record.get("start_line"), record.get("end_line")
    return (type(start) is int and type(end) is int and start > 0
            and end >= start and end <= _line_count(source))


def owner_coordinates_for_runtime(runtime: object, path: str) -> dict[str, object] | None:
    if owner_coordinates_mode() != "on":
        return None
    normalized = str(path).replace("\\", "/")
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None or normalized not in editable_product_paths(runtime):
        return None
    source = _read_product_file(Path(root), normalized)
    return owner_coordinate_data(normalized, source) if source is not None else None


def _module_level_label(lines: Sequence[str], start: int, end: int) -> str:
    first = lines[start - 1].strip()[:72] if 0 < start <= len(lines) else ""
    if start == end:
        return f"module level: `{first}` (line {start}, no enclosing function)"
    return f"module level: `{first}` spans lines {start}-{end} (no enclosing function)"


# L112: 64 entries thrashed: state-owner inventory outlines every product file (~500-900)
# per call, so each call evicted the whole cache.  Pure function; size only.
@functools.lru_cache(maxsize=4096)
def _outline_cached(path: str, source: str, capability_key: str) -> Mapping[str, object]:
    from mc_repo_intel_vendor.ts_dispatch import outline_source
    return outline_source(source, path)


@profiled_function_span("outline_resolution")
def _outline_for(path: str, source: str) -> Mapping[str, object]:
    """Resolve an outline with parser capability in the cache identity."""

    from mc_repo_intel_vendor.ts_dispatch import (
        outline_capability_key,
        report_outline_parser_capability,
    )
    outline = _outline_cached(path, source, outline_capability_key(path, source))
    report_outline_parser_capability(outline)
    return outline


def _python_enclosing(source: str, line: int, lines: Sequence[str]) -> str | None:
    rows = [row for row in symbol_outline(source)
            if int(row["start_line"]) <= line <= int(row["end_line"])]
    if rows:
        row = max(rows, key=lambda item: (int(item["start_line"]), -int(item["end_line"])))
        return (f"inside `{row['qualified_name']}` "
                f"(lines {row['start_line']}-{row['end_line']})")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    for node in tree.body:
        start = int(getattr(node, "lineno", 0) or 0)
        end = int(getattr(node, "end_lineno", start) or start)
        if start <= line <= end:
            return _module_level_label(lines, start, end)
    return None


def _outlined_function_at_line(source: str, outline: Mapping[str, object],
                               line: int) -> tuple[Mapping[str, object], int, int] | None:
    """Find an outlined function span, extending degraded signature-only rows safely."""
    rows = [row for row in outline.get("functions", ()) or ()
            if isinstance(row, Mapping) and int(row.get("line", 0) or 0) <= line]
    if not rows:
        return None
    rows.sort(key=lambda item: int(item.get("line", 0) or 0))
    lines = source.splitlines()
    candidates: list[tuple[Mapping[str, object], int, int]] = []
    for index, row in enumerate(rows):
        start = int(row.get("line", 0) or 0)
        end = int(row.get("end_line", start) or start)
        if end < line:
            _span_start, bracket_end = _bracket_span(lines, max(0, start - 1))
            next_start = (int(rows[index + 1].get("line", 0) or 0)
                          if index + 1 < len(rows) else len(lines) + 1)
            end = max(end, min(bracket_end + 1, next_start - 1))
        if start <= line <= end:
            candidates.append((row, start, end))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[2] - item[1], -item[1]))


def _outlined_owner(outline: Mapping[str, object], line: int) -> str:
    """Resolve a missing degraded-parser parent from the existing type outline."""
    owners: list[tuple[int, str]] = []
    for row in outline.get("types", ()) or ():
        if not isinstance(row, Mapping):
            continue
        start = int(row.get("line", 0) or 0)
        if start > line:
            continue
        if row.get("name"):
            owners.append((start, str(row["name"])))
    return max(owners, default=(0, "module"))[1]


def _candidate_symbol(path: str, source: str, line: int) -> tuple[str, str, int, int] | None:
    """Return ``(name, owner, start, end)`` for the executable symbol at ``line``.

    The source-window owner is deliberately resolved through the same outline
    authorities as the enclosing label.  This keeps the candidate identity
    language-generic without introducing a second parser for any language.
    """
    if path.casefold().endswith(".py"):
        rows = [row for row in symbol_outline(source)
                if int(row["start_line"]) <= line <= int(row["end_line"])]
        if not rows:
            return None
        row = min(rows, key=lambda item: (
            int(item["end_line"]) - int(item["start_line"]),
            -int(item["start_line"]),
        ))
        qualified = str(row.get("qualified_name") or row.get("name") or "")
        name = str(row.get("name") or qualified.rsplit(".", 1)[-1])
        owner = qualified.rsplit(".", 1)[0] if "." in qualified else "module"
        return name, owner, int(row["start_line"]), int(row["end_line"])

    try:
        outline = _outline_for(path, source)
    except (ImportError, OSError, TypeError, ValueError):
        return None
    outlined = _outlined_function_at_line(source, outline, line)
    if outlined is None:
        return None
    row, start, end = outlined
    name = str(row.get("name") or "")
    if not name:
        return None
    owner = str(row.get("parent") or _outlined_owner(outline, start))
    return name, owner, start, end


def _outline_enclosing(path: str, source: str, line: int, lines: Sequence[str]) -> str | None:
    outline = _outline_for(path, source)
    functions = [row for row in outline.get("functions", ()) or ()
                 if isinstance(row, Mapping)
                 and int(row.get("line", 0) or 0) <= line <= int(row.get("end_line", 0) or 0)]
    if functions:
        row = max(functions, key=lambda item: (int(item["line"]), -int(item["end_line"])))
        name = ".".join(str(part) for part in (row.get("parent"), row.get("name")) if part)
        return f"inside `{name}` (lines {row['line']}-{row['end_line']})"
    types = [row for row in outline.get("types", ()) or ()
             if isinstance(row, Mapping)
             and int(row.get("line", 0) or 0) <= line <= int(row.get("end_line", 0) or 0)]
    if types:
        row = max(types, key=lambda item: int(item["line"]))
        return f"inside `{row.get('name')}` (lines {row['line']}-{row['end_line']})"
    for key in ("module_statements", "imports"):
        for row in outline.get(key, ()) or ():
            if (isinstance(row, Mapping)
                    and int(row.get("line", 0) or 0) <= line <= int(row.get("end_line", 0) or 0)):
                return _module_level_label(lines, int(row["line"]), int(row["end_line"]))
    return None


def enclosing_label(path: str, source: str, line: int) -> str:
    """Host label for one source line: enclosing symbol span, else module-level statement."""
    lines = source.splitlines()
    label = None
    if path.endswith(".py"):
        label = _python_enclosing(source, line, lines)
    else:
        try:
            label = _outline_enclosing(path, source, line, lines)
        except (ImportError, OSError, TypeError, ValueError):
            label = None
    if label is not None:
        return label
    start, end = _bracket_span(lines, max(0, line - 1))
    return _module_level_label(lines, start + 1, end + 1)


@profiled_function_span("request_source_literal_site_inventory")
def literal_site_inventory(root: Path, editable_paths: Sequence[str],
                           tokens: Sequence[str], *, origins: Mapping[str, str] | None = None,
                           max_sites: int = _MAX_SITES) -> list[Site]:
    """Every lexical match, typed as executable code, comment, or docstring."""
    origins = origins or {}
    patterns = [(token, _literal_pattern(token, origins.get(token, ""))) for token in tokens]
    sites: list[Site] = []
    for path in editable_paths:
        source = _read_product_file(Path(root), path)
        if source is None:
            continue
        lines = source.splitlines()
        docstring_lines = _python_docstring_lines(source) if path.casefold().endswith(".py") else frozenset()
        revision = _digest(source)
        language = _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.casefold(), "unknown")
        for index, line in enumerate(lines, 1):
            role = _site_role(path, source, index, docstring_lines)
            for token, pattern in patterns:
                if not pattern.search(line):
                    continue
                sites.append(Site(
                    path, index, line.rstrip(), enclosing_label(path, source, index), token,
                    role, language, revision,
                    *(_candidate_symbol(path, source, index) or ("", "", 0, 0)),
                ))
                if len(sites) >= max_sites:
                    return sites
    return sites


def changed_lines(parent: str, current: str) -> frozenset[int]:
    """1-based current-file lines an edit changed relative to the parent revision."""
    matcher = difflib.SequenceMatcher(None, parent.splitlines(), current.splitlines(),
                                      autojunk=False)
    changed: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed.update(range(j1 + 1, j2 + 1))
    return frozenset(changed)


def _parent_sources(runtime: object) -> dict[str, str]:
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    files = row.get("files") if isinstance(row, Mapping) else None
    result: dict[str, str] = {}
    if isinstance(files, Mapping):
        for path, payload in files.items():
            if isinstance(payload, Mapping) and isinstance(payload.get("parent_src"), str):
                result[str(path)] = payload["parent_src"]
    return result


def _revision12(root: Path, paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8") + b"\0")
        try:
            digest.update((Path(root) / path).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    # Keep the historical helper name for callers, but expose the complete
    # revision. Truncation would make entry/import facts collision-prone and
    # could leave a dependency-backed cache apparently current.
    return "sha256:" + digest.hexdigest()


def render_literal_inventory(sites: Sequence[Site], tokens: Sequence[str], *,
                             origins: Mapping[str, str], revision: str,
                             covered: int) -> str:
    lines = [f"{INVENTORY_HEADER} | host grep over declared product files at revision {revision}]"]
    for token in tokens:
        if not any(site.token == token for site in sites):
            continue
        original = origins.get(token)
        lines.append(f"query: {token}" + (f"   (normalized from the ticket literal \"{original}\")"
                                         if original else ""))
    executable = sum(site.role == "executable" for site in sites)
    comments = sum(site.role == "comment" for site in sites)
    docstrings = sum(site.role == "docstring" for site in sites)
    lines.append(f"sites={len(sites)}   executable_sites={executable}   comments={comments}   docstrings={docstrings}   "
                 f"covered_by_current_edits={covered}   "
                 f"textual_edit_coverage={covered}   resolved_use_definition=UNKNOWN   "
                 "observed_behavior=UNKNOWN")
    many = len({site.token for site in sites}) > 1
    symbol_candidates: dict[str, set[tuple[str, str, int, int]]] = {}
    for site in sites:
        if site.role != "executable" or not site.symbol_name:
            continue
        symbol_candidates.setdefault(site.symbol_name, set()).add((
            site.path, site.symbol_owner, site.symbol_start_line, site.symbol_end_line,
        ))
    colliding_names = frozenset(
        name for name, candidates in symbol_candidates.items() if len(candidates) > 1
    )
    for site in sites[:_MAX_RENDERED_SITES]:
        text = site.text if len(site.text) <= 100 else site.text[:97] + "..."
        owner_label = ""
        if site.role == "executable" and site.symbol_name in colliding_names:
            owner_label = (f"   owner={site.path}::{site.symbol_owner}::"
                           f"{site.symbol_name}")
        lines.append(f"- {site.path}:{site.line}   `{text}`   {site.enclosing}   role={site.role}"
                     f"   language={site.language}   revision={site.revision}"
                     + (f"   [query {site.token}]" if many else "")
                     + owner_label)
    if len(sites) > _MAX_RENDERED_SITES:
        lines.append(f"- ... {len(sites) - _MAX_RENDERED_SITES} more site(s) not listed")
    lines.append("Line numbers are host-counted at this revision. covered_by_current_edits is a "
                 "textual-edit count only; it does not establish resolved use/definition or "
                 "observed behavior. Lexical, delivered-span, textual-edit, resolved-use, and "
                 "observed-behavior coverage are separate facts.")
    return "\n".join(lines)


def literal_inventory_block(runtime: object) -> tuple[str | None, dict[str, object]]:
    """(rendered block or None, journal summary).  Rendered when N >= 2 or a site is module level."""
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None:
        return None, {}
    ticket = _ticket_text(runtime)
    tokens = literal_tokens(ticket, exclude=(entry_symbol(runtime),))
    paths = editable_product_paths(runtime)
    if not tokens or not paths:
        return None, {}
    origins = _token_origins(ticket)
    sites = literal_site_inventory(Path(root), paths, tokens, origins=origins)
    parents = _parent_sources(runtime)
    covered = 0
    changed_by_path: dict[str, frozenset[int]] = {}
    for site in sites:
        if site.path not in changed_by_path:
            current = _read_product_file(Path(root), site.path)
            parent = parents.get(site.path)
            changed_by_path[site.path] = (changed_lines(parent, current)
                                          if parent is not None and current is not None
                                          else frozenset())
        if site.role == "executable":
            covered += site.line in changed_by_path[site.path]
    site_paths = sorted({site.path for site in sites})
    current_sources = {path: _read_product_file(Path(root), path) for path in site_paths}
    current_revisions = {
        path: source_revision(source) for path, source in current_sources.items()
        if isinstance(source, str)
    }
    summary = {
        "tokens": list(tokens),
        "sites": len(sites),
        "lexical_sites": len(sites),
        "executable_sites": sum(site.role == "executable" for site in sites),
        "comment_sites": sum(site.role == "comment" for site in sites),
        "docstring_sites": sum(site.role == "docstring" for site in sites),
        "covered": covered,
        "textual_edit_coverage": covered,
        "delivered_spans": 0,
        "resolved_use_definition": "UNKNOWN",
        "observed_behavior": "UNKNOWN",
        "paths": site_paths,
        "source_revisions": current_revisions,
        "parent_revisions": {
            path: source_revision(parent) for path, parent in parents.items()
            if path in {site.path for site in sites}
        },
        "current_parent_identity": {
            path: {
                "current_revision": current_revisions.get(path),
                "parent_revision": source_revision(parents[path]) if path in parents else None,
                "status": ("UNKNOWN" if path not in parents or path not in current_revisions else
                           "UNCHANGED" if current_revisions[path] == source_revision(parents[path])
                           else "CHANGED"),
            }
            for path in site_paths
        },
    }
    module_level = any(site.enclosing.startswith("module level") for site in sites)
    if len(sites) < 2 and not module_level:
        return None, summary
    revision = _revision12(Path(root), paths)
    return render_literal_inventory(sites, tokens, origins=origins, revision=revision,
                                    covered=covered), summary


# ----------------------------------------------------------------- entry symbol verification (P3)

def _declared_definitions(root: Path, paths: Sequence[str], symbol: str) -> list[tuple[str, int]]:
    short = symbol.rsplit(".", 1)[-1]
    found: list[tuple[str, int]] = []
    for path in paths:
        source = _read_product_file(root, path)
        if source is None:
            continue
        if path.endswith(".py"):
            located = locate_symbol(source, symbol) or locate_symbol(source, short)
            if located is not None:
                found.append((path, located[0]))
            continue
        try:
            outline = _outline_for(path, source)
        except (ImportError, OSError, TypeError, ValueError):
            continue
        for row in [*(outline.get("functions", ()) or ()), *(outline.get("types", ()) or ())]:
            if isinstance(row, Mapping) and row.get("name") == short:
                found.append((path, int(row.get("line", 0) or 0)))
                break
    return found


def _declared_references(root: Path, paths: Sequence[str], symbol: str) -> int:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol.rsplit('.', 1)[-1])}(?![A-Za-z0-9_])")
    count = 0
    for path in paths:
        source = _read_product_file(root, path)
        if source is not None:
            count += sum(1 for line in source.splitlines() if pattern.search(line))
    return count


ENTRY_ABSENT_STATUSES = frozenset({"NOT_FOUND", "REFERENCED_ONLY"})


def entry_marked_new(runtime: object) -> bool:
    """Q1: the task row declares its entry as a NEW symbol (``kind: "new_symbol"``)."""
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    if not isinstance(row, Mapping):
        return False
    entry = row.get("entry") if isinstance(row.get("entry"), Mapping) else {}
    return "new_symbol" in {str(row.get("kind") or ""), str(row.get("entry_kind") or ""),
                            str(entry.get("kind") or "")}


def public_test_paths(runtime: object) -> tuple[str, ...]:
    """Q1: declared public test files (selectors stripped), existing, no evaluator scope."""
    root = getattr(getattr(runtime, "owner", None), "root", None)
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    if root is None or not isinstance(row, Mapping):
        return ()
    values: list[object] = []
    for key in ("public_check_paths", "public_tests", "test_files", "files", "editable_paths"):
        value = row.get(key)
        values.extend(value.keys() if isinstance(value, Mapping) else
                      value if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                      else ())
    base = Path(root).resolve()
    result: list[str] = []
    for value in values:
        relative = str(value or "").split("::", 1)[0].strip().replace("\\", "/").removeprefix("./")
        if (not relative or relative.startswith("/") or not _TEST_PATH.search(relative)
                or any(part.casefold() in _FORBIDDEN_PARTS for part in relative.split("/"))):
            continue
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        if candidate.is_file():
            result.append(relative)
    return tuple(dict.fromkeys(result))[:_MAX_EDITABLE_FILES]


def entry_verification(runtime: object) -> dict[str, object] | None:
    """Host ``where_defined`` of the task row's entry symbol (declared files first).

    Q1: the declared public test files are scanned too.  A symbol with 0
    definitions that the tests reference is ``REFERENCED_ONLY`` (not
    ``NOT_FOUND``); a row marked ``kind: "new_symbol"`` records ``new_symbol``.
    """
    symbol = entry_symbol(runtime)
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if not symbol or root is None:
        return None
    base = Path(root).resolve()
    paths = editable_product_paths(runtime)
    revision = _revision12(base, paths)
    definitions = _declared_definitions(base, paths, symbol)
    if definitions:
        return {"symbol": symbol, "status": "FOUND", "definitions": definitions,
                "revision": revision, "paths": list(paths)}
    references = _declared_references(base, paths, symbol)
    tests = public_test_paths(runtime)
    test_definitions = _declared_definitions(base, tests, symbol) if tests else []
    test_references = _declared_references(base, tests, symbol) if tests else 0
    from kernel_repo_intel import host_where_defined
    payload = host_where_defined(base, symbol)
    status = str(payload.get("status", "UNAVAILABLE"))
    coverage = payload.get("coverage", {}) if isinstance(payload.get("coverage"), Mapping) else {}
    result: dict[str, object] = {
        "symbol": symbol, "revision": revision, "paths": list(paths),
        "references": references, "repo_status": status,
        "coverage": coverage,
        "suggestions": payload.get("suggestions", []),
        "test_paths": list(tests), "test_references": test_references,
        "new_symbol": entry_marked_new(runtime),
    }
    definition = payload.get("definition") if isinstance(payload.get("definition"), Mapping) else None
    if status == "FOUND" and definition is not None:
        result.update({"status": "FOUND_OUTSIDE_DECLARED", "definition": dict(definition)})
    elif test_definitions:
        path, line = test_definitions[0]
        result.update({"status": "FOUND_OUTSIDE_DECLARED",
                       "definition": {"file": path, "lineno": line}})
    elif test_references:
        result["status"] = "REFERENCED_ONLY"
    elif status in {"UNAVAILABLE", "PARTIAL", "UNKNOWN", "STALE"}:
        result["status"] = status
    else:
        result["status"] = "NOT_FOUND"
    from kernel_fivefix_control import NegativeLookupLedger, classify_negative_lookup
    coverage_complete = bool(paths) and (
        coverage.get("complete") is True
        or (type(coverage.get("indexed")) is int and coverage.get("indexed", 0) >= len(paths))
    )
    negative_lookup = classify_negative_lookup(
        status=result["status"],
        complete=coverage_complete,
        query=symbol,
        scope="|".join(paths) or "declared_product_scope",
        product_revision=revision,
    )
    ledger = getattr(runtime, "_fivefix_negative_lookup_ledger", None)
    if not isinstance(ledger, NegativeLookupLedger):
        ledger = NegativeLookupLedger()
        try:
            setattr(runtime, "_fivefix_negative_lookup_ledger", ledger)
        except (AttributeError, TypeError):
            pass
    negative_lookup["new_fact"] = ledger.record(negative_lookup)
    result["negative_lookup"] = negative_lookup
    return result


def render_entry_verification(
        result: Mapping[str, object] | None, *,
        workspace_root: str | Path | None = None,
        revision: str | None = None, journal: object = None) -> str | None:
    if _workspace_truth.enabled():
        return _workspace_truth.render_entry_fact(
            result, workspace_root, revision=revision, journal=journal,
            header_prefix=ENTRY_HEADER,
        )
    return _workspace_truth.render_legacy_entry_verification(
        result, header_prefix=ENTRY_HEADER,
    )


# ----------------------------------------------------------------- module head + import fact (P3)

def module_head_end(source: str, path: str = "") -> int:
    """Last line of the module head: the line before the first top-level definition."""
    lines = source.splitlines()
    first_def = _first_definition_line(source, path)
    end = (first_def - 1) if first_def else len(lines)
    return max(0, min(end, _MODULE_HEAD_MAX_LINES, len(lines)))


def _first_definition_line(source: str, path: str) -> int | None:
    if path.endswith(".py") or not path:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return None
        starts = [min([node.lineno, *(item.lineno for item in node.decorator_list)])
                  for node in tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return min(starts) if starts else None
    try:
        outline = _outline_for(path, source)
    except (ImportError, OSError, TypeError, ValueError):
        return None
    starts = [int(row.get("line", 0) or 0)
              for row in [*(outline.get("functions", ()) or ()), *(outline.get("types", ()) or ())]
              if isinstance(row, Mapping) and not row.get("parent")]
    starts = [line for line in starts if line > 0]
    return min(starts) if starts else None


def _construction_names(ticket: str) -> tuple[str, ...]:
    names: list[str] = [match.group(1) for match in _CALL_NAME.finditer(ticket)]
    for match in _CODE_IDENT.finditer(ticket):
        value = match.group(1)
        if _code_identifier(value):
            names.extend((value, value.rsplit(".", 1)[-1]))
    names.extend(word for word in _WORD.findall(ticket) if word.casefold() not in _WORD_STOPWORDS)
    return tuple(dict.fromkeys(
        name for name in names
        if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*", name)
    ))[:30]


@functools.lru_cache(maxsize=32)
def _import_fact_cached(root: str, path: str, digest: str,
                        names: tuple[str, ...], dependency_revision: str = "") -> Mapping[str, object] | None:
    del digest, dependency_revision  # cache keys bind target and dependency revisions
    from kernel_repo_intel import import_construction_fact
    return import_construction_fact(Path(root), path, names)


def module_head_blocks(runtime: object,
                       entry: Mapping[str, object] | None = None) -> list[str]:
    """``[MODULE HEAD]`` (+ ``[IMPORT GRAPH FACT]``) for the target when P3 triggers."""
    root = getattr(getattr(runtime, "owner", None), "root", None)
    path = _path_from_runtime(runtime)
    if root is None or not path:
        return []
    blocks: list[str] = []
    if owner_coordinates_mode() == "on":
        coordinates = owner_coordinates_for_runtime(runtime, path)
        if coordinates and coordinates.get("owners_listed"):
            blocks.append(owner_coordinate_block(coordinates))
    ticket = _ticket_text(runtime)
    absent = bool(entry) and entry.get("status") in {"NOT_FOUND", "REFERENCED_ONLY",
                                                     "FOUND_OUTSIDE_DECLARED"}
    if not absent and not _MODULE_HEAD_TRIGGER.search(ticket):
        return blocks
    source = _read_product_file(Path(root), path)
    if source is None:
        return blocks
    fact = None
    generic_fact = None
    if path.endswith(".py"):
        names = _construction_names(ticket)
        if names:
            plan = closure_plan_for_runtime(runtime)
            connected = tuple(str(item) for item in (plan or {}).get("paths", (path,)))
            dependency_revision = _revision12(Path(root), connected)
            fact = _import_fact_cached(
                str(Path(root).resolve()), path, _digest(source), names, dependency_revision,
            )
            generic_fact = initialization_fact(Path(root), path, names, ticket=ticket)
    else:
        names = _construction_names(ticket)
        if names:
            generic_fact = initialization_fact(Path(root), path, names, ticket=ticket)
    head = render_module_head(path, source, fact)
    if head:
        blocks.append(head)
    if fact:
        blocks.append(render_import_fact(fact, revision=_revision12(Path(root), (path,))))
    if generic_fact is not None:
        blocks.append(render_initialization_fact(generic_fact))
    return blocks


def render_module_head(path: str, source: str,
                       fact: Mapping[str, object] | None = None) -> str | None:
    lines = source.splitlines()
    end = module_head_end(source, path)
    if end < 1:
        return None
    try:
        outline = _outline_for(path, source)
    except (ImportError, OSError, TypeError, ValueError):
        outline = {}
    rows: list[tuple[int, int, str]] = []
    for key in ("imports", "module_statements"):
        for row in outline.get(key, ()) or ():
            if not isinstance(row, Mapping):
                continue
            start = int(row.get("line", 0) or 0)
            stop = int(row.get("end_line", start) or start)
            if 1 <= start <= end:
                rows.append((start, stop, lines[start - 1].strip() if start <= len(lines) else ""))
    rows.sort()
    fact_line = int(fact.get("line", 0) or 0) if fact else 0
    rendered = [f"{MODULE_HEAD_HEADER} | {path} lines 1-{end}, host outline with line numbers]"]
    for start, stop, text in rows[:_MODULE_HEAD_ROWS]:
        span = f"{start}-{stop}" if stop > start else f"{start}"
        text = text if len(text) <= 100 else text[:97] + "..."
        marker = (f"   <- first statement that transitively constructs `{fact.get('name')}` "
                  "(see IMPORT GRAPH FACT)" if fact and start == fact_line else "")
        rendered.append(f"  {span:<9} `{text}`{marker}")
    if len(rows) > _MODULE_HEAD_ROWS:
        rendered.append(f"  ... {len(rows) - _MODULE_HEAD_ROWS} more module-level statement(s)")
    first_def = _first_definition_line(source, path)
    if first_def and first_def <= len(lines):
        rendered.append(f"  {first_def:<9} first definition: `{lines[first_def - 1].strip()[:90]}`")
    return "\n".join(rendered)


def render_import_fact(fact: Mapping[str, object], *, revision: str) -> str:
    constructor = fact.get("constructor") if isinstance(fact.get("constructor"), Mapping) else {}
    chain = " -> ".join(str(item) for item in fact.get("chain", ()) or ())
    return (f"{IMPORT_FACT_HEADER} | host, Python import closure at revision {revision}]\n"
            f"  First statement in {fact.get('path')} that transitively constructs "
            f"`{fact.get('name')}`: line {fact.get('line')} (`{fact.get('statement')}`)\n"
            f"  constructor site: {constructor.get('path')}:{constructor.get('line')} "
             f"`{constructor.get('text')}`" + (f"   (import chain: {chain})" if chain else ""))


def render_initialization_fact(fact: Mapping[str, object]) -> str:
    """Render language-generic ownership without claiming behavior was observed."""
    return ("[INITIALIZATION / CONSTRUCTION FACT]\n"
            f"language={fact.get('language', 'unknown')} status={fact.get('status', 'UNKNOWN')} "
            f"source={fact.get('path', 'UNKNOWN')} "
            f"source_revision={fact.get('source_revision', 'UNKNOWN')} "
            f"dependency_revision={fact.get('dependency_revision', 'UNKNOWN')}\n"
            f"adapter={fact.get('adapter', 'UNKNOWN')} "
            f"line={fact.get('line', 'UNKNOWN')} reason={fact.get('reason', 'outline-backed source fact')}")


def site_fact_blocks(runtime: object, entry: Mapping[str, object] | None
                     ) -> tuple[list[str], dict[str, object]]:
    """All P2/P3 sidecars for one provider request (entry verification is supplied cached)."""
    blocks: list[str] = []
    owner = getattr(runtime, "owner", None)
    rendered_entry = render_entry_verification(
        entry, workspace_root=getattr(owner, "root", None),
        journal=getattr(runtime, "journal", None),
    )
    if rendered_entry:
        blocks.append(rendered_entry)
    module_blocks = module_head_blocks(runtime, entry)
    blocks.extend(module_blocks)
    inventory, summary = literal_inventory_block(runtime)
    if inventory:
        blocks.append(inventory)
    if owner_coordinates_mode() == "on":
        summary["owners_listed"] = sum(
            1 for block in module_blocks for line in block.splitlines()
            if line.startswith("  owner=")
        )
    return blocks, summary


__all__ = [
    "ENTRY_ABSENT_STATUSES", "ENTRY_HEADER", "IMPORT_FACT_HEADER", "OWNER_COORDS_HEADER",
    "entry_marked_new",
    "public_test_paths", "INVENTORY_HEADER", "MODULE_HEAD_HEADER",
    "SCHEMA", "SITE_FACTS_ENV", "Site", "WINDOW_MARKER", "bound_runtime_sources",
    "bound_source_message", "changed_lines", "editable_product_paths", "enclosing_label",
    "entry_symbol", "entry_verification", "is_source_window", "literal_inventory_block",
    "literal_site_inventory", "literal_tokens", "module_head_blocks", "module_head_end",
    "parse_model_source_window", "parse_window", "project_model_source_messages",
    "project_source_absence", "project_source_window", "render_entry_verification",
    "render_initialization_fact", "render_literal_inventory",
    "owner_coordinate_block", "owner_coordinate_data", "owner_coordinate_is_current",
    "owner_coordinates_for_runtime", "owner_coordinates_mode", "site_fact_blocks",
    "site_facts_mode", "partial_source_paths_for_deferred_delivery",
    "verify_final_source_delivery", "verify_window",
    "verify_model_source_window", "verify_window_projection",
    "settle_blank_source_windows", "record_deferred_source_paths",
]


# Flag OFF (STATEFRAME_TRUTHFUL_READ_PANELS default): the r2FO functions verbatim.
def _normalized_deferred_paths_r2fo(paths: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in paths if str(path).strip()
    ))


def _render_deferred_source_notice_r2fo(paths: Sequence[object]) -> str:
    """Render the one host-authored notice that proves source deferral."""
    normalized = _normalized_deferred_paths_r2fo(paths)
    if not normalized:
        raise ValueError("deferred source notice requires at least one path")
    return _workspace_truth.render_legacy_deferred_notice(
        DEFERRED_SOURCE_NOTICE_HEADER, normalized,
    )


def _upsert_deferred_source_notice_r2fo(
        messages: list[dict[str, object]], paths: Sequence[object],
) -> None:
    normalized = _normalized_deferred_paths_r2fo(paths)
    if not normalized:
        return
    content = _render_deferred_source_notice_r2fo(normalized)
    last_exchange_index = max(
        (index for index, message in enumerate(messages)
         if isinstance(message, Mapping)
         and message.get("role") in ("assistant", "tool")),
        default=-1,
    )
    if last_exchange_index >= 0:
        current_tail_start = last_exchange_index + 1
    else:
        task_index = next(
            (index for index, message in enumerate(messages)
             if isinstance(message, Mapping)
             and message.get("role") == "user"
             and message.get("_request_budget_section") == "task"),
            None,
        )
        if task_index is None:
            task_index = next(
                (index for index, message in enumerate(messages)
                 if isinstance(message, Mapping) and message.get("role") == "user"),
                None,
            )
        current_tail_start = task_index + 1 if task_index is not None else len(messages)

    # Budget/diet retries can revise the deferred path set several times before
    # dispatch. Replace only notices staged after the latest conversation turn;
    # already-sent notices stay prefix-stable in the transcript.
    messages[:] = [
        message for index, message in enumerate(messages)
        if not (
            index >= current_tail_start
            and isinstance(message, Mapping)
            and message.get("role") == "user"
            and message.get("_request_budget_section") == "stage_evidence"
            and str(message.get("content", "")).startswith(
                DEFERRED_SOURCE_NOTICE_HEADER,
            )
        )
    ]
    notice = {
        "role": "user",
        "content": content,
        "_request_budget_section": "stage_evidence",
    }
    if any(
            isinstance(message, Mapping)
            and message.get("role") == "user"
            and message.get("content") == content
            for message in messages):
        return
    # Existing transcript turns are prefix-stable; differing current notices
    # are appended at the tail instead of rewriting an earlier user message.
    messages.append(notice)




def render_deferred_source_notice(
        paths: Sequence[object], *,
        delivered_source_ranges: Sequence[tuple[str, str, int, int]] = (),
        workspace_root: str | Path | None = None,
        revision: str | None = None, messages: Sequence[object] = (),
        journal: object = None,
) -> str:
    if _workspace_truth.enabled():
        return _workspace_truth.render_deferred_source_facts(
            DEFERRED_SOURCE_NOTICE_HEADER, _normalized_deferred_paths(paths),
            workspace_root, messages=messages, revision=revision, journal=journal,
        )
    from kernel_request_diet import truthful_read_panels_enabled
    if not truthful_read_panels_enabled():
        return _render_deferred_source_notice_r2fo(paths)
    return _render_truthful_deferred_source_notice(
        paths, delivered_source_ranges=delivered_source_ranges)


def _upsert_live_workspace_deferred_notice(messages, paths, **kwargs):
    normalized = _normalized_deferred_paths(paths)
    workspace_root = kwargs.get("workspace_root")
    revision = kwargs.get("revision")
    journal = kwargs.get("journal")
    messages[:] = [
        message for message in messages
        if not (
            isinstance(message, Mapping)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].startswith(DEFERRED_SOURCE_NOTICE_HEADER)
        )
    ]
    if not normalized:
        return
    content = _workspace_truth.render_deferred_source_facts(
        DEFERRED_SOURCE_NOTICE_HEADER, normalized, workspace_root,
        messages=messages, revision=revision, journal=journal,
    )
    messages.append({
        "role": "user", "content": content,
        "_request_budget_section": "stage_evidence",
    })


def upsert_deferred_source_notice(messages, paths, *args, **kwargs):
    if _workspace_truth.enabled():
        return _upsert_live_workspace_deferred_notice(messages, paths, **kwargs)
    from kernel_request_diet import truthful_read_panels_enabled
    if not truthful_read_panels_enabled():
        return _upsert_deferred_source_notice_r2fo(messages, paths, *args, **kwargs)
    return _upsert_truthful_deferred_source_notice(messages, paths, *args, **kwargs)
