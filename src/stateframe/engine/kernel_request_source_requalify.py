"""Exact live-source repair for a required path missing from the final wire."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _path_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.replace("\\", "/")
    parts = path.split("/")
    if (not path or path.startswith("/")
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in parts)
            or "\x00" in path or "\n" in path or "\r" in path):
        return None
    return path


def _required_span(
        units: Sequence[object], path: str, source: str,
) -> tuple[int, int] | None:
    source_lines = source.splitlines(keepends=True)
    if not source_lines:
        return None
    bounds: list[tuple[int, int]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        if _path_value(unit.get("path")) != path:
            continue
        start = unit.get("start_line")
        end = unit.get("end_line")
        if start is None and end is None:
            return 1, len(source_lines)
        if (type(start) is not int or type(end) is not int
                or start < 1 or end < start or end > len(source_lines)):
            return None
        bounds.append((start, end))
    if not bounds:
        return None
    return min(start for start, _end in bounds), max(end for _start, end in bounds)


def _render_live_source_block(
        root: Path, path: str, revision: str, source: str,
        start: int, end: int,
) -> str | None:
    from kernel_f4_source_ranges import exact_current_source_span
    from kernel_request_source_window import source_revision

    if source_revision(source) != revision:
        return None
    exact = exact_current_source_span(root, path, revision, start, end)
    source_lines = source.splitlines(keepends=True)
    if (exact is None or exact != "".join(source_lines[start - 1:end])
            or start < 1 or end > len(source_lines)):
        return None
    rendered_source = exact if exact.endswith(("\n", "\r")) else exact + "\n"
    return (
        f"[Current source: {path}]\n{rendered_source}"
        f"Coverage: lines {start}-{end}, columns 0-0; "
        "current bytes are delivered for this range."
    )


def _has_source_block(messages: Sequence[object], path: str) -> bool:
    markers = (
        f"[Current source: {path}]",
        f"[Working source: {path}]",
        f"[Retained source: {path}]",
        f"[Observed source: {path}]",
        f"[read page] path={path} ",
        f"[read reuse] path={path} ",
        f"CURRENT_SOURCE {path} ",
        f"Observed source {path}",
    )
    return any(
        isinstance(message, Mapping)
        and isinstance(message.get("content"), str)
        and any(marker in message["content"] for marker in markers)
        for message in messages
    )


def _append_user_source_message(messages: list[object], content: str) -> bool:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (isinstance(message, Mapping) and message.get("role") == "user"
                and isinstance(message.get("content"), str)):
            messages.insert(index + 1, {"role": "user", "content": content})
            return True
    return False


def _read_tool_available(tools: object) -> bool:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return False
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else (
            tool.get("name") if isinstance(tool, Mapping) else None
        )
        if name == "read":
            return True
    return False


def _drop_source_requirement(plan: Mapping[str, object], path: str) -> dict[str, object] | None:
    dropped = copy.deepcopy(dict(plan))
    raw_units = dropped.get("units")
    if (not isinstance(raw_units, Sequence)
            or isinstance(raw_units, (str, bytes, bytearray))):
        return None
    remaining_units = [
        unit for unit in raw_units
        if not isinstance(unit, Mapping) or _path_value(unit.get("path")) != path
    ]
    if len(remaining_units) == len(raw_units):
        return None
    dropped["units"] = remaining_units

    raw_paths = dropped.get("paths")
    if raw_paths is not None:
        if (not isinstance(raw_paths, Sequence)
                or isinstance(raw_paths, (str, bytes, bytearray))):
            return None
        dropped["paths"] = [
            value for value in raw_paths
            if _path_value(value) != path
        ]

    raw_revisions = dropped.get("revisions")
    if raw_revisions is not None:
        if not isinstance(raw_revisions, Mapping):
            return None
        revisions = dict(raw_revisions)
        revisions.pop(path, None)
        dropped["revisions"] = revisions

    raw_sources = dropped.get("source_contents")
    if isinstance(raw_sources, Mapping):
        sources = dict(raw_sources)
        sources.pop(path, None)
        dropped["source_contents"] = sources

    raw_deferred = dropped.get("deferred_paths", ())
    if isinstance(raw_deferred, Sequence) and not isinstance(
            raw_deferred, (str, bytes, bytearray)):
        dropped["deferred_paths"] = [
            value for value in raw_deferred if _path_value(value) != path
        ]

    if _path_value(dropped.get("target_path")) == path:
        next_path = next((
            _path_value(unit.get("path"))
            for unit in remaining_units if isinstance(unit, Mapping)
            and _path_value(unit.get("path")) is not None
        ), None)
        if next_path is None:
            dropped.pop("target_path", None)
        else:
            dropped["target_path"] = next_path

    raw_dropped = dropped.get("dropped_source_paths", ())
    prior = (
        [value for value in raw_dropped if isinstance(value, str) and value.strip()]
        if isinstance(raw_dropped, Sequence)
        and not isinstance(raw_dropped, (str, bytes, bytearray))
        else []
    )
    dropped["dropped_source_paths"] = list(dict.fromkeys((*prior, path)))
    return dropped


def attempt_missing_source_requalification(
        runtime: object, plan: Mapping[str, object],
        sources: Mapping[str, str] | None, body: Mapping[str, object], wire: bytes,
        candidates: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, object], bytes, dict[str, object], Mapping[str, object],
    list[tuple[str, str]],
] | None:
    """Attach missing selected source exactly, or drop it only when it exceeds the cap."""
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES
    from kernel_request_source_window import (
        source_delivery_satisfied, verify_final_source_delivery,
    )

    if len(wire) > int(HARD_MAX_REQUEST_BYTES):
        return None
    messages = body.get("messages")
    units = plan.get("units")
    revisions = plan.get("revisions")
    if (not isinstance(messages, list)
            or not isinstance(units, Sequence)
            or isinstance(units, (str, bytes, bytearray))
            or not isinstance(revisions, Mapping)
            or not candidates):
        return None

    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        return None
    try:
        root = Path(root_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    source_map = sources if isinstance(sources, Mapping) else {}
    from kernel_stateframe_framefit5 import (
        assemble_framefit5_request,
        has_framefit5_state,
    )
    if has_framefit5_state(messages):
        candidate_body = copy.deepcopy(dict(body))
        candidate_messages = candidate_body.get("messages")
        if not isinstance(candidate_messages, list):
            return None
        active_capture = getattr(runtime, "_active_capture", None)
        assembled_messages = assemble_framefit5_request(
            candidate_messages,
            candidate_messages,
            task_message=getattr(active_capture, "_framefit5_task_message", None),
            source_plan=plan,
            sources=source_map,
            frame_request_allocation=getattr(
                active_capture, "frame_request_allocation", None,
            ),
        )
        candidate_body["messages"] = assembled_messages
        candidate_wire = json.dumps(candidate_body).encode("utf-8")
        if len(candidate_wire) <= int(HARD_MAX_REQUEST_BYTES):
            assembled_plan = copy.deepcopy(dict(plan))
            assembled_delivery = verify_final_source_delivery(
                assembled_plan,
                assembled_messages,
                payload=candidate_wire,
                sources=source_map,
            )
            if source_delivery_satisfied(assembled_delivery):
                return (
                    candidate_body,
                    candidate_wire,
                    assembled_plan,
                    assembled_delivery,
                    [(str(row.get("path")), "framefit5_reassembled_source")
                     for row in candidates if isinstance(row.get("path"), str)],
                )
    required: list[tuple[str, str, str, int, int]] = []
    for candidate in candidates:
        path = _path_value(candidate.get("path"))
        revision = revisions.get(path) if path is not None else None
        source = source_map.get(path) if path is not None else None
        if (path is None or not isinstance(revision, str) or not revision
                or not isinstance(source, str) or _has_source_block(messages, path)):
            return None
        span = _required_span(units, path, source)
        if span is None:
            return None
        start, end = span
        required.append((path, revision, source, start, end))

    candidate_body = copy.deepcopy(dict(body))
    candidate_messages = candidate_body.get("messages")
    if not isinstance(candidate_messages, list):
        return None
    actions: list[tuple[str, str]] = []
    for path, revision, source, start, end in required:
        block = _render_live_source_block(root, path, revision, source, start, end)
        if block is None or not _append_user_source_message(candidate_messages, block):
            return None
        actions.append((path, "rendered_live_source"))

    candidate_wire = json.dumps(candidate_body).encode("utf-8")
    if len(candidate_wire) <= int(HARD_MAX_REQUEST_BYTES):
        qualified_plan = copy.deepcopy(dict(plan))
        qualified = verify_final_source_delivery(
            qualified_plan, candidate_messages, payload=candidate_wire,
            sources=source_map,
        )
        if source_delivery_satisfied(qualified):
            return candidate_body, candidate_wire, qualified_plan, qualified, actions
        return None

    tools = body.get("tools")
    if not _read_tool_available(tools):
        return None
    fallback_body = copy.deepcopy(dict(body))
    fallback_messages = fallback_body.get("messages")
    if not isinstance(fallback_messages, list):
        return None
    fallback_plan = copy.deepcopy(dict(plan))
    dropped_actions: list[tuple[str, str]] = []
    for path, _revision, _source, start, end in required:
        fallback_plan = _drop_source_requirement(fallback_plan, path)
        if fallback_plan is None:
            return None
        route = (
            f"read(path={json.dumps(path, ensure_ascii=False)}, "
            f"start_line={start}, end_line={end})"
        )
        fact = f"[Host fact] {path} is not attached; {route} loads it"
        if not _append_user_source_message(fallback_messages, fact):
            return None
        dropped_actions.append((path, "requirement_dropped_with_read_route"))

    fallback_wire = json.dumps(fallback_body).encode("utf-8")
    if len(fallback_wire) > int(HARD_MAX_REQUEST_BYTES):
        return None
    qualified = verify_final_source_delivery(
        fallback_plan, fallback_messages, payload=fallback_wire, sources=source_map,
    )
    if not source_delivery_satisfied(qualified):
        return None
    return fallback_body, fallback_wire, fallback_plan, qualified, dropped_actions
