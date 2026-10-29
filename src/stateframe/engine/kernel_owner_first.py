"""Opt-in owner-first request allocation and source-coordinate receipts."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence

ALLOC_ENV = "STATEFRAME_OWNER_FIRST_ALLOC"
COORDS_ENV = "STATEFRAME_OWNER_COORDS"
PRIORITY_ORDER = (
    "authorization_ticket",
    "admitted_action_schemas",
    "current_blocker_receipt_facts",
    "requested_owner_caller_source",
    "optional_module_import_context",
)


def switch_enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(name, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{name} must be on or off")
    return value == "on"


def allocation_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return switch_enabled(ALLOC_ENV, environ)


def coordinates_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return switch_enabled(COORDS_ENV, environ)


def startup_organ_wiring(emit: object) -> None:
    """Journal both opt-in switches when the runtime journal becomes available."""
    if not callable(emit):
        return
    for name, enabled in ((ALLOC_ENV, allocation_enabled()),
                          (COORDS_ENV, coordinates_enabled())):
        emit("organ_wiring_state", producer="kernel_owner_first", organ=name,
             state="on" if enabled else "off", boundary="startup", dispatch=False)


def _message_key(message: Mapping[str, object]) -> str:
    identity = "\0".join((str(message.get("role", "")),
                          str(message.get("tool_call_id", "")),
                          str(message.get("content", ""))))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _message_category(message: Mapping[str, object], requested_paths: set[str]) -> str:
    content = str(message.get("content") or "")
    section = str(message.get("_request_budget_section") or "").strip().lower()
    if (section in {"authorized_ticket", "ticket"}
            or content.lstrip().startswith(("[AUTHORIZED TICKET", "Task:", "Ticket:", "Brief:"))):
        return "authorization_ticket"
    if section in {"stage_evidence", "completion_contract", "failure_packet", "blocker", "receipt"} \
            or any(marker in content[:512].casefold() for marker in (
                "current blocker", "blocker receipt", "host limit", "current obligation",
                "missing check", "latest failure", "failure evidence:",
            )):
        return "current_blocker_receipt_facts"
    if content.startswith(("[MODULE HEAD", "[IMPORT GRAPH FACT",
                           "[INITIALIZATION / CONSTRUCTION FACT")):
        return "optional_module_import_context"
    try:
        from kernel_request_source_window import parse_window
        metadata = parse_window(content)
    except (ImportError, TypeError, ValueError):
        metadata = None
    if isinstance(metadata, Mapping):
        path = str(metadata.get("path", "")).replace("\\", "/")
        return ("requested_owner_caller_source" if path in requested_paths
                else "optional_module_import_context")
    match = re.match(r"^\[(?:Current|Working) source: ([^\]\n]+)\]", content)
    if match:
        path = match.group(1).replace("\\", "/")
        return ("requested_owner_caller_source" if path in requested_paths
                else "optional_module_import_context")
    return "other"


def build_priority_plan(body: Mapping[str, object], runtime: object | None) -> dict[str, object]:
    """Capture request priorities and revision-bound requested ranges once."""
    source_plan = getattr(runtime, "_request_source_closure", None)
    source_plan = source_plan if isinstance(source_plan, Mapping) else {}
    raw_revisions = source_plan.get("revisions", {})
    revisions = ({str(path).replace("\\", "/"): str(revision)
                  for path, revision in raw_revisions.items()}
                 if isinstance(raw_revisions, Mapping) else {})
    requested_paths: list[str] = []
    requested_ranges: list[dict[str, object]] = []
    try:
        from kernel_request_source_window import _path_from_runtime
        target = _path_from_runtime(runtime) if runtime is not None else None
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        target = None
    if target:
        requested_paths.append(str(target).replace("\\", "/"))
    units = source_plan.get("units", ())
    if isinstance(units, Sequence) and not isinstance(units, (str, bytes)):
        for raw in units:
            if not isinstance(raw, Mapping):
                continue
            path = str(raw.get("path", "")).replace("\\", "/")
            if not path:
                continue
            try:
                priority = int(raw.get("priority", 99))
            except (TypeError, ValueError):
                priority = 99
            role = str(raw.get("role", ""))
            if path == target or (priority <= 1 and role in {
                    "body", "statement", "target_reservation", "caller", "callsite"}):
                requested_paths.append(path)
                start, end = raw.get("start_line"), raw.get("end_line")
                revision = str(raw.get("revision") or revisions.get(path, ""))
                if (type(start) is int and type(end) is int and 1 <= start <= end
                        and revision and revision not in {"UNKNOWN", "UNVERIFIED"}):
                    requested_ranges.append({
                        "path": path, "revision": revision,
                        "start_line": start, "end_line": end,
                        "role": role,
                        "owner": str(raw.get("owner", raw.get("symbol_owner", ""))),
                    })
    requested_paths = list(dict.fromkeys(requested_paths))
    path_set = set(requested_paths)
    counts = {name: 0 for name in PRIORITY_ORDER}
    message_plan: list[dict[str, str]] = []
    protected_keys: set[str] = set()
    messages = body.get("messages", ())
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            category = _message_category(message, path_set)
            if category in counts:
                counts[category] += 1
            message_plan.append({"key": _message_key(message), "category": category})
            if category in {"authorization_ticket", "current_blocker_receipt_facts",
                            "requested_owner_caller_source"}:
                protected_keys.add(_message_key(message))
    tools = body.get("tools", ())
    admitted_actions: list[str] = []
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
            if isinstance(name, str) and name:
                admitted_actions.append(name)
    return {
        "schema": "stateframe-owner-first-allocation/1",
        "priority_order": list(PRIORITY_ORDER),
        "requested_paths": requested_paths,
        "requested_ranges": requested_ranges,
        "source_revisions": revisions,
        "message_plan": message_plan,
        "protected_message_keys": sorted(protected_keys),
        "message_counts": counts,
        "admitted_action_schemas": admitted_actions,
        "recall_inventory": [],
        "delivered_ranges": [],
        "dropped_ranges": [],
    }


def protected_priority_messages(body: Mapping[str, object], plan: Mapping[str, object]
                                ) -> tuple[Mapping[str, object], ...]:
    messages = body.get("messages", ())
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ()
    requested_paths = {str(path).replace("\\", "/")
                       for path in plan.get("requested_paths", ()) or ()}
    return tuple(message for message in messages if isinstance(message, Mapping)
                 and _message_category(message, requested_paths) in {
                     "authorization_ticket", "current_blocker_receipt_facts",
                     "requested_owner_caller_source",
                 })


def optional_recall_record(message: Mapping[str, object], plan: Mapping[str, object]
                           ) -> dict[str, object] | None:
    """Extract truthful path/range handles from an optional source fragment."""
    content = str(message.get("content") or "")
    path, revision = "", ""
    ranges: list[dict[str, int]] = []
    try:
        from kernel_request_source_window import parse_window
        metadata = parse_window(content)
    except (ImportError, TypeError, ValueError):
        metadata = None
    if isinstance(metadata, Mapping):
        path = str(metadata.get("path", "")).replace("\\", "/")
        revision = str(metadata.get("revision", ""))
        for row in metadata.get("ranges", ()) or ():
            if not isinstance(row, Mapping):
                continue
            start, end = row.get("start_line"), row.get("end_line")
            if type(start) is int and type(end) is int and 1 <= start <= end:
                ranges.append({"start_line": start, "end_line": end})
    else:
        head = re.search(r"^\[MODULE HEAD \| ([^\s]+) lines ", content, re.MULTILINE)
        imported = re.search(r"First statement in ([^\s]+) that", content)
        init = re.search(r"\bsource=([^\s]+).*?\bsource_revision=([^\s]+)", content, re.DOTALL)
        if head:
            path = head.group(1).replace("\\", "/")
            first = content.splitlines()[0] if content.splitlines() else ""
            span = re.search(r"lines (\d+)-(\d+)", first)
            if span:
                ranges.append({"start_line": int(span.group(1)), "end_line": int(span.group(2))})
        elif imported:
            path = imported.group(1).replace("\\", "/")
            found_revision = re.search(r"at revision ([^\]\s]+)", content)
            revision = found_revision.group(1) if found_revision else ""
        elif init:
            path, revision = init.group(1).replace("\\", "/"), init.group(2)
    if not path:
        return None
    if not revision:
        revisions = plan.get("source_revisions", {})
        revision = str(revisions.get(path, "")) if isinstance(revisions, Mapping) else ""
    if not revision or revision in {"UNKNOWN", "UNVERIFIED"}:
        return None
    return {"path": path, "revision": revision, "ranges": ranges}


def render_recall_inventory(records: Sequence[Mapping[str, object]]) -> str:
    unique: dict[tuple[str, str], list[dict[str, int]]] = {}
    for row in records:
        path, revision = str(row.get("path", "")), str(row.get("revision", ""))
        if not path or not revision:
            continue
        spans = unique.setdefault((path, revision), [])
        for raw in row.get("ranges", ()) or ():
            if not isinstance(raw, Mapping):
                continue
            start, end = raw.get("start_line"), raw.get("end_line")
            if type(start) is int and type(end) is int and start > 0 and end >= start:
                spans.append({"start_line": start, "end_line": end})
    lines = ["[REVISION-BOUND RECALL INVENTORY]"]
    for (path, revision), spans in sorted(unique.items()):
        spans = sorted({(row["start_line"], row["end_line"]) for row in spans})
        ranges = ",".join(f"{start}-{end}" for start, end in spans)
        location = f" lines={ranges}" if ranges else ""
        first_line = spans[0][0] if spans else 1
        quoted_path = json.dumps(path, ensure_ascii=False)
        lines.append(f"path={quoted_path} revision={revision}{location}; page with "
                     f"read(path={quoted_path}, start_line={first_line})")
    return "\n".join(lines)


def finalize_range_plan(plan: dict[str, object], messages: Sequence[object]) -> dict[str, object]:
    """Derive delivered and dropped spans from the final serialized message list."""
    from kernel_request_source_window import parse_model_source_window, parse_window

    delivered: list[dict[str, object]] = []
    inventory: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        if content.startswith("[REVISION-BOUND RECALL INVENTORY]"):
            for raw_path, revision in re.findall(
                    r'(?m)^path=("(?:[^"\\]|\\.)*") revision=([^\s;]+)', content):
                try:
                    path = json.loads(raw_path)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(path, str):
                    inventory.append({"path": path, "revision": revision})
        metadata = parse_window(content) or parse_model_source_window(content)
        if not isinstance(metadata, Mapping):
            continue
        path = str(metadata.get("path", "")).replace("\\", "/")
        revision = str(metadata.get("revision", ""))
        for row in metadata.get("ranges", ()) or ():
            if (isinstance(row, Mapping) and type(row.get("start_line")) is int
                    and type(row.get("end_line")) is int and path and revision):
                delivered.append({"path": path, "revision": revision,
                                  "start_line": row["start_line"], "end_line": row["end_line"]})

    candidates: list[dict[str, object]] = []
    def add_candidate(raw: Mapping[str, object]) -> None:
        path, revision = str(raw.get("path", "")), str(raw.get("revision", ""))
        start, end = raw.get("start_line"), raw.get("end_line")
        if (path and revision and type(start) is int and type(end) is int
                and start > 0 and end >= start):
            candidates.append({**dict(raw), "path": path, "revision": revision,
                               "start_line": start, "end_line": end})

    for raw in plan.get("dropped_ranges", ()) or ():
        if isinstance(raw, Mapping):
            add_candidate(raw)
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str) and content.startswith("[DROPPED SOURCE RANGES]"):
            for match in re.finditer(r"(?m)^- (.+):(\d+)-(\d+) revision=([^\s]+) reason=(.+)$", content):
                path, start, end, revision, reason = match.groups()
                add_candidate({"path": path, "revision": revision,
                               "start_line": int(start), "end_line": int(end), "reason": reason})
    for raw in plan.get("requested_ranges", ()) or ():
        if isinstance(raw, Mapping):
            add_candidate({**dict(raw), "reason": str(raw.get("reason") or "requested_range_not_delivered")})

    dropped: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for candidate in candidates:
        path, revision = str(candidate["path"]), str(candidate["revision"])
        start, end = int(candidate["start_line"]), int(candidate["end_line"])
        covered = sorted((int(row["start_line"]), int(row["end_line"])) for row in delivered
                         if row["path"] == path and row["revision"] == revision
                         and int(row["end_line"]) >= start and int(row["start_line"]) <= end)
        cursor = start
        segments: list[tuple[int, int]] = []
        for left, right in covered:
            if left > cursor:
                segments.append((cursor, min(end, left - 1)))
            cursor = max(cursor, right + 1)
            if cursor > end:
                break
        if cursor <= end:
            segments.append((cursor, end))
        reason = str(candidate.get("reason") or "source_range_not_delivered")
        for left, right in segments:
            if left > right:
                continue
            key = (path, revision, left, right, reason)
            if key not in seen:
                seen.add(key)
                dropped.append({**candidate, "path": path, "revision": revision,
                                "start_line": left, "end_line": right, "reason": reason})
    plan["delivered_ranges"] = delivered
    plan["dropped_ranges"] = dropped
    plan["recall_inventory"] = inventory
    _write_final_dropped_notice(messages, dropped)
    return allocation_summary(plan)


def _write_final_dropped_notice(messages: Sequence[object], dropped: Sequence[Mapping[str, object]]) -> None:
    if not isinstance(messages, list):
        return
    indices = [index for index, message in enumerate(messages)
               if isinstance(message, Mapping)
               and str(message.get("content", "")).startswith("[DROPPED SOURCE RANGES]")]
    if not dropped:
        for index in reversed(indices):
            del messages[index]
        return
    lines = ["[DROPPED SOURCE RANGES]",
             "The source text for these previously read ranges is absent from this request.",
             "Pinned host range anchors (path:start-end); re-read before relying on the text:"]
    for row in sorted(dropped, key=lambda item: (
            str(item.get("path", "")), int(item.get("start_line", 0)),
            int(item.get("end_line", 0)), str(item.get("revision", "")),
            str(item.get("reason", "")))):
        lines.append(f"- {row['path']}:{row['start_line']}-{row['end_line']} "
                     f"revision={row['revision']} reason={row.get('reason', 'source_range_not_delivered')}")
    notice = {"role": "user", "content": "\n".join(lines),
              "_request_budget_section": "stage_evidence"}
    if indices:
        messages[indices[0]] = notice
        for index in reversed(indices[1:]):
            del messages[index]
    else:
        messages.append(notice)


def allocation_summary(plan: Mapping[str, object]) -> dict[str, object]:
    counts = plan.get("message_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    return {
        "schema": plan.get("schema", "stateframe-owner-first-allocation/1"),
        "priority_order": list(plan.get("priority_order", PRIORITY_ORDER)),
        "authorization_messages": int(counts.get("authorization_ticket", 0)),
        "admitted_action_schemas": list(plan.get("admitted_action_schemas", ()) or ()),
        "blocker_receipt_messages": int(counts.get("current_blocker_receipt_facts", 0)),
        "requested_paths": list(plan.get("requested_paths", ()) or ()),
        "recall_inventory": [dict(row) for row in plan.get("recall_inventory", ()) or ()
                             if isinstance(row, Mapping)],
        "delivered_ranges": [dict(row) for row in plan.get("delivered_ranges", ()) or ()
                             if isinstance(row, Mapping)],
        "dropped_ranges": [dict(row) for row in plan.get("dropped_ranges", ()) or ()
                           if isinstance(row, Mapping)],
    }


def host_limit_outcome(plan: Mapping[str, object], *, request_bytes: int,
                       max_request_bytes: int, prompt_tokens: int,
                       max_prompt_tokens: int) -> dict[str, object]:
    requested = next((row for row in plan.get("requested_ranges", ())
                      if isinstance(row, Mapping)), None)
    path = str(requested.get("path", "")) if requested else ""
    if not path:
        path = str(next(iter(plan.get("requested_paths", ()) or ()), "authorized source"))
    start = int(requested.get("start_line", 1)) if requested else 1
    end = int(requested.get("end_line", start + 99)) if requested else start + 99
    revision = str(requested.get("revision", "UNKNOWN")) if requested else "UNKNOWN"
    page_end = min(end, start + 99)
    quoted_path = json.dumps(path, ensure_ascii=False)
    return {
        "status": "HOST_CONTRACT_LIMIT",
        "reason": "mandatory request content cannot fit the final wire budget",
        "request_bytes": int(request_bytes), "max_request_bytes": int(max_request_bytes),
        "prompt_tokens": int(prompt_tokens), "max_prompt_tokens": int(max_prompt_tokens),
        "source_path": path, "source_revision": revision,
        "paging_path": {"tool": "read", "path": path, "revision": revision,
                        "start_line": start, "end_line": page_end,
                        "instruction": f"read(path={quoted_path}, start_line={start}, end_line={page_end})"},
    }


__all__ = [
    "ALLOC_ENV", "COORDS_ENV", "PRIORITY_ORDER", "allocation_enabled",
    "allocation_summary", "build_priority_plan", "coordinates_enabled",
    "finalize_range_plan", "host_limit_outcome", "optional_recall_record",
    "protected_priority_messages", "render_recall_inventory", "startup_organ_wiring",
    "switch_enabled",
]
