"""Opt-in projections that keep the model-facing frame aligned with host facts.

This module is deliberately presentation-only.  It does not decide whether an
action was authorized, whether work is complete, or whether a check is green;
it only makes the already-recorded rows unambiguous at the render boundary.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_DISPLAY_TRUTH"
SCHEMA = "stateframe-display-truth/1"

# These names are the single producer/consumer contract for the new display
# fields.  Keep them here rather than repeating spellings in runtime hooks.
OPEN_WORK_FIELD = "open_work"
READ_RECEIPTS_FIELD = "read_receipts"
READ_RECEIPT_KIND = "read_receipt"
PREVIOUS_ACTIONS_FIELD = "previous_actions"
REVISION_ALIASES_FIELD = "revision_aliases"
PREVIOUS_ACTION_LABEL = "Previous action"

_OPEN_WORK_FIELDS = (
    OPEN_WORK_FIELD,
    "pending_multi_definition_edits",
    "unresolved_obligations",
    "obligations",
)
_ACTION_FIELDS = (
    PREVIOUS_ACTIONS_FIELD,
    "action_history",
    "actions",
    "recent_actions",
    "call_history",
    "batch_actions",
    "batch_calls",
    "native_batch_receipts",
    "frame_integrity_receipts",
)
_CALL_FIELDS = ("calls", "actions", "batch_calls", "tool_calls", "call_results",
                "batch", "results", "items")

_READ_MARKERS = {
    "read", "text", "recall_evidence", "read_receipt", "read_observation",
    "workspace_read", "source_read", "read_unchanged",
}
_REFUSED_MARKERS = {
    "REFUSED", "REJECTED", "DENIED", "NOT_RUN", "NO_OP", "ERROR", "TIMEOUT",
    "UNKNOWN", "NO_EXECUTION_RECEIPT", "REJECTED_CONTRACT", "REFUSED_PREDISPATCH",
}
_APPLIED_MARKERS = {
    "APPLIED", "OK", "ACCEPTED", "PASS", "PASSED", "GREEN", "SUCCESS",
    "CERTIFIED", "REUSED_VERIFICATION", "NO_OP_SUCCESS",
}
_WORKSPACE_REVISION_KEYS = {
    "revision", "current_revision", "workspace_revision", "product_revision",
    "after_revision", "before_revision", "checked_revision", "pass_revision",
    "green_revision", "binding_revision", "candidate_revision",
    "basis_revision",
}
_FILE_REVISION_KEYS = {
    "file_revision", "source_revision", "artifact_revision", "working_revision",
}
_SKIP_REVISION_CONTAINERS = {REVISION_ALIASES_FIELD, "identity_aliases", "current_sources"}
_ALIAS_RE = re.compile(r"^(?:W|F)\d+$")
_PATH_ALIAS_RE = re.compile(r"@F\d+$")
_DIGEST_RE = re.compile(r"sha256:[0-9a-fA-F]{64}")
_NATIVE_RECEIPT_RE = re.compile(
    r"^call\s+(?P<index>\S+)\s+(?P<tool>[^:]+):\s*"
    r"(?P<status>[A-Za-z_]+)(?:\(reason=(?P<reason>.*)\))?$"
)


def mode(value: str | None = None) -> str:
    """Return the validated switch value; the default is intentionally OFF."""
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    raw = str(raw).strip().lower()
    if raw not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return raw


def enabled() -> bool:
    return mode() == "on"


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _compact_text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _normalized(value: object) -> str:
    return "_".join(str(value or "").upper().split())


def is_read_receipt(row: object) -> bool:
    """Recognize read evidence without treating ordinary edits as reads."""
    if not isinstance(row, Mapping):
        return False
    if row.get("read") is True or row.get("source_read") is True:
        return True
    for key in ("read_evidence", "read_receipt", "read_observation"):
        if key in row:
            return True
    for key in ("kind", "tool", "action", "type", "receipt_kind", "producer",
                "result_kind", "operation"):
        value = _normalized(row.get(key))
        if value in {marker.upper() for marker in _READ_MARKERS}:
            return True
        if "READ_RECEIPT" in value or value.endswith("_READ") or value.startswith("READ_"):
            return True
    # Some older projections retained predicate=edit_result while carrying a
    # read witness.  The explicit evidence marker is what disambiguates it.
    return ("predicate" in row and "edit_result" == str(row.get("predicate", "")).lower()
            and any(key in row for key in ("read_path", "read_revision", "read_evidence")))


def _compact_read_receipt(row: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"kind": READ_RECEIPT_KIND}
    for key in ("id", "receipt_id", "path", "requested_path", "status", "revision",
                "file_revision", "step"):
        if key in row and row[key] not in (None, ""):
            result[key] = copy.deepcopy(row[key])
    if "path" not in result:
        for key in ("read_path", "source_path", "target_path"):
            if row.get(key):
                result["path"] = copy.deepcopy(row[key])
                break
    return result


def _split_open_rows(value: object) -> tuple[object, list[dict[str, object]]]:
    if isinstance(value, Mapping):
        if is_read_receipt(value):
            return [], [_compact_read_receipt(value)]
        for key in ("items", "rows", "assessments"):
            rows = value.get(key)
            if _sequence(rows):
                kept, reads = _split_open_rows(rows)
                result = copy.deepcopy(dict(value))
                result[key] = kept
                return result, reads
        return copy.deepcopy(value), []
    if not _sequence(value):
        return copy.deepcopy(value), []
    kept: list[object] = []
    reads: list[dict[str, object]] = []
    for row in value:
        if is_read_receipt(row):
            reads.append(_compact_read_receipt(row))
        else:
            kept.append(copy.deepcopy(row))
    return kept, reads


def _action_children(value: Mapping[str, object]) -> tuple[str, Sequence[object]] | None:
    for key in _CALL_FIELDS:
        children = value.get(key)
        if _sequence(children):
            return key, children
    return None


def _flatten_action_rows(value: object, *, batch_id: object = None,
                         inherited_index: object = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        local_batch = next((value.get(key) for key in
                            ("batch_id", "origin_request_id", "origin")
                            if value.get(key) not in (None, "")), batch_id)
        children = _action_children(value)
        if children is not None:
            _key, values = children
            for index, child in enumerate(values):
                rows.extend(_flatten_action_rows(
                    child, batch_id=local_batch, inherited_index=index))
            return rows
        if isinstance(value.get("call"), Mapping):
            rows.extend(_flatten_action_rows(value["call"], batch_id=local_batch,
                                             inherited_index=inherited_index))
            return rows
        if isinstance(value.get("action"), Mapping):
            nested = copy.deepcopy(dict(value["action"]))
            for key in ("result", "outcome", "status", "result_class", "reason", "detail",
                        "step", "basis_revision", "revision"):
                if key not in nested and value.get(key) not in (None, ""):
                    nested[key] = copy.deepcopy(value[key])
            rows.extend(_flatten_action_rows(
                nested, batch_id=local_batch, inherited_index=inherited_index))
            return rows
        if any(value.get(key) is not None for key in
               ("tool", "name", "action", "kind", "status", "result", "outcome",
                "result_class", "receipt_status", "effect_status")):
            row = copy.deepcopy(dict(value))
            if local_batch not in (None, "") and not any(row.get(key) for key in
                                                          ("batch_id", "origin_request_id", "origin")):
                row["batch_id"] = local_batch
            if inherited_index is not None and not any(row.get(key) is not None for key in
                                                        ("index", "call_index", "batch_index")):
                row["index"] = inherited_index
            rows.append(row)
        return rows
    if _sequence(value):
        for index, child in enumerate(value):
            rows.extend(_flatten_action_rows(child, batch_id=batch_id,
                                             inherited_index=index))
    return rows


def _action_sources(frame: Mapping[str, object], action_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in _ACTION_FIELDS:
        if field in frame:
            rows.extend(_flatten_action_rows(frame.get(field)))
    rows.extend(_flatten_action_rows(action_records))
    if not rows and isinstance(frame.get("last_action"), Mapping):
        rows.extend(_flatten_action_rows(frame["last_action"]))
    return rows


def _action_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    batch = row.get("batch_id", row.get("origin_request_id", row.get("origin")))
    index = row.get("index", row.get("call_index", row.get("batch_index")))
    call_id = row.get("call_id", row.get("tool_call_id", row.get("id")))
    if batch not in (None, "") or index is not None or call_id not in (None, ""):
        return (str(batch), str(index), str(call_id), str(row.get("tool", row.get("name", ""))))
    return (str(row.get("step", "")), str(row.get("tool", row.get("name", row.get("kind", "")))),
            _normalized(row.get("result", row.get("status", row.get("outcome", "")))))


def _action_status(row: Mapping[str, object]) -> tuple[str, str | None]:
    raw = next((row.get(key) for key in
                ("result", "outcome", "status", "result_class", "receipt_status", "effect_status")
                if row.get(key) not in (None, "")), None)
    status = _normalized(raw)
    refused = (status in _REFUSED_MARKERS or "REFUS" in status or "REJECT" in status
               or "DENI" in status or "NOT_RUN" in status or "ERROR" in status)
    applied = status in _APPLIED_MARKERS or status.endswith("_APPLIED")
    reason = next((row.get(key) for key in
                   ("reason", "failure_reason", "detail", "error", "observation", "feedback")
                   if row.get(key) not in (None, "")), None)
    if applied and not refused:
        return "APPLIED", None
    if refused or not status:
        return "REFUSED", _compact_text(reason or "no host application receipt")
    # A display row must never imply that an unrecognized result was applied.
    return "REFUSED", _compact_text(reason or f"host result={status}")


def _compact_action_rows(frame: Mapping[str, object],
                         action_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    raw_rows = _action_sources(frame, action_records)
    if not raw_rows:
        return []
    # Runtime records can contain the same call as the frame's native receipt;
    # one identity must render once.
    unique: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in raw_rows:
        identity = _action_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        result, reason = _action_status(row)
        tool = next((row.get(key) for key in ("tool", "name", "action", "kind")
                     if row.get(key) not in (None, "")), "unknown")
        compact: dict[str, object] = {"tool": str(tool), "result": result}
        index = next((row.get(key) for key in ("index", "call_index", "batch_index")
                      if row.get(key) is not None), None)
        if index is not None:
            compact["index"] = index
        else:
            compact["index"] = len(unique)
        batch = next((row.get(key) for key in ("batch_id", "origin_request_id", "origin")
                      if row.get(key) not in (None, "")), None)
        if batch is not None:
            compact["batch_id"] = str(batch)
        if row.get("step") is not None:
            compact["step"] = row["step"]
        if reason is not None and result == "REFUSED":
            compact["reason"] = reason
        for key in ("revision", "basis_revision", "current_revision", "product_revision",
                    "file_revision", "source_revision"):
            if row.get(key) is not None:
                compact[key] = copy.deepcopy(row[key])
                break
        unique.append(compact)
    return unique[-16:]


def _unique(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, (Mapping, list, tuple, set)) or value in (None, ""):
            continue
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def _collect_revision_values(value: object, *, key: str = "", workspace: list[object] | None = None,
                             files: list[object] | None = None, file_paths: dict[str, object] | None = None) -> tuple[list[object], list[object], dict[str, object]]:
    workspace = workspace if workspace is not None else []
    files = files if files is not None else []
    file_paths = file_paths if file_paths is not None else {}
    if isinstance(value, Mapping):
        if key == "artifact_revisions":
            for path, revision in value.items():
                if revision not in (None, ""):
                    files.append(revision)
                    file_paths[str(path)] = revision
            return workspace, files, file_paths
        if key in _SKIP_REVISION_CONTAINERS:
            return workspace, files, file_paths
        for child_key, child in value.items():
            name = str(child_key)
            if name in _WORKSPACE_REVISION_KEYS:
                if child not in (None, ""):
                    workspace.append(child)
                continue
            if name in _FILE_REVISION_KEYS:
                if child not in (None, ""):
                    files.append(child)
                continue
            _collect_revision_values(child, key=name, workspace=workspace, files=files,
                                     file_paths=file_paths)
    elif _sequence(value):
        for child in value:
            _collect_revision_values(child, key=key, workspace=workspace, files=files,
                                     file_paths=file_paths)
    return workspace, files, file_paths


def _current_workspace_value(frame: Mapping[str, object], values: list[object]) -> object:
    for key in ("workspace_revision", "current_revision", "product_revision", "revision"):
        if frame.get(key) not in (None, ""):
            return frame[key]
    binding = frame.get("binding")
    if isinstance(binding, Mapping):
        for key in ("product_revision", "current_revision", "revision"):
            if binding.get(key) not in (None, ""):
                return binding[key]
    snapshot = frame.get("workspace_snapshot")
    if isinstance(snapshot, str) and snapshot.startswith("W"):
        return snapshot
    artifacts = frame.get("artifact_revisions")
    if isinstance(artifacts, Mapping):
        for value in artifacts.values():
            if value not in (None, ""):
                return value
    return values[-1] if values else None


def _revision_maps(frame: Mapping[str, object]) -> tuple[dict[str, str], dict[str, str], dict[str, object]]:
    workspace, files, file_paths = _collect_revision_values(frame)
    workspace_values = _unique(workspace)
    file_values = _unique(files)
    current = _current_workspace_value(frame, workspace)
    current_text = None if current in (None, "") else str(current)
    if current_text and current_text not in workspace_values:
        workspace_values.append(current_text)
    workspace_aliases: dict[str, str] = {}
    if workspace_values:
        if current_text not in workspace_values:
            current_text = workspace_values[-1]
        others = [item for item in workspace_values if item != current_text]
        for index, item in enumerate(others):
            workspace_aliases[item] = f"W{index}"
        workspace_aliases[current_text] = "W1"
        # If there are more historical values, keep W1 reserved for current.
        next_alias = 2
        for item in others:
            if workspace_aliases[item] == "W0":
                continue
            workspace_aliases[item] = f"W{next_alias}"
            next_alias += 1
    file_aliases = {item: f"F{index}" for index, item in enumerate(file_values, 1)}
    meta: dict[str, object] = {}
    if workspace_aliases:
        current_alias = workspace_aliases.get(current_text, "W1")
        ordered = sorted(
            (workspace_aliases[item] for item in workspace_values if item in workspace_aliases),
            key=lambda item: int(item[1:]) if item[1:].isdigit() else item,
        )
        meta["workspace"] = {"current": current_alias, "distinct": ordered}
    if file_aliases:
        meta["files"] = {
            path.replace("\\", "/").lstrip("./"): file_aliases.get(str(value), "F0")
            for path, value in sorted(file_paths.items())
        }
    return workspace_aliases, file_aliases, meta


def _alias_value(value: object, aliases: Mapping[str, str]) -> object:
    if value in (None, "") or isinstance(value, Mapping):
        return value
    text = str(value)
    if text in aliases:
        return aliases[text]
    if _ALIAS_RE.match(text) or _PATH_ALIAS_RE.search(text):
        return value
    return value


def _rewrite_revisions(value: object, workspace: Mapping[str, str], files: Mapping[str, str],
                       *, key: str = "", source: object = None) -> object:
    if isinstance(value, Mapping):
        source_map = source if isinstance(source, Mapping) else {}
        result: dict[object, object] = {}
        for raw_key, child in value.items():
            name = str(raw_key)
            source_child = source_map.get(raw_key)
            if name == "artifact_revisions" and isinstance(child, Mapping):
                result[name] = {
                    str(path).replace("\\", "/").lstrip("./"): _alias_value(revision, files)
                    for path, revision in child.items()
                }
            elif name in _WORKSPACE_REVISION_KEYS:
                candidate = source_child if source_child not in (None, "") else child
                result[raw_key] = _alias_value(candidate, workspace)
            elif name in _FILE_REVISION_KEYS:
                candidate = source_child if source_child not in (None, "") else child
                result[raw_key] = _alias_value(candidate, files)
            elif name == "workspace_snapshot":
                result[raw_key] = (next(iter(workspace.values()), child)
                                   if workspace else child)
            elif name in _SKIP_REVISION_CONTAINERS:
                result[raw_key] = copy.deepcopy(child)
            else:
                result[raw_key] = _rewrite_revisions(
                    child, workspace, files, key=name, source=source_child)
        return result
    if _sequence(value):
        source_values = source if _sequence(source) else ()
        return [_rewrite_revisions(
            child, workspace, files, key=key,
            source=source_values[index] if index < len(source_values) else None)
                for index, child in enumerate(value)]
    return value


def render_previous_actions(frame: Mapping[str, object]) -> str:
    """Render all retained calls as one compact, typed line."""
    rows = frame.get(PREVIOUS_ACTIONS_FIELD)
    rows = rows if _sequence(rows) else _compact_action_rows(frame, ())
    if not rows:
        return ""
    rendered: list[str] = []
    for fallback, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        index = row.get("index", fallback)
        tool = row.get("tool", "unknown")
        result = row.get("result", "REFUSED")
        text = f"call {index} {tool}: {result}"
        if result == "REFUSED":
            text += f" (reason={_compact_text(row.get('reason') or 'no host application receipt')})"
        rendered.append(text)
    return f"{PREVIOUS_ACTION_LABEL}: " + "; ".join(rendered) if rendered else ""


def project_frame(frame: Mapping[str, object], *, source_frame: Mapping[str, object] | None = None,
                  action_records: Sequence[Mapping[str, object]] = ()) -> Mapping[str, object]:
    """Project only display data; OFF returns the original object untouched."""
    if not enabled():
        return frame
    if not isinstance(frame, Mapping):
        return frame
    source = source_frame if isinstance(source_frame, Mapping) else frame
    projected: dict[str, object] = copy.deepcopy(dict(frame))
    read_rows: list[dict[str, object]] = []
    for field in _OPEN_WORK_FIELDS:
        if field not in projected:
            continue
        projected[field], found = _split_open_rows(projected[field])
        read_rows.extend(found)
    if read_rows:
        existing = projected.get(READ_RECEIPTS_FIELD)
        existing_rows = ([_compact_read_receipt(row) if isinstance(row, Mapping) else row
                          for row in existing]
                         if _sequence(existing) else [])
        projected[READ_RECEIPTS_FIELD] = existing_rows + read_rows

    actions = _compact_action_rows(source, action_records)
    if actions:
        projected[PREVIOUS_ACTIONS_FIELD] = actions
        projected.pop("last_action", None)
        # The compact field is the display consumer; retaining every source
        # history beside it would recreate the duplicate the projection is
        # meant to collapse.
        for field in _ACTION_FIELDS:
            if field != PREVIOUS_ACTIONS_FIELD:
                projected.pop(field, None)

    workspace, files, aliases = _revision_maps(source)
    projected = _rewrite_revisions(projected, workspace, files, source=source)
    if aliases:
        if isinstance(aliases.get("workspace"), Mapping):
            projected["workspace_snapshot"] = aliases["workspace"].get("current", "W1")
        projected[REVISION_ALIASES_FIELD] = aliases
    return projected


def _native_receipt_rows(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    latest: list[dict[str, object]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or "[NATIVE BATCH RECEIPTS]" not in content:
            continue
        rows: list[dict[str, object]] = []
        active = False
        for line in content.splitlines():
            if line == "[NATIVE BATCH RECEIPTS]":
                active = True
                continue
            if not active:
                continue
            match = _NATIVE_RECEIPT_RE.match(line.strip())
            if match:
                index = match.group("index")
                rows.append({
                    "batch_index": int(index) if index.isdigit() else index,
                    "tool": match.group("tool").strip(),
                    "status": match.group("status"),
                    "reason": match.group("reason") or "",
                })
        if rows:
            latest = rows
    return latest


def _replace_working_action(content: str, frame: Mapping[str, object]) -> str:
    rendered = render_previous_actions(frame)
    if not rendered or not isinstance(content, str):
        return content
    lines = content.splitlines()
    replaced = False
    output: list[str] = []
    for line in lines:
        lower = line.lower()
        is_old = (lower.startswith("last_action:")
                  or "previous action " in lower
                  or lower.startswith("resolved: the action you chose"))
        if is_old:
            if not replaced:
                prefix = "HOST_FACT: " if line.startswith("HOST_FACT:") else ""
                output.append(prefix + rendered)
                replaced = True
            continue
        output.append(line)
    if not replaced and output and output[0].startswith("[WORKING_VIEW"):
        output.insert(1, rendered)
    return "\n".join(output)


def _replace_terminal_action(content: str, frame: Mapping[str, object]) -> str:
    rendered = render_previous_actions(frame)
    if not rendered:
        return content
    lines = content.splitlines()
    for index, line in enumerate(lines):
        lower = line.lower()
        if ("your_last_step:" in lower or "last_edit:" in lower or "last_action:" in lower):
            lines[index] = rendered
            return "\n".join(lines)
    return content


def _replace_unified_revisions(content: str, frame: Mapping[str, object]) -> str:
    workspace, _files, _meta = _revision_maps(frame)
    if not workspace:
        return content
    for raw, alias in sorted(workspace.items(), key=lambda item: len(item[0]), reverse=True):
        if _DIGEST_RE.fullmatch(raw):
            content = content.replace(raw, alias)
        else:
            content = re.sub(r"(current_revision|revision|epoch)=" + re.escape(raw) + r"\b",
                             lambda match: match.group(1) + "=" + alias, content)
    return content


def transform_working_view(message: Mapping[str, object], *, frame: Mapping[str, object]) -> dict:
    result = copy.deepcopy(dict(message))
    content = result.get("content")
    if isinstance(content, str):
        result["content"] = _replace_working_action(content, frame)
    return result


def transform_messages(messages: Sequence[Mapping[str, object]], *, frame: Mapping[str, object] | None = None,
                       action_records: Sequence[Mapping[str, object]] = ()) -> Sequence[Mapping[str, object]]:
    """Apply the display projection to already-composed frame messages."""
    if not enabled():
        return messages
    native_rows = _native_receipt_rows(messages)
    # The pinned native receipt block is the authoritative complete batch. Do
    # not append runtime closure records to it a second time.
    combined_actions = native_rows if native_rows else list(action_records)
    public_frame = (project_frame(frame, action_records=combined_actions)
                    if isinstance(frame, Mapping) else None)
    result: list[Mapping[str, object]] = []
    last_state: Mapping[str, object] | None = None
    for message in messages:
        copied = copy.deepcopy(dict(message)) if isinstance(message, Mapping) else message
        content = copied.get("content") if isinstance(copied, Mapping) else None
        if isinstance(content, str) and content.startswith("[StateFrame]\n"):
            try:
                value = json.loads(content.split("\n", 1)[1])
            except (IndexError, TypeError, ValueError, json.JSONDecodeError):
                if public_frame is not None:
                    copied["content"] = _replace_terminal_action(content, public_frame)
            else:
                if isinstance(value, Mapping):
                    source = frame if isinstance(frame, Mapping) else value
                    projected = project_frame(value, source_frame=source,
                                              action_records=combined_actions)
                    copied["content"] = "[StateFrame]\n" + json.dumps(
                        projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    last_state = projected
        elif isinstance(content, str) and content.startswith("[WORKING_VIEW"):
            working_frame = public_frame or last_state
            if working_frame is not None:
                copied["content"] = _replace_working_action(content, working_frame)
        elif isinstance(content, str) and content.startswith("[STATEFRAME UNIFIED"):
            copied["content"] = _replace_unified_revisions(
                content, frame if isinstance(frame, Mapping) else (public_frame or {}))
        result.append(copied)
    return result


__all__ = [
    "MODE_ENV", "SCHEMA", "OPEN_WORK_FIELD", "READ_RECEIPTS_FIELD",
    "READ_RECEIPT_KIND", "PREVIOUS_ACTIONS_FIELD", "REVISION_ALIASES_FIELD",
    "PREVIOUS_ACTION_LABEL",
    "enabled", "is_read_receipt", "mode", "project_frame", "render_previous_actions",
    "transform_messages", "transform_working_view",
]
