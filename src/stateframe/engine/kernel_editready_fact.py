"""Truthful, opt-in edit-ready facts derived only from bound read tool results."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from kernel_request_diet import _source_identity


MODE_ENV = "STATEFRAME_EDITREADY"
_VALID_MODES = frozenset({"off", "on"})
_EDITREADY_FACT_PREFIX = "[host] The current source of "
_EDITREADY_FACT_SUFFIX = " is in this request (see the read pages above). You can edit it now."
_PROTECTED_PANEL_PREFIXES = (
    "[Request diet: dropped source ranges]",
    "[Deferred required source paths]",
    "[GREENFACT]",
    "[Done path]",
    "[Done-path]",
)
_TEST_PARTS = frozenset({"test", "tests", "__tests__"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = values.get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _path_values(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.keys())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            item.get("path") if isinstance(item, Mapping) else item
            for item in value
        )
    return ()


def _normalize_repo_path(value: object, root: Path) -> str | None:
    raw = str(value or "").strip().replace("\\", "/").removeprefix("./")
    if not raw:
        return None
    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            relative = candidate.resolve().relative_to(root).as_posix()
        else:
            relative = PurePosixPath(raw).as_posix()
    except (OSError, ValueError):
        return None
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    if any(part.casefold() in _TEST_PARTS for part in parts):
        return None
    if PurePosixPath(parts[-1]).name.casefold().startswith("test_"):
        return None
    try:
        (root / Path(*parts)).resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    return PurePosixPath(*parts).as_posix()


def _editable_paths(runtime: object) -> frozenset[str]:
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        return frozenset()
    try:
        root = Path(root_value).resolve()
    except (OSError, TypeError, ValueError):
        return frozenset()

    values: list[object] = [getattr(runtime, "target", None)]
    holders: list[object] = [
        getattr(runtime, "edit_contract", None),
        getattr(owner, "edit_contract", None),
        getattr(owner, "task_row", None),
    ]
    for holder in holders:
        if not isinstance(holder, Mapping):
            continue
        values.extend(_path_values(holder.get("editable_paths", ())))
        values.extend(_path_values(holder.get("target_files", ())))
        values.extend(_path_values(holder.get("files", ())))
        authority = holder.get("edit_authority")
        if isinstance(authority, Mapping):
            values.extend(_path_values(authority.get("editable_paths", ())))
    paths = (_normalize_repo_path(value, root) for value in values)
    return frozenset(path for path in paths if path is not None)


def _read_call_ids(messages: Sequence[Mapping[str, object]]) -> frozenset[str]:
    bound: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else call.get("name")
            call_id = call.get("id")
            if name == "read" and isinstance(call_id, str) and call_id:
                bound.add(call_id)
    return frozenset(bound)


def _bound_read_results(
    messages: Sequence[Mapping[str, object]],
) -> Sequence[Mapping[str, object]]:
    bound_call_ids: set[str] = set()
    results: list[Mapping[str, object]] = []
    for message in messages:
        if message.get("role") == "assistant":
            bound_call_ids.update(_read_call_ids((message,)))
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id in bound_call_ids:
                results.append(message)
    return results


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if start < 1 or end < start:
            continue
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def editable_source_ranges(
    messages: Sequence[Mapping[str, object]], runtime: object,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return merged ranges from source-bearing results of bound read calls."""
    editable = _editable_paths(runtime)
    if not editable:
        return {}
    owner = getattr(runtime, "owner", None)
    try:
        root = Path(getattr(owner, "root")).resolve()
    except (OSError, TypeError, ValueError):
        return {}
    collected: dict[str, list[tuple[int, int]]] = {}
    for message in _bound_read_results(messages):
        identity = _source_identity(message)
        if identity is None or identity[0] != "read":
            continue
        path = _normalize_repo_path(identity[1], root)
        source_range = identity[3]
        if path not in editable or not isinstance(source_range, tuple) or len(source_range) != 2:
            continue
        start, end = source_range
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            continue
        collected.setdefault(path, []).append((start, end))
    return {
        path: _merge_ranges(collected[path])
        for path in sorted(collected)
        if collected[path]
    }


def format_editready_fact(
    ranges: Mapping[str, Sequence[tuple[int, int]]],
) -> str | None:
    """Render the pinned EDITREADY sentence shape, with stable path/range order."""
    clauses: list[str] = []
    for path in sorted(ranges):
        spans = _merge_ranges(tuple(ranges[path]))
        if not spans:
            continue
        rendered = ", ".join(
            str(start) if start == end else f"{start}-{end}"
            for start, end in spans
        )
        clauses.append(f"{path} lines {rendered}")
    if not clauses:
        return None
    source_text = clauses[0]
    for clause in clauses[1:]:
        source_text += f"; and of {clause}"
    return (
        "[host] The current source of "
        + source_text
        + " is in this request (see the read pages above). You can edit it now."
    )


def build_editready_fact(
    messages: Sequence[Mapping[str, object]], runtime: object,
) -> str | None:
    return format_editready_fact(editable_source_ranges(messages, runtime))


def _safe_panel(message: Mapping[str, object]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    stripped = content.lstrip()
    if stripped.startswith(_PROTECTED_PANEL_PREFIXES):
        return False
    lowered = stripped.casefold()
    if lowered.startswith(("[done", "[greenfact", "[request diet:", "[deferred required source paths]")):
        return False
    if lowered.startswith(("task:", "[current source:", "[working source:", "[host fact:")):
        return False
    return True


def _is_generated_editready_fact(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(_EDITREADY_FACT_PREFIX)
        and content.endswith(_EDITREADY_FACT_SUFFIX)
        and " is in this request (see the read pages above)." in content
    )


def _reconcile_existing_fact(
    copied: list[dict[str, object]],
    existing_indexes: Sequence[int],
    fact: str,
    *,
    path_count: int,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    if not existing_indexes:
        return None
    exact_index = next(
        (index for index in existing_indexes if copied[index].get("content") == fact),
        None,
    )
    selected_index = exact_index if exact_index is not None else existing_indexes[0]
    retained: list[dict[str, object]] = []
    output_index = None
    existing = set(existing_indexes)
    for index, message in enumerate(copied):
        if index not in existing:
            retained.append(message)
        elif index == selected_index:
            output_index = len(retained)
            retained.append({**message, "content": fact})
    return retained, {
        "editready_schema": "stateframe-editready/1",  # never "schema": journal-RESERVED (r2EY ALL-arm step-0 crash)
        "status": (
            "already_present"
            if exact_index is not None and len(existing_indexes) == 1
            else "reconciled"
        ),
        "fact_sha256": hashlib.sha256(fact.encode("utf-8")).hexdigest(),
        "path_count": path_count,
        "message_index": output_index,
        "stale_fact_removed": len(existing_indexes) > 1 or exact_index is None,
    }


def apply_editready_fact(
    messages: Sequence[Mapping[str, object]], runtime: object,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Reconcile the range fact against current bound reads; preserve protected panels."""
    if not enabled():
        return [dict(message) for message in messages], None
    ranges = editable_source_ranges(messages, runtime)
    fact = format_editready_fact(ranges)
    copied = [dict(message) for message in messages]
    existing_indexes = [
        index for index, message in enumerate(copied)
        if _is_generated_editready_fact(message)
    ]
    if fact is None:
        retained = [
            message for index, message in enumerate(copied)
            if index not in existing_indexes
        ]
        return retained, {
            "editready_schema": "stateframe-editready/1",  # never "schema": journal-RESERVED (r2EY ALL-arm step-0 crash)
            "status": "no_bound_editable_read_ranges",
            "stale_fact_removed": bool(existing_indexes),
        }

    reconciled = _reconcile_existing_fact(
        copied, existing_indexes, fact, path_count=len(ranges),
    )
    if reconciled is not None:
        return reconciled

    candidates = [
        index for index, message in enumerate(copied)
        if _safe_panel(message)
    ]
    preferred = [
        index for index in candidates
        if str(copied[index].get("content", "")).lstrip().casefold().startswith("[read reuse]")
    ]
    protected_positions = [
        index for index, message in enumerate(copied)
        if message.get("role") == "user"
        and str(message.get("content", "")).lstrip().startswith(_PROTECTED_PANEL_PREFIXES[:2])
    ]
    after_read_order = [
        index for index in candidates
        if protected_positions and index > max(protected_positions)
    ]
    before_read_order = [
        index for index in candidates
        if protected_positions and index < min(protected_positions)
    ]
    if preferred and (
        not protected_positions or preferred[-1] < min(protected_positions)
    ):
        # A read-reuse placeholder before the protected read-order notices is
        # the exact spec slot in the pinned multi-file row.
        selected = preferred[-1]
    elif after_read_order:
        # Keep the truthful notice group intact and use its nearest safe panel
        # on the following side, not an unrelated read-reuse panel at turn end.
        selected = after_read_order[0]
    elif before_read_order:
        selected = before_read_order[-1]
    elif preferred:
        selected = preferred[-1]
    else:
        selected = candidates[-1] if candidates else None

    if selected is None:
        copied.append({"role": "user", "content": fact})
        action = "appended"
    else:
        copied[selected] = {**copied[selected], "content": fact}
        action = "replaced_panel"
    return copied, {
        "editready_schema": "stateframe-editready/1",  # never "schema": journal-RESERVED (r2EY ALL-arm step-0 crash)
        "status": action,
        "fact_sha256": hashlib.sha256(fact.encode("utf-8")).hexdigest(),
        "path_count": len(ranges),
        "message_index": selected,
    }


__all__ = [
    "MODE_ENV",
    "apply_editready_fact",
    "build_editready_fact",
    "editable_source_ranges",
    "enabled",
    "format_editready_fact",
    "mode",
]
