"""Refresh model-facing read and check evidence after an applied product edit."""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path


_EDIT_KINDS = frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"})
_CHECK_KINDS = frozenset({"test", "run_tests", "pytest", "check", "verification"})


def _relative(root: Path, value: object) -> str | None:
    from kernel_request_source_window import _safe_framefit_path

    if not isinstance(value, str) or not value:
        return None
    try:
        path = Path(value)
        if path.is_absolute():
            value = path.relative_to(root).as_posix()
    except (OSError, TypeError, ValueError):
        return None
    return _safe_framefit_path(value)


def _changed_paths(root: Path, receipt: Mapping[str, object], changed_paths: object) -> set[str]:
    values: list[object] = [receipt.get("target_path")]
    changed = receipt.get("changed_files")
    if isinstance(changed, Sequence) and not isinstance(changed, (str, bytes, bytearray)):
        values.extend(row.get("path") if isinstance(row, Mapping) else row for row in changed)
    if isinstance(changed_paths, Sequence) and not isinstance(changed_paths, (str, bytes, bytearray)):
        values.extend(changed_paths)
    elif isinstance(changed_paths, str):
        values.append(changed_paths)
    return {relative for value in values if (relative := _relative(root, value)) is not None}


def _current_source(root: Path, path: str, binding: Mapping[str, object]) -> tuple[str, str] | None:
    from kernel_stateframe_headcarry import _read_source

    raw = _read_source(root, path)
    if raw is None:
        return None
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    files = binding.get("product_files")
    if isinstance(files, Mapping) and files.get(path) != revision:
        return None
    try:
        return raw.decode("utf-8"), revision
    except UnicodeError:
        return None


def _refresh_window(row: Mapping[str, object], source: str, revision: str) -> dict[str, object] | None:
    start, end = row.get("start_line"), row.get("end_line")
    lines = source.splitlines(keepends=True)
    if (type(start) is not int or type(end) is not int or start < 1
            or end < start or start > len(lines)):
        return None
    end = min(end, len(lines))
    content = "".join(lines[start - 1:end])
    return {**dict(row), "revision": revision, "start_line": start, "end_line": end,
            "start_column": 0, "end_column": None, "clip_reason": None,
            "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "byte_length": len(content.encode())}


def stage_post_edit_windows(runtime: object, path: object) -> None:
    """Remember read ranges before low-level edit callbacks invalidate pins."""
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        return
    try:
        relative = _relative(Path(root_value).resolve(), path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return
    if relative is None:
        return
    rows = getattr(runtime, "_read_reuse_visibility_candidates", None)
    if not isinstance(rows, list):
        return
    pending = getattr(runtime, "_post_edit_pending_windows", None)
    pending = dict(pending) if isinstance(pending, Mapping) else {}
    existing = list(pending.get(relative, ()))
    existing.extend(copy.deepcopy(row) for row in rows
                    if isinstance(row, Mapping) and row.get("path") == relative)
    if existing:
        pending[relative] = existing[-64:]
        runtime._post_edit_pending_windows = pending


def refresh_after_applied_edit(
        runtime: object, receipt: Mapping[str, object], binding: Mapping[str, object],
        changed_paths: object = (),
) -> dict[str, str]:
    """Replace old read candidates on edited paths with current, bound source bytes.

    An unavailable or raced file retires its old windows; it can never leave
    pre-edit content labelled current in the next request.
    """
    if receipt.get("kind") not in _EDIT_KINDS or receipt.get("result_class") != "APPLIED":
        return {}
    root = Path(runtime.owner.root).resolve()
    paths = _changed_paths(root, receipt, changed_paths)
    current = {path: _current_source(root, path, binding) for path in paths}
    targets = [runtime]
    capture = getattr(runtime, "_active_capture", None)
    if capture is not None and capture is not runtime:
        targets.append(capture)
    pending = getattr(runtime, "_post_edit_pending_windows", None)
    pending = dict(pending) if isinstance(pending, Mapping) else {}
    for target in targets:
        for attribute in ("_read_reuse_visibility_candidates", "_read_reuse_visibility_dropped"):
            rows = getattr(target, attribute, None)
            if not isinstance(rows, list):
                continue
            if attribute.endswith("_candidates") and target is runtime:
                rows = [*rows, *(row for path in paths for row in pending.get(path, ()))]
            refreshed = []
            seen = set()
            for row in rows:
                if (not isinstance(row, Mapping) or not isinstance(row.get("path"), str)
                        or row["path"] not in paths):
                    refreshed.append(row)
                    continue
                if attribute.endswith("_dropped"):
                    continue
                source = current.get(str(row["path"]))
                replacement = _refresh_window(row, *source) if source is not None else None
                if replacement is None:
                    continue
                key = (replacement["path"], replacement["revision"],
                       replacement["start_line"], replacement["end_line"],
                       replacement["content_sha256"])
                if key not in seen:
                    refreshed.append(replacement)
                    seen.add(key)
            setattr(target, attribute, refreshed)
    # A detached request must read its new closure from the bound workspace,
    # never an optional empirical cache populated by an earlier frame.
    for attribute in ("_empirical_frame_sources", "_empirical_current_revisions"):
        cache = getattr(runtime, attribute, None)
        if isinstance(cache, dict):
            for path in paths:
                cache.pop(path, None)
    for path in paths:
        pending.pop(path, None)
    runtime._post_edit_pending_windows = pending
    revisions = {path: source[1] for path, source in current.items() if source is not None}
    previous = getattr(runtime, "_post_edit_refresh", {})
    prior_paths = previous.get("paths", {}) if isinstance(previous, Mapping) else {}
    runtime._post_edit_refresh = {
        "paths": {**prior_paths, **{path: revisions.get(path, "UNAVAILABLE") for path in paths}},
        "product_revision": binding.get("product_revision", binding.get("artifact")),
        "turn": receipt.get("turn"),
    }
    return revisions


def _check_revision(row: Mapping[str, object]) -> str | None:
    for key in ("checked_revision", "product_revision", "revision"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("binding", "after_binding"):
        value = row.get(key)
        if isinstance(value, Mapping):
            revision = value.get("product_revision", value.get("artifact"))
            if isinstance(revision, str) and revision:
                return revision
    return None


def _stale_check(row: Mapping[str, object], revision: str | None, edit_turn: object) -> dict:
    if row.get("status") == "STALE":
        return dict(row)
    checked = _check_revision(row)
    if checked == revision and revision is not None:
        return dict(row)
    turn = row.get("turn", row.get("step"))
    if checked is None and type(turn) is int and type(edit_turn) is int and turn > edit_turn:
        return dict(row)
    result = dict(row)
    result.update(status="STALE", relevance="historical_workspace_revision",
                  checked_revision=checked or "UNKNOWN")
    for key in ("result", "result_class", "outcome", "verdict"):
        if key in result:
            result[key] = "STALE"
    for key in ("feedback", "detail", "observation", "summary", "output"):
        if key in result:
            result[key] = "Earlier check predates the applied edit; run it on the current revision."
    return result


def retire_pre_edit_source_windows(
        body: Mapping[str, object], refreshed_paths: Mapping[str, str],
        *, protected_messages: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Remove superseded generated source windows before fresh ones are pinned.

    The canonical source-window parser identifies host rendered pages. The
    newest original user suffix remains immutable even if it resembles one.
    """
    from kernel_request_source_window import parse_model_source_window

    result = copy.deepcopy(dict(body))
    messages = result.get("messages")
    if not isinstance(messages, list):
        return result
    protected = (protected_messages if isinstance(protected_messages, Sequence)
                 and not isinstance(protected_messages, (str, bytes, bytearray)) else ())
    retained = []
    for message in messages:
        if (not isinstance(message, Mapping) or message.get("role") != "user"
                or message in protected):
            retained.append(message)
            continue
        metadata = parse_model_source_window(message.get("content"))
        path = metadata.get("path") if isinstance(metadata, Mapping) else None
        if (not isinstance(metadata, Mapping)
                or not isinstance(path, str) or path not in refreshed_paths
                or metadata.get("revision") == refreshed_paths[path]):
            retained.append(message)
    result["messages"] = retained
    return result


def project_post_edit_frame(
        frame: Mapping[str, object], refreshed_paths: Mapping[str, str], *,
        current_product_revision: str | None = None, edit_turn: object = None,
        candidates: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Project refreshed windows and revision-labelled checks without altering audit rows."""
    if not refreshed_paths:
        return dict(frame)
    result = copy.deepcopy(dict(frame))
    current = current_product_revision or next(iter(refreshed_paths.values()))
    by_range = {(row["path"], row.get("start_line"), row.get("end_line")): row
                for row in candidates if isinstance(row, Mapping)
                and isinstance(row.get("path"), str)}
    reads = result.get("recent_reads")
    if isinstance(reads, list):
        for row in reads:
            if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                    or row["path"] not in refreshed_paths):
                continue
            if row.get("file_revision", row.get("revision")) == refreshed_paths[row["path"]]:
                continue
            key = (row.get("path"), row.get("start_line"), row.get("end_line",
                   row.get("selected_end_line")))
            window = by_range.get(key)
            if window is None:
                row.update(content="", status="post_edit_source_unavailable",
                           revision_relevance="historical_file_revision")
            else:
                row.update(content=window["content"], file_revision=window["revision"],
                           omitted_chars=0, status="post_edit_refreshed_source_range",
                           revision_relevance="same_file_revision")
    for key in ("recent_test_feedback", "test_receipts"):
        rows = result.get(key)
        if isinstance(rows, list):
            result[key] = [_stale_check(row, current, edit_turn)
                           if isinstance(row, Mapping) else row for row in rows]
    negatives = result.get("recent_negative_actions")
    if isinstance(negatives, list):
        result["recent_negative_actions"] = [
            _stale_check(row, current, edit_turn)
            if isinstance(row, Mapping) and row.get("tool", row.get("kind")) in _CHECK_KINDS
            else row for row in negatives
        ]
    latest = result.get("latest_receipts")
    if isinstance(latest, dict):
        for key, row in latest.items():
            if isinstance(row, Mapping) and row.get("kind", key) in _CHECK_KINDS:
                latest[key] = _stale_check(row, current, edit_turn)
    return result


__all__ = ["project_post_edit_frame", "refresh_after_applied_edit",
           "retire_pre_edit_source_windows", "stage_post_edit_windows"]
