"""Final-wire visibility pins for source ranges returned by read."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import kernel_workspace_truth as _workspace_truth


_MAX_CANDIDATES = 64
_MAX_CANDIDATE_BYTES = 1_000_000
_MAX_DROPPED_SPANS = 2048
_REUSE_PATH = re.compile(r"\[read reuse\] path=([^\s]+)")
_REUSE_RANGE = re.compile(r"\blines=(\d+)-(\d+)\b")
_COVERAGE_RANGE = re.compile(r"Coverage: lines (\d+)-(\d+), columns \d+-\d+;")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


_BASE_SOURCE_PROOF_FAILURES = frozenset({
    "final_wire_proof_missing", "final_wire_proof_incomplete",
})
_EXPANDED_SOURCE_PROOF_FAILURES = _BASE_SOURCE_PROOF_FAILURES | {"source_body_blank"}
_BOUNDARY_SOURCE_PROOF_FAILURES = _EXPANDED_SOURCE_PROOF_FAILURES | {
    "deferred_source_notice_unverified",
}


def source_proof_deferral_enabled() -> bool:
    """Return whether final-wire missing/blank proofs may use the read route."""
    value = os.environ.get("STATEFRAME_SOURCE_PROOF_DEFERRAL", "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError("STATEFRAME_SOURCE_PROOF_DEFERRAL must be on or off")
    return value == "on"


def _remember_dropped_span(target: object, row: Mapping[str, object]) -> None:
    descriptor = {
        key: row[key] for key in ("path", "revision", "start_line", "end_line")
        if key in row
    }
    existing = getattr(target, "_read_reuse_visibility_dropped", None)
    dropped = list(existing) if isinstance(existing, list) else []
    key = tuple(descriptor.get(name) for name in (
        "path", "revision", "start_line", "end_line"))
    if not any(
        isinstance(item, Mapping)
        and tuple(item.get(name) for name in (
            "path", "revision", "start_line", "end_line")) == key
        for item in dropped
    ):
        dropped.append(descriptor)
    try:
        setattr(target, "_read_reuse_visibility_dropped", dropped[-_MAX_DROPPED_SPANS:])
    except (AttributeError, TypeError):
        pass


def remember_read_span(
        capture: object, root: Path, path: Path, revision: object,
        metadata: Mapping[str, object], content: object,
) -> None:
    """Keep a bounded digest-bound candidate until the next request is serialized."""
    if not isinstance(content, str) or not content:
        return
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, TypeError, ValueError):
        return
    if not relative or relative.startswith("../"):
        return
    effective = metadata.get("effective_range")
    effective = effective if isinstance(effective, Mapping) else metadata
    start = effective.get("start_line", metadata.get("start_line"))
    end = effective.get("end_line", metadata.get("selected_end_line"))
    if (type(start) is not int or type(end) is not int or start < 1 or end < start):
        return
    revision_text = str(revision or metadata.get("revision") or "")
    if not _SHA256.fullmatch(revision_text):
        return
    payload = content.encode("utf-8")
    candidate = {
        "path": relative,
        "revision": revision_text,
        "start_line": start,
        "end_line": end,
        "start_column": effective.get("start_column", 0),
        "end_column": effective.get("end_column"),
        "clip_reason": metadata.get("clip_reason"),
        "content": content,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
    }
    targets = [capture]
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    if _workspace_truth.enabled():
        turn_source = runtime if runtime is not None else capture
        turn = getattr(turn_source, "current_step", None)
        if type(turn) is int and turn >= 1:
            candidate["turn"] = turn
    if runtime is not None and runtime is not capture:
        targets.append(runtime)
    for target in targets:
        existing = getattr(target, "_read_reuse_visibility_candidates", None)
        candidates = list(existing) if isinstance(existing, list) else []
        key = (relative, revision_text, start, end, candidate["content_sha256"])
        candidates = [
            row for row in candidates
            if not isinstance(row, Mapping) or (
                row.get("path"), row.get("revision"), row.get("start_line"),
                row.get("end_line"), row.get("content_sha256"),
            ) != key
        ]
        candidates.append(candidate)
        while (len(candidates) > _MAX_CANDIDATES
               or sum(int(row.get("byte_length", 0)) for row in candidates
                      if isinstance(row, Mapping)) > _MAX_CANDIDATE_BYTES):
            removed = candidates.pop(0)
            if isinstance(removed, Mapping):
                _remember_dropped_span(target, removed)
        try:
            setattr(target, "_read_reuse_visibility_candidates", candidates)
        except (AttributeError, TypeError):
            pass


def _relative(value: object) -> str:
    return str(value or "").replace("\\", "/").removeprefix("./").strip("/")


def source_delivery_identity(value: Mapping[str, object]) -> dict[str, object] | None:
    """Normalize the exact path/revision/range/digest proven in a provider request."""
    raw_path = str(value.get("path") or "").replace("\\", "/").strip()
    if raw_path.startswith("/") or re.match(r"^[A-Za-z]:/", raw_path):
        return None
    path = _relative(raw_path)
    revision = str(value.get("revision") or "").strip()
    effective = value.get("effective_range")
    effective = effective if isinstance(effective, Mapping) else {}
    start = effective.get("start_line", value.get("start_line"))
    end = effective.get(
        "end_line", value.get("end_line", value.get("selected_end_line")),
    )
    digest_value = (
        value.get("content_sha256") or value.get("page_bytes_sha256")
        or value.get("content_hash")
    )
    digest = str(digest_value or "").strip().lower().removeprefix("sha256:")
    content = value.get("content")
    if (not path or any(part == ".." for part in path.split("/"))
            or not _SHA256.fullmatch(revision)
            or type(start) is not int or type(end) is not int
            or start < 1 or end < start
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        return None
    if isinstance(content, str) and hashlib.sha256(content.encode("utf-8")).hexdigest() != digest:
        return None
    return {
        "path": path,
        "revision": revision,
        "start_line": start,
        "end_line": end,
        "content_sha256": digest,
    }


def editable_read_paths(runtime: object) -> set[str]:
    """Use edit authority, not readable closure dependencies, for read priority."""
    contract = getattr(runtime, "edit_contract", None)
    authority = contract.get("edit_authority") if isinstance(contract, Mapping) else None
    paths = authority.get("editable_paths") if isinstance(authority, Mapping) else None
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes, bytearray)):
        return set()
    from kernel_request_source_window import _safe_framefit_path
    return {path for value in paths if (path := _safe_framefit_path(value)) is not None}


def _contract_paths(runtime: object) -> set[str]:
    paths: set[str] = set()
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is not None:
        try:
            root = Path(root_value).resolve()
            target = getattr(runtime, "target", None)
            if target is not None:
                try:
                    paths.add(Path(target).resolve().relative_to(root).as_posix())
                except (OSError, TypeError, ValueError):
                    pass
        except (OSError, TypeError, ValueError):
            pass
    row = getattr(owner, "task_row", {})
    if isinstance(row, Mapping):
        values: list[object] = []
        for key in ("path", "rel", "allowed_product_paths", "new_file_roots"):
            value = row.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                values.extend(value)
        entry = row.get("entry")
        if isinstance(entry, Mapping):
            values.append(entry.get("path"))
        for value in values:
            relative = _relative(value)
            if relative and not relative.startswith("/") and ".." not in Path(relative).parts:
                paths.add(relative)
    required_caller = _relative(getattr(owner, "required_caller_path", ""))
    if required_caller and not required_caller.startswith("/") and ".." not in Path(required_caller).parts:
        paths.add(required_caller)
    plan = getattr(runtime, "_request_source_closure", None)
    if isinstance(plan, Mapping):
        for value in plan.get("paths", ()) if isinstance(plan.get("paths"), Sequence) else ():
            relative = _relative(value)
            if relative and not relative.startswith("/") and ".." not in Path(relative).parts:
                paths.add(relative)
        units = plan.get("units")
        if isinstance(units, Sequence) and not isinstance(units, (str, bytes, bytearray)):
            for unit in units:
                if isinstance(unit, Mapping):
                    relative = _relative(unit.get("path"))
                    if relative and not relative.startswith("/") and ".." not in Path(relative).parts:
                        paths.add(relative)
        deferred = plan.get("deferred_paths")
        if isinstance(deferred, Sequence) and not isinstance(deferred, (str, bytes, bytearray)):
            paths.update(_relative(value) for value in deferred if _relative(value))
    try:
        from kernel_request_source_window import editable_product_paths
        paths.update(editable_product_paths(runtime))
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return paths


def _source_range_block_position(
        content: object, candidate: Mapping[str, object],
) -> tuple[int, int] | None:
    if not isinstance(content, str):
        return None
    path = str(candidate.get("path", ""))
    start, end = candidate.get("start_line"), candidate.get("end_line")
    if not path or type(start) is not int or type(end) is not int:
        return None
    header = re.match(
        rf"^\[(?:Current|Working|Retained|Observed) source:\s*"
        rf"{re.escape(path)}\]\n",
        content,
    )
    coverage = _COVERAGE_RANGE.search(content)
    if (header is None or coverage is None
            or tuple(map(int, coverage.groups())) != (start, end)
            or coverage.start() < header.end()):
        return None
    return header.end(), coverage.end()


def _source_message(content: object, candidate: Mapping[str, object]) -> bool:
    if not isinstance(content, str):
        return False
    path = str(candidate.get("path", ""))
    body = str(candidate.get("content", ""))
    start, end = candidate.get("start_line"), candidate.get("end_line")
    revision = str(candidate.get("revision", ""))
    if (not path or not body or body not in content
            or type(start) is not int or type(end) is not int):
        return False

    headers = (
        f"[Current source: {path}]",
        f"[Working source: {path}]",
        f"[Retained source: {path}]",
        f"[Observed source: {path}]",
        f"[read page] path={path} ",
        f"CURRENT_SOURCE {path} ",
        f"Observed source {path}",
    )
    if not any(content.startswith(header) for header in headers):
        return False

    coverage = _COVERAGE_RANGE.search(content)
    if coverage is not None:
        return tuple(map(int, coverage.groups())) == (start, end)

    first_line = content.splitlines()[0] if content.splitlines() else ""
    page = re.match(
        r"^\[read page\]\s+path=(\S+)\s+revision=(sha256:[0-9a-f]{64})\s+"
        r"lines=(\d+)-(\d+)/\d+\b",
        first_line,
    )
    if page is not None:
        return (page.group(1) == path and page.group(2) == revision
                and (int(page.group(3)), int(page.group(4))) == (start, end))

    marker = "[Source metadata]\n"
    if marker in content:
        metadata_line = content.split(marker, 1)[1].split("\n", 1)[0]
        try:
            metadata = json.loads(metadata_line)
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, Mapping):
            metadata_path = str(metadata.get("path", ""))
            metadata_revision = str(metadata.get("revision", metadata.get("file_revision", "")))
            metadata_start = metadata.get("start_line")
            metadata_end = metadata.get("selected_end_line", metadata.get("end_line"))
            return (
                metadata_path == path
                and metadata_revision == revision
                and type(metadata_start) is int and type(metadata_end) is int
                and (metadata_start, metadata_end) == (start, end)
            )

    try:
        from kernel_request_source_window import parse_model_source_window
        projected = parse_model_source_window(content)
    except (ImportError, AttributeError, TypeError, ValueError):
        projected = None
    if isinstance(projected, Mapping) and str(projected.get("path", "")) == path:
        ranges = projected.get("ranges")
        if isinstance(ranges, list):
            return any(
                isinstance(row, Mapping)
                and row.get("start_line") == start and row.get("end_line") == end
                and (not revision or str(projected.get("revision", "")) == revision)
                for row in ranges
            )

    complete = re.search(
        r"Coverage: COMPLETE FILE; lines (\d+)-(\d+) of (\d+);", content)
    return bool(
        complete
        and (int(complete.group(1)), int(complete.group(2))) == (start, end)
        and int(complete.group(2)) == int(complete.group(3))
        and start == 1
    )


def _panel_boundary(messages: Sequence[object]) -> int:
    try:
        from kernel_request_source_window import _is_optional_panel
    except ImportError:
        return len(messages)
    return next((index for index, item in enumerate(messages) if _is_optional_panel(item)),
                len(messages))


def _visible_before_panels(messages: Sequence[object], candidate: Mapping[str, object]) -> bool:
    boundary = _panel_boundary(messages)
    return any(
        index < boundary and isinstance(message, Mapping)
        and _source_message(message.get("content"), candidate)
        for index, message in enumerate(messages)
    )


def _read_current_span_status(
        runtime: object, candidate: Mapping[str, object],
) -> tuple[str | None, str, str | None]:
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        return None, "workspace_root_unavailable", None
    try:
        root = Path(root_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, "workspace_root_unavailable", None
    relative = _relative(candidate.get("path"))
    relative_path = Path(relative)
    if (not relative or relative_path.is_absolute()
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in relative_path.parts)):
        return None, "path_outside_authorized_shape", None
    try:
        unresolved = root / relative_path
        probe = root
        for part in relative_path.parts:
            probe = probe / part
            if probe.is_symlink():
                return None, "symlink_path", None
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
        if not path.is_file():
            return None, "not_a_regular_file", None
        raw = path.read_bytes()
        source = raw.decode("utf-8")
    except (OSError, RuntimeError, UnicodeError, TypeError, ValueError):
        return None, "current_source_unavailable", None
    revision = str(candidate.get("revision", ""))
    current_revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    if revision != current_revision:
        return None, "revision_changed", None
    lines = source.splitlines(keepends=True)
    start, end = int(candidate["start_line"]), int(candidate["end_line"])
    if not 1 <= start <= end <= len(lines):
        return None, "range_outside_current_source", None
    start_column = candidate.get("start_column", 0)
    end_column = candidate.get("end_column")
    if type(start_column) is not int or start_column < 0:
        return None, "invalid_effective_column", None
    if end_column is not None and (type(end_column) is not int or end_column < start_column):
        return None, "invalid_effective_column", None
    if start == end:
        body = lines[start - 1][start_column:end_column]
    else:
        selected = lines[start - 1:end]
        selected[0] = selected[0][start_column:]
        if end_column is not None:
            selected[-1] = selected[-1][:end_column]
        body = "".join(selected)
    expected = str(candidate.get("content", ""))
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != candidate.get("content_sha256"):
        return None, "current_span_digest_mismatch", None
    if body != expected:
        return None, "current_span_bytes_mismatch", None
    return body, "CURRENT", source


def _read_current_span(runtime: object, candidate: Mapping[str, object]) -> str | None:
    body, _status, _source = _read_current_span_status(runtime, candidate)
    return body


def _full_line_source_message(
        candidate: Mapping[str, object], source: str,
) -> tuple[dict[str, object], str] | None:
    """Render complete source lines from the same verified revision as a read page."""
    path = str(candidate.get("path", ""))
    revision = str(candidate.get("revision", ""))
    start, end = candidate.get("start_line"), candidate.get("end_line")
    if (not path or not _SHA256.fullmatch(revision)
            or type(start) is not int or type(end) is not int):
        return None
    if "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest() != revision:
        return None
    lines = source.splitlines(keepends=True)
    if not 1 <= start <= end <= len(lines):
        return None
    complete = "".join(lines[start - 1:end])
    raw = complete.encode("utf-8")
    if len(raw) > _MAX_CANDIDATE_BYTES:
        return None
    full_candidate = {
        **dict(candidate),
        "content": complete,
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
        "start_column": 0,
        "end_column": None,
        "clip_reason": None,
    }
    try:
        from kernel_request_source_window import (
            _MAX_PAGE_CHARS,
            _MAX_PAGE_LINES,
            _render_ranges,
            project_source_window,
        )

        if len(raw) > _MAX_PAGE_CHARS or end - start + 1 > _MAX_PAGE_LINES:
            return None
        window = _render_ranges(
            lines,
            [{"start_line": start, "end_line": end}],
            path=path,
            revision=revision,
            total_lines=len(lines),
        )
        rendered = project_source_window(window)
    except (ImportError, TypeError, ValueError):
        return None
    if not isinstance(rendered, str) or not _source_message(rendered, full_candidate):
        return None
    return full_candidate, rendered


def _refresh_current_source_snapshot(
        runtime: object, path: str, revision: str, source: str,
) -> str:
    """Align final source verification with the exact workspace bytes just proven."""
    raw = source.encode("utf-8")
    if "sha256:" + hashlib.sha256(raw).hexdigest() != revision:
        return "revision_mismatch"
    frame_sources = getattr(runtime, "_empirical_frame_sources", None)
    if not isinstance(frame_sources, dict):
        return "source_map_unavailable"
    previous = frame_sources.get(path)
    previous = previous if isinstance(previous, Mapping) else {}
    previous_content = previous.get("content")
    if (previous.get("revision") == revision and previous_content == source):
        return "already_current"
    row = dict(previous)
    row.update({"revision": revision, "full": True, "content": source})
    row.setdefault("reference", f"frame:{path}@{revision}")
    try:
        from kernel_read_fixes import source_header
        row.setdefault("header", source_header(path, role="working"))
    except (ImportError, AttributeError, TypeError, ValueError):
        row.setdefault("header", f"[Current source: {path}]")
    frame_sources[path] = row
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        previous_digest = (
            hashlib.sha256(previous_content.encode("utf-8")).hexdigest()
            if isinstance(previous_content, str) else None
        )
        emit(
            "read_reuse_source_snapshot_reconciled",
            producer="kernel_read_reuse_visibility",
            path=path,
            revision=revision,
            previous_content_sha256=previous_digest,
            current_content_sha256=hashlib.sha256(raw).hexdigest(),
            reason="verified_current_read_span",
            dispatch=False,
        )
    return "reconciled"


def proven_deferred_source_paths(runtime: object, paths: Sequence[object]) -> set[str]:
    """Return deferred paths whose required units are covered by current read spans."""
    plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        return set()
    raw_units = plan.get("units", ())
    units = raw_units if isinstance(raw_units, Sequence) \
        and not isinstance(raw_units, (str, bytes, bytearray)) else ()
    revisions = plan.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    raw_candidates = getattr(runtime, "_read_reuse_visibility_candidates", ())
    candidates = raw_candidates if isinstance(raw_candidates, list) else ()
    proven: set[str] = set()
    for raw_path in paths:
        path = _relative(raw_path)
        if not path:
            continue
        required = [
            unit for unit in units
            if isinstance(unit, Mapping) and _relative(unit.get("path")) == path
        ]
        if not required:
            continue
        covered = True
        for unit in required:
            start, end = unit.get("start_line"), unit.get("end_line")
            revision = str(unit.get("revision") or revisions.get(path) or "")
            if (type(start) is not int or type(end) is not int or start < 1 or end < start
                    or not _SHA256.fullmatch(revision)):
                covered = False
                break
            exact_match = False
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_start = candidate.get("start_line")
                candidate_end = candidate.get("end_line")
                if (
                    _relative(candidate.get("path")) == path
                    and candidate.get("revision") == revision
                    and type(candidate_start) is int and type(candidate_end) is int
                    and candidate_start <= start and candidate_end >= end
                    and _read_current_span(runtime, candidate) is not None
                ):
                    exact_match = True
                    break
            if not exact_match:
                covered = False
                break
        if covered:
            proven.add(path)
    return proven


def _notice_descriptor(message: object) -> dict[str, object] | None:
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    path_match = _REUSE_PATH.search(content)
    if path_match is None:
        return None
    descriptor: dict[str, object] = {"path": _relative(path_match.group(1))}
    revision = re.search(r"\brevision=(sha256:[0-9a-f]{64})\b", content)
    if revision is not None:
        descriptor["revision"] = revision.group(1)
    line_range = _REUSE_RANGE.search(content)
    if line_range is None:
        effective = re.search(r"\beffective=(\{[^;]*\});", content)
        if effective is not None:
            try:
                value = json.loads(effective.group(1))
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            if isinstance(value, Mapping):
                start = value.get("start_line")
                end = value.get("end_line", value.get("selected_end_line"))
                if type(start) is int and type(end) is int:
                    descriptor["start_line"], descriptor["end_line"] = start, end
    else:
        descriptor["start_line"], descriptor["end_line"] = map(int, line_range.groups())
    return descriptor


def _notice_matches(message: object, candidate: Mapping[str, object]) -> bool:
    descriptor = _notice_descriptor(message)
    if descriptor is None or descriptor.get("path") != _relative(candidate.get("path")):
        return False
    revision = descriptor.get("revision")
    if revision and revision != candidate.get("revision"):
        return False
    start, end = descriptor.get("start_line"), descriptor.get("end_line")
    if start is None or end is None:
        return True
    return start == candidate.get("start_line") and end == candidate.get("end_line")


def _message_for_missing(path: str, start: object, end: object, *,
                         clip_reason: object = None, start_column: object = None,
                         end_column: object = None, byte_length: object = None) -> dict[str, object]:
    if _workspace_truth.enabled():
        line_range = (
            (start, end) if type(start) is int and type(end) is int and start >= 1 and end >= start
            else None
        )
        fact = _workspace_truth.probe_live_workspace(
            None, "receipt", path, line_range=line_range,
        )
        return {
            "role": "user",
            "content": _workspace_truth.render_read_visibility_fact(fact),
        }
    content = _workspace_truth.render_legacy_read_missing(
        path, start, end, clip_reason=clip_reason, start_column=start_column,
        end_column=end_column, byte_length=byte_length,
    )
    return {"role": "user", "content": content}


def _message_for_candidate(candidate: Mapping[str, object], *, visible: bool) -> dict[str, object]:
    path = str(candidate["path"])
    start, end = int(candidate["start_line"]), int(candidate["end_line"])
    clip_reason = candidate.get("clip_reason")
    turn = candidate.get("turn")
    turn_part = (
        f" turn={turn}" if _workspace_truth.enabled() and type(turn) is int and turn >= 1
        else ""
    )
    if visible:
        if str(clip_reason or "") in {"max_chars", "max_bytes"}:
            content = (f"[read reuse] path={path} revision={candidate['revision']}{turn_part} "
                       f"delivered span lines={start}-{end} "
                       f"columns={candidate.get('start_column', 0)}-{candidate.get('end_column')}; "
                       f"clip_reason={clip_reason}; delivered_bytes={candidate.get('byte_length')}; "
                       "those exact delivered bytes are present in this request.")
        else:
            content = (f"[read reuse] path={path} revision={candidate['revision']}{turn_part} "
                       f"lines={start}-{end}; exact delivered source bytes are present in this request.")
        return {"role": "user", "content": content}
    return _message_for_missing(
        path, start, end, clip_reason=clip_reason,
        start_column=candidate.get("start_column", 0),
        end_column=candidate.get("end_column"),
        byte_length=candidate.get("byte_length"),
    )


def _missing_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row.get(name) for name in ("path", "start_line", "end_line", "revision"))


def _missing_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "path", "start_line", "end_line", "revision", "clip_reason",
        "start_column", "end_column", "byte_length",
    )
    return {key: candidate[key] for key in fields if key in candidate}


def _dedupe_missing(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        unique[_missing_key(row)] = dict(row)
    return list(unique.values())


def _rewrite_reuse_notices(
        messages: Sequence[object], candidates: Sequence[Mapping[str, object]],
        missing_rows: Sequence[Mapping[str, object]],
) -> tuple[list[object], list[dict[str, object]]]:
    missing = [dict(row) for row in missing_rows]
    rewritten: list[object] = []
    for message in messages:
        descriptor = _notice_descriptor(message)
        if descriptor is None:
            rewritten.append(message)
            continue
        candidate = next(
            (item for item in candidates if _notice_matches(message, item)), None)
        if candidate is not None:
            already_missing = any(
                _missing_key(row) == _missing_key({
                    "path": candidate.get("path"),
                    "start_line": candidate.get("start_line"),
                    "end_line": candidate.get("end_line"),
                    "revision": candidate.get("revision"),
                })
                for row in missing
            )
            visible = not already_missing and _visible_before_panels(messages, candidate)
            rewritten.append(_message_for_candidate(candidate, visible=visible))
            if not visible:
                missing.append({
                    key: candidate[key] for key in (
                        "path", "start_line", "end_line", "revision")
                })
            continue
        row = {key: descriptor[key] for key in (
            "path", "start_line", "end_line", "revision") if key in descriptor}
        if row:
            missing.append(row)
        rewritten.append(_message_for_missing(
            str(descriptor["path"]), descriptor.get("start_line"), descriptor.get("end_line")))
    return rewritten, _dedupe_missing(missing)


def _ensure_missing_notices(
        messages: list[object], missing_rows: Sequence[Mapping[str, object]],
) -> list[object]:
    result = list(messages)
    for row in _dedupe_missing(missing_rows):
        path = str(row.get("path", ""))
        start, end = row.get("start_line"), row.get("end_line")
        line_label = (f"lines={start}-{end}" if type(start) is int and type(end) is int
                      else "line range unavailable")
        prefixes = (*_workspace_truth.READ_REUSE_MISSING_PREFIXES, "[read reuse] path=")
        if any(
            isinstance(message, Mapping)
            and str(message.get("content", "")).startswith(prefixes)
            and f"path={path} " in str(message.get("content", ""))
            and line_label in str(message.get("content", ""))
            for message in result
        ):
            continue
        result.insert(_panel_boundary(result), _message_for_missing(path, start, end))
    return result


def _append_final_visibility_notices(
        messages: Sequence[object], candidates: Sequence[Mapping[str, object]],
        missing_rows: Sequence[Mapping[str, object]],
) -> list[object]:
    """Append a fresh visibility receipt without rewriting prior conversation."""
    result = list(messages)
    additions: list[dict[str, object]] = []
    missing_keys = {_missing_key(row) for row in _dedupe_missing(missing_rows)}
    for candidate in candidates:
        key = _missing_key(candidate)
        visible = key not in missing_keys
        notice = _message_for_candidate(candidate, visible=visible)
        if notice not in result and notice not in additions:
            additions.append(notice)
    for row in _dedupe_missing(missing_rows):
        notice = _message_for_missing(
            str(row.get("path", "")), row.get("start_line"), row.get("end_line"),
            clip_reason=row.get("clip_reason"),
            start_column=row.get("start_column"),
            end_column=row.get("end_column"),
            byte_length=row.get("byte_length"),
        )
        if notice not in result and notice not in additions:
            additions.append(notice)
    result.extend(additions)
    return result


def _resolved_deferred_paths_for_plan(
        runtime: object, active_plan: object, request_plan: object,
        deferred_paths: Sequence[str],
) -> set[str]:
    if isinstance(request_plan, dict) and active_plan is request_plan:
        return set()
    return proven_deferred_source_paths(runtime, tuple(deferred_paths))


def _install_exact_source_block(
        messages: list[object], candidate: Mapping[str, object],
        source_message: Mapping[str, object], insert_at: int,
        panel_boundary: int | None,
) -> bool:
    stale_index = next((
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "user"
        and _source_range_block_position(message.get("content"), candidate) is not None
    ), None)
    if stale_index is None:
        messages.insert(insert_at, dict(source_message))
        return True
    stale_message = messages[stale_index]
    stale_content = stale_message.get("content")
    rendered_content = source_message.get("content")
    position = _source_range_block_position(stale_content, candidate)
    coverage = _COVERAGE_RANGE.search(stale_content) if isinstance(stale_content, str) else None
    if (not isinstance(stale_content, str) or not isinstance(rendered_content, str)
            or position is None or coverage is None or "\n" not in rendered_content):
        return False
    body_start, _coverage_end = position
    newline = stale_content.find("\n", coverage.end())
    suffix_start = len(stale_content) if newline < 0 else newline + 1
    rendered_tail = rendered_content.split("\n", 1)[1]
    separator = "\n" if suffix_start < len(stale_content) else ""
    repaired_message = {
        **dict(stale_message),
        "content": (
            stale_content[:body_start] + rendered_tail
            + separator + stale_content[suffix_start:]
        ),
    }
    if panel_boundary is not None and stale_index >= panel_boundary:
        messages.pop(stale_index)
        messages.insert(min(insert_at, panel_boundary), repaired_message)
    else:
        messages[stale_index] = repaired_message
    return True


def _prepare_final_visibility_core(
        capture: object, runtime: object, body: dict[str, object], *,
        max_request_bytes: int | None = None, allow_pins: bool = True,
        preserve_deferred: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Pin current delivered spans at the final request boundary."""
    from kernel_lean import enabled as lean_enabled
    append_only = lean_enabled()
    candidates_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for source in (capture, runtime):
        raw_candidates = getattr(source, "_read_reuse_visibility_candidates", None)
        if not isinstance(raw_candidates, list):
            continue
        for row in raw_candidates:
            if not isinstance(row, Mapping):
                continue
            candidate = dict(row)
            key = (candidate.get("path"), candidate.get("revision"),
                   candidate.get("start_line"), candidate.get("end_line"),
                   candidate.get("content_sha256"))
            candidates_by_key[key] = candidate
    candidates = list(candidates_by_key.values())
    dropped_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for source in (capture, runtime):
        raw_dropped = getattr(source, "_read_reuse_visibility_dropped", None)
        if not isinstance(raw_dropped, list):
            continue
        for row in raw_dropped:
            if not isinstance(row, Mapping):
                continue
            descriptor = dict(row)
            key = tuple(descriptor.get(name) for name in (
                "path", "revision", "start_line", "end_line"))
            dropped_by_key[key] = descriptor
    dropped = list(dropped_by_key.values())
    request_plan = getattr(capture, "_request_source_closure_override", None)
    active_plan = request_plan
    if not isinstance(active_plan, Mapping):
        active_plan = getattr(runtime, "_request_source_closure", None)
    deferred = active_plan.get("deferred_paths", ()) if isinstance(active_plan, Mapping) else ()
    deferred_paths = {str(path) for path in deferred} if isinstance(deferred, Sequence) \
        and not isinstance(deferred, (str, bytes, bytearray)) else set()
    resolved_deferred = set() if preserve_deferred else _resolved_deferred_paths_for_plan(
        runtime, active_plan, request_plan, tuple(deferred_paths),
    )
    if resolved_deferred:
        remaining = sorted(deferred_paths - resolved_deferred)
        if isinstance(active_plan, dict):
            active_plan["deferred_paths"] = remaining
        current_deferred = getattr(runtime, "_request_deferred_source_paths", ())
        if isinstance(current_deferred, Sequence) and not isinstance(
                current_deferred, (str, bytes, bytearray)):
            setattr(runtime, "_request_deferred_source_paths", tuple(
                path for path in current_deferred if _relative(path) not in resolved_deferred
            ))
    if preserve_deferred:
        # A deferred closure may still have useful exact editable read windows.
        # Propose those to the single fit below; CURRENT and byte checks still
        # run before pinning, and no closure obligation is discharged here.
        editable = editable_read_paths(runtime)
        candidates = [row for row in candidates
                      if row.get("path") not in deferred_paths or row.get("path") in editable]
        dropped = [row for row in dropped if row.get("path") not in deferred_paths]
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, {"candidates": [], "pins_added": 0, "missing_ranges": [], "result": True}
    if not candidates:
        working = copy.deepcopy(body)
        if append_only:
            missing = _dedupe_missing(dropped)
            working["messages"] = _append_final_visibility_notices(
                messages, (), missing,
            )
        else:
            rewritten, missing = _rewrite_reuse_notices(messages, (), dropped)
            working["messages"] = _ensure_missing_notices(rewritten, missing)
        return working, {
            "candidates": [], "pins_added": 0, "missing_ranges": missing,
            "result": not missing,
        }
    authorized = _contract_paths(runtime)
    candidates = [
        item for item in candidates
        if str(item.get("path", "")) in authorized
        and isinstance(item.get("content"), str)
        and str(item.get("revision", ""))
        and type(item.get("start_line")) is int
        and type(item.get("end_line")) is int
        and item["start_line"] >= 1 and item["end_line"] >= item["start_line"]
    ]
    candidates.sort(key=lambda item: (
        0 if _is_target(runtime, str(item["path"])) else 1,
        str(item["path"]), int(item["start_line"]), int(item["end_line"]),
    ))
    working = copy.deepcopy(body)
    current_messages = working.get("messages")
    if not isinstance(current_messages, list):
        return body, {"candidates": [], "pins_added": 0, "missing_ranges": [], "result": True}
    pinned = 0
    checked: list[dict[str, object]] = []
    candidate_statuses: list[dict[str, object]] = []
    missing: list[dict[str, object]] = [
        row for row in dropped
        if row.get("path") and type(row.get("start_line")) is int
        and type(row.get("end_line")) is int and row.get("revision")
    ]
    for candidate in candidates:
        exact, current_status, current_source = _read_current_span_status(
            runtime, candidate)
        candidate_statuses.append({
            "path": candidate.get("path"),
            "revision": candidate.get("revision"),
            "start_line": candidate.get("start_line"),
            "end_line": candidate.get("end_line"),
            "content_sha256": candidate.get("content_sha256"),
            "current_status": current_status,
            "snapshot_status": None,
        })
        if exact is None:
            missing.append(_missing_candidate(candidate))
            checked.append(candidate)
            continue
        snapshot_status = _refresh_current_source_snapshot(
            runtime, str(candidate["path"]), str(candidate["revision"]),
            str(current_source or ""),
        )
        candidate_statuses[-1]["snapshot_status"] = snapshot_status
        rendered_source = _full_line_source_message(candidate, str(current_source or ""))
        if rendered_source is None:
            missing.append(_missing_candidate(candidate))
            checked.append(candidate)
            continue
        current_candidate, rendered_content = rendered_source
        currently_visible = (
            any(isinstance(message, Mapping)
                and _source_message(message.get("content"), current_candidate)
                for message in current_messages)
            if append_only else _visible_before_panels(current_messages, current_candidate)
        )
        if currently_visible:
            checked.append(current_candidate)
            continue
        if not allow_pins:
            missing.append(_missing_candidate(candidate))
            checked.append(candidate)
            continue
        source_message = {
            "role": "user",
            "content": rendered_content,
        }
        candidate_body = copy.deepcopy(working)
        candidate_messages = candidate_body.get("messages")
        if not isinstance(candidate_messages, list):
            missing.append(_missing_candidate(candidate))
            checked.append(candidate)
            continue
        panel_boundary: int | None = None
        if append_only:
            insert_at = len(candidate_messages)
        else:
            notice_index = next((index for index, message in enumerate(candidate_messages)
                                 if _notice_matches(message, candidate)), None)
            panel_boundary = _panel_boundary(candidate_messages)
            insert_at = (
                min(notice_index, panel_boundary)
                if notice_index is not None else panel_boundary
            )
        if not _install_exact_source_block(
                candidate_messages, current_candidate, source_message,
                insert_at, panel_boundary):
            missing.append(_missing_candidate(candidate))
            checked.append(candidate)
            continue
        encoded = json.dumps(candidate_body).encode("utf-8")
        if max_request_bytes is not None and len(encoded) > max_request_bytes:
            missing.append(_missing_candidate(candidate))
            checked.append(candidate)
            continue
        working = candidate_body
        current_messages = candidate_messages
        pinned += 1
        checked.append(current_candidate)
    messages = working.get("messages")
    if isinstance(messages, list):
        if append_only:
            working["messages"] = _append_final_visibility_notices(
                messages, checked, missing,
            )
        else:
            rewritten, missing = _rewrite_reuse_notices(messages, checked, missing)
            working["messages"] = _ensure_missing_notices(rewritten, missing)
    return working, {
        "candidates": checked,
        "candidate_statuses": candidate_statuses,
        "pins_added": pinned,
        "missing_ranges": missing,
        "result": not missing,
    }


def _runtime_workspace_context(runtime: object) -> tuple[object, str | None, object]:
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    journal = getattr(runtime, "journal", None)
    revision = None
    try:
        binding = runtime.binding()
        if isinstance(binding, Mapping):
            value = binding.get("product_revision")
            revision = value if isinstance(value, str) else None
    except Exception:
        revision = None
    return root, revision, journal


def _refresh_workspace_truth_notices(
        capture: object, runtime: object, body: dict[str, object], *,
        preparation: object | None = None,
) -> dict[str, object]:
    if not _workspace_truth.enabled():
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    root, revision, journal = _runtime_workspace_context(runtime)
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("[read reuse]"):
            continue
        if ("Current workspace fact unavailable" not in content
                and "No current read receipt" not in content):
            continue
        path_match = _REUSE_PATH.search(content)
        range_match = _REUSE_RANGE.search(content)
        if path_match is None:
            continue
        requested_range = (
            (int(range_match.group(1)), int(range_match.group(2)))
            if range_match is not None else None
        )
        fact = _workspace_truth.read_current(
            root, path_match.group(1), requested_range,
            messages=messages, revision=revision, journal=journal,
        )
        messages[index] = {
            **dict(message),
            "content": _workspace_truth.render_read_visibility_fact(fact),
        }
    if preparation is not None:
        # Budget recovery may clear runtime-local deferrals while the immutable
        # request snapshot still requires its notice on the final wire.
        from kernel_request_reprepare import _source_delivery_inputs
        plan, _sources = _source_delivery_inputs(capture, runtime, preparation)
    else:
        plan = getattr(capture, "_request_source_closure_override", None)
        if not isinstance(plan, Mapping):
            plan = getattr(runtime, "_request_source_closure", None)
    deferred = plan.get("deferred_paths", ()) if isinstance(plan, Mapping) else ()
    allocation = getattr(capture, "frame_request_allocation", None)
    if allocation is not None:
        allocation.validate()
        # Historical read facts and runtime caches cannot discharge this request.
        deferred = tuple(dict.fromkeys(
            row["path"] for row in allocation.source_obligations
            if row["disposition"] == "DEFERRED"
        ))
    elif not isinstance(plan, Mapping):
        deferred = getattr(runtime, "_request_deferred_source_paths", ())
    if isinstance(deferred, Sequence) and not isinstance(deferred, (str, bytes, bytearray)):
        from kernel_request_source_window import upsert_deferred_source_notice
        upsert_deferred_source_notice(
            messages, deferred, workspace_root=root, revision=revision, journal=journal,
        )
    return body


def prepare_final_visibility(
        capture: object, runtime: object, body: dict[str, object], *,
        max_request_bytes: int | None = None, allow_pins: bool = True,
        preparation: object | None = None,
        preserve_deferred: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    refresh = getattr(runtime, "_post_edit_refresh", None)
    if isinstance(refresh, Mapping) and isinstance(refresh.get("paths"), Mapping):
        from kernel_post_edit_refresh import retire_pre_edit_source_windows
        body = retire_pre_edit_source_windows(
            body, refresh["paths"],
            protected_messages=getattr(capture, "_framefit5_newest_user_suffix", ()),
        )
    prepared, receipt = _prepare_final_visibility_core(
        capture, runtime, body, max_request_bytes=max_request_bytes,
        allow_pins=allow_pins,
        preserve_deferred=preserve_deferred,
    )
    return _refresh_workspace_truth_notices(
        capture, runtime, prepared, preparation=preparation,
    ), receipt


def _is_target(runtime: object, relative: str) -> bool:
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    target = getattr(runtime, "target", None)
    if root_value is not None and target is not None:
        try:
            return Path(target).resolve().relative_to(Path(root_value).resolve()).as_posix() == relative
        except (OSError, TypeError, ValueError):
            return False
    row = getattr(owner, "task_row", {})
    if isinstance(row, Mapping):
        entry = row.get("entry")
        if isinstance(entry, Mapping) and _relative(entry.get("path")) == relative:
            return True
        return _relative(row.get("path")) == relative
    return False


def _candidate_delivery_in_wire(
        messages: Sequence[object], candidate: Mapping[str, object],
        panel_boundary: int,
) -> tuple[bool, dict[str, object] | None]:
    visible = any(
        _source_message(message.get("content"), candidate) and index < panel_boundary
        for index, message in enumerate(messages) if isinstance(message, Mapping)
    )
    return visible, source_delivery_identity(candidate) if visible else None


def _candidate_is_predeclared_missing(
        candidate: Mapping[str, object], missing: Sequence[object],
) -> bool:
    keys = ("path", "revision", "start_line", "end_line")
    return any(
        isinstance(row, Mapping)
        and all(row.get(key) == candidate.get(key) for key in keys)
        for row in missing
    )


def verify_final_visibility(wire: bytes, check: Mapping[str, object]) -> dict[str, object]:
    candidates = check.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    try:
        body = json.loads(wire)
    except (TypeError, ValueError, UnicodeDecodeError):
        body = {}
    messages = body.get("messages") if isinstance(body, Mapping) else None
    messages = messages if isinstance(messages, list) else []
    from kernel_cache_layout_view import source_messages
    messages = source_messages(messages, unwrap_pages=True)
    missing = list(check.get("missing_ranges", ())) \
        if isinstance(check.get("missing_ranges"), list) else []
    delivered_sources: list[dict[str, object]] = []
    panel_boundary = _panel_boundary(messages)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            missing.append({"code": "invalid_source_candidate"})
            continue
        predeclared_missing = _candidate_is_predeclared_missing(candidate, missing)
        visible, identity = _candidate_delivery_in_wire(
            messages, candidate, panel_boundary,
        )
        if visible and identity is not None and not predeclared_missing:
            if identity not in delivered_sources:
                delivered_sources.append(identity)
            continue
        row = {key: candidate[key] for key in (
            "path", "start_line", "end_line", "revision") if key in candidate}
        if row not in missing:
            missing.append(row)
    return {
        "result": not missing,
        "missing_ranges": missing,
        "delivered_sources": delivered_sources,
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
    }


def _raise_boundary_read_menu_conflict(
        capture: object, paths: Sequence[str], authority: str | None,
) -> None:
    from kernel_boundary_recovery import BoundaryRecoveryMenuConflict

    menu_authority = authority or "request_menu"
    code = (
        "greenlock_menu_excludes_required_read"
        if menu_authority == "greenlock_frozen_menu"
        else "owner_menu_excludes_required_read"
    )
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if callable(emit):
        emit(
            "boundary_recovery_menu_conflict",
            producer="kernel_read_reuse_visibility",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            authority=menu_authority,
            code=code,
            path_count=len(paths),
            outcome="HARNESS_CONTRACT",
            failure_origin="harness",
            dispatch=False,
        )
    raise BoundaryRecoveryMenuConflict(
        f"{menu_authority} excludes the read route required by deferred sources",
        code=code,
    )


def _retain_capture_deferred_paths(capture: object, plan: dict[str, object]) -> None:
    previous = getattr(capture, "_request_source_closure_override", None)
    previous_paths = previous.get("deferred_paths", ()) if isinstance(previous, Mapping) else ()
    current_paths = plan.get("deferred_paths", ())
    from kernel_request_source_window import _normalized_deferred_paths

    def normalized(value: object) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return ()
        return _normalized_deferred_paths(value)

    prior = normalized(previous_paths)
    if prior:
        plan["deferred_paths"] = list(dict.fromkeys(
            (*prior, *normalized(current_paths)),
        ))


def _message_claims_source_path(content: str, path: str) -> bool:
    markers = (
        f"[Current source: {path}]",
        f"[Working source: {path}]",
        f"[Retained source: {path}]",
        f"[Observed source: {path}]",
        f"[read page] path={path} ",
        f"CURRENT_SOURCE {path} ",
        f"Observed source {path}",
        f"[read reuse] path={path} ",
    )
    if any(marker in content for marker in markers):
        return True
    try:
        from kernel_request_diet import source_window_path
        if source_window_path(content) == path:
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    try:
        from kernel_request_source_window import parse_model_source_window, parse_window
        for parser in (parse_model_source_window, parse_window):
            metadata = parser(content)
            if isinstance(metadata, Mapping):
                candidate = str(metadata.get("path", "")).replace("\\", "/")
                if candidate == path:
                    return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return False


def _drop_deferred_notice_paths(content: str, dropped: Sequence[str]) -> str | None:
    """Remove dropped paths from the deferred-read notice; preserve unrelated paths."""
    header = "[Deferred required source paths]"
    if not content.startswith(header):
        return content
    separator = "\n" if "\n" in content else r"\n"
    lines = content.split(separator)
    if len(lines) < 3:
        return content
    from kernel_request_source_window import _normalized_deferred_paths

    dropped_set = set(_normalized_deferred_paths(dropped))
    path_lines: list[str] = []
    remaining: list[str] = []
    removed = False
    for line in lines[2:]:
        stripped = line.strip()
        if stripped.startswith("- "):
            raw_path = stripped[2:].strip()
            normalized = _normalized_deferred_paths((raw_path,))
            if normalized and normalized[0] in dropped_set:
                removed = True
                continue
            path_lines.append(line)
        else:
            remaining.append(line)
    if not removed:
        return content
    if not path_lines:
        return None
    return separator.join((*lines[:2], *path_lines, *remaining)).rstrip()


def _remove_source_paths_from_plan(
        plan: dict[str, object], dropped: Sequence[str],
) -> None:
    from kernel_request_source_window import _normalized_deferred_paths

    dropped_set = set(dropped)

    def path_of(value: object) -> str:
        if isinstance(value, Mapping):
            value = value.get("path", "")
        return next(iter(_normalized_deferred_paths((value,))), "")

    raw_units = plan.get("units", ())
    units = raw_units if isinstance(raw_units, Sequence) and not isinstance(
        raw_units, (str, bytes, bytearray),
    ) else ()
    plan["units"] = [
        dict(unit) if isinstance(unit, Mapping) else unit
        for unit in units if path_of(unit) not in dropped_set
    ]
    raw_paths = plan.get("paths", ())
    if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes, bytearray)):
        plan["paths"] = [path for path in raw_paths if path_of(path) not in dropped_set]
    for name in (
        "revisions", "dependency_revisions", "identities", "adapters", "coverage",
        "delivery", "final_delivery", "source_contents",
    ):
        value = plan.get(name)
        if isinstance(value, Mapping):
            plan[name] = {
                str(key): item for key, item in value.items()
                if path_of(key) not in dropped_set
            }
    for name in ("omissions", "skipped_files", "lexical_sites"):
        value = plan.get(name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            plan[name] = [item for item in value if path_of(item) not in dropped_set]
    if path_of(plan.get("target_path", "")) in dropped_set:
        plan["target_path"] = ""
    previous_deferred = plan.get("deferred_paths", ())
    if isinstance(previous_deferred, Sequence) and not isinstance(
            previous_deferred, (str, bytes, bytearray)):
        plan["deferred_paths"] = [
            path for path in previous_deferred if path_of(path) not in dropped_set
        ]
    plan["dropped_source_paths"] = list(dict.fromkeys((
        *_normalized_deferred_paths(plan.get("dropped_source_paths", ())),
        *dropped,
    )))
    for key in (
        "delivery_proofs", "proof_failures", "integrity_omissions", "context_digest",
        "submitted_http_bytes_digest", "proof_status", "final_delivery_status",
        "submitted_payload_hash",
    ):
        plan.pop(key, None)


def _drop_unproven_source_claims(
        capture: object, plan: dict[str, object], body: dict[str, object],
        source_values: object, paths: Sequence[str],
) -> bool:
    """Remove unverified source claims when the frozen recovery menu has no read tool."""
    from kernel_request_source_window import (
        _normalized_deferred_paths, source_delivery_satisfied, verify_final_source_delivery,
    )

    raw_units = plan.get("units", ())
    units = raw_units if isinstance(raw_units, Sequence) and not isinstance(
        raw_units, (str, bytes, bytearray),
    ) else ()
    selected_paths = {
        normalized[0]
        for unit in units if isinstance(unit, Mapping)
        and (normalized := _normalized_deferred_paths((unit.get("path", ""),)))
    }
    dropped = tuple(path for path in _normalized_deferred_paths(paths) if path in selected_paths)
    if not dropped:
        return False
    dropped_set = set(dropped)
    messages: list[dict[str, object]] = []
    for raw_message in body.get("messages", ()):
        if not isinstance(raw_message, Mapping):
            continue
        message = dict(raw_message)
        content = message.get("content")
        if not isinstance(content, str):
            messages.append(message)
            continue
        if any(_message_claims_source_path(content, path) for path in dropped):
            continue
        pruned_notice = _drop_deferred_notice_paths(content, dropped)
        if pruned_notice is None:
            continue
        content = pruned_notice
        if any(match.group(1) in dropped_set for match in _REUSE_PATH.finditer(content)):
            remaining = [
                line for line in content.splitlines()
                if not any(f"[read reuse] path={path} " in line for path in dropped)
            ]
            content = "\n".join(remaining)
            if not content.strip():
                continue
            message["content"] = content
        messages.append(message)
    messages.append({
        "role": "user",
        "content": (
            "[Source proof notice] Unverified source excerpts were withheld. "
            "Do not rely on earlier excerpts; continue only from the remaining "
            "verified request context."
        ),
    })

    _remove_source_paths_from_plan(plan, dropped)

    payload = {"messages": messages, "tools": body.get("tools", [])}
    verified = verify_final_source_delivery(
        plan, messages,
        sources=source_values if isinstance(source_values, Mapping) else None,
        payload=payload,
    )
    if not source_delivery_satisfied(verified):
        return False
    body["messages"] = messages
    setattr(capture, "_request_source_closure_override", plan)
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if callable(emit):
        emit(
            "request_source_claim_dropped",
            producer="kernel_read_reuse_visibility",
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            path_count=len(dropped),
            reason="read_route_unavailable",
            proof_status=verified.get("proof_status"),
            status=verified.get("status"),
            dispatch=False,
        )
    return True


def defer_unproven_source_claims(capture: object, runtime: object,
                                preparation: object,
                                body: dict[str, object], *,
                                boundary_recovery: bool = False,
                                menu_authority: str | None = None) -> bool:
    """Defer source gaps when read is available; otherwise drop their unverified claims."""
    del runtime
    expanded_deferral = source_proof_deferral_enabled() or boundary_recovery
    raw_plan = getattr(preparation, "source_plan", None)
    raw_sources = getattr(preparation, "sources", None)
    try:
        plan_value = json.loads(raw_plan) if isinstance(raw_plan, (bytes, str)) else None
        source_values = json.loads(raw_sources) if isinstance(raw_sources, (bytes, str)) else None
    except (TypeError, ValueError):
        return False
    if not isinstance(plan_value, Mapping) or not isinstance(body.get("messages"), list):
        return False
    original = copy.deepcopy(dict(plan_value))
    from kernel_request_budget import (
        _bounded_delivery_omitted_unit_paths,
        _materialize_bounded_omission_units,
    )
    from kernel_request_source_window import (
        _executable_deferred_read_route, _normalized_deferred_paths,
        partial_source_paths_for_deferred_delivery, record_deferred_source_paths,
        source_delivery_satisfied, verify_final_source_delivery,
    )
    _retain_capture_deferred_paths(capture, original)
    omission_paths = tuple(dict.fromkeys((
        *_materialize_bounded_omission_units(original),
        *_bounded_delivery_omitted_unit_paths(original),
    )))
    initial = verify_final_source_delivery(
        copy.deepcopy(original), body["messages"],
        sources=source_values if isinstance(source_values, Mapping) else None,
        payload={"messages": body["messages"], "tools": body.get("tools", [])},
    )
    if source_delivery_satisfied(initial):
        return False
    integrity_omissions = initial.get("integrity_omissions", ())
    if (not isinstance(integrity_omissions, Sequence)
            or isinstance(integrity_omissions, (str, bytes, bytearray))
            or integrity_omissions):
        return False
    proof_failures = initial.get("proof_failures", ())
    if (not isinstance(proof_failures, Sequence)
            or isinstance(proof_failures, (str, bytes, bytearray))):
        return False
    tool_names = set()
    for tool in body.get("tools", ()) if isinstance(body.get("tools"), Sequence) else ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            tool_names.add(name)
    has_read = "read" in tool_names
    missing_paths: set[str] = set()
    eligible_failures = (
        _BOUNDARY_SOURCE_PROOF_FAILURES if boundary_recovery else
        _EXPANDED_SOURCE_PROOF_FAILURES if expanded_deferral else
        _BASE_SOURCE_PROOF_FAILURES
    )
    if proof_failures:
        for failure in proof_failures:
            if not isinstance(failure, Mapping):
                return False
            kind = str(failure.get("kind", ""))
            path = str(failure.get("path", "")).strip()
            if kind == "deferred_source_notice_unverified" and not has_read:
                raw_paths = failure.get("paths", ())
                if isinstance(raw_paths, Sequence) and not isinstance(
                        raw_paths, (str, bytes, bytearray)):
                    missing_paths.update(_normalized_deferred_paths(raw_paths))
                continue
            if kind not in eligible_failures or not path:
                return False
            missing_paths.add(path)
    else:
        # A bounded source closure can be byte-perfect yet cover only part of a
        # selected path. Defer that path only when the existing receipt proves
        # its revision-bound read route, workspace authorization, and unit handles.
        missing_paths.update(partial_source_paths_for_deferred_delivery(original, initial))
    missing_paths.update(omission_paths)
    if not missing_paths:
        return False
    selected_paths: set[str] = set()
    for unit in original.get("units", ()):
        if not isinstance(unit, Mapping):
            continue
        normalized = _normalized_deferred_paths((unit.get("path", ""),))
        if normalized:
            selected_paths.add(normalized[0])
    if not has_read:
        if not boundary_recovery:
            return False
        previous_deferred = original.get("deferred_paths", ())
        if isinstance(previous_deferred, Sequence) and not isinstance(
                previous_deferred, (str, bytes, bytearray)):
            missing_paths.update(_normalized_deferred_paths(previous_deferred))
        dropped_paths = tuple(sorted(missing_paths & selected_paths))
        if not dropped_paths:
            return False
        return _drop_unproven_source_claims(
            capture, original, body, source_values, dropped_paths,
        )
    revisions = original.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    delivery = original.get("delivery")
    delivery = delivery if isinstance(delivery, Mapping) else {}
    if not expanded_deferral and tool_names != {"read"}:
        return False
    deferred: list[str] = []
    for path in sorted(missing_paths & selected_paths):
        revision = str(revisions.get(path, "")).strip()
        row = delivery.get(path)
        route_value = row.get("retrieval_route") if isinstance(row, Mapping) else None
        if not isinstance(route_value, Mapping):
            route_value = {}
        units = [
            unit for unit in original.get("units", ())
            if isinstance(unit, Mapping) and str(unit.get("path")) == path
        ]
        handles = [
            str(unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle")
                or f"range:{unit.get('start_line', 0)}-{unit.get('end_line', 0)}")
            for unit in units if int(unit.get("start_line", 0) or 0) >= 1
            and int(unit.get("end_line", 0) or 0) >= int(unit.get("start_line", 0) or 0)
        ]
        route = dict(route_value)
        if (not _executable_deferred_read_route(path, revision, route)
                and (has_read or boundary_recovery) and revision
                and revision not in {"UNKNOWN", "None", "null", ""}
                and handles and isinstance(original.get("workspace_authorization"), Mapping)
                and original["workspace_authorization"].get("scope")
                and original["workspace_authorization"].get("generation")):
            ranges = sorted({
                (int(unit["start_line"]), int(unit["end_line"]))
                for unit in units
                if type(unit.get("start_line")) is int and type(unit.get("end_line")) is int
                and int(unit["start_line"]) >= 1 and int(unit["end_line"]) >= int(unit["start_line"])
            })
            route = {
                "status": "EXECUTABLE_BOUNDED_READ", "tool": "read",
                "path": path, "revision": revision, "unit_handles": handles,
                "ranges": [list(item) for item in ranges],
            }
        route.update({"path": path, "revision": revision, "unit_handles": handles})
        if (revision and handles and all(handles)
                and _executable_deferred_read_route(path, revision, route)):
            deferred.append(path)
    if not deferred:
        return False

    messages = [dict(message) for message in body["messages"] if isinstance(message, Mapping)]
    for path in deferred:
        prefixes = (
            f"[Current source: {path}]\n", f"[Working source: {path}]\n",
            f"[Retained source: {path}]\n", f"[Observed source: {path}]\n",
            f"[read page] path={path} ", f"CURRENT_SOURCE {path} ",
            f"Observed source {path}",
        )
        kept: list[dict[str, object]] = []
        for message in messages:
            content = str(message.get("content", ""))
            matches = content.startswith(prefixes)
            if not matches:
                try:
                    from kernel_request_diet import source_window_path
                    matches = source_window_path(content) == path
                except (ImportError, AttributeError, TypeError, ValueError):
                    matches = False
            if not matches:
                kept.append(message)
        messages = kept
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and f"[read reuse] path={path} " in content:
                message["content"] = re.sub(
                    r"\[read reuse\][^\n]*",
                    f"[read reuse] source visibility is limited in this request; path={path}.",
                    content, count=1,
                )
    if not record_deferred_source_paths(original, messages, deferred):
        return False
    verified = verify_final_source_delivery(
        original, messages,
        sources=source_values if isinstance(source_values, Mapping) else None,
        payload={"messages": messages, "tools": body.get("tools", [])},
    )
    if not source_delivery_satisfied(verified):
        return False
    body["messages"] = messages
    setattr(capture, "_request_source_closure_override", original)
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    if callable(emit):
        emit("request_source_claim_deferred", producer="kernel_read_reuse_visibility",
             request_id=getattr(capture, "request_id", None),
             step=getattr(capture, "step", None), paths=deferred,
             proof_status=verified.get("proof_status"),
             status=verified.get("status"), dispatch=False)
    return True


def clear_candidates(capture: object, *, path: object | None = None) -> None:
    """Retire pins from the capture and active runtime at task end or after an edit."""
    targets = [capture]
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    if runtime is not None and runtime is not capture:
        targets.append(runtime)

    edited = None
    if path is not None:
        try:
            owner = getattr(capture, "owner", None)
            root_value = getattr(owner, "root", None)
            edited_path = Path(path)
            if root_value is not None:
                root = Path(root_value).resolve()
                edited_path = (edited_path if edited_path.is_absolute()
                               else root / edited_path).resolve()
                edited_path = edited_path.relative_to(root)
            edited = edited_path.as_posix()
        except (OSError, RuntimeError, TypeError, ValueError):
            edited = str(path).replace("\\", "/")

    for target in targets:
        for attr in (
            "_read_reuse_visibility_candidates", "_read_reuse_visibility_dropped",
        ):
            rows = getattr(target, attr, None)
            if not isinstance(rows, list):
                continue
            if edited is None:
                retained = []
            else:
                retained = [
                    row for row in rows
                    if not isinstance(row, Mapping) or _relative(row.get("path")) != edited
                ]
            try:
                setattr(target, attr, retained)
            except (AttributeError, TypeError):
                pass


def retire_changed_revision_candidates(capture: object, runtime: object) -> int:
    """Discharge read pins whose file was edited before a source replan."""
    retired = 0
    for target in (capture, runtime):
        for attr in (
            "_read_reuse_visibility_candidates", "_read_reuse_visibility_dropped",
        ):
            rows = getattr(target, attr, None)
            if not isinstance(rows, list):
                continue
            retained = []
            for row in rows:
                if isinstance(row, Mapping) and _read_current_span_status(runtime, row)[1] == "revision_changed":
                    retired += 1
                else:
                    retained.append(row)
            setattr(target, attr, retained)
    return retired


__all__ = [
    "clear_candidates", "prepare_final_visibility", "remember_read_span",
    "proven_deferred_source_paths", "verify_final_visibility",
]
