"""Closure/plan proposals from already frozen source maps, without disk reads."""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json


def render_frozen_span(metadata, descriptor, payload):
    """Rebuild the existing window grammar for one independently selected span.

    Splitting a multi-range window changes its continuation count and handle.
    Neither the old aggregate metadata nor its old expansion handle is valid
    for the new selected range. No workspace bytes are needed to derive them.
    """
    from kernel_request_source_window import MODEL_WINDOW_SCHEMA

    row = {"start_line": descriptor["start_line"], "end_line": descriptor["end_line"],
           "anchors": list(descriptor.get("anchors", ())),
           "trailing_newline": payload.endswith("\n")}
    path, revision, total = metadata["path"], metadata["revision"], metadata["total_lines"]
    count = row["end_line"] - row["start_line"] + 1
    if type(total) is not int or not 0 < count <= total:
        raise ValueError("frozen window exceeds declared source length")
    status = "COMPLETE" if count == total else "CONTINUATION_AVAILABLE"
    handle = ("expand:" + hashlib.sha256(json.dumps({"path": path, "revision": revision,
        "ranges": [row]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if count < total else None)
    selected = {"schema": MODEL_WINDOW_SCHEMA, "owner": metadata.get("owner") or "source_closure",
        "path": path, "revision": revision, "sha256": revision.removeprefix("sha256:"),
        "total_lines": total, "ranges": [row], "status": status, "absence": "NONE",
        "continuation": {"status": status, "omitted_lines": total - count, "expansion_handle": handle}}
    prefix = (f"[Current source: {path}]\n[Model source window]\n" +
        json.dumps(selected, sort_keys=True, separators=(",", ":")) +
        f"\n[Source window content]\n--- source lines {row['start_line']}-{row['end_line']} ---\n")
    page = prefix + payload + ("" if payload.endswith("\n") else "\n") + "[End source window]"
    return page, len(prefix.encode())


def frozen_plan_windows(runtime, existing, current_revisions):
    """Offer every exact plan unit, including units absent from legacy panels.

    Successful captures remain the source of read exposure debt. These windows
    are request-local alternatives and cannot promote a retired capture back to
    FULL merely because the same unit appears in the closure plan.
    """
    from kernel_read_receipts import DecodedSpan, ReceiptIdentity, line_digests_with_numbers
    from kernel_request_source_window import (
        _render_ranges, _safe_framefit_path, project_source_window, source_revision,
    )
    from kernel_working_set_snapshot import SourceSlot, SnapshotConstructionError

    plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        return (), {}
    plan = copy.deepcopy(dict(plan))
    declared = plan.get("revisions", {})
    contents = plan.get("source_contents", {})
    cached = copy.deepcopy(getattr(runtime, "_empirical_frame_sources", {}))
    if not isinstance(declared, Mapping) or not isinstance(contents, Mapping):
        raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
    proposed = []
    revisions = {}
    ordinal = max((slot.admission_ordinal for slot in existing), default=0)
    seen = {(slot.identity.path, slot.identity.revision, slot.identity.first_line,
             slot.identity.last_line, slot.payload) for slot in existing}
    for unit in plan.get("units", ()):
        if not isinstance(unit, Mapping):
            raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
        path, first, last = unit.get("path"), unit.get("start_line"), unit.get("end_line")
        if (not isinstance(path, str) or _safe_framefit_path(path) != path
                or type(first) is not int or type(last) is not int or first < 1 or last < first):
            raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
        source = contents.get(path)
        if source is None and isinstance(cached, Mapping):
            row = cached.get(path)
            source = row.get("content") if isinstance(row, Mapping) else None
        revision = declared.get(path)
        if (not isinstance(source, str) or source_revision(source) != revision
                or (path in current_revisions and current_revisions[path] != revision)):
            raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
        lines = source.splitlines(keepends=True)
        if last > len(lines):
            raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
        payload = "".join(lines[first - 1:last])
        identity_key = path, revision, first, last, payload
        revisions[path] = revision
        if identity_key in seen:
            continue
        seen.add(identity_key)
        descriptor = {"start_line": first, "end_line": last,
                      "anchors": list(unit.get("anchors", ()))}
        page = project_source_window(_render_ranges(lines, (descriptor,), path=path,
            revision=revision, total_lines=len(lines)), owner="source_closure")
        if not isinstance(page, str):
            raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
        payload_start = page.index(f"--- source lines {first}-{last} ---\n")
        payload_start += len(f"--- source lines {first}-{last} ---\n")
        checksum = hashlib.sha256(payload.encode()).hexdigest()
        identity = ReceiptIdentity(path, revision, first, last, None, None, checksum,
            DecodedSpan(0, len(lines[last - 1]), "model-source-window/1"))
        evidence = line_digests_with_numbers(payload, first)
        key = "closure:" + hashlib.sha256(json.dumps(identity_key).encode()).hexdigest()
        ordinal += 1
        role = str(unit.get("role") or "")
        state = "PLAN" if role in {"plan", "plan_window", "planned_edit"} else "CLOSURE"
        proposed.append(SourceSlot(key, (key,), identity, payload, page, *evidence,
            json.dumps({"path": path, "start_line": first, "end_line": last}).encode(),
            ordinal, 0, state, kind="tests" if path.startswith("tests/") else "context",
            payload_start=len(page[:payload_start].encode())))
    return tuple(proposed), revisions
