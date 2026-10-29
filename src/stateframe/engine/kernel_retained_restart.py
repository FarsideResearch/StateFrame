"""Replay retained source and bounded read focus at a trusted worker checkpoint."""
from __future__ import annotations

import copy

from kernel_measurement import digest

SCHEMA = "retained-worker-checkpoint/1"
FOCUS_FIELDS = {"reference", "line", "column", "symbol"}


def empty_focus() -> dict:
    return {"reference": "", "line": 1, "column": 0, "symbol": ""}


def snapshot(holder) -> dict:
    working = holder.working_set
    return {
        "schema": SCHEMA,
        "configuration": {
            "delivery": holder.delivery, "read_focus": holder.read_focus,
            **({"edit_focus": True} if getattr(holder, "edit_focus", False) else {}),
            **({"lossless_paging": True} if getattr(holder, "lossless_paging", False) else {}),
            "public_identity": holder.public_references.mode,
            "working_set_capacity": working.limit if working is not None else 0,
        },
        "focus": {
            "reference": holder.requested, "line": holder.requested_start_line,
            "column": holder.requested_column, "symbol": holder.requested_symbol,
        },
        "source_cursor": len(holder.archive.seen_events),
        "source_digest": digest(holder.archive.seen_events),
        "working_set": {
            "spans": copy.deepcopy(list(working.spans.values())),
            "recent_removals": copy.deepcopy(working.removed),
            "edit_focused": sorted(working.edit_focused),
        } if working is not None else None,
    }


def _validate_focus(archive, focus: dict) -> None:
    if not isinstance(focus, dict) or set(focus) != FOCUS_FIELDS:
        raise ValueError("invalid retained focus fields")
    if not focus["reference"]:
        if focus != empty_focus():
            raise ValueError("invalid empty retained focus")
        return
    if focus["reference"] not in archive.revisions:
        raise ValueError("retained focus references missing source")
    archive.recall(focus["reference"], {}, start_line=focus["line"],
                   start_column=focus["column"], symbol=focus["symbol"], continuation=True)


def _span(archive, raw: dict) -> dict:
    fields = {"reference", "path", "revision", "start_line", "start_column",
              "max_lines", "max_chars", "content_hash", "span_id"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("invalid retained working span fields")
    value = archive.revisions.get(raw["reference"])
    if value is None or any(value[key] != raw[key] for key in ("path", "revision")):
        raise ValueError("retained working span references missing source")
    identity = {key: value for key, value in raw.items() if key not in {"reference", "span_id"}}
    if raw["span_id"] != "read-span:" + digest(identity):
        raise ValueError("retained working span identity mismatch")
    page = archive.recall(raw["reference"], {raw["path"]: raw["revision"]},
        start_line=raw["start_line"], start_column=raw["start_column"],
        max_lines=raw["max_lines"], max_chars=raw["max_chars"], continuation=True)
    if not page["content"] or digest(page["content"]) != raw["content_hash"]:
        raise ValueError("retained working span bytes mismatch")
    return copy.deepcopy(raw)


def restore(holder, saved: dict, records: list[dict], boundary: int) -> None:
    """Fold durable events without emitting another source observation or action."""
    initial = snapshot(holder)
    if holder.archive.seen_events or initial["focus"] != empty_focus():
        raise ValueError("retained holder used before restoration")
    if (not isinstance(saved, dict) or set(saved) != set(initial)
            or saved["schema"] != SCHEMA
            or saved["configuration"] != initial["configuration"]):
        raise ValueError("retained checkpoint configuration mismatch")
    if type(boundary) is not int or boundary < 1:
        raise ValueError("invalid retained checkpoint boundary")
    prefix = [event for event in records if event["sequence"] <= boundary]
    if len(prefix) != boundary:
        raise ValueError("retained checkpoint prefix is incomplete")
    for sequence, event in enumerate(prefix, 1):
        if (event.get("schema") != "stateframe-audit/1"
                or event.get("run_id") != holder.journal.run_id
                or event.get("event_id") != f"{holder.journal.run_id}:{sequence}"
                or type(event.get("sequence")) is not int or event["sequence"] != sequence):
            raise ValueError("retained checkpoint identity or sequence mismatch")
        inner = event.get("event", {})
        if (event["kind"] == "world_knowledge_event"
                and isinstance(inner, dict) and inner.get("schema") == "retained-source/1"
                and inner.get("scope") != holder.journal.run_id):
            raise ValueError("foreign retained source scope")
    # Replay into detached projections; validation failures leave live state untouched.
    from detreason.world_knowledge import WorldKnowledgeStore
    from detreason.source_archive import SourceArchive
    candidate = WorldKnowledgeStore(scope=holder.world.scope)
    limits = {key: getattr(holder.archive, key) for key in (
        "max_bytes", "max_file_bytes", "max_revisions", "max_steps", "max_events")}
    archive = SourceArchive(holder.archive.scope, **limits)
    candidate.source_archive = archive
    candidate.replay(prefix)
    focus, spans, removed, edit_focused = empty_focus(), {}, [], set()
    working = holder.working_set
    closed_steps = {}
    for event in prefix:
        kind = event["kind"]
        if kind == "action_outcome_closed":
            record = event["record"]
            closed_steps[record["step_record_id"]] = record
            if (holder.delivery and record.get("tool") != "recall_evidence"
                    and record.get("result") not in {"REJECTED_CONTRACT", "DENIED_PATH"}):
                focus = empty_focus()
        elif kind == "retained_read_focused":
            if not holder.read_focus:
                raise ValueError("read focus event in disabled profile")
            focus = {"reference": event["reference"], "line": event["start_line"],
                     "column": 0, "symbol": ""}
            value = archive.revisions.get(focus["reference"])
            if value is None or any(value[key] != event[key] for key in ("path", "revision")):
                raise ValueError("read focus differs from observed source")
            record = closed_steps.get(event.get("step_record_id"), {})
            evidence = record.get("read_evidence", {})
            binding = record.get("after_binding", {})
            observed_hashes = {**binding.get("product_files", {}), **binding.get("harness_files", {})}
            if (record.get("tool") != "read" or record.get("result") != "OK"
                    or evidence.get("producer") != "workspace_read"
                    or evidence.get("authority") != "HOST_OBSERVED_READ"
                    or evidence.get("path") != event["path"]
                    or evidence.get("file_revision") != event["revision"]
                    or evidence.get("start_line") != event["start_line"]
                    or observed_hashes.get(event["path"]) != event["revision"]):
                raise ValueError("read focus differs from closed source observation")
            _validate_focus(archive, focus)
        elif kind == "retained_edit_focused":
            if not getattr(holder, "edit_focus", False):
                raise ValueError("edit focus event in disabled profile")
            record = closed_steps.get(event.get("step_record_id"), {})
            path, revision = event["path"], event["revision"]
            value = archive.revisions.get(event["reference"])
            if (record.get("tool") not in {"edit", "str_replace", "file_create"}
                    or record.get("result") != "APPLIED"
                    or record.get("after_binding", {}).get("product_files", {}).get(path) != revision
                    or record.get("before_binding", {}).get("product_files", {}).get(path) == revision
                    or value is None or value["path"] != path or value["revision"] != revision):
                raise ValueError("edit focus differs from closed mutation")
            matching = [span_id for span_id, span in spans.items()
                        if span["reference"] == event["reference"]
                        and span["path"] == path and span["revision"] == revision]
            if len(matching) != 1:
                raise ValueError("edit focus has no retained current span")
            edit_focused.add(matching[0])
        elif kind == "retained_recall":
            focus = copy.deepcopy(event.get("selection"))
            _validate_focus(archive, focus)
            if focus["reference"] != event["reference"]:
                raise ValueError("recall selection differs from source reference")
        elif kind == "read_working_set_remembered":
            if working is None:
                raise ValueError("working span in disabled profile")
            span = _span(archive, event["span"])
            spans.pop(span["span_id"], None)
            spans[span["span_id"]] = span
        elif kind == "read_working_set_removed":
            if working is None or event["span_id"] not in spans:
                raise ValueError("working removal has no remembered span")
            span = spans[event["span_id"]]
            keys = ("span_id", "path", "revision", "start_line", "start_column")
            notice = {key: event[key] for key in keys}
            if any(notice[key] != span[key] for key in keys):
                raise ValueError("working removal differs from remembered span")
            reason = event["reason"]
            if reason not in {"capacity_eviction", "revision_changed_or_missing"}:
                raise ValueError("unknown working removal reason")
            if reason == "capacity_eviction" and (
                    len(spans) <= working.limit or next(iter(spans)) != event["span_id"]):
                raise ValueError("invalid working capacity eviction")
            spans.pop(event["span_id"])
            removed = [*removed[-7:], {**notice, "reason": reason}]
    if working is not None and len(spans) > working.limit:
        raise ValueError("retained working set exceeds capacity")
    reconstructed = {
        **initial, "focus": focus,
        "source_cursor": len(archive.seen_events),
        "source_digest": digest(archive.seen_events),
        "working_set": {"spans": list(spans.values()), "recent_removals": removed,
                        "edit_focused": sorted(edit_focused)}
                       if working is not None else None,
    }
    if reconstructed != saved:
        raise ValueError("retained checkpoint differs from journal replay")
    # Adopt validated data by ownership, preserving aliases and durable callbacks.
    archive_fields = ("blobs", "revisions", "latest", "steps", "graph", "used_bytes", "seen_events")
    world_fields = ("_facts", "_logical", "_events", "_seen_event_ids", "_observation_sequence",
                    "_invalidation_records", "_path_revisions", "_proposals", "_trace", "current_revision")
    archive_state = {name: getattr(archive, name) for name in archive_fields}
    world_state = {name: getattr(candidate, name) for name in world_fields}
    holder.archive.__dict__.update(archive_state)
    holder.world.__dict__.update(world_state)
    holder.public_references._public.clear()
    holder.public_references._internal.clear()
    holder.requested = focus["reference"]
    holder.requested_start_line, holder.requested_column = focus["line"], focus["column"]
    holder.requested_symbol = focus["symbol"]
    if working is not None:
        working.spans, working.removed = spans, removed
        working.edit_focused = edit_focused
