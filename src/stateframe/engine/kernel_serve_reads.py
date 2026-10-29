"""STATEFRAME_TOKENLAB_SERVE_READS (token lab v11): serve in-frame reads, and say what is in the frame.

Measured (703 ldev_05 v6a, classify.py): 5 reads of files whose current bytes were already in the request were
answered with a READ_UNCHANGED/ALREADY_PRESENT stub or a stub-sweep REFUSED_PREDISPATCH; each such turn cost a
full reasoning burst (34k completion tokens in total) while the boss simply receives the bytes and moves on.

On: (a) `read_dedup_observation` and `dispatch_refusal` both stand down, so the ordinary read path runs and the
file content is returned as a normal tool result (journaled as ``read_served_from_frame`` with what would have
answered otherwise); (b) one "[Files in frame]" user notice lists every ``[Current source: <path>]`` block in the
composed F4 request, placed right after the last source block (rank tier of a generic notice, before compaction
and fit); it describes excerpts only because ordinary reads still fetch current disk bytes for their requested
range.  Off (default): nothing here is called; requests and receipts are byte-identical.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_TOKENLAB_SERVE_READS"
MANIFEST_KEY = "stateframe_tokenlab_serve_reads"
EVENT_SERVED = "read_served_from_frame"
EVENT_OUTCOME = "read_serve_outcome"
EVENT_LISTED = "frame_files_listed"
PRODUCER = "kernel_serve_reads"
FILES_HEADER = "[Files in frame]"
SOURCE_PREFIX = "[Current source: "


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def note_served(runtime: object, path: object, *, would_have: str, predicate: str) -> bool:
    """Record a matched counterfactual admission, without claiming a read succeeded."""
    normalised = str(path or "").replace("\\", "/").lstrip("./")
    key = normalised
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return False
    pending = getattr(runtime, "_tokenlab_serve_reads_pending", None)
    if not isinstance(pending, set):
        pending = set()
        try:
            setattr(runtime, "_tokenlab_serve_reads_pending", pending)
        except (AttributeError, TypeError):
            return False
    if key in pending:
        return False
    pending.add(key)
    try:
        emit(EVENT_SERVED, producer=PRODUCER, path=normalised, would_have=str(would_have),
             predicate=str(predicate), counterfactual=True, saved_refusal=False,
             phase="admission", dispatch=False)
    except Exception:  # noqa: BLE001 - telemetry never changes the read
        pending.discard(key)
        return False
    return True


def note_read_outcome(runtime: object, path: object, result_class: object, *, bytes_read: int = 0) -> bool:
    """Close pending serve admissions with the actual ordinary-read result."""
    normalised = str(path or "").replace("\\", "/").lstrip("./")
    pending = getattr(runtime, "_tokenlab_serve_reads_pending", None)
    if not isinstance(pending, set):
        return False
    matched = {item for item in pending if item == normalised}
    if not matched:
        return False
    pending.difference_update(matched)
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        return False
    try:
        emit(EVENT_OUTCOME, producer=PRODUCER, path=normalised,
             result_class=str(result_class), bytes_read=max(0, int(bytes_read)),
             phase="read_complete", dispatch=False)
    except Exception:  # noqa: BLE001 - telemetry never changes the read
        return False
    return True


def files_in_frame_notice(messages: Sequence[Mapping[str, object]]) -> tuple[list[dict], list[str]]:
    """Insert ONE [Files in frame] notice after the last [Current source: ...] user message.

    Pure.  Returns (messages, listed paths); unchanged input + [] when there is no source block or the notice is
    already present (idempotent across sizing trials)."""
    out = [dict(m) for m in messages]
    paths: list[str] = []
    last = -1
    for index, message in enumerate(out):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        if content.startswith(FILES_HEADER):
            # Rebuild stale notices after a reducer deletes or windows a source
            # block; retaining old paths would claim delivery that is no longer
            # present in the final serialized frame.
            continue
        head = content.split("\n", 1)[0]
        source_prefix = next((prefix for prefix in (SOURCE_PREFIX, "[Working source: ")
                              if head.startswith(prefix)), None)
        if source_prefix is not None and head.endswith("]"):
            path = head[len(source_prefix):-1].strip()
            if path and path not in paths:
                paths.append(path)
            last = index
    if not paths:
        return [message for message in out if not (
            isinstance(message.get("content"), str)
            and message["content"].startswith(FILES_HEADER)
        )], []
    out = [message for message in out if not (
        isinstance(message.get("content"), str)
        and message["content"].startswith(FILES_HEADER)
    )]
    last = max(
        (index for index, message in enumerate(out)
         if message.get("role") == "user"
         and isinstance(message.get("content"), str)
         and message["content"].split("\n", 1)[0].startswith((SOURCE_PREFIX, "[Working source: "))),
        default=-1,
    )
    notice = (f"{FILES_HEADER} Source excerpts for these paths are included in this request under their "
              f"[Current source: ...] blocks; read fetches current disk content for the requested range: "
              f"{', '.join(paths)}.")
    out.insert(last + 1, {"role": "user", "content": notice})
    return out, paths


__all__ = ["EVENT_LISTED", "EVENT_OUTCOME", "EVENT_SERVED", "FILES_HEADER", "MANIFEST_KEY", "MODE_ENV", "PRODUCER",
           "enabled", "files_in_frame_notice", "mode", "note_read_outcome", "note_served"]
