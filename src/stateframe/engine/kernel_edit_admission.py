"""Typed request-bound authority for admitting edits to verified source."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from kernel_source_view import SourceView, VerifiedSpan, source_view


@dataclass(frozen=True)
class EditHandle:
    """An edit capability bound to one request and one source revision."""

    request_id: str
    path: str
    source_sha256: str
    spans: tuple[VerifiedSpan, ...]
    mode: Literal["FILE", "SPAN"]


class EditNotAdmitted(Exception):
    """An edit is outside its request-bound source authority or has stale bytes."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"edit not admitted for {path}: {reason}")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _line_count(source: bytes) -> int:
    """Count lines the way the engine does: newline bytes, plus an unterminated tail."""
    count = source.count(b"\n")
    if source and not source.endswith(b"\n"):
        count += 1
    return count


def _split_lines(source: bytes) -> list[bytes]:
    """Split on newline bytes only, keeping each terminator, so len() == _line_count()."""
    if not source:
        return []
    parts = source.split(b"\n")
    lines = [part + b"\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _span_matches(span: VerifiedSpan, lines: Sequence[bytes], line_count: int) -> bool:
    """A span is valid only if it lies inside the current file and its bytes hash to its sha."""
    if not 1 <= span.start <= span.end <= line_count:
        return False
    payload = b"".join(lines[span.start - 1:span.end])
    return _sha(payload) == span.sha256


def _merge(bounds: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge verified inclusive bounds; adjacent bounds join, so each merged run is maximal."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(bounds):
        if merged and start <= merged[-1][1] + 1:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _valid_range(touched: object, line_count: int) -> bool:
    if not isinstance(touched, tuple) or len(touched) != 2:
        return False
    start, end = touched
    if type(start) is not int or type(end) is not int:
        return False
    return 1 <= start <= end <= line_count


def _normalize_path(path: str) -> str:
    """Normalize a touched path; traversal and absolute paths are never valid write targets."""
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"invalid touched path: {path!r}")
    return os.path.normpath(path)


def issue_handles(view: SourceView, *, request_id: str) -> EditHandle | None:
    """Issue FILE authority only for complete views; future SPAN needs trusted input.

    Edit admission is the authoritative boundary. The fitter can never downgrade
    FILE to SPAN. A sha mismatch means the handle is stale. Historical receipts
    never authorize an edit.
    """
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not view.complete:
        return None
    return EditHandle(request_id, view.path, view.source_sha256, view.spans, "FILE")


def admit_edit(handle: EditHandle, *, current_source: bytes,
               touched_lines: tuple[int, int]) -> None:
    """Enforce source revision and touched-line bounds for an edit.

    Checks run in a fixed order: stale_revision, invalid_range, span_bytes_mismatch,
    file_incomplete, outside_span. Spans only enter coverage after their bytes verify.
    """
    if _sha(current_source) != handle.source_sha256:
        raise EditNotAdmitted(handle.path, "stale_revision")

    line_count = _line_count(current_source)
    if not _valid_range(touched_lines, line_count):
        raise EditNotAdmitted(handle.path, "invalid_range")

    lines = _split_lines(current_source)
    verified: list[tuple[int, int]] = []
    for span in handle.spans:
        if not _span_matches(span, lines, line_count):
            raise EditNotAdmitted(handle.path, "span_bytes_mismatch")
        verified.append((span.start, span.end))
    merged = _merge(verified)

    if handle.mode == "FILE" and merged != [(1, line_count)]:
        raise EditNotAdmitted(handle.path, "file_incomplete")

    start, end = touched_lines
    if not any(lo <= start and end <= hi for lo, hi in merged):
        raise EditNotAdmitted(handle.path, "outside_span")


def expire_handles(handles: Sequence[EditHandle], *,
                   touched_paths: Iterable[str]) -> tuple[EditHandle, ...]:
    """Drop authority over every path an unadmitted write touched; keep the rest in order."""
    touched = {_normalize_path(path) for path in touched_paths}
    return tuple(handle for handle in handles
                 if os.path.normpath(handle.path) not in touched)


_DETAIL_CAP = 300


def _missing_gaps(total: int, spans: Iterable[VerifiedSpan]) -> list[tuple[int, int]]:
    """Inclusive 1..total ranges not covered by verified spans."""
    gaps: list[tuple[int, int]] = []
    cursor = 1
    for start, end in _merge((span.start, span.end) for span in spans):
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= total:
        gaps.append((cursor, total))
    return gaps


def _refusal_detail(path: str, gaps: Sequence[tuple[int, int]], total: int) -> str:
    """Tell the model which lines to read; the ranges are cut to keep the detail capped."""
    head = f"edit needs the complete current source of {path}; read the missing lines first: "
    ranges = ", ".join(f"{start}-{end}" for start, end in gaps) or f"1-{total}"
    text = head + ranges
    if len(text) > _DETAIL_CAP:
        text = text[:_DETAIL_CAP - 3] + "..."
    return text


def admit_unit_edit(messages: Sequence[Mapping[str, object]], *, path: str, source: bytes,
                    request_id: str,
                    legacy_complete: Callable[[object], bool]) -> tuple[bool, str, str]:
    """Admit a subframe edit only from a complete, byte-verified view of the frozen request.

    Returns (ok, reason, detail); reason is the journal reason and is "" when ok.
    An empty source has nothing to verify and is admitted. legacy_complete takes one
    message content and reports whether it is a full current-source block (allocator-OFF shape).
    """
    if not source:
        return True, "", ""
    try:
        source.decode("utf-8")
    except UnicodeDecodeError:
        decodable = False
    else:
        decodable = True
    if decodable and any(legacy_complete(message.get("content"))
                         for message in messages if isinstance(message, Mapping)):
        return True, "", ""

    total = _line_count(source)
    view = source_view(messages, path=path, source=source)
    try:
        handle = issue_handles(view, request_id=request_id)
    except ValueError:
        return False, "no_request_id", _refusal_detail(path, [(1, total)], total)
    if handle is None:
        return False, "incomplete_view", _refusal_detail(
            path, _missing_gaps(total, view.spans), total)
    try:
        admit_edit(handle, current_source=source, touched_lines=(1, total))
    except EditNotAdmitted as error:
        return False, error.reason, _refusal_detail(
            path, _missing_gaps(total, view.spans), total)
    return True, "", ""
