"""Range-aware F4 source identity and coverage checks."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


def _integer(value: object, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def source_span(row: Mapping[str, object]) -> tuple[int, int, int, int] | None:
    """Return line/column coverage when a read carries a trustworthy range."""
    content = row.get("content")
    if not isinstance(content, str):
        return None
    effective = row.get("effective_range")
    effective = effective if isinstance(effective, Mapping) else row
    start = _integer(effective.get("start_line", row.get("start_line")), 1)
    if start < 1:
        return None
    end_value = effective.get(
        "end_line",
        effective.get("selected_end_line", row.get("selected_end_line", row.get("shown_end_line"))),
    )
    if end_value in (None, ""):
        line_count = max(1, len(content.splitlines()))
        end = start + line_count - 1
    else:
        end = max(start, _integer(end_value, start))
    start_column = _integer(effective.get("start_column", row.get("start_column")), 0)
    end_column = _integer(
        effective.get("end_column", row.get("shown_end_column")), 0)
    return start, end, start_column, end_column


def current_source_text(root: Path, relative_path: object) -> str | None:
    """Read a current UTF-8 source file without following symlinks or entering backups."""
    raw_path = str(relative_path or "").replace("\\", "/")
    candidate = Path(raw_path)
    if (not raw_path or "\x00" in raw_path or candidate.is_absolute()
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in candidate.parts)):
        return None
    try:
        base = root.resolve()
        probe = base
        for part in candidate.parts:
            probe = probe / part
            if probe.is_symlink():
                return None
        resolved = probe.resolve(strict=True)
        resolved.relative_to(base)
        if not resolved.is_file():
            return None
        decoded = resolved.read_bytes().decode("utf-8")
    except (OSError, RuntimeError, UnicodeError, TypeError, ValueError):
        return None
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


def exact_current_source_span(
        root: Path, relative_path: object, revision: object,
        start_line: int, end_line: int,
) -> str | None:
    """Return exact current text for a revision-bound, authorized line span."""
    raw_path = str(relative_path or "").replace("\\", "/")
    candidate = Path(raw_path)
    if (not raw_path or candidate.is_absolute()
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in candidate.parts)):
        return None
    try:
        base = root.resolve()
        current = base / candidate
        probe = base
        for part in candidate.parts:
            probe = probe / part
            if probe.is_symlink():
                return None
        resolved = current.resolve(strict=True)
        resolved.relative_to(base)
        if not resolved.is_file():
            return None
        raw = resolved.read_bytes()
        decoded = raw.decode("utf-8")
        normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
        expected = str(revision or "")
        raw_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        decoded_digest = "sha256:" + hashlib.sha256(decoded.encode("utf-8")).hexdigest()
        normalized_digest = "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if expected in {raw_digest, decoded_digest}:
            source = decoded
        elif expected == normalized_digest:
            # Path.read_text (used by source planning) applies universal newline decoding.
            source = normalized
        else:
            return None
        lines = source.splitlines(keepends=True)
    except (OSError, RuntimeError, UnicodeError, TypeError, ValueError):
        return None
    if (type(start_line) is not int or type(end_line) is not int
            or start_line < 1 or end_line < start_line or end_line > len(lines)):
        return None
    return "".join(lines[start_line - 1:end_line])


def source_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    path = str(row.get("path") or "")
    revision = str(row.get("revision") or row.get("file_revision") or "")
    span = source_span(row)
    if span is not None:
        return (path, revision, *span)
    content = row.get("content")
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:24]
    return (path, revision, "unknown-range", digest)


def covers(existing: Mapping[str, object], requested: Mapping[str, object]) -> bool:
    """True only when current bytes from one revision cover requested bytes."""
    if (str(existing.get("path") or "") != str(requested.get("path") or "")
            or str(existing.get("revision") or existing.get("file_revision") or "")
            != str(requested.get("revision") or requested.get("file_revision") or "")):
        return False
    left, right = source_span(existing), source_span(requested)
    if left is None or right is None:
        return source_identity(existing) == source_identity(requested)
    return (left[0] <= right[0] and left[1] >= right[1]
            and (left[0] != right[0] or left[2] <= right[2]))


class SourceRangeLedger:
    """Small per-render ledger that suppresses only duplicate/covered source."""

    def __init__(self) -> None:
        self._delivered: list[Mapping[str, object]] = []

    def covered(self, row: Mapping[str, object]) -> bool:
        return any(covers(existing, row) for existing in self._delivered)

    def remember(self, row: Mapping[str, object]) -> None:
        self._delivered.append(row)


def range_notice(row: Mapping[str, object]) -> str:
    span = source_span(row)
    if span is None:
        return ""
    start, end, start_column, end_column = span
    return (f"\nCoverage: lines {start}-{end}, columns {start_column}-{end_column}; "
            "current bytes are delivered for this range.")


def delivered_view(row: Mapping[str, object]) -> tuple[str, str] | None:
    """Content and coverage notice that claim exactly the complete lines delivered.

    Live benchmark r2AR (batch 1028, reproduced offline): a retained read clipped at
    6,000 characters still claimed "lines 1-120 ... current bytes are delivered";
    the final-wire verifier rejected the range (source_range_bytes_mismatch) and
    the request died HARNESS_CONTRACT. A clipped page now claims only its complete
    lines; the unclipped case is byte-identical to ``content + range_notice(row)``.
    Returns None when no whole line can be claimed (the block must not be rendered
    as current source).
    """
    content = row.get("content")
    if not isinstance(content, str):
        return None
    span = source_span(row)
    if span is None:
        return content, ""
    start, end, start_column, _end_column = span
    lines = content.splitlines(keepends=True)
    terminated = content.endswith(("\n", "\r"))
    total = row.get("total_lines")
    eof_tail = (not terminated and type(total) is int
                and start + len(lines) - 1 == total)
    if (start_column == 0 and len(lines) == end - start + 1
            and (terminated or eof_tail)):
        return content, range_notice(row)
    if start_column != 0:
        return None
    keep = len(lines) if (terminated or eof_tail) else len(lines) - 1
    keep = min(keep, end - start + 1)
    if keep < 1:
        return None
    return ("".join(lines[:keep]),
            f"\nCoverage: lines {start}-{start + keep - 1}, columns 0-0; "
            "current bytes are delivered for this range.")


__all__ = [
    "SourceRangeLedger", "covers", "current_source_text", "delivered_view", "exact_current_source_span",
    "range_notice", "source_identity", "source_span",
]
