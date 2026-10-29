"""Opt-in host guard for destructive source rewrites and shell commands.

The guard is deliberately side-effect free.  Callers ask for a refusal before
their existing writer or subprocess path runs, then return the receipt without
changing the legacy writer when the switch is off.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from collections.abc import Mapping
from typing import Iterable


MODE_ENV = "STATEFRAME_EDIT_SHRINK_GUARD"
_SHRINK_REASON = "shrink exceeds both 50% of original lines and 20 lines absolute"
_COMMAND_PREFIX = "REJECTED_DESTRUCTIVE_COMMAND"
_SHRINK_PREFIX = "REJECTED_SHRINK"
RISK_MODE_ENV = "STATEFRAME_EDIT_RISK_POLICY"
RISK_UNKNOWN = "UNKNOWN"
RISK_ALLOW = "ALLOW"
RISK_TRIGGER = "PROVISIONAL"
MAX_DIFF_BYTES = 2_000_000
MAX_DIFF_FILES = 256


@dataclass(frozen=True)
class DiffMetrics:
    """Pure replacement-breadth arithmetic for one edit proposal."""

    added_bytes: int
    deleted_bytes: int
    added_lines: int
    deleted_lines: int
    before_bytes: int
    after_bytes: int
    before_lines: int
    after_lines: int
    replacement_ratio: float
    byte_replacement_ratio: float
    changed_lines: int
    touched_files: int
    paths: tuple[str, ...]
    cumulative_added_bytes: int
    cumulative_deleted_bytes: int
    cumulative_changed_lines: int
    changed_span_overlap: float
    changed_spans: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    existing_file: bool
    cost_bounded: bool
    risk: str

    @property
    def triggered(self) -> bool:
        return self.risk == RISK_TRIGGER

    def as_dict(self) -> dict[str, object]:
        return {
            "added_bytes": self.added_bytes,
            "deleted_bytes": self.deleted_bytes,
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "before_lines": self.before_lines,
            "after_lines": self.after_lines,
            "replacement_ratio": self.replacement_ratio,
            "byte_replacement_ratio": self.byte_replacement_ratio,
            "changed_lines": self.changed_lines,
            "touched_files": self.touched_files,
            "paths": list(self.paths),
            "cumulative_added_bytes": self.cumulative_added_bytes,
            "cumulative_deleted_bytes": self.cumulative_deleted_bytes,
            "cumulative_changed_lines": self.cumulative_changed_lines,
            "changed_span_overlap": self.changed_span_overlap,
            "changed_spans": {
                path: [list(span) for span in spans]
                for path, spans in self.changed_spans
            },
            "existing_file": self.existing_file,
            "cost_bounded": self.cost_bounded,
            "risk": self.risk,
        }


def _metric_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("edit source must be str or bytes")


def _metric_lines(value: bytes) -> list[bytes]:
    return value.splitlines(keepends=True)


def _edit_parts(value: object) -> tuple[dict[str, bytes], bool]:
    """Normalize one file or a path->bytes mapping without parsing a language."""
    if isinstance(value, Mapping):
        parts = {str(path).replace("\\", "/"): _metric_bytes(content)
                 for path, content in value.items()}
        return parts, True
    return {"<single>": _metric_bytes(value)}, True


def _prior_spans(accumulated_diff: object, paths: tuple[str, ...]) -> dict[str, tuple[tuple[int, int], ...]]:
    if not isinstance(accumulated_diff, Mapping):
        return {}
    raw = accumulated_diff.get("changed_spans", {})
    if isinstance(raw, Mapping):
        return {
            str(path): tuple(
                (int(item[0]), int(item[1]))
                for item in raw_value
                if isinstance(item, (tuple, list)) and len(item) == 2
                and isinstance(item[0], int) and isinstance(item[1], int)
            )
            for path, raw_value in raw.items()
            if isinstance(raw_value, (tuple, list))
        }
    if isinstance(raw, (tuple, list)) and len(paths) == 1:
        return {paths[0]: tuple(
            (int(item[0]), int(item[1])) for item in raw
            if isinstance(item, (tuple, list)) and len(item) == 2
        )}
    return {}


def _line_diff(before: bytes, after: bytes) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    old_lines, new_lines = _metric_lines(before), _metric_lines(after)
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = deleted = 0
    spans: list[tuple[int, int]] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        deleted += old_end - old_start
        added += new_end - new_start
        if new_end > new_start:
            spans.append((new_start, new_end))
    return added, deleted, tuple(spans)


def measure_edit(before: object, after: object, accumulated_diff: object = None) -> DiffMetrics:
    """Measure incremental and cumulative replacement breadth.

    The function is language-neutral and intentionally has no filesystem or
    evaluator dependency.  If bounded arithmetic cannot be completed, its
    result is ``risk=UNKNOWN`` rather than zero risk.
    """
    before_parts, _ = _edit_parts(before)
    after_parts, _ = _edit_parts(after)
    paths = tuple(sorted(set(before_parts) | set(after_parts)))
    prior = accumulated_diff if isinstance(accumulated_diff, DiffMetrics) else accumulated_diff
    prior_map = prior.as_dict() if isinstance(prior, DiffMetrics) else (prior if isinstance(prior, Mapping) else {})
    total_size = sum(len(before_parts.get(path, b"")) + len(after_parts.get(path, b"")) for path in paths)
    bounded = len(paths) <= MAX_DIFF_FILES and total_size <= MAX_DIFF_BYTES
    added_bytes = deleted_bytes = added_lines = deleted_lines = 0
    before_bytes = after_bytes = before_lines = after_lines = 0
    changed_paths: list[str] = []
    current_spans: dict[str, tuple[tuple[int, int], ...]] = {}
    for path in paths:
        old = before_parts.get(path, b"")
        new = after_parts.get(path, b"")
        before_bytes += len(old)
        after_bytes += len(new)
        before_lines += len(_metric_lines(old))
        after_lines += len(_metric_lines(new))
        if old == new:
            continue
        changed_paths.append(path)
        added, deleted, spans = _line_diff(old, new)
        added_lines += added
        deleted_lines += deleted
        if bounded:
            byte_added, byte_deleted = _byte_delta(old, new)
            added_bytes += byte_added
            deleted_bytes += byte_deleted
        else:
            added_bytes += max(0, len(new) - len(old))
            deleted_bytes += max(0, len(old) - len(new))
        current_spans[path] = spans
    # For bounded inputs, byte additions/deletions above are computed by exact
    # opcodes.  Unbounded inputs retain safe size bounds but cannot make a risk
    # decision.
    changed_lines = added_lines + deleted_lines
    denominator = before_lines + after_lines
    byte_denominator = before_bytes + after_bytes
    ratio = (changed_lines / denominator) if denominator else 0.0
    byte_ratio = ((added_bytes + deleted_bytes) / byte_denominator) if byte_denominator else 0.0
    previous_spans = _prior_spans(prior_map, paths)
    overlap_total = overlap_denominator = 0
    for path, spans in current_spans.items():
        for start, end in spans:
            length = max(0, end - start)
            overlap_denominator += length
            for old_start, old_end in previous_spans.get(path, ()):
                overlap_total += max(0, min(end, old_end) - max(start, old_start))
    overlap = (overlap_total / overlap_denominator) if overlap_denominator else 0.0
    cumulative_added = int(prior_map.get("cumulative_added_bytes", prior_map.get("added_bytes", 0)) or 0) + added_bytes
    cumulative_deleted = int(prior_map.get("cumulative_deleted_bytes", prior_map.get("deleted_bytes", 0)) or 0) + deleted_bytes
    cumulative_changed = int(prior_map.get("cumulative_changed_lines", prior_map.get("changed_lines", 0)) or 0) + changed_lines
    if not bounded:
        risk = RISK_UNKNOWN
    elif len(changed_paths) and before_bytes > 0 and ratio > 0.50 and changed_lines > 200:
        risk = RISK_TRIGGER
    else:
        risk = RISK_ALLOW
    return DiffMetrics(
        added_bytes=added_bytes, deleted_bytes=deleted_bytes,
        added_lines=added_lines, deleted_lines=deleted_lines,
        before_bytes=before_bytes, after_bytes=after_bytes,
        before_lines=before_lines, after_lines=after_lines,
        replacement_ratio=ratio, byte_replacement_ratio=byte_ratio,
        changed_lines=changed_lines, touched_files=len(changed_paths),
        paths=tuple(changed_paths), cumulative_added_bytes=cumulative_added,
        cumulative_deleted_bytes=cumulative_deleted, cumulative_changed_lines=cumulative_changed,
        changed_span_overlap=overlap, changed_spans=tuple(sorted(current_spans.items())),
        existing_file=before_bytes > 0,
        cost_bounded=bounded, risk=risk,
    )


def _byte_delta(before: bytes, after: bytes) -> tuple[int, int]:
    added = deleted = 0
    for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
            a=list(before), b=list(after), autojunk=False).get_opcodes():
        if tag in {"insert", "replace"}:
            added += new_end - new_start
        if tag in {"delete", "replace"}:
            deleted += old_end - old_start
    return added, deleted


def replacement_breadth_receipt(before: object, after: object, accumulated_diff: object = None) -> dict[str, object]:
    """Return the factual shadow result used by every writer seam."""
    metrics = measure_edit(before, after, accumulated_diff)
    return {
        "schema": "stateframe-edit-risk/1",
        "mode": "shadow",
        "risk": metrics.risk,
        "metrics": metrics.as_dict(),
        "admission_changed": False,
    }


def edit_admission(before: object, after: object, accumulated_diff: object = None) -> str:
    """Return the shadow admission class; no class refuses a legitimate write."""
    metrics = measure_edit(before, after, accumulated_diff)
    if not metrics.cost_bounded:
        return RISK_UNKNOWN
    return RISK_TRIGGER if metrics.triggered else RISK_ALLOW


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def risk_mode() -> str:
    value = os.environ.get(RISK_MODE_ENV, "off").strip().lower()
    if value not in {"off", "on", "shadow"}:
        raise ValueError(f"{RISK_MODE_ENV} must be off, shadow, or on")
    return value


def line_count(source: str) -> int:
    """Count logical source lines using the same convention as ``splitlines``."""
    if not isinstance(source, str):
        raise TypeError("source must be text")
    return len(source.splitlines())


def _path_value(path: object) -> Path:
    return path if isinstance(path, Path) else Path(str(path))


def _path_label(path: object, root: Path | None = None) -> str:
    value = _path_value(path)
    if root is not None:
        try:
            return value.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return value.as_posix().replace("\\", "/")


def brief_allows_shrink(public_brief: object, path: object) -> bool:
    """Recognize an explicit path-scoped rewrite/remove instruction.

    A generic mention of cleanup is not enough.  The public brief must name the
    target (or its basename) and use a destructive rewrite verb.
    """
    if not isinstance(public_brief, str) or not public_brief.strip():
        return False
    text = re.sub(r"\s+", " ", public_brief).strip().lower()
    label = _path_value(path).as_posix().replace("\\", "/").lstrip("./").lower()
    basename = _path_value(path).name.lower()
    if not label or (label not in text and basename not in text):
        return False
    return bool(re.search(r"\b(?:rewrite|rewritten|remove|removed|delete|deleted|replace|rebuild)\b", text))


def shrink_receipt(path: object, before_source: str, after_source: str, *,
                   root: Path | None = None, public_brief: object = None) -> str | None:
    """Return the exact typed shrink receipt, or ``None`` before a permitted write."""
    if not enabled():
        return None
    before_lines = line_count(before_source)
    after_lines = line_count(after_source)
    delta = before_lines - after_lines
    if (delta <= 20 or before_lines <= 0 or 2 * after_lines >= before_lines
            or brief_allows_shrink(public_brief, path)):
        return None
    return (f"{_SHRINK_PREFIX} {_path_label(path, root)} {before_lines} {after_lines} "
            f"{_SHRINK_REASON}")


def guard_write(path: object, after_source: str, *, root: Path | None = None,
                public_brief: object = None) -> str | None:
    """Check an existing file before a writer mutates it.

    New files have no prior line count and therefore cannot be a shrink.  No
    filesystem read is performed while the switch is off.
    """
    # Q7 post-green scope lock (own switch; reads only while an evidence refusal is open).
    from kernel_scope_lock import guard_path as scope_lock_guard
    held = scope_lock_guard(path, after_source)
    if held is not None:
        return held
    target = _path_value(path)
    if target.is_file() and risk_mode() != "off":
        before_source = target.read_text(encoding="utf-8", errors="replace")
        # All existing writers already cross this seam.  Record the breadth
        # metric in active runtime, while leaving admission provisional.
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            runtime = ACTIVE_RUNTIME.get()
            runtime._last_edit_risk_receipt = replacement_breadth_receipt(before_source, after_source)
        except (ImportError, AttributeError, LookupError, TypeError, ValueError):
            pass
    if not enabled():
        return None
    if not target.is_file():
        return None
    before_source = target.read_text(encoding="utf-8", errors="replace")
    return shrink_receipt(
        target, before_source, after_source, root=root,
        public_brief=public_brief,
    )


def _command_words(command: str) -> list[str]:
    try:
        words = shlex.split(command)
    except ValueError:
        return []
    if words and words[0] == "test":
        words = words[1:]
    return words


def _git_reason(words: Iterable[str]) -> str | None:
    values = list(words)
    for index, word in enumerate(values):
        if word != "git":
            continue
        tail = values[index + 1:]
        # Accept global git options such as ``-C checkout`` before the verb.
        cursor = 0
        while cursor < len(tail):
            if tail[cursor] in {"-C", "--git-dir", "--work-tree"}:
                cursor += 2
                continue
            if tail[cursor].startswith("-"):
                cursor += 1
                continue
            break
        tail = tail[cursor:]
        if not tail:
            continue
        if tail[0] == "reset" and "--hard" in tail[1:4]:
            return "git reset --hard"
        if tail[0] == "checkout" and "--" in tail[1:4]:
            return "git checkout --"
        if tail[0] == "clean":
            return "git clean"
    return None


def _rm_workspace_target(words: list[str], root: Path | None) -> bool:
    for index, word in enumerate(words):
        if word != "rm":
            continue
        cursor = index + 1
        force = recursive = False
        targets: list[str] = []
        while cursor < len(words):
            token = words[cursor]
            if token == "--":
                targets.extend(words[cursor + 1:])
                break
            if token == "--force":
                force = True
            elif token == "--recursive":
                recursive = True
            elif token.startswith("-") and token != "-":
                force = force or "f" in token[1:]
                recursive = recursive or "r" in token[1:]
            else:
                targets.append(token)
            cursor += 1
        if not (force and recursive):
            continue
        if not targets or root is None:
            return True
        workspace = root.resolve()
        for target in targets:
            if target.startswith("$"):
                return True
            candidate = Path(target)
            if not candidate.is_absolute():
                candidate = workspace / candidate
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved == workspace or workspace in resolved.parents:
                return True
    return False


def command_receipt(command: object, *, root: Path | None = None) -> str | None:
    """Reject destructive command proposals before any subprocess dispatch."""
    if not enabled():
        return None
    raw = str(command or "").strip()
    words = _command_words(raw)
    git_reason = _git_reason(words)
    reason = git_reason
    if reason is None and _rm_workspace_target(words, root):
        reason = "rm -rf targets the workspace"
    if reason is None:
        return None
    normalized = re.sub(r"\s+", " ", raw)
    return f"{_COMMAND_PREFIX} {normalized} {reason}"


destructive_command_receipt = command_receipt


__all__ = [
    "MODE_ENV",
    "brief_allows_shrink",
    "command_receipt",
    "destructive_command_receipt",
    "DiffMetrics",
    "edit_admission",
    "enabled",
    "guard_write",
    "line_count",
    "measure_edit",
    "mode",
    "risk_mode",
    "replacement_breadth_receipt",
    "shrink_receipt",
]
