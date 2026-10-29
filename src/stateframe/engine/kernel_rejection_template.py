"""Opt-in str_replace templates for rejected source edits."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

ENV = "STATEFRAME_REJECTION_TEMPLATE"
WIRE_MARKER = "[kernel] Use str_replace for this change."
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_SNIPPET_LINES = 60
MAX_SNIPPET_BYTES = 4096
_FORBIDDEN_PARTS = frozenset({
    "evaluator", "grader", "graders", "gold", "hidden", "hidden_tests",
    "private", "backups",
})
_PYTHON_HEADER = re.compile(
    r"^[ \t]*(?:(?:async[ \t]+)?def|class)[ \t]+([A-Za-z_]\w*)\b"
)


@dataclass(frozen=True)
class _Definition:
    identity: tuple[str, str, str]
    name: str
    signature: str
    start_line: int
    end_line: int
    source: str | None


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(ENV, "off") or "off").strip().lower() == "on"


def _line_slice(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[max(0, start_line - 1):max(0, end_line)])


def _python_definitions(source: str) -> list[_Definition]:
    tree = ast.parse(source)
    result: list[_Definition] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start_line = min(
            [int(node.lineno), *(int(item.lineno) for item in node.decorator_list)]
        )
        end_line = int(getattr(node, "end_lineno", node.lineno))
        first_body_line = int(getattr(node.body[0], "lineno", node.lineno))
        signature_end = max(int(node.lineno), first_body_line - 1)
        signature = _line_slice(source, int(node.lineno), signature_end).strip()
        if not signature:
            signature = _line_slice(source, int(node.lineno), int(node.lineno)).strip()
        result.append(_Definition(
            identity=(type(node).__name__, str(node.name), ""),
            name=str(node.name),
            signature=signature[:500],
            start_line=start_line,
            end_line=end_line,
            source=_line_slice(source, start_line, end_line),
        ))
    return result


def _python_invalid_proposal_changes(
    current: list[_Definition], proposed: str
) -> list[_Definition]:
    headers: list[tuple[str, int]] = []
    for line_number, line in enumerate(proposed.splitlines(), 1):
        match = _PYTHON_HEADER.match(line)
        if match is not None:
            headers.append((match.group(1), line_number))
    counts: dict[str, int] = {}
    for name, _ in headers:
        counts[name] = counts.get(name, 0) + 1
    current_counts: dict[str, int] = {}
    for item in current:
        current_counts[item.name] = current_counts.get(item.name, 0) + 1
    present = {name for name, _ in headers if counts.get(name) == 1}
    changed = [
        item for item in current
        if item.name in present
        and current_counts.get(item.name) == 1
        and item.source is not None
        and item.source not in proposed
    ]
    return sorted(changed, key=lambda item: item.start_line)[:2]


def _parse_python_changes(current: str, proposed: str) -> list[_Definition]:
    current_defs = _python_definitions(current)
    try:
        proposed_defs = _python_definitions(proposed)
    except SyntaxError:
        return _python_invalid_proposal_changes(current_defs, proposed)

    current_counts: dict[tuple[str, str, str], int] = {}
    proposed_counts: dict[tuple[str, str, str], int] = {}
    for item in current_defs:
        current_counts[item.identity] = current_counts.get(item.identity, 0) + 1
    for item in proposed_defs:
        proposed_counts[item.identity] = proposed_counts.get(item.identity, 0) + 1
    proposed_by_identity = {item.identity: item for item in proposed_defs}
    changed = [
        item for item in current_defs
        if current_counts.get(item.identity) == 1
        and proposed_counts.get(item.identity) == 1
        and item.identity in proposed_by_identity
        and item.source != proposed_by_identity[item.identity].source
    ]
    return sorted(changed, key=lambda item: item.start_line)[:2]


def _outline_definitions(source: str, relpath: str) -> tuple[bool, list[_Definition]]:
    from mc_repo_intel_vendor.ts_dispatch import outline_source, structural_spans

    outline = outline_source(source, relpath)
    structure = structural_spans(source, relpath)
    spans = structure.get("spans")
    if (not bool(structure.get("degraded")) and isinstance(spans, list)):
        encoded = source.encode("utf-8")
        result = []
        for item in spans:
            if not isinstance(item, Mapping) or item.get("kind") not in {"function", "type"}:
                continue
            if item.get("owner") not in (None, ""):
                continue
            try:
                start = int(item["start"])
                end = int(item["end"])
                start_line = int(item["line"])
                end_line = int(item["end_line"])
                name = str(item["name"])
                signature = str(item.get("signature") or name)[:500]
                snippet = encoded[start:end].decode("utf-8")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError):
                continue
            if start < 0 or end <= start or end > len(encoded) or not name:
                continue
            result.append(_Definition(
                identity=(str(item["kind"]), name, ""),
                name=name,
                signature=signature,
                start_line=start_line,
                end_line=end_line,
                source=snippet,
            ))
        return True, result

    # A degraded outline can identify a signature, but cannot authorize a source splice.
    fallback: list[_Definition] = []
    for key, kind in (("types", "type"), ("functions", "function")):
        records = outline.get(key)
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or "")
            signature = str(item.get("signature") or "")
            try:
                start_line = int(item.get("line", 0))
                end_line = int(item.get("end_line", start_line))
            except (TypeError, ValueError):
                continue
            if not name or start_line < 1 or end_line < start_line:
                continue
            owner = str(item.get("parent") or "")
            fallback.append(_Definition(
                identity=(kind, name, owner),
                name=name,
                signature=signature[:500],
                start_line=start_line,
                end_line=end_line,
                source=None,
            ))
    return False, fallback


def _nonpython_changes(current: str, proposed: str, relpath: str) -> list[_Definition]:
    current_exact, current_defs = _outline_definitions(current, relpath)
    proposed_exact, proposed_defs = _outline_definitions(proposed, relpath)
    current_counts: dict[tuple[str, str, str], int] = {}
    proposed_counts: dict[tuple[str, str, str], int] = {}
    for item in current_defs:
        current_counts[item.identity] = current_counts.get(item.identity, 0) + 1
    for item in proposed_defs:
        proposed_counts[item.identity] = proposed_counts.get(item.identity, 0) + 1
    proposed_by_identity = {item.identity: item for item in proposed_defs}
    changed: list[_Definition] = []
    for item in current_defs:
        if (current_counts.get(item.identity) != 1
                or proposed_counts.get(item.identity) != 1
                or item.identity not in proposed_by_identity):
            continue
        candidate = proposed_by_identity[item.identity]
        if current_exact and proposed_exact:
            did_change = item.source != candidate.source
        else:
            # A degraded outline supports signature-only evidence, never a guessed body diff.
            did_change = item.signature != candidate.signature
        if did_change:
            changed.append(item)
    return sorted(changed, key=lambda item: item.start_line)[:2]


def _safe_relative_path(
    root: Path,
    path: object,
    *,
    path_allowed: Callable[[Path], bool] | None,
) -> tuple[Path, str] | None:
    if not isinstance(path, (str, Path)) or not str(path):
        return None
    try:
        resolved_root = root.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve(strict=True)
        relative = candidate.relative_to(resolved_root).as_posix()
        if any(part.casefold() in _FORBIDDEN_PARTS for part in Path(relative).parts):
            return None
        if not candidate.is_file():
            return None
        if path_allowed is not None and not path_allowed(candidate):
            return None
        if candidate.stat().st_size > MAX_SOURCE_BYTES:
            return None
        return candidate, relative
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _definition_text(item: _Definition, relpath: str) -> str:
    if (item.source is not None
            and item.end_line - item.start_line + 1 <= MAX_SNIPPET_LINES
            and len(item.source.encode("utf-8")) <= MAX_SNIPPET_BYTES):
        fence = chr(96) * 3
        if fence in item.source:
            fence += chr(96)
        text = (
            f"Exact current text of {chr(96)}{item.name}{chr(96)} in {relpath} "
            f"(copy it verbatim as old_str):\n{fence}\n{item.source}"
        )
        if not item.source.endswith("\n"):
            text += "\n"
        return text + fence
    signature = item.signature.strip().replace("\n", " ")[:500] or item.name
    return (
        f"Current definition {chr(96)}{item.name}{chr(96)} in {relpath}: "
        f"signature {chr(96)}{signature}{chr(96)}; "
        f"lines {item.start_line}-{item.end_line}."
    )


def _bounded_proposal(proposed: object) -> str | None:
    if not isinstance(proposed, str):
        return None
    try:
        return proposed if len(proposed.encode("utf-8")) <= MAX_SOURCE_BYTES else None
    except UnicodeEncodeError:
        return None


def _definitions_for_source(
    current_path: Path,
    relpath: str,
    proposed: str,
) -> tuple[str, list[_Definition]] | None:
    try:
        current = current_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    try:
        if current_path.suffix.casefold() == ".py":
            changed = _parse_python_changes(current, proposed)
        else:
            changed = _nonpython_changes(current, proposed, relpath)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, SyntaxError):
        return None
    return (relpath, changed) if changed else None


def _definition_changes(
    root: Path,
    path: object,
    proposed: object,
    *,
    path_allowed: Callable[[Path], bool] | None = None,
) -> tuple[str, list[_Definition]] | None:
    bounded = _bounded_proposal(proposed)
    if bounded is None:
        return None
    safe = _safe_relative_path(root, path, path_allowed=path_allowed)
    if safe is None:
        return None
    current_path, relpath = safe
    return _definitions_for_source(current_path, relpath, bounded)


def has_definition_diff(
    root: Path,
    path: object,
    proposed: object,
    *,
    path_allowed: Callable[[Path], bool] | None = None,
) -> bool:
    """Check for a valid source-definition diff without requiring the notice flag."""
    return _definition_changes(
        root, path, proposed, path_allowed=path_allowed
    ) is not None


def build_notice(
    root: Path,
    path: object,
    proposed: object,
    *,
    path_allowed: Callable[[Path], bool] | None = None,
) -> str | None:
    """Build a bounded template from authorized current source only."""
    if not enabled():
        return None
    changes = _definition_changes(
        root, path, proposed, path_allowed=path_allowed
    )
    if changes is None:
        return None
    relpath, changed = changes
    return WIRE_MARKER + " " + "\n".join(
        _definition_text(item, relpath) for item in changed[:2]
    )


def annotate_rejection(
    observation: str,
    *,
    rejected: bool,
    root: Path,
    path: object,
    proposed: object,
    path_allowed: Callable[[Path], bool] | None = None,
) -> str:
    """Append one current-source template to a rejected edit result."""
    if (not enabled() or not rejected or not isinstance(observation, str)
            or WIRE_MARKER in observation):
        return observation
    notice = build_notice(
        root, path, proposed, path_allowed=path_allowed)
    return observation if not notice else observation + "\n" + notice


def wire_receipt(wire: bytes) -> dict[str, object] | None:
    """Prove the template survived into the exact serialized request body."""
    if not enabled():
        return None
    try:
        payload = json.loads(wire)
        messages = payload.get("messages") if isinstance(payload, Mapping) else None
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(messages, list):
        return None
    indexes = [
        index for index, item in enumerate(messages)
        if isinstance(item, Mapping) and isinstance(item.get("content"), str)
        and WIRE_MARKER in item["content"]
    ]
    if not indexes:
        return None
    return {
        "message_indexes": indexes[:16],
        "occurrences": len(indexes),
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
    }


def floor_notice_receipt(
        wire: bytes, *, runtime: object | None = None,
) -> dict[str, object] | None:
    """Prove the rejected-edit notice is on wire while the capability floor is active."""
    from kernel_capability_floor import enabled as capability_floor_enabled

    if not capability_floor_enabled(runtime):
        return None
    receipt = wire_receipt(wire)
    if receipt is None:
        return None
    return {**receipt, "notice_event": "rejection_template_notice_fired"}
