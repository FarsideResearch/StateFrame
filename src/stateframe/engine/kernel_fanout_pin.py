"""Select current source and failing-test windows for a partial fan-out."""
from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from kernel_fanout_continuation import (
    MAX_PIN_BYTES, MAX_SOURCE_EXCERPT_BYTES, MAX_SOURCE_SCAN_BYTES,
    _bytes, _failure_for, _paths, _relative_path,
    _selectors, _truncate_utf8,
)


def _source_file(root: Path | None, relative: object) -> tuple[Path, bytes] | None:
    path_text = _relative_path(relative)
    if root is None or path_text is None:
        return None
    try:
        if root.is_symlink():
            return None
        candidate = root
        for part in PurePosixPath(path_text).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                return None
        root_resolved = root.resolve(strict=True)
        path = root_resolved.joinpath(*PurePosixPath(path_text).parts)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
            return None
        if resolved.stat().st_size > MAX_SOURCE_SCAN_BYTES:
            return None
        return resolved, resolved.read_bytes()
    except (OSError, RuntimeError, ValueError):
        return None


def _top_level_blocks(source: str) -> tuple[list[str], list[tuple[str, ast.AST]]]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return lines, []
    blocks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            blocks.append((node.name, node))
    return lines, blocks


def _block_text(lines: Sequence[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        return ""
    return "\n".join(lines[start - 1:end]).rstrip()


def _failure_function(lines: Sequence[str], nodeid: object, location: object) -> ast.AST | None:
    source = "\n".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    name = str(nodeid or "").split("::")[-1]
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", name)
    desired = match.group(1) if match else ""
    line_match = re.search(r":(\d+)(?::|$)", str(location or ""))
    line_number = int(line_match.group(1)) if line_match else None
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    exact = next((node for node in candidates if node.name == desired), None)
    if exact is not None:
        return exact
    if line_number is not None:
        return next((node for node in candidates
                     if node.lineno <= line_number <= (node.end_lineno or node.lineno)), None)
    return None


def _call_names(node: ast.AST) -> set[str]:
    result = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        function = item.func
        name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
        if name:
            result.add(name)
    return result


def _identifier_tokens(text: str) -> set[str]:
    parts = (part for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text)
             for part in token.split("_"))
    return {part.lower() for part in parts
            if len(part) > 2 and part.lower() not in {"assert", "quote", "expected", "observed"}}


def _source_windows(
        source: str, test_source: str | None, failure: Mapping[str, object],
) -> tuple[list[tuple[int, int, str]], str]:
    source_lines, source_blocks = _top_level_blocks(source)
    test_lines = test_source.splitlines() if test_source is not None else []
    test_node = _failure_function(test_lines, failure.get("nodeid"), failure.get("location")) if test_lines else None
    assertion = str(failure.get("assertion") or "")
    tokens = _identifier_tokens(assertion)
    definitions = {name: node for name, node in source_blocks}
    reachable: dict[str, int] = {}
    if test_node is not None:
        roots = _call_names(test_node) & definitions.keys()
        frontier = [(name, 0) for name in roots]
        while frontier:
            name, depth = frontier.pop(0)
            if name in reachable or depth > 5:
                continue
            reachable[name] = depth
            frontier.extend((child, depth + 1) for child in _call_names(definitions[name])
                            if child in definitions and child not in reachable)
    test_calls = _call_names(test_node) if test_node is not None else set()
    preferred = {name for name in reachable if name in test_calls}
    preferred.update(name for name in reachable if tokens & _identifier_tokens(name))
    if not preferred:
        preferred.update(name for name, _node in source_blocks if tokens & _identifier_tokens(name))
    if not preferred and source_blocks:
        preferred.add(source_blocks[0][0])

    ordered = sorted((name for name, _node in source_blocks if name in preferred),
                     key=lambda name: source_blocks.index((name, definitions[name])))
    chosen = []
    for name in ordered:
        node = definitions[name]
        text = _block_text(source_lines, node)
        if text:
            chosen.append((node.lineno, node.end_lineno or node.lineno, text))
    if not chosen and source_lines:
        line_match = re.search(r":(\d+)(?::|$)", str(failure.get("location") or ""))
        center = int(line_match.group(1)) if line_match else 1
        start, end = max(1, center - 5), min(len(source_lines), center + 5)
        chosen.append((start, end, "\n".join(source_lines[start - 1:end])))

    const_names: set[str] = set()
    if chosen:
        try:
            tree = ast.parse(source)
            selected_nodes = [definitions[name] for name in ordered if name in definitions]
            referenced = {node.id for selected in selected_nodes for node in ast.walk(selected)
                          if isinstance(node, ast.Name) and node.id.isupper()}
            for node in tree.body:
                targets = (node.targets if isinstance(node, ast.Assign) else
                           [node.target] if isinstance(node, ast.AnnAssign) else [])
                names = {target.id for target in targets if isinstance(target, ast.Name)}
                const_names.update(names & referenced)
        except SyntaxError:
            pass
        try:
            module_nodes = ast.parse(source).body if source_blocks else ()
        except SyntaxError:
            module_nodes = ()
        for node in module_nodes:
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, ast.AnnAssign) else [])
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & const_names:
                text = _block_text(source_lines, node)
                if text:
                    chosen.append((node.lineno, node.end_lineno or node.lineno, text))
    chosen.sort(key=lambda item: item[0])

    test_window = ""
    if test_node is not None:
        test_window = _block_text(test_lines, test_node)
        if not test_window and test_lines:
            start = max(1, test_node.lineno - 3)
            end = min(len(test_lines), (test_node.end_lineno or test_node.lineno) + 3)
            test_window = "\n".join(test_lines[start - 1:end])
    elif test_lines:
        line_match = re.search(r":(\d+)(?::|$)", str(failure.get("location") or ""))
        center = int(line_match.group(1)) if line_match else 1
        start, end = max(1, center - 4), min(len(test_lines), center + 4)
        test_window = "\n".join(test_lines[start - 1:end])
    return chosen, test_window


def _excerpt_from_windows(windows: Sequence[tuple[int, int, str]], assertion: str) -> str:
    if not windows:
        return "source unavailable"
    tokens = _identifier_tokens(assertion)
    attributes = set()
    attribute_leads = set()
    for name in re.findall(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)", assertion):
        attributes.update(_identifier_tokens(name))
        attribute_leads.add(name.split("_", 1)[0].lower())
    function_rows = [row for row in windows if row[2].lstrip().startswith(("def ", "async def "))]
    chosen = function_rows[0] if function_rows else windows[-1]
    best_score = (-1, -1, -1)
    for row in function_rows:
        name_match = re.search(r"(?:async )?def\s+([A-Za-z_][A-Za-z0-9_]*)", row[2])
        if name_match:
            name_tokens = _identifier_tokens(name_match.group(1))
            score = (len(attribute_leads & name_tokens),
                     len(attributes & name_tokens), len(tokens & name_tokens))
            if score > best_score:
                best_score = score
                chosen = row
    preceding = [row for row in windows if row[0] < chosen[0] and row[2].split("=", 1)[0].strip().isupper()]
    text = "\n".join([*(row[2] for row in preceding[-1:]), chosen[2]])
    return _truncate_utf8(text, MAX_SOURCE_EXCERPT_BYTES)


def source_revision_and_pin(
        disposition: Mapping[str, object], units: Sequence[Mapping[str, object]],
        remaining: Sequence[str], root_value: str | Path | None,
) -> tuple[str, str, str, list[dict[str, object]]]:
    if not remaining:
        return "", "", "", []
    root = Path(root_value) if isinstance(root_value, (str, Path)) else None
    failures: dict[str, Mapping[str, object] | None] = {
        unit_id: _failure_for(unit_id, disposition) for unit_id in remaining
    }
    pin_sections: list[str] = ["[PINNED FANOUT SOURCE WINDOWS]"]
    revision_line = "source_revision=unavailable"
    excerpt_line = "source unavailable"
    events: list[dict[str, object]] = []
    total_source_and_test_bytes = 0

    def degrade_if_oversized() -> bool:
        requested = _bytes("\n".join(pin_sections))
        if requested <= MAX_PIN_BYTES:
            return False
        events.append({
            "kind": "fanout_pin_degraded",
            "reason": "pin_exceeds_budget",
            "unit_ids": list(remaining),
            "requested_bytes": requested,
            "source_and_test_bytes": total_source_and_test_bytes,
            "cap_bytes": MAX_PIN_BYTES,
        })
        return True

    for unit_id in remaining:
        unit = next((row for row in units if row.get("id") == unit_id), None)
        if unit is None:
            continue
        failure = failures[unit_id] or {}
        owns = _paths(unit.get("owns"))
        selectors = _selectors(unit)
        for path_index, path_text in enumerate(owns):
            source = _source_file(root, path_text)
            if source is None:
                continue
            source_path, source_bytes = source
            try:
                source_text = source_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            test_path_text = next((candidate.split("::", 1)[0] for candidate in selectors
                                   if candidate.split("::", 1)[0].endswith(".py")), None)
            test_file = _source_file(root, test_path_text) if test_path_text else None
            test_text = None
            test_bytes = b""
            if test_file is not None:
                try:
                    test_text = test_file[1].decode("utf-8")
                    test_bytes = test_file[1]
                except UnicodeDecodeError:
                    test_text = None
            source_windows, test_window = _source_windows(source_text, test_text, failure)
            if path_index == 0:
                digest = hashlib.sha256(source_bytes).hexdigest()
                revision_line = f"source_revision=sha256:{digest}; excerpt:"
                excerpt_line = _excerpt_from_windows(source_windows, str(failure.get("assertion") or ""))
            source_ranges = ",".join(f"{start}-{end}" for start, end, _text in source_windows) or "unavailable"
            pin_sections.extend([
                f"[Current source window: {path_text}]",
                f"revision=sha256:{hashlib.sha256(source_bytes).hexdigest()} lines={source_ranges}",
            ])
            if degrade_if_oversized():
                return revision_line, _truncate_utf8(excerpt_line, MAX_SOURCE_EXCERPT_BYTES), "", events
            for _start, _end, text in source_windows:
                pin_sections.append(text)
                total_source_and_test_bytes += _bytes(text)
                if degrade_if_oversized():
                    return revision_line, _truncate_utf8(excerpt_line, MAX_SOURCE_EXCERPT_BYTES), "", events
            if test_path_text and test_file is not None and test_window:
                test_bytes = test_file[1]
                test_lines = test_text.splitlines() if test_text is not None else []
                test_node = _failure_function(test_lines, failure.get("nodeid"), failure.get("location"))
                start = getattr(test_node, "lineno", 1) if test_node is not None else 1
                end = getattr(test_node, "end_lineno", start) if test_node is not None else start
                pin_sections.extend([
                    f"[Current test window: {test_path_text}]",
                    f"revision=sha256:{hashlib.sha256(test_bytes).hexdigest()} lines={start}-{end}",
                    test_window,
                ])
                total_source_and_test_bytes += _bytes(test_window)
                if degrade_if_oversized():
                    return revision_line, _truncate_utf8(excerpt_line, MAX_SOURCE_EXCERPT_BYTES), "", events
    pin_text = "\n".join(pin_sections) if len(pin_sections) > 1 else ""
    if not pin_text:
        events.append({"kind": "fanout_pin_degraded", "reason": "source_window_unavailable",
                       "unit_ids": list(remaining), "cap_bytes": MAX_PIN_BYTES})
    elif _bytes(pin_text) > MAX_PIN_BYTES:
        events.append({"kind": "fanout_pin_degraded", "reason": "pin_exceeds_budget",
                       "unit_ids": list(remaining), "requested_bytes": _bytes(pin_text),
                       "source_and_test_bytes": total_source_and_test_bytes,
                       "cap_bytes": MAX_PIN_BYTES})
        pin_text = ""
    excerpt = _truncate_utf8(excerpt_line, MAX_SOURCE_EXCERPT_BYTES)
    return revision_line, excerpt, pin_text, events
