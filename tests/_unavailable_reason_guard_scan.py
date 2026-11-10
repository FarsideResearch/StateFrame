"""Fast test-side implementation of the unavailable-event source scan."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from _class_guard_scanners import (
    UnavailableScan,
    _call_event_name,
    _call_has_reason,
    _function_scope,
    _is_event_field,
    _parse_file,
    _qualified_name,
    _scope_nodes,
)


_EVENT_KEYWORDS = {"event", "kind", "name", "type"}


def _may_resolve_unavailable(value: ast.AST) -> bool:
    if isinstance(value, ast.Name):
        return True
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str) and value.value.endswith("_unavailable")
    if isinstance(value, ast.Dict):
        return any(
            _is_event_field(key) and _may_resolve_unavailable(item)
            for key, item in zip(value.keys, value.values)
        )
    return False


def _possible_event_values(call: ast.Call) -> tuple[ast.AST, ...]:
    name = _qualified_name(call.func).rsplit(".", 1)[-1]
    if name == "append":
        if not isinstance(call.func, ast.Attribute) or not call.args:
            return ()
        receiver = call.func.value
        collection = (receiver.id if isinstance(receiver, ast.Name)
                      else receiver.attr if isinstance(receiver, ast.Attribute) else "")
        return (call.args[0],) if collection in {"events", "event_log", "journal_events"} else ()
    if name not in {"emit", "_emit", "write"}:
        return ()
    values = list(call.args[:2])
    values.extend(item.value for item in call.keywords if item.arg in _EVENT_KEYWORDS)
    return tuple(values)


def _event_value_needs_scope(value: ast.AST) -> bool:
    if isinstance(value, ast.Name):
        return True
    if isinstance(value, ast.Dict):
        return any(
            _is_event_field(key) and _event_value_needs_scope(item)
            for key, item in zip(value.keys, value.values)
        )
    return False


def _event_needs_scope(values: tuple[ast.AST, ...]) -> bool:
    return any(_event_value_needs_scope(value) for value in values)


def _reason_needs_scope(call: ast.Call) -> bool:
    if any(item.arg == "reason" and isinstance(item.value, ast.Name)
           for item in call.keywords):
        return True
    for value in (*call.args, *(item.value for item in call.keywords)):
        if isinstance(value, ast.Name):
            return True
        if isinstance(value, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "reason"
                and isinstance(item, ast.Name)
                for key, item in zip(value.keys, value.values)):
            return True
    return False


def scan_unavailable_events_fast(paths: Iterable[Path], *, root: Path) -> UnavailableScan:
    """Keep the guard's semantics while walking each parsed tree only once."""
    scanned = with_reason = 0
    violations: list[str] = []
    root = root.resolve()
    for path in sorted(Path(item) for item in paths):
        resolved = path.resolve()
        if "_unavailable" not in resolved.read_text(encoding="utf-8"):
            continue
        stat = resolved.stat()
        tree = _parse_file(str(resolved), stat.st_mtime_ns, stat.st_size)

        # ast.walk is breadth-first. Building its ordered node list alongside the
        # parent map avoids the scanner's separate full-tree walks for each.
        nodes = [tree]
        parents: dict[ast.AST, ast.AST] = {}
        for parent in nodes:
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
                nodes.append(child)

        label = resolved.relative_to(root).as_posix()
        scope_cache: dict[ast.AST | None, list[ast.AST]] = {}
        for call in (node for node in nodes if isinstance(node, ast.Call)):
            event_values = _possible_event_values(call)
            if not event_values or not any(_may_resolve_unavailable(value)
                                           for value in event_values):
                continue
            scope = _function_scope(call, parents)
            if _event_needs_scope(event_values) or _reason_needs_scope(call):
                if scope not in scope_cache:
                    scope_cache[scope] = nodes if scope is None else _scope_nodes(scope, scope)
                scope_nodes = scope_cache[scope]
            else:
                scope_nodes = []
            event = _call_event_name(call, scope_nodes)
            if event is None:
                continue
            scanned += 1
            function = getattr(scope, "name", "<module>")
            site = f"{label}:{function}:{call.lineno} ({event})"
            if _call_has_reason(call, scope_nodes):
                with_reason += 1
            else:
                violations.append(site)
    return UnavailableScan(scanned, with_reason, tuple(violations))
