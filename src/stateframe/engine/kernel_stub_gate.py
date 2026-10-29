"""Kernel-owned STUB GATE: refuse `done` once while the target still has unimplemented stubs.

Measured motivation (R-147): 6/16 failed JobQueue finals declared `done` with functions still raising
NotImplementedError (parent had 7, gold 0). This check is host-observed from the current file bytes,
costs no model call, and is soft: it challenges at most once per distinct file content, so the model
can always finish (e.g. when a stub is intentional) by replying `done` again.
"""
from __future__ import annotations

import ast
import hashlib
import os

MODE_ENV = "STATEFRAME_STUB_GATE"
MAX_LISTED = 8


def stub_gate_mode() -> str:
    """Return the strictly validated feature mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _raises_not_implemented(node: ast.AST) -> bool:
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def _definitions_in_scope(body: list[ast.stmt]) -> list[ast.AST]:
    """Return def/class statements in one lexical scope, in source order."""
    found: list[ast.AST] = []
    definition_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

    def collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, definition_types):
                found.append(child)
            else:
                collect(child)

    for statement in body:
        if isinstance(statement, definition_types):
            found.append(statement)
        else:
            collect(statement)
    return found


def _live_definitions(body: list[ast.stmt]) -> list[ast.AST]:
    """Return only the last def/class binding for each name in a scope."""
    definitions = _definitions_in_scope(body)
    last: dict[str, ast.AST] = {}
    for definition in definitions:
        last[definition.name] = definition
    return [definition for definition in definitions
            if last[definition.name] is definition]


def unimplemented_functions(source: str) -> list[str]:
    """Qualified names of functions whose own body raises NotImplementedError.

    Nested functions are attributed to themselves, not their parent. Unparseable source yields []
    (the gate never blocks on a file it cannot read).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[str] = []

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for child in _live_definitions(body):
            if isinstance(child, ast.ClassDef):
                visit(child.body, f"{prefix}{child.name}.")
            else:
                name = f"{prefix}{child.name}"
                own = [n for n in ast.walk(child) if _raises_not_implemented(n)
                       and _owner(child, n) is child]
                if own:
                    found.append(name)
                visit(child.body, f"{name}.")

    visit(tree.body, "")
    return found


def duplicate_definitions(source: str) -> list[str]:
    """Qualified module/class names bound more than once by def/class statements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    duplicates: list[str] = []

    def visit(body: list[ast.stmt], prefix: str, binding_scope: bool) -> None:
        definitions = _definitions_in_scope(body)
        if binding_scope:
            counts: dict[str, int] = {}
            for definition in definitions:
                counts[definition.name] = counts.get(definition.name, 0) + 1
            seen: set[str] = set()
            for definition in definitions:
                qualified = f"{prefix}{definition.name}"
                if counts[definition.name] > 1 and qualified not in seen:
                    duplicates.append(qualified)
                    seen.add(qualified)
        for child in definitions:
            qualified = f"{prefix}{child.name}"
            if isinstance(child, ast.ClassDef):
                visit(child.body, f"{qualified}.", True)
            else:
                visit(child.body, f"{qualified}.", False)

    visit(tree.body, "", True)
    return duplicates


def _owner(root: ast.AST, target: ast.AST) -> ast.AST | None:
    """Innermost function under `root` (inclusive) that contains `target`."""
    owner = None

    def walk(node: ast.AST, current: ast.AST | None) -> bool:
        nonlocal owner
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current = node
        if node is target:
            owner = current
            return True
        return any(walk(c, current) for c in ast.iter_child_nodes(node))

    walk(root, None)
    return owner


class StubGate:
    """One soft challenge per distinct target content."""

    def __init__(self, mode: str | None = None):
        self.mode = mode if mode is not None else stub_gate_mode()
        self._challenged: set[str] = set()

    def evaluate(self, path: str, source: str) -> tuple[bool, str]:
        if self.mode != "on" or not isinstance(source, str):
            return True, ""
        stubs = unimplemented_functions(source)
        if not stubs:
            return True, ""
        key = hashlib.sha256(f"{path}\0{source}".encode()).hexdigest()
        if key in self._challenged:
            return True, ""
        self._challenged.add(key)
        listed = ", ".join(stubs[:MAX_LISTED]) + (f" (+{len(stubs) - MAX_LISTED} more)"
                                                  if len(stubs) > MAX_LISTED else "")
        return False, (f"[kernel] done not accepted yet (HOST_OBSERVED in the current {path}): "
                       f"these functions still raise NotImplementedError: {listed}. "
                       "Implement them as the brief requires. If a stub is intentional, reply done again.")
