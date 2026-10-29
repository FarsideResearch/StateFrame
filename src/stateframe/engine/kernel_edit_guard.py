"""Optional guardrails for class-scoped ``edit`` and live structural findings."""
from __future__ import annotations

import ast
import os
from collections.abc import Iterable

from kernel_stub_gate import duplicate_definitions, unimplemented_functions


MODE_ENV = "STATEFRAME_EDIT_GUARD"


def edit_guard_mode() -> str:
    """Return the strictly validated edit-guard mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    """Return whether the guarded edit/finding behavior is active."""
    return edit_guard_mode() == "on"


def _definition_nodes(body: Iterable[ast.stmt], prefix: str = "") -> list[tuple[str, ast.AST]]:
    definitions: list[tuple[str, ast.AST]] = []
    definition_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for statement in body:
        if not isinstance(statement, definition_types):
            continue
        qualified = f"{prefix}{statement.name}"
        definitions.append((qualified, statement))
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.extend(_definition_nodes(statement.body, qualified + "."))
    return definitions


def _last_definition_lines(source: str) -> dict[str, int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    last: dict[str, ast.AST] = {}
    for qualified, node in _definition_nodes(tree.body):
        last[qualified] = node
    return {name: int(getattr(node, "lineno", 0)) for name, node in last.items()}


def _top_level_names(tree: ast.Module) -> list[str]:
    definition_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return [statement.name for statement in tree.body
            if isinstance(statement, definition_types)]


def _module_prefix(source: str, tree: ast.Module) -> bool:
    """Whether source begins with module-only material rather than its target definition."""
    if not tree.body:
        return False
    first = tree.body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return True
    return isinstance(first, (ast.Import, ast.ImportFrom))


def edit_rejection(source: str, target: str, function_source: str) -> str | None:
    """Return a precise rejection for an unsafe top-level function/class replacement.

    ``None`` means the existing edit contract may continue.  This function is intentionally
    side-effect free so the caller can reject before writing any partial splice.
    """
    if not enabled():
        return None
    try:
        source_tree = ast.parse(source)
        replacement_tree = ast.parse(function_source)
    except SyntaxError:
        return None

    names = _top_level_names(replacement_tree)
    if len(names) > 1:
        if os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on":
            source_names = set(_top_level_names(source_tree))
            missing = [name for name in names if name not in source_names]
            if not missing:
                pass
            else:
                return ("edit REJECTED: multi-definition edit cannot apply atomically; "
                        "unapplied definitions " + ", ".join(missing)
                        + "; use `str_replace` instead")
        else:
            joined = ", ".join(names)
            return (f"edit REJECTED: function_source defines {joined}; `edit` replaces only "
                    f"`{target}`. Use `str_replace` with the existing file path for a whole-file "
                    "replacement, or edit each class separately.")
    if _module_prefix(function_source, replacement_tree):
        if os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() != "on":
            return ("edit REJECTED: function_source starts with a module docstring/imports; "
                    f"`edit` replaces only `{target}`. Use `str_replace` with the existing file "
                    "path for a whole-file replacement, or edit each class separately.")
    if os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on":
        for statement in replacement_tree.body[1:]:
            valid_alias = (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, (ast.Name, ast.Attribute))
            )
            if not valid_alias:
                try:
                    rendered = ast.unparse(statement)
                except (AttributeError, ValueError):
                    rendered = type(statement).__name__
                return ("edit REJECTED: unsupported trailing top-level statement "
                        f"`{rendered}`; use `str_replace` instead")

    target_nodes = [statement for statement in source_tree.body
                    if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and statement.name == target]
    if not target_nodes:
        return None
    if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import definition_source
        old = definition_source(source, target_nodes[0])
    else:
        old = ast.get_source_segment(source, target_nodes[0])
    if not old:
        return None
    candidate = source.replace(old, function_source, 1)
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import duplicate_decorator_rejection
        rejection = duplicate_decorator_rejection(source, candidate)
        if rejection is not None:
            return rejection
    duplicate_names = [name for name in duplicate_definitions(candidate) if "." not in name]
    if duplicate_names:
        joined = ", ".join(duplicate_names)
        return (f"edit REJECTED: the resulting file has duplicate top-level definitions "
                f"{joined}; `edit` replaces only `{target}`. Use `str_replace` with the "
                "existing file path for a whole-file replacement, or edit each class separately.")
    return None


def host_observed_findings(source: str, *, force: bool = False) -> list[str]:
    """Describe open live structural defects, using the shared stub/duplicate detectors."""
    if not enabled() and not force:
        return []
    duplicate_names = [name for name in duplicate_definitions(source) if "." not in name]
    stubs = unimplemented_functions(source)
    lines = _last_definition_lines(source)
    findings: list[str] = []
    try:
        tree = ast.parse(source)
        top_level_counts = {}
        for statement in tree.body:
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                top_level_counts[statement.name] = top_level_counts.get(statement.name, 0) + 1
    except SyntaxError:
        top_level_counts = {}

    for name in duplicate_names:
        line = lines.get(name, 0)
        count = top_level_counts.get(name, 2)
        findings.append(
            f"HOST_OBSERVED: {name} defined {count} times; Python uses the last copy at line "
            f"{line} (last-bound definition)."
        )
    for name in stubs:
        line = lines.get(name, 0)
        findings.append(
            f"HOST_OBSERVED: live {name} is the last-bound copy at line {line} and raises "
            "NotImplementedError."
        )
    return findings


__all__ = ["MODE_ENV", "edit_guard_mode", "enabled", "edit_rejection",
           "host_observed_findings"]
