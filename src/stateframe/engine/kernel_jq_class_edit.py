"""Opt-in, class-scoped AST edits for the JobQueue repair path.

The historical edit executor resolves a class name through a first-match span.  That
is unsafe for a module containing siblings, repeated class names, or decorators: the
replacement can be applied to the wrong definition or leave two decorator copies.
This module makes the target resolution explicit and returns a typed refusal whenever
the source does not identify exactly one class.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import textwrap
from dataclasses import dataclass
from typing import Mapping, Sequence


MODE_ENV = "STATEFRAME_JQ_CLASS_EDIT"
VALID_MODES = {"off", "on"}
RENDER_HEADER = "[JQ CLASS EDIT | HOST_CAPTURED]"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


@dataclass(frozen=True)
class ClassTarget:
    qualified_name: str
    node: ast.ClassDef


@dataclass(frozen=True)
class ClassEditPlan:
    """The class mechanism's result before the caller performs a guarded write."""

    handled: bool
    candidate: str | None = None
    rejection: str | None = None


def _class_targets(tree: ast.AST) -> tuple[ClassTarget, ...]:
    found: list[ClassTarget] = []

    class Collector(ast.NodeVisitor):
        stack: list[str]

        def __init__(self) -> None:
            self.stack = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            qualified = ".".join((*self.stack, node.name))
            found.append(ClassTarget(qualified, node))
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    Collector().visit(tree)
    return tuple(found)


def _matching_targets(targets: Sequence[ClassTarget], name: str) -> tuple[ClassTarget, ...]:
    if "." in name:
        return tuple(item for item in targets if item.qualified_name == name)
    return tuple(item for item in targets if item.node.name == name)


def _relative(path: pathlib.Path, root: pathlib.Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return path.as_posix() if path.is_absolute() else path.as_posix()


def _typed_refusal(
    code: str,
    path: pathlib.Path,
    target: str,
    *,
    root: pathlib.Path | None = None,
    hint: str,
    definitions: Sequence[Mapping[str, object]] = (),
) -> str:
    path_text = _relative(path, root)
    payload: dict[str, object] = {
        "alternatives": [{
            "tool": "str_replace",
            "path": path_text,
            "hint": hint[:600],
        }],
        "code": code,
        "path": path_text,
        "status": "REJECTED",
        "symbol": target,
    }
    if definitions:
        payload["definitions"] = [dict(item) for item in definitions]
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _definition_rows(targets: Sequence[ClassTarget]) -> tuple[dict[str, object], ...]:
    return tuple({
        "column": item.node.col_offset,
        "kind": "class",
        "line": item.node.lineno,
        "name": item.node.name,
        "qualified_name": item.qualified_name,
    } for item in targets)


def _line_offsets(source: str) -> tuple[int, ...]:
    offsets = [0]
    for index, character in enumerate(source):
        if character == "\n":
            offsets.append(index + 1)
    return tuple(offsets)


def _char_column(line: str, byte_column: int) -> int:
    """Convert an AST col_offset (a UTF-8 BYTE count, per the Python grammar) into
    a character index within ``line``.  Treating col_offset as a character index
    directly corrupts any line containing multi-byte UTF-8 characters before the
    target column."""
    if byte_column <= 0:
        return 0
    encoded = line.encode("utf-8")
    return len(encoded[:byte_column].decode("utf-8"))


def _offset(source: str, offsets: Sequence[int], line: int, column: int) -> int:
    line_start = offsets[line - 1]
    line_end = offsets[line] - 1 if line < len(offsets) else len(source)
    line_text = source[line_start:line_end]
    return line_start + _char_column(line_text, column)


def _decorator_start(source: str, offsets: Sequence[int], node: ast.AST) -> int:
    """Decorator expression nodes start AFTER the ``@`` token (and any whitespace
    following it), never at the ``@`` itself.  Walk left over the intervening
    whitespace to the ``@`` so decorator source, including it, is preserved."""
    pos = _offset(source, offsets, getattr(node, "lineno"), getattr(node, "col_offset"))
    cursor = pos - 1
    while cursor >= 0 and source[cursor] in " \t":
        cursor -= 1
    if cursor >= 0 and source[cursor] == "@":
        return cursor
    return pos


def _node_span(source: str, node: ast.AST, *, include_decorators: bool = False) -> tuple[int, int]:
    offsets = _line_offsets(source)
    decorators = getattr(node, "decorator_list", ())
    if include_decorators and decorators:
        start = _decorator_start(source, offsets, decorators[0])
    else:
        start = _offset(source, offsets, getattr(node, "lineno"), getattr(node, "col_offset"))
    end_line = getattr(node, "end_lineno")
    end_col = getattr(node, "end_col_offset")
    end = _offset(source, offsets, end_line, end_col)
    return (start, end)


def _source_segment(source: str, node: ast.AST, *, include_decorators: bool = False) -> str:
    start, end = _node_span(source, node, include_decorators=include_decorators)
    return source[start:end]


def _reindent(segment: str, column: int) -> str:
    """Reindent a dedented multi-line segment for splicing at ``column``.

    The FIRST line is spliced immediately after ``current[:start]``, which already
    contains that column's leading whitespace (the caller never re-adds it) -- so
    only continuation lines get the explicit ``column``-space prefix.  Prefixing
    every line double-indents the first one and can produce an unparsable nested
    replacement (or, harmlessly, an over-indented top-level one).
    """
    normalized = textwrap.dedent(segment)
    lines = normalized.splitlines()
    if not lines:
        return normalized
    prefix = " " * column
    reindented = [lines[0]] + [prefix + line if line else line for line in lines[1:]]
    return "\n".join(reindented)


def _decorator_signature(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(ast.dump(item, include_attributes=False) for item in node.decorator_list)


def _inflight_enabled() -> bool:
    import kernel_inflight_fixes
    return kernel_inflight_fixes.enabled()


def _add_missing_imports(candidate: str, imports: Sequence[ast.stmt], replacement: str) -> str:
    """Insert payload import lines the module lacks, after its last top-level import (else after docstring /
    ``__future__``).  Never removes or reorders existing imports.  An unparseable candidate is returned unchanged
    so the caller's own validity check reports it."""
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return candidate
    present = {ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))}
    missing: list[ast.stmt] = []
    for node in imports:
        text = ast.unparse(node)
        if text not in present and all(text != ast.unparse(item) for item in missing):
            missing.append(node)
    if not missing:
        return candidate

    def render(nodes: Sequence[ast.stmt]) -> str:
        return "".join(ast.unparse(node) + "\n" for node in nodes) + "\n"

    future_imports = [node for node in missing
                      if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    if future_imports:
        tree = ast.parse(candidate)
        insertion = 0
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            insertion = _node_span(candidate, tree.body[0])[1]
            while insertion < len(candidate) and candidate[insertion] in "\r\n":
                insertion += 1
        candidate = candidate[:insertion] + render(future_imports) + candidate[insertion:]

    ordinary_imports = [node for node in missing if node not in future_imports]
    if not ordinary_imports:
        return candidate
    tree = ast.parse(candidate)
    anchor = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            anchor = node.end_lineno or anchor
        elif (anchor == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
              and isinstance(node.value.value, str) and node is tree.body[0]):
            anchor = node.end_lineno or 0
        else:
            break
    lines = candidate.splitlines(keepends=True)
    if anchor and lines and not lines[anchor - 1].endswith("\n"):
        lines[anchor - 1] += "\n"
    block = render(ordinary_imports)
    return "".join(lines[:anchor]) + block + "".join(lines[anchor:])


def prepare_class_edit(
    current: str,
    target: str,
    replacement: str,
    *,
    path: pathlib.Path,
    root: pathlib.Path | None = None,
) -> ClassEditPlan:
    """Resolve one class and produce a candidate without writing it.

    ``handled=False`` deliberately leaves ordinary function edits on the historical
    path.  Once either side identifies a class, all malformed, missing, mismatched,
    and ambiguous cases are handled here and refused explicitly.
    """
    try:
        source_tree = ast.parse(current, filename=str(path))
    except SyntaxError:
        return ClassEditPlan(handled=False)
    targets = _matching_targets(_class_targets(source_tree), target)

    replacement = textwrap.dedent(replacement)
    try:
        replacement_tree = ast.parse(replacement, filename=str(path))
    except SyntaxError:
        if not targets:
            return ClassEditPlan(handled=False)
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "INVALID_CLASS_PAYLOAD", path, target, root=root,
                hint=(f"class payload for {target} is not valid Python; "
                      "read the current class and submit one complete class"),
            ),
        )

    payload_classes = [node for node in replacement_tree.body if isinstance(node, ast.ClassDef)]
    # r45 in-flight fix 1 (STATEFRAME_INFLIGHT_FIXES): import lines next to the class are not a second definition.
    payload_imports = ([node for node in replacement_tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
                       if _inflight_enabled() else [])
    payload_body_len = len(replacement_tree.body) - len(payload_imports)
    if not payload_classes:
        if not targets:
            return ClassEditPlan(handled=False)
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "TARGET_KIND_MISMATCH", path, target, root=root,
                hint=(f"{target} is a class; submit one complete class definition, "
                      "not a function or method body"),
                definitions=_definition_rows(targets),
            ),
        )
    if len(payload_classes) != 1 or payload_body_len != 1:
        # r44 M1: like its siblings, only own the edit when the target resolves to a
        # class; a multi-definition payload aimed at a function belongs to the legacy /
        # boundary-checks multi-definition path.
        if not targets:
            return ClassEditPlan(handled=False)
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "INVALID_CLASS_PAYLOAD", path, target, root=root,
                hint=("class-scoped edit accepts exactly one complete class definition; "
                      "use str_replace for module-level changes"),
            ),
        )
    if not targets:
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "NO_SUCH_CLASS", path, target, root=root,
                hint=f"no unique class named {target} exists in {path.name}",
            ),
        )
    if len(targets) > 1:
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "AMBIGUOUS_CLASS_TARGET", path, target, root=root,
                hint=(f"{target} names multiple classes; use a qualified class name "
                      "or str_replace with exact surrounding text"),
                definitions=_definition_rows(targets),
            ),
        )

    selected = targets[0]
    payload = payload_classes[0]
    if payload.name != selected.node.name:
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "CLASS_NAME_MISMATCH", path, target, root=root,
                hint=(f"target {selected.qualified_name} requires a payload named "
                      f"{selected.node.name}; never rename a class implicitly"),
                definitions=_definition_rows((selected,)),
            ),
        )
    if (selected.node.decorator_list and payload.decorator_list
            and _decorator_signature(selected.node) != _decorator_signature(payload)):
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "DECORATOR_MISMATCH", path, target, root=root,
                hint=(f"preserve the existing decorators on {selected.qualified_name}; "
                      "resubmit the class with matching decorators"),
                definitions=_definition_rows((selected,)),
            ),
        )

    # A decorated target is replaced from its ``class`` keyword onward.  This leaves
    # the original decorator lines byte-for-byte intact and prevents duplicate
    # decorators when native payloads repeat the class header.  If the target has no
    # decorators, a decorated payload is allowed and its decorator source is applied.
    include_payload_decorators = not selected.node.decorator_list and bool(payload.decorator_list)
    payload_segment = _source_segment(
        replacement, payload, include_decorators=include_payload_decorators,
    )
    candidate_segment = _reindent(payload_segment, selected.node.col_offset)
    start, end = _node_span(current, selected.node)
    candidate = current[:start] + candidate_segment + current[end:]
    if payload_imports:
        candidate = _add_missing_imports(candidate, payload_imports, replacement)
        import kernel_inflight_fixes
        kernel_inflight_fixes.emit(None, "class_edit_imports", stage="acted", path=str(path), target=target,
                                   imports=len(payload_imports))
    try:
        ast.parse(candidate, filename=str(path))
        if _inflight_enabled():
            # ast.parse accepts a late ``from __future__`` import; only compile() rejects it.
            # Compile (never execute) so an approved candidate is always loadable.
            compile(candidate, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        return ClassEditPlan(
            handled=True,
            rejection=_typed_refusal(
                "INVALID_CLASS_RESULT", path, target, root=root,
                hint=f"class edit would produce invalid Python ({exc.msg}); nothing was written",
            ),
        )
    return ClassEditPlan(handled=True, candidate=candidate)


def class_edit_context(source: str, target: str, *, path: str = "") -> dict[str, object]:
    """Return the host-owned scope fact used by the model-facing frame."""
    try:
        tree = ast.parse(source, filename=path or "candidate.py")
    except SyntaxError:
        return {
            "schema": "stateframe-jq-class-edit/1",
            "authority": "HOST_CAPTURED",
            "path": path,
            "target": target,
            "status": "INVALID_SOURCE",
            "scope": "named-class-only",
            "preserves": ["siblings", "decorators", "module-level-code"],
        }
    matches = _matching_targets(_class_targets(tree), target)
    status = "READY" if len(matches) == 1 else "AMBIGUOUS" if len(matches) > 1 else "NO_SUCH_CLASS"
    context: dict[str, object] = {
        "schema": "stateframe-jq-class-edit/1",
        "authority": "HOST_CAPTURED",
        "path": path,
        "target": target,
        "status": status,
        "scope": "named-class-only",
        "preserves": ["siblings", "decorators", "module-level-code"],
        "ambiguity": "typed-refusal",
    }
    if matches:
        context["definitions"] = list(_definition_rows(matches))
    return context


def render_context(context: Mapping[str, object]) -> str:
    preserves = context.get("preserves", ())
    if isinstance(preserves, (str, bytes)):
        preserve_text = str(preserves)
    else:
        preserve_text = ",".join(str(item) for item in preserves)
    return "\n".join((
        RENDER_HEADER,
        f"path={context.get('path', '')}",
        f"target={context.get('target', '')}",
        f"status={context.get('status', 'UNKNOWN')}",
        f"scope={context.get('scope', 'named-class-only')}",
        f"preserves={preserve_text}",
        f"ambiguous={context.get('ambiguity', 'typed-refusal')}",
    ))


__all__ = [
    "MODE_ENV",
    "RENDER_HEADER",
    "ClassEditPlan",
    "class_edit_context",
    "enabled",
    "mode",
    "prepare_class_edit",
    "render_context",
]
