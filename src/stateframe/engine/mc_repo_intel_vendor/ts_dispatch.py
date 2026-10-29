"""Tree-sitter dispatch boundaries and language-neutral source outlines.

Pinned companion of Mission Control's ``call_graph.py``. It announces runtime
dispatch boundaries instead of inventing static edges, and exposes the shared
outline contract used by qprobe and the kernel adapter; neither path raises.
"""

from __future__ import annotations

import ast
import contextvars
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .call_graph import DispatchBoundary

logger = logging.getLogger(__name__)

_OutlineCapabilityListener = Callable[[str, str, str, str], None]
_OUTLINE_CAPABILITY_LISTENER: contextvars.ContextVar[_OutlineCapabilityListener | None] = (
    contextvars.ContextVar("outline_capability_listener", default=None)
)


def bind_outline_capability_listener(listener: _OutlineCapabilityListener) -> contextvars.Token:
    """Bind run-local parser capability reporting to the current execution context."""
    return _OUTLINE_CAPABILITY_LISTENER.set(listener)


def reset_outline_capability_listener(token: contextvars.Token) -> None:
    """Restore the previous run-local parser capability listener."""
    _OUTLINE_CAPABILITY_LISTENER.reset(token)


def report_outline_parser_capability(outline: Mapping[str, object]) -> None:
    """Report only safe parser metadata for an outline actually consumed."""
    listener = _OUTLINE_CAPABILITY_LISTENER.get()
    if listener is None:
        return
    language = outline.get("language")
    parser = outline.get("parser")
    capability = outline.get("capability")
    completeness = outline.get("completeness")
    if all(isinstance(value, str) and value for value in (language, parser, capability, completeness)):
        listener(language, parser, capability, completeness)


def _node_attr(node, name: str, default=None):
    """Read both the 0.25 method API and the newer property API."""
    value = getattr(node, name, default)
    return value() if callable(value) else value


def _node_type(node) -> str:
    return str(_node_attr(node, "type", _node_attr(node, "kind", "")) or "")


def _node_children(node) -> list:
    children = _node_attr(node, "children", None)
    if children is not None:
        return list(children)
    count = int(_node_attr(node, "child_count", 0) or 0)
    child = getattr(node, "child", None)
    return [child(index) for index in range(count)] if callable(child) else []


def _node_named_children(node) -> list:
    children = _node_attr(node, "named_children", None)
    if children is not None:
        return list(children)
    count = int(_node_attr(node, "named_child_count", 0) or 0)
    child = getattr(node, "named_child", None)
    return [child(index) for index in range(count)] if callable(child) else []


def _node_field(node, name: str):
    getter = getattr(node, "child_by_field_name", None)
    return getter(name) if callable(getter) else None


def _root_node(tree):
    return _node_attr(tree, "root_node")


def _node_position(node, name: str) -> tuple[int, int]:
    point = _node_attr(node, name, None)
    if point is None:
        point = _node_attr(node, "start_point" if name == "start_position" else "end_point", None)
    if point is None:
        return 0, 0
    row = getattr(point, "row", None)
    column = getattr(point, "column", None)
    if row is not None:
        return int(row), int(column or 0)
    return int(point[0]), int(point[1])


def _node_has_error(node) -> bool:
    value = _node_attr(node, "has_error", False)
    return bool(value)

_STRING_KEYED_DISPATCH = frozenset({
    "emit", "dispatch", "send", "publish", "trigger", "fire",
    "handle", "route", "invoke", "$emit", "on",
})
_SCRIPT_BLOCK_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def _node_text(source: bytes, node) -> str:
    start = int(_node_attr(node, "start_byte", 0) or 0)
    end = int(_node_attr(node, "end_byte", start) or start)
    data = source if isinstance(source, bytes) else source.encode("utf-8", errors="replace")
    return data[start:end].decode("utf-8", errors="replace")


def _scan_ts(source: str, relpath: str, line_offset: int = 0) -> list[DispatchBoundary]:
    from .ts_extract import _parse
    try:
        import tree_sitter_language_pack as tslp
    except ImportError:
        return []
    lower = relpath.lower()
    lang = "tsx" if lower.endswith(".tsx") else (
        "javascript" if lower.endswith((".js", ".jsx", ".mjs", ".cjs"))
        else "typescript")
    try:
        parser = tslp.get_parser(lang)
        data = source.encode("utf-8", errors="replace")
        root = _root_node(_parse(parser, data))
    except Exception as exc:  # noqa: BLE001
        logger.debug("ts_dispatch: %s parse failed for %s: %s", lang, relpath, exc)
        return []

    boundaries: list[DispatchBoundary] = []

    def line(node) -> int:
        return _node_position(node, "start_position")[0] + 1 + line_offset

    def classify(call_node, func) -> None:
        func_type = _node_type(func)
        if func_type == "subscript_expression":
            func_children = _node_children(func)
            base = func_children[0] if func_children else None
            index = _node_field(func, "index")
            base_text = _node_text(data, base) if base is not None else "?"
            key = _node_text(data, index) if index is not None else "?"
            boundaries.append(DispatchBoundary(
                file=relpath, lineno=line(call_node), kind="computed_subscript",
                key_expr=key,
                note=(f"flow continues dynamically here (key={key}); callee is looked up "
                      f"in `{base_text}` at runtime — grep for where `{base_text}` is populated"),
            ))
            return
        if func_type == "import":
            args = _node_field(call_node, "arguments")
            key = _node_text(data, args) if args is not None else "?"
            boundaries.append(DispatchBoundary(
                file=relpath, lineno=line(call_node), kind="dynamic_import", key_expr=key,
                note=(f"flow continues dynamically here via dynamic import({key}); grep for "
                      "the literal paths the argument can take"),
            ))
            return
        if func_type != "member_expression":
            return
        prop = _node_field(func, "property")
        attr = _node_text(data, prop) if prop is not None else None
        if attr not in _STRING_KEYED_DISPATCH:
            return
        args = _node_field(call_node, "arguments")
        first = _node_named_children(args)[0] if args is not None and _node_named_children(args) else None
        if first is None or _node_type(first) != "string":
            return
        key = _node_text(data, first)
        receiver = _node_field(func, "object")
        recv = _node_text(data, receiver) if receiver is not None else "?"
        boundaries.append(DispatchBoundary(
            file=relpath, lineno=line(call_node), kind="string_keyed_bus", key_expr=key,
            note=(f"flow continues dynamically here (key={key}); `{recv}.{attr}` routes to "
                  f"a handler registered under {key} elsewhere — grep for {key}"),
        ))

    def walk(node) -> None:
        if _node_type(node) == "call_expression":
            function = _node_field(node, "function")
            if function is not None:
                classify(node, function)
        for child in _node_children(node):
            walk(child)

    try:
        walk(root)
    except RecursionError:  # pragma: no cover - defensive parser boundary
        return boundaries
    return boundaries


def dispatch_boundaries_in_file(source: str, relpath: str) -> list[DispatchBoundary]:
    """Return JS/TS/Vue dispatch boundaries; never raises."""
    try:
        if relpath.lower().endswith(".vue"):
            result: list[DispatchBoundary] = []
            for match in _SCRIPT_BLOCK_RE.finditer(source):
                body = match.group(1) or ""
                offset = source[:match.start(1)].count("\n")
                result.extend(_scan_ts(body, relpath, line_offset=offset))
            return result
        return _scan_ts(source, relpath)
    except Exception as exc:  # noqa: BLE001 - graph boundary is fail-open
        logger.debug("ts_dispatch: boundary scan failed for %s: %s", relpath, exc)
        return []


# ── Language-neutral source outline ──────────────────────────────────────────

OUTLINE_SCHEMA = "mc-outline/1"

_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cp": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".vue": "vue",
}

_PARSER_NAME = {"c_sharp": "csharp"}

_IDENTIFIER_TYPES = frozenset({
    "identifier", "type_identifier", "field_identifier", "property_identifier",
    "constant", "package_identifier", "class_name", "namespace_name", "name",
})

_BODY_TYPES = frozenset({
    "block", "statement_block", "class_body", "interface_body", "body_statement",
    "declaration_list", "compound_statement", "field_declaration_list",
    "enum_body", "enum_variant_list", "argument_list",
})

_CALL_TYPES = frozenset({
    "call", "call_expression", "method_invocation", "new_expression",
    "object_creation_expression", "macro_invocation",
})

_PHP_CALL_TYPES = _CALL_TYPES | frozenset({
    "function_call_expression", "member_call_expression", "scoped_call_expression",
})
_PHP_NAMESPACE_BODY_TYPES = frozenset({"compound_statement", "namespace_body", "declaration_list"})


@dataclass(frozen=True)
class _OutlineSpec:
    """Declarative node taxonomy; extraction logic stays language-neutral."""

    import_types: frozenset[str]
    module_types: frozenset[str]
    type_types: dict[str, str]
    function_types: frozenset[str]
    call_types: frozenset[str] = _CALL_TYPES


_PYTHON_SPEC = _OutlineSpec(
    import_types=frozenset({"import_statement", "import_from_statement"}),
    module_types=frozenset({"assignment", "annotated_assignment", "augmented_assignment", "expression_statement"}),
    type_types={"class_definition": "class"},
    function_types=frozenset({"function_definition", "async_function_definition"}),
)
_GO_SPEC = _OutlineSpec(
    import_types=frozenset({"import_declaration"}),
    module_types=frozenset({"var_declaration", "const_declaration", "expression_statement"}),
    type_types={"type_spec": "type"},
    function_types=frozenset({"function_declaration", "method_declaration"}),
)
_RUST_SPEC = _OutlineSpec(
    import_types=frozenset({"use_declaration", "extern_crate_declaration"}),
    module_types=frozenset({"static_item", "const_item", "let_declaration", "expression_statement"}),
    type_types={"struct_item": "struct", "enum_item": "enum", "trait_item": "trait", "impl_item": "impl", "type_item": "type"},
    function_types=frozenset({"function_item", "function_signature_item"}),
)
_TS_SPEC = _OutlineSpec(
    import_types=frozenset({"import_statement"}),
    module_types=frozenset({"lexical_declaration", "variable_declaration", "export_statement", "expression_statement"}),
    type_types={"class_declaration": "class", "interface_declaration": "interface", "type_alias_declaration": "type", "enum_declaration": "enum"},
    function_types=frozenset({"function_declaration", "method_definition", "method_signature"}),
)
_JAVA_SPEC = _OutlineSpec(
    import_types=frozenset({"import_declaration", "package_declaration"}),
    module_types=frozenset({"field_declaration", "static_initializer", "expression_statement"}),
    type_types={"class_declaration": "class", "interface_declaration": "interface", "enum_declaration": "enum", "annotation_type_declaration": "interface"},
    function_types=frozenset({"method_declaration", "constructor_declaration"}),
)
_C_SPEC = _OutlineSpec(
    import_types=frozenset({"preproc_include", "preproc_def"}),
    module_types=frozenset({"declaration", "expression_statement"}),
    type_types={"struct_specifier": "struct", "enum_specifier": "enum", "union_specifier": "struct"},
    function_types=frozenset({"function_definition"}),
)
_CPP_SPEC = _OutlineSpec(
    import_types=frozenset({"preproc_include", "preproc_def", "using_declaration", "using_directive"}),
    module_types=frozenset({"declaration", "expression_statement"}),
    type_types={"class_specifier": "class", "struct_specifier": "struct", "enum_specifier": "enum", "namespace_definition": "module"},
    function_types=frozenset({"function_definition"}),
)
_CSHARP_SPEC = _OutlineSpec(
    import_types=frozenset({"using_directive", "extern_alias_directive"}),
    module_types=frozenset({"global_statement", "local_declaration_statement", "field_declaration", "expression_statement"}),
    type_types={"class_declaration": "class", "struct_declaration": "struct", "interface_declaration": "interface", "enum_declaration": "enum", "record_declaration": "class"},
    function_types=frozenset({"method_declaration", "constructor_declaration", "local_function_statement"}),
)
_RUBY_SPEC = _OutlineSpec(
    import_types=frozenset(),
    module_types=frozenset({"assignment", "call", "expression_statement"}),
    type_types={"class": "class", "module": "module"},
    function_types=frozenset({"method", "singleton_method"}),
)
_PHP_SPEC = _OutlineSpec(
    import_types=frozenset({"namespace_definition", "namespace_use_declaration", "use_declaration"}),
    module_types=frozenset({
        "expression_statement", "const_declaration", "global_declaration",
        "echo_statement", "unset_statement",
    }),
    type_types={
        "class_declaration": "class", "interface_declaration": "interface",
        "trait_declaration": "trait", "enum_declaration": "enum",
    },
    function_types=frozenset({"function_definition", "method_declaration"}),
    call_types=_PHP_CALL_TYPES,
)

_OUTLINE_SPECS = {
    "python": _PYTHON_SPEC,
    "go": _GO_SPEC,
    "rust": _RUST_SPEC,
    "typescript": _TS_SPEC,
    "tsx": _TS_SPEC,
    "javascript": _TS_SPEC,
    "java": _JAVA_SPEC,
    "c": _C_SPEC,
    "cpp": _CPP_SPEC,
    "c_sharp": _CSHARP_SPEC,
    "ruby": _RUBY_SPEC,
    "php": _PHP_SPEC,
}


def _detected_language(relpath: str, source: str) -> tuple[str, str]:
    suffix = Path(relpath).suffix.casefold()
    if suffix in _LANGUAGE_BY_EXTENSION:
        return _LANGUAGE_BY_EXTENSION[suffix], "extension"
    first_line = (source.splitlines() or [""])[0].casefold()
    if first_line.startswith("#!"):
        for marker, language in (
            ("python", "python"), ("ruby", "ruby"), ("node", "javascript"),
            ("deno", "javascript"), ("bun", "javascript"), ("go", "go"),
        ):
            if marker in first_line:
                return language, "shebang"
    return "unknown", "fallback"


def detect_language(relpath: str, source: str = "") -> str:
    """Return the canonical language selected by extension, then shebang."""
    return _detected_language(str(relpath), source)[0]


def _parse_source(language: str, source: str):
    import tree_sitter_language_pack as tslp

    parser = tslp.get_parser(_PARSER_NAME.get(language, language))
    try:
        return parser.parse(source.encode("utf-8", errors="replace"))
    except TypeError:
        return parser.parse(source)


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().split())


def _line_record(node) -> dict[str, int]:
    start = _node_position(node, "start_position")[0] + 1
    end = _node_position(node, "end_position")[0] + 1
    return {"line": start, "end_line": max(start, end)}


def _node_name(node, source: bytes) -> str:
    for field_name in ("name", "declarator", "declaration", "left"):
        candidate = _node_field(node, field_name)
        if candidate is not None:
            name = _node_name(candidate, source)
            if name:
                return name
    node_type = _node_type(node)
    if node_type in _IDENTIFIER_TYPES or node_type in {"qualified_identifier", "scoped_identifier"}:
        return _normalise_text(_node_text(source, node))
    for child in _node_named_children(node):
        name = _node_name(child, source)
        if name:
            return name
    return ""


def _body_node(node):
    for child in _node_named_children(node):
        if _node_type(child) in _BODY_TYPES:
            if _node_type(child) == "argument_list":
                body = _node_field(node, "body")
                if body is not None:
                    return body
            return child
    for field_name in ("body", "block", "declaration_list"):
        candidate = _node_field(node, field_name)
        if candidate is not None:
            return candidate
    return None


def _signature(node, source: bytes) -> str:
    body = _body_node(node)
    start = int(_node_attr(node, "start_byte", 0) or 0)
    end = int(_node_attr(body, "start_byte", 0) or 0) if body is not None else 0
    text = _node_text(source, node) if body is None or end <= start else source[start:end].decode("utf-8", errors="replace")
    if "\n" in text:
        text = text.splitlines()[0]
    if not text.strip():
        text = _node_text(source, node).splitlines()[0] if _node_text(source, node) else ""
    return _normalise_text(text).rstrip(";")


def _walk_nodes(node) -> Iterable:
    yield node
    for child in _node_named_children(node):
        yield from _walk_nodes(child)


def _call_records(node, source: bytes, spec: _OutlineSpec) -> list[dict[str, object]]:
    records = []
    call_types = spec.call_types
    for candidate in _walk_nodes(node):
        if _node_type(candidate) not in call_types:
            continue
        records.append({
            "expression": _normalise_text(_node_text(source, candidate)),
            **_line_record(candidate),
        })
    records.sort(key=lambda item: (int(item["line"]), str(item["expression"])))
    return records


def _is_import_node(node, language: str, spec: _OutlineSpec, source: bytes) -> bool:
    node_type = _node_type(node)
    if node_type in spec.import_types:
        return True
    return language == "ruby" and _normalise_text(_node_text(source, node)).startswith("require ")


def _import_record(node, source: bytes, language: str) -> dict[str, object]:
    text = _normalise_text(_node_text(source, node))
    module = None
    names: list[str] = []
    for child in _node_named_children(node):
        child_type = _node_type(child)
        if child_type in {
            "string", "interpreted_string_literal", "system_lib_string", "scoped_identifier",
            "dotted_name", "identifier", "constant", "namespace_name", "qualified_name",
            "namespace_use_clause", "namespace_aliasing_clause", "name",
        }:
            value = _normalise_text(_node_text(source, child))
            if module is None:
                module = value.strip("'\"")
            names.append(value)
    if language == "ruby" and text.startswith("require "):
        module = text.removeprefix("require ").strip().strip("'\"")
    kind = "namespace" if language == "php" and _node_type(node) == "namespace_definition" else "import"
    if language == "php" and module is None:
        match = re.match(r"(?:namespace|use)\s+(.+?);?$", text)
        if match:
            module = match.group(1).rstrip(";").strip()
            names = [module]
    return {"kind": kind, "module": module, "names": names, "source": text, **_line_record(node)}


def _type_kind(node, spec: _OutlineSpec) -> str | None:
    node_type = _node_type(node)
    kind = spec.type_types.get(node_type)
    if kind is not None:
        if node_type == "type_spec":
            for child in _node_named_children(node):
                child_kind = {
                    "struct_type": "struct",
                    "interface_type": "interface",
                }.get(_node_type(child))
                if child_kind in {"struct", "interface"}:
                    return child_kind
        return kind
    if node_type == "type_declaration":
        return None
    return None


def _module_kind(node, source: bytes, spec: _OutlineSpec) -> str:
    text = _normalise_text(_node_text(source, node))
    if _node_type(node) == "package_declaration" or text.startswith("package "):
        return "package"
    if _node_type(node) == "namespace_definition" or text.startswith("namespace "):
        return "namespace"
    if _node_type(node) in spec.call_types or any(_node_type(child) in spec.call_types for child in _node_named_children(node)):
        return "call"
    if "=" in text or " new " in f" {text} ":
        return "assignment"
    return "declaration"


_PYTHON_STATEMENT_TYPES = frozenset({
    "import_statement", "import_from_statement", "future_import_statement", "class_definition",
    "function_definition", "async_function_definition", "decorated_definition", "if_statement", "for_statement",
    "while_statement", "try_statement", "with_statement", "match_statement", "return_statement", "pass_statement",
    "break_statement", "continue_statement", "global_statement", "nonlocal_statement", "assert_statement",
    "delete_statement", "raise_statement", "print_statement", "exec_statement", "type_alias_statement",
    "comment", "ERROR",
})
_PYTHON_ASSIGNMENT_TYPES = frozenset({"assignment", "augmented_assignment"})


def _python_module_node_type(node, top_level: bool) -> str | None:
    """The stdlib-ast module statement type for a tree-sitter Python node, or None when it is not one."""
    node_type = _node_type(node)
    target = node
    if node_type == "expression_statement":
        children = _node_named_children(node)
        if len(children) != 1 or _node_type(children[0]) not in _PYTHON_ASSIGNMENT_TYPES:
            return "expression_statement"
        target = children[0]
    elif node_type not in _PYTHON_ASSIGNMENT_TYPES:
        # A bare expression where the binding exposes no expression_statement wrapper.
        return "expression_statement" if top_level and node_type not in _PYTHON_STATEMENT_TYPES else None
    if _node_type(target) == "assignment" and _node_field(target, "type") is not None:
        return "annotated_assignment"
    return _node_type(target)


def _module_record(node, source: bytes, spec: _OutlineSpec, scope: str = "module",
                   node_type: str | None = None) -> dict[str, object]:
    return {
        "kind": _module_kind(node, source, spec),
        "scope": scope,
        "node_type": node_type or _node_type(node),
        "source": _normalise_text(_node_text(source, node)),
        "calls": _call_records(node, source, spec),
        **_line_record(node),
    }


def _static_member(node, source: bytes) -> bool:
    return bool(re.search(r"\b(?:static|const|readonly|shared|final)\b", _node_text(source, node)))


def _tree_outline(source: str, relpath: str, language: str, detection: str, root, spec: _OutlineSpec) -> dict[str, object]:
    source_bytes = source.encode("utf-8", errors="replace")
    imports: list[dict[str, object]] = []
    module_statements: list[dict[str, object]] = []
    types: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []

    def visit(
        node,
        parent_type: str | None = None,
        top_level: bool = False,
        transparent_module_scope: bool = False,
    ) -> None:
        node_type = _node_type(node)
        effective_top_level = top_level or transparent_module_scope
        if effective_top_level and _is_import_node(node, language, spec, source_bytes):
            imports.append(_import_record(node, source_bytes, language))
        type_kind = _type_kind(node, spec)
        current_parent = parent_type
        if type_kind is not None:
            type_name = _node_name(node, source_bytes) or _signature(node, source_bytes)
            type_record = {
                "kind": type_kind,
                "name": type_name,
                "signature": _signature(node, source_bytes),
                "methods": [],
                **_line_record(node),
            }
            types.append(type_record)
            current_parent = type_name or parent_type
        elif node_type in spec.function_types:
            function_name = _node_name(node, source_bytes) or "<anonymous>"
            function_parent = parent_type
            if language == "go" and function_parent is None:
                receiver = _LEXICAL_RECEIVER_METHOD_RE.search(_signature(node, source_bytes))
                if receiver is not None:
                    function_parent = receiver.group("parent")
            functions.append({
                "kind": "method" if function_parent else "function",
                "name": function_name,
                "parent": function_parent,
                "signature": _signature(node, source_bytes),
                **_line_record(node),
            })
        module_scope = effective_top_level or (parent_type is not None and _static_member(node, source_bytes))
        if language == "python":
            python_type = _python_module_node_type(node, effective_top_level) if module_scope else None
            if python_type is not None and not (effective_top_level and _is_import_node(node, language, spec, source_bytes)):
                module_statements.append(_module_record(
                    node, source_bytes, spec, "type" if parent_type else "module", python_type))
        elif node_type in spec.module_types and module_scope \
                and not (effective_top_level and _is_import_node(node, language, spec, source_bytes)):
            module_statements.append(_module_record(node, source_bytes, spec, "type" if parent_type else "module"))
        child_top_level = effective_top_level if (node_type == "namespace_definition" or transparent_module_scope) else False
        child_transparent_scope = node_type == "namespace_definition" or (
            transparent_module_scope and node_type in _PHP_NAMESPACE_BODY_TYPES
        )
        for child in _node_named_children(node):
            visit(child, current_parent, child_top_level, child_transparent_scope)

    for child in _node_named_children(root):
        visit(child, None, True)
    for type_record in types:
        type_name = str(type_record["name"])
        type_record["methods"] = [item for item in functions if item.get("parent") == type_name]
    return {
        "schema": OUTLINE_SCHEMA,
        "file": str(relpath),
        "language": language,
        "detection": detection,
        "parser": "tree-sitter",
        "degraded": False,
        "capability": "AVAILABLE",
        "completeness": "KNOWN",
        "imports": imports,
        "module_statements": module_statements,
        "types": types,
        "functions": functions,
    }


_PYTHON_MODULE_NODE_TYPES = {
    ast.Assign: "assignment",
    ast.AnnAssign: "annotated_assignment",
    ast.AugAssign: "augmented_assignment",
    ast.Expr: "expression_statement",
}


# Same line split as CPython's ast._splitlines_no_ff (PSF license): \r\n, \n, \r only.
_PY_LINE_PATTERN = re.compile(r"(.*?(?:\r\n|\n|\r|$))")
_PY_LINES_CACHE: list[object] = [None, ()]


def _python_source_lines(source: str) -> tuple[str, ...]:
    """Split ``source`` once per outline; ast.get_source_segment re-splits per node (O(n^2))."""
    if _PY_LINES_CACHE[0] is not source:
        _PY_LINES_CACHE[1] = tuple(match[0] for match in _PY_LINE_PATTERN.finditer(source))
        _PY_LINES_CACHE[0] = source
    return _PY_LINES_CACHE[1]  # type: ignore[return-value]


def _python_ast_source(source: str, node: ast.AST) -> str:
    """Byte-identical to ``ast.get_source_segment(source, node) or ""`` (unpadded)."""
    try:
        if node.end_lineno is None or node.end_col_offset is None:
            return ""
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
        col_offset = node.col_offset
        end_col_offset = node.end_col_offset
    except AttributeError:
        return ""
    lines = _python_source_lines(source)
    if end_lineno == lineno:
        return lines[lineno].encode()[col_offset:end_col_offset].decode()
    first = lines[lineno].encode()[col_offset:].decode()
    last = lines[end_lineno].encode()[:end_col_offset].decode()
    return first + "".join(lines[lineno + 1:end_lineno]) + last


def _python_ast_line_record(node: ast.AST, *, start_line: int | None = None) -> dict[str, int]:
    start = int(start_line or getattr(node, "lineno", 1) or 1)
    end = int(getattr(node, "end_lineno", start) or start)
    return {"line": start, "end_line": max(start, end)}


def _python_ast_definition_line(
    source_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    fallback = int(getattr(node, "lineno", 1) or 1)
    name = str(getattr(node, "name", ""))
    keyword = "class" if isinstance(node, ast.ClassDef) else "def"
    decorators = getattr(node, "decorator_list", ())
    decorator_lines = [int(getattr(item, "lineno", fallback) or fallback) for item in decorators]
    first_line = min(decorator_lines, default=fallback)
    body = getattr(node, "body", ())
    last_line = int(getattr(body[0], "lineno", fallback) or fallback) if body else fallback
    declaration = re.compile(rf"^[ \t]*(?:async[ \t]+)?{keyword}[ \t]+{re.escape(name)}\b")
    for line_number in range(first_line, last_line + 1):
        if line_number <= len(source_lines) and declaration.match(source_lines[line_number - 1]):
            return line_number
    return fallback


def _python_ast_definition_signature(
    source_bytes: bytes,
    line_starts: list[int],
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    start_line: int,
) -> str:
    body = getattr(node, "body", ())
    if not body:
        return ""
    body_node = body[0]
    start_index = min(max(start_line - 1, 0), len(line_starts) - 1)
    end_line = int(getattr(body_node, "lineno", start_line) or start_line)
    end_index = min(max(end_line - 1, 0), len(line_starts) - 1)
    start = line_starts[start_index] + int(getattr(node, "col_offset", 0) or 0)
    end = line_starts[end_index] + int(getattr(body_node, "col_offset", 0) or 0)
    if end < start or end > len(source_bytes):
        return ""
    header = source_bytes[start:end].decode("utf-8", errors="replace")
    first_line = header.splitlines()[0] if header else ""
    return _normalise_text(first_line).rstrip(";")


def _python_ast_call_records(node: ast.AST, source: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        records.append({
            "expression": _normalise_text(_python_ast_source(source, candidate)),
            **_python_ast_line_record(candidate),
        })
    records.sort(key=lambda item: (int(item["line"]), str(item["expression"])))
    return records


def _python_ast_import_record(node: ast.Import | ast.ImportFrom, source: str) -> dict[str, object]:
    text = _normalise_text(_python_ast_source(source, node))
    names: list[str] = []
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        names.append(node.module)
    for alias in node.names:
        # _import_record only inspects direct named children; tree-sitter wraps
        # aliased imports in aliased_import, so those names remain omitted here.
        if alias.asname is None and alias.name != "*":
            names.append(alias.name)
    return {
        "kind": "import",
        "module": names[0] if names else None,
        "names": names,
        "source": text,
        **_python_ast_line_record(node),
    }


def _python_ast_module_record(
    node: ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Expr,
    source: str,
    parent_type: str | None,
) -> dict[str, object]:
    text = _normalise_text(_python_ast_source(source, node))
    direct_call = any(isinstance(child, ast.Call) for child in ast.iter_child_nodes(node))
    if direct_call:
        kind = "call"
    elif "=" in text or " new " in f" {text} ":
        kind = "assignment"
    else:
        kind = "declaration"
    return {
        "kind": kind,
        "scope": "type" if parent_type else "module",
        "node_type": _PYTHON_MODULE_NODE_TYPES[type(node)],
        "source": text,
        "calls": _python_ast_call_records(node, source),
        **_python_ast_line_record(node),
    }


def _ast_python_outline(source: str, relpath: str, detection: str) -> dict[str, object]:
    """Build the Python outline with stdlib AST using the tree outline contract."""
    tree = ast.parse(source)
    source_bytes = source.encode("utf-8", errors="replace")
    source_lines = source.splitlines()
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line.encode("utf-8", errors="replace")))

    imports: list[dict[str, object]] = []
    module_statements: list[dict[str, object]] = []
    types: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []

    def visit(node: ast.AST, parent_type: str | None = None, top_level: bool = False) -> None:
        current_parent = parent_type
        if top_level and isinstance(node, (ast.Import, ast.ImportFrom)):
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__" and node.level == 0):
                imports.append(_python_ast_import_record(node, source))

        if isinstance(node, ast.ClassDef):
            start_line = _python_ast_definition_line(source_lines, node)
            type_name = node.name or _python_ast_definition_signature(
                source_bytes, line_starts, node, start_line)
            types.append({
                "kind": "class",
                "name": type_name,
                "signature": _python_ast_definition_signature(
                    source_bytes, line_starts, node, start_line),
                "methods": [],
                **_python_ast_line_record(node, start_line=start_line),
            })
            current_parent = type_name or parent_type
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = _python_ast_definition_line(source_lines, node)
            functions.append({
                "kind": "method" if parent_type else "function",
                "name": node.name or "<anonymous>",
                "parent": parent_type,
                "signature": _python_ast_definition_signature(
                    source_bytes, line_starts, node, start_line),
                **_python_ast_line_record(node, start_line=start_line),
            })

        module_node_type = _PYTHON_MODULE_NODE_TYPES.get(type(node))
        if module_node_type is not None:
            text = _python_ast_source(source, node)
            is_static_member = bool(re.search(r"\b(?:static|const|readonly|shared|final)\b", text))
            if top_level or (parent_type is not None and is_static_member):
                module_statements.append(_python_ast_module_record(node, source, parent_type))

        for child in ast.iter_child_nodes(node):
            visit(child, current_parent, False)

    for statement in tree.body:
        visit(statement, None, True)

    for type_record in types:
        type_name = str(type_record["name"])
        type_record["methods"] = [item for item in functions if item.get("parent") == type_name]

    return {
        "schema": OUTLINE_SCHEMA,
        "file": str(relpath),
        "language": "python",
        "detection": detection,
        "parser": "ast",
        "degraded": False,
        "capability": "AVAILABLE",
        "completeness": "KNOWN",
        "imports": imports,
        "module_statements": module_statements,
        "types": types,
        "functions": functions,
    }


_LEXICAL_IMPORT_RE = re.compile(r"^(?:import\b|from\b|use\b|namespace\b|#\s*include\b|using\b|require\b|package\b)")
_LEXICAL_TYPE_RE = re.compile(r"\b(class|struct|enum|interface|trait|impl|module)\s+([A-Za-z_$][\w$]*)")
_LEXICAL_FUNCTION_RE = re.compile(r"\b(?:async\s+)?(?:def|fn|func|function)\s+([A-Za-z_$][\w$]*)")
_LEXICAL_RECEIVER_METHOD_RE = re.compile(
    r"\bfunc\s*\(\s*[A-Za-z_$][\w$]*\s+\*?(?P<parent>[A-Za-z_$][\w$]*)\s*\)\s*"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\("
)
_LEXICAL_METHOD_RE = re.compile(
    r"\b(?:(?:async|public|private|protected|static|readonly|override|abstract|get|set)\s+)*"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^>\n]*>)?\s*\([^(){}\n]*\)"
    r"(?:\s*:\s*[^={]+)?"
    r"(?:\s+throws\s+[A-Za-z_$][\w$.,\s<>?]*)?\s*\{"
)
_LEXICAL_METHOD_KEYWORDS = frozenset({"if", "for", "while", "switch", "catch", "return"})
_LEXICAL_CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(")


def _lexical_calls(text: str, line: int) -> list[dict[str, object]]:
    return [{"expression": match.group(1) + "(...)", "line": line, "end_line": line} for match in _LEXICAL_CALL_RE.finditer(text)]


def _lexical_outline(source: str, relpath: str, language: str, detection: str, reason: str) -> dict[str, object]:
    imports: list[dict[str, object]] = []
    module_statements: list[dict[str, object]] = []
    types: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []
    brace_depth = 0
    type_scopes: list[tuple[str, int]] = []
    php_namespace_depth: int | None = None
    php_type_depths: list[int] = []
    for line_number, raw_line in enumerate(source.splitlines(), 1):
        stripped = raw_line.strip()
        while type_scopes and brace_depth < type_scopes[-1][1]:
            type_scopes.pop()
        if language == "php":
            while php_type_depths and brace_depth < php_type_depths[-1]:
                php_type_depths.pop()
            if php_namespace_depth is not None and brace_depth < php_namespace_depth:
                php_namespace_depth = None
        if not stripped or stripped.startswith(("#", "//", "/*", "*")) and not stripped.startswith("#include"):
            brace_depth += raw_line.count("{") - raw_line.count("}")
            continue
        source_line = _normalise_text(raw_line)
        if _LEXICAL_IMPORT_RE.match(stripped):
            import_kind = "namespace" if language == "php" and stripped.startswith("namespace ") else "import"
            imports.append({"kind": import_kind, "module": None, "names": [], "source": source_line, "line": line_number, "end_line": line_number})
        type_match = _LEXICAL_TYPE_RE.search(stripped)
        if type_match:
            type_name = type_match.group(2)
            types.append({"kind": type_match.group(1), "name": type_name, "signature": source_line, "methods": [], "line": line_number, "end_line": line_number})
            if "{" in raw_line:
                type_scopes.append((type_name, brace_depth + 1))
            if language == "php" and "{" in raw_line:
                php_type_depths.append(brace_depth + 1)
        receiver_match = _LEXICAL_RECEIVER_METHOD_RE.search(stripped) if language == "go" else None
        function_match = _LEXICAL_FUNCTION_RE.search(stripped)
        method_match = _LEXICAL_METHOD_RE.search(stripped) if not function_match else None
        function_name = (receiver_match.group("name") if receiver_match else
                         function_match.group(1) if function_match else
                         method_match.group("name") if method_match else None)
        if function_name and function_name not in _LEXICAL_METHOD_KEYWORDS:
            parent = (receiver_match.group("parent") if receiver_match else
                      type_scopes[-1][0] if type_scopes and method_match else None)
            functions.append({"kind": "method" if parent else "function", "name": function_name,
                              "parent": parent, "signature": source_line, "line": line_number,
                              "end_line": line_number})
        if language == "php" and stripped.startswith("namespace ") and "{" in raw_line:
            php_namespace_depth = brace_depth + 1
        php_namespace_scope = php_namespace_depth is not None and brace_depth == php_namespace_depth and not php_type_depths
        top_level = (brace_depth == 0 and not php_type_depths) or php_namespace_scope
        assignment = bool(re.search(r"(?:^|\b)(?:var|const|let|static|final|val|mut)\b|:=|(?<![=!<>])=(?!=|>)|\bnew\s+", stripped))
        direct_call = bool(_LEXICAL_CALL_RE.search(stripped)) and not function_name and not type_match
        if (top_level or stripped.startswith(("static ", "const ", "final "))) and (assignment or direct_call or stripped.startswith("package ")):
            module_statements.append({
                "kind": "call" if direct_call else ("package" if stripped.startswith("package ") else "assignment"),
                "scope": "module",
                "node_type": "lexical",
                "source": source_line,
                "calls": _lexical_calls(stripped, line_number),
                "line": line_number,
                "end_line": line_number,
            })
        brace_depth += raw_line.count("{") - raw_line.count("}")
    for type_record in types:
        type_name = str(type_record["name"])
        type_record["methods"] = [item for item in functions if item.get("parent") == type_name]
    return {
        "schema": OUTLINE_SCHEMA,
        "file": str(relpath),
        "language": language,
        "detection": detection,
        "parser": "lexical",
        "degraded": True,
        "capability": "DEGRADED",
        "completeness": "UNKNOWN",
        "degraded_reason": reason,
        "imports": imports,
        "module_statements": module_statements,
        "types": types,
        "functions": functions,
    }


_SFC_OPEN_RE = re.compile(r"<(?P<tag>template|script|style)\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_SFC_TEMPLATE_TOKEN_RE = re.compile(r"<\s*(?P<closing>/)?\s*template\b[^>]*>", re.IGNORECASE)
_TEMPLATE_ATTR_RE = re.compile(
    r'''(?P<name>(?:[:@#]|v-)?[A-Za-z_][\w:.-]*)(?:\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s"'=<>`]+)))?'''
)
_COMPOSABLE_CALL_RE = re.compile(r"\b(use[A-Z][A-Za-z0-9_$]*)\s*\(")
_VUE_MACRO_RE = re.compile(r"\b(defineProps|defineEmits|defineModel|defineExpose)\s*(?:<[^()\n]*>)?\s*\(")
_VUE_REACTIVE_DECLARATION_RE = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(ref|reactive|computed|watch)\s*\("
)
_VUE_REACTIVE_CALL_RE = re.compile(r"\b(ref|reactive|computed|watch)\s*\(")

_HTML_TEMPLATE_TAGS = frozenset({
    "a", "abbr", "address", "area", "article", "aside", "audio", "b", "base", "bdi",
    "bdo", "blockquote", "body", "br", "button", "canvas", "caption", "cite", "code",
    "col", "colgroup", "data", "datalist", "dd", "del", "details", "dfn", "dialog",
    "div", "dl", "dt", "em", "embed", "fieldset", "figcaption", "figure", "footer",
    "form", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr",
    "html", "i", "iframe", "img", "input", "ins", "kbd", "label", "legend", "li",
    "link", "main", "map", "mark", "menu", "meta", "meter", "nav", "noscript", "object",
    "ol", "optgroup", "option", "output", "p", "picture", "pre", "progress", "q", "rp",
    "rt", "ruby", "s", "samp", "script", "section", "select", "small", "source", "span",
    "strong", "style", "sub", "summary", "sup", "table", "tbody", "td", "template", "textarea",
    "tfoot", "th", "thead", "time", "title", "tr", "track", "u", "ul", "var", "video", "wbr",
    "component", "keep-alive", "slot", "teleport", "transition", "transition-group", "svg", "path",
})


def _line_at_offset(source: str, offset: int, base_line: int = 1) -> int:
    return base_line + source.count("\n", 0, max(0, offset))


def _sfc_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _TEMPLATE_ATTR_RE.finditer(raw_attrs):
        value = match.group("double")
        if value is None:
            value = match.group("single")
        if value is None:
            value = match.group("bare")
        attrs[match.group("name").casefold()] = value or ""
    return attrs


def _find_sfc_close(source: str, tag: str, body_start: int) -> tuple[int, int, bool]:
    if tag.casefold() != "template":
        close = re.search(rf"</\s*{re.escape(tag)}\s*>", source[body_start:], re.IGNORECASE)
        if close is None:
            return len(source), len(source), False
        return body_start + close.start(), body_start + close.end(), True

    depth = 1
    for match in _SFC_TEMPLATE_TOKEN_RE.finditer(source, body_start):
        if match.group("closing"):
            depth -= 1
        else:
            depth += 1
        if depth == 0:
            return match.start(), match.end(), True
    return len(source), len(source), False


def _extract_sfc_blocks(source: str) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    cursor = 0
    while True:
        match = _SFC_OPEN_RE.search(source, cursor)
        if match is None:
            break
        tag = match.group("tag").casefold()
        body_start = match.end()
        body_end, close_end, closed = _find_sfc_close(source, tag, body_start)
        attrs = _sfc_attrs(match.group("attrs") or "")
        content_start_line = _line_at_offset(source, body_start)
        content_end_line = _line_at_offset(source, body_end)
        blocks.append({
            "tag": tag,
            "kind": "script_setup" if tag == "script" and "setup" in attrs else tag,
            "attrs": attrs,
            "start_line": _line_at_offset(source, match.start()),
            "content_start_line": content_start_line,
            "content_end_line": content_end_line,
            "end_line": _line_at_offset(source, body_end),
            "closed": closed,
            "degraded": not closed,
            "body_start": body_start,
            "body_end": body_end,
        })
        if not closed:
            break
        cursor = close_end
    return blocks


def _offset_outline_lines(value: object, line_offset: int) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key in {"line", "end_line"} and isinstance(child, int):
                value[key] = child + line_offset
            else:
                _offset_outline_lines(child, line_offset)
    elif isinstance(value, list):
        for child in value:
            _offset_outline_lines(child, line_offset)


def _vue_grammar_available() -> bool:
    try:
        import tree_sitter_language_pack as tslp
        return "vue" in {str(name).casefold() for name in tslp.available_languages()}
    except Exception:  # noqa: BLE001 - optional grammar probe
        return False


def _vue_feature_record(source: str, match: re.Match[str], line_offset: int, **extra: object) -> dict[str, object]:
    line = _line_at_offset(source, match.start()) + line_offset
    return {
        "source": _normalise_text(match.group(0)),
        "line": line,
        "end_line": line,
        **extra,
    }


def _vue_script_features(source: str, line_offset: int) -> dict[str, list[dict[str, object]]]:
    macros = [
        _vue_feature_record(source, match, line_offset, name=match.group(1), kind="macro")
        for match in _VUE_MACRO_RE.finditer(source)
    ]
    reactivity = []
    for match in _VUE_REACTIVE_DECLARATION_RE.finditer(source):
        reactivity.append(_vue_feature_record(
            source, match, line_offset, name=match.group(1), kind=match.group(2), declaration=True,
        ))
    for match in _VUE_REACTIVE_CALL_RE.finditer(source):
        if any(item["line"] == _line_at_offset(source, match.start()) + line_offset and item["kind"] == match.group(1) for item in reactivity):
            continue
        reactivity.append(_vue_feature_record(
            source, match, line_offset, name=None, kind=match.group(1), declaration=False,
        ))
    composables = [
        _vue_feature_record(source, match, line_offset, name=match.group(1), kind="composable")
        for match in _COMPOSABLE_CALL_RE.finditer(source)
    ]
    return {"macros": macros, "reactivity": reactivity, "composables": composables}


def _iter_template_tags(source: str) -> Iterable[tuple[bool, str, str, int, int]]:
    cursor = 0
    while cursor < len(source):
        start = source.find("<", cursor)
        if start < 0:
            return
        if source.startswith("<!--", start):
            end_comment = source.find("-->", start + 4)
            cursor = len(source) if end_comment < 0 else end_comment + 3
            continue
        head = re.match(r"<\s*(/?)\s*([A-Za-z][\w:.-]*)", source[start:])
        if head is None:
            cursor = start + 1
            continue
        closing = bool(head.group(1))
        name = head.group(2)
        attr_start = start + head.end()
        quote: str | None = None
        end = attr_start
        while end < len(source):
            char = source[end]
            if quote:
                if char == quote:
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == ">":
                break
            end += 1
        if end >= len(source):
            yield closing, name, source[attr_start:], start, len(source)
            return
        yield closing, name, source[attr_start:end], start, end + 1
        cursor = end + 1


def _template_component(name: str) -> bool:
    lower = name.casefold()
    return lower.startswith("v-") or (name[:1].isupper()) or (
        "-" in name and lower not in _HTML_TEMPLATE_TAGS
    )


def _template_attr_value(match: re.Match[str]) -> str:
    return match.group("double") or match.group("single") or match.group("bare") or ""


def _template_outline(source: str, block: dict[str, object], vue_grammar: bool) -> dict[str, object]:
    body_start = int(block["body_start"])
    body_end = int(block["body_end"])
    body = source[body_start:body_end]
    base_line = int(block["content_start_line"])
    components: list[dict[str, object]] = []
    slots: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    directives: list[dict[str, object]] = []

    for closing, tag_name, raw_attrs, tag_start, tag_end in _iter_template_tags(body):
        if closing:
            continue
        tag_line = _line_at_offset(body, tag_start, base_line)
        tag_end_line = _line_at_offset(body, tag_end, base_line)
        tag_attrs = []
        for attr_match in _TEMPLATE_ATTR_RE.finditer(raw_attrs):
            attr_name = attr_match.group("name")
            attr_line = _line_at_offset(body, tag_start + attr_match.start(), base_line)
            tag_attrs.append((attr_name, _template_attr_value(attr_match), attr_line))

        component: dict[str, object] | None = None
        if _template_component(tag_name):
            component = {
                "name": tag_name,
                "line": tag_line,
                "end_line": tag_end_line,
                "props": [],
                "events": [],
                "directives": [],
            }
            components.append(component)

        for attr_name, value, attr_line in tag_attrs:
            lower_name = attr_name.casefold()
            record = {
                "name": attr_name,
                "value": value,
                "line": attr_line,
                "end_line": attr_line,
            }
            if lower_name in {"ref", ":ref", "v-bind:ref"}:
                refs.append({"name": value, "line": attr_line, "end_line": attr_line, "bound": lower_name != "ref"})
                continue
            if lower_name.startswith("#") or lower_name.startswith("v-slot"):
                slot_name = attr_name[1:] if lower_name.startswith("#") else attr_name.split(":", 1)[1] if ":" in attr_name else "default"
                slots.append({"name": slot_name or "default", "line": attr_line, "end_line": attr_line, "component": tag_name})
                continue
            if lower_name.startswith("@") or lower_name.startswith("v-on:"):
                event_name = attr_name[1:] if lower_name.startswith("@") else attr_name.split(":", 1)[1]
                record["name"] = event_name.split(".", 1)[0]
                if component is not None:
                    component["events"].append(record)
                continue
            if lower_name.startswith(":") or lower_name.startswith("v-bind:"):
                prop_name = attr_name[1:] if lower_name.startswith(":") else attr_name.split(":", 1)[1]
                record["name"] = prop_name or "*"
                if component is not None:
                    component["props"].append(record)
                continue
            if lower_name.startswith("v-"):
                directive = {"name": attr_name, "value": value, "line": attr_line, "end_line": attr_line}
                directives.append(directive)
                if component is not None:
                    component["directives"].append(directive)
                continue
            if tag_name.casefold() == "slot" and lower_name == "name":
                slots.append({"name": value or "default", "line": attr_line, "end_line": attr_line, "component": None})

    partial = not bool(block.get("closed"))
    degraded = partial or not vue_grammar
    reason = "unclosed-template-block" if partial else ("vue-grammar-unavailable" if not vue_grammar else None)
    return {
        "parser": "tree-sitter" if vue_grammar else "lexical",
        "degraded": degraded,
        "capability": "DEGRADED" if degraded else "AVAILABLE",
        "completeness": "UNKNOWN" if degraded else "KNOWN",
        "degraded_reason": reason,
        "line": int(block["start_line"]),
        "end_line": int(block["end_line"]),
        "components": components,
        "slots": slots,
        "refs": refs,
        "directives": directives,
    }


def _vue_outline(source: str, relpath: str, detection: str) -> dict[str, object]:
    blocks = _extract_sfc_blocks(source)
    vue_grammar = _vue_grammar_available()
    imports: list[dict[str, object]] = []
    module_statements: list[dict[str, object]] = []
    types: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []
    script_blocks: list[dict[str, object]] = []
    script_degraded = False
    reasons: list[str] = []

    for block in blocks:
        if block["tag"] != "script":
            continue
        body = source[int(block["body_start"]):int(block["body_end"])]
        attrs = block["attrs"]
        lang = str(attrs.get("lang", "")).casefold()
        language = "typescript" if lang in {"", "ts", "typescript"} else ("tsx" if lang == "tsx" else "javascript")
        spec = _OUTLINE_SPECS[language]
        try:
            tree = _parse_source(language, body)
            root = _root_node(tree)
            if root is None or _node_has_error(root):
                raise ValueError("syntax-error")
            script_outline = _tree_outline(body, relpath, language, "extension", root, spec)
        except Exception as exc:  # noqa: BLE001 - one malformed block must not abort the SFC
            script_outline = _lexical_outline(body, relpath, language, "extension", type(exc).__name__)
            script_degraded = True
            reasons.append(f"script:{type(exc).__name__}")
        report_outline_parser_capability(script_outline)
        line_offset = int(block["content_start_line"]) - 1
        _offset_outline_lines(script_outline, line_offset)
        script_outline["file"] = relpath
        imports.extend(script_outline["imports"])
        module_statements.extend(script_outline["module_statements"])
        types.extend(script_outline["types"])
        functions.extend(script_outline["functions"])
        features = _vue_script_features(body, line_offset)
        script_blocks.append({
            "kind": block["kind"],
            "language": language,
            "start_line": block["start_line"],
            "content_start_line": block["content_start_line"],
            "content_end_line": block["content_end_line"],
            "end_line": block["end_line"],
            "degraded": bool(script_outline["degraded"]),
            "outline": script_outline,
            **features,
        })

    template_block = next((block for block in blocks if block["tag"] == "template"), None)
    if template_block is None:
        template = {
            "parser": "lexical",
            "degraded": True,
            "capability": "DEGRADED",
            "completeness": "UNKNOWN",
            "degraded_reason": "missing-template-block",
            "line": 0,
            "end_line": 0,
            "components": [], "slots": [], "refs": [], "directives": [],
        }
        reasons.append("missing-template-block")
    else:
        template = _template_outline(source, template_block, vue_grammar)
        if template["degraded"]:
            reasons.append(str(template["degraded_reason"]))

    if not vue_grammar:
        reasons.append("vue-grammar-unavailable")
    degraded = bool(script_degraded or reasons)
    result: dict[str, object] = {
        "schema": OUTLINE_SCHEMA,
        "file": str(relpath),
        "language": "vue",
        "detection": detection,
        "parser": "tree-sitter" if script_blocks and not script_degraded else "lexical",
        "degraded": degraded,
        "capability": "DEGRADED" if degraded else "AVAILABLE",
        "completeness": "UNKNOWN" if degraded else "KNOWN",
        "imports": imports,
        "module_statements": module_statements,
        "types": types,
        "functions": functions,
        "blocks": [
            {key: value for key, value in block.items() if key not in {"body_start", "body_end", "attrs"}}
            | {"attributes": block["attrs"]}
            for block in blocks
        ],
        "script_blocks": script_blocks,
        "template": template,
    }
    if degraded:
        result["degraded_reason"] = ";".join(dict.fromkeys(reasons))
    return result


def _reported_outline(outline: dict[str, object]) -> dict[str, object]:
    report_outline_parser_capability(outline)
    return outline


def _python_ast_failure_outline(
    relpath: str,
    detection: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema": OUTLINE_SCHEMA,
        "file": str(relpath),
        "language": "python",
        "detection": detection,
        "parser": "ast",
        "degraded": True,
        "capability": "DEGRADED",
        "completeness": "UNKNOWN",
        "degraded_reason": reason,
        "imports": [],
        "module_statements": [],
        "types": [],
        "functions": [],
    }


def _python_ast_fallback(source: str, relpath: str, detection: str) -> dict[str, object]:
    try:
        outline = _ast_python_outline(source, relpath, detection)
    except SyntaxError:
        outline = _lexical_outline(source, relpath, "python", detection, "syntax-error")
    except Exception as exc:  # noqa: BLE001 - keep parser faults distinct from syntax errors
        logger.debug("Python AST outline failed for %s: %s", relpath, type(exc).__name__)
        outline = _python_ast_failure_outline(relpath, detection, "ast-analysis-error")
    return _reported_outline(outline)


# Structure outlines for markup/data files (html, css, json, toml, yaml). Additive: these languages are detected
# here only, never by _detected_language, so detect_language/structural_spans and every code-outline consumer keep
# their behavior; the outline carries "sections" and leaves imports/types/functions empty. A parse failure falls
# back to the original unsupported-language lexical outline.
_STRUCTURE_LANGUAGE_BY_EXTENSION = {
    ".html": "html", ".htm": "html", ".css": "css", ".json": "json",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
}
_STRUCTURE_MAX_RECORDS = 500
_STRUCTURE_MAX_DEPTH = 6
_STRUCTURE_NAME_CHARS = 120
_HTML_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_CSS_AT_RULES = frozenset({"media_statement", "keyframes_statement", "supports_statement", "import_statement",
                           "namespace_statement", "charset_statement", "at_rule"})


def _structure_text(source: bytes, node) -> str:
    return " ".join(_node_text(source, node).split())[:_STRUCTURE_NAME_CHARS]


def _structure_key(source: bytes, node) -> str:
    """A key node's text without its quotes (json string, toml quoted/dotted key, yaml scalar)."""
    if node is None:
        return ""
    text = _structure_text(source, node)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text


def _structure_records(language: str, source: bytes, root) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def add(kind: str, name: str, node, depth: int) -> None:
        if name and len(records) < _STRUCTURE_MAX_RECORDS:
            records.append({
                "kind": kind,
                "name": name,
                "line": _node_position(node, "start_position")[0] + 1,
                "end_line": _node_position(node, "end_position")[0] + 1,
                "depth": depth,
                "node_type": _node_type(node),
                "parser": "tree-sitter",
            })

    def html_tag_and_id(element) -> tuple[str, str]:
        tag, element_id = "", ""
        for child in _node_named_children(element):
            if _node_type(child) not in {"start_tag", "self_closing_tag"}:
                continue
            for part in _node_named_children(child):
                if _node_type(part) == "tag_name":
                    tag = _structure_text(source, part).lower()
                elif _node_type(part) == "attribute":
                    pieces = _node_named_children(part)
                    if pieces and _structure_text(source, pieces[0]).lower() == "id" and len(pieces) > 1:
                        element_id = _structure_key(source, pieces[-1])
            break
        return tag, element_id

    def visit(node, depth: int) -> None:
        if len(records) >= _STRUCTURE_MAX_RECORDS or depth > _STRUCTURE_MAX_DEPTH:
            return
        node_type = _node_type(node)
        child_depth = depth
        if language == "html" and node_type in {"element", "script_element", "style_element"}:
            tag, element_id = html_tag_and_id(node)
            if element_id:
                add("element", f"{tag}#{element_id}", node, depth)
            elif tag in _HTML_HEADINGS:
                text = " ".join(_structure_text(source, c) for c in _node_named_children(node) if _node_type(c) == "text")
                add("heading", f"{tag}: {text}".strip(), node, depth)
            child_depth = depth + 1
        elif language == "css" and node_type == "rule_set":
            selectors = next((c for c in _node_named_children(node) if _node_type(c) == "selectors"), None)
            add("rule", _structure_text(source, selectors) if selectors is not None else "", node, depth)
            child_depth = depth + 1
        elif language == "css" and node_type in _CSS_AT_RULES:
            block = next((c for c in _node_named_children(node) if _node_type(c) == "block"), None)
            end = int(_node_attr(block, "start_byte", 0) or 0) if block is not None else int(_node_attr(node, "end_byte", 0) or 0)
            start = int(_node_attr(node, "start_byte", 0) or 0)
            add("at-rule", " ".join(source[start:end].decode("utf-8", "replace").split()).rstrip(";")[:_STRUCTURE_NAME_CHARS],
                node, depth)
            child_depth = depth + 1
        elif language == "json" and node_type == "pair":
            add("key", _structure_key(source, _node_field(node, "key")), node, depth)
            child_depth = depth + 1
        elif language == "toml" and node_type in {"table", "table_array_element"}:
            key = next((c for c in _node_named_children(node) if _node_type(c) in {"bare_key", "dotted_key", "quoted_key"}), None)
            name = _structure_text(source, key) if key is not None else ""
            add("table" if node_type == "table" else "array-table",
                f"[{name}]" if node_type == "table" else f"[[{name}]]", node, depth)
            child_depth = depth + 1
        elif language == "toml" and node_type == "pair":
            key = next((c for c in _node_named_children(node) if _node_type(c) in {"bare_key", "dotted_key", "quoted_key"}), None)
            add("key", _structure_key(source, key), node, depth)
            child_depth = depth + 1
        elif language == "yaml" and node_type in {"block_mapping_pair", "flow_pair"}:
            add("key", _structure_key(source, _node_field(node, "key")), node, depth)
            child_depth = depth + 1
        for child in _node_named_children(node):
            visit(child, child_depth)

    visit(root, 0)
    return records


def _structure_outline(source: str, relpath: str) -> dict[str, object] | None:
    language = _STRUCTURE_LANGUAGE_BY_EXTENSION.get(Path(relpath).suffix.casefold())
    if language is None:
        return None
    try:
        tree = _parse_source(language, source)
        root = _root_node(tree)
        if root is None or _node_has_error(root):
            return None
        sections = _structure_records(language, source.encode("utf-8", errors="replace"), root)
    except Exception as exc:  # noqa: BLE001 - outline is an analysis boundary
        logger.debug("structure outline failed for %s: %s", relpath, type(exc).__name__)
        return None
    return {
        "schema": OUTLINE_SCHEMA,
        "file": str(relpath),
        "language": language,
        "detection": "extension",
        "parser": "tree-sitter",
        "degraded": False,
        "capability": "AVAILABLE",
        "completeness": "KNOWN" if len(sections) < _STRUCTURE_MAX_RECORDS else "UNKNOWN",
        "imports": [],
        "module_statements": [],
        "types": [],
        "functions": [],
        "sections": sections,
    }


def outline_source(source: str, relpath: str = "") -> dict[str, object]:
    """Return one non-throwing, language-neutral source outline."""
    text = str(source)
    language, detection = _detected_language(str(relpath), text)
    if language == "vue":
        outline = _vue_outline(text, str(relpath), detection)
        return _reported_outline(outline)
    if language == "unknown":
        structure = _structure_outline(text, str(relpath))
        if structure is not None:
            return _reported_outline(structure)
    spec = _OUTLINE_SPECS.get(language)
    if spec is None:
        return _reported_outline(_lexical_outline(
            text, str(relpath), language, detection, "unsupported-language"))
    try:
        tree = _parse_source(language, text)
        root = _root_node(tree)
        has_syntax_error = root is not None and _node_has_error(root)
    except Exception as exc:  # noqa: BLE001 - outline is an analysis boundary
        logger.debug("outline parse failed for %s: %s", relpath, type(exc).__name__)
        if language == "python":
            return _python_ast_fallback(text, str(relpath), detection)
        return _reported_outline(_lexical_outline(
            text, str(relpath), language, detection, type(exc).__name__))
    if root is None or has_syntax_error:
        if language == "python":
            return _python_ast_fallback(text, str(relpath), detection)
        return _reported_outline(_lexical_outline(
            text, str(relpath), language, detection, "syntax-error"))
    return _reported_outline(_tree_outline(
        text, str(relpath), language, detection, root, spec))


def _structural_span_records(
    source: bytes,
    language: str,
    root,
    spec: _OutlineSpec,
    *,
    byte_offset: int = 0,
    line_offset: int = 0,
    owner: str | None = None,
) -> list[dict[str, object]]:
    """Return parser-owned byte spans; degraded outlines never reach this path."""
    records: list[dict[str, object]] = []

    def visit(node, current_owner: str | None = owner) -> None:
        node_type = _node_type(node)
        next_owner = current_owner
        type_kind = _type_kind(node, spec)
        if node_type == "namespace_definition":
            name = _node_name(node, source) or _signature(node, source)
            next_owner = name or current_owner
            kind = "namespace"
        elif type_kind is not None:
            name = _node_name(node, source) or _signature(node, source)
            next_owner = name or current_owner
            kind = "type"
        elif node_type in spec.function_types:
            name = _node_name(node, source) or "<anonymous>"
            function_owner = current_owner
            if language == "go" and function_owner is None:
                receiver = _LEXICAL_RECEIVER_METHOD_RE.search(_signature(node, source))
                if receiver is not None:
                    function_owner = receiver.group("parent")
            kind = "method" if function_owner else "function"
            next_owner = current_owner
        else:
            name = ""
            kind = ""

        if kind:
            start = int(_node_attr(node, "start_byte", 0) or 0) + byte_offset
            end = int(_node_attr(node, "end_byte", start - byte_offset) or 0) + byte_offset
            if end > start:
                record: dict[str, object] = {
                    "kind": kind,
                    "name": name,
                    "owner": current_owner if kind in {"function", "method"} else None,
                    "signature": _signature(node, source),
                    "start": start,
                    "end": end,
                    "line": _node_position(node, "start_position")[0] + 1 + line_offset,
                    "end_line": _node_position(node, "end_position")[0] + 1 + line_offset,
                    "node_type": node_type,
                    "parser": "tree-sitter",
                }
                if language == "go" and kind == "method" and record["owner"] is None:
                    receiver = _LEXICAL_RECEIVER_METHOD_RE.search(str(record["signature"]))
                    if receiver is not None:
                        record["owner"] = receiver.group("parent")
                records.append(record)

        for child in _node_named_children(node):
            visit(child, next_owner)

    for child in _node_named_children(root):
        visit(child)
    return records


def structural_spans(source: str, relpath: str = "") -> dict[str, object]:
    """Resolve exact parser-backed definition/type spans without lexical fallback.

    This API is intentionally stricter than :func:`outline_source`: a degraded
    outline can inform navigation, but it cannot authorize a structural splice.
    Vue SFC block boundaries are handled by the existing balanced block scanner;
    script members still require their actual language parser.
    """
    text = str(source)
    language, detection = _detected_language(str(relpath), text)
    source_bytes = text.encode("utf-8", errors="replace")
    if language == "vue":
        blocks = _extract_sfc_blocks(text)
        if not blocks or any(not bool(block.get("closed")) for block in blocks):
            return {"language": language, "parser": "unavailable", "degraded": True,
                    "detection": detection, "spans": [], "reason": "unclosed-sfc-block"}
        spans: list[dict[str, object]] = []
        script_parser_available = True
        for index, block in enumerate(blocks):
            start_char = text.rfind("<", 0, int(block["body_start"]))
            end_char = text.find(">", int(block["body_end"])) + 1
            close_match = re.search(rf"</\s*{re.escape(str(block['tag']))}\s*>", text[int(block["body_end"]):], re.IGNORECASE)
            if close_match is not None:
                end_char = int(block["body_end"]) + close_match.end()
            spans.append({
                "kind": "sfc-block", "name": f"{block['tag']}[{index}]",
                "owner": None, "signature": text[start_char:min(end_char, len(text))].splitlines()[0],
                "start": len(text[:start_char].encode("utf-8")),
                "end": len(text[:min(end_char, len(text))].encode("utf-8")),
                "line": int(block["start_line"]), "end_line": int(block["end_line"]),
                "node_type": str(block["tag"]), "parser": "sfc-structure",
                "block_index": index, "block_tag": str(block["tag"]),
            })
            if block["tag"] != "script":
                continue
            body = text[int(block["body_start"]):int(block["body_end"])]
            attrs = block["attrs"]
            lang = str(attrs.get("lang", "")).casefold()
            script_language = "typescript" if lang in {"", "ts", "typescript"} else ("tsx" if lang == "tsx" else "javascript")
            spec = _OUTLINE_SPECS[script_language]
            try:
                tree = _parse_source(script_language, body)
                root = _root_node(tree)
                if root is None or _node_has_error(root):
                    raise ValueError("syntax-error")
                body_start = len(text[:int(block["body_start"])].encode("utf-8"))
                spans.extend(_structural_span_records(
                    body.encode("utf-8"), script_language, root, spec,
                    byte_offset=body_start, line_offset=int(block["content_start_line"]) - 1,
                ))
            except Exception as exc:  # noqa: BLE001 - parser availability is evidence
                script_parser_available = False
                spans.append({
                    "kind": "parser-status", "name": f"script[{index}]", "owner": None,
                    "signature": "", "start": 0, "end": 0, "parser": "unavailable",
                    "reason": type(exc).__name__,
                })
        return {
            "language": language, "parser": "sfc-structure",
            "degraded": not script_parser_available, "detection": detection, "spans": spans,
        }
    spec = _OUTLINE_SPECS.get(language)
    if spec is None:
        return {"language": language, "parser": "unavailable", "degraded": True,
                "detection": detection, "spans": [], "reason": "unsupported-language"}
    try:
        tree = _parse_source(language, text)
        root = _root_node(tree)
        if root is None or _node_has_error(root):
            raise ValueError("syntax-error")
        return {
            "language": language, "parser": "tree-sitter", "degraded": False,
            "detection": detection, "spans": _structural_span_records(source_bytes, language, root, spec),
        }
    except Exception as exc:  # noqa: BLE001 - unavailable parsers refuse structural edits
        return {"language": language, "parser": "unavailable", "degraded": True,
                "detection": detection, "spans": [], "reason": type(exc).__name__}


def _python_outline_parser_identity() -> tuple[str, str, str]:
    try:
        import tree_sitter_language_pack as tslp

        tslp.get_parser(_PARSER_NAME.get("python", "python"))
    except Exception:
        return "ast", "AVAILABLE", "stdlib"
    return "tree-sitter", "AVAILABLE", str(getattr(tslp, "__version__", "unknown"))


def outline_capability_key(relpath: str, source: str = "") -> str:
    """Return parser capability identity for outline caches."""

    language, detection = _detected_language(str(relpath), str(source))
    implementation = format(id(_parse_source), "x")
    if language == "vue":
        try:
            capability = "AVAILABLE" if _vue_grammar_available() else "DEGRADED"
        except Exception as exc:  # noqa: BLE001 - capability is evidence
            capability = f"DEGRADED:{type(exc).__name__}"
        return f"vue:{capability}:{detection}:{implementation}"
    spec = _OUTLINE_SPECS.get(language)
    if spec is None:
        return f"{language}:UNSUPPORTED:{detection}:{implementation}"
    if language == "python":
        parser, capability, version = _python_outline_parser_identity()
        return f"{language}:{parser}:{capability}:{version}:{detection}:{implementation}"
    try:
        import tree_sitter_language_pack as tslp

        tslp.get_parser(_PARSER_NAME.get(language, language))
    except Exception as exc:  # noqa: BLE001 - capability is evidence, not a parse failure
        return f"{language}:DEGRADED:{type(exc).__name__}:{implementation}"
    version = str(getattr(tslp, "__version__", "unknown"))
    return f"{language}:AVAILABLE:{version}:{detection}:{implementation}"


def outline_file(file: str | Path) -> dict[str, object]:
    """Read one source file and return an outline without ever raising."""
    path = Path(file)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - file analysis is fail-open
        language, detection = _detected_language(str(path), "")
        return _reported_outline(_lexical_outline(
            "", str(path), language, detection, type(exc).__name__))
    return outline_source(source, str(path))


__all__ = [
    "OUTLINE_SCHEMA", "detect_language", "dispatch_boundaries_in_file", "outline_capability_key",
    "structural_spans",
    "outline_file", "outline_source",
]
