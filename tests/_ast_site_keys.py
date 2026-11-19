"""Line-independent AST identities for reviewed engine sites in tests."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BroadExceptSite:
    module: str
    scope: str
    clause: str
    shape: str

    @property
    def label(self) -> str:
        return f"{self.module}.{self.scope}: except {self.clause} [site AST {self.shape}]"


@dataclass(frozen=True)
class ProtocolLiteralSite:
    module: str
    scope: str
    legacy_members: tuple[str, ...]
    shape: str = ""

    @property
    def label(self) -> str:
        members = ", ".join(self.legacy_members)
        suffix = f" [literal AST {self.shape}]" if self.shape else ""
        return f"{self.module}.{self.scope}: literal {{{members}}}{suffix}"


def _read_source(module_path: Path, source_text: str | None) -> str:
    return source_text if source_text is not None else module_path.read_text(encoding="utf-8")


def _fingerprint(node: ast.AST) -> str:
    shape = ast.dump(node, include_attributes=False)
    return hashlib.sha256(shape.encode("utf-8")).hexdigest()[:12]


def _replace_node_segment(source_text: str, node: ast.AST, replacement: str) -> str:
    if node.lineno is None or node.end_lineno is None \
            or node.col_offset is None or node.end_col_offset is None:
        raise AssertionError("AST node has no source span")
    source_lines = source_text.encode("utf-8").splitlines(keepends=True)
    start = sum(len(line) for line in source_lines[:node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in source_lines[:node.end_lineno - 1]) + node.end_col_offset
    encoded = source_text.encode("utf-8")
    return (encoded[:start] + replacement.encode("utf-8") + encoded[end:]).decode("utf-8")


def _scope_names(tree: ast.Module) -> dict[ast.AST, str]:
    scopes: dict[ast.AST, str] = {}

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parents = (*parents, node.name)
        scopes[node] = ".".join(parents) if parents else "<module>"
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, ())
    return scopes


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _protocol_literal_site(
        module_path: Path,
        node: ast.AST,
        scope: str,
        splice_name: str,
        sentinel: str,
) -> ProtocolLiteralSite | None:
    if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return None
    splices_protocol = any(
        isinstance(item, ast.Starred)
        and isinstance(item.value, ast.Name)
        and item.value.id == splice_name
        for item in node.elts
    )
    string_members = tuple(sorted({
        item.value for item in node.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }))
    if not splices_protocol or sentinel not in string_members:
        return None

    supported_shape = all(
        (isinstance(item, ast.Constant) and isinstance(item.value, str))
        or (isinstance(item, ast.Starred)
            and isinstance(item.value, ast.Name)
            and item.value.id == splice_name)
        for item in node.elts
    )
    return ProtocolLiteralSite(
        module_path.stem,
        scope,
        string_members,
        "" if supported_shape else _fingerprint(node),
    )


def _is_broad_except(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    return any(
        isinstance(node, ast.Name) and node.id in {"Exception", "BaseException"}
        for node in ast.walk(handler.type)
    )


def _except_site(
        module_path: Path, handler: ast.ExceptHandler, scope: str, try_node: ast.AST,
) -> BroadExceptSite:
    clause = ast.unparse(handler.type) if handler.type is not None else "bare"
    try_body = getattr(try_node, "body", None)
    if not isinstance(try_body, list) or not all(isinstance(item, ast.stmt) for item in try_body):
        raise AssertionError("except handler has no containing try body")
    try_shape = ast.dump(ast.Module(body=try_body, type_ignores=[]), include_attributes=False)
    handler_shape = ast.dump(handler, include_attributes=False)
    shape = hashlib.sha256(f"{try_shape}\0{handler_shape}".encode("utf-8")).hexdigest()[:12]
    return BroadExceptSite(module_path.stem, scope, clause, shape)


def find_broad_except_sites(
        module_path: Path, source_text: str | None = None) -> tuple[BroadExceptSite, ...]:
    """Find broad except clauses by scope and their containing try AST, never by line."""
    tree = ast.parse(_read_source(module_path, source_text), filename=str(module_path))
    scopes = _scope_names(tree)
    parents = _parents(tree)
    found: list[BroadExceptSite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not _is_broad_except(node):
            continue
        try_node = parents.get(node)
        if try_node is None:
            raise AssertionError("except handler is missing its try construct")
        try_body = getattr(try_node, "body", None)
        if not isinstance(try_body, list) or not try_body:
            raise AssertionError("except handler has no containing try body")
        found.append(_except_site(module_path, node, scopes[node], try_node))
    return tuple(sorted(found, key=lambda site: site.label))


def rename_broad_except_site(
        module_path: Path,
        source_text: str,
        target: BroadExceptSite,
        replacement: str = "LookupError",
) -> str:
    """Rename one located broad clause in a source copy for mutation proofs."""
    tree = ast.parse(source_text, filename=str(module_path))
    scopes = _scope_names(tree)
    parents = _parents(tree)
    matches: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not _is_broad_except(node):
            continue
        try_node = parents.get(node)
        if try_node is not None and _except_site(module_path, node, scopes[node], try_node) == target:
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(f"{target.label}: expected one mutable clause, found {len(matches)}")

    segment = ast.get_source_segment(source_text, matches[0])
    if not segment:
        raise AssertionError(f"{target.label}: could not read source segment")
    old_clause = "except:" if target.clause == "bare" else f"except {target.clause}"
    new_clause = f"except {replacement}"
    changed_segment = segment.replace(old_clause, new_clause, 1)
    if changed_segment == segment:
        raise AssertionError(f"{target.label}: source clause did not match its AST key")
    return _replace_node_segment(source_text, matches[0], changed_segment)


def find_protocol_literal_sites(
        module_path: Path,
        source_text: str | None = None,
        *,
        splice_name: str = "PROTOCOL_EDIT_TOOL_NAMES",
        sentinel: str = "str_replace",
) -> tuple[ProtocolLiteralSite, ...]:
    """Find protocol-spliced literal containers by scope and exact legacy members."""
    tree = ast.parse(_read_source(module_path, source_text), filename=str(module_path))
    scopes = _scope_names(tree)
    found: list[ProtocolLiteralSite] = []
    for node in ast.walk(tree):
        site = _protocol_literal_site(module_path, node, scopes[node], splice_name, sentinel)
        if site is not None:
            found.append(site)
    return tuple(sorted(found, key=lambda site: site.label))


def rename_protocol_literal_member(
        module_path: Path,
        source_text: str,
        target: ProtocolLiteralSite,
        old_member: str,
        new_member: str,
        *,
        splice_name: str = "PROTOCOL_EDIT_TOOL_NAMES",
        sentinel: str = "str_replace",
) -> str:
    """Rename one member in a located literal copy for mutation proofs."""
    tree = ast.parse(source_text, filename=str(module_path))
    scopes = _scope_names(tree)
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        site = _protocol_literal_site(module_path, node, scopes[node], splice_name, sentinel)
        if site == target:
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(f"{target.label}: expected one mutable literal, found {len(matches)}")

    segment = ast.get_source_segment(source_text, matches[0])
    if not segment:
        raise AssertionError(f"{target.label}: could not read literal source segment")
    old_tokens = (f'"{old_member}"', f"'{old_member}'")
    changed_segment = segment
    for token in old_tokens:
        if token in segment:
            quote = token[0]
            changed_segment = segment.replace(token, f"{quote}{new_member}{quote}", 1)
            break
    if changed_segment == segment:
        raise AssertionError(f"{target.label}: member {old_member!r} was not in the literal")
    return _replace_node_segment(source_text, matches[0], changed_segment)
