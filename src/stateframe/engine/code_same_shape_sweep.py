"""Small Python code-pack for the same-shape sibling sweep.

The kernel owns the evidence and obligation vocabulary.  This module only
turns a typed semantic signature into a bounded Python AST/token query and
returns source locations as behavioral instances.  It deliberately does not
execute product code or mutate the checkout.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import tokenize
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_OPERATIONS = frozenset({
    "attribute_read", "call", "subscript_read", "text",
})
MAX_QUERY_TEXT = 240
MAX_SITE_SNIPPET = 240
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".pytest_cache", ".ruff_cache", "__pycache__",
    ".mutation", ".test-tmp",
})


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _bounded_text(value: object, limit: int = MAX_QUERY_TEXT) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _bounded_strings(value: object, *, limit: int = 12,
                     item_limit: int = 120) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text[:item_limit])
    return tuple(result[:limit])


@dataclass(frozen=True, slots=True)
class PythonSite:
    """One match, identified by behavior rather than a changed line."""

    path: str
    line: int
    column: int
    end_line: int
    end_column: int
    qualname: str
    operation: str
    subject: str
    snippet: str
    behavior_id: str
    score: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "qualname": self.qualname,
            "operation": self.operation,
            "subject": self.subject,
            "snippet": self.snippet,
            "behavior_id": self.behavior_id,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class CompiledPythonQuery:
    operation: str
    contract: str
    attribute: str | None
    callee: str | None
    receiver: str | None
    query_text: str
    query_digest: str
    roles: tuple[str, ...] = ()
    safe_lookalikes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "language": "python",
            "operation": self.operation,
            "contract": self.contract,
            "attribute": self.attribute,
            "callee": self.callee,
            "receiver": self.receiver,
            "query_text": self.query_text,
            "query_digest": self.query_digest,
            "roles": list(self.roles),
            "safe_lookalikes": list(self.safe_lookalikes),
        }


@dataclass(frozen=True, slots=True)
class PythonPatternValidation:
    accepted: bool
    reason: str
    base_matches: int
    candidate_matches: int
    repaired_matches: int
    safe_lookalike_matches: int
    base_sites: tuple[PythonSite, ...] = ()
    candidate_sites: tuple[PythonSite, ...] = ()
    repaired_site_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "base_matches": self.base_matches,
            "candidate_matches": self.candidate_matches,
            "repaired_matches": self.repaired_matches,
            "safe_lookalike_matches": self.safe_lookalike_matches,
            "base_sites": [site.to_dict() for site in self.base_sites],
            "candidate_sites": [site.to_dict() for site in self.candidate_sites],
            "repaired_site_ids": list(self.repaired_site_ids),
        }


def _normalise_operation(value: object) -> str:
    operation = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "attribute": "attribute_read",
        "attribute_access": "attribute_read",
        "attribute_read_on_object": "attribute_read",
        "callsite": "call",
        "subscript": "subscript_read",
        "grep": "text",
        "literal": "text",
    }.get(operation, operation)


def _signature_value(signature: object, *names: str, default: object = None) -> object:
    if isinstance(signature, Mapping):
        for name in names:
            if name in signature and signature[name] is not None:
                return signature[name]
    for name in names:
        if hasattr(signature, name):
            value = getattr(signature, name)
            if value is not None:
                return value
    return default


def compile_python_query(signature: object) -> CompiledPythonQuery:
    """Compile a typed signature into a deterministic AST/token query.

    A syntax-only proposal is not enough: the contract is carried into the
    compiled query and later into every review obligation.  Unknown operations
    fail closed so a model cannot silently broaden the scan to arbitrary text.
    """
    contract = str(_signature_value(signature, "contract", "violated_contract",
                                    default="")).strip()
    if not contract:
        raise ValueError("same-shape signature requires a non-empty contract")
    if len(contract) > 400:
        raise ValueError("same-shape signature contract is too long")
    operation = _normalise_operation(_signature_value(
        signature, "operation", "defect_kind", "shape", "node_type", default=""))
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported same-shape operation: {operation!r}")
    attribute = _signature_value(signature, "attribute", "field", "member")
    callee = _signature_value(signature, "callee", "function", "call")
    receiver = _signature_value(signature, "receiver", "receiver_name", "object")
    query = _signature_value(signature, "query", "literal", "grep")
    attribute = str(attribute).strip() if attribute is not None else None
    callee = str(callee).strip() if callee is not None else None
    receiver = str(receiver).strip() if receiver is not None else None
    query = str(query).strip() if query is not None else None
    if operation == "attribute_read" and attribute and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", attribute):
        raise ValueError("attribute must be a Python identifier")
    if operation == "call" and callee and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.]*", callee):
        raise ValueError("callee must be a dotted Python name")
    if operation == "text" and not query:
        raise ValueError("text same-shape signatures require query text")
    if operation == "attribute_read" and not attribute:
        raise ValueError("attribute_read signatures require an attribute")
    if operation == "call" and not callee:
        raise ValueError("call signatures require a callee")
    roles = _bounded_strings(_signature_value(signature, "roles", default=()))
    safe = _bounded_strings(_signature_value(
        signature, "safe_lookalikes", "safe_receivers", default=()))
    if operation == "attribute_read":
        query_text = f"{receiver or '$OBJECT'}.{attribute}"
    elif operation == "call":
        query_text = f"{callee}(...)"
    elif operation == "subscript_read":
        query_text = f"{receiver or '$OBJECT'}[...]"
    else:
        query_text = _bounded_text(query)
    query_text = _bounded_text(query_text)
    identity = {
        "operation": operation, "contract": contract, "attribute": attribute,
        "callee": callee, "receiver": receiver, "query_text": query_text,
        "roles": roles, "safe_lookalikes": safe,
    }
    return CompiledPythonQuery(
        operation=operation, contract=contract, attribute=attribute,
        callee=callee, receiver=receiver, query_text=query_text,
        query_digest=_digest(identity), roles=roles, safe_lookalikes=safe,
    )


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _source_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError):
        return ast.dump(node, annotate_fields=False, include_attributes=False)


def _callee_text(node: ast.Call) -> str:
    return _source_text(node.func)


def _safe_match(query: CompiledPythonQuery, node: ast.AST, subject: str) -> bool:
    if not query.safe_lookalikes:
        return False
    root = _root_name(node.value) if isinstance(node, ast.Attribute) else None
    values = {subject, root or ""}
    values.update(part for part in subject.split(".") if part)
    return any(item in values for item in query.safe_lookalikes)


def _matches(query: CompiledPythonQuery, node: ast.AST) -> bool:
    if query.operation == "attribute_read":
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            return False
        if query.attribute and node.attr != query.attribute:
            return False
        if query.receiver:
            root = _root_name(node.value)
            if root != query.receiver and _source_text(node.value) != query.receiver:
                return False
        return not _safe_match(query, node, _source_text(node))
    if query.operation == "call":
        if not isinstance(node, ast.Call):
            return False
        return _callee_text(node) == query.callee
    if query.operation == "subscript_read":
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
            return False
        if query.receiver:
            root = _root_name(node.value)
            if root != query.receiver and _source_text(node.value) != query.receiver:
                return False
        return not _safe_match(query, node, _source_text(node))
    return False


class _SiteVisitor(ast.NodeVisitor):
    def __init__(self, source: str, path: str, query: CompiledPythonQuery):
        self.source = source
        self.path = path
        self.query = query
        self.scopes: list[str] = []
        self.ordinals: dict[tuple[str, str, str], int] = {}
        self.sites: list[PythonSite] = []
        self.lines = source.splitlines()

    @property
    def qualname(self) -> str:
        return ".".join(self.scopes) if self.scopes else "<module>"

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scopes.append(name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def generic_visit(self, node: ast.AST) -> None:
        if _matches(self.query, node):
            subject = _source_text(node)
            key = (self.qualname, self.query.operation, subject)
            ordinal = self.ordinals.get(key, 0)
            self.ordinals[key] = ordinal + 1
            line = int(getattr(node, "lineno", 1))
            column = int(getattr(node, "col_offset", 0)) + 1
            end_line = int(getattr(node, "end_lineno", line))
            end_column = int(getattr(node, "end_col_offset", column - 1)) + 1
            snippet = (self.lines[line - 1].strip()
                       if 0 < line <= len(self.lines) else subject)
            snippet = snippet[:MAX_SITE_SNIPPET]
            behavior_id = _digest({
                "path": self.path, "qualname": self.qualname,
                "operation": self.query.operation, "subject": subject,
                "ordinal": ordinal,
            })
            score = 1 + (2 if self.query.receiver else 0) + (2 if self.query.attribute else 0)
            self.sites.append(PythonSite(
                path=self.path, line=line, column=column, end_line=end_line,
                end_column=end_column, qualname=self.qualname,
                operation=self.query.operation, subject=subject,
                snippet=snippet, behavior_id=behavior_id, score=score,
            ))
        super().generic_visit(node)


def _parse(source: str, *, path: str) -> ast.AST:
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"cannot scan {path}: syntax error at line {error.lineno}") from error


def scan_python_source(source: str, *, path: str,
                       query: CompiledPythonQuery) -> tuple[PythonSite, ...]:
    if not isinstance(source, str):
        raise TypeError("source must be text")
    if query.operation == "text":
        return _scan_text_source(source, path=path, query=query)
    tree = _parse(source, path=path)
    visitor = _SiteVisitor(source, path, query)
    visitor.visit(tree)
    return tuple(visitor.sites)


def _scan_text_source(source: str, *, path: str,
                      query: CompiledPythonQuery) -> tuple[PythonSite, ...]:
    """Literal/grep channel that ignores comments and string contents."""
    lines = source.splitlines()
    try:
        tokens = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        return ()
    by_line: dict[int, list[tokenize.TokenInfo]] = {}
    for token in tokens:
        by_line.setdefault(token.start[0], []).append(token)
    found: list[PythonSite] = []
    ordinal = 0
    needle = query.query_text
    for line_number, line in enumerate(lines, 1):
        for match in re.finditer(re.escape(needle), line):
            start, end = match.span()
            blocked = any(
                token.type in {tokenize.COMMENT, tokenize.STRING}
                and token.start[1] < end and token.end[1] > start
                for token in by_line.get(line_number, ())
            )
            if blocked:
                continue
            subject = line[start:end]
            behavior_id = _digest({
                "path": path, "qualname": "<module>",
                "operation": query.operation, "subject": subject,
                "ordinal": ordinal,
            })
            ordinal += 1
            found.append(PythonSite(
                path=path, line=line_number, column=start + 1,
                end_line=line_number, end_column=end + 1,
                qualname="<module>", operation=query.operation,
                subject=subject, snippet=line[:MAX_SITE_SNIPPET],
                behavior_id=behavior_id, score=1,
            ))
    return tuple(found)


def _normal_path(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def scan_python_workspace(root: str | Path, query: CompiledPythonQuery, *,
                          overrides: Mapping[str, str] | None = None,
                          include_paths: Sequence[str] = (),
                          maximum: int = 256) -> tuple[PythonSite, ...]:
    """Scan the current candidate, with an in-memory overlay for its edit."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError("maximum must be a positive integer")
    checkout = Path(root).resolve()
    if not checkout.is_dir():
        raise ValueError("same-shape scan root must be a directory")
    overlay = {_normal_path(path): source for path, source in (overrides or {}).items()}
    paths: dict[str, Path] = {}
    for path in checkout.rglob("*.py"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(checkout)
        if any(part in _SKIP_DIRS or part.startswith(".mutation")
               for part in relative.parts):
            continue
        paths[relative.as_posix()] = path
    for raw in include_paths:
        relative = _normal_path(raw)
        if relative:
            paths.setdefault(relative, checkout / relative)
    for relative in overlay:
        paths.setdefault(relative, checkout / relative)
    sites: list[PythonSite] = []
    for relative in sorted(paths):
        source = overlay.get(relative)
        path = paths[relative]
        if source is None:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
        try:
            sites.extend(scan_python_source(source, path=relative, query=query))
        except ValueError:
            # A broken sibling is not silently called a match.  The host's
            # separate import/syntax checks own that evidence.
            continue
        if len(sites) >= maximum:
            break
    return tuple(sites[:maximum])


def _anchor_value(anchor: object, name: str, default: object = None) -> object:
    if isinstance(anchor, Mapping):
        return anchor.get(name, default)
    return getattr(anchor, name, default)


def _anchor_matches(site: PythonSite, *, path: str, anchor: object) -> bool:
    anchor_path = _anchor_value(anchor, "path")
    if anchor_path and _normal_path(anchor_path) != _normal_path(path):
        return False
    line = _anchor_value(anchor, "line")
    if type(line) is int and site.line != line:
        return False
    column = _anchor_value(anchor, "column")
    if type(column) is int and site.column != column:
        return False
    qualname = _anchor_value(anchor, "qualname")
    if qualname and site.qualname != str(qualname):
        return False
    text = _anchor_value(anchor, "text")
    if text and str(text) not in site.snippet and str(text) not in site.subject:
        return False
    return True


def validate_python_query(
    query: CompiledPythonQuery,
    base_source: str,
    candidate_source: str,
    *,
    path: str,
    anchor: object = None,
    safe_lookalike_sources: Iterable[str] = (),
) -> PythonPatternValidation:
    """Require a real base match and reject a query that still matches its fix."""
    try:
        base_sites = scan_python_source(base_source, path=path, query=query)
    except (TypeError, ValueError) as error:
        return PythonPatternValidation(False, f"base scan failed: {error}", 0, 0, 0, 0)
    try:
        candidate_sites = scan_python_source(candidate_source, path=path, query=query)
    except (TypeError, ValueError) as error:
        return PythonPatternValidation(False, f"candidate scan failed: {error}",
                                       len(base_sites), 0, 0, 0,
                                       base_sites=base_sites)
    safe_sites: list[PythonSite] = []
    for index, source in enumerate(safe_lookalike_sources):
        try:
            safe_sites.extend(scan_python_source(source, path=f"<safe:{index}>", query=query))
        except (TypeError, ValueError):
            continue
    if not base_sites:
        return PythonPatternValidation(False, "pattern did not match the base occurrence",
                                       0, len(candidate_sites), 0, len(safe_sites),
                                       candidate_sites=candidate_sites)
    if anchor is not None:
        repaired = tuple(site for site in candidate_sites
                         if _anchor_matches(site, path=path, anchor=anchor))
    elif len(candidate_sites) >= len(base_sites):
        # Without a typed anchor, retaining the full base cardinality is not
        # enough evidence that the repaired occurrence disappeared.
        repaired = candidate_sites[:len(base_sites)]
    else:
        repaired = ()
    if safe_sites:
        reason = "pattern matches a safe lookalike"
    elif repaired:
        reason = "pattern still matches the repaired occurrence"
    else:
        reason = "pattern matches the base occurrence and rejects the repaired one"
    return PythonPatternValidation(
        accepted=not safe_sites and not repaired,
        reason=reason,
        base_matches=len(base_sites), candidate_matches=len(candidate_sites),
        repaired_matches=len(repaired), safe_lookalike_matches=len(safe_sites),
        base_sites=base_sites, candidate_sites=candidate_sites,
        repaired_site_ids=tuple(site.behavior_id for site in repaired),
    )


__all__ = [
    "CompiledPythonQuery", "PythonPatternValidation", "PythonSite",
    "SUPPORTED_OPERATIONS", "compile_python_query", "scan_python_source",
    "scan_python_workspace", "validate_python_query",
]
