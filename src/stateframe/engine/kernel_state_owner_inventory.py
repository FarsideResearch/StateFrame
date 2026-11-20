"""Q2 (E2): host state-owner inventory for a stateful ticket.

When the ticket's state is a persisted field (a column, struct field, class
attribute) the expert first lists who declares it, who reads it, who writes it,
and which helpers produce its value, with exact host line numbers.  The model
then designs the transition instead of searching for a writer that does not
exist.  Everything here is a host grep/outline fact over the parent tree,
computed once per run (the inventory describes the PARENT) and rendered beside
the other site facts, so it counts against the request budget.

Owner fields are resolved language-generically: type spans come from the
shared parser outline (``repo_intel_core.ts_dispatch.outline_source``;
Python uses stdlib AST when tree-sitter is unavailable; Go, Rust, TS/JS, Java,
and PHP keep their tree-sitter parser path. Field declarations are lines inside
a type span and outside its methods.

Trigger: a ticket literal (code identifier) names a declared field, OR the entry
symbol is absent (E1 ``NOT_FOUND``/``REFERENCED_ONLY``/``new_symbol``) and a
ticket word pair (``printable resource`` -> ``printable_resource*``) names one.

``STATEFRAME_STATE_OWNER_INVENTORY`` (default ``on``).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import sys
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from importlib.machinery import ExtensionFileLoader
from importlib.util import find_spec
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import kernel_parse_cache as _parse_cache

if TYPE_CHECKING:
    from stateframe.workspace_start_snapshot import StartupObservation
    from kernel_workspace_state import WorkspaceState
    from kernel_workspace_tree import Directory, File

ENV = "STATEFRAME_STATE_OWNER_INVENTORY"
HEADER = "[STATE OWNER INVENTORY"
SOURCE_SUFFIXES = frozenset({".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs",
                             ".java", ".kt", ".php", ".rb", ".cs", ".vue"})
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
                        ".mypy_cache", ".pytest_cache", "vendor", "migrations", "alembic"})
_MAX_FILES = 600
_MAX_FILE_BYTES = 600_000
_MAX_FIELDS = 6
_MAX_SITES = 6
_MAX_HELPERS = 4
_HASH_HELPER_WORDS = frozenset({"sha1", "sha256", "sha384", "sha512", "digest", "checksum"})
_MAX_CHARS = 2400
_MAX_CONTEXT_SITES = 8
_INVENTORY_CACHE_KIND = "owner-inventory"
_INVENTORY_CACHE_VERSION = 1
_MAX_MEMO_ENTRIES = 4
_MAX_MEMO_CHARS = 256_000
_MEMO: OrderedDict[str, str] = OrderedDict()
_MEMO_LOCK = RLock()
_GENERIC_DECL = re.compile(
    r"^\s*(?:(?:pub(?:\([a-z]+\))?|public|private|protected|internal|readonly|static|final|var|val"
    r"|let|const|declare|override)\s+)*\$?(?P<name>[A-Za-z_]\w*)\s*\??\s*[:=](?!=)")
_GO_FIELD = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s+[\*\[\]\w.]")
_TYPED_DECL = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|static|final|readonly|transient|volatile)\s+)*"
    r"[\w<>\[\],.?]+\s+(?P<name>[A-Za-z_]\w*)\s*(?:[;=]|\{\s*get)")
_DEF_LINE = re.compile(r"^\s*(?:async\s+)?(?:def|fn|func|function)\b|^\s*(?:pub\s+)?fn\b")
_TYPE_LINE = re.compile(
    r"^\s*(?:export\s+)?(?:type\s+(?P<struct>[A-Za-z_$][A-Za-z0-9_$]*)\s+struct|"
    r"interface\s+(?P<interface>[A-Za-z_$][A-Za-z0-9_$]*)|"
    r"class\s+(?P<class>[A-Za-z_$][A-Za-z0-9_$]*))\b"
)
_PAIR_WORD = re.compile(r"[A-Za-z][a-z]{2,}")


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value


@dataclass(frozen=True)
class Site:
    path: str
    line: int
    text: str


@dataclass
class OwnerField:
    name: str
    path: str
    line: int
    text: str
    owner: str
    readers: list[Site] = dc_field(default_factory=list)
    writers: list[Site] = dc_field(default_factory=list)
    contexts: list[Site] = dc_field(default_factory=list)


@dataclass
class Helper:
    name: str
    path: str
    line: int
    text: str
    callers: list[Site] = dc_field(default_factory=list)
    importers: list[Site] = dc_field(default_factory=list)


@dataclass
class Inventory:
    candidates: tuple[str, ...]
    fields: list[OwnerField]
    helpers: list[Helper]
    revision: str
    files_scanned: int
    complete: bool = True
    skipped_files: tuple[dict[str, object], ...] = ()
    dependency_revision: str = ""
    identities: dict[str, dict[str, object]] = dc_field(default_factory=dict)
    parser_capabilities: dict[str, str] = dc_field(default_factory=dict)

    @property
    def writer_count(self) -> int:
        return sum(len(item.writers) for item in self.fields)

    @property
    def writer_status(self) -> str:
        return "KNOWN" if self.complete else "UNKNOWN"

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (self.dependency_revision,
                tuple(sorted(self.parser_capabilities.items())),
                self.complete,
                tuple(sorted(
                    (str(item.get("path", "")), str(item.get("status", "")),
                     str(item.get("reason", "")))
                    for item in self.skipped_files
                )),
                tuple(sorted(
                    (path, str(value.get("current_revision", "")),
                     str(value.get("parent_revision", "")), str(value.get("status", "")))
                    for path, value in self.identities.items()
                )))

    def summary(self) -> dict[str, object]:
        return {"candidates": list(self.candidates), "files_scanned": self.files_scanned,
                "fields": [f"{item.path}:{item.line}:{item.name}" for item in self.fields],
                "readers": sum(len(item.readers) for item in self.fields),
                "writers": self.writer_count,
                "writer_status": self.writer_status,
                "complete": self.complete,
                "skipped_files": [dict(item) for item in self.skipped_files],
                "dependency_revision": self.dependency_revision,
                "identities": {key: dict(value) for key, value in self.identities.items()},
                "parser_capabilities": dict(self.parser_capabilities),
                "helpers": {item.name: len(item.callers) for item in self.helpers},
                "revision": self.revision}


# ------------------------------------------------------------------ candidates

def owner_candidates(ticket: str, focus: str = "", *, pairs: bool = False) -> tuple[str, ...]:
    """Identifier-shaped ticket literals, plus snake-joined word pairs when ``pairs``."""
    from kernel_request_source_window import _WORD_STOPWORDS, literal_tokens
    result: list[str] = []
    for token in literal_tokens(ticket, focus):
        value = token.rstrip("_")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", value):
            continue
        if len(value.rsplit(".", 1)[-1]) < 4:
            continue
        result.append(value)
        terminal = value.rsplit(".", 1)[-1]
        if terminal != value:
            result.append(terminal)
    for wildcard in re.findall(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)_\*", f"{ticket}\n{focus}"):
        result.append(wildcard)
    if pairs:
        for sentence in re.split(r"[.;:\n]", f"{ticket}\n{focus}"):
            words = [word.lower() for word in _PAIR_WORD.findall(sentence)]
            for first, second in zip(words, words[1:]):
                if first in _WORD_STOPWORDS or second in _WORD_STOPWORDS:
                    continue
                result.append(f"{first}_{second}")
    return tuple(dict.fromkeys(item for item in result if item))[:24]


def _matches(name: str, candidates: Sequence[str]) -> bool:
    folded = name.casefold()
    return any(folded == cand.casefold() or folded.startswith(cand.casefold() + "_")
               for cand in candidates)


# ------------------------------------------------------------------ scanning

@dataclass(frozen=True)
class ProductFileScan:
    paths: tuple[str, ...]
    skipped_files: tuple[dict[str, object], ...]
    complete: bool


def scan_product_files(root: Path, *, limit: int = _MAX_FILES) -> ProductFileScan:
    from kernel_request_source_window import _FORBIDDEN_PARTS, _TEST_PATH
    base = Path(root).resolve()
    found: list[str] = []
    skipped: list[dict[str, object]] = []
    truncated = False
    for current, dirs, files in os.walk(base):
        dirs[:] = sorted(item for item in dirs if not item.startswith(".")
                         and item not in _SKIP_DIRS and item.casefold() not in _FORBIDDEN_PARTS)
        for name in sorted(files):
            if Path(name).suffix not in SOURCE_SUFFIXES:
                continue
            relative = (Path(current) / name).relative_to(base).as_posix()
            if _TEST_PATH.search(relative):
                continue
            if len(found) >= limit:
                truncated = True
                skipped.append({"path": relative, "status": "PARTIAL",
                                "reason": "product file scan limit reached"})
                break
            found.append(relative)
        if truncated:
            break
    return ProductFileScan(tuple(found[:limit]), tuple(skipped), not truncated)


def product_files(root: Path, *, limit: int = _MAX_FILES) -> list[str]:
    """Compatibility projection; completeness lives in ``scan_product_files``."""
    return list(scan_product_files(root, limit=limit).paths)


def _read(root: Path, relative: str, *, observed: File | None = None) -> str | None:
    path = Path(root) / relative
    try:
        if observed is not None:
            if observed.size > _MAX_FILE_BYTES:
                return None
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    return None
                data = stream.read(observed.size + 1)
            if len(data) != observed.size or hashlib.sha256(data).digest() != observed.sha256:
                return None
            return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _compact_python_outline(tree: ast.Module, source: str) -> dict[str, object] | None:
    """Project just the helper rows needed from a successfully parsed Python AST."""
    from repo_intel_core.ts_dispatch import _python_ast_definition_line

    # The normal fallback visits every node recursively. Use it for deep trees
    # so any failure near its recursion limit still determines completeness.
    frames = 0
    frame = sys._getframe()
    while frame is not None:
        frames += 1
        frame = frame.f_back
    depth_limit = min(128, (sys.getrecursionlimit() - frames) // 2)
    functions = []
    lines = None

    def visit(node: ast.AST, parent: str | None, depth: int) -> None:
        nonlocal lines
        if depth >= depth_limit:
            raise RecursionError
        kind = type(node)
        if kind is ast.Name or kind is ast.Constant:
            return
        if kind is ast.ClassDef:
            parent = node.name
        elif kind is ast.FunctionDef or kind is ast.AsyncFunctionDef:
            if lines is None:
                lines = source.splitlines()
            functions.append({
                "name": node.name,
                "parent": parent,
                "line": _python_ast_definition_line(lines, node),
            })
        for field in node._fields:
            value = getattr(node, field, None)
            if isinstance(value, ast.AST):
                visit(value, parent, depth + 1)
            elif type(value) is list:
                for child in value:
                    if isinstance(child, ast.AST):
                        visit(child, parent, depth + 1)

    try:
        for statement in tree.body:
            visit(statement, None, 1)
    except RecursionError:
        return None
    return {"language": "python", "parser": "ast", "capability": "AVAILABLE",
            "completeness": "KNOWN", "functions": functions}


def _python_definition_outline(path: str, source: str) -> Mapping[str, object]:
    from repo_intel_core.ts_dispatch import (
        _OUTLINE_SPECS, _lexical_outline, _node_has_error, _parse_source,
        _python_ast_failure_outline, _python_ast_fallback, _root_node, _tree_outline,
        report_outline_parser_capability,
    )

    # Attempt the same parser as the full outline, even when the capability
    # probe predicts AST fallback. A successful grammar parse still wins.
    try:
        tree = _parse_source("python", source)
        root = _root_node(tree)
        valid = root is not None and not _node_has_error(root)
    except Exception:
        valid = False
    if valid:
        outline = _tree_outline(source, path, "python", "extension", root, _OUTLINE_SPECS["python"])
    else:
        try:
            outline = _compact_python_outline(_parse_cache.parse(source), source)
        except SyntaxError:
            outline = _lexical_outline(source, path, "python", "extension", "syntax-error")
        except Exception:
            outline = _python_ast_failure_outline(path, "extension", "ast-analysis-error")
        if outline is None:
            return _python_ast_fallback(source, path, "extension", definitions_only=True)
    report_outline_parser_capability(outline)
    return outline


class _UncacheableOutline(Exception):
    def __init__(self, outline: Mapping[str, object]):
        self.outline = outline


def _outline(path: str, source: str, *, policy: str | None = None,
             lexical_miss: bool = False, exact: tuple[object, ...] | None = None) -> Mapping[str, object]:
    try:
        from repo_intel_core.ts_dispatch import (
            outline_capability_key,
            outline_source,
            report_outline_parser_capability,
        )
        if policy is None:
            policy = outline_capability_key(path, source)
        compact = (lexical_miss and path.endswith(".py") and sys.implementation.name == "cpython"
                   and policy.startswith("python:ast:AVAILABLE:stdlib:extension:"))
        if exact is None:
            outline = _parse_cache.derived_source(
                ("owner_inventory.definitions", policy, compact), source, path,
                lambda text: (_python_definition_outline(path, text) if compact else
                              outline_source(text, path, definitions_only=True)),
            )
            report_outline_parser_capability(outline)
            return outline
        def build(text: str) -> Mapping[str, object]:
            outline = (_python_definition_outline(path, text) if compact else
                       outline_source(text, path, definitions_only=True))
            if exact and path.endswith(".py") and (outline.get("parser") != exact[0] or outline.get("degraded")
                          or outline.get("completeness") != "KNOWN"):
                raise _UncacheableOutline(outline)
            if not exact or not path.endswith(".py"):
                return outline
            # Declaration, helper and context consumers need only these rows.
            return {
                **{key: outline[key] for key in ("language", "parser", "capability", "completeness")},
                "types": tuple({key: row.get(key) for key in ("name", "kind", "line", "end_line")}
                               for row in outline.get("types", ())),
                "functions": tuple({key: row.get(key) for key in ("name", "parent", "line", "end_line")}
                                   for row in outline.get("functions", ())),
            }

        # An empty proof requires a live parse.
        if exact == () or (exact and not path.endswith(".py")):
            outline = build(source)
        elif exact is None and not compact:
            outline = _parse_cache.derived_source(
                ("owner_inventory.definitions", policy), source, path,
                lambda text: outline_source(text, path, definitions_only=True),
            )
        else:
            key = ("owner_inventory.definitions", policy, compact)
            outline = _parse_cache.derived_source(key + exact, source, path, build)
        report_outline_parser_capability(outline)
        return outline
    except _UncacheableOutline as failure:
        report_outline_parser_capability(failure.outline)
        return failure.outline
    except (ImportError, OSError, TypeError, ValueError):
        return {}


def _capability(outline: Mapping[str, object]) -> str:
    if not outline:
        return "UNAVAILABLE"
    return "DEGRADED" if outline.get("degraded") else "AVAILABLE"


def _source_revision_vector(
        sources: Mapping[str, str], parent_sources: Mapping[str, str],
) -> tuple[str, dict[str, dict[str, object]]]:
    from kernel_tree_digest import compatible_text_revision
    identities: dict[str, dict[str, object]] = {}
    for path in sorted(sources):
        source = sources[path]
        current = "sha256:" + hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()
        parent = parent_sources.get(path)
        parent_revision = ("sha256:" + hashlib.sha256(parent.encode("utf-8", "surrogatepass")).hexdigest()
                           if isinstance(parent, str) else None)
        identities[path] = {
            "current_revision": current,
            "parent_revision": parent_revision,
            "status": ("UNKNOWN" if parent is None else
                       "UNCHANGED" if source == parent else "CHANGED"),
        }
    return compatible_text_revision("stateframe.owner_sources/2", dict(sources),
                                    terminator=False), identities


def _declared_name(line: str, kind: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "//", "/*", "*", "@", "def ", "return ")):
        return None
    for pattern in (_GENERIC_DECL, _TYPED_DECL, *((_GO_FIELD,) if kind == "struct" else ())):
        match = pattern.match(line)
        if match and match.group("name") not in {"self", "this", "return", "if", "for", "else"}:
            return match.group("name")
    return None


def _correlation_key(name: str) -> str:
    """Bound a persisted-field group to its stable leading declaration stem."""
    parts = [part for part in name.rstrip("*").split("_") if part]
    return "_".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else name)


_DefinitionSpans = tuple[tuple[str, int, int], ...]


def _python_definition_spans(tree: ast.Module) -> tuple[_DefinitionSpans, _DefinitionSpans]:
    classes: list[tuple[str, int, int]] = []
    functions: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append((node.name, node.lineno, node.end_lineno or node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node.name, node.lineno, node.end_lineno or node.lineno))
    return tuple(classes), tuple(functions)


def _python_spans(source: str) -> tuple[_DefinitionSpans, _DefinitionSpans]:
    return _parse_cache.derived("owner_inventory.python_spans", source, "<unknown>",
                                _python_definition_spans)


def declared_fields(path: str, source: str, candidates: Sequence[str], *,
                    outline: Mapping[str, object] | None = None) -> list[OwnerField]:
    """Resolve a named field plus its bounded correlated declaration group."""
    if outline is None:
        outline = _outline(path, source)
    lines = source.splitlines()
    spans = None
    if path.casefold().endswith(".py"):
        try:
            spans = _python_spans(source)
        except (SyntaxError, ValueError):
            pass
    if spans is not None:
        classes, functions = spans
        methods = [(start, end) for _, start, end in functions]
        type_rows = [
            {"line": start, "end_line": end, "name": name, "kind": "class"}
            for name, start, end in classes
        ]
    else:
        methods = [(int(row.get("line", 0) or 0), int(row.get("end_line", 0) or 0))
                   for row in outline.get("functions", ()) or () if isinstance(row, Mapping)]
        type_rows = [row for row in outline.get("types", ()) or () if isinstance(row, Mapping)]
        if not type_rows:
            for number, line in enumerate(lines, 1):
                match = _TYPE_LINE.match(line)
                if match:
                    name = next((value for value in match.groups() if value), "")
                    kind = ("struct" if match.group("struct") else
                            "interface" if match.group("interface") else "class")
                    type_rows.append({"line": number, "end_line": number, "name": name, "kind": kind})
    expanded_types: list[Mapping[str, object]] = []
    source_lines = lines
    for row in type_rows:
        start = int(row.get("line", 0) or 0)
        end = int(row.get("end_line", start) or start)
        if start > 0 and end <= start and start <= len(source_lines):
            depth = source_lines[start - 1].count("{") - source_lines[start - 1].count("}")
            cursor = start
            while depth > 0 and cursor < len(source_lines):
                cursor += 1
                depth += source_lines[cursor - 1].count("{") - source_lines[cursor - 1].count("}")
            end = cursor
        expanded_types.append({**dict(row), "line": start, "end_line": end})
    type_rows = expanded_types
    candidate_keys = {_correlation_key(str(item)) for item in candidates if str(item)}
    found: list[OwnerField] = []
    for row in type_rows:
        start, end = int(row.get("line", 0) or 0), int(row.get("end_line", 0) or 0)
        declared: list[OwnerField] = []
        for number in range(start + 1, min(end, len(lines)) + 1):
            if any(m_start <= number <= m_end for m_start, m_end in methods):
                continue
            name = _declared_name(lines[number - 1], str(row.get("kind") or ""))
            if name:
                declared.append(OwnerField(name, path, number, lines[number - 1].strip(),
                                           str(row.get("name") or "")))
        matched = [item for item in declared if _matches(item.name, candidates)]
        if matched:
            matched_keys = {_correlation_key(item.name) for item in matched}
            found.extend(item for item in declared
                         if _correlation_key(item.name) in candidate_keys | matched_keys)
    return found


def _site_patterns(name: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    escaped = re.escape(name)
    write = re.compile(
        rf"\.{escaped}\s*(?::\s*[^=\n]+)?\s*(?:=(?!=)|\+=|-=)|"
        rf"(?<![\w.$]){escaped}\s*(?::\s*[^=\n]+)?\s*=(?!=)"
        rf"|setattr\([^,]+,\s*['\"]{escaped}['\"]|\[['\"]{escaped}['\"]\]\s*=(?!=)")
    read = re.compile(rf"(?:\.|\$|->|\[['\"]){escaped}(?![\w])|getattr\([^,]+,\s*['\"]{escaped}['\"]")
    return read, write


def _field_contexts(path: str, source: str, sites: Sequence[Site], *,
                    outline: Mapping[str, object] | None = None) -> list[Site]:
    """Attach bounded enclosing effect/commit/catch context to field sites."""
    if outline is None:
        outline = _outline(path, source)
    lines = source.splitlines()
    contexts: list[Site] = []
    functions = [row for row in outline.get("functions", ()) or ()
                 if isinstance(row, Mapping)]
    if path.casefold().endswith(".py"):
        try:
            _, spans = _python_spans(source)
            functions = [
                {"line": start, "end_line": end, "name": name}
                for name, start, end in spans
            ]
        except (SyntaxError, ValueError):
            pass
    for site in sites:
        line = site.line
        containing = [row for row in functions
                      if int(row.get("line", 0) or 0) <= line <= int(row.get("end_line", 0) or 0)]
        if containing:
            row = min(containing, key=lambda item: int(item.get("end_line", 0) or 0)
                      - int(item.get("line", 0) or 0))
            start = int(row.get("line", 0) or 0)
            text = lines[start - 1].strip() if 0 < start <= len(lines) else str(row.get("name", ""))
            contexts.append(Site(path, start, text))
            for number in range(max(start, line - 4), min(len(lines), line + 5) + 1):
                text = lines[number - 1].strip()
                if re.search(r"\b(?:commit|flush|rollback|except|catch|defer|return|await)\b",
                             text, re.IGNORECASE):
                    contexts.append(Site(path, number, text))
    unique: list[Site] = []
    seen: set[tuple[str, int, str]] = set()
    for site in contexts:
        key = (site.path, site.line, site.text)
        if key not in seen:
            seen.add(key)
            unique.append(site)
    return unique[:_MAX_CONTEXT_SITES]


def _field_sites(path: str, lines: Sequence[str], name: str) -> tuple[list[Site], list[Site]]:
    read, write = _site_patterns(name)
    readers: list[Site] = []
    writers: list[Site] = []
    for number, line in enumerate(lines, 1):
        text = line.strip()
        if text.startswith(("#", "//")):
            continue
        if write.search(line):
            writers.append(Site(path, number, text))
        elif read.search(line):
            readers.append(Site(path, number, text))
    return readers, writers


@dataclass
class _FileFacts:
    path: str
    source: str
    outline: Mapping[str, object]
    blob: bytes | None = None
    language: str = ""
    policy: str = ""
    _candidates: tuple[str, ...] = ()
    _declarations: list[OwnerField] = dc_field(default_factory=list)
    _sites: dict[str, tuple[list[Site], list[Site]]] = dc_field(default_factory=dict)
    _contexts: dict[tuple[Site, ...], list[Site]] = dc_field(default_factory=dict)
    _lines: list[str] | None = None
    _content: bytes | None = None
    _revision: str | None = None

    @property
    def lines(self) -> list[str]:
        if self._lines is None:
            self._lines = self.source.splitlines()
        return self._lines

    @property
    def content(self) -> bytes:
        if self._content is None:
            self._content = self.source.encode("utf-8")
        return self._content

    @property
    def revision(self) -> str:
        if self._revision is None:
            self._revision = "sha256:" + hashlib.sha256(self.content).hexdigest()
        return self._revision

    def _load_declarations(self, candidates: tuple[str, ...], needles: Sequence[str], *,
                           matches: bool | None = None) -> None:
        if candidates == self._candidates:
            return
        if matches is None:
            matches = _has_candidate(self.source, needles)
        self._declarations = (declared_fields(self.path, self.source, candidates,
                                             outline=self.outline) if matches else [])
        self._candidates = candidates

    def declarations(self, candidates: tuple[str, ...],
                     needles: Sequence[str]) -> list[OwnerField]:
        self._load_declarations(candidates, needles)
        return [OwnerField(row.name, row.path, row.line, row.text, row.owner)
                for row in self._declarations]

    def sites(self, name: str) -> tuple[list[Site], list[Site]]:
        if name not in self._sites:
            if len(self._sites) >= 2 * _MAX_FIELDS:
                self._sites.clear()
            self._sites[name] = (_field_sites(self.path, self.lines, name)
                                 if name in self.source else ([], []))
        return self._sites[name]

    def contexts(self, sites: Sequence[Site]) -> list[Site]:
        key = tuple(sites)
        if key not in self._contexts:
            if len(self._contexts) >= 2 * _MAX_FIELDS:
                self._contexts.clear()
            self._contexts[key] = _field_contexts(self.path, self.source, sites,
                                                   outline=self.outline)
        return self._contexts[key]


def _has_candidate(source: str, needles: Sequence[str]) -> bool:
    folded = source.casefold()
    return any(needle in folded for needle in needles)


def _file_facts(path: str, source: str, candidates: tuple[str, ...], needles: Sequence[str],
                policy: str, *, blob: bytes | None = None, language: str = "",
                exact: tuple[object, ...] | None = None, revision: str | None = None) -> _FileFacts:
    matches = _has_candidate(source, needles)
    fact = _FileFacts(path, source, _outline(path, source, policy=policy, lexical_miss=not matches,
                                            exact=exact),
                      blob=blob, language=language, policy=policy, _revision=revision)
    # Collect declaration spans while the outline's AST is still resident.
    fact._load_declarations(candidates, needles, matches=matches)
    return fact


def _helpers(files: Sequence[str], sources: Mapping[str, str],
             fields: Sequence[OwnerField], candidates: Sequence[str],
             facts: Mapping[str, _FileFacts] | None) -> list[Helper]:
    stems = {word for item in [*(f.name for f in fields), *candidates]
             for word in item.lower().split("_") if len(word) >= 4}
    explicit = {str(item).rsplit(".", 1)[-1].casefold() for item in candidates}
    explicit.update(item.name.casefold() for item in fields)
    scored: list[tuple[tuple[int, int, str, int], Helper]] = []
    for path in files:
        source = sources.get(path)
        if source is None:
            continue
        fact = facts[path] if facts is not None else None
        outline = fact.outline if fact is not None else _outline(path, source)
        for row in outline.get("functions", ()) or ():
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "")
            qualified = ".".join(str(part) for part in (row.get("parent"), name) if part)
            words = set(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().split("_"))
            shared = len(words & stems)
            exact = int(name.casefold() in explicit or qualified.casefold() in explicit)
            prefix = int(any(name.casefold().startswith(item + "_") for item in explicit))
            # Hashing helpers can define a single persisted checksum stem.
            # Other fuzzy matches still require two shared tokens.
            hash_match = bool(words & stems & _HASH_HELPER_WORDS)
            if exact or prefix or shared >= 2 or hash_match:
                lines = fact.lines if fact is not None else source.splitlines()
                line = int(row.get("line", 0) or 0)
                text = lines[line - 1].strip() if 0 < line <= len(lines) else name
                score = (exact * 100 + prefix * 25 + shared,
                         exact, path, line)
                scored.append((score, Helper(name, path, line, text)))
    scored.sort(key=lambda item: (-item[0][0], -item[0][1], item[0][2], item[0][3]))
    helpers = [helper for _, helper in scored[:_MAX_HELPERS]]
    for helper in helpers:
        call = re.compile(rf"(?<![\w.]){re.escape(helper.name)}\s*\(|\.{re.escape(helper.name)}\s*\(")
        imported = re.compile(rf"\bimport\b.*\b{re.escape(helper.name)}\b")
        for path in files:
            source = sources.get(path)
            if source is None or helper.name not in source:
                continue
            lines = facts[path].lines if facts is not None else source.splitlines()
            for number, line in enumerate(lines, 1):
                if path == helper.path and number == helper.line:
                    continue
                if _DEF_LINE.match(line):
                    continue
                if call.search(line):
                    helper.callers.append(Site(path, number, line.strip()))
                elif imported.search(line):
                    helper.importers.append(Site(path, number, line.strip()))
    return helpers


def build_inventory(root: Path, candidates: Sequence[str], *,
                    files: Sequence[str] | None = None,
                    parent_sources: Mapping[str, str] | None = None,
                    scan: ProductFileScan | None = None,
                    _facts: Mapping[str, _FileFacts] | None = None,
                    _parent_digests: Mapping[str, str] | None = None,
                    _include_contexts: bool = True) -> Inventory | None:
    """Host inventory with explicit scan completeness and parent identities."""
    base = Path(root).resolve()
    candidates = tuple(candidates)
    if not candidates:
        return None
    file_scan = (scan if scan is not None else
                 ProductFileScan(tuple(str(item) for item in files), (), True)
                 if files is not None else scan_product_files(base))
    files = list(file_scan.paths)
    skipped = list(file_scan.skipped_files)
    sources: dict[str, str] = {}
    parser_capabilities: dict[str, str] = {}
    fields: list[OwnerField] = []
    needles = tuple(cand.casefold() for cand in candidates)
    for path in files:
        fact = _facts.get(path) if _facts is not None else None
        source = (_read(base, path) if _facts is None else
                  fact.source if fact is not None else None)
        if source is None:
            parser_capabilities[path] = "UNAVAILABLE"
            skipped.append({"path": path, "status": "UNKNOWN",
                            "reason": "file is unreadable or exceeds bounded size"})
            continue
        sources[path] = source
        outline = fact.outline if fact is not None else _outline(path, source)
        parser_capabilities[path] = _capability(outline)
        if fact is not None:
            fields.extend(fact.declarations(candidates, needles))
        else:
            folded = source.casefold()
            if any(needle in folded for needle in needles):
                fields.extend(declared_fields(path, source, candidates, outline=outline))
    complete = (file_scan.complete and not skipped
                and all(value == "AVAILABLE" for value in parser_capabilities.values()))
    if not fields and complete:
        return None
    fields = fields[:_MAX_FIELDS]
    declarations: dict[str, set[int]] = {}
    for item in fields:
        declarations.setdefault(item.path, set()).add(item.line)
    for item in fields:
        for path, source in sources.items():
            fact = _facts[path] if _facts is not None else None
            if fact is None and item.name not in source:
                continue
            readers, writers = (fact.sites(item.name) if fact is not None else
                                _field_sites(path, source.splitlines(), item.name))
            excluded = declarations.get(path, ())
            readers = [site for site in readers if site.line not in excluded]
            writers = [site for site in writers if site.line not in excluded]
            item.readers.extend(readers)
            item.writers.extend(writers)
            if _include_contexts and (readers or writers):
                sites = readers + writers
                item.contexts.extend(fact.contexts(sites) if fact is not None
                                     else _field_contexts(path, source, sites))
    parent_sources = parent_sources or {}
    if _facts is None or _parent_digests is None:
        dependency_revision, identities = _source_revision_vector(sources, parent_sources)
    else:
        dependency_revision, identities = _pinned_revision(_facts, parent_sources, _parent_digests)
    helpers = _helpers(files, sources, fields, candidates, _facts)
    return Inventory(candidates, fields, helpers,
                     dependency_revision[:12], len(sources), complete=complete,
                     skipped_files=tuple(skipped), dependency_revision=dependency_revision,
                     identities=identities, parser_capabilities=parser_capabilities)


# ------------------------------------------------------------------ rendering

def _clip(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def render(inventory: Inventory) -> str:
    lines = [f"{HEADER} | host grep over the parent at revision {inventory.revision}: fields "
              f"matching {', '.join(inventory.candidates[:4])}; {inventory.files_scanned} product "
              f"files scanned; completeness={inventory.writer_status}; "
              f"parser={','.join(sorted(set(inventory.parser_capabilities.values())) or {'UNKNOWN'})}]"]
    for item in inventory.fields:
        lines.append(f"field    {item.path}:{item.line}  `{_clip(item.text)}`"
                     + (f"  (in `{item.owner}`)" if item.owner else ""))
    for item in inventory.fields:
        if item.readers:
            shown = "; ".join(f"{site.path}:{site.line} `{_clip(site.text, 60)}`"
                              for site in item.readers[:_MAX_SITES])
            more = f" (+{len(item.readers) - _MAX_SITES} more)" if len(item.readers) > _MAX_SITES else ""
            lines.append(f"readers  {item.name}: {len(item.readers)} site(s): {shown}{more}")
        else:
            lines.append(f"readers  {item.name}: 0 sites in product code")
    if inventory.writer_count == 0:
        if inventory.complete:
            lines.append("writers  0 sites in product code set any listed field (only the declaration "
                         "and tests mention them)")
        else:
            lines.append("writers  UNKNOWN: 0 observed in the bounded scan; skipped files prevent "
                         "a global absence claim")
    else:
        for item in inventory.fields:
            for site in item.writers[:_MAX_SITES]:
                lines.append(f"writer   {item.name}: {site.path}:{site.line} `{_clip(site.text, 70)}`")
    for helper in inventory.helpers:
        lines.append(f"helper   {helper.path}:{helper.line}  `{_clip(helper.text)}`")
        callers = (f"callers  {helper.name}: {len(helper.callers)} call site(s) in product code")
        if helper.callers:
            callers += ": " + "; ".join(f"{site.path}:{site.line}" for site in helper.callers[:_MAX_SITES])
        elif helper.importers:
            callers += " (" + "; ".join(f"{site.path}:{site.line} imports it"
                                        for site in helper.importers[:3]) + ", never called)"
        lines.append(callers)
    if inventory.skipped_files:
        for item in inventory.skipped_files[:_MAX_SITES]:
            lines.append(f"skipped  {item.get('path')} status={item.get('status')} "
                         f"reason={item.get('reason')}")
    lines.append("All counts are bounded host grep/outline facts. Zero writers is global absence "
                 "only when completeness=KNOWN; otherwise the result is UNKNOWN.")
    text = "\n".join(lines)
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS - 4] + "\n..."


# ------------------------------------------------------------------ runtime seam

def _inventory_store(runtime: object):
    if getattr(runtime, "_template_conformance_resumed", False):
        return None
    from kernel_measurement import startup_pilot_enabled

    if not startup_pilot_enabled():
        return None
    try:
        import stateframe.send1_probe_cache as send1_probe_cache
    except ModuleNotFoundError as error:
        if error.name != "stateframe.send1_probe_cache":
            raise
        return None
    from kernel_tool_tape import replay_mode_active
    return None if replay_mode_active() or not send1_probe_cache.enabled() else send1_probe_cache


def _exact_python_policy(policy: str) -> tuple[tuple[object, ...], str | None]:
    """Identify the stdlib parser or the audited, installed Python grammar."""
    from repo_intel_core import ts_dispatch

    parser = ts_dispatch._parse_source
    if (sys.implementation.name != "cpython"
            or getattr(getattr(parser, "__code__", None), "co_filename", None) != ts_dispatch.__file__):
        return (), None

    parser_files = ()
    pack = sys.modules.get("tree_sitter_language_pack")
    audited_shim = getattr(pack, "__stateframe_grammar_adapter__", False)
    if audited_shim:
        from stateframe import ts_grammars

        getter = getattr(pack, "get_parser", None)
        # The direct child entry point can load the same adapter under its
        # top-level name. Check the implementation file and the live provider.
        adapter = sys.modules.get(getattr(getter, "__module__", ""))
        if (getattr(adapter, "__file__", None) != ts_grammars.__file__
                or getter is not adapter.get_parser
                or getattr(pack, "get_language", None) is not adapter.get_language
                or getattr(getattr(getter, "__code__", None), "co_filename", None) != adapter.__file__
                or adapter.GRAMMARS.get("python") != ("tree_sitter_python", "language")):
            return (), None
        parser_files = (adapter.__file__,)

    if policy.startswith("python:ast:AVAILABLE:stdlib:extension:"):
        # The installed shim can only load the named local grammar. A broken
        # grammar present on disk does not establish stable AST fallback.
        optional = "tree_sitter_python" if audited_shim else "tree_sitter_language_pack"
        try:
            if find_spec(optional) is not None:
                return (), None
        except (ImportError, ValueError):
            return (), None
        implementation = "ast"
    elif audited_shim and policy.startswith("python:tree-sitter:AVAILABLE:"):
        library = sys.modules.get("tree_sitter._binding")
        grammar = sys.modules.get("tree_sitter_python._binding")
        if (not all(isinstance(getattr(module, "__loader__", None), ExtensionFileLoader)
                    for module in (library, grammar))
                or sys.modules["tree_sitter"].Parser is not library.Parser
                or sys.modules["tree_sitter_python"].language is not grammar.language):
            return (), None
        implementation = "tree-sitter"
        parser_files += (library.__file__, grammar.__file__)
    else:
        return (), None

    # Both outline walkers can fail near the recursion limit.
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    code = _inventory_implementation((__file__, ts_dispatch.__file__, _parse_cache.__file__, *parser_files))
    return (implementation, _INVENTORY_CACHE_VERSION, sys.version, sys.getrecursionlimit(),
            depth, policy.rsplit(":", 1)[0]), code


@lru_cache(maxsize=8)
def _inventory_implementation(files: tuple[str, ...]) -> str | None:
    digest = hashlib.sha256()
    try:
        for path in files:
            with open(path, "rb") as source:
                digest.update(hashlib.file_digest(source, "sha256").digest())
    except OSError:
        return None
    return digest.hexdigest()


def _startup_source_scan(root: Path) -> tuple[ProductFileScan, StartupObservation] | None:
    try:
        from stateframe.workspace_start_snapshot import startup_observation
    except ModuleNotFoundError as error:
        if error.name != "stateframe.workspace_start_snapshot":
            raise
        return None
    from kernel_request_source_window import _FORBIDDEN_PARTS, _TEST_PATH

    observed = startup_observation(root, "state-owner-inventory")
    if observed is None:
        return None
    by_dir: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for path in observed.files:
        *parent, name = path.split("/")
        if (Path(name).suffix in SOURCE_SUFFIXES and not _TEST_PATH.search(path)
                and not any(part.startswith(".") or part in _SKIP_DIRS
                            or part.casefold() in _FORBIDDEN_PARTS for part in parent)):
            by_dir[tuple(parent)].append(path)
    paths = [path for parent in sorted(by_dir) for path in sorted(by_dir[parent])]
    if len(paths) <= _MAX_FILES:
        return ProductFileScan(tuple(paths), (), True), observed
    skipped = {"path": paths[_MAX_FILES], "status": "PARTIAL",
               "reason": "product file scan limit reached"}
    return ProductFileScan(tuple(paths[:_MAX_FILES]), (skipped,), False), observed


def _memo_key(inputs: object) -> str:
    return hashlib.sha256(json.dumps(inputs, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _decode_inventory(payload: object, scan: ProductFileScan, candidates: tuple[str, ...],
                      parent_digests: Mapping[str, str], revisions: Mapping[str, str | None], *,
                      include_contexts: bool) -> Inventory | None:
    """Decode detached facts from the private store, checking the current scan binding."""
    if not isinstance(payload, str) or len(payload) > _MAX_MEMO_CHARS:
        raise ValueError("invalid inventory cache payload")
    row = json.loads(payload)
    if row is None and scan.complete and len(revisions) == len(scan.paths):
        return None
    if not isinstance(row, dict) or set(row) != {"dependency_revision", "identities", "fields", "helpers"}:
        raise ValueError("invalid inventory cache record")

    def revision(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def site(value: object, extra: set[str]) -> Site:
        if (not isinstance(value, dict) or set(value) != {"path", "line", "text"} | extra
                or not isinstance(value["path"], str) or value["path"] not in identities
                or type(value["line"]) is not int or not 0 < value["line"] <= _MAX_FILE_BYTES
                or not isinstance(value["text"], str)):
            raise ValueError("invalid inventory site")
        return Site(value["path"], value["line"], value["text"])

    def sites(values: object) -> list[Site]:
        if not isinstance(values, list):
            raise ValueError("invalid inventory sites")
        return [site(value, set()) for value in values]

    digest = row["dependency_revision"]
    identities = row["identities"]
    if not revision(digest) or not isinstance(identities, dict) or identities.keys() != revisions.keys():
        raise ValueError("invalid inventory source identities")
    for path, identity in identities.items():
        parent = parent_digests.get(path)
        if (not isinstance(identity, dict)
                or set(identity) != {"current_revision", "parent_revision", "status"}):
            raise ValueError("invalid inventory source identity")
        current = identity["current_revision"]
        bound_revision = revisions[path]
        if (not isinstance(current, str) or not current.startswith("sha256:") or not revision(current[7:])
                or (bound_revision is not None and current != bound_revision)
                or identity["parent_revision"] != parent
                or identity["status"] != ("UNKNOWN" if parent is None else
                                          "UNCHANGED" if current == parent else "CHANGED")):
            raise ValueError("inventory source identity mismatch")
    skipped = (*scan.skipped_files, *(
        {"path": path, "status": "UNKNOWN", "reason": "file is unreadable or exceeds bounded size"}
        for path in scan.paths if path not in identities
    ))
    complete = scan.complete and not skipped
    fields = row["fields"]
    helpers = row["helpers"]
    if (not isinstance(fields, list) or len(fields) > _MAX_FIELDS
            or not isinstance(helpers, list) or len(helpers) > _MAX_HELPERS
            or (not fields and complete)):
        raise ValueError("invalid inventory rows")

    decoded_fields = []
    for value in fields:
        location = site(value, {"name", "owner", "readers", "writers", "contexts"})
        if (not isinstance(value["name"], str) or not isinstance(value["owner"], str)
                or (not include_contexts and value["contexts"] != [])):
            raise ValueError("invalid inventory declaration")
        decoded_fields.append(OwnerField(value["name"], location.path, location.line, location.text,
                                        value["owner"], sites(value["readers"]), sites(value["writers"]),
                                        sites(value["contexts"])))
    decoded_helpers = []
    for value in helpers:
        location = site(value, {"name", "callers", "importers"})
        if not isinstance(value["name"], str):
            raise ValueError("invalid inventory helper")
        decoded_helpers.append(Helper(value["name"], location.path, location.line, location.text,
                                      sites(value["callers"]), sites(value["importers"])))
    return Inventory(candidates, decoded_fields, decoded_helpers, digest[:12], len(identities),
                     complete=complete, skipped_files=skipped, dependency_revision=digest,
                     identities=identities, parser_capabilities={
                         path: "AVAILABLE" if path in identities else "UNAVAILABLE" for path in scan.paths
                     })


def _remember_inventory(fingerprint: str, payload: str) -> None:
    with _MEMO_LOCK:
        _MEMO[fingerprint] = payload
        _MEMO.move_to_end(fingerprint)
        if len(_MEMO) > _MAX_MEMO_ENTRIES:
            _MEMO.popitem(last=False)


def _lookup_inventory(store, fingerprint: str, persistent: bool, scan: ProductFileScan,
                      candidates: tuple[str, ...], parent_digests: Mapping[str, str],
                      revisions: Mapping[str, str | None], *,
                      include_contexts: bool) -> tuple[bool, Inventory | None]:
    with _MEMO_LOCK:
        payload = _MEMO.get(fingerprint)
        if payload is not None:
            _MEMO.move_to_end(fingerprint)
    if payload is None and persistent:
        try:
            facts = store.lookup(_INVENTORY_CACHE_KIND, fingerprint)
            if isinstance(facts, dict) and set(facts) == {"inventory"}:
                payload = facts["inventory"]
        except OSError:
            pass
    try:
        inventory = _decode_inventory(payload, scan, candidates, parent_digests, revisions,
                                      include_contexts=include_contexts)
    except (TypeError, ValueError, RecursionError):
        return False, None
    _remember_inventory(fingerprint, payload)
    return True, inventory


def _save_inventory(store, fingerprint: str, persistent: bool, inventory: Inventory | None) -> None:
    from itertools import chain

    facts = None
    if inventory is not None:
        # Avoid materializing an unbounded serialization of occurrences. The
        # rendered panel is small but callers can inspect the full site lists.
        sites = chain(
            ((field, field.readers, field.writers, field.contexts) for field in inventory.fields),
            ((helper, helper.callers, helper.importers) for helper in inventory.helpers),
        )
        remaining = _MAX_MEMO_CHARS
        for row, *groups in sites:
            remaining -= len(row.text)
            for group in groups:
                if len(group) > remaining // 32:
                    return
                remaining -= sum(32 + len(site.path) + len(site.text) for site in group)
            if remaining < 0:
                return
        facts = {"dependency_revision": inventory.dependency_revision, "identities": inventory.identities,
                 "fields": inventory.fields, "helpers": inventory.helpers}
    payload = json.dumps(facts, default=vars, ensure_ascii=True, separators=(",", ":"))
    if len(payload) > _MAX_MEMO_CHARS:
        return
    _remember_inventory(fingerprint, payload)
    if persistent:
        store.publish(_INVENTORY_CACHE_KIND, fingerprint, {"inventory": payload})


def _admitted_files(root: Path, scan: ProductFileScan, startup: StartupObservation) -> dict[str, File] | None:
    """Bind the size cutoff to current metadata before trusting startup hashes."""
    from kernel_workspace_tree import File

    files = {}
    try:
        for path in scan.paths:
            info = (root / path).stat(follow_symlinks=False)
            if info.st_mode != startup.modes[path] or not stat.S_ISREG(info.st_mode):
                return None
            files[path] = File(info.st_mode, info.st_size, bytes.fromhex(startup.files[path]))
    except OSError:
        return None
    return files


def _exact_inventory(root: Path, candidates: tuple[str, ...], parent: Mapping[str, str], *,
                     include_contexts: bool, entry: tuple[object, ...], store,
                     use_startup: bool = True) -> tuple[Inventory | None, tuple[object, ...]]:
    from repo_intel_core.ts_dispatch import (
        detect_language, outline_capability_key, report_outline_parser_capability,
    )

    python_policy = outline_capability_key("source.py")
    exact, implementation = _exact_python_policy(python_policy)
    if exact:
        exact += (implementation or python_policy,)
    parent_digests = {path: "sha256:" + _parse_cache.source_digest(source)
                      for path, source in sorted(parent.items())}
    admission = _startup_source_scan(root) if exact and use_startup else None
    scan, startup = admission if admission is not None else (scan_product_files(root), None)
    reuse = bool(exact) and all(path.endswith(".py") for path in scan.paths)
    files = _admitted_files(root, scan, startup) if reuse and startup is not None else None
    if startup is not None and files is None:
        scan = scan_product_files(root)
        reuse = bool(exact) and all(path.endswith(".py") for path in scan.paths)

    # With an audited view, the raw digest binds the normalized text as well.
    # Without that view, read the full bounded scan before considering a hit.
    sources = {}
    revisions = {}
    unknown_read = False
    if files is None:
        for path in scan.paths:
            source = _read(root, path)
            if source is not None:
                sources[path] = source
                revisions[path] = "sha256:" + _parse_cache.source_digest(source)
            elif reuse:
                try:
                    unknown_read |= (root / path).stat().st_size <= _MAX_FILE_BYTES
                except OSError:
                    unknown_read = True
    else:
        revisions = dict.fromkeys(path for path, file in files.items() if file.size <= _MAX_FILE_BYTES)

    def inventory_key():
        content = (("read", [(path, revisions.get(path)) for path in scan.paths]) if files is None else
                   ("admitted", startup.exclusion_digest, startup.overlay_digest,
                    [(path, file.digest.hex()) for path, file in files.items()]))
        return _memo_key((exact, entry, candidates, include_contexts, _MAX_FILES, _MAX_FILE_BYTES,
                          _MAX_FIELDS, _MAX_HELPERS, _MAX_CONTEXT_SITES, scan.complete,
                          scan.skipped_files, content, tuple(parent_digests.items())))

    key = None
    cacheable = reuse and not unknown_read
    if cacheable and getattr(store, "reads_available", lambda: True)():
        key = inventory_key()
        hit, inventory = _lookup_inventory(store, key, implementation is not None, scan, candidates,
                                          parent_digests, revisions,
                                          include_contexts=include_contexts)
        if hit:
            # The lookup can outlast the admitted workspace generation.
            if files is not None and not startup.validate_for_use("state-owner-inventory"):
                return _exact_inventory(root, candidates, parent, include_contexts=include_contexts,
                                        entry=entry, store=store, use_startup=False)
            if (inventory is None and scan.paths) or (inventory is not None and inventory.files_scanned):
                report_outline_parser_capability({"language": "python", "parser": exact[0],
                                                  "capability": "AVAILABLE", "completeness": "KNOWN"})
            return inventory, ("exact", key)

    if files is not None:
        for path in revisions:
            source = _read(root, path, observed=files[path])
            if source is None:
                # Reopen through the normal scanner if the admitted read could
                # not establish the expected bytes and failure identity.
                return _exact_inventory(root, candidates, parent, include_contexts=include_contexts,
                                        entry=entry, store=store, use_startup=False)
            sources[path] = source
            revisions[path] = "sha256:" + _parse_cache.source_digest(source)

    policy = {"python": python_policy} if all(path.endswith(".py") for path in sources) else _parser_policy()
    needles = tuple(candidate.casefold() for candidate in candidates)
    facts = {}
    for path, source in sources.items():
        effective = policy[detect_language(path, source)]
        facts[path] = _file_facts(path, source, candidates, needles, effective, exact=exact,
                                 revision=revisions[path])
    inventory = build_inventory(root, candidates, scan=scan, parent_sources=parent,
                                _facts=facts, _parent_digests=parent_digests,
                                _include_contexts=include_contexts)
    # Matching reads bind individual files. Recheck membership before using the assembled inventory.
    if files is not None and not startup.validate_for_use("state-owner-inventory"):
        return _exact_inventory(root, candidates, parent, include_contexts=include_contexts,
                                entry=entry, store=store, use_startup=False)
    if cacheable and all(
            fact.outline.get("parser") == exact[0] and fact.outline.get("completeness") == "KNOWN"
            and _capability(fact.outline) == "AVAILABLE" for fact in facts.values()):
        from stateframe.startup_cache_writes import defer

        try:
            defer(_INVENTORY_CACHE_KIND, lambda: _save_inventory(
                store, key if key is not None else inventory_key(), implementation is not None, inventory))
        except OSError:
            pass
    return inventory, ("exact", key)


def _legacy_inventory(runtime: object, root: Path, candidates: tuple[str, ...],
                      parent: Mapping[str, str], *,
                      include_contexts: bool, store=None,
                      entry: tuple[object, ...] = ()) -> tuple[Inventory | None, tuple[object, ...]]:
    from repo_intel_core.ts_dispatch import detect_language

    if store is not None:
        with _parse_cache.gc_paused():
            return _exact_inventory(root, candidates, parent, include_contexts=include_contexts,
                                    entry=entry, store=store)
    scan = scan_product_files(root)
    policy = _parser_policy()
    facts: dict[str, _FileFacts] = {}
    needles = tuple(candidate.casefold() for candidate in candidates)
    with _parse_cache.gc_paused():
        for path in scan.paths:
            source = _read(root, path)
            if source is None:
                continue
            effective = policy[detect_language(path, source)]
            fact = _FileFacts(path, source, _outline(path, source, policy=effective))
            # Collect declaration spans while the outline's AST is still resident.
            fact._load_declarations(candidates, needles)
            facts[path] = fact
        revisions = tuple((path, fact.revision, _capability(fact.outline))
                          for path, fact in facts.items())
        parent_digests = {
            path: "sha256:" + _parse_cache.source_digest(source)
            for path, source in sorted(parent.items())
        }
        key = (candidates, include_contexts, tuple(policy.items()), revisions,
               scan.paths, scan.skipped_files, scan.complete, tuple(parent_digests.items()))
        cached = getattr(runtime, "_state_owner_inventory", False)
        if cached is not False and getattr(runtime, "_state_owner_inventory_cache_key", None) == key:
            return cached, key
        inventory = build_inventory(root, candidates, scan=scan, parent_sources=parent,
                                    _facts=facts, _parent_digests=parent_digests,
                                    _include_contexts=include_contexts)
    return inventory, key


def _pinned_files(tree: Directory) -> tuple[ProductFileScan, dict[str, File]]:
    from kernel_request_source_window import _FORBIDDEN_PARTS, _TEST_PATH
    from kernel_workspace_tree import Directory, File, children

    paths: list[str] = []
    files: dict[str, File] = {}
    pending = [("", tree)]
    while pending:
        prefix, directory = pending.pop()
        subdirs: list[tuple[str, Directory]] = []
        for name, entry in sorted(((os.fsdecode(name), entry) for name, entry in children(directory)),
                                  key=lambda row: row[0]):
            path = prefix + name
            if isinstance(entry, Directory):
                if (not name.startswith(".") and name not in _SKIP_DIRS
                        and name.casefold() not in _FORBIDDEN_PARTS):
                    subdirs.append((path + "/", entry))
                continue
            if Path(name).suffix not in SOURCE_SUFFIXES or _TEST_PATH.search(path):
                continue
            if len(paths) >= _MAX_FILES:
                skipped = {"path": path, "status": "PARTIAL",
                           "reason": "product file scan limit reached"}
                return ProductFileScan(tuple(paths), (skipped,), False), files
            paths.append(path)
            # A link target or a special file is outside the pinned text scope.
            if isinstance(entry, File) and entry.size <= _MAX_FILE_BYTES:
                files[path] = entry
        pending.extend(reversed(subdirs))
    return ProductFileScan(tuple(paths), (), True), files


def _parser_policy() -> dict[str, str]:
    from repo_intel_core.ts_dispatch import detect_language, outline_capability_key

    # One capability probe per language. The extra suffix covers shebang detection
    # on extensions the dispatcher does not recognize, such as .kt.
    samples = {detect_language(path): path for path in
               (f"source{suffix}" for suffix in sorted(SOURCE_SUFFIXES))}
    policy = {language: outline_capability_key(path) for language, path in samples.items()}
    policy["vue"] = "|".join(policy[language] for language in
                             ("vue", "typescript", "tsx", "javascript"))
    return policy


def _pinned_revision(facts: Mapping[str, _FileFacts], parent: Mapping[str, str],
                     parent_digests: Mapping[str, str]) -> tuple[str, dict[str, dict[str, object]]]:
    from kernel_tree_digest import compatible_revision

    rows = []
    identities: dict[str, dict[str, object]] = {}
    for path in sorted(facts):
        fact = facts[path]
        parent_revision = parent_digests.get(path)
        identities[path] = {
            "current_revision": fact.revision,
            "parent_revision": parent_revision,
            "status": ("UNKNOWN" if parent_revision is None else
                       "UNCHANGED" if fact.source == parent[path] else "CHANGED"),
        }
        rows.append((path, fact.content, "file"))
    return compatible_revision("stateframe.owner_sources/2", rows), identities


@dataclass
class _InventoryProjection:
    tree: bytes
    scan: ProductFileScan
    files: dict[str, File]
    key: tuple[object, ...]
    parent: dict[str, str]
    parent_digests: dict[str, str]
    facts: dict[str, _FileFacts]
    inventory: Inventory | None


# Keep derived source payloads out of checkpoint state. The shared default is
# process-local, and the weak keys retire each projection with its fence.
_PROJECTIONS: ContextVar[WeakKeyDictionary[WorkspaceState, _InventoryProjection]] = ContextVar(
    "state_owner_projections", default=WeakKeyDictionary())


def _project_inventory(root: Path, tree: Directory, candidates: tuple[str, ...],
                       parent: dict[str, str],
                       previous: _InventoryProjection | None, *,
                       include_contexts: bool) -> _InventoryProjection:
    from repo_intel_core.ts_dispatch import detect_language

    policy = _parser_policy()
    key = (candidates, include_contexts, tuple(policy.items()))
    same_parent = previous is not None and parent == previous.parent
    if previous is not None and tree.digest == previous.tree:
        if key == previous.key and same_parent and len(previous.facts) == len(previous.files):
            return previous
        scan, files = previous.scan, previous.files
    else:
        scan, files = _pinned_files(tree)

    if same_parent and previous is not None:
        parent_digests = previous.parent_digests
    else:
        parent_digests = {
            path: (previous.parent_digests[path]
                   if previous is not None and previous.parent.get(path) == source
                   else "sha256:" + hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest())
            for path, source in parent.items()
        }

    facts: dict[str, _FileFacts] = {}
    needles = tuple(candidate.casefold() for candidate in candidates)
    for path, entry in files.items():
        old = previous.facts.get(path) if previous is not None else None
        if old is not None and old.blob == entry.sha256:
            if old.policy == policy[old.language]:
                facts[path] = old
                continue
            source, language = old.source, old.language
        else:
            source = _read(root, path, observed=entry)
            if source is None:
                continue
            language = detect_language(path, source)
        effective = policy[language]
        fact = _FileFacts(path, source, _outline(path, source, policy=effective),
                          blob=entry.sha256, language=language, policy=effective)
        fact._load_declarations(candidates, needles)
        facts[path] = fact
    inventory = build_inventory(root, candidates, scan=scan, parent_sources=parent,
                                _facts=facts, _parent_digests=parent_digests,
                                _include_contexts=include_contexts)
    return _InventoryProjection(tree.digest, scan, files, key, parent, parent_digests,
                                facts, inventory)


def _pinned_state(root: Path) -> WorkspaceState | None:
    if os.environ.get("STATEFRAME_WORKSPACE_OBJECTS") != "on":
        return None
    from kernel_workspace_consumers import consumers_for

    consumers = consumers_for(root)
    return consumers.state if consumers is not None else None


def _pinned_inventory(root: Path, state: WorkspaceState,
                      candidates: tuple[str, ...], parent: dict[str, str], *,
                      include_contexts: bool) -> Inventory | None:
    from kernel_workspace_state import WorkspaceUnavailable

    boundary = "state-owner-inventory"
    try:
        with state.pin(boundary) as snapshot:
            projections = _PROJECTIONS.get()
            with _parse_cache.gc_paused():
                projection = _project_inventory(root, snapshot.tree, candidates, parent,
                                                projections.get(state), include_contexts=include_contexts)
            if not snapshot.validate_for_use(boundary):
                raise WorkspaceUnavailable("state owner observation changed")
            projections[state] = projection
            return projection.inventory
    except WorkspaceUnavailable:
        if not candidates:
            return None
        return Inventory(candidates, [], [], "", 0, complete=False, skipped_files=(
            {"path": "", "status": "UNKNOWN", "reason": "workspace observation unavailable"},
        ))


def _entry_absent(runtime: object) -> bool:
    entry = getattr(runtime, "_site_facts_entry", None)
    if not isinstance(entry, Mapping):
        return False
    return (str(entry.get("status")) in {"NOT_FOUND", "REFERENCED_ONLY"}
            or bool(entry.get("new_symbol")))


def runtime_inventory(runtime: object) -> Inventory | None:
    """Cache the parent inventory only while its source identity is current."""

    supplied = getattr(runtime, "_state_owner_inventory", False)
    # Isolated host-contract fixtures and an already-admitted host receipt may
    # provide an immutable inventory while the optional filesystem scanner is
    # disabled.  Do not erase that parent fact merely because STATEFRAME's
    # live scan mode is off.
    if isinstance(supplied, Inventory) and (
            mode() != "on" or getattr(getattr(runtime, "owner", None), "root", None) is None):
        return supplied
    inventory = None
    root = getattr(getattr(runtime, "owner", None), "root", None)
    current_key: tuple[object, ...] = ("disabled",)
    if root is not None and mode() == "on":
        # The first request uses the declarations and sites, but neither its
        # owner panel nor its other consumers read enclosing context.
        include_contexts = getattr(runtime, "_frame_budget_request_count", None) != 0
        from kernel_request_source_window import _ticket_text, entry_symbol
        ticket = _ticket_text(runtime)
        absent = _entry_absent(runtime)
        symbol = entry_symbol(runtime)
        candidates = tuple(item for item in owner_candidates(ticket, pairs=absent)
                           if item != symbol and item != symbol.rsplit(".", 1)[-1])
        from kernel_request_source_window import _parent_sources
        parent_sources = _parent_sources(runtime)
        root = Path(root)
        state = _pinned_state(root)
        # Both source paths report parser capability even for an empty inventory.
        if state is not None:
            current_key = ("pinned",)
            inventory = _pinned_inventory(root, state, candidates, parent_sources,
                                          include_contexts=include_contexts)
        else:
            store = _inventory_store(runtime) if candidates else None
            entry = ()
            source_entry = getattr(runtime, "_site_facts_entry", None)
            if store is not None:
                entry = (symbol, absent)
                if isinstance(source_entry, Mapping):
                    entry += tuple(str(source_entry.get(key) or "") for key in
                                   ("symbol", "source_ref", "status", "revision", "dependency_revision"))
            inventory, current_key = _legacy_inventory(
                runtime, root, candidates, parent_sources, include_contexts=include_contexts,
                store=store, entry=entry)
    try:
        setattr(runtime, "_state_owner_inventory", inventory)
        setattr(runtime, "_state_owner_inventory_cache_key", current_key)
    except (AttributeError, TypeError):
        pass
    return inventory


def owner_writer_count(runtime: object) -> int | None:
    inventory = getattr(runtime, "_state_owner_inventory", None)
    if getattr(runtime, "_state_owner_inventory_cache_key", None) == ("pinned",):
        inventory = runtime_inventory(runtime)
    return (inventory.writer_count if isinstance(inventory, Inventory) and inventory.complete
            else None)


__all__ = ["ENV", "HEADER", "Helper", "Inventory", "OwnerField", "ProductFileScan", "Site",
           "build_inventory", "declared_fields", "mode", "owner_candidates", "owner_writer_count",
           "product_files", "render", "runtime_inventory", "scan_product_files"]
