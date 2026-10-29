"""Host-computed where-used closure for accepted product edits.

The organ is deliberately independent of model wording.  An accepted edit is
compared as source text, changed symbols and string literals are extracted by
the host, and a fresh repository code pack supplies definitions, references,
imports, and literal occurrences.  The model can close a site with evidence
or a host-accepted dispute; it cannot make an unvisited site disappear.
"""
from __future__ import annotations

import ast
import collections
import contextvars
import difflib
import hashlib
import io
import json
import os
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


MODE_ENV = "STATEFRAME_WHERE_USED_CLOSURE"
SCHEMA = "where-used-closure/1"
EVENT_KIND = "where_used_closure_fired"
SITE_UNREVIEWED = "SITE_UNREVIEWED"
# A reference is a CHECK_CANDIDATE, not a violation. Only the
# explicit enforce tier turns open sites into completion-blocking evidence gaps (ADMITTED).
TIER_ENV = "STATEFRAME_WHERE_USED_TIER"
TIER_CHECK_CANDIDATE = "CHECK_CANDIDATE"
TIER_ADMITTED = "ADMITTED"
DISPUTE_GROUNDS = frozenset({
    "WRONG_BINDING", "COMPATIBILITY_ADAPTER", "OUTSIDE_SUPPORTED_SCOPE",
    "STALE_OR_INVALID_WITNESS", "WRONG_CONTRACT",
})
RESOLUTION_MENU = (
    "A listed site is a place the change MAY reach, not a required edit. Compatibility can be kept by a consumer "
    "fix, an alias, an adapter, or a regenerated client; after reading a site you may mark it NOT_AFFECTED."
)
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".venv", "venv", "node_modules", "dist", "build",
})
_ACTIVE_JOURNAL: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "where_used_journal", default=None)
_ACTIVE_LEDGER: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "where_used_ledger", default=None)


def mode() -> str:
    """Return the strict, default-off organ switch."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def tier() -> str:
    """Return the strict gating tier: advisory (default, never blocks) or enforce (open sites block done)."""
    value = os.environ.get(TIER_ENV, "advisory").strip().lower()
    if value not in {"advisory", "enforce"}:
        raise ValueError(f"{TIER_ENV} must be advisory or enforce")
    return value


def bind_context(*, journal: object | None = None,
                 ledger: object | None = None) -> tuple[object | None, object | None]:
    """Bind optional request-local defaults for small host integrations."""
    old = (_ACTIVE_JOURNAL.get(), _ACTIVE_LEDGER.get())
    _ACTIVE_JOURNAL.set(journal)
    _ACTIVE_LEDGER.set(ledger)
    return old


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    digest: str
    readable: bool = True

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "digest": self.digest,
                "readable": self.readable}


@dataclass(frozen=True)
class WhereUsedSite:
    identity: str
    path: str
    kind: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    text: str = ""
    source: str = "ast"

    @property
    def span(self) -> dict[str, int]:
        return {"start_line": self.start_line, "end_line": self.end_line,
                "start_col": self.start_col, "end_col": self.end_col}

    @property
    def site_id(self) -> str:
        payload = json.dumps({
            "identity": self.identity, "path": self.path, "kind": self.kind,
            "span": self.span, "text": self.text,
        }, sort_keys=True, ensure_ascii=False).encode()
        return "site-" + hashlib.sha256(payload).hexdigest()[:20]

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id, "identity": self.identity, "path": self.path,
            "kind": self.kind, "span": self.span, "text": self.text,
            "source": self.source,
        }


@dataclass
class SiteObligation:
    obligation_id: str
    identity: str
    path: str
    span: dict[str, int]
    kind: str
    site_id: str
    reason_code: str = SITE_UNREVIEWED
    status: str = "OPEN"
    disposition: str | None = None
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()

    @property
    def open(self) -> bool:
        return self.status == "OPEN"

    def close(self, disposition: str, *, reason: str = "",
              evidence_refs: Iterable[str] = ()) -> None:
        self.status = "CLOSED"
        self.disposition = str(disposition)
        self.reason = str(reason)
        self.evidence_refs = tuple(str(item) for item in evidence_refs if str(item))

    def to_dict(self) -> dict[str, object]:
        return {
            "obligation_id": self.obligation_id, "reason_code": self.reason_code,
            "identity": self.identity, "path": self.path, "span": dict(self.span),
            "kind": self.kind, "site_id": self.site_id, "status": self.status,
            "open": self.open, "disposition": self.disposition, "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "tier": TIER_ADMITTED if tier() == "enforce" else TIER_CHECK_CANDIDATE,
        }


@dataclass(frozen=True)
class IdentityChange:
    identity: str
    kinds: tuple[str, ...]
    old_count: int
    new_count: int

    def to_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "kinds": list(self.kinds),
                "old_count": self.old_count, "new_count": self.new_count}


@dataclass(frozen=True)
class ImportEdge:
    source: str
    target: str
    module: str
    line: int

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target,
                "module": self.module, "line": self.line}


@dataclass
class CodePack:
    root: Path
    files: tuple[FileEntry, ...]
    contents: dict[str, str]
    sites: tuple[WhereUsedSite, ...]
    imports: tuple[ImportEdge, ...]
    include_literals: bool = True
    top_level_definitions: tuple[WhereUsedSite, ...] = ()

    def find(self, identity: str) -> tuple[WhereUsedSite, ...]:
        """Find AST sites plus exact raw literal occurrences, including spaces."""
        wanted = str(identity)
        found: list[WhereUsedSite] = [site for site in self.sites if site.identity == wanted]
        if self.include_literals and wanted:
            for site in _literal_sites_in_contents(self.contents, wanted):
                # Parsed Python already has a typed AST site for a name or
                # string on this line.  Keep raw grep as the fallback for
                # malformed Python and as the primary channel for registries,
                # fixtures, and other non-Python files.
                if site.path.endswith(".py") and any(
                        existing.path == site.path
                        and existing.start_line == site.start_line
                        and existing.identity == wanted
                        for existing in found):
                    continue
                found.append(site)
        return _dedupe_sites(found)

    def siblings(self, path: str) -> tuple[WhereUsedSite, ...]:
        """Return top-level definitions in the edited file as closure context."""
        return tuple(site for site in self.top_level_definitions
                     if site.path == path and site.kind == "definition")

    def file_list(self, prefix: str | None = None) -> tuple[FileEntry, ...]:
        if not prefix:
            return self.files
        prefix = prefix.strip("/")
        return tuple(item for item in self.files
                     if item.path == prefix or item.path.startswith(prefix + "/"))


@dataclass
class WhereUsedReport:
    fired: bool
    edit_path: str = ""
    changed_identities: tuple[str, ...] = ()
    identity_changes: tuple[IdentityChange, ...] = ()
    sites: tuple[WhereUsedSite, ...] = ()
    touched_site_ids: tuple[str, ...] = ()
    obligations: tuple[SiteObligation, ...] = ()
    files: tuple[FileEntry, ...] = ()
    imports: tuple[ImportEdge, ...] = ()
    event_id: str | None = None
    journaled: bool = False
    literal_channel: bool = True

    @property
    def open_obligations(self) -> tuple[SiteObligation, ...]:
        return tuple(item for item in self.obligations if item.open)

    def to_dict(self) -> dict[str, object]:
        if not self.fired:
            return {"schema": SCHEMA, "fired": False}
        return {
            "schema": SCHEMA, "fired": True, "edit_path": self.edit_path,
            "changed_identities": list(self.changed_identities),
            "identity_changes": [item.to_dict() for item in self.identity_changes],
            "sites": [item.to_dict() for item in self.sites],
            "touched_site_ids": list(self.touched_site_ids),
            "obligations": [item.to_dict() for item in self.obligations],
            "open_obligation_ids": [item.obligation_id for item in self.open_obligations],
            "files": [item.to_dict() for item in self.files],
            "imports": [item.to_dict() for item in self.imports],
            "event_id": self.event_id, "journaled": self.journaled,
            "literal_channel": self.literal_channel,
        }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_paths(root: Path, scope: Path | None = None) -> tuple[Path, ...]:
    root = root.resolve()
    start = (scope or root).resolve()
    if not start.is_relative_to(root):
        raise ValueError("where-used scope must be inside the repository root")
    if start.is_file():
        candidates = [start]
    else:
        candidates = []
        stack = [start]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
            except (OSError, NotADirectoryError):
                continue
            for entry in entries:
                relative = Path(entry.path).relative_to(root)
                if any(part in _SKIP_DIRS for part in relative.parts):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        candidates.append(Path(entry.path))
                except OSError:
                    continue
    paths = [path for path in candidates
             if path.is_file() and not any(part in _SKIP_DIRS for part in path.relative_to(root).parts)]
    return tuple(sorted(paths, key=lambda item: _relative(root, item)))


def list_files(root: str | Path, scope: str | Path | None = None,
               *, limit: int | None = None) -> tuple[FileEntry, ...]:
    root_path = Path(root).resolve()
    scope_path = None if scope is None else (root_path / scope if not Path(scope).is_absolute()
                                             else Path(scope))
    rows: list[FileEntry] = []
    for path in _iter_paths(root_path, scope_path):
        try:
            raw = path.read_bytes()
            raw.decode("utf-8")
        except (OSError, UnicodeError):
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            rows.append(FileEntry(_relative(root_path, path), size, "", readable=False))
            continue
        rows.append(FileEntry(_relative(root_path, path), len(raw),
                              hashlib.sha256(raw).hexdigest()))
    return tuple(rows if limit is None else rows[:max(0, int(limit))])


def _read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
        if b"\x00" in raw:
            return None
        return raw.decode("utf-8")
    except (OSError, UnicodeError):
        return None


def _source_lines(source: str) -> list[str]:
    return source.splitlines() or [""]


def _node_site(identity: str, path: str, kind: str, node: ast.AST,
               source: str, *, source_kind: str = "ast") -> WhereUsedSite:
    start_line = int(getattr(node, "lineno", 1) or 1)
    end_line = int(getattr(node, "end_lineno", start_line) or start_line)
    start_col = int(getattr(node, "col_offset", 0) or 0)
    end_col = int(getattr(node, "end_col_offset", start_col) or start_col)
    lines = _source_lines(source)
    text = lines[start_line - 1].strip() if 0 < start_line <= len(lines) else ""
    return WhereUsedSite(identity, path, kind, start_line, end_line,
                         start_col, end_col, text, source_kind)


class _AstPackVisitor(ast.NodeVisitor):
    def __init__(self, source: str, path: str):
        self.source = source
        self.path = path
        self.sites: list[WhereUsedSite] = []
        self.top_level_definitions: list[WhereUsedSite] = []
        self.imports: list[tuple[str, str, int]] = []
        self._definition_depth = 0

    def _add(self, identity: str, kind: str, node: ast.AST) -> None:
        if identity:
            self.sites.append(_node_site(identity, self.path, kind, node, self.source))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        site = _node_site(node.name, self.path, "definition", node, self.source)
        self.sites.append(site)
        if self._definition_depth == 0:
            self.top_level_definitions.append(site)
        self._definition_depth += 1
        self.generic_visit(node)
        self._definition_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        site = _node_site(node.name, self.path, "definition", node, self.source)
        self.sites.append(site)
        if self._definition_depth == 0:
            self.top_level_definitions.append(site)
        self._definition_depth += 1
        self.generic_visit(node)
        self._definition_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node: ast.arg) -> None:
        self._add(node.arg, "definition", node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._add(node.id, "definition" if isinstance(node.ctx, ast.Store) else "reference", node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add(node.attr, "reference", node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value:
            self._add(node.value, "literal", node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self._add(node.arg, "reference", node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name, "import", node)
            if alias.asname:
                self._add(alias.asname, "definition", node)
            self.imports.append((alias.name, alias.name, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = ("." * int(node.level or 0)) + (node.module or "")
        for alias in node.names:
            self._add(alias.name, "import", node)
            self._add(alias.asname or alias.name.split(".")[-1], "definition", node)
        self.imports.append((module, module, node.lineno))


def _module_target(root: Path, source_path: str, module: str) -> str | None:
    module = module.lstrip(".")
    if not module:
        return None
    direct = root.joinpath(*module.split("."))
    candidates = (direct.with_suffix(".py"), direct / "__init__.py")
    for target in candidates:
        if target.is_file():
            return _relative(root, target)
    # A relative import is resolved approximately from the source package.
    base = Path(source_path).parent
    target = root.joinpath(base, *module.split("."))
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file():
            return _relative(root, candidate)
    return None


def build_code_pack(root: str | Path, *, include_literals: bool = True,
                    overlays: Mapping[str, str] | None = None) -> CodePack:
    root_path = Path(root).resolve()
    normalized_overlays: dict[str, str] = {}
    for key, value in (overlays or {}).items():
        relative = PurePosixPath(str(key).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("overlay path escapes the workspace root")
        normalized = relative.as_posix()
        if not normalized or normalized == ".":
            raise ValueError("overlay path is empty")
        normalized_overlays[normalized] = str(value)
    overlays = normalized_overlays
    file_rows = list(list_files(root_path))
    known_paths = {item.path for item in file_rows}
    for relative, source in overlays.items():
        if relative in known_paths:
            continue
        raw = str(source).encode("utf-8")
        file_rows.append(FileEntry(relative, len(raw), hashlib.sha256(raw).hexdigest()))
    file_rows.sort(key=lambda item: item.path)
    contents: dict[str, str] = {}
    sites: list[WhereUsedSite] = []
    top_level_definitions: list[WhereUsedSite] = []
    imports: list[ImportEdge] = []
    for entry in file_rows:
        path = root_path / entry.path
        source = overlays.get(entry.path)
        if source is None:
            source = _read_text(path)
        if source is None:
            continue
        contents[entry.path] = source
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(source, filename=entry.path)
        except (SyntaxError, ValueError, TypeError):
            continue
        visitor = _AstPackVisitor(source, entry.path)
        visitor.visit(tree)
        sites.extend(visitor.sites)
        top_level_definitions.extend(visitor.top_level_definitions)
        for module, raw_module, line in visitor.imports:
            target = _module_target(root_path, entry.path, raw_module)
            if target:
                imports.append(ImportEdge(entry.path, target, module, line))
    return CodePack(
        root_path, tuple(file_rows), contents, tuple(_dedupe_sites(sites)),
        tuple(sorted(set(imports), key=lambda item: (item.source, item.line, item.target))),
        include_literals=include_literals,
        top_level_definitions=tuple(_dedupe_sites(top_level_definitions)))


def _literal_pattern(literal: str) -> re.Pattern[str] | None:
    value = str(literal)
    if not value:
        return None
    parts = re.split(r"\s+", value.strip())
    if not parts or not any(parts):
        return None
    return re.compile(r"\s+".join(re.escape(part) for part in parts))


def _literal_sites_in_contents(contents: Mapping[str, str], literal: str) -> tuple[WhereUsedSite, ...]:
    pattern = _literal_pattern(literal)
    if pattern is None:
        return ()
    found: list[WhereUsedSite] = []
    for path, source in sorted(contents.items()):
        lines = source.splitlines()
        for match in pattern.finditer(source):
            start_line = source.count("\n", 0, match.start()) + 1
            end_line = source.count("\n", 0, match.end()) + 1
            line_start = source.rfind("\n", 0, match.start()) + 1
            end_col = match.end() - (source.rfind("\n", 0, match.end()) + 1)
            text = lines[start_line - 1].strip() if lines and start_line <= len(lines) else ""
            found.append(WhereUsedSite(literal, path, "literal", start_line, end_line,
                                       match.start() - line_start, end_col, text,
                                       "literal_grep"))
    return tuple(found)


def literal_grep(root: str | Path, literal: str, *, pack: CodePack | None = None,
                 overlays: Mapping[str, str] | None = None) -> tuple[WhereUsedSite, ...]:
    """Search a literal as a single value; internal whitespace remains matchable."""
    if pack is not None:
        return tuple(site for site in pack.find(literal) if site.source == "literal_grep")
    root_path = Path(root).resolve()
    rows = list_files(root_path)
    contents = {}
    overlays = overlays or {}
    for entry in rows:
        source = overlays.get(entry.path)
        if source is None:
            source = _read_text(root_path / entry.path)
        if source is not None:
            contents[entry.path] = source
    return _literal_sites_in_contents(contents, literal)


def _dedupe_sites(sites: Iterable[WhereUsedSite]) -> tuple[WhereUsedSite, ...]:
    result: list[WhereUsedSite] = []
    seen: set[str] = set()
    for site in sites:
        if site.site_id in seen:
            continue
        seen.add(site.site_id)
        result.append(site)
    return tuple(sorted(result, key=lambda item: (
        item.path, item.start_line, item.start_col, item.identity, item.kind, item.source)))


def _identity_counter(source: str) -> collections.Counter[tuple[str, str]]:
    counter: collections.Counter[tuple[str, str]] = collections.Counter()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type == tokenize.NAME:
                    counter[("symbol", token.string)] += 1
                elif token.type == tokenize.STRING:
                    try:
                        value = ast.literal_eval(token.string)
                    except (ValueError, SyntaxError):
                        value = token.string
                    if isinstance(value, str) and value:
                        counter[("literal", value)] += 1
        except (IndentationError, tokenize.TokenError, UnicodeError):
            for value in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source):
                counter[("symbol", value)] += 1
        return counter
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            counter[("symbol", node.name)] += 1
        elif isinstance(node, ast.arg):
            counter[("symbol", node.arg)] += 1
        elif isinstance(node, ast.Name):
            counter[("symbol", node.id)] += 1
        elif isinstance(node, ast.Attribute):
            counter[("symbol", node.attr)] += 1
        elif isinstance(node, ast.alias):
            counter[("symbol", node.name)] += 1
            if node.asname:
                counter[("symbol", node.asname)] += 1
        elif isinstance(node, ast.keyword) and node.arg:
            counter[("symbol", node.arg)] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
            counter[("literal", node.value)] += 1
    return counter


def extract_identity_changes(old_source: str | None, new_source: str | None) -> tuple[IdentityChange, ...]:
    old_counter = _identity_counter(old_source or "")
    new_counter = _identity_counter(new_source or "")
    values = sorted({value for _, value in set(old_counter) | set(new_counter)})
    changes: list[IdentityChange] = []
    for value in values:
        old_by_kind = {kind: count for (kind, item), count in old_counter.items()
                       if item == value and count}
        new_by_kind = {kind: count for (kind, item), count in new_counter.items()
                       if item == value and count}
        if old_by_kind == new_by_kind:
            continue
        old_count = sum(old_by_kind.values())
        new_count = sum(new_by_kind.values())
        kinds = tuple(sorted(set(old_by_kind) | set(new_by_kind)))
        changes.append(IdentityChange(value, kinds, old_count, new_count))
    return tuple(changes)


def extract_changed_identities(old_source: str | None, new_source: str | None) -> tuple[str, ...]:
    return tuple(item.identity for item in extract_identity_changes(old_source, new_source))


def _changed_lines(before: str, after: str) -> set[int]:
    old_lines = before.splitlines()
    new_lines = after.splitlines()
    touched: set[int] = set()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        if tag != "equal":
            if new_start == new_end:
                touched.add(max(1, new_start))
            else:
                touched.update(range(new_start + 1, new_end + 1))
    return touched


def _obligation_for(site: WhereUsedSite) -> SiteObligation:
    return SiteObligation(
        obligation_id=f"where-used:{site.site_id}", identity=site.identity,
        path=site.path, span=site.span, kind=site.kind, site_id=site.site_id)


def _journal_emit(journal: object | None, *, edit_path: str,
                  changed_identities: Sequence[str], site_count: int,
                  open_count: int, files: int, imports: int,
                  open_obligation_ids: Sequence[str], edit_receipt_id: str | None,
                  literal_channel: bool) -> str | None:
    if journal is None:
        return None
    event = journal.emit(
        EVENT_KIND,
        producer="kernel_where_used",
        authority="HOST_OBSERVED",
        fired=True,
        edit_path=edit_path,
        edit_receipt_id=edit_receipt_id,
        changed_identities=list(changed_identities),
        changed_identity_count=len(changed_identities),
        site_count=site_count,
        open_site_count=open_count,
        open_obligation_ids=list(open_obligation_ids),
        file_count=files,
        import_edge_count=imports,
        literal_channel=literal_channel,
        open_obligation_count=open_count,
        dispatch=False,
    )
    if isinstance(event, Mapping):
        value = event.get("event_id")
        return str(value) if value else None
    return None


def analyze_accepted_edit(root: str | Path, path: str | Path,
                          before: str | None, after: str | None, *,
                          journal: object | None = None,
                          ledger: "WhereUsedLedger | None" = None,
                          edit_receipt_id: str | None = None,
                          include_literals: bool = True) -> WhereUsedReport:
    """Compute closure for one host-accepted edit.

    When the switch is off this returns a constant projection and does no file
    scan, journal call, or obligation mutation.
    """
    if mode() != "on":
        return WhereUsedReport(False, literal_channel=include_literals)
    root_path = Path(root).resolve()
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = root_path / path_obj
    relative = _relative(root_path, path_obj)
    before_text = before or ""
    after_text = after or ""
    changes = extract_identity_changes(before_text, after_text)
    identities = tuple(item.identity for item in changes)
    pack = build_code_pack(root_path, include_literals=include_literals,
                           overlays={relative: after_text})
    before_pack = build_code_pack(root_path, include_literals=include_literals,
                                  overlays={relative: before_text})
    found: list[WhereUsedSite] = []
    for identity in identities:
        found.extend(pack.find(identity))
        # Keep the removed definition/literal visible as a touched site.  The
        # current pack proves the new identity and all remaining references;
        # the before pack proves what the accepted edit replaced.
        found.extend(before_pack.find(identity))
    # A class/function edit must also expose untouched top-level siblings in the
    # same file.  This is the small but important JobQueue-vs-Job/QueueStats
    # closure that a single-class patch otherwise hides.
    if identities:
        found.extend(pack.siblings(relative))
        found.extend(before_pack.siblings(relative))
    sites = _dedupe_sites(found)
    changed_lines = _changed_lines(before_text, after_text)
    touched = tuple(site.site_id for site in sites
                    if site.path == relative and site.start_line in changed_lines
                    and site.identity in identities)
    touched_set = set(touched)
    obligations = tuple(_obligation_for(site) for site in sites if site.site_id not in touched_set)
    active_journal = journal if journal is not None else _ACTIVE_JOURNAL.get()
    if active_journal is None:
        raise RuntimeError("where-used closure requires a journal when enabled")
    event_id = _journal_emit(
        active_journal, edit_path=relative, changed_identities=identities,
        site_count=len(sites), open_count=len(obligations), files=len(pack.files),
        imports=len(pack.imports), edit_receipt_id=edit_receipt_id,
        open_obligation_ids=tuple(item.obligation_id for item in obligations),
        literal_channel=include_literals)
    report = WhereUsedReport(
        True, relative, identities, changes, sites, touched, obligations,
        pack.files, pack.imports, event_id, event_id is not None, include_literals)
    if ledger is not None:
        ledger._ingest_report(report)
    return report


observe_accepted_edit = analyze_accepted_edit


class WhereUsedLedger:
    """Revision-local obligation ledger for closure and completion authority."""

    def __init__(self, *, journal: object | None = None):
        self.journal = journal
        self._obligations: dict[str, SiteObligation] = {}
        self._reports: list[WhereUsedReport] = []
        self._read_receipts: dict[str, dict[str, object]] = {}
        # Optional N2 hook (STATEFRAME_PARTIAL_OBSERVATION): callable(refs, claim=)
        # returning a rejection string when cited evidence is host-cut.
        self.coverage_gate = None
        self._authority_required: set[str] = set()
        self._authority_passed: set[str] = set()
        self._edit_keys: set[str] = set()

    def observe_edit(self, root: str | Path, path: str | Path,
                     before: str | None, after: str | None, *,
                     edit_receipt_id: str | None = None) -> WhereUsedReport:
        root_path = Path(root).resolve()
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = root_path / path_obj
        key = hashlib.sha256((str(path_obj.resolve()) + "\0" + (before or "") + "\0" +
                              (after or "")).encode()).hexdigest()
        if key in self._edit_keys:
            for report in reversed(self._reports):
                if report.edit_path == _relative(root_path, path_obj):
                    return report
        self._edit_keys.add(key)
        report = analyze_accepted_edit(
            root_path, path_obj, before, after, journal=self.journal, ledger=None,
            edit_receipt_id=edit_receipt_id)
        self._ingest_report(report)
        return report

    def _ingest_report(self, report: WhereUsedReport) -> None:
        if not report.fired:
            return
        self._reports.append(report)
        touched_keys = {(site.identity, site.path, site.start_line, site.kind)
                        for site in report.sites if site.site_id in set(report.touched_site_ids)}
        for item in self._obligations.values():
            key = (item.identity, item.path, int(item.span.get("start_line", 0)), item.kind)
            if item.open and key in touched_keys:
                item.close("EDIT_TOUCH", reason="accepted edit touched this site",
                           evidence_refs=(report.event_id or "",))
                self._journal_close(item, authority="HOST_OBSERVED")
        for item in report.obligations:
            existing = self._obligations.get(item.obligation_id)
            if existing is None:
                self._obligations[item.obligation_id] = item
            elif existing.status != "OPEN":
                # A later accepted edit makes the same site a fresh obligation.
                existing.status = "OPEN"
                existing.disposition = None
                existing.reason = ""
                existing.evidence_refs = ()

    def all_obligations(self) -> tuple[SiteObligation, ...]:
        return tuple(self._obligations.values())

    def open_obligations(self) -> tuple[SiteObligation, ...]:
        return tuple(item for item in self._obligations.values() if item.open)

    def record_read_receipt(self, obligation_id: str,
                            receipt: Mapping[str, object]) -> None:
        if not isinstance(receipt, Mapping):
            raise TypeError("read receipt must be a mapping")
        receipt_id = str(receipt.get("receipt_id", receipt.get("event_id", ""))).strip()
        if not receipt_id:
            raise ValueError("read receipt needs receipt_id")
        if obligation_id not in self._obligations:
            raise KeyError(obligation_id)
        self._read_receipts[str(obligation_id)] = dict(receipt)

    def _close(self, obligation_id: str, disposition: str, *, reason: str,
               evidence_refs: Iterable[str] = (), authority: str = "MODEL_DISPOSITION") -> SiteObligation:
        item = self._obligations.get(str(obligation_id))
        if item is None:
            raise KeyError(obligation_id)
        if not reason.strip():
            raise ValueError("site closure requires a reason")
        item.close(disposition, reason=reason, evidence_refs=evidence_refs)
        self._journal_close(item, authority=authority)
        return item

    def _journal_close(self, item: SiteObligation, *, authority: str) -> None:
        if self.journal is None:
            return
        self.journal.emit(
            "where_used_site_closed",
            producer="kernel_where_used",
            authority=authority,
            obligation_id=item.obligation_id,
            disposition=item.disposition,
            reason=item.reason,
            evidence_refs=list(item.evidence_refs),
            dispatch=False,
        )

    def dispose_not_affected(self, obligation_id: str, *, reason: str,
                             read_receipt: str | None) -> SiteObligation:
        if not read_receipt or not str(read_receipt).strip():
            raise ValueError("NOT_AFFECTED requires a read receipt")
        receipt = self._read_receipts.get(str(obligation_id))
        if receipt is None or str(receipt.get("receipt_id", "")) != str(read_receipt):
            raise ValueError("read receipt does not match the site obligation")
        if self.coverage_gate is not None:
            rejection = self.coverage_gate((str(read_receipt),), claim="NOT_AFFECTED")
            if rejection:
                raise ValueError(rejection)
        return self._close(obligation_id, "NOT_AFFECTED", reason=reason,
                           evidence_refs=(str(read_receipt),), authority="MODEL_DISPOSITION")

    def accept_dispute(self, obligation_id: str, *, reason: str, ground: str | None = None,
                       host_accepted: bool = False) -> SiteObligation:
        if ground is not None and ground not in DISPUTE_GROUNDS:
            raise ValueError(f"dispute ground must be one of {sorted(DISPUTE_GROUNDS)}")
        if ground is not None:
            reason = f"{ground}: {reason}"
        if not host_accepted:
            raise ValueError("only the host can accept a site dispute")
        return self._close(obligation_id, "HOST_ACCEPTED_DISPUTE", reason=reason,
                           authority="HOST_ACCEPTED_DISPUTE")

    def required_evidence_gaps(self) -> tuple[object, ...]:
        from kernel_completion_authority import RequiredEvidenceGap
        return tuple(RequiredEvidenceGap(
            reason_code="REQUIRED_EVIDENCE_GAP",
            obligation_id=item.obligation_id,
            detail=(f"where-used site {item.identity} at {item.path}:"
                    f"{item.span.get('start_line', 0)} is unreviewed"),
            evidence_refs=item.evidence_refs,
        ) for item in self.open_obligations())

    def render(self) -> str:
        return json.dumps({"schema": SCHEMA,
                           "open_obligations": [item.to_dict() for item in self.open_obligations()]},
                          sort_keys=True, ensure_ascii=False)

    def frame_projection(self) -> dict[str, object]:
        """Bounded host-fact view that lets the model visit or dispute sites."""
        latest = self._reports[-1] if self._reports else None
        return {
            "schema": SCHEMA,
            "authority": "HOST_OBSERVED",
            "tier": TIER_ADMITTED if tier() == "enforce" else TIER_CHECK_CANDIDATE,
            "blocks_completion": tier() == "enforce",
            "resolution": RESOLUTION_MENU,
            "changed_identities": list(latest.changed_identities) if latest else [],
            "open_obligations": [item.to_dict() for item in self.open_obligations()],
            "open_count": len(self.open_obligations()),
            "event_id": latest.event_id if latest else None,
            "file_count": len(latest.files) if latest else 0,
            "import_edge_count": len(latest.imports) if latest else 0,
        }


def sync_completion_authority(authority: object, ledger: WhereUsedLedger) -> object:
    """Project open site debt into the existing completion authority (enforce tier only; §62)."""
    if mode() != "on":
        return authority
    if tier() != "enforce":
        return authority
    for item in ledger.open_obligations():
        obligation_id = item.obligation_id
        required = getattr(authority, "required_evidence", frozenset())
        if obligation_id not in required:
            authority.require_evidence(obligation_id, reason_code="REQUIRED_EVIDENCE_GAP")
        ledger._authority_required.add(obligation_id)
    for obligation_id in tuple(ledger._authority_required):
        if any(item.obligation_id == obligation_id for item in ledger.open_obligations()):
            continue
        if obligation_id in ledger._authority_passed:
            continue
        digest = getattr(authority, "candidate_digest", None)
        revision = getattr(authority, "current_revision", None)
        authority.record_receipt({
            "receipt_id": f"where-used-pass:{obligation_id}:{revision or digest or 'current'}",
            "obligation_id": obligation_id, "status": "PASS",
            "candidate_digest": digest, "revision": revision,
        })
        ledger._authority_passed.add(obligation_id)
    return authority


def render_file_list(root: str | Path, scope: str | Path | None = None,
                     *, limit: int = 200) -> str:
    rows = list_files(root, scope, limit=limit)
    return "\n".join(f"{item.path}\t{item.size} bytes" for item in rows) or "(no files)"


__all__ = [
    "CodePack", "EVENT_KIND", "FileEntry", "IdentityChange", "ImportEdge",
    "SITE_UNREVIEWED", "SiteObligation", "WhereUsedLedger", "WhereUsedReport",
    "WhereUsedSite", "analyze_accepted_edit", "bind_context", "build_code_pack",
    "enabled", "extract_changed_identities", "extract_identity_changes", "list_files",
    "literal_grep", "mode", "observe_accepted_edit", "render_file_list",
    "sync_completion_authority",
]
