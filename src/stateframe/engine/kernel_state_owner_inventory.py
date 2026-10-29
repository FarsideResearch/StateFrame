"""Q2 (E2): host state-owner inventory for a stateful ticket.

When the ticket's state is a persisted field (a column, struct field, class
attribute) the expert first lists who declares it, who reads it, who writes it,
and which helpers produce its value, with exact host line numbers.  The model
then designs the transition instead of searching for a writer that does not
exist.  Everything here is a host grep/outline fact over the parent tree,
computed once per run (the inventory describes the PARENT) and rendered beside
the other site facts, so it counts against the request budget.

Owner fields are resolved language-generically: type spans come from the
shared parser outline (``mc_repo_intel_vendor.ts_dispatch.outline_source``;
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
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dc_field
from pathlib import Path

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


def _read(root: Path, relative: str) -> str | None:
    path = Path(root) / relative
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _outline(path: str, source: str) -> Mapping[str, object]:
    try:
        from kernel_request_source_window import _outline_for
        return _outline_for(path, source)
    except (ImportError, OSError, TypeError, ValueError):
        return {}


def _parser_capability(path: str, source: str) -> str:
    outline = _outline(path, source)
    if not outline:
        return "UNAVAILABLE"
    return "DEGRADED" if outline.get("degraded") else "AVAILABLE"


def _source_revision_vector(
        sources: Mapping[str, str], parent_sources: Mapping[str, str],
) -> tuple[str, dict[str, dict[str, object]]]:
    digest = hashlib.sha256()
    identities: dict[str, dict[str, object]] = {}
    for path in sorted(sources):
        source = sources[path]
        current = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
        parent = parent_sources.get(path)
        parent_revision = ("sha256:" + hashlib.sha256(parent.encode("utf-8")).hexdigest()
                           if isinstance(parent, str) else None)
        identities[path] = {
            "current_revision": current,
            "parent_revision": parent_revision,
            "status": ("UNKNOWN" if parent is None else
                       "UNCHANGED" if source == parent else "CHANGED"),
        }
        digest.update(path.encode() + b"\0" + source.encode("utf-8", "replace"))
    return digest.hexdigest(), identities


def _runtime_cache_key(
        root: Path, candidates: Sequence[str], parent_sources: Mapping[str, str],
) -> tuple[object, ...]:
    """Fingerprint current files, parser capability, and parent identity."""

    scan = scan_product_files(root)
    sources: dict[str, str] = {}
    capabilities: dict[str, str] = {}
    for path in scan.paths:
        source = _read(root, path)
        if source is None:
            capabilities[path] = "UNAVAILABLE"
            continue
        sources[path] = source
        capabilities[path] = _parser_capability(path, source)
    dependency_revision, _identities = _source_revision_vector(sources, parent_sources)
    parent_revision = tuple(
        (path, "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest())
        for path, source in sorted(parent_sources.items())
    )
    return (
        tuple(candidates), dependency_revision, tuple(sorted(capabilities.items())),
        tuple(scan.paths), tuple(scan.skipped_files), scan.complete, parent_revision,
    )


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


def declared_fields(path: str, source: str, candidates: Sequence[str]) -> list[OwnerField]:
    """Resolve a named field plus its bounded correlated declaration group."""
    outline = _outline(path, source)
    lines = source.splitlines()
    method_rows = [row for row in outline.get("functions", ()) or () if isinstance(row, Mapping)]
    type_rows = [row for row in outline.get("types", ()) or () if isinstance(row, Mapping)]
    if not type_rows:
        for number, line in enumerate(lines, 1):
            match = _TYPE_LINE.match(line)
            if match:
                name = next((value for value in match.groups() if value), "")
                kind = "struct" if match.group("struct") else "interface" if match.group("interface") else "class"
                type_rows.append({"line": number, "end_line": number,
                                  "name": name, "kind": kind})
    if path.casefold().endswith(".py"):
        try:
            tree = ast.parse(source)
            method_rows = [
                {"line": node.lineno, "end_line": node.end_lineno or node.lineno}
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            type_rows = [
                {"line": node.lineno, "end_line": node.end_lineno or node.lineno,
                 "name": node.name, "kind": "class"}
                for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
        except (SyntaxError, ValueError):
            pass
    expanded_types: list[Mapping[str, object]] = []
    source_lines = source.splitlines()
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
    methods = [(int(row.get("line", 0) or 0), int(row.get("end_line", 0) or 0))
               for row in method_rows]
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


def _field_contexts(path: str, source: str, sites: Sequence[Site]) -> list[Site]:
    """Attach bounded enclosing effect/commit/catch context to field sites."""
    outline = _outline(path, source)
    lines = source.splitlines()
    contexts: list[Site] = []
    functions = [row for row in outline.get("functions", ()) or ()
                 if isinstance(row, Mapping)]
    if path.casefold().endswith(".py"):
        try:
            tree = ast.parse(source)
            functions = [
                {"line": node.lineno, "end_line": node.end_lineno or node.lineno,
                 "name": node.name}
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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


def _helpers(root: Path, files: Sequence[str], sources: Mapping[str, str],
             fields: Sequence[OwnerField], candidates: Sequence[str]) -> list[Helper]:
    stems = {word for item in [*(f.name for f in fields), *candidates]
             for word in item.lower().split("_") if len(word) >= 4}
    explicit = {str(item).rsplit(".", 1)[-1].casefold() for item in candidates}
    explicit.update(item.name.casefold() for item in fields)
    scored: list[tuple[tuple[int, int, str, int], Helper]] = []
    for path in files:
        source = sources.get(path)
        if source is None:
            continue
        lines = source.splitlines()
        for row in _outline(path, source).get("functions", ()) or ():
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
            if source is None:
                continue
            for number, line in enumerate(source.splitlines(), 1):
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
                    scan: ProductFileScan | None = None) -> Inventory | None:
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
    for path in files:
        source = _read(base, path)
        if source is None:
            parser_capabilities[path] = "UNAVAILABLE"
            skipped.append({"path": path, "status": "UNKNOWN",
                            "reason": "file is unreadable or exceeds bounded size"})
            continue
        sources[path] = source
        parser_capabilities[path] = _parser_capability(path, source)
        if any(cand.casefold() in source.casefold() for cand in candidates):
            fields.extend(declared_fields(path, source, candidates))
    if not fields:
        if (file_scan.complete and not skipped
                and all(value == "AVAILABLE" for value in parser_capabilities.values())):
            return None
        parent_sources = parent_sources or {}
        dependency_revision, identities = _source_revision_vector(sources, parent_sources)
        return Inventory(candidates, [], _helpers(base, files, sources, [], candidates),
                         dependency_revision[:12], len(sources), complete=False,
                         skipped_files=tuple(skipped),
                         dependency_revision=dependency_revision, identities=identities,
                         parser_capabilities=parser_capabilities)
    fields = fields[:_MAX_FIELDS]
    declarations = {(item.path, item.line) for item in fields}
    for item in fields:
        read, write = _site_patterns(item.name)
        for path, source in sources.items():
            if item.name not in source:
                continue
            for number, line in enumerate(source.splitlines(), 1):
                if (path, number) in declarations or line.strip().startswith(("#", "//")):
                    continue
                if write.search(line):
                    item.writers.append(Site(path, number, line.strip()))
                elif read.search(line):
                    item.readers.append(Site(path, number, line.strip()))
        for context_path, context_source in sources.items():
            sites = [site for site in [*item.readers, *item.writers]
                     if site.path == context_path]
            if sites:
                item.contexts.extend(_field_contexts(context_path, context_source, sites))
    parent_sources = parent_sources or {}
    dependency_revision, identities = _source_revision_vector(sources, parent_sources)
    complete = (file_scan.complete and not skipped
                and all(value == "AVAILABLE" for value in parser_capabilities.values()))
    return Inventory(candidates, fields, _helpers(base, files, sources, fields, candidates),
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
        from kernel_request_source_window import _ticket_text, entry_symbol
        ticket = _ticket_text(runtime)
        absent = _entry_absent(runtime)
        symbol = entry_symbol(runtime)
        candidates = tuple(item for item in owner_candidates(ticket, pairs=absent)
                           if item != symbol and item != symbol.rsplit(".", 1)[-1])
        from kernel_request_source_window import _parent_sources
        parent_sources = _parent_sources(runtime)
        current_key = _runtime_cache_key(Path(root), candidates, parent_sources)
        cached = getattr(runtime, "_state_owner_inventory", False)
        if (cached is not False
                and getattr(runtime, "_state_owner_inventory_cache_key", None) == current_key):
            return cached
        inventory = build_inventory(Path(root), candidates, parent_sources=parent_sources)
    try:
        setattr(runtime, "_state_owner_inventory", inventory)
        setattr(runtime, "_state_owner_inventory_cache_key", current_key)
    except (AttributeError, TypeError):
        pass
    return inventory


def owner_writer_count(runtime: object) -> int | None:
    inventory = getattr(runtime, "_state_owner_inventory", None)
    return (inventory.writer_count if isinstance(inventory, Inventory) and inventory.complete
            else None)


__all__ = ["ENV", "HEADER", "Helper", "Inventory", "OwnerField", "ProductFileScan", "Site",
           "build_inventory", "declared_fields", "mode", "owner_candidates", "owner_writer_count",
           "product_files", "render", "runtime_inventory", "scan_product_files"]
