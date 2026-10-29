"""Public-only localization for multi-file repair rows.

The solver may use a ticket, declared parent sources, and a deterministic
parent symbol index.  Nothing in this module accepts evaluator/fix material.
That boundary makes a private evaluator location non-interfering by
construction rather than by filtering a leaked location after the fact.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


_QUALIFIED = re.compile(
    r"(?<![A-Za-z0-9_$./-])"
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)*"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    # A final period can be sentence punctuation, but a period followed by
    # another identifier/path character is still part of a longer name.
    r"(?![A-Za-z0-9_$/-]|\.(?=[A-Za-z0-9_$./-]))"
)
_BACKTICKED = re.compile(
    r"`\s*((?:[A-Za-z_$][A-Za-z0-9_$]*\.)*"
    r"[A-Za-z_$][A-Za-z0-9_$]*)(?:\s*\(\s*\))?\s*`"
)
_INVOKED = re.compile(
    r"(?<![A-Za-z0-9_$./-])"
    r"((?:[A-Za-z_$][A-Za-z0-9_$]*\.)*"
    r"[A-Za-z_$][A-Za-z0-9_$]*)\s*\(\s*\)"
)
_PATH = re.compile(r"(?<![A-Za-z0-9_./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?(?:def|fn|func|function)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
_TYPE_DEFINITION = re.compile(
    r"^\s*(?:export\s+)?(?:class|interface|struct|type)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
_STOPWORDS = frozenset({
    "after", "before", "change", "changed", "code", "file", "files", "from",
    "into", "module", "public", "return", "source", "task", "test", "the",
    "this", "ticket", "with",
})
_PRIVATE_MARKERS = re.compile(
    r"(?i)(?:grader|graders|gold|private|hidden|oracle|sentinel|"
    r"expected[_ -]?patch|grade[_ -]?history|task[_ -]?id|report[_ -]?text)"
)
_PRIVATE_PATH_PARTS = frozenset({
    ".git", ".stateframe",
    "__pycache__", "node_modules", "dist", "build", "coverage", ".mission",
    "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
    "evaluator", "evaluators", "oracle", "sentinel", "reports",
})


@dataclass(frozen=True)
class ParentSymbol:
    path: str
    name: str
    terminal: str
    line: int
    kind: str


def _code_shaped(value: str, *, explicit: bool = False) -> bool:
    normalized = value.strip()
    marked = normalized.startswith("`") and normalized.endswith("`")
    if marked:
        normalized = normalized[1:-1].strip()
    invoked = bool(re.search(r"\(\s*\)$", normalized))
    if invoked:
        normalized = re.sub(r"\s*\(\s*\)$", "", normalized)
    terminal = normalized.rsplit(".", 1)[-1]
    return ("_" in terminal or "$" in terminal
            or bool(re.search(r"[a-z][A-Z]", terminal))
            or terminal.startswith("_") or marked or explicit or invoked)


def public_identifiers(ticket: str) -> tuple[str, ...]:
    """Return explicit qualified/code-shaped identifiers in ticket order."""
    values: list[tuple[int, str, bool]] = []
    for match in _QUALIFIED.finditer(str(ticket)):
        value = match.group(0)
        if _code_shaped(value) or "." in value:
            values.append((match.start(), value, False))
    for pattern in (_BACKTICKED, _INVOKED):
        for match in pattern.finditer(str(ticket)):
            value = match.group(1)
            if _code_shaped(value, explicit=True) or "." in value:
                values.append((match.start(1), value, True))
    seen: set[str] = set()
    result: list[str] = []
    for _position, value, explicit in sorted(values):
        key = value.casefold()
        if key not in seen and (explicit or key not in _STOPWORDS):
            seen.add(key)
            result.append(value)
    return tuple(result)


def public_ticket(ticket: str) -> str:
    """Project prose to public locator text before any selection or indexing.

    Private/evaluator material is dropped before regex extraction, so adding a
    hidden sentinel cannot alter public candidates, cache keys, or segments.
    """
    lines: list[str] = []
    for line in str(ticket).splitlines():
        marker = _PRIVATE_MARKERS.search(line)
        public_line = line[:marker.start()] if marker is not None else line
        if public_line.strip():
            lines.append(public_line)
    return "\n".join(lines)


def public_path(path: object) -> str | None:
    raw = str(path or "").replace("\\", "/")
    if not raw or raw.startswith("/") or "\x00" in raw:
        return None
    raw_parts = raw.split("/")
    if any(part == ".." for part in raw_parts):
        return None
    while raw_parts and raw_parts[0] == ".":
        raw_parts.pop(0)
    if not raw_parts or any(part in {"", "."} for part in raw_parts):
        return None
    normalized = "/".join(raw_parts)
    parts = normalized.split("/")
    if any(part.casefold() in _PRIVATE_PATH_PARTS or part.startswith(".") for part in parts):
        return None
    return normalized


def _python_symbols(path: str, source: str) -> list[ParentSymbol]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    result: list[ParentSymbol] = []

    def visit(nodes: Sequence[ast.AST], prefix: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = ".".join((*prefix, node.name))
            result.append(ParentSymbol(path, name, node.name, int(node.lineno),
                                       "class" if isinstance(node, ast.ClassDef) else "function"))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    return result


def _lexical_symbols(path: str, source: str) -> list[ParentSymbol]:
    """Conservative fallback for parser misses across the public language set."""

    result: list[ParentSymbol] = []
    in_script = path.casefold().endswith(".vue")
    script_open = False
    patterns = (
        (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|fn|func|function)\s+"
                    r"([A-Za-z_$][\w$]*)"), "function"),
        (re.compile(r"^\s*(?:export\s+)?(?:class|interface|struct|trait|enum|type)\s+([A-Za-z_$][\w$]*)"), "type"),
        (re.compile(r"^\s*(?:namespace|module|package)\s+([A-Za-z_$][\w$.]*)"), "namespace"),
        (re.compile(r"^\s*(?:public|private|protected|static|async|final|override|virtual|\s)*"
                    r"([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?:[:{]|=>)"), "method"),
    )
    for line_number, line in enumerate(source.splitlines(), 1):
        if in_script:
            if re.search(r"<script\b", line, re.IGNORECASE):
                script_open = True
                continue
            if re.search(r"</script\s*>", line, re.IGNORECASE):
                script_open = False
                continue
            if not script_open:
                continue
        if line.lstrip().startswith(("#", "//", "/*", "*", "<!--", "--")):
            continue
        for pattern, kind in patterns:
            match = pattern.search(line)
            if match is not None:
                result.append(ParentSymbol(path, match.group(1), match.group(1), line_number, kind))
                break
        alias_match = re.search(
            r"\b(?:export|import)\s*\{[^}\n]*\b([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)",
            line,
        )
        if alias_match is not None:
            result.append(ParentSymbol(path, alias_match.group(2), alias_match.group(2), line_number, "alias"))
    return result


def _outline_symbols(path: str, source: str) -> list[ParentSymbol]:
    """Use the shared outline for polyglot definitions before lexical fallback."""

    try:
        from mc_repo_intel_vendor.ts_dispatch import outline_source

        outline = outline_source(source, path)
        result: list[ParentSymbol] = []
        for group, kind in (("types", "type"), ("functions", "function")):
            for row in outline.get(group, ()) or ():
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                parent = str(row.get("parent") or "").strip()
                qualified = ".".join(part for part in (parent, name) if part)
                result.append(ParentSymbol(
                    path, qualified, name,
                    int(row.get("line", 0) or 0), kind,
                ))
        lexical = _lexical_symbols(path, source)
        known = {(item.name, item.line, item.kind) for item in result}
        result.extend(item for item in lexical if (item.name, item.line, item.kind) not in known)
        if result:
            return result
    except (ImportError, OSError, TypeError, ValueError):
        pass

    return _lexical_symbols(path, source)


def build_parent_index(parent_sources: Mapping[str, str]) -> tuple[ParentSymbol, ...]:
    """Index definitions from public parent bytes in canonical path order."""
    result: list[ParentSymbol] = []
    for path in sorted(parent_sources):
        source = parent_sources[path]
        relative = public_path(path)
        if relative is None:
            continue
        result.extend(_python_symbols(relative, source) if relative.endswith(".py")
                      else _outline_symbols(relative, source))
    return tuple(result)


def _path_score(path: str, ticket: str, declared: Sequence[str]) -> tuple[int, int, str]:
    normalized = path.replace("\\", "/")
    path_hits = tuple(match.group(0) for match in _PATH.finditer(ticket))
    exact = any(normalized == item or normalized.endswith("/" + item) for item in path_hits)
    mention = normalized.casefold() in ticket.casefold()
    canonical_declared = tuple(sorted(dict.fromkeys(str(item) for item in declared)))
    return (0 if exact else 1 if mention else 2,
            canonical_declared.index(normalized) if normalized in canonical_declared else len(canonical_declared),
            normalized)


def _symbol_match(symbol: ParentSymbol, identifier: str) -> tuple[int, int, str] | None:
    wanted = identifier.casefold()
    qualified = symbol.name.casefold()
    terminal = symbol.terminal.casefold()
    if qualified == wanted:
        return (0, 0, symbol.name)
    if terminal == wanted.rsplit(".", 1)[-1]:
        # A qualified ticket name still outranks a bare lexical terminal, but
        # terminal resolution keeps `pkg.module.fn` usable in source that
        # defines only `fn`.
        return (1, 0 if "." in identifier else 1, symbol.name)
    return None


def derive_public_location(ticket: str, declared_paths: Sequence[str],
                           parent_sources: Mapping[str, str]) -> dict[str, object]:
    """Derive a stable public row location from ticket and parent bytes only."""
    declared = tuple(dict.fromkeys(path for path in (public_path(item) for item in declared_paths)
                                   if path))
    if not declared:
        return {"changed_functions": {}, "entry": {}, "fn": "", "path": "", "rel": ""}
    index = build_parent_index({path: parent_sources[path] for path in declared
                                if path in parent_sources})
    identifiers = public_identifiers(public_ticket(ticket))
    candidates: list[tuple[tuple[int, int, int, int, str, str], ParentSymbol]] = []
    for symbol in index:
        if symbol.path not in declared:
            continue
        path_rank = _path_score(symbol.path, public_ticket(ticket), declared)
        for identifier in identifiers:
            match = _symbol_match(symbol, identifier)
            if match is not None:
                # A path explicitly named by the public ticket is stronger than
                # a bare terminal match in an unrelated declared file.  The
                # evaluator never participates in this ordering.
                candidates.append(((match[0], path_rank[0], match[1], path_rank[1],
                                    symbol.name, symbol.path), symbol))
    if candidates:
        symbol = min(candidates, key=lambda item: item[0])[1]
        location = {
            "path": symbol.path,
            "rel": symbol.path,
            "fn": symbol.name,
            "entry": {"path": symbol.path, "fn": symbol.name},
            "changed_functions": {symbol.path: [symbol.name]},
        }
        return location
    path = min(declared, key=lambda item: _path_score(item, ticket, declared))
    return {
        "path": path,
        "rel": path,
        "fn": "",
        "entry": {"path": path, "fn": ""},
        "changed_functions": {},
    }


def public_candidate_sites(
        ticket: str,
        declared_paths: Sequence[str],
        parent_sources: Mapping[str, str],
        *,
        max_candidates: int = 3,
) -> dict[str, object]:
    """Return bounded public symbol sites without selecting an edit target."""
    bound = max(1, min(int(max_candidates), 3))
    declared = tuple(dict.fromkeys(path for path in (public_path(item) for item in declared_paths)
                                   if path))
    safe_ticket = public_ticket(ticket)
    identifiers = public_identifiers(safe_ticket)
    index = build_parent_index({path: parent_sources[path] for path in declared
                                if path in parent_sources})
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for symbol in index:
        if symbol.path not in declared:
            continue
        ranked_matches = [(_symbol_match(symbol, identifier), identifier)
                          for identifier in identifiers]
        ranked_matches = [item for item in ranked_matches if item[0] is not None]
        matches = [identifier for _rank, identifier in ranked_matches]
        if not matches:
            continue
        identity = (symbol.path, symbol.name, symbol.line)
        if identity in seen:
            continue
        seen.add(identity)
        best_match = min(item[0] for item in ranked_matches)
        path_rank = _path_score(symbol.path, safe_ticket, declared)
        rows.append({
            "path": symbol.path, "symbol": symbol.name, "line": symbol.line,
            "kind": symbol.kind, "matched_identifiers": matches[:4],
            "selection_sources": ["public_ticket", "declared_source", "lexical_match"],
            "_rank": (best_match[0], path_rank[0], best_match[1], path_rank[2],
                      symbol.path, symbol.line, symbol.name),
        })
    rows.sort(key=lambda row: row["_rank"])
    for row in rows:
        row.pop("_rank", None)
    if len(rows) > bound:
        return {
            "status": "PARTIAL", "candidates": rows[:bound],
            "omissions": [{"kind": "candidate_bound", "omitted": len(rows) - bound}],
        }
    if len(rows) == 1:
        status = "FOUND"
    elif len(rows) > 1:
        status = "AMBIGUOUS"
    else:
        status = "NOT_FOUND" if identifiers else "UNAVAILABLE"
    return {"status": status, "candidates": rows, "omissions": []}


def public_location_fingerprint(location: Mapping[str, object]) -> str:
    """Canonical JSON-compatible projection useful for noninterference tests."""
    entry = location.get("entry") if isinstance(location.get("entry"), Mapping) else {}
    return "|".join((str(location.get("path", "")), str(entry.get("path", "")),
                      str(location.get("fn", "")), str(entry.get("fn", "")),
                      str(location.get("rel", ""))))


__all__ = ["ParentSymbol", "build_parent_index", "derive_public_location",
           "public_candidate_sites", "public_identifiers", "public_location_fingerprint",
           "public_path", "public_ticket"]
