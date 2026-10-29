"""Incremental symbol/caller navigation without owning change application."""
from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass

from kernel_gap_switches import emit, truth_enabled


SWITCH = "STATEFRAME_GAPS_NAVIGATION"
PRODUCER = "kernel_gap_navigation"
MAX_RESULTS = 8


@dataclass(frozen=True)
class SymbolEntry:
    name: str
    path: str
    kind: str
    aliases: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class NavigationIndex:
    revision: str
    entries: tuple[SymbolEntry, ...]
    importers: tuple[tuple[str, str], ...] = ()

    def definitions(self, name: str) -> tuple[SymbolEntry, ...]:
        wanted = str(name).strip()
        return tuple(entry for entry in self.entries if entry.name == wanted)

    def callers(self, name: str) -> tuple[str, ...]:
        wanted = str(name).strip()
        paths = [path for symbol, path in self.importers if symbol == wanted]
        return tuple(dict.fromkeys(paths))[:MAX_RESULTS]

    def to_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "entries": [entry.__dict__ for entry in self.entries],
            "importers": [{"symbol": symbol, "path": path}
                          for symbol, path in self.importers],
        }


@dataclass(frozen=True)
class NavigationResult:
    query: str
    definitions: tuple[SymbolEntry, ...]
    callers: tuple[str, ...]
    ambiguous: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "definitions": [entry.__dict__ for entry in self.definitions],
            "callers": list(self.callers),
            "ambiguous": self.ambiguous,
            "reason": self.reason,
        }


def _module_name(path: str) -> str:
    value = str(path).replace("\\", "/").rsplit(".", 1)[0].replace("/", ".")
    return value[:-9] if value.endswith(".__init__") else value


def _entry_for_file(path: str, source: str) -> tuple[list[SymbolEntry], list[tuple[str, str]]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return [], []
    entries: list[SymbolEntry] = []
    importers: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            entries.append(SymbolEntry(node.name, path, type(node).__name__))
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            for alias in node.names:
                name = str(alias.name)
                imported_as = str(alias.asname or name)
                importers.append((name, path))
                if imported_as != name:
                    entries.append(SymbolEntry(imported_as, path, "alias", (name,)))
                if module:
                    importers.append((f"{module}.{name}", path))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name)
                imported_as = str(alias.asname or name.split(".", 1)[0])
                importers.append((name, path))
                if imported_as != name:
                    entries.append(SymbolEntry(imported_as, path, "alias", (name,)))
    return entries, importers


def build_index(sources: Mapping[str, str], *, revision: str,
                journal: object | None = None) -> NavigationIndex | None:
    if not truth_enabled(SWITCH):
        return None
    entries: list[SymbolEntry] = []
    importers: list[tuple[str, str]] = []
    for path, source in sorted(sources.items()):
        file_entries, file_importers = _entry_for_file(str(path), str(source))
        entries.extend(file_entries)
        importers.extend(file_importers)
    index = NavigationIndex(str(revision), tuple(entries), tuple(importers))
    emit(journal, "gaps_navigation_indexed", producer=PRODUCER, enabled_value=True,
         revision=index.revision, symbol_count=len(index.entries),
         importer_count=len(index.importers))
    return index


def merge_index(index: NavigationIndex, sources: Mapping[str, str], *, revision: str,
                journal: object | None = None) -> NavigationIndex | None:
    """Re-index only changed visible files; no transaction or apply semantics."""
    if not truth_enabled(SWITCH):
        return None
    changed_paths = set(str(path) for path in sources)
    entries = [entry for entry in index.entries if entry.path not in changed_paths]
    importers = [row for row in index.importers if row[1] not in changed_paths]
    delta = build_index(sources, revision=str(revision), journal=None)
    if delta is not None:
        entries.extend(delta.entries)
        importers.extend(delta.importers)
    merged = NavigationIndex(str(revision), tuple(entries), tuple(importers))
    emit(journal, "gaps_navigation_incremental", producer=PRODUCER, enabled_value=True,
         revision=merged.revision, changed_file_count=len(changed_paths),
         symbol_count=len(merged.entries))
    return merged


def locate(index: NavigationIndex, query: str, *, path_hint: str = "",
           journal: object | None = None) -> NavigationResult:
    wanted = str(query).strip()
    matches = tuple(entry for entry in index.definitions(wanted)
                    if not path_hint or entry.path == path_hint)
    if not matches and path_hint:
        matches = index.definitions(wanted)
    callers = index.callers(wanted)
    ambiguous = len({entry.path for entry in matches}) > 1
    reason = "ambiguous_definition" if ambiguous else (
        "definition_found" if matches else "not_indexed")
    result = NavigationResult(wanted, matches[:MAX_RESULTS], callers, ambiguous, reason)
    emit(journal, "gaps_navigation_lookup", producer=PRODUCER, enabled_value=True,
         query=wanted, definition_count=len(result.definitions),
         caller_count=len(result.callers), ambiguous=result.ambiguous, reason=result.reason)
    return result


def render_navigation(result: NavigationResult | None) -> str:
    if result is None:
        return ""
    return "[Symbol and caller navigation]\n" + json.dumps(result.to_dict(), sort_keys=True,
                                                               separators=(",", ":"))


__all__ = [
    "NavigationIndex", "NavigationResult", "PRODUCER", "SWITCH", "SymbolEntry",
    "build_index", "locate", "merge_index", "render_navigation",
]
