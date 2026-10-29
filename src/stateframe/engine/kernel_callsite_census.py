"""Bounded, source-adjacent static call-site census for prepared requests.

This presentation-only adapter reuses catalog content handles, never runs
workspace code, and does not merge unresolved short names into relation groups.
"""
from __future__ import annotations

import ast
import base64
import builtins
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Literal

from kernel_context import token_estimate
from kernel_census_index_reuse import (
    acquire_index,
    index_source_bytes as census_index_source_bytes,
    reuse_enabled as census_index_reuse_enabled,
    structural_revision,
    validation_source_byte_limit as census_validation_source_byte_limit,
    validation_source_cap_mib as census_validation_source_cap_mib,
)
from kernel_repo_catalog import (
    StructuralIndexError,
    build_structural_index,
    scan_entry_set,
)
from kernel_source_presentation import strip_callsite_census
from kernel_measurement import profiled_function_span
import kernel_turn0_persistence as _turn0


SCHEMA = "stateframe-callsite-census/1"
ROUTE_ID = "R_shown_crossfile"
MODE_ENV = "STATEFRAME_CALLSITE_CENSUS"
EDIT_MODE_ENV = "STATEFRAME_CALLSITE_CENSUS_EDIT"
MAX_DISPLAYED_ROWS = 12
MAX_SERVING_TOKENS = 600
EXTRACTOR_VERSION = "python-bound-call-sites/2"
_CENSUS_PERSIST_SCHEMA = "callsite-census-structure/1"
MAX_CACHED_ANALYSIS_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_ERROR_MESSAGE_CHARS = 240
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|"
    r"authorization|credential|client[_-]?secret)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_PROVIDER_SECRET = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^/\s,;]+/)*[^/\s,;]+")
_HEADER = "[host source relations; static call sites of constructors used in the shown source]"
HEADER_TEXT = _HEADER
_END = "[/host source relations]"
_SOURCE_HEADER = re.compile(
    r"^\[(?P<label>Current|Working|Observed|Retained|Repair) source: "
    r"(?P<path>[^\]\n]+)\]\n"
)
_LEGACY_HEADER = re.compile(
    r"^CURRENT_SOURCE (?P<path>[^\s]+)(?: version=(?P<revision>[^\s]+))?\n"
)
_FORBIDDEN_PARTS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "grader", "graders",
    "gold", "private", "hidden", "hidden_tests", "backups", "tests",
})
_NON_PYTHON_CODE_SUFFIXES = frozenset({
    ".go", ".rs", ".ts", ".mts", ".cts", ".tsx", ".js", ".mjs", ".cjs", ".jsx",
    ".java", ".c", ".h", ".cc", ".cp", ".cpp", ".cxx", ".hh", ".hpp", ".hxx",
    ".cs", ".rb", ".php", ".vue",
})
_FACTORY_PREFIXES = ("create_", "make_", "build_", "new_")
_CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)


class CensusError(ValueError):
    """A census input cannot be safely admitted."""


@dataclass(frozen=True, slots=True)
class CensusPacket:
    schema: str
    workspace_namespace: str
    index_revision: str
    extractor_versions: tuple[str, ...]
    route_id: str
    decision_kind: str
    selected_source_handles: tuple[str, ...]
    source_span_set_digest: str
    coverage: Mapping[str, object]
    groups: tuple[Mapping[str, object], ...]
    render: Mapping[str, object]
    payload: str
    cache_key: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "workspace_namespace": self.workspace_namespace,
            "index_revision": self.index_revision,
            "extractor_versions": list(self.extractor_versions),
            "route_id": self.route_id,
            "decision_kind": self.decision_kind,
            "selected_source_handles": list(self.selected_source_handles),
            "source_span_set_digest": self.source_span_set_digest,
            "coverage": dict(self.coverage),
            "groups": [dict(group) for group in self.groups],
            "render": dict(self.render),
        }


@dataclass(frozen=True, slots=True)
class CensusDelivery:
    messages: tuple[dict[str, object], ...]
    status: str
    packet: CensusPacket | None = None
    header_path: str | None = None
    header_message_index: int | None = None
    payload_digest: str | None = None
    decision_kind: str = ""
    mode: str = "off"
    error_type: str | None = None
    placement: str = "after_source_header"
    source_facts: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _Source:
    path: str
    text: str
    start_line: int | None
    end_line: int | None
    window_ranges: tuple[tuple[int, int], ...] = ()
    window_content: str | None = None


@dataclass(frozen=True, slots=True)
class _Definition:
    identity: str
    display: str
    kind: Literal["class", "function"]


@dataclass(frozen=True, slots=True)
class _Binding:
    kind: Literal["module", "symbol"]
    identity: str
    display: str
    module_path: str | None = None
    definition: _Definition | None = None
    strategy: str = ""


@dataclass(slots=True)
class _Scope:
    kind: Literal["module", "class", "function", "lambda"]
    bindings: dict[str, _Binding]
    shadows: set[str]


@dataclass(frozen=True, slots=True)
class _Call:
    identity: str
    display: str
    resolution: str
    strategy: str
    owner: str
    path: str
    source_handle: str
    start_byte: int
    end_byte: int
    line: int
    column: int
    shown: bool


@dataclass(slots=True)
class _CensusStructure:
    revision: str
    scope_key: str
    parser_version: str
    parsed: dict[str, tuple[object, bytes, str]]
    trees: dict[str, ast.Module]
    definitions: dict[str, dict[str, _Definition]]
    calls_by_path: dict[str, tuple[_Call, ...]]
    unresolved_by_path: dict[str, int]
    partial_paths: set[str]
    skipped_paths: set[str]
    importers_by_path: dict[str, set[str]]


@dataclass(slots=True)
class _IndexReuseReport:
    status: str = "not_run"
    reason: str | None = "not_run"
    index_source_bytes_admitted: int = 0
    validation_cap_mib: int = 32
    limit_kind: str | None = None
    limit_observed: int | None = None
    limit_allowed: int | None = None


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_error_message(error: Exception) -> str:
    try:
        message = " ".join(str(error).split())
    except Exception:
        message = type(error).__name__
    message = _SECRET_ASSIGNMENT.sub(r"\1=[redacted]", message)
    message = _BEARER_SECRET.sub("Bearer [redacted]", message)
    message = _PROVIDER_SECRET.sub("[redacted]", message)
    message = _AWS_ACCESS_KEY.sub("[redacted]", message)
    message = re.sub(r"(?i)https?://[^\s,;]+", "[url]", message)
    message = _ABSOLUTE_PATH.sub("[path]", message)
    if len(message) > _MAX_ERROR_MESSAGE_CHARS:
        return message[:_MAX_ERROR_MESSAGE_CHARS - 3] + "..."
    return message


def _error_site(error: Exception) -> str | None:
    try:
        frames = traceback.extract_tb(error.__traceback__)
    except (AttributeError, TypeError, ValueError):
        return None
    if not frames:
        return None
    frame = frames[-1]
    function = re.sub(r"[^A-Za-z0-9_.<>-]", "_", frame.name)[:80] or "unknown"
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(frame.filename).name)[:120] or "unknown"
    return f"{function}@{filename}:{frame.lineno}"


def _safe_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if (not raw or path.is_absolute() or re.match(r"^[A-Za-z]:", raw)
            or ".." in path.parts
            or any(part.casefold() in _FORBIDDEN_PARTS or part.startswith(".") for part in path.parts)):
        raise CensusError("source path is outside the public workspace scope")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise CensusError("source path is empty")
    return normalized


def _span(metadata: Mapping[str, object]) -> tuple[int | None, int | None]:
    value = metadata.get("span")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        start, end = value[0], value[1]
        if type(start) is int and type(end) is int and start >= 1 and end >= start:
            return start, end
    start = metadata.get("start_line", metadata.get("line"))
    end = metadata.get("end_line", metadata.get("line_end", start))
    if type(start) is int and type(end) is int and start >= 1 and end >= start:
        return start, end
    return None, None


def _strip_census_presentation(content: str) -> str:
    """Remove census copies, including the slot inside a model source window."""
    clean = strip_callsite_census(content)
    if clean != content:
        return clean
    from kernel_request_source_window import (
        MODEL_WINDOW_MARKER,
        WINDOW_CONTENT_MARKER,
        parse_model_source_window,
    )

    if parse_model_source_window(content) is None:
        return content
    lines = content.splitlines(keepends=True)
    source_content_index = next((index for index, line in enumerate(lines)
                                 if line.rstrip("\r\n") == WINDOW_CONTENT_MARKER), None)
    if source_content_index is None:
        return content
    marker_index = next((index for index, line in enumerate(lines[:3])
                         if line.rstrip("\r\n") == MODEL_WINDOW_MARKER), None)
    if marker_index is None:
        return content
    header_index = next((index for index in range(marker_index + 2, source_content_index)
                         if lines[index].rstrip("\r\n") == _HEADER), None)
    if header_index is None:
        return content
    end_index = next((index for index in range(header_index + 1, source_content_index)
                      if lines[index].rstrip("\r\n") == _END), None)
    if end_index is None:
        return content
    del lines[header_index:end_index + 1]
    return "".join(lines)


def _source_block(content: object) -> tuple[_Source, int] | None:
    if not isinstance(content, str):
        return None
    clean = _strip_census_presentation(content)
    match = _SOURCE_HEADER.match(clean)
    metadata: dict[str, object] = {}
    if match:
        raw_path, body, header_end = match.group("path"), clean[match.end():], match.end()
    else:
        legacy = _LEGACY_HEADER.match(clean)
        if legacy is None:
            return None
        raw_path, body, header_end = legacy.group("path"), clean[legacy.end():], legacy.end()
    try:
        path = _safe_path(raw_path)
    except CensusError:
        return None
    if body.startswith("[Model source window]\n") or body.startswith("[Model source window]\r\n"):
        from kernel_request_source_window import (
            MODEL_WINDOW_MARKER,
            parse_model_source_window,
            verify_model_source_window,
        )

        metadata = parse_model_source_window(clean)
        if (metadata is None
                or not verify_model_source_window(clean, path=path)
                or metadata.get("path") != path):
            return None
        ranges = metadata.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            return None
        spans: list[tuple[int, int]] = []
        for row in ranges:
            if not isinstance(row, Mapping):
                return None
            start, end = row.get("start_line"), row.get("end_line")
            if type(start) is not int or type(end) is not int or start < 1 or end < start:
                return None
            spans.append((start, end))
        marker_at = clean.find(MODEL_WINDOW_MARKER, header_end)
        line_break = "\r\n" if clean[marker_at:].startswith(MODEL_WINDOW_MARKER + "\r\n") else "\n"
        metadata_end = clean.find(
            line_break,
            marker_at + len(MODEL_WINDOW_MARKER) + len(line_break),
        )
        if metadata_end < 0:
            return None
        return _Source(path, "", None, None, tuple(spans), clean), metadata_end + len(line_break)
    if body.startswith("[Source metadata]\n"):
        marker = f"\n[Source content: {path}]\n"
        end_marker = f"\n[End source content: {path}]"
        if marker not in body or not body.endswith(end_marker):
            return None
        encoded, body = body[len("[Source metadata]\n"):].split(marker, 1)
        try:
            parsed = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, Mapping):
            return None
        metadata.update(parsed)
        body = body[:-len(end_marker)]
        if body.endswith("\n"):
            body = body[:-1]
    elif body.startswith("Coverage: "):
        _, separator, body = body.partition("\n")
        if not separator:
            return None
    start, end = _span(metadata)
    return _Source(path, body, start, end), header_end


# Native read results arrive as role=tool "[read page] path=P revision=sha256:H lines=S-E/T ..."
# followed by one status line and then the delivered source lines. Live benchmark r2AR (batch
# 1028): census only parsed "[... source: P]" user blocks, saw no source on any turn and
# reported route_off 13/13, so the model never received the call-site list.
_READ_PAGE_HEADER = re.compile(
    r"^\[read page\] path=(?P<path>\S+) revision=(?P<revision>sha256:[0-9a-f]{64}) "
    r"lines=(?P<start>\d+)-(?P<end>\d+)/\d+[^\n]*\n[^\n]*\n"
)


def _read_page_block(content: object) -> tuple[str, str, int, int, str] | None:
    """Parse one host read page into (path, revision, start, end, body); None if not one."""
    if not isinstance(content, str) or not content.startswith("[read page] "):
        return None
    match = _READ_PAGE_HEADER.match(content)
    if match is None:
        return None
    try:
        path = _safe_path(match.group("path"))
    except CensusError:
        return None
    start, end = int(match.group("start")), int(match.group("end"))
    if start < 1 or end < start:
        return None
    return path, match.group("revision"), start, end, content[match.end():]


def _verified_read_page_span(index: object, page: tuple[str, str, int, int, str]
                             ) -> tuple[int, int, str] | None:
    """Accept a read page only when it matches the CURRENT indexed bytes exactly.

    The page revision must equal the indexed content digest (stale reads are dropped),
    and the delivered body must start with exactly those source lines (host suffixes
    such as read notices may follow).
    """
    path, revision, start, end, body = page
    try:
        handle, raw = index.read_content(path)
    except (LookupError, UnicodeError, OSError, StructuralIndexError):
        return None
    if revision != handle.content_digest:
        return None
    selected = _line_slice(raw, start, end)
    if selected is None:
        return None
    try:
        expected = selected.decode(handle.encoding, errors="strict")
    except (LookupError, UnicodeError):
        return None
    # The last delivered line may lack its newline (end of file or a page boundary).
    if not body.startswith(expected.rstrip("\r\n")):
        return None
    return start, end, handle.content_id


def _line_slice(body: bytes, start: int, end: int) -> bytes | None:
    lines = body.splitlines(keepends=True)
    if start < 1 or end < start or end > len(lines):
        return None
    return b"".join(lines[start - 1:end])


def _same_shown_source(indexed: bytes, shown: bytes) -> bool:
    if indexed == shown:
        return True
    if shown.endswith((b"\n", b"\r")):
        return False
    if indexed.endswith(b"\r\n"):
        return indexed[:-2] == shown
    if indexed.endswith((b"\n", b"\r")):
        return indexed[:-1] == shown
    return False


def _exact_source_span(body: bytes, shown: bytes) -> tuple[int, int] | None:
    """Recover a unique line span when a source label carries an exact excerpt."""
    if not shown:
        return None
    lines = body.splitlines(keepends=True)
    shown_lines = shown.splitlines(keepends=True)
    width = len(shown_lines)
    if not width or width > len(lines):
        return None
    matches: list[tuple[int, int]] = []
    for start in range(len(lines) - width + 1):
        candidate = b"".join(lines[start:start + width])
        if _same_shown_source(candidate, shown):
            matches.append((start + 1, start + width))
            if len(matches) > 1:
                return None
    return matches[0] if matches else None


def _module_display(path: str) -> str:
    parts = list(PurePosixPath(path).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = PurePosixPath(path).stem
    return ".".join(parts)


def _definition(path: str, name: str, kind: Literal["class", "function"]) -> _Definition:
    return _Definition(f"repository:{path}::{name}", f"{_module_display(path)}.{name}", kind)


def _module_names(paths: Sequence[str]) -> dict[str, str | None]:
    candidates: dict[str, set[str]] = {}
    for path in paths:
        pure = PurePosixPath(path)
        if pure.suffix != ".py":
            continue
        parts = list(pure.parts)
        if parts[-1] == "__init__.py":
            parts.pop()
        else:
            parts[-1] = pure.stem
        if parts:
            candidates.setdefault(".".join(parts), set()).add(path)
    return {name: next(iter(rows)) if len(rows) == 1 else None for name, rows in candidates.items()}


def _module_candidates(paths: Sequence[str]) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    for path in paths:
        pure = PurePosixPath(path)
        if pure.suffix != ".py":
            continue
        parts = list(pure.parts)
        if parts[-1] == "__init__.py":
            parts.pop()
        else:
            parts[-1] = pure.stem
        if parts:
            candidates.setdefault(".".join(parts), set()).add(path)
    return candidates


def _import_target_paths(
    tree: ast.Module,
    path: str,
    module_candidates: Mapping[str, set[str]],
) -> set[str]:
    """Map syntax-only imports to every possibly affected repository module."""
    targets: set[str] = set()

    def include(module_name: str | None) -> None:
        if not module_name:
            return
        targets.update(
            candidate
            for name, paths in module_candidates.items()
            if name == module_name or name.startswith(module_name + ".")
            for candidate in paths
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name if alias.asname else alias.name.split(".", 1)[0]
                include(imported)
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(path, node)
            include(base)
            if base:
                for alias in node.names:
                    if alias.name != "*":
                        include(f"{base}.{alias.name}")
    return targets


def _build_importer_map(
    trees: Mapping[str, ast.Module],
    module_candidates: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    importers: dict[str, set[str]] = {}
    for source_path, tree in trees.items():
        for target_path in _import_target_paths(tree, source_path, module_candidates):
            if target_path != source_path:
                importers.setdefault(target_path, set()).add(source_path)
    return importers


def _definitions(tree: ast.Module, path: str) -> dict[str, _Definition]:
    found: dict[str, _Definition] = {}

    def walk(statements: Sequence[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in statements:
            if isinstance(node, ast.ClassDef):
                name = ".".join((*prefix, node.name))
                found[name] = _definition(path, name, "class")
                walk(node.body, (*prefix, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not prefix:
                found[node.name] = _definition(path, node.name, "function")
    walk(tree.body)
    return found


def _import_base(path: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module or None
    parts = list(PurePosixPath(path).parts[:-1])
    level = node.level - 1
    if level > len(parts):
        return None
    if level:
        parts = parts[:-level]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts) or None


def _import_binding(node: ast.stmt, alias: ast.alias, path: str,
                    modules: Mapping[str, str | None],
                    definitions: Mapping[str, Mapping[str, _Definition]]) -> tuple[str, _Binding] | None:
    if isinstance(node, ast.Import):
        local = alias.asname or alias.name.split(".", 1)[0]
        imported = alias.name if alias.asname else alias.name.split(".", 1)[0]
        internal = modules.get(imported)
        if internal:
            return local, _Binding("module", f"repository-module:{internal}",
                                   _module_display(internal), internal, strategy="repository_import")
        if any(name == imported or name.startswith(imported + ".") for name in modules):
            return None
        return local, _Binding("module", f"external:{imported}", imported,
                               strategy="external_import_alias")
    if not isinstance(node, ast.ImportFrom) or alias.name == "*":
        return None
    base = _import_base(path, node)
    if not base:
        return None
    local = alias.asname or alias.name
    child = modules.get(f"{base}.{alias.name}")
    if child:
        return local, _Binding("module", f"repository-module:{child}",
                               _module_display(child), child, strategy="repository_import_alias")
    base_path = modules.get(base)
    if base_path:
        definition = definitions.get(base_path, {}).get(alias.name)
        if definition is None:
            return None
        return local, _Binding("symbol", definition.identity, definition.display,
                               definition=definition, strategy="repository_import_alias")
    if any(name == base or name.startswith(base + ".") for name in modules):
        return None
    canonical = f"{base}.{alias.name}"
    imported = _Definition(f"external:{canonical}", canonical,
                           "class" if alias.name[:1].isupper() else "function")
    return local, _Binding("symbol", imported.identity, imported.display,
                           definition=imported, strategy="external_import_alias")


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()

    def visit(child: ast.AST) -> None:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        for nested in ast.iter_child_nodes(child):
            visit(nested)
    visit(node)
    return names


def _scope(statements: Sequence[ast.stmt], kind: Literal["module", "class", "function"],
           path: str, owner: tuple[str, ...], modules: Mapping[str, str | None],
           definitions: Mapping[str, Mapping[str, _Definition]],
           args: ast.arguments | None = None) -> _Scope:
    bindings: dict[str, _Binding] = {}
    shadows: set[str] = set()

    def add(name: str, binding: _Binding) -> None:
        previous = bindings.get(name)
        if previous is not None and previous.identity != binding.identity:
            bindings.pop(name, None)
            shadows.add(name)
        elif name not in shadows:
            bindings[name] = binding

    def scan(rows: Sequence[ast.stmt]) -> None:
        for node in rows:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith(_FACTORY_PREFIXES):
                    target = _definition(path, ".".join((*owner, node.name)), "function")
                    bindings[node.name] = _Binding(
                        "symbol", target.identity, target.display,
                        definition=target, strategy="repository_local_definition",
                    )
                    shadows.discard(node.name)
                else:
                    bindings.pop(node.name, None)
                    shadows.add(node.name)
                continue
            if isinstance(node, ast.ClassDef):
                target = _definition(path, ".".join((*owner, node.name)), "class")
                bindings[node.name] = _Binding(
                    "symbol", target.identity, target.display,
                    definition=target, strategy="repository_local_definition",
                )
                shadows.discard(node.name)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names or ():
                    resolved = _import_binding(node, alias, path, modules, definitions)
                    if resolved is None:
                        shadows.add(alias.asname or alias.name.split(".", 1)[0])
                    else:
                        add(*resolved)
                continue
            shadows.update(_assigned_names(node))
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                shadows.update(node.names)
            if isinstance(node, _CONTROL):
                for key in ("body", "orelse", "finalbody"):
                    children = getattr(node, key, ())
                    if isinstance(children, Sequence):
                        scan(children)
                for handler in getattr(node, "handlers", ()) or ():
                    if handler.name:
                        shadows.add(handler.name)
                    scan(handler.body)
                for case in getattr(node, "cases", ()) or ():
                    scan(case.body)

    scan(statements)
    if args is not None:
        shadows.update(arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs))
        if args.vararg is not None:
            shadows.add(args.vararg.arg)
        if args.kwarg is not None:
            shadows.add(args.kwarg.arg)
    for name in shadows:
        bindings.pop(name, None)
    return _Scope(kind, bindings, shadows)


def _parts(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _parts(node.value)
        return (*prefix, node.attr) if prefix else None
    return None


def _constructor_name(name: str) -> bool:
    return bool(name) and (name[0].isupper() or name.startswith(_FACTORY_PREFIXES))


def _bound_name(name: str, scopes: Sequence[_Scope]) -> _Binding | None:
    for index in range(len(scopes) - 1, -1, -1):
        scope = scopes[index]
        if scope.kind == "class" and any(item.kind in {"function", "lambda"}
                                          for item in scopes[index + 1:]):
            continue
        if name in scope.shadows:
            return None
        if name in scope.bindings:
            return scope.bindings[name]
    return None


def _resolve(parts: tuple[str, ...], scopes: Sequence[_Scope],
             definitions: Mapping[str, Mapping[str, _Definition]]) -> tuple[str, str, str, str] | None:
    if not parts or parts[0] in vars(builtins):
        return None
    binding = _bound_name(parts[0], scopes)
    if binding is None:
        return None
    attrs = parts[1:]
    if binding.kind == "module":
        if not attrs:
            return None
        if binding.module_path is None:
            if not _constructor_name(attrs[-1]):
                return None
            canonical = binding.identity.removeprefix("external:") + "." + ".".join(attrs)
            return "external:" + canonical, canonical, "statically_bound_external_import", binding.strategy
        definition = definitions.get(binding.module_path, {}).get(".".join(attrs))
        if definition is None or not _constructor_name(attrs[-1]):
            return None
        return definition.identity, definition.display, "resolved_repository_symbol", binding.strategy
    definition = binding.definition
    if definition is None:
        return None
    if attrs:
        if definition.identity.startswith("external:") and _constructor_name(attrs[-1]):
            canonical = definition.identity.removeprefix("external:") + "." + ".".join(attrs)
            return "external:" + canonical, canonical, "statically_bound_external_import", binding.strategy
        return None
    tail = definition.display.rsplit(".", 1)[-1]
    if definition.kind != "class" and not tail.startswith(_FACTORY_PREFIXES):
        return None
    resolution = ("statically_bound_external_import" if definition.identity.startswith("external:")
                  else "resolved_repository_symbol")
    return definition.identity, definition.display, resolution, binding.strategy


class _Collector(ast.NodeVisitor):
    def __init__(self, path: str, handle: str, source: str, tree: ast.Module,
                 module_scope: _Scope, modules: Mapping[str, str | None],
                 definitions: Mapping[str, Mapping[str, _Definition]],
                 shown: Sequence[tuple[int, int]]) -> None:
        self.path, self.handle, self.source, self.tree = path, handle, source, tree
        self.scopes = [module_scope]
        self.modules, self.definitions = modules, definitions
        self.shown = tuple(shown)
        self.owner: list[str] = []
        self.calls: list[_Call] = []
        self.unresolved = 0
        self.offsets = [0]
        for line in source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line.encode("utf-8")))

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in (*node.decorator_list, *node.args.defaults,
                           *(value for value in node.args.kw_defaults if value is not None)):
            self.visit(expression)
        self.owner.append(node.name)
        current = _scope(node.body, "function", self.path, tuple(self.owner[:-1]),
                         self.modules, self.definitions, node.args)
        self.scopes.append(current)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()
        self.owner.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *(item.value for item in node.keywords)):
            self.visit(expression)
        self.owner.append(node.name)
        current = _scope(node.body, "class", self.path, tuple(self.owner[:-1]),
                         self.modules, self.definitions)
        self.scopes.append(current)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()
        self.owner.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        names = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        if node.args.vararg is not None:
            names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            names.add(node.args.kwarg.arg)
        self.scopes.append(_Scope("lambda", {}, names))
        self.visit(node.body)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        parts = _parts(node.func)
        if parts:
            resolved = _resolve(parts, self.scopes, self.definitions)
            if resolved is None:
                if (_constructor_name(parts[-1])
                        and parts[-1] not in vars(builtins)):
                    self.unresolved += 1
            else:
                identity, display, resolution, strategy = resolved
                line = int(getattr(node, "lineno", 1))
                end_line = int(getattr(node, "end_lineno", line))
                column = int(getattr(node, "col_offset", 0))
                end_column = int(getattr(node, "end_col_offset", column))
                start_byte = self.offsets[min(line - 1, len(self.offsets) - 1)] + column
                end_byte = self.offsets[min(end_line - 1, len(self.offsets) - 1)] + end_column
                self.calls.append(_Call(
                    identity, display, resolution, strategy,
                    ".".join(self.owner) or "<module>", self.path, self.handle,
                    start_byte, end_byte, line, column,
                    any(start <= line <= end for start, end in self.shown),
                ))
        self.generic_visit(node)


# Catalog outlines are language-generic; this relation extractor is not. Its binding
# and shadowing rules are built on Python's ast.Module/import semantics, so other code
# languages remain explicitly partial until they have a parser-backed resolver.
def _source_paths(index: object) -> tuple[dict[str, tuple[object, bytes, str]], int, int]:
    parsed: dict[str, tuple[object, bytes, str]] = {}
    partial = skipped = 0
    for module in getattr(index, "modules", ()):
        path = str(module.path)
        if _excluded(path):
            skipped += 1
            continue
        if not path.endswith(".py"):
            skipped += 1
            if PurePosixPath(path).suffix.casefold() in _NON_PYTHON_CODE_SUFFIXES:
                partial += 1
            continue
        try:
            handle, raw = index.read_content(path)
            source = raw.decode(handle.encoding, errors="strict")
            ast.parse(source, filename=path)
        except (LookupError, UnicodeError, OSError, StructuralIndexError, SyntaxError, ValueError):
            partial += 1
            continue
        parsed[path] = (handle, raw, source)
        if bool(getattr(module, "degraded", False)):
            partial += 1
    return parsed, partial, skipped


def _runtime_task_scope(runtime: object) -> str:
    identity_fields = (
        "tenant_id", "workspace_id", "task_bundle_id", "bundle_id", "task_id",
        "assignment_id", "run_id", "id",
    )
    owner = getattr(runtime, "owner", None)
    task_row = getattr(owner, "task_row", None)
    identities: dict[str, str] = {}
    for label, source in (("runtime", runtime), ("owner", owner), ("task", task_row)):
        if source is None:
            continue
        for name in identity_fields:
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if isinstance(value, (str, int)) and str(value).strip():
                identities[f"{label}.{name}"] = str(value)
    return _digest({"runtime_instance": id(runtime), "identities": identities})


def _index_scope_key(runtime: object, root: Path) -> str:
    parser_version = f"{EXTRACTOR_VERSION}:ast-{sys.version_info.major}.{sys.version_info.minor}"
    return _digest({
        "root": root.resolve().as_posix(),
        "runtime_task_scope": _runtime_task_scope(runtime),
        "parser_version": parser_version,
    })


def _analysis_source_size(index: object) -> int:
    return sum(
        int(getattr(getattr(module, "content", None), "byte_length", 0))
        for module in getattr(index, "modules", ())
        if str(getattr(module, "path", "")).endswith(".py")
        and not _excluded(str(getattr(module, "path", "")))
    )


def _parse_census_source(
    index: object,
    path: str,
) -> tuple[tuple[object, bytes, str] | None, ast.Module | None,
             dict[str, _Definition] | None, bool, bool]:
    """Parse one selected Python source without importing or executing it."""
    module = index.module(path)
    if module is None:
        return None, None, None, True, False
    if _excluded(path):
        return None, None, None, False, True
    if not path.endswith(".py"):
        partial = PurePosixPath(path).suffix.casefold() in _NON_PYTHON_CODE_SUFFIXES
        return None, None, None, partial, True
    try:
        handle, raw = index.read_content(path)
        source = raw.decode(handle.encoding, errors="strict")
        tree = ast.parse(source, filename=path)
    except (LookupError, UnicodeError, OSError, StructuralIndexError, SyntaxError, ValueError):
        return None, None, None, True, False
    return (handle, raw, source), tree, _definitions(tree, path), bool(module.degraded), False


def _collect_file_calls(
    path: str,
    parsed: tuple[object, bytes, str],
    tree: ast.Module,
    modules: Mapping[str, str | None],
    definitions: Mapping[str, Mapping[str, _Definition]],
) -> tuple[tuple[_Call, ...], int]:
    handle, _raw, source = parsed
    module_scope = _scope(tree.body, "module", path, (), modules, definitions)
    collector = _Collector(
        tree=tree, path=path, handle=handle.content_id, source=source,
        module_scope=module_scope, modules=modules, definitions=definitions, shown=(),
    )
    collector.visit(tree)
    return tuple(collector.calls), collector.unresolved


@profiled_function_span("census_structure_build")
def _build_census_structure(
    index: object,
    *,
    scope_key: str,
    parser_version: str,
) -> _CensusStructure:
    parsed, _partial_count, _skipped_count = _source_paths(index)
    trees: dict[str, ast.Module] = {}
    definitions: dict[str, dict[str, _Definition]] = {}
    partial_paths: set[str] = set()
    skipped_paths: set[str] = set()
    for module in getattr(index, "modules", ()):
        path = str(module.path)
        if _excluded(path):
            skipped_paths.add(path)
            continue
        if not path.endswith(".py"):
            skipped_paths.add(path)
            if PurePosixPath(path).suffix.casefold() in _NON_PYTHON_CODE_SUFFIXES:
                partial_paths.add(path)
            continue
        if path not in parsed:
            partial_paths.add(path)
            continue
        if bool(getattr(module, "degraded", False)):
            partial_paths.add(path)
        try:
            tree = ast.parse(parsed[path][2], filename=path)
        except (SyntaxError, ValueError):
            partial_paths.add(path)
            parsed.pop(path, None)
            continue
        trees[path] = tree
        definitions[path] = _definitions(tree, path)

    parsed = {str(module.path): parsed[str(module.path)]
              for module in getattr(index, "modules", ()) if str(module.path) in parsed}
    trees = {path: trees[path] for path in parsed if path in trees}
    definitions = {path: definitions[path] for path in parsed if path in definitions}
    modules = _module_names(tuple(parsed))
    calls_by_path: dict[str, tuple[_Call, ...]] = {}
    unresolved_by_path: dict[str, int] = {}
    for path in parsed:
        tree = trees.get(path)
        if tree is None:
            continue
        calls_by_path[path], unresolved_by_path[path] = _collect_file_calls(
            path, parsed[path], tree, modules, definitions,
        )
    importers = _build_importer_map(trees, _module_candidates(tuple(parsed)))
    return _CensusStructure(
        structural_revision(index, scope_key=scope_key, parser_version=parser_version),
        scope_key, parser_version, parsed, trees, definitions, calls_by_path,
        unresolved_by_path, partial_paths, skipped_paths, importers,
    )


def _runtime_persistence_task_id(runtime: object) -> str:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", None)
    if isinstance(row, Mapping):
        value = row.get("task_id")
        if isinstance(value, str) and value.strip():
            return value
    value = getattr(owner, "task_id", "")
    return str(value or "")


def _census_persistence_context(
    index: object,
    runtime: object,
    *,
    parser_version: str,
    journal: object | None,
) -> tuple[_turn0.CacheContext | None, tuple[str, ...], tuple[tuple[str, str], ...]]:
    if not _turn0.enabled():
        return None, (), ()
    raw_entries = getattr(index, "scan_entries", None)
    if not bool(getattr(index, "scan_stable", False)) or raw_entries is None:
        _turn0.emit_cache_outcome(journal, name="census_index", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="census_index", reason="scan_unstable")
        return None, (), ()
    entries = tuple((str(path), str(kind)) for path, kind in raw_entries)
    walked_paths = tuple(path for path, kind in entries if kind == "source")
    module_paths = tuple(str(module.path) for module in getattr(index, "modules", ()))
    if _analysis_source_size(index) > MAX_CACHED_ANALYSIS_SOURCE_BYTES:
        _turn0.emit_cache_outcome(journal, name="census_index", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="census_index", reason="limits")
        return None, walked_paths, entries
    try:
        module_version = _turn0.implementation_version(
            f"{_CENSUS_PERSIST_SCHEMA}:{EXTRACTOR_VERSION}:{parser_version}",
            Path(__file__),
        )
        namespace = getattr(index, "namespace", None)
        namespace_value = namespace.to_dict() if callable(getattr(namespace, "to_dict", None)) else {}
        extra_key = {
            "namespace": namespace_value,
            "parser_version": parser_version,
            "selected_module_paths": list(module_paths),
            "scan_entries": [list(entry) for entry in entries],
            "index_revision": structural_revision(
                index, scope_key="", parser_version=parser_version,
            ),
        }
        context, reason = _turn0.make_context(
            Path(getattr(index, "root")),
            task_id=_runtime_persistence_task_id(runtime),
            name="census_index",
            module_version=module_version or "",
            walked_paths=walked_paths,
            extra_key=extra_key,
        )
    except Exception:
        context, reason = None, "context_unavailable"
    if context is None:
        _turn0.emit_cache_outcome(journal, name="census_index", outcome="miss")
        _turn0.emit_cache_bypass(
            journal, name="census_index", reason=reason or "context_unavailable",
        )
    return context, walked_paths, entries


def _ast_value_to_payload(value: object) -> object:
    if isinstance(value, ast.AST):
        return {
            "$node": type(value).__name__,
            "fields": {
                name: _ast_value_to_payload(getattr(value, name))
                for name in value._fields
            },
            "attributes": {
                name: _ast_value_to_payload(getattr(value, name, None))
                for name in value._attributes
            },
        }
    if isinstance(value, list):
        return [_ast_value_to_payload(item) for item in value]
    if isinstance(value, tuple):
        return {"$tuple": [_ast_value_to_payload(item) for item in value]}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("AST contains a non-finite complex constant")
        return {"$complex": [value.real, value.imag]}
    if value is Ellipsis:
        return {"$ellipsis": True}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("AST contains an unsupported JSON value")


def _ast_value_from_payload(value: object, *, depth: int = 0) -> object:
    if depth > 256:
        raise ValueError("cached AST exceeds the depth limit")
    if isinstance(value, list):
        return [_ast_value_from_payload(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if set(value) == {"$tuple"} and isinstance(value["$tuple"], list):
            return tuple(_ast_value_from_payload(item, depth=depth + 1) for item in value["$tuple"])
        if set(value) == {"$bytes"} and isinstance(value["$bytes"], str):
            encoded = value["$bytes"].encode("ascii")
            decoded = base64.b64decode(encoded, validate=True)
            if base64.b64encode(decoded) != encoded:
                raise ValueError("cached AST bytes are not canonical")
            return decoded
        if set(value) == {"$complex"} and isinstance(value["$complex"], list):
            parts = value["$complex"]
            if (len(parts) != 2
                    or any(type(part) not in {int, float} or not math.isfinite(part) for part in parts)):
                raise ValueError("cached AST complex constant is malformed")
            return complex(parts[0], parts[1])
        if set(value) == {"$ellipsis"} and value["$ellipsis"] is True:
            return Ellipsis
        if set(value) != {"$node", "fields", "attributes"}:
            raise ValueError("cached AST value is malformed")
        name = value["$node"]
        fields = value["fields"]
        attributes = value["attributes"]
        node_type = getattr(ast, name, None) if isinstance(name, str) else None
        if (not isinstance(node_type, type) or node_type is ast.AST
                or not issubclass(node_type, ast.AST)
                or not isinstance(fields, Mapping) or not isinstance(attributes, Mapping)
                or set(fields) != set(node_type._fields)
                or set(attributes) != set(node_type._attributes)):
            raise ValueError("cached AST node is malformed")
        decoded_fields = {
            key: _ast_value_from_payload(item, depth=depth + 1)
            for key, item in fields.items()
        }
        decoded_attributes = {
            key: _ast_value_from_payload(item, depth=depth + 1)
            for key, item in attributes.items()
        }
        node = node_type(**decoded_fields)
        for key, item in decoded_attributes.items():
            setattr(node, key, item)
        return node
    if (value is None or isinstance(value, (bool, int, str))
            or (isinstance(value, float) and math.isfinite(value))):
        return value
    raise ValueError("cached AST scalar is malformed")


def _census_call_payload(call: _Call) -> dict[str, object]:
    return {
        "identity": call.identity,
        "display": call.display,
        "resolution": call.resolution,
        "strategy": call.strategy,
        "owner": call.owner,
        "path": call.path,
        "source_handle": call.source_handle,
        "start_byte": call.start_byte,
        "end_byte": call.end_byte,
        "line": call.line,
        "column": call.column,
        "shown": call.shown,
    }


def _census_call_from_payload(value: object, *, paths: set[str]) -> _Call:
    fields = {
        "identity", "display", "resolution", "strategy", "owner", "path",
        "source_handle", "start_byte", "end_byte", "line", "column", "shown",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("cached call row is malformed")
    text_fields = (
        "identity", "display", "resolution", "strategy", "owner", "path", "source_handle",
    )
    if (any(not isinstance(value[field], str) for field in text_fields)
            or value["path"] not in paths
            or any(type(value[field]) is not int for field in ("start_byte", "end_byte", "line", "column"))
            or type(value["shown"]) is not bool
            or value["start_byte"] < 0 or value["end_byte"] < value["start_byte"]
            or value["line"] < 1 or value["column"] < 0):
        raise ValueError("cached call fields are malformed")
    return _Call(**{field: value[field] for field in fields})


def _census_structure_payload(index: object, structure: _CensusStructure, *,
                              parser_version: str) -> dict[str, object]:
    module_paths = tuple(str(module.path) for module in getattr(index, "modules", ()))
    modules = {str(module.path): module for module in getattr(index, "modules", ())}
    parsed_paths = tuple(path for path in module_paths if path in structure.parsed)
    if set(parsed_paths) != set(structure.parsed):
        raise ValueError("coverage check parsed paths do not match the structural index")
    source_digests = {
        path: str(modules[path].content.content_digest) for path in parsed_paths
    }
    return {
        "schema": _CENSUS_PERSIST_SCHEMA,
        "parser_version": parser_version,
        "index_revision": structural_revision(index, scope_key="", parser_version=parser_version),
        "module_paths": list(module_paths),
        "parsed_paths": list(parsed_paths),
        "source_digests": source_digests,
        "trees": {path: _ast_value_to_payload(structure.trees[path]) for path in parsed_paths},
        "definitions": {
            path: {
                name: {
                    "identity": definition.identity,
                    "display": definition.display,
                    "kind": definition.kind,
                }
                for name, definition in structure.definitions[path].items()
            }
            for path in parsed_paths
        },
        "calls_by_path": {
            path: [_census_call_payload(call) for call in structure.calls_by_path[path]]
            for path in parsed_paths
        },
        "unresolved_by_path": {
            path: structure.unresolved_by_path[path] for path in parsed_paths
        },
        "partial_paths": sorted(structure.partial_paths),
        "skipped_paths": sorted(structure.skipped_paths),
        "importers_by_path": {
            path: sorted(sources) for path, sources in structure.importers_by_path.items()
        },
    }


def _census_structure_from_payload(
    index: object,
    value: object,
    *,
    scope_key: str,
    parser_version: str,
) -> _CensusStructure:
    fields = {
        "schema", "parser_version", "index_revision", "module_paths", "parsed_paths",
        "source_digests", "trees", "definitions", "calls_by_path", "unresolved_by_path",
        "partial_paths", "skipped_paths", "importers_by_path",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("cached coverage check structure is malformed")
    module_paths = tuple(str(module.path) for module in getattr(index, "modules", ()))
    module_set = set(module_paths)
    if (value["schema"] != _CENSUS_PERSIST_SCHEMA
            or value["parser_version"] != parser_version
            or value["index_revision"] != structural_revision(
                index, scope_key="", parser_version=parser_version,
            )
            or value["module_paths"] != list(module_paths)):
        raise ValueError("cached coverage check structure belongs to another index")
    raw_parsed = value["parsed_paths"]
    if (not isinstance(raw_parsed, list)
            or any(not isinstance(path, str) or path not in module_set for path in raw_parsed)
            or len(set(raw_parsed)) != len(raw_parsed)
            or any(not path.endswith(".py") or _excluded(path) for path in raw_parsed)):
        raise ValueError("cached coverage check parsed path set is malformed")
    parsed_set = set(raw_parsed)
    parsed_paths = tuple(path for path in module_paths if path in parsed_set)
    if raw_parsed != list(parsed_paths):
        raise ValueError("cached coverage check parsed paths are not canonical")
    trees_value = value["trees"]
    definitions_value = value["definitions"]
    calls_value = value["calls_by_path"]
    unresolved_value = value["unresolved_by_path"]
    digests_value = value["source_digests"]
    if any(not isinstance(item, Mapping) for item in (
            trees_value, definitions_value, calls_value, unresolved_value, digests_value)):
        raise ValueError("cached coverage check maps are malformed")
    expected_map_keys = set(parsed_paths)
    if any(set(item) != expected_map_keys for item in (
            trees_value, definitions_value, calls_value, unresolved_value, digests_value)):
        raise ValueError("cached coverage check maps do not match parsed paths")
    modules = {str(module.path): module for module in getattr(index, "modules", ())}
    parsed: dict[str, tuple[object, bytes, str]] = {}
    trees: dict[str, ast.Module] = {}
    definitions: dict[str, dict[str, _Definition]] = {}
    calls_by_path: dict[str, tuple[_Call, ...]] = {}
    unresolved_by_path: dict[str, int] = {}
    for path in parsed_paths:
        expected_digest = modules[path].content.content_digest
        if digests_value[path] != expected_digest:
            raise ValueError("cached coverage check source digest is stale")
        handle, raw = index.read_content(path)
        if handle.content_digest != expected_digest:
            raise ValueError("cached coverage check content handle is stale")
        source = raw.decode(handle.encoding, errors="strict")
        tree = _ast_value_from_payload(trees_value[path])
        if not isinstance(tree, ast.Module):
            raise ValueError("cached coverage check tree is not a module")
        raw_definitions = definitions_value[path]
        if not isinstance(raw_definitions, Mapping):
            raise ValueError("cached coverage check definition map is malformed")
        found: dict[str, _Definition] = {}
        for name, item in raw_definitions.items():
            if (not isinstance(name, str) or not name or not isinstance(item, Mapping)
                    or set(item) != {"identity", "display", "kind"}
                    or any(not isinstance(item[field], str) for field in ("identity", "display", "kind"))
                    or item["kind"] not in {"class", "function"}):
                raise ValueError("cached coverage check definition row is malformed")
            found[name] = _Definition(item["identity"], item["display"], item["kind"])
        raw_calls = calls_value[path]
        if not isinstance(raw_calls, list):
            raise ValueError("cached coverage check call list is malformed")
        calls = tuple(_census_call_from_payload(call, paths=module_set) for call in raw_calls)
        unresolved = unresolved_value[path]
        if type(unresolved) is not int or unresolved < 0:
            raise ValueError("cached coverage check unresolved count is malformed")
        parsed[path] = (handle, raw, source)
        trees[path] = tree
        definitions[path] = found
        calls_by_path[path] = calls
        unresolved_by_path[path] = unresolved
    partial_paths = value["partial_paths"]
    skipped_paths = value["skipped_paths"]
    if (not isinstance(partial_paths, list) or not isinstance(skipped_paths, list)
            or any(not isinstance(path, str) or path not in module_set
                   for path in (*partial_paths, *skipped_paths))
            or partial_paths != sorted(set(partial_paths))
            or skipped_paths != sorted(set(skipped_paths))):
        raise ValueError("cached coverage check coverage paths are malformed")
    raw_importers = value["importers_by_path"]
    if not isinstance(raw_importers, Mapping):
        raise ValueError("cached coverage check importer map is malformed")
    importers: dict[str, set[str]] = {}
    for target, sources in raw_importers.items():
        if (not isinstance(target, str) or target not in expected_map_keys
                or not isinstance(sources, list)
                or any(not isinstance(path, str) or path not in expected_map_keys for path in sources)
                or sources != sorted(set(sources))):
            raise ValueError("cached coverage check importer row is malformed")
        importers[target] = set(sources)
    return _CensusStructure(
        structural_revision(index, scope_key=scope_key, parser_version=parser_version),
        scope_key, parser_version, parsed, trees, definitions, calls_by_path,
        unresolved_by_path, set(partial_paths), set(skipped_paths), importers,
    )


def _load_persisted_census_structure(
    index: object,
    runtime: object,
    *,
    scope_key: str,
    parser_version: str,
    journal: object | None,
) -> tuple[_turn0.CacheContext | None, _CensusStructure | None,
           tuple[str, ...], tuple[tuple[str, str], ...]]:
    context, walked_paths, entries = _census_persistence_context(
        index, runtime, parser_version=parser_version, journal=journal,
    )
    if context is None:
        return None, None, walked_paths, entries
    payload, reason = _turn0.read_payload(context)
    if payload is None or reason is not None:
        if reason not in {None, "missing"}:
            _turn0.emit_cache_bypass(journal, name="census_index", reason=reason)
        _turn0.emit_cache_outcome(journal, name="census_index", outcome="miss")
        return context, None, walked_paths, entries
    try:
        structure = _census_structure_from_payload(
            index, payload, scope_key=scope_key, parser_version=parser_version,
        )
        current_entries = scan_entry_set(Path(getattr(index, "root")))
        if (current_entries != entries
                or not _turn0.sources_unchanged(context, walked_paths)):
            raise ValueError("coverage check source inventory changed during cache read")
    except Exception:
        _turn0.emit_cache_outcome(journal, name="census_index", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="census_index", reason="payload_invalid")
        return context, None, walked_paths, entries
    _turn0.emit_cache_outcome(journal, name="census_index", outcome="hit")
    return context, structure, walked_paths, entries


def _save_persisted_census_structure(
    context: _turn0.CacheContext,
    index: object,
    structure: _CensusStructure,
    *,
    walked_paths: tuple[str, ...],
    entries: tuple[tuple[str, str], ...],
    parser_version: str,
    journal: object | None,
) -> None:
    try:
        current_entries = scan_entry_set(Path(getattr(index, "root")))
        if (current_entries != entries
                or not _turn0.sources_unchanged(context, walked_paths)):
            _turn0.emit_cache_bypass(journal, name="census_index", reason="source_changed")
            return
        payload = _census_structure_payload(index, structure, parser_version=parser_version)
        reason = _turn0.write_payload(context, payload)
        if reason:
            _turn0.emit_cache_bypass(journal, name="census_index", reason=reason)
    except Exception:
        _turn0.emit_cache_bypass(journal, name="census_index", reason="serialize_failed")


def _load_persisted_census_structure_fail_open(
    index: object,
    runtime: object,
    *,
    scope_key: str,
    parser_version: str,
    journal: object | None,
) -> tuple[_turn0.CacheContext | None, _CensusStructure | None,
           tuple[str, ...], tuple[tuple[str, str], ...]]:
    try:
        return _load_persisted_census_structure(
            index,
            runtime,
            scope_key=scope_key,
            parser_version=parser_version,
            journal=journal,
        )
    except Exception:
        _turn0.emit_cache_bypass(journal, name="census_index", reason="load_failed")
        return None, None, (), ()


def _census_persistence_context_fail_open(
    index: object,
    runtime: object,
    *,
    parser_version: str,
    journal: object | None,
) -> tuple[_turn0.CacheContext | None, tuple[str, ...], tuple[tuple[str, str], ...]]:
    try:
        return _census_persistence_context(
            index, runtime, parser_version=parser_version, journal=journal,
        )
    except Exception:
        _turn0.emit_cache_bypass(
            journal, name="census_index", reason="context_unavailable",
        )
        return None, (), ()


def _save_persisted_census_structure_fail_open(
    context: _turn0.CacheContext,
    index: object,
    structure: _CensusStructure,
    *,
    walked_paths: tuple[str, ...],
    entries: tuple[tuple[str, str], ...],
    parser_version: str,
    journal: object | None,
) -> None:
    try:
        _save_persisted_census_structure(
            context,
            index,
            structure,
            walked_paths=walked_paths,
            entries=entries,
            parser_version=parser_version,
            journal=journal,
        )
    except Exception:
        _turn0.emit_cache_bypass(journal, name="census_index", reason="write_failed")


@profiled_function_span("census_structure_refresh")
def _refresh_census_structure(
    index: object,
    previous: _CensusStructure,
    *,
    changed_paths: Sequence[str],
    scope_key: str,
    parser_version: str,
) -> _CensusStructure:
    """Refresh changed source plus direct dependent call edges, preserving file order."""
    indexed_paths = tuple(str(module.path) for module in getattr(index, "modules", ()))
    changed = set(changed_paths)
    if not changed or not changed.issubset(set(indexed_paths)):
        return _build_census_structure(index, scope_key=scope_key, parser_version=parser_version)

    parsed = dict(previous.parsed)
    trees = dict(previous.trees)
    definitions = dict(previous.definitions)
    calls_by_path = dict(previous.calls_by_path)
    unresolved_by_path = dict(previous.unresolved_by_path)
    partial_paths = set(previous.partial_paths)
    skipped_paths = set(previous.skipped_paths)
    affected = set(changed)
    for path in changed:
        affected.update(previous.importers_by_path.get(path, ()))
        parsed.pop(path, None)
        trees.pop(path, None)
        definitions.pop(path, None)
        calls_by_path.pop(path, None)
        unresolved_by_path.pop(path, None)
        partial_paths.discard(path)
        skipped_paths.discard(path)

        source, tree, found, partial, skipped = _parse_census_source(index, path)
        if source is not None and tree is not None and found is not None:
            parsed[path] = source
            trees[path] = tree
            definitions[path] = found
        if partial:
            partial_paths.add(path)
        if skipped:
            skipped_paths.add(path)

    module_candidates = _module_candidates(indexed_paths)
    importers = {target: set(sources) for target, sources in previous.importers_by_path.items()}
    for target in tuple(importers):
        importers[target].difference_update(changed)
        if not importers[target]:
            del importers[target]
    for path in changed:
        tree = trees.get(path)
        if tree is None:
            continue
        for target in _import_target_paths(tree, path, module_candidates):
            if target != path:
                importers.setdefault(target, set()).add(path)
    for path in changed:
        affected.update(importers.get(path, ()))

    parsed = {path: parsed[path] for path in indexed_paths if path in parsed}
    trees = {path: trees[path] for path in parsed if path in trees}
    definitions = {path: definitions[path] for path in parsed if path in definitions}
    modules = _module_names(tuple(parsed))
    for path in parsed:
        if path not in affected:
            continue
        tree = trees.get(path)
        if tree is None:
            calls_by_path.pop(path, None)
            unresolved_by_path.pop(path, None)
            continue
        calls_by_path[path], unresolved_by_path[path] = _collect_file_calls(
            path, parsed[path], tree, modules, definitions,
        )
    calls_by_path = {path: calls_by_path[path] for path in parsed if path in calls_by_path}
    unresolved_by_path = {
        path: unresolved_by_path[path] for path in parsed if path in unresolved_by_path
    }
    return _CensusStructure(
        structural_revision(index, scope_key=scope_key, parser_version=parser_version),
        scope_key, parser_version, parsed, trees, definitions, calls_by_path,
        unresolved_by_path, partial_paths, skipped_paths, importers,
    )


@profiled_function_span("census_call_projection")
def _project_call_facts(
    structure: _CensusStructure,
    shown: Mapping[str, Sequence[tuple[int, int, str]]],
) -> tuple[list[_Call], int]:
    calls: list[_Call] = []
    for path, file_calls in structure.calls_by_path.items():
        ranges = shown.get(path, ())
        calls.extend(
            replace(call, shown=any(start <= call.line <= end for start, end, _ in ranges))
            for call in file_calls
        )
    return calls, sum(structure.unresolved_by_path.values())


def _excluded(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (PurePosixPath(path).name.startswith("test_")
            or any(part.startswith(".") or part.casefold() in _FORBIDDEN_PARTS for part in parts))


def _editable_paths(runtime: object) -> tuple[str, ...]:
    """The task's declared editable product files (host-owned), safe and non-excluded."""
    try:
        from kernel_request_source_window import editable_product_paths
        raw = editable_product_paths(runtime)
    except Exception:  # noqa: BLE001 - census is presentation-only; no declared paths is valid
        return ()
    paths: list[str] = []
    for value in raw or ():
        try:
            path = _safe_path(str(value).replace("\\", "/").strip())
        except CensusError:
            continue
        if not _excluded(path):
            paths.append(path)
    return tuple(sorted(set(paths)))


def _declared_source_paths(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    paths: set[str] = set()
    for message in messages:
        if message.get("role") == "tool":
            page = _read_page_block(message.get("content"))
            if page is not None and not _excluded(page[0]):
                paths.add(page[0])
            continue
        if message.get("role") != "user":
            continue
        parsed = _source_block(message.get("content"))
        if parsed is not None and not _excluded(parsed[0].path):
            paths.add(parsed[0].path)
    return tuple(sorted(paths))


def _shown_sources(index: object, messages: Sequence[Mapping[str, object]]) -> tuple[
        dict[str, list[tuple[int, int, str]]], tuple[str, ...], dict[int, str]]:
    indexed = {module.path: module for module in getattr(index, "modules", ())}
    shown: dict[str, list[tuple[int, int, str]]] = {}
    handles: set[str] = set()
    paths_by_message: dict[int, str] = {}
    for index_in_messages, message in enumerate(messages):
        if message.get("role") == "tool":
            page = _read_page_block(message.get("content"))
            if (page is None or page[0] not in indexed or not page[0].endswith(".py")
                    or _excluded(page[0])):
                continue
            span = _verified_read_page_span(index, page)
            if span is None:
                continue
            shown.setdefault(page[0], []).append(span)
            handles.add(span[2])
            paths_by_message[index_in_messages] = page[0]
            continue
        if message.get("role") != "user":
            continue
        parsed = _source_block(message.get("content"))
        if parsed is None:
            continue
        block, _ = parsed
        module = indexed.get(block.path)
        # Relation resolution currently accepts Python AST only; non-Python source
        # blocks can be indexed for coverage but must not be represented as extracted.
        if module is None or not block.path.endswith(".py") or _excluded(block.path):
            continue
        try:
            handle, raw = index.read_content(block.path)
        except (LookupError, UnicodeError, OSError, StructuralIndexError):
            continue
        if block.window_content is not None:
            from kernel_request_source_window import (
                parse_model_source_window,
                verify_model_source_window,
            )

            try:
                source = raw.decode(handle.encoding, errors="strict")
            except (LookupError, UnicodeError):
                continue
            metadata = parse_model_source_window(block.window_content)
            if (metadata is None
                    or metadata.get("revision") != handle.content_digest
                    or metadata.get("sha256") != handle.content_digest.removeprefix("sha256:")
                    or not verify_model_source_window(
                        block.window_content, path=block.path, source=source)):
                continue
            shown.setdefault(block.path, []).extend(
                (start, end, handle.content_id)
                for start, end in block.window_ranges
            )
        else:
            try:
                content_bytes = block.text.encode(handle.encoding, errors="strict")
            except (LookupError, UnicodeError):
                continue
            if content_bytes == raw:
                shown.setdefault(block.path, []).append(
                    (1, max(1, len(raw.splitlines())), handle.content_id))
            elif block.start_line is not None and block.end_line is not None:
                selected = _line_slice(raw, block.start_line, block.end_line)
                if selected is None or not _same_shown_source(selected, content_bytes):
                    continue
                shown.setdefault(block.path, []).append(
                    (block.start_line, block.end_line, handle.content_id))
            else:
                span = _exact_source_span(raw, content_bytes)
                if span is None:
                    continue
                shown.setdefault(block.path, []).append(
                    (span[0], span[1], handle.content_id))
        handles.add(handle.content_id)
        paths_by_message[index_in_messages] = block.path
    return shown, tuple(sorted(handles)), paths_by_message


def _root(runtime: object) -> Path:
    owner = getattr(runtime, "owner", None)
    candidate = getattr(owner, "root", None)
    if candidate is None:
        candidate = getattr(runtime, "repo_root", None)
    if candidate is None:
        raise CensusError("workspace root is unavailable")
    supplied = Path(candidate).expanduser()
    if supplied.is_symlink():
        raise CensusError("workspace root is a symlink")
    root = supplied.resolve()
    if not root.is_dir() or root.name.casefold() in _FORBIDDEN_PARTS:
        raise CensusError("workspace root is outside the public source scope")
    return root


def _decision_kind(runtime: object, frame: object) -> str:
    values: list[object] = []
    owner = getattr(runtime, "owner", None)
    owner_row = getattr(owner, "task_row", None)
    for source in (frame, runtime, owner, owner_row):
        if isinstance(source, Mapping):
            if source.get("edit_frame") is True:
                return "edit"
            values.extend(source.get(key) for key in
                          ("frame_kind", "decision_kind", "decision_stage", "decision_type", "stage"))
        else:
            if getattr(source, "edit_frame", False) is True:
                return "edit"
            values.extend(getattr(source, key, None) for key in
                          ("frame_kind", "decision_kind", "decision_stage", "decision_type", "stage"))
    normalized = [
        str(item or "").strip().casefold().replace("-", "_").replace(" ", "_")
        for item in values
    ]
    if any(value in {"edit", "edit_frame", "stage_edit"} for value in normalized):
        return "edit"
    if any(value in {"stage_l", "l", "stage", "planning", "plan"} for value in normalized):
        return "stage_l"
    if any(value not in {"", "none", "unknown", "unspecified"} for value in normalized):
        return "other"
    return "stage_l"


def _catalog_revision(index: object) -> str:
    files = []
    for module in getattr(index, "modules", ()):
        handle = getattr(module, "content", None)
        files.append((str(getattr(module, "path", "")),
                      str(getattr(handle, "content_id", "")),
                      str(getattr(handle, "content_digest", ""))))
    stats = getattr(index, "stats", None)
    return _digest({
        "namespace": index.namespace.to_dict() if hasattr(index.namespace, "to_dict") else {},
        "files": files,
        "extractor": EXTRACTOR_VERSION,
        "scan_scope": getattr(stats, "scan_scope", "unknown"),
        "candidate_files": getattr(stats, "candidate_files", None),
        "indexed_files": getattr(stats, "indexed_files", len(getattr(index, "modules", ()))),
        "omitted_files": getattr(stats, "omitted_files", 0),
        "priority_files_omitted": getattr(stats, "priority_files_omitted", 0),
    })


def _capture_public_sources(index: object, packet: CensusPacket) -> tuple[Mapping[str, object], ...]:
    """Acquire bounded current bodies through this census generation's reader."""
    paths = sorted({str(site["path"]) for group in packet.groups
                    for site in group.get("sites", ()) if isinstance(site, Mapping)})
    if not paths or len(paths) > 64:
        return ()
    rows: list[Mapping[str, object]] = []
    total = 0
    for path in paths:
        module = index.module(path)
        if module is None or module.content.byte_length > 262_144:
            return ()
        try:
            handle, raw = index.read_content(path, max_bytes=262_144)
        except (LookupError, UnicodeError, OSError, StructuralIndexError):
            return ()
        if len(raw) > 262_144:
            return ()
        text = raw.decode(handle.encoding, errors="strict")
        encoded = text.encode("utf-8", errors="strict")
        total += len(encoded)
        if len(encoded) > 262_144 or total > 1_048_576:
            return ()
        rows.append({"path": path, "handle": handle.content_id, "text": text,
                     "sha256": hashlib.sha256(encoded).hexdigest(),
                     "raw_sha256": hashlib.sha256(raw).hexdigest()})
    return tuple(rows)


def _unindexed_candidate_files(candidate_files: object, indexed_files: object,
                               omitted_files: object) -> int:
    if (type(candidate_files) is int and type(indexed_files) is int
            and type(omitted_files) is int):
        return max(0, candidate_files - indexed_files - omitted_files)
    return 0


def _partial_coverage_details(
    *,
    candidate_files: object,
    catalogued_files: int,
    files_omitted: int,
    priority_files_omitted: int,
    files_unindexed: int,
    parse_partial: int,
) -> tuple[str, ...]:
    details: list[str] = []
    if files_omitted:
        if type(candidate_files) is int:
            details.append(
                f"structural index {catalogued_files} of {candidate_files} candidate files "
                f"({files_omitted} omitted)"
            )
        else:
            details.append(f"structural index omitted {files_omitted} candidate files")
    if priority_files_omitted:
        details.append(f"{priority_files_omitted} priority source paths omitted")
    if files_unindexed:
        details.append(f"{files_unindexed} selected source files could not be indexed")
    if parse_partial:
        details.append(f"{parse_partial} files had partial or unsupported parsing")
    return tuple(details) or ("source coverage is incomplete",)


@profiled_function_span("callsite_census_build_packet")
def _build_packet(runtime: object, root: Path, messages: Sequence[Mapping[str, object]], *,
                  decision_kind: str,
                  token_counter: Callable[[object], int],
                  source_facts_out: list[Mapping[str, object]] | None = None,
                  reuse_report: _IndexReuseReport | None = None,
                  index_reuse: bool = True,
                  incremental_mode: str = "off",
                  environ: Mapping[str, str] | None = None,
                  request_id: object = None) -> CensusPacket | None:
    editable = _editable_paths(runtime)
    priority_paths = tuple(dict.fromkeys((*_declared_source_paths(messages), *editable)))
    cache_scope_key: str | None = None
    parser_version = f"{EXTRACTOR_VERSION}:ast-{sys.version_info.major}.{sys.version_info.minor}"
    lease = None
    if index_reuse:
        cache_scope_key = _index_scope_key(runtime, root)
        lease = acquire_index(
            runtime, root, scope_key=cache_scope_key, parser_version=parser_version,
            priority_paths=priority_paths, enabled=True, build_index_fn=build_structural_index,
            environ=environ,
        )
        index = lease.index
        if reuse_report is not None:
            reuse_report.status = lease.reuse_status
            reuse_report.reason = lease.reuse_reason
            reuse_report.validation_cap_mib = census_validation_source_cap_mib(environ)
            reuse_report.index_source_bytes_admitted = (
                census_index_source_bytes(lease.cache.index)
                + census_index_source_bytes(lease.cache.overlay_index)
                if lease.cache is not None else 0
            )
            observed_source_bytes = census_index_source_bytes(lease.index)
            source_byte_limit = census_validation_source_byte_limit(environ)
            if lease.reuse_reason == "limits" and observed_source_bytes > source_byte_limit:
                reuse_report.limit_kind = "source_bytes"
                reuse_report.limit_observed = observed_source_bytes
                reuse_report.limit_allowed = source_byte_limit
        try:
            setattr(runtime, "_stateframe_callsite_census_index_reuse_status", lease.reuse_status)
            setattr(runtime, "_stateframe_callsite_census_index_reuse_reason", lease.reuse_reason)
        except (AttributeError, TypeError):
            pass
    else:
        if reuse_report is not None:
            reuse_report.status = "off"
            reuse_report.reason = "disabled"
        # Preserve the pre-S1 path exactly when the independently switchable cache is off.
        index = build_structural_index(
            root,
            # Declared editable files are indexed first so the file cap can never hide them.
            priority_paths=priority_paths,
            include_call_graph=False,
        )
        try:
            setattr(runtime, "_stateframe_callsite_census_index_reuse_status", "off")
            setattr(runtime, "_stateframe_callsite_census_index_reuse_reason", "disabled")
        except (AttributeError, TypeError):
            pass
    shown, source_handles, _ = _shown_sources(index, messages)
    if not shown:
        return None
    revision = _catalog_revision(index)
    spans = [(path, start, end, handle) for path, rows in sorted(shown.items())
             for start, end, handle in rows]
    span_digest = _digest(spans)
    cache_key_fields: dict[str, object] = {
        "root": str(root),
        "namespace": index.namespace.to_dict() if hasattr(index.namespace, "to_dict") else {},
        "index_revision": revision,
        "source_span_set_digest": span_digest,
        "extractor": EXTRACTOR_VERSION,
        "editable": list(editable),
    }
    if cache_scope_key is not None:
        cache_key_fields["run_scope"] = cache_scope_key
    cache_key = _digest(cache_key_fields)
    cached = getattr(runtime, "_stateframe_callsite_census_packet", None)
    if isinstance(cached, CensusPacket) and cached.cache_key == cache_key:
        if source_facts_out is not None:
            source_facts_out.extend(_capture_public_sources(index, cached))
        return cached if cached.decision_kind == decision_kind else replace(cached, decision_kind=decision_kind)

    parse_loop_partial = False
    if index_reuse and lease is not None:
        current_structure_revision = structural_revision(
            index, scope_key=cache_scope_key or "", parser_version=parser_version,
        )
        previous_structure = lease.cache.analysis if lease.cache is not None else None
        if (isinstance(previous_structure, _CensusStructure)
                and previous_structure.revision == current_structure_revision
                and not lease.changed_paths):
            structure = previous_structure
        elif (isinstance(previous_structure, _CensusStructure) and lease.incremental
              and previous_structure.scope_key == cache_scope_key
              and previous_structure.parser_version == parser_version
              and incremental_mode in {"on", "shadow"}):
            try:
                incremental_structure = _refresh_census_structure(
                    index, previous_structure, changed_paths=lease.changed_paths,
                    scope_key=cache_scope_key or "", parser_version=parser_version,
                )
            except Exception:
                # A failed cache refresh falls through to the existing cold computation.
                incremental_structure = None
            if incremental_structure is None:
                structure = _build_census_structure(
                    index, scope_key=cache_scope_key or "", parser_version=parser_version,
                )
            elif incremental_mode == "shadow":
                cold_structure = _build_census_structure(
                    index, scope_key=cache_scope_key or "", parser_version=parser_version,
                )
                _emit_incremental_shadow(
                    runtime,
                    request_id=request_id,
                    matches=_same_census_structure(incremental_structure, cold_structure),
                    changed_paths=lease.changed_paths,
                )
                # Shadow always serves the cold result. It never depends on the candidate path.
                structure = cold_structure
            else:
                structure = incremental_structure
        else:
            persistence_context = None
            persistence_paths: tuple[str, ...] = ()
            persistence_entries: tuple[tuple[str, str], ...] = ()
            structure = None
            journal = getattr(runtime, "journal", None)
            if incremental_mode == "off" and _turn0.enabled():
                (
                    persistence_context,
                    structure,
                    persistence_paths,
                    persistence_entries,
                ) = _load_persisted_census_structure_fail_open(
                    index,
                    runtime,
                    scope_key=cache_scope_key or "",
                    parser_version=parser_version,
                    journal=journal,
                )
            if structure is None:
                structure = _build_census_structure(
                    index, scope_key=cache_scope_key or "", parser_version=parser_version,
                )
                if persistence_context is not None and incremental_mode == "off":
                    _save_persisted_census_structure_fail_open(
                        persistence_context,
                        index,
                        structure,
                        walked_paths=persistence_paths,
                        entries=persistence_entries,
                        parser_version=parser_version,
                        journal=journal,
                    )
        if (lease.cache is not None and _analysis_source_size(index) <= MAX_CACHED_ANALYSIS_SOURCE_BYTES):
            lease.cache.analysis = structure
        elif lease.cache is not None:
            lease.cache.analysis = None
        if (previous_structure is not None
                and previous_structure is not structure
                and lease.incremental
                and incremental_mode == "off"
                and _turn0.enabled()):
            journal = getattr(runtime, "journal", None)
            persistence_context, persistence_paths, persistence_entries = (
                _census_persistence_context_fail_open(
                    index, runtime, parser_version=parser_version, journal=journal,
                )
            )
            if persistence_context is not None:
                _turn0.emit_cache_outcome(journal, name="census_index", outcome="miss")
                _save_persisted_census_structure_fail_open(
                    persistence_context,
                    index,
                    structure,
                    walked_paths=persistence_paths,
                    entries=persistence_entries,
                    parser_version=parser_version,
                    journal=journal,
                )
        parsed = structure.parsed
        trees = structure.trees
        definitions = structure.definitions
        parse_partial = len(structure.partial_paths)
        files_skipped = len(structure.skipped_paths)
        calls, unresolved = _project_call_facts(structure, shown)
    else:
        parsed, parse_partial, files_skipped = _source_paths(index)
        trees: dict[str, ast.Module] = {}
        definitions: dict[str, dict[str, _Definition]] = {}
        for path, (_, _, source) in parsed.items():
            try:
                tree = ast.parse(source, filename=path)
            except (SyntaxError, ValueError):
                parse_loop_partial = True
                continue
            trees[path] = tree
            definitions[path] = _definitions(tree, path)

        calls = []
        unresolved = 0
        modules = _module_names(tuple(parsed))
        for path, tree in trees.items():
            handle, _, source = parsed[path]
            module_scope = _scope(tree.body, "module", path, (), modules, definitions)
            collector = _Collector(tree=tree, path=path, handle=handle.content_id,
                                   source=source, module_scope=module_scope,
                                   modules=modules, definitions=definitions,
                                   shown=[(start, end) for start, end, _ in shown.get(path, ())])
            collector.visit(tree)
            calls.extend(collector.calls)
            unresolved += collector.unresolved
    index_stats = getattr(index, "stats", None)
    candidate_files = getattr(index_stats, "candidate_files", None)
    catalogued_files = getattr(index_stats, "indexed_files", len(getattr(index, "modules", ())))
    files_omitted = getattr(index_stats, "omitted_files", 0)
    priority_files_omitted = getattr(index_stats, "priority_files_omitted", 0)
    unindexed_files = _unindexed_candidate_files(
        candidate_files, catalogued_files, files_omitted,
    )
    scan_truncated = files_omitted > 0
    partial = bool(parse_partial or scan_truncated or priority_files_omitted or unindexed_files)
    partial = bool(partial or parse_loop_partial)

    grouped: dict[str, list[_Call]] = {}
    for call in calls:
        group_id = _digest({"relation_kind": "constructor_call", "binding": call.identity})
        grouped.setdefault(group_id, []).append(call)
    eligible: list[tuple[str, list[_Call]]] = []
    shown_paths = set(shown)

    def _unread_declared(site: _Call) -> bool:
        return not site.shown and site.path in editable and site.path not in shown_paths

    for group_id, sites in grouped.items():
        shown_sites = [site for site in sites if site.shown and site.path in shown_paths]
        crossfile = len(shown_sites) >= 2 and len({site.path for site in shown_sites}) >= 2
        # Unread declared site: the model edits one declared
        # file and never reads the other declared file's matching construction site, so a
        # "shown in >=2 files" rule never fires. A constructor shown at least once that also
        # occurs in a declared editable file the model has NOT read is listed too; the index
        # is measured from the repo bytes, and unread rows are labelled as such.
        unread_declared = bool(shown_sites) and any(
            site.path in editable and site.path not in shown_paths for site in sites)
        if crossfile or unread_declared:
            eligible.append((group_id, sorted(sites, key=lambda site:
                                              (not site.shown, not _unread_declared(site), site.path, site.line, site.column))))
    if not eligible:
        return None
    eligible.sort(key=lambda row: min(
        (site.path, site.line, site.column) for site in row[1] if site.shown))

    displayed: dict[str, list[_Call]] = {group_id: [] for group_id, _ in eligible}
    row_order = [site for _, sites in eligible for site in sites if site.shown]
    # Unread declared sites outrank other unshown rows in the shared display budget.
    row_order += [site for _, sites in eligible for site in sites
                  if not site.shown and _unread_declared(site)]
    row_order += [site for _, sites in eligible for site in sites
                  if not site.shown and not _unread_declared(site)]

    def render() -> tuple[str, int, int]:
        rows: list[str] = [_HEADER]
        if partial:
            details = _partial_coverage_details(
                candidate_files=candidate_files,
                catalogued_files=catalogued_files,
                files_omitted=files_omitted,
                priority_files_omitted=priority_files_omitted,
                files_unindexed=unindexed_files,
                parse_partial=parse_partial,
            )
            rows.append(
                "[coverage partial; " + "; ".join(details) +
                "; relation totals are lower bounds]"
            )
        shown_count_by_group = {
            group_id: sum(site.shown for site in sites)
            for group_id, sites in eligible
        }
        for group_id, sites in eligible:
            chosen = displayed[group_id]
            if not chosen:
                continue
            canonical = sites[0].display
            omitted = len(sites) - len(chosen)
            rows.append(
                f"group {canonical}: {shown_count_by_group[group_id]} shown / {len(sites)} indexed; "
                f"{len(chosen)} displayed" + (f"; {omitted} omitted" if omitted else "")
            )
            rows.extend(
                f"  {'shown  ' if site.shown else 'unshown'} {site.path}::{site.owner} L{site.line}"
                for site in chosen
            )
        payload = "\n".join(rows)
        try:
            count = int(token_counter({"messages": [{"role": "user", "content": payload + "\n" + _END}]}))
        except (TypeError, ValueError, OverflowError):
            count = MAX_SERVING_TOKENS + 1
        omitted_groups = sum(not displayed[group_id] for group_id, _ in eligible)
        return payload, count, omitted_groups

    for site in row_order:
        if sum(len(rows) for rows in displayed.values()) >= MAX_DISPLAYED_ROWS:
            break
        group_id = _digest({"relation_kind": "constructor_call", "binding": site.identity})
        displayed[group_id].append(site)
        _, count, _ = render()
        if count > MAX_SERVING_TOKENS:
            displayed[group_id].pop()
    payload, serving_tokens, omitted_groups = render()
    if serving_tokens > MAX_SERVING_TOKENS or not any(
            site.shown for rows in displayed.values() for site in rows):
        return None

    packet_groups = []
    for group_id, sites in eligible:
        selected = displayed[group_id]
        packet_groups.append({
            "group_id": group_id,
            "relation_kind": "constructor_call",
            "canonical_binding": sites[0].identity,
            "resolution": sites[0].resolution,
            "indexed_sites": len(sites),
            "total_relation": "lower_bound" if partial else "exact",
            "shown_sites": sum(site.shown for site in sites),
            "displayed_sites": len(selected),
            "omitted_sites": len(sites) - len(selected),
            "sites": [{
                "site_id": _digest((site.identity, site.source_handle, site.path,
                                    site.start_byte, site.end_byte)),
                "source_handle": site.source_handle,
                "path": site.path,
                "owner": site.owner,
                "start_byte": site.start_byte,
                "end_byte": site.end_byte,
                "line": site.line,
                "column": site.column,
                "relation_kind": "constructor_call",
                "binding_strategy": site.strategy,
                "shown": site.shown,
            } for site in sites],
        })
    namespace = index.namespace.to_dict() if hasattr(index.namespace, "to_dict") else {}
    coverage = {
        "status": "PARTIAL" if partial else "COMPLETE",
        "roots": [root.name],
        "exclusions_digest": _digest(sorted(_FORBIDDEN_PARTS)),
        "files_indexed": len(trees),
        "files_catalogued": catalogued_files,
        "files_available": candidate_files,
        "files_omitted": files_omitted,
        "priority_files_omitted": priority_files_omitted,
        "files_unindexed": unindexed_files,
        "files_partial": parse_partial,
        "files_skipped": files_skipped,
        "unresolved_count": unresolved,
        "scan_scope": getattr(index_stats, "scan_scope", "unknown"),
        "scan_truncated": scan_truncated,
    }
    packet = CensusPacket(
        SCHEMA, json.dumps(namespace, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        revision, (EXTRACTOR_VERSION,), ROUTE_ID, decision_kind,
        source_handles, span_digest, coverage, tuple(packet_groups),
        {"status": "RENDERED", "payload_digest": _digest(payload),
         "token_count": serving_tokens, "omitted_group_count": omitted_groups},
        payload, cache_key,
    )
    if not bool(coverage["scan_truncated"]) and not priority_files_omitted and not unindexed_files:
        try:
            setattr(runtime, "_stateframe_callsite_census_packet", packet)
        except (AttributeError, TypeError):
            pass
    if source_facts_out is not None:
        source_facts_out.extend(_capture_public_sources(index, packet))
    return packet


def _workspace_mode(runtime: object, frame: object, environ: Mapping[str, str] | None) -> tuple[str, str]:
    kind = _decision_kind(runtime, frame)
    if kind not in {"stage_l", "edit"}:
        return kind, "off"
    source = os.environ if environ is None else environ
    is_edit = kind == "edit"
    name = EDIT_MODE_ENV if is_edit else MODE_ENV
    # Coverage checks use verified callsite evidence before assigning relevance.
    # vs 6/55): the EDIT-frame header ships ON by default. It keeps its own flag so it can be switched off alone.
    default = "on"
    attr = "stateframe_callsite_census_edit_mode" if is_edit else "stateframe_callsite_census_mode"
    configured = getattr(runtime, attr, None)
    value = str(configured) if configured is not None else source.get(name, default)
    return kind, value


def _incremental_mode(environ: Mapping[str, str] | None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get("STATEFRAME_CENSUS_INCREMENTAL", "off") or "off").strip().lower()
    if value not in {"off", "shadow", "on"}:
        return "off"
    return value


def _same_census_structure(left: _CensusStructure, right: _CensusStructure) -> bool:
    """Compare every parsed fact that can affect the emitted census packet."""
    return (
        left.revision == right.revision
        and left.scope_key == right.scope_key
        and left.parser_version == right.parser_version
        and left.parsed == right.parsed
        and left.definitions == right.definitions
        and left.calls_by_path == right.calls_by_path
        and left.unresolved_by_path == right.unresolved_by_path
        and left.partial_paths == right.partial_paths
        and left.skipped_paths == right.skipped_paths
        and left.importers_by_path == right.importers_by_path
    )


def _emit_incremental_shadow(
        runtime: object, *, request_id: object, matches: bool, changed_paths: Sequence[str],
) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        return
    try:
        emit(
            "callsite_census_incremental_shadow",
            producer="kernel_callsite_census",
            request_id=request_id,
            matches_cold=matches,
            changed_file_count=len(changed_paths),
            dispatch=False,
        )
    except Exception:
        pass


def _copy_message(message: Mapping[str, object]) -> dict[str, object]:
    return dict(message)


@profiled_function_span("callsite_census_prepare_request")
def prepare_request(
        runtime: object, frame: object, messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]], *, frame_ceiling: int,
        request_id: object = None,
        request_token_counter: Callable[[object], int] = token_estimate,
        census_token_counter: Callable[[object], int] = token_estimate,
        environ: Mapping[str, str] | None = None,
        collect_source_facts: bool = False) -> CensusDelivery:
    """Attach one near-source header after final source selection and budgeting."""
    original = tuple(_copy_message(message) for message in messages)
    decision_kind = "other"
    mode = "off"
    incremental_mode = "off"
    source_facts: list[Mapping[str, object]] = []
    census_started_ns: int | None = None
    census_reuse = _IndexReuseReport()
    census_reuse.validation_cap_mib = census_validation_source_cap_mib(environ)

    def finish(status: str, *, packet: CensusPacket | None = None,
               delivered: Sequence[dict[str, object]] | None = None,
               header_path: str | None = None,
               header_message_index: int | None = None,
               payload_digest: str | None = None,
               error_type: str | None = None,
               error_message: str | None = None,
               error_site: str | None = None,
               placement: str = "after_source_header") -> CensusDelivery:
        result = CensusDelivery(
            tuple(delivered) if delivered is not None else original,
            status, packet, header_path, header_message_index, payload_digest,
            decision_kind, mode, error_type, placement, tuple(source_facts),
        )
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            try:
                emit("callsite_census_outcome", producer="kernel_callsite_census",
                     request_id=request_id, status=status, decision_kind=decision_kind,
                     mode=mode, error_type=error_type, error_message=error_message,
                     error_site=error_site, dispatch=False,
                      census_wall_ns=(time.perf_counter_ns() - census_started_ns
                                      if census_started_ns is not None else None),
                      index_reuse=census_reuse.status,
                      index_reuse_reason=census_reuse.reason,
                      index_source_bytes_admitted=census_reuse.index_source_bytes_admitted,
                      validation_cap_mib=census_reuse.validation_cap_mib,
                      limit_kind=census_reuse.limit_kind,
                      limit_observed=census_reuse.limit_observed,
                      limit_allowed=census_reuse.limit_allowed)
            except Exception as error:  # noqa: BLE001 - observability must not break provider dispatch
                logging.getLogger(__name__).error(
                    "callsite census outcome journal emit failed",
                    extra={"error_type": type(error).__name__},
                )
        return result

    try:
        decision_kind, mode = _workspace_mode(runtime, frame, environ)
    except Exception as error:  # noqa: BLE001 - expose bounded failure details on the request outcome
        return finish(
            "unavailable",
            error_type=type(error).__name__,
            error_message=_safe_error_message(error),
            error_site=_error_site(error),
        )
    if mode not in {"off", "shadow", "on"}:
        return finish("invalid_flag", error_type="CensusError")
    if mode == "off":
        census_reuse.reason = "disabled"
        return finish("off")
    try:
        incremental_mode = _incremental_mode(environ)
    except ValueError as error:
        return finish(
            "invalid_flag", error_type="CensusError",
            error_message=_safe_error_message(error), error_site=_error_site(error),
        )
    try:
        index_reuse = census_index_reuse_enabled(environ)
    except ValueError as error:
        return finish(
            "invalid_flag", error_type="CensusError",
            error_message=_safe_error_message(error), error_site=_error_site(error),
        )
    try:
        root = _root(runtime)
        census_started_ns = time.perf_counter_ns()
        census_reuse.status = "on" if index_reuse else "off"
        census_reuse.reason = "scan_unstable" if index_reuse else "disabled"
        packet = _build_packet(runtime, root, messages, decision_kind=decision_kind,
                                token_counter=census_token_counter,
                                source_facts_out=source_facts if collect_source_facts else None,
                                 reuse_report=census_reuse,
                                 index_reuse=index_reuse,
                                 incremental_mode=incremental_mode,
                                 environ=environ,
                                 request_id=request_id)
    except Exception as error:  # noqa: BLE001 - report the bounded failure details
        return finish(
            "unavailable",
            error_type=type(error).__name__,
            error_message=_safe_error_message(error),
            error_site=_error_site(error),
        )
    if packet is None:
        return finish("route_off")
    if mode == "shadow":
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            try:
                emit("callsite_census_shadow", producer="kernel_callsite_census",
                     request_id=request_id, route_id=packet.route_id,
                     packet_digest=packet.render.get("payload_digest"),
                     selected_source_count=len(packet.selected_source_handles), dispatch=False)
            except Exception as error:  # noqa: BLE001 - keep the provider request observable
                logging.getLogger(__name__).error(
                    "callsite census shadow journal emit failed",
                    extra={"error_type": type(error).__name__},
                )
        return finish("shadow", packet=packet)

    candidate = [_copy_message(message) for message in messages]
    # Older turns can remain in the transcript. Collapse their presentation
    # copies, then place the current revision's packet once in this wire request.
    for message in candidate:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _strip_census_presentation(content)
    shown_paths = {
        str(site.get("path"))
        for group in packet.groups
        for site in group.get("sites", ())
        if isinstance(site, Mapping) and site.get("shown") is True
    }
    target_index = next((
        index for index, message in enumerate(candidate)
        if message.get("role") == "user"
        and (parsed := _source_block(message.get("content"))) is not None
        and parsed[0].path in shown_paths
    ), None)
    if target_index is None:
        # Native read pages (role=tool) are the normal source carrier in live runs.
        # Anchor after the newest shown page's header so the shared prefix stays long.
        page_index = next((
            index for index in range(len(candidate) - 1, -1, -1)
            if candidate[index].get("role") == "tool"
            and (page := _read_page_block(candidate[index].get("content"))) is not None
            and page[0] in shown_paths
        ), None)
        if page_index is None:
            return finish("source_header_missing", packet=packet)
        content = candidate[page_index]["content"]
        header = _READ_PAGE_HEADER.match(content)
        if header is None:
            return finish("source_header_missing", packet=packet)
        candidate[page_index]["content"] = (
            content[:header.end()] + packet.payload + "\n" + _END + "\n" + content[header.end():]
        )
        try:
            request_tokens = int(request_token_counter({"messages": candidate, "tools": list(tools)}))
            ceiling = max(0, int(frame_ceiling))
        except Exception as error:  # noqa: BLE001 - always report the budget failure type
            return finish("budget_unavailable", packet=packet, error_type=type(error).__name__)
        if request_tokens > ceiling:
            return finish("request_cap", packet=packet)
        digest = _digest({"messages": candidate, "tools": list(tools)})
        page = _read_page_block(candidate[page_index]["content"])
        if page is None:
            return finish("source_header_missing", packet=packet)
        return finish("fired", packet=packet, delivered=candidate, header_path=page[0],
                      header_message_index=page_index, payload_digest=digest,
                      placement="after_read_page_header")
    content = candidate[target_index].get("content")
    if not isinstance(content, str):
        return finish("source_header_missing", packet=packet)
    selected_source = _source_block(content)
    if selected_source is None:
        return finish("source_header_missing", packet=packet)
    insertion_offset = selected_source[1]
    if not 0 <= insertion_offset <= len(content):
        return finish("source_header_missing", packet=packet)
    placement = ("after_source_metadata" if selected_source[0].window_content is not None
                 else "after_source_header")
    candidate[target_index]["content"] = (
        content[:insertion_offset] + packet.payload + "\n" + _END + "\n"
        + content[insertion_offset:]
    )
    try:
        request_tokens = int(request_token_counter({"messages": candidate, "tools": list(tools)}))
        ceiling = max(0, int(frame_ceiling))
    except Exception as error:  # noqa: BLE001 - always report the budget failure type
        return finish("budget_unavailable", packet=packet, error_type=type(error).__name__)
    if request_tokens > ceiling:
        return finish("request_cap", packet=packet)
    digest = _digest({"messages": candidate, "tools": list(tools)})
    selected_source = _source_block(candidate[target_index]["content"])
    if selected_source is None:
        return finish("source_header_missing", packet=packet)
    path = selected_source[0].path
    return finish("fired", packet=packet, delivered=candidate, header_path=path,
                  header_message_index=target_index, payload_digest=digest,
                  placement=placement)


__all__ = [
    "CensusDelivery", "CensusError", "CensusPacket", "EDIT_MODE_ENV",
    "HEADER_TEXT", "MAX_DISPLAYED_ROWS", "MAX_SERVING_TOKENS", "MODE_ENV", "ROUTE_ID",
    "SCHEMA", "prepare_request",
]
