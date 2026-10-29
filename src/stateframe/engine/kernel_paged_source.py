"""Lossless, revision-bound source descriptors for oversized frame inputs.

The model-facing descriptor is deliberately only an address book.  The host
keeps the complete source text and its exact digest; reads and edits continue
to resolve against the full current file rather than against a cached page.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence

import kernel_parse_cache as _parse_cache


SCHEMA = "stateframe-paged-source/1"
HANDLE_SCHEMA = "stateframe-source-handle/1"
PAGED_SOURCE_PREFIX = "[Paged source:"


def source_revision(source: str) -> str:
    """Return the full UTF-8 source digest used by existing frame bindings."""
    if not isinstance(source, str):
        raise TypeError("paged source must be text")
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _qualified_name(stack: Sequence[str], name: str) -> str:
    return ".".join((*stack, name)) if stack else name


def symbol_outline(source: str) -> tuple[dict[str, object], ...]:
    """Return deterministic top-level and nested Python symbol ranges."""
    if not isinstance(source, str):
        return _symbol_outline(source)
    # L112: literal-site labelling outlined the same file once per site (360x/step).
    # Memoized by content; each call gets fresh row dicts, so no caller can alias.
    rows = _parse_cache.derived_source("paged_source.symbol_outline", source, "", _symbol_outline)
    return tuple(dict(row) for row in rows)


def _symbol_outline(source: str) -> tuple[dict[str, object], ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    rows: list[dict[str, object]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _visit_symbol(self, node: ast.AST, kind: str, name: str) -> None:
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            qualified = _qualified_name(self.stack, name)
            rows.append({
                "kind": kind,
                "name": name,
                "qualified_name": qualified,
                "start_line": start,
                "end_line": end,
                "start_column": int(getattr(node, "col_offset", 0) or 0),
                "end_column": int(getattr(node, "end_col_offset", 0) or 0),
            })
            self.stack.append(name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_symbol(node, "class", node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_symbol(node, "function", node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_symbol(node, "async_function", node.name)

    Visitor().visit(tree)
    return tuple(rows)


def _handle(path: str, revision: str, *, symbol: str | None,
            start_line: int, end_line: int) -> dict[str, object]:
    kind = "symbol" if symbol else "range"
    label = symbol or f"lines:{start_line}-{end_line}"
    return {
        "schema": HANDLE_SCHEMA,
        "kind": kind,
        "handle": f"source-range:{path}:{revision}:{label}:{start_line}-{end_line}",
        "path": path,
        "revision": revision,
        "symbol": symbol,
        "start_line": start_line,
        "end_line": end_line,
    }


def _outline_digest(outline: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(list(outline), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_descriptor(path: str, source: str, *, revision: str | None = None,
                     ceiling_tokens: int = 8192) -> dict[str, object]:
    """Build a compact address book without placing source bytes on the wire."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("paged source requires a path")
    if not isinstance(source, str):
        raise TypeError("paged source must be text")
    normalized = path.replace("\\", "/").lstrip("./")
    full_revision = revision or source_revision(source)
    expected_revision = source_revision(source)
    if full_revision != expected_revision:
        raise ValueError("paged source revision does not match full source")
    lines = _source_lines(source)
    outline = symbol_outline(source)
    handles = [
        _handle(normalized, full_revision, symbol=str(row["qualified_name"]),
                start_line=int(row["start_line"]), end_line=int(row["end_line"]))
        for row in outline
    ]
    total_lines = len(lines)
    handles.append(_handle(normalized, full_revision, symbol=None,
                           start_line=1, end_line=max(1, total_lines)))
    descriptor = {
        "schema": SCHEMA,
        "path": normalized,
        "revision": full_revision,
        "bytes": len(source.encode("utf-8")),
        "characters": len(source),
        "lines": total_lines,
        "complete": True,
        "content_in_frame": False,
        "outline_digest": _outline_digest(outline),
        "outline": [dict(row) for row in outline],
        "handles": handles,
        "read_contract": {
            "tool": "read",
            "symbol_argument": "qualified_name",
            "range_arguments": {
                "start_line": "1-based", "end_line": "inclusive",
                "line": "start_line alias", "max_lines": "bounded",
            },
            "max_chars": 64000,
            "max_lines": 1000,
            "full_revision_required": True,
        },
        "frame_ceiling_tokens": max(1, int(ceiling_tokens)),
    }
    return descriptor


def render_descriptor(path: str, source: str, *, revision: str | None = None,
                      ceiling_tokens: int = 8192) -> str:
    descriptor = build_descriptor(path, source, revision=revision,
                                  ceiling_tokens=ceiling_tokens)
    return (f"{PAGED_SOURCE_PREFIX} {descriptor['path']}]\n"
            + json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\nFull source is available only through revision-bound read pages; no source bytes are omitted from the host identity.")


def parse_descriptor(message: str) -> dict[str, object] | None:
    if not isinstance(message, str) or not message.startswith(PAGED_SOURCE_PREFIX):
        return None
    lines = message.splitlines()
    if len(lines) < 2:
        return None
    try:
        value = json.loads(lines[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_descriptor(message: str, *, path: str, source: str,
                        revision: str | None = None) -> dict[str, object]:
    descriptor = parse_descriptor(message)
    expected_revision = revision or source_revision(source)
    normalized = path.replace("\\", "/").lstrip("./")
    if descriptor is None or descriptor.get("schema") != SCHEMA:
        raise ValueError("paged source descriptor is missing or malformed")
    if descriptor.get("path") != normalized:
        raise ValueError("paged source descriptor path differs from current target")
    if descriptor.get("revision") != expected_revision:
        raise ValueError("paged source descriptor revision differs from current source")
    if descriptor.get("bytes") != len(source.encode("utf-8")):
        raise ValueError("paged source descriptor byte identity differs from current source")
    if descriptor.get("characters") != len(source) or descriptor.get("lines") != len(_source_lines(source)):
        raise ValueError("paged source descriptor size differs from current source")
    if descriptor.get("complete") is not True or descriptor.get("content_in_frame") is not False:
        raise ValueError("paged source descriptor must declare complete host identity")
    return descriptor


def locate_symbol(source: str, symbol: str) -> tuple[int, int] | None:
    requested = str(symbol or "").strip().replace("::", ".")
    if not requested:
        return None
    rows = symbol_outline(source)
    exact = [row for row in rows if row.get("qualified_name") == requested]
    candidates = exact or [row for row in rows if row.get("name") == requested]
    if not candidates:
        return None
    row = candidates[0]
    return int(row["start_line"]), int(row["end_line"])


def page_window_args(args: Mapping[str, object], source: str) -> tuple[int | None, int | None, str | None]:
    """Resolve symbol/range arguments before the bounded source pager runs."""
    symbol = args.get("symbol")
    if symbol not in (None, ""):
        located = locate_symbol(source, str(symbol))
        if located is None:
            return None, None, "SYMBOL_NOT_FOUND"
        return located[0], located[1], None
    start = args.get("start_line", args.get("line"))
    end = args.get("end_line")
    if start in (None, "") and end in (None, ""):
        return None, None, None
    try:
        start_line = max(1, int(start)) if start not in (None, "") else 1
    except (TypeError, ValueError):
        start_line = 1
    try:
        end_line = max(start_line, int(end)) if end not in (None, "") else None
    except (TypeError, ValueError):
        end_line = None
    return start_line, end_line, None


__all__ = [
    "HANDLE_SCHEMA", "PAGED_SOURCE_PREFIX", "SCHEMA", "build_descriptor",
    "locate_symbol", "page_window_args", "parse_descriptor", "render_descriptor",
    "source_revision", "symbol_outline", "validate_descriptor",
]
