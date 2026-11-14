"""Structure maps for the view ladder's `symbols` / `outline` rungs.

Pure text analysis of the HELD range of an artifact (P1: never the whole file unless the
whole file is what was held). Every row carries an exact `L<a>-<b>` anchor so the model can
ask for that range in one step, and every row's text is a slice of the delivered lines
(stripped, multi-line headers joined with a single space), never regenerated from a tree.

Parsers: Python via stdlib `ast` (dedent retry on an indented slice), a generic line scanner
for other code, `json`/`tomllib` key trees, a YAML/Markdown text scan and a head/tail/errors
map for logs. `parser=` names the parser actually used, so the model knows how much to trust
the map; the strings are this renderer's own (`ast`, `scan`, `json`, `tomllib`, `log`).
"""
from __future__ import annotations

import ast
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol

FACT_CAP = 24
LOG_HEAD = 10
LOG_TAIL = 20
LOG_ERRORS = 30
LOG_THRESHOLD_LINES = 200
GENERIC_SIGNATURE_WIDTH = 120

_ERROR_LINE = re.compile(r"(?i)\b(error|exception|traceback|fatal|panic|failed|denied|refused)\b")
_GENERIC_DEF = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?(?:default\s+)?(?:pub(?:\(crate\))?\s+)?(?:async\s+)?(?:static\s+)?"
    r"(?P<kind>def|class|function|fn|func|struct|impl|interface|type|enum|trait|module|const|let|var|val|object)\b"
    r"\s*(?P<name>[A-Za-z_$][\w$.:<>]*)?"
)
_GENERIC_ARROW = re.compile(
    r"^(?P<indent>[ \t]*)(?P<name>[A-Za-z_$][\w$]*)\s*(?:=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>|\([^)]*\)\s*\{)"
)
_GENERIC_IMPORT = re.compile(r"^\s*(import|from|use|require\(|#include|using)\b")
_YAML_KEY = re.compile(r"^(\s*)([^\s#:][^:]*):\s*(.*)$")
_MD_HEADING = re.compile(r"^#{1,6}\s")
_TOML_TABLE = re.compile(r"^\[.*\]$")
_TOML_KEY = re.compile(r"^\w+\s*=")
_SHEBANG = re.compile(r"^#!.*\b(python|bash|sh|zsh|node)\b")

_EXTENSIONS = {
    ".py": "python", ".pyi": "python",
    ".js": "code", ".jsx": "code", ".ts": "code", ".tsx": "code", ".mjs": "code", ".cjs": "code",
    ".go": "code", ".rs": "code", ".java": "code", ".kt": "code", ".scala": "code", ".c": "code",
    ".h": "code", ".cc": "code", ".cpp": "code", ".hpp": "code", ".cs": "code", ".rb": "code",
    ".php": "code", ".sh": "code", ".bash": "code", ".zsh": "code", ".swift": "code", ".vue": "code",
    ".lua": "code", ".pl": "code", ".sql": "code",
    ".json": "json", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".rst": "markdown", ".txt": "markdown",
    ".log": "log", ".out": "log",
}


class HeldRange(Protocol):
    display: str
    held_start: int
    held_end: int

    @property
    def lines(self) -> list[str]: ...


@dataclass(frozen=True)
class Row:
    """One map row. `kind` is `def` for definition rows (the only rows `outline` keeps)."""

    start: int
    end: int | None
    depth: int
    text: str
    kind: str = "def"
    name: str = ""
    qualified: str = ""
    enclosing: bool = False
    continues: bool = False
    header_end: int | None = None  # last line whose text the row shows (defaults to `end`)

    @property
    def lines(self) -> range:
        return range(self.start, (self.end or self.start) + 1)

    @property
    def shown(self) -> range:
        return range(self.start, (self.header_end if self.header_end is not None else (self.end or self.start)) + 1)


# Rows every rung keeps; `outline` drops the symbols-only detail rows.
SYMBOLS_ONLY_KINDS = frozenset({"import", "constant", "decorator", "note"})


@dataclass
class Outline:
    parser: str
    rows: list[Row] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    docstring_lines: list[int] = field(default_factory=list)
    shown_lines: set[int] = field(default_factory=set)
    definitions: int = 0
    reread_symbol: str | None = None
    line_endings_mixed: bool = False
    applicable: bool = True
    keep_order: bool = False  # log maps are block-ordered (head, tail, errors), not line-sorted
    held_start: int = 1
    held_end: int = 0

    @property
    def omitted_record(self) -> str:
        """Complete record of what the map does not show (bodies, docstrings, mixed endings)."""
        omitted = [n for n in range(self.held_start, self.held_end + 1)
                   if n not in self.shown_lines and n not in self.docstring_lines and n not in self._blank]
        parts: list[str] = []
        if self.definitions or omitted:
            parts.append(f"bodies of {self.definitions} definitions (lines {_format_lines(omitted)})")
        if self.docstring_lines:
            parts.append("docstrings at " + ", ".join(str(n) for n in sorted(set(self.docstring_lines))))
        if self.line_endings_mixed:
            parts.append("line_endings=mixed")
        return ", ".join(parts) if parts else "nothing"

    _blank: set[int] = field(default_factory=set, repr=False)


def _format_lines(numbers: list[int]) -> str:
    if not numbers:
        return "none"
    ranges: list[list[int]] = []
    for n in sorted(set(numbers)):
        if ranges and n == ranges[-1][1] + 1:
            ranges[-1][1] = n
        else:
            ranges.append([n, n])
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


# --- language ------------------------------------------------------------------------------

def detect_language(display: str, text: str) -> str:
    """Extension, then shebang sniff, else `text`."""
    suffix = PurePosixPath(display).suffix.lower()
    if suffix in _EXTENSIONS:
        return _EXTENSIONS[suffix]
    first = text.split("\n", 1)[0]
    match = _SHEBANG.match(first)
    if match:
        return "python" if match.group(1) == "python" else "code"
    return "text"


def looks_binary(lines: list[str]) -> bool:
    text = "".join(lines)
    if not text:
        return False
    if "\x00" in text:
        return True
    replaced = text.count("�")
    return replaced > 0 and replaced * 20 > len(text)


def _mixed_endings(lines: list[str]) -> bool:
    crlf = any(line.endswith("\r\n") for line in lines)
    lf = any(line.endswith("\n") and not line.endswith("\r\n") for line in lines)
    return crlf and lf


def _blank_lines(held: HeldRange) -> set[int]:
    return {held.held_start + i for i, line in enumerate(held.lines) if not line.strip()}


# --- entry ---------------------------------------------------------------------------------

def outline_for(held: HeldRange, *, language: str | None = None, file_text: str | None = None) -> Outline | None:
    """Structure map of the held range, or None when no map applies (binary, empty)."""
    lines = held.lines
    if not lines or looks_binary(lines):
        return None
    language = language or detect_language(held.display, "".join(lines))
    builders = {
        "python": _python_outline,
        "code": _generic_outline,
        "json": _json_outline,
        "toml": _toml_outline,
        "yaml": _yaml_outline,
        "markdown": _markdown_outline,
        "log": _log_outline,
        "text": _text_outline,
    }
    outline = builders.get(language, _text_outline)(held, file_text)
    if outline is None:
        return None
    outline.held_start, outline.held_end = held.held_start, held.held_end
    outline._blank = _blank_lines(held)
    outline.line_endings_mixed = _mixed_endings(lines)
    if not outline.keep_order:
        outline.rows.sort(key=lambda row: (not row.enclosing, row.start, row.depth))
    for row in outline.rows:
        if not row.enclosing and row.kind not in ("key", "note"):  # key trees show no source lines
            outline.shown_lines.update(n for n in row.shown if held.held_start <= n <= held.held_end)
    if outline.applicable and not outline.rows:
        outline.applicable = False
    return outline


# --- python ---------------------------------------------------------------------------------

def _python_outline(held: HeldRange, file_text: str | None) -> Outline | None:
    tree, offset, source_lines = _parse_python(held, file_text)
    if tree is None:
        fallback = _generic_outline(held, file_text)
        return fallback
    outline = Outline(parser="ast")
    a, b = held.held_start, held.held_end

    def line_text(n: int) -> str:
        """Exact source line `n`: held text inside the range, file text outside (enclosing rows)."""
        if a <= n <= b:
            return held.lines[n - a]
        return source_lines[n - 1 - offset] if 0 <= n - 1 - offset < len(source_lines) else ""

    def header_text(start: int, end: int) -> str:
        return " ".join(line_text(n).strip() for n in range(start, end + 1))

    def node_line(node: ast.AST) -> int:
        return int(getattr(node, "lineno", 0) or 0) + offset

    def node_end(node: ast.AST) -> int:
        return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0) + offset

    def header_end(node: ast.AST) -> int:
        start = node_line(node)
        parts: list[ast.AST] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            parts.extend(args.posonlyargs + args.args + args.kwonlyargs)
            parts.extend(arg for arg in (args.vararg, args.kwarg) if arg is not None)
            parts.extend(d for d in args.defaults + [k for k in args.kw_defaults if k is not None])
            if node.returns is not None:
                parts.append(node.returns)
        elif isinstance(node, ast.ClassDef):
            parts.extend(node.bases)
            parts.extend(node.keywords)
        end = max([start] + [node_end(part) for part in parts])
        body_start = node_line(node.body[0]) if getattr(node, "body", None) else end
        if body_start <= start:
            return start
        # A closing paren / colon on its own line belongs to the header.
        while end < body_start - 1 and not line_text(end).strip().endswith(":"):
            end += 1
        return min(end, body_start - 1) if body_start - 1 >= start else start

    def docstring_lines(node: ast.AST) -> list[int]:
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            return [n for n in range(node_line(body[0]), node_end(body[0]) + 1) if a <= n <= b]
        return []

    first_symbol: str | None = None
    first_any: str | None = None

    def visit(node: ast.AST, depth: int, stack: tuple[str, ...]) -> None:
        nonlocal first_symbol, first_any
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start, end = node_line(child), node_end(child)
                if end < a or start > b:
                    continue
                qualified = ".".join(stack + (child.name,))
                if start < a:
                    outline.rows.append(Row(start, None, depth, header_text(start, header_end(child)),
                                            name=child.name, qualified=qualified, enclosing=True))
                else:
                    h_end = header_end(child)
                    for decorator in child.decorator_list:
                        d_start, d_end = node_line(decorator), node_end(decorator)
                        if a <= d_start <= b:
                            outline.rows.append(Row(d_start, min(d_end, b), depth, header_text(d_start, min(d_end, b)),
                                                    kind="decorator", name=child.name, qualified=qualified))
                    outline.rows.append(Row(start, min(end, b), depth, header_text(start, min(h_end, b)),
                                            name=child.name, qualified=qualified, continues=end > b,
                                            header_end=min(h_end, b)))
                    outline.definitions += 1
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if first_any is None:
                            first_any = qualified
                        if first_symbol is None and not (child.name.startswith("__") and child.name.endswith("__")):
                            first_symbol = qualified
                    elif first_any is None:
                        first_any = qualified
                outline.docstring_lines.extend(docstring_lines(child))
                visit(child, depth + 1, stack + (child.name,))
            elif isinstance(node, ast.Module) and isinstance(child, (ast.Import, ast.ImportFrom)):
                start, end = node_line(child), node_end(child)
                if a <= start <= b:
                    outline.rows.append(Row(start, min(end, b), depth, header_text(start, min(end, b)), kind="import"))
            elif isinstance(node, ast.Module) and isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                if all(isinstance(t, ast.Name) for t in targets):
                    start, end = node_line(child), node_end(child)
                    if a <= start <= b:
                        outline.rows.append(Row(start, min(end, b), depth, line_text(start).strip(), kind="constant",
                                                name=", ".join(t.id for t in targets if isinstance(t, ast.Name)),
                                                header_end=start))
            else:
                visit(child, depth, stack)

    outline.docstring_lines.extend(docstring_lines(tree))
    visit(tree, 0, ())
    outline.facts = _python_facts(tree, offset, a, b)
    outline.reread_symbol = first_symbol or first_any
    return outline


_TERMINATORS = ("\r\n", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def _lf_joined(units: list[str]) -> str:
    """One LF per delivered unit, so `ast` line k is delivered line k even around form feeds / lone CRs."""
    out = []
    for unit in units:
        for terminator in _TERMINATORS:
            if unit.endswith(terminator):
                unit = unit[:-len(terminator)]
                break
        out.append(unit + "\n")
    return "".join(out)


def _parse_python(held: HeldRange, file_text: str | None) -> tuple[ast.AST | None, int, list[str]]:
    """Parse the whole file when given, else the held slice (dedent retry). Returns (tree, line offset, lines)."""
    if file_text is not None:
        file_lines = file_text.splitlines(keepends=True)
        try:
            return ast.parse(_lf_joined(file_lines)), 0, file_lines
        except (SyntaxError, ValueError):
            pass
    source = _lf_joined(held.lines)
    offset = held.held_start - 1
    for candidate in (source, textwrap.dedent(source)):
        try:
            return ast.parse(candidate), offset, held.lines
        except (SyntaxError, ValueError):
            continue
    return None, offset, held.lines


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _python_facts(tree: ast.AST, offset: int, a: int, b: int) -> list[str]:
    """Typed facts the parser established, in source order, capped at FACT_CAP."""
    facts: list[tuple[int, int, str]] = []
    module_level_imports = {id(n) for n in getattr(tree, "body", []) if isinstance(n, (ast.Import, ast.ImportFrom))}
    for node in ast.walk(tree):
        line = int(getattr(node, "lineno", 0) or 0) + offset
        col = int(getattr(node, "col_offset", 0) or 0)
        if not (a <= line <= b):
            continue
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                facts.append((line, col, f"call={name} at={line}"))
        elif isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            name = _dotted(target)
            if name:
                facts.append((line, col, f"raise={name} at={line}"))
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in module_level_imports:
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "."]
            else:
                names = [alias.name for alias in node.names]
            for name in names:
                facts.append((line, col, f"import={name} at={line}"))
    facts.sort(key=lambda item: (item[0], item[1], item[2]))
    return [text for _, _, text in facts[:FACT_CAP]]


# --- generic code -----------------------------------------------------------------------------

def _indent_width(text: str) -> int:
    return len(text.expandtabs(4)) - len(text.expandtabs(4).lstrip(" "))


def _generic_outline(held: HeldRange, file_text: str | None) -> Outline:
    outline = Outline(parser="scan")
    lines = held.lines
    a = held.held_start
    open_defs: list[tuple[int, int, int]] = []  # (indent, row index, brace depth at open)
    first_symbol: str | None = None
    for i, raw in enumerate(lines):
        n = a + i
        stripped = raw.strip()
        if not stripped:
            continue
        if _GENERIC_IMPORT.match(raw):
            outline.rows.append(Row(n, n, 0, stripped[:GENERIC_SIGNATURE_WIDTH], kind="import"))
            continue
        match = _GENERIC_DEF.match(raw) or _GENERIC_ARROW.match(raw)
        if match is None:
            continue
        indent = _indent_width(match.group("indent"))
        depth = sum(1 for open_indent, _, _ in open_defs if open_indent < indent)
        name = match.groupdict().get("name") or ""
        kind = match.groupdict().get("kind") or "fn"
        end = _generic_block_end(lines, i, indent) + a
        outline.rows.append(Row(n, min(end, held.held_end), depth, stripped[:GENERIC_SIGNATURE_WIDTH],
                                name=name, qualified=name, continues=False, header_end=n))
        outline.definitions += 1
        if first_symbol is None and name and kind not in ("const", "let", "var", "val"):
            first_symbol = name
        open_defs = [item for item in open_defs if item[0] < indent] + [(indent, len(outline.rows) - 1, 0)]
    outline.reread_symbol = first_symbol
    if not any(row.kind == "def" for row in outline.rows) and len(lines) > LOG_THRESHOLD_LINES:
        return _log_outline(held, file_text)
    return outline


def _generic_block_end(lines: list[str], index: int, indent: int) -> int:
    """0-based index of the block's last line: brace balance when braces open, else indentation."""
    depth = 0
    saw_brace = False
    for j in range(index, len(lines)):
        text = lines[j]
        opens, closes = text.count("{"), text.count("}")
        if opens or closes:
            saw_brace = True
        depth += opens - closes
        if saw_brace and depth <= 0 and j >= index and (opens or closes):
            return j
        if j > index and not saw_brace:
            stripped = text.strip()
            if stripped and not stripped.startswith(("#", "//")) and _indent_width(text) <= indent:
                return j - 1
    return len(lines) - 1


# --- data files ---------------------------------------------------------------------------------

def _type_label(value: object) -> str:
    if isinstance(value, dict):
        return f"map[{len(value)}]"
    if isinstance(value, list):
        inner = _type_label(value[0]).split("[")[0] if value else "empty"
        return f"list[{len(value)}] of {inner}"
    if isinstance(value, str):
        return f"str({len(value)})" if len(value) > 32 else "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return type(value).__name__


def _key_tree(value: object, held: HeldRange) -> list[Row]:
    rows: list[Row] = []
    if not isinstance(value, dict):
        rows.append(Row(held.held_start, held.held_end, 0, f"root: {_type_label(value)}", kind="key"))
        return rows
    for key in sorted(value, key=str):
        child = value[key]
        rows.append(Row(held.held_start, held.held_end, 0, f"{key}: {_type_label(child)}", kind="key"))
        if isinstance(child, dict):
            for sub in sorted(child, key=str):
                rows.append(Row(held.held_start, held.held_end, 1, f"{sub}: {_type_label(child[sub])}", kind="key"))
    return rows


def _json_outline(held: HeldRange, file_text: str | None) -> Outline:
    text = "".join(held.lines)
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as exc:
        outline = _text_outline(held, file_text)
        outline.parser = "json"
        outline.rows.insert(0, Row(held.held_start, held.held_start, 0, f"parse_error: {_one_line(str(exc))}", kind="note"))
        return outline
    outline = Outline(parser="json")
    outline.rows = _key_tree(value, held)
    return outline


def _one_line(text: str) -> str:
    return " ".join(text.split())[:GENERIC_SIGNATURE_WIDTH]


def _toml_outline(held: HeldRange, file_text: str | None) -> Outline:
    text = "".join(held.lines)
    try:
        import tomllib
        value = tomllib.loads(text)
    except Exception:  # ImportError below 3.11 or a parse error on a partial slice
        outline = Outline(parser="scan")
        for i, raw in enumerate(held.lines):
            n = held.held_start + i
            stripped = raw.strip()
            if _TOML_TABLE.match(stripped):
                outline.rows.append(Row(n, n, 0, stripped[:GENERIC_SIGNATURE_WIDTH], kind="key"))
            elif _TOML_KEY.match(stripped):
                outline.rows.append(Row(n, n, 1, stripped.split("=", 1)[0].strip(), kind="key"))
        return outline
    outline = Outline(parser="tomllib")
    outline.rows = _key_tree(value, held)
    return outline


def _yaml_outline(held: HeldRange, file_text: str | None) -> Outline:
    outline = Outline(parser="scan")
    lines = held.lines
    nested_indent: int | None = None
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\r\n")
        match = _YAML_KEY.match(raw)
        i += 1
        if match is None or raw.lstrip().startswith("#"):
            continue
        indent = len(match.group(1).expandtabs(2))
        if indent == 0:
            nested_indent = None
        elif nested_indent is None:
            nested_indent = indent
        elif indent != nested_indent:
            continue
        depth = 0 if indent == 0 else 1
        key, rest = match.group(2).strip(), match.group(3).strip()
        if rest and not rest.startswith("#"):
            value = "scalar"
        else:
            items = 0
            j = i
            while j < len(lines):
                nxt = lines[j].rstrip("\r\n")
                if not nxt.strip():
                    j += 1
                    continue
                if len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                if nxt.lstrip().startswith("- "):
                    items += 1
                j += 1
            value = f"list[{items}]" if items else "map"
        n = held.held_start + i - 1
        outline.rows.append(Row(n, n, depth, f"{key}: {value}", kind="key"))
    return outline


def _markdown_outline(held: HeldRange, file_text: str | None) -> Outline:
    outline = Outline(parser="scan")
    lines = held.lines
    for i, raw in enumerate(lines):
        text = raw.rstrip("\r\n")
        n = held.held_start + i
        if _MD_HEADING.match(text):
            level = len(text) - len(text.lstrip("#"))
            outline.rows.append(Row(n, n, level - 1, text.strip(), kind="heading"))
        elif i + 1 < len(lines) and text.strip() and re.fullmatch(r"(=+|-+)\s*", lines[i + 1].rstrip("\r\n")) \
                and len(lines[i + 1].strip()) >= 3:
            depth = 0 if lines[i + 1].strip().startswith("=") else 1
            outline.rows.append(Row(n, n + 1, depth, text.strip(), kind="heading"))
    if not outline.rows:
        return _log_outline(held, file_text) if len(lines) > LOG_THRESHOLD_LINES else outline
    return outline


def _text_outline(held: HeldRange, file_text: str | None) -> Outline:
    outline = _generic_outline(held, file_text)
    if outline.rows or outline.parser == "log":
        return outline
    if len(held.lines) > LOG_THRESHOLD_LINES:
        return _log_outline(held, file_text)
    return outline


def _log_outline(held: HeldRange, file_text: str | None) -> Outline:
    outline = Outline(parser="log", keep_order=True)
    lines = held.lines
    a = held.held_start

    def row(n: int, text: str, depth: int) -> Row:
        return Row(n, n, depth, f"L{n} {text.rstrip(chr(10)).rstrip(chr(13))[:GENERIC_SIGNATURE_WIDTH]}", kind="log")

    outline.rows.append(Row(a, a + min(LOG_HEAD, len(lines)) - 1, 0, "head", kind="block"))
    for i in range(min(LOG_HEAD, len(lines))):
        outline.rows.append(row(a + i, lines[i], 1))
    tail_start = max(LOG_HEAD, len(lines) - LOG_TAIL)
    if tail_start < len(lines):
        outline.rows.append(Row(a + tail_start, a + len(lines) - 1, 0, "tail", kind="block"))
        for i in range(tail_start, len(lines)):
            outline.rows.append(row(a + i, lines[i], 1))
    errors: list[tuple[int, str, int]] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(lines):
        if not _ERROR_LINE.search(raw):
            continue
        key = raw.rstrip("\r\n")
        if key in seen:
            errors[seen[key]] = (errors[seen[key]][0], key, errors[seen[key]][2] + 1)
            continue
        if len(errors) >= LOG_ERRORS:
            continue
        seen[key] = len(errors)
        errors.append((a + i, key, 1))
    if errors:
        outline.rows.append(Row(errors[0][0], errors[-1][0], 0, "errors", kind="block"))
        for n, text, count in errors:
            suffix = f" (x{count})" if count > 1 else ""
            outline.rows.append(Row(n, n, 1, f"L{n} {text[:GENERIC_SIGNATURE_WIDTH]}{suffix}", kind="log"))
    outline.reread_symbol = None
    return outline
