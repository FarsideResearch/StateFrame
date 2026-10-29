"""Opt-in, task-named definition read pointers for StateFrame."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

ENV = "STATEFRAME_READ_POINTER"
MODE_ENV = "STATEFRAME_READ_POINTER_MODE"
INLINE_ENV = "STATEFRAME_READ_POINTER_INLINE"
EDIT_MENU_ENV = "STATEFRAME_READ_POINTER_EDIT_MENU"
EDIT_MENU_TURNS_ENV = "STATEFRAME_READ_POINTER_EDIT_MENU_TURNS"
WIRE_MARKER = "[kernel] TASK.md names"
INLINE_MARKER = "[kernel] TASK.md definitions (current source)"
INLINE_FINAL_INSTRUCTION = (
    "You have now seen every file TASK.md names. Your next action should be an edit "
    "(str_replace) to a file you have already read. Do not re-read."
)
MAX_INLINE_DEFINITIONS = 3
MAX_INLINE_LINES = 90
MAX_INLINE_SOURCE_BYTES = 1_000_000
MAX_EDIT_MENU_TURNS = 8
_FORBIDDEN_PARTS = frozenset({
    "evaluator", "grader", "graders", "gold", "hidden", "hidden_tests",
    "private", "backups",
})
_INLINE_FORBIDDEN_PARTS = _FORBIDDEN_PARTS | frozenset({
    "eval", "evals", "evaluation", "evaluations", "spec", "specs",
    "specification", "specifications", "test", "tests", "testing",
    "testdata", "__tests__",
})
_SYMBOL = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){0,7}\Z")
_BACKTICKED = re.compile(r"\x60([^\x60]{1,240})\x60")
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".go", ".rs", ".java",
})
_REPO_CACHE_ENV = "STATEFRAME_REPO_CACHE_DIR"
_REPO_CACHE_ROOT = Path(tempfile.gettempdir()) / "stateframe" / "read-pointer-repo-intel"
_REPO_INTEL_LOCK = threading.RLock()


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(ENV, "off") or "off").strip().lower() == "on"


def inline_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return enabled(values) and str(
        values.get(INLINE_ENV, "off") or "off"
    ).strip().lower() == "on"


def edit_menu_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    if not enabled(values):
        return False
    configured = values.get(EDIT_MENU_ENV)
    if configured is None:
        return inline_enabled(values)
    setting = str(configured or "").strip().lower()
    if setting == "on":
        return True
    if setting == "off":
        return False
    return False


def edit_menu_should_arm(
    *,
    inline_delivered: bool,
    source_definitions_read: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if environ is None else environ
    if not edit_menu_enabled(values):
        return False
    configured = values.get(EDIT_MENU_ENV)
    if configured is None:
        return inline_enabled(values) and inline_delivered
    if str(configured or "").strip().lower() == "on":
        return inline_delivered or source_definitions_read
    return False


def edit_menu_turn_limit(environ: Mapping[str, str] | None = None) -> int:
    values = os.environ if environ is None else environ
    raw = str(values.get(EDIT_MENU_TURNS_ENV, str(MAX_EDIT_MENU_TURNS)) or "").strip()
    try:
        requested = int(raw, 10)
    except (TypeError, ValueError):
        requested = MAX_EDIT_MENU_TURNS
    return max(1, min(MAX_EDIT_MENU_TURNS, requested))


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(MODE_ENV, "once") or "once").strip().lower()
    return value if value in {"once", "until_read"} else "once"


def extract_symbols(task_text: object) -> tuple[str, ...]:
    """Return bounded code-shaped task identifiers, preserving backtick syntax."""
    if not isinstance(task_text, str) or not task_text:
        return ()
    text = task_text[:20000]
    token_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:\./)?(?:[A-Za-z0-9_.-]+/)+"
        r"[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|vue|go|rs|java)(?![A-Za-z0-9_])"
        r"|(?<![A-Za-z0-9_/])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){0,7}(?:\(\))?"
        r"(?![A-Za-z0-9_])"
    )
    candidates: list[tuple[int, str, bool]] = [
        (match.start(), match.group(1).strip(), True)
        for match in _BACKTICKED.finditer(text)
    ]
    candidates.extend((match.start(), match.group(0).strip(), False)
                      for match in token_pattern.finditer(text))
    found: list[str] = []
    for _position, raw, explicit in sorted(candidates, key=lambda item: item[0]):
        symbol = raw.removeprefix("./")
        path_like = "/" in symbol and Path(symbol).suffix.lower() in {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".go", ".rs", ".java",
        }
        lookup = symbol[:-2] if symbol.endswith("()") else symbol
        if path_like:
            shaped = True
        else:
            shaped = (
                "_" in lookup or "." in lookup
                or bool(re.search(r"[a-z][A-Z]", lookup))
                or raw.endswith("()")
            )
        valid = path_like or bool(_SYMBOL.fullmatch(lookup))
        if (not valid or (not explicit and not shaped)
                or lookup in found):
            continue
        found.append(lookup)
        if len(found) >= 16:
            break
    return tuple(found)


def _definition_symbols(task_text: object) -> tuple[str, ...]:
    """Keep resolvable identifiers while leaving file paths out of definition lists."""
    return tuple(
        symbol for symbol in extract_symbols(task_text)
        if Path(symbol).suffix.lower() not in _SOURCE_EXTENSIONS
    )


def _definition_line_bounds(
    definition: Mapping[str, object],
) -> tuple[int, int | None] | None:
    line_value = definition.get("lineno", definition.get("line"))
    if isinstance(line_value, bool):
        return None
    try:
        line = int(line_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if line < 1:
        return None
    end_value = definition.get("end_lineno", definition.get("end_line"))
    if end_value is None or isinstance(end_value, bool):
        return line, None
    try:
        end_line = int(end_value)
    except (TypeError, ValueError, OverflowError):
        return line, None
    return (line, end_line) if end_line >= line else (line, None)


def _host_definition(
    root: Path,
    symbol: str,
    *,
    path_allowed: Callable[[Path], bool] | None,
) -> dict[str, object] | None:
    try:
        from kernel_repo_intel import operation

        runtime = SimpleNamespace(owner=SimpleNamespace(root=root))
        # Keep the host parser cache outside read-only release workspaces.
        with _REPO_INTEL_LOCK:
            previous_cache = os.environ.get(_REPO_CACHE_ENV)
            os.environ[_REPO_CACHE_ENV] = str(_REPO_CACHE_ROOT)
            try:
                queries = [symbol]
                leaf = symbol.rsplit(".", 1)[-1]
                if leaf != symbol:
                    queries.append(leaf)
                payload = None
                for query in queries:
                    raw = operation(runtime, "where_defined", {"symbol": query})
                    candidate_payload = json.loads(raw) if isinstance(raw, str) else None
                    if (isinstance(candidate_payload, Mapping)
                            and candidate_payload.get("status") == "FOUND"):
                        candidate_definition = candidate_payload.get("definition")
                        if not isinstance(candidate_definition, Mapping):
                            continue
                        candidate_file = str(candidate_definition.get("file") or "").replace(
                            "\\", "/")
                        module_parts = symbol.split(".")[:-1]
                        if module_parts:
                            module_suffix = Path(*module_parts).with_suffix(".py").as_posix()
                            if not candidate_file.endswith(module_suffix):
                                continue
                        payload = candidate_payload
                        break
            finally:
                if previous_cache is None:
                    os.environ.pop(_REPO_CACHE_ENV, None)
                else:
                    os.environ[_REPO_CACHE_ENV] = previous_cache
        if not isinstance(payload, Mapping) or payload.get("status") != "FOUND":
            return None
        definition = payload.get("definition")
        if not isinstance(definition, Mapping):
            return None
        relative = str(definition.get("file") or "").replace("\\", "/")
        if not relative:
            return None
        candidate = (root / relative).resolve()
        resolved_root = root.resolve()
        rel = candidate.relative_to(resolved_root).as_posix()
        module_parts = symbol.split(".")[:-1]
        if module_parts and not rel.endswith(
                Path(*module_parts).with_suffix(".py").as_posix()):
            return None
        if any(part.casefold() in _FORBIDDEN_PARTS for part in Path(rel).parts):
            return None
        if not candidate.is_file() or (root / rel).is_symlink():
            return None
        if path_allowed is not None and not path_allowed(candidate):
            return None
        line_bounds = _definition_line_bounds(definition)
        if line_bounds is None:
            return None
        line, end_line = line_bounds
        return {
            "symbol": symbol,
            "path": rel,
            "line": line,
            "end_line": end_line,
            "kind": str(definition.get("kind") or "definition"),
            "signature": definition.get("signature"),
        }
    except (
        ImportError, OSError, RuntimeError, TypeError, ValueError,
        OverflowError, AttributeError, json.JSONDecodeError,
    ):
        return None


def _relative_definition(
    root: Path,
    symbol: str,
    *,
    path_allowed: Callable[[Path], bool] | None,
) -> dict[str, object] | None:
    definition = _host_definition(root, symbol, path_allowed=path_allowed)
    if definition is None:
        return None
    return {
        "symbol": definition["symbol"],
        "path": definition["path"],
        "line": definition["line"],
    }


def all_task_definitions_read(runtime: object, task_text: object) -> bool:
    """Return true only when every explicit task symbol resolved and was read."""
    symbols = _definition_symbols(task_text)
    state = getattr(runtime, "_kernel_read_pointer_state", None)
    items = state.get("items") if isinstance(state, Mapping) else None
    if not symbols or not isinstance(items, list):
        return False
    by_symbol = {
        str(item.get("symbol")): item
        for item in items if isinstance(item, Mapping)
        and str(item.get("symbol")) in symbols
    }
    return len(by_symbol) == len(symbols) and all(
        by_symbol.get(symbol, {}).get("read") is True for symbol in symbols
    )


def resolve_task_symbols(
    root: Path,
    task_text: object,
    *,
    path_allowed: Callable[[Path], bool] | None = None,
) -> tuple[dict[str, object], ...]:
    """Resolve only unique host where_defined results inside the authorized workspace."""
    results: list[dict[str, object]] = []
    for symbol in extract_symbols(task_text):
        definition = _relative_definition(root, symbol, path_allowed=path_allowed)
        if definition is not None:
            results.append(definition)
    return tuple(results)


def _inline_path_allowed(relative: str) -> bool:
    for part in Path(relative).parts:
        folded = part.casefold()
        tokens = {token for token in re.split(r"[._-]+", folded) if token}
        if (folded.startswith(".")
                or tokens.intersection(_INLINE_FORBIDDEN_PARTS)):
            return False
    return True


def _format_signature(definition: Mapping[str, object]) -> str:
    symbol = str(definition.get("symbol") or "")
    name = symbol.rsplit(".", 1)[-1]
    kind = str(definition.get("kind") or "definition")
    signature = definition.get("signature")
    if isinstance(signature, str) and signature.strip():
        return signature.strip()
    if not isinstance(signature, Mapping) or kind == "class":
        return f"{kind} {name}"

    positional = [
        str(value) for value in (*signature.get("posonly", ()), *signature.get("args", ()))
        if str(value)
    ]
    defaults = signature.get("defaults", 0)
    if isinstance(defaults, int) and not isinstance(defaults, bool) and defaults > 0:
        for index in range(max(0, len(positional) - defaults), len(positional)):
            positional[index] += "=…"
    if signature.get("has_star_args"):
        positional.append("*args")
    positional.extend(str(value) for value in signature.get("kwonly", ()) if str(value))
    if signature.get("has_star_kwargs"):
        positional.append("**kwargs")
    return f"{kind} {name}({', '.join(positional)})"


def _python_definition_end_line(
    source: str,
    symbol: str,
    start_line: int,
) -> int | None:
    """Recover a Python definition's end when repo-intel has only its start."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    name = symbol.rsplit(".", 1)[-1]
    matches: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != name:
            continue
        node_start = min(
            [int(node.lineno), *(int(item.lineno) for item in node.decorator_list)]
        )
        if node_start == start_line or int(node.lineno) == start_line:
            end_line = getattr(node, "end_lineno", None)
            if isinstance(end_line, int) and end_line >= start_line:
                matches.append(end_line)
    return matches[0] if len(matches) == 1 else None


def _bounded_python_definition_source(
    path: Path,
    symbol: str,
    start_line: int,
) -> tuple[int, str] | None:
    with path.open("rb") as source_file:
        raw_source = source_file.read(MAX_INLINE_SOURCE_BYTES + 1)
    if len(raw_source) > MAX_INLINE_SOURCE_BYTES:
        return None
    source = raw_source.decode("utf-8")
    end_line = _python_definition_end_line(source, symbol, start_line)
    return (end_line, source) if end_line is not None else None


def _definition_source_lines(
    path: Path,
    start_line: int,
    end_line: int,
    source: str | None,
) -> list[str] | None:
    if source is not None:
        lines = source.splitlines()[start_line - 1:end_line]
    else:
        lines = []
        with path.open("r", encoding="utf-8", newline="") as source_file:
            for number, line in enumerate(source_file, start=1):
                if number > end_line:
                    break
                if number >= start_line:
                    lines.append(line.rstrip("\r\n"))
    return lines if len(lines) == end_line - start_line + 1 else None


def _inline_definition_block(
    root: Path,
    definition: Mapping[str, object],
) -> str | None:
    relative = str(definition.get("path") or "")
    if not relative or not _inline_path_allowed(relative):
        return None
    try:
        target = root / relative
        if target.is_symlink():
            return None
        resolved = target.resolve()
        resolved.relative_to(root.resolve())
        start = int(definition["line"])
        if start < 1:
            return None
        end_value = definition.get("end_line")
        inline_source: str | None = None
        if end_value is None:
            recovered = _bounded_python_definition_source(
                resolved, str(definition.get("symbol") or ""), start
            )
            if recovered is None:
                return None
            end_value, inline_source = recovered
        end = int(end_value)
        if end < start:
            return None
        signature = _format_signature(definition)
        label = f"{definition['symbol']} ({relative})"
        if end - start + 1 > MAX_INLINE_LINES:
            return f"{label}\nSignature: {signature}\nlines {start}-{end}"
        lines = _definition_source_lines(resolved, start, end, inline_source)
        if lines is None:
            return None
        return f"{label}, lines {start}-{end}\n" + "\n".join(lines)
    except (OSError, RuntimeError, UnicodeError, TypeError, ValueError, KeyError):
        return None


def _build_inline_notice(
    root: Path,
    task_text: object,
    items: list[dict[str, object]],
    *,
    path_allowed: Callable[[Path], bool] | None,
) -> str | None:
    symbols = _definition_symbols(task_text)
    if not symbols or len(symbols) > MAX_INLINE_DEFINITIONS:
        return None
    resolved_items = {
        str(item.get("symbol")): item for item in items
        if isinstance(item, Mapping) and str(item.get("symbol")) in symbols
    }
    if len(resolved_items) != len(symbols) or any(
        symbol not in resolved_items for symbol in symbols
    ):
        return None

    blocks: list[str] = []
    for symbol in symbols:
        definition = _host_definition(root, symbol, path_allowed=path_allowed)
        if definition is None:
            return None
        block = _inline_definition_block(root, definition)
        if block is None:
            return None
        blocks.append(block)
    return (
        INLINE_MARKER + "\n" + "\n\n".join(blocks) + "\n"
        + INLINE_FINAL_INSTRUCTION
    )


def _line_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 1 else None


def _read_interval(delivery: object) -> tuple[int, int] | None:
    if not isinstance(delivery, Mapping):
        return None
    effective = delivery.get("effective_range")
    values = effective if isinstance(effective, Mapping) else delivery
    start = _line_number(values.get("start_line"))
    end = _line_number(values.get("end_line"))
    if end is None:
        end = _line_number(delivery.get("selected_end_line"))
    if start is None:
        start = _line_number(delivery.get("start_line"))
    return (start, end) if start is not None and end is not None and end >= start else None


def _read_path(value: object) -> str | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    return str(value).replace("\\", "/").strip("/")


def _unread_key(items: list[dict[str, object]]) -> tuple[tuple[str, str, int], ...]:
    return tuple(sorted(
        (str(item["symbol"]), str(item["path"]), int(item["line"]))
        for item in items if not item.get("read")
    ))


def _render_notice(items: list[dict[str, object]]) -> str:
    labels = []
    for item in items:
        symbol = str(item["symbol"])
        path = str(item["path"])
        line = int(item["line"])
        labels.append(f"{chr(96)}{symbol}{chr(96)} ({path}, def at line {line})")
    next_item = next(item for item in items if not item.get("read"))
    next_path = str(next_item["path"])
    next_line = int(next_item["line"])
    start = max(1, next_line - 20)
    end = next_line + 70
    notice = f"{WIRE_MARKER} " + " and ".join(labels)
    notice += f". Next step: read {next_path} lines {start}..{end}."
    read_paths = sorted({str(item["path"]) for item in items if item.get("read")})
    for path in read_paths:
        notice += f" {path} has already been read."
    return notice


def observe_tool_result(
    runtime: object,
    observation: str,
    *,
    root: Path,
    task_text: object,
    path_allowed: Callable[[Path], bool] | None = None,
    read_path: object = None,
    read_delivery: object = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Append current-source guidance to the same native tool result."""
    if not enabled(environ) or not isinstance(observation, str):
        return observation
    try:
        state = getattr(runtime, "_kernel_read_pointer_state", None)
        if not isinstance(state, dict):
            state = {
                "items": list(resolve_task_symbols(
                    root, task_text, path_allowed=path_allowed)),
                "read": set(),
                "fired": set(),
            }
            setattr(runtime, "_kernel_read_pointer_state", state)
        items = state.get("items")
        if not isinstance(items, list) or not items:
            return observation
        relative = _read_path(read_path)
        interval = _read_interval(read_delivery)
        if relative is not None and interval is not None:
            start, end = interval
            for item in items:
                if (str(item.get("path")) == relative
                        and start <= int(item.get("line", 0)) <= end):
                    state["read"].add((
                        str(item["symbol"]), str(item["path"]), int(item["line"])))
                    item["read"] = True
        unread_key = _unread_key(items)
        if not unread_key:
            return observation
        if mode(environ) == "once":
            fired = state.get("fired")
            if not isinstance(fired, set):
                fired = set()
                state["fired"] = fired
            if unread_key in fired:
                return observation
            fired.add(unread_key)
        if WIRE_MARKER in observation:
            return observation
        inline_text = None
        if inline_enabled(environ):
            inline_text = _build_inline_notice(
                root, task_text, items, path_allowed=path_allowed
            )
        if inline_text is not None:
            for item in items:
                item["read"] = True
            state["inline_text"] = inline_text
            return (
                observation + "\n" + WIRE_MARKER
                + " Current source definitions follow.\n" + inline_text
            )
        return observation + "\n" + _render_notice(items)
    except (AttributeError, TypeError, ValueError, KeyError):
        return observation


def wire_receipt(wire: bytes) -> dict[str, object] | None:
    """Prove the instruction survived into the exact serialized request body."""
    if not enabled():
        return None
    try:
        payload = json.loads(wire)
        messages = payload.get("messages") if isinstance(payload, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(messages, list):
        return None
    indexes = [
        index for index, item in enumerate(messages)
        if isinstance(item, Mapping) and isinstance(item.get("content"), str)
        and WIRE_MARKER in item["content"]
    ]
    if not indexes:
        return None
    return {
        "message_indexes": indexes[:16],
        "occurrences": len(indexes),
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
    }


def inline_wire_receipt(wire: bytes) -> dict[str, object] | None:
    """Prove both the inline source block and its exact next-action text are on wire."""
    if not inline_enabled():
        return None
    try:
        payload = json.loads(wire)
        messages = payload.get("messages") if isinstance(payload, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(messages, list):
        return None
    indexes = [
        index for index, item in enumerate(messages)
        if isinstance(item, Mapping)
        and isinstance(item.get("content"), str)
        and INLINE_MARKER in item["content"]
        and INLINE_FINAL_INSTRUCTION in item["content"]
    ]
    if not indexes:
        return None
    return {
        "message_indexes": indexes[:16],
        "occurrences": len(indexes),
        "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest(),
    }
