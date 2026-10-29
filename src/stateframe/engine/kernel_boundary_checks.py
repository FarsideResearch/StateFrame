"""Public-material compiler for boundary checks boundary contract cells.

The compiler is intentionally conservative. It reads only a task brief,
declared public product files, and declared public tests. It never opens a
task directory, follows a private-manifest pointer, or treats the current
candidate as an oracle. The result is an ordinary:class:`CheckBundle`, so
the existing host executor and reducer remain the sole verdict path.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path, PurePosixPath

import kernel_parse_cache as _parse_cache
from kernel_live_projection import is_harness_path

from kernel_obligation_assess import (
    ADMITTED,
    APPLIES,
    APPLICABILITY_UNKNOWN,
    CheckBundle,
    CheckDefinition,
    ContractCell,
    ObligationSpec,
    PROVISIONAL,
    TEMPLATES,
    UNBOUND,
    VALIDATED,
    digest,
)
from kernel_check_generator_fixes import (
    check_is_gating, coverage_disposition, enabled as generator_fixes_enabled,
    enforce_generated_checks, normative_clause,
)


MODE_ENV = "STATEFRAME_BOUNDARY_CHECKS"
MODEL_SHAPE_MODE_ENV = "STATEFRAME_TOKENLAB_MODEL_SHAPE_GUARD"
SCHEMA = "stateframe-boundary-compiler/1"
MAX_FILES = 128
MAX_SOURCE_BYTES = 4_000_000
MAX_TEST_BYTES = 1_000_000
MAX_CELLS = 512
MAX_CHECKS = 256

_FORBIDDEN_PARTS = {
    "private", "hidden", "hidden_tests", "near_miss", "gold", "oracle",
    "grader", "graders", "grading", "harness",
}
_STRUCTURAL_KEYS = {"path", "json", "status", "body", "error"}
_FIELD_STOPWORDS = {"to", "result"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BACKTICK = re.compile(r"`([^`]+)`")
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_HTTP_STATUS = re.compile(r"\bHTTP\s+(\d{3})\b|\bstatus\s+(\d{3})\b", re.I)
_REPAIR_ERROR_STATUS = re.compile(
    r"\b(?:reject|refuse|return|respond)\b.*?\b(?:as\s+(?:an?\s+)?)?"
    r"(?:HTTP\s*)?((?:4\d\d|4xx))\b", re.I | re.S)


def model_shape_guard_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(MODEL_SHAPE_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODEL_SHAPE_MODE_ENV} must be off or on")
    return value == "on"


@dataclass(frozen=True, slots=True)
class SubjectRef:
    """Role-qualified public boundary evidence for one named subject."""

    boundary: str
    representation: str
    path: str
    name: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {
            "boundary": self.boundary,
            "representation": self.representation,
            "path": self.path,
            "name": self.name,
            "role": self.role,
        }


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _safe_path(value: object, *, test: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("public boundary material requires a path")
    path = value.replace("\\", "/")
    pure = PurePosixPath(path)
    if (pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path
            or "\x00" in path or (pure.parts and ":" in pure.parts[0])
            or pure.suffix != ".py" or any(part.startswith(".") for part in pure.parts)
            or any(part.lower() in _FORBIDDEN_PARTS or "backup" in part.lower()
                   for part in pure.parts)):
        raise ValueError("unsafe public boundary path")
    is_test_role = is_harness_path(Path(path))
    if test and not is_test_role:
        raise ValueError("public boundary tests must identify a test-role file")
    if not test and is_test_role:
        raise ValueError("public boundary product sources must not be harness files")
    return path


def _source_value(payload: object) -> str | None:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, Mapping):
        return None
    # fixed_src is deliberately excluded: it is an evaluation/oracle-side
    # field in repair rows, not public source material.
    for key in ("source", "src", "content", "parent_src"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _collect_files(value: object, *, test: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        items = ((value.get("path"), value),) if "path" in value \
            and _source_value(value) is not None else value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = ((item.get("path"), item) for item in value if isinstance(item, Mapping))
    else:
        items = ()
    total = 0
    for raw_path, payload in items:
        path = _safe_path(raw_path, test=test)
        source = _source_value(payload)
        if not isinstance(source, str) or not source.strip():
            continue
        if path in result:
            raise ValueError(f"duplicate public boundary file: {path}")
        total += len(source.encode("utf-8"))
        if total > (MAX_TEST_BYTES if test else MAX_SOURCE_BYTES):
            raise ValueError("public boundary material exceeds its size bound")
        _parse_cache.parse(source, path)
        result[path] = source
    if len(result) > MAX_FILES:
        raise ValueError("too many public boundary files")
    return dict(sorted(result.items()))


def _public_tests(value: object) -> dict[str, str]:
    if isinstance(value, str):
        return {"tests/test_public_boundary.py": value}
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        items = ((value.get("path"), value),) if "path" in value \
            and _source_value(value) is not None else value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = ((item.get("path"), item) for item in value if isinstance(item, Mapping))
    else:
        items = ()
    for raw_path, payload in items:
        path = _safe_path(raw_path, test=True)
        source = _source_value(payload)
        if not isinstance(source, str) or not source.strip():
            continue
        if path in result:
            raise ValueError(f"duplicate public boundary test: {path}")
        if len(source.encode("utf-8")) > MAX_TEST_BYTES:
            raise ValueError("public boundary test exceeds its size bound")
        _parse_cache.parse(source, path)
        result[path] = source
    return dict(sorted(result.items()))


def _split_public_files(value: object) -> tuple[object, object]:
    """Separate an optional mixed public ``files`` collection safely."""
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [(item.get("path"), item) for item in value if isinstance(item, Mapping)]
    else:
        return value, ()
    sources: dict[object, object] = {}
    tests: dict[object, object] = {}
    for path, payload in items:
        if isinstance(path, str) and is_harness_path(Path(path.replace("\\", "/"))):
            tests[path] = payload
        else:
            sources[path] = payload
    return sources, tests


def public_material_from_row(row: Mapping[str, object]) -> tuple[str, dict[str, str], dict[str, str]]:
    """Project a worker row to ``(brief, product_sources, public_tests)``.

    Only explicit public fields are inspected.  In particular, this function
    does not look at ``oracle`` or at any path-valued field that could point to
    a non-public task directory.
    """
    if not isinstance(row, Mapping):
        raise TypeError("public boundary row must be a mapping")
    brief = ""
    brief_keys = (("_bridge_ordinary_brief", "ordinary_brief", "brief", "msg",
                   "symptom", "clause") if generator_fixes_enabled() else
                  ("_bridge_ordinary_brief", "brief", "msg", "symptom", "clause"))
    for key in brief_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            brief = value
            break
    if not brief and isinstance(row.get("public_brief"), str):
        brief = str(row["public_brief"])
    raw_files = row.get("files", row.get("public_sources", ()))
    source_files, embedded_tests = _split_public_files(raw_files)
    sources = _collect_files(source_files, test=False)
    tests_value: object = row.get("public_tests", row.get("public_test_files", ()))
    tests = _public_tests(tests_value)
    if not tests:
        source = next((row.get(key) for key in (
            "public_test_source", "public_test_src", "test_source", "test_src")
                       if ((isinstance(row.get(key), str) and row.get(key).strip())
                           or (isinstance(row.get(key), Mapping)
                               and _source_value(row.get(key))))), None)
        if isinstance(source, str):
            path = row.get("public_test_path", row.get("test_path", "tests/test_public_boundary.py"))
            tests = _public_tests({str(path): source})
        elif isinstance(source, Mapping):
            path = source.get("path", row.get("public_test_path", "tests/test_public_boundary.py"))
            tests = _public_tests({str(path): source})
    # Some public manifests expose tests in ``files``.  They are accepted only
    # after the explicit public test collection, never from an oracle mapping.
    if not tests:
        tests = _public_tests(embedded_tests)
    return brief, sources, tests


@dataclass(frozen=True, slots=True)
class _Example:
    entry_point: str
    fixture: dict[str, object]
    expected: object
    selection: str
    line: int


@dataclass(frozen=True, slots=True)
class BoundaryCompilationReport:
    task_id: str
    bundle: CheckBundle | None
    bound_cells: tuple[ContractCell, ...]
    unknown_cells: tuple[ContractCell, ...]

    @property
    def bound(self) -> tuple[ContractCell, ...]:
        return self.bound_cells

    @property
    def unknown(self) -> tuple[ContractCell, ...]:
        return self.unknown_cells

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "task_id": self.task_id,
            "bound_cells": [cell.to_dict() for cell in self.bound_cells],
            "unknown_cells": [cell.to_dict() for cell in self.unknown_cells],
            "check_ids": list(self.bundle and [check.check_id for check in self.bundle.checks] or ()),
        }


def _non_json_literal(value: object) -> str | None:
    """Return the type name of the first non-JSON leaf, or None when strict JSON.

    Public material can contain Python-only literals (``...`` placeholders,
    bytes, sets, complex numbers).  They must never be bound into a contract
    cell or check: every cell is journaled with strict canonical JSON.
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        pass
    else:
        return None
    stack = [value]
    while stack:
        item = stack.pop()
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and item != item or item in (float("inf"), float("-inf")):
                return "non_finite_float"
            continue
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key is not None and not isinstance(key, (str, int, float, bool)):
                    return type(key).__name__
                stack.append(child)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            return type(item).__name__
    return "non_json"


def _module_for_path(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    # Repository-root paths such as ``backend/app/service.py`` still expose
    # the public package as ``app.service``.  The repo prefix is not part of
    # Python's product namespace and must not make public imports unbound.
    if "app" in parts:
        parts = parts[parts.index("app"):]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _canonical_module(name: str) -> str:
    """Drop repository prefixes while retaining the public ``app`` package."""
    parts = name.split(".")
    return ".".join(parts[parts.index("app"):]) if "app" in parts else name


def _literal(node: ast.AST, values: Mapping[str, object]) -> tuple[bool, object]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(node, ast.Name) and node.id in values:
            return True, copy.deepcopy(values[node.id])
        return False, None


def _imports(tree: ast.AST, module: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    package = module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                target_module = _canonical_module(item.name)
                if not (target_module == "app" or target_module.startswith("app.")):
                    continue
                aliases[item.asname or item.name.split(".")[-1]] = target_module
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: max(0, len(package) - node.level + 1)]
                target_module = ".".join((*base, *(node.module.split(".") if node.module else ())))
            else:
                target_module = _canonical_module(node.module or "")
                if not (target_module == "app" or target_module.startswith("app.")):
                    continue
            for item in node.names:
                if item.name == "*":
                    continue
                aliases[item.asname or item.name] = f"{target_module}.{item.name}"
    return aliases


def _expr_name(node: ast.AST, aliases: Mapping[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _entry_point(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    name = _expr_name(node, aliases)
    if not name.startswith("app."):
        return None
    parts = name.split(".")
    module_index = next((index for index, part in enumerate(parts[1:], 1)
                         if part in {"models", "service", "api", "cli", "parser", "merge",
                                     "money", "validation"}), None)
    if module_index is not None and module_index + 1 < len(parts):
        module = ".".join(parts[:module_index + 1])
        attribute = ".".join(parts[module_index + 1:])
    else:
        module, _, attribute = name.rpartition(".")
    if not module or not attribute:
        return None
    return f"{module}:{attribute}"


def _public_examples(tests: Mapping[str, str]) -> tuple[_Example, ...]:
    examples: list[_Example] = []
    seen: set[tuple[str, str]] = set()
    for path, source in tests.items():
        tree = _parse_cache.parse(source, path)
        aliases = _imports(tree, "tests." + PurePosixPath(path).stem)
        values: dict[str, object] = {}
        assignments: dict[str, tuple[str, dict[str, object], int]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                ok, value = _literal(node.value, values)
                if ok:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            values[target.id] = value
                if isinstance(node.value, ast.Call):
                    entry = _entry_point(node.value.func, aliases)
                    fixture = _fixture(node.value, values)
                    if entry and fixture is not None:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                assignments[target.id] = (entry, fixture, node.lineno)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            calls: list[tuple[ast.Call, str, dict[str, object]]] = []
            for candidate in ast.walk(node.test):
                if not isinstance(candidate, ast.Call):
                    continue
                entry = _entry_point(candidate.func, aliases)
                fixture = _fixture(candidate, values)
                if entry and fixture is not None:
                    calls.append((candidate, entry, fixture))
            expected: object = None
            if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                for side in (node.test.left, node.test.comparators[0]):
                    if isinstance(side, ast.Call):
                        continue
                    ok, value = _literal(side, values)
                    if ok:
                        expected = value
                        break
                    if isinstance(side, ast.Name) and side.id in assignments:
                        continue
            if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                for side in (node.test.left, node.test.comparators[0]):
                    if isinstance(side, ast.Name) and side.id in assignments:
                        entry, fixture, line = assignments[side.id]
                        expected = _comparison_literal(node.test, side, values)
                        calls.append((ast.Call(func=ast.Name(id="_assigned", ctx=ast.Load()),
                                               args=[], keywords=[]), entry, fixture))
                        calls[-1][0].lineno = line
                        break
            if calls:
                for call, entry, fixture in calls:
                    key = (entry, repr((fixture, expected)))
                    if key in seen:
                        continue
                    seen.add(key)
                    examples.append(_Example(entry, fixture, copy.deepcopy(expected), path,
                                             getattr(call, "lineno", node.lineno)))
        # Invocation witnesses are still useful for a public call without an
        # equality assertion, but do not invent an expected output.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            entry = _entry_point(node.func, aliases)
            fixture = _fixture(node, values)
            if not entry or fixture is None:
                continue
            key = (entry, repr((fixture, None)))
            if key in seen:
                continue
            seen.add(key)
            examples.append(_Example(entry, fixture, None, path, node.lineno))
    return tuple(examples)


def _comparison_literal(compare: ast.Compare, variable: ast.Name,
                        values: Mapping[str, object]) -> object:
    for side in (compare.left, *compare.comparators):
        if side is variable:
            continue
        ok, value = _literal(side, values)
        if ok:
            return value
    return None


def _fixture(call: ast.Call, values: Mapping[str, object]) -> dict[str, object] | None:
    args: list[object] = []
    for node in call.args:
        ok, value = _literal(node, values)
        if not ok:
            return None
        args.append(value)
    kwargs: dict[str, object] = {}
    for item in call.keywords:
        if item.arg is None:
            return None
        ok, value = _literal(item.value, values)
        if not ok:
            return None
        kwargs[item.arg] = value
    result: dict[str, object] = {"args": args, "kwargs": kwargs}
    if not kwargs:
        result.pop("kwargs")
    return result


def _function_signatures(sources: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for path, source in sources.items():
        module = _module_for_path(path)
        tree = _parse_cache.parse(source, path)
        # Direct-body owner per node; a later ClassDef in walk order wins, exactly
        # as the former per-function rescan's ``parents[-1]`` did (it was O(n^2)).
        owner: dict[int, str] = {}
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in parent.body:
                    owner[id(child)] = parent.name
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            prefix = module
            if id(node) in owner:
                prefix += "." + owner[id(node)]
            names = [item.arg for item in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            if node.args.vararg:
                names.append(node.args.vararg.arg)
            result[f"{prefix}.{node.name}"] = tuple(names)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                initializer = next((item for item in node.body
                                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and item.name == "__init__"), None)
                if initializer is None:
                    result[f"{module}.{node.name}"] = tuple(
                        item.target.id for item in node.body
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name))
                else:
                    result[f"{module}.{node.name}"] = tuple(
                        item.arg for item in (
                            *initializer.args.posonlyargs, *initializer.args.args,
                            *initializer.args.kwonlyargs,
                        ) if item.arg not in {"self", "cls"})
    return result


def _function_parameter_shapes(sources: Mapping[str, str]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Keep the starter's positional/keyword parameter shape for compatibility probes."""
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for path, source in sources.items():
        module = _module_for_path(path)
        tree = _parse_cache.parse(source, path)

        def shape(arguments: ast.arguments, *, drop_receiver: bool = False) -> dict[str, tuple[str, ...]]:
            posonly = tuple(item.arg for item in arguments.posonlyargs)
            positional = tuple(item.arg for item in arguments.args)
            keyword_only = tuple(item.arg for item in arguments.kwonlyargs)
            if drop_receiver:
                posonly = tuple(item for item in posonly if item not in {"self", "cls"})
                positional = tuple(item for item in positional if item not in {"self", "cls"})
            return {"posonly": posonly, "positional": positional,
                    "keyword_only": keyword_only}

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parent = next((candidate for candidate in ast.walk(tree)
                           if isinstance(candidate, ast.ClassDef)
                           and any(child is node for child in candidate.body)), None)
            prefix = f"{module}.{parent.name}." if parent is not None else f"{module}."
            result[prefix + node.name] = shape(node.args, drop_receiver=parent is not None)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            initializer = next((item for item in node.body
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and item.name == "__init__"), None)
            result[f"{module}.{node.name}"] = shape(
                initializer.args, drop_receiver=True) if initializer is not None else {
                    "posonly": (), "positional": (), "keyword_only": ()}
    return result


def _source_definitions(sources: Mapping[str, str]) -> set[str]:
    result: set[str] = set()
    for path, source in sources.items():
        module = _module_for_path(path)
        tree = _parse_cache.parse(source, path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                result.add(f"{module}.{node.name}")
                if isinstance(node, ast.ClassDef):
                    result.update(f"{module}.{node.name}.{item.name}"
                                  for item in node.body
                                  if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
                    node.value, (ast.Name, ast.Attribute, ast.Lambda)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                result.update(f"{module}.{target.id}" for target in targets
                              if isinstance(target, ast.Name))
    return result


def _public_symbol_entries(sources: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Return top-level public symbols with their source module."""
    entries: list[tuple[str, str]] = []
    for path, source in sources.items():
        module = _module_for_path(path)
        tree = _parse_cache.parse(source, path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                entries.append((module, node.name))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if isinstance(node.value, (ast.Name, ast.Attribute, ast.Lambda)):
                    entries.extend((module, target.id) for target in targets
                                   if isinstance(target, ast.Name))
    return tuple(entries)


def _source_input_keys(sources: Mapping[str, str]) -> tuple[str, ...]:
    """Collect literal mapping keys used by public product code."""
    found: set[str] = set()
    for path, source in sources.items():
        tree = _parse_cache.parse(source, path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                literal = node.slice.value if isinstance(node.slice, ast.Constant) else None
                if isinstance(literal, str) and _IDENTIFIER.fullmatch(literal):
                    found.add(literal)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and node.args:
                literal = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
                if isinstance(literal, str) and _IDENTIFIER.fullmatch(literal):
                    found.add(literal)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.update(item.arg for item in (
                    *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                ) if item.arg.endswith("_id") or item.arg in {"username", "handle"})
    return tuple(sorted(found))


def _reachable_surfaces(sources: Mapping[str, str], examples: Sequence[_Example]) -> dict[str, set[str]]:
    """Resolve a small, explicit Python call graph over public app sources."""
    definitions = _source_definitions(sources)
    nodes: dict[str, ast.AST] = {}
    aliases_by_module: dict[str, dict[str, str]] = {}
    for path, source in sources.items():
        module = _module_for_path(path)
        tree = _parse_cache.parse(source, path)
        aliases_by_module[module] = _imports(tree, module)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nodes[f"{module}.{node.name}"] = node
            elif isinstance(node, ast.ClassDef):
                nodes[f"{module}.{node.name}"] = node
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        nodes[f"{module}.{node.name}.{item.name}"] = item
    graph: dict[str, set[str]] = defaultdict(set)
    for name, node in nodes.items():
        module = name.rsplit(".", 1)[0] if "." in name else name
        aliases = aliases_by_module.get(module, {})
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            target = _expr_name(call.func, aliases)
            if target.startswith("app."):
                if target in definitions:
                    graph[name].add(target)
                else:
                    module_name, _, attr = target.rpartition(".")
                    candidate = f"{module_name}.{attr}"
                    if candidate in definitions:
                        graph[name].add(candidate)
    result: dict[str, set[str]] = {}
    for example in examples:
        start = example.entry_point.replace(":", ".")
        todo = [start]
        visited: set[str] = set()
        while todo:
            item = todo.pop()
            if item in visited:
                continue
            visited.add(item)
            todo.extend(graph.get(item, ()))
        result.setdefault(example.entry_point, set()).update(visited)
    return result


def _keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _IDENTIFIER.fullmatch(key):
                result.add(key)
            result.update(_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_keys(item))
    return result


def _literal_identifier(node: ast.AST | None) -> str | None:
    try:
        value = ast.literal_eval(node) if node is not None else None
    except (ValueError, TypeError, SyntaxError):
        value = None
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _public_subject_refs(sources: Mapping[str, str], tests: Mapping[str, str] = ()) -> tuple[SubjectRef, ...]:
    """Collect role-qualified subjects from declared public AST material.

    The source AST can establish a boundary role, but never supplies the
    contract by itself.  Public assertions are also subjects because they
    identify the representation the test actually exercises.
    """
    refs: list[SubjectRef] = []

    def add(boundary: str, representation: str, path: str, name: str, role: str) -> None:
        if not _IDENTIFIER.fullmatch(name):
            return
        refs.append(SubjectRef(boundary, representation, path, name, role))

    source_items = sources.items() if isinstance(sources, Mapping) else ()
    test_items = tests.items() if isinstance(tests, Mapping) else ()
    for path, source in (*source_items, *test_items):
        try:
            tree = _parse_cache.parse(source, path)
        except (SyntaxError, TypeError, ValueError):
            continue
        is_test = is_harness_path(Path(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def in_assertion(node: ast.AST) -> bool:
            current = parents.get(node)
            while current is not None:
                if isinstance(current, ast.Assert):
                    return True
                current = parents.get(current)
            return False

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for argument in (*node.args.posonlyargs, *node.args.args,
                                 *node.args.kwonlyargs):
                    add("public", "PARAMETER", path, argument.arg, "function_parameter")
                if node.args.vararg:
                    add("public", "PARAMETER", path, node.args.vararg.arg, "vararg_parameter")
                if node.args.kwarg:
                    add("public", "PARAMETER", path, node.args.kwarg.arg, "keyword_parameter")
            elif isinstance(node, ast.keyword) and node.arg:
                add("public", "CLI_OPTION" if is_test else "PARAMETER",
                    path, node.arg, "call_keyword")
            elif isinstance(node, ast.Subscript):
                name = _literal_identifier(node.slice)
                if name is None:
                    continue
                segment = ast.get_source_segment(source, node.value) or ""
                representation = ("HTTP_BODY" if re.search(
                    r"\b(?:json|body|payload|request)\b", segment, re.I)
                    else "OUTPUT_FIELD" if is_test and in_assertion(node) else "PARAMETER")
                add("public", representation, path, name,
                    "asserted_field" if is_test and in_assertion(node) else "boundary_field")
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    name = _literal_identifier(key)
                    if name is None:
                        continue
                    segment = ast.get_source_segment(source, node) or ""
                    representation = ("HTTP_BODY" if re.search(
                        r"['\"](?:json|body|payload)['\"]\s*:", segment, re.I)
                        else "OUTPUT_FIELD" if is_test and in_assertion(node) else "PARAMETER")
                    add("public", representation, path, name,
                        "assertion_mapping_key" if is_test and in_assertion(node)
                        else "mapping_key")
    return tuple(dict.fromkeys(refs))


def _public_ast_field_names(sources: Mapping[str, str]) -> set[str]:
    """Return names that public source ASTs actually use as parameters/keys."""
    return {ref.name for ref in _public_subject_refs(sources)}


def _public_assertion_names(tests: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """Return literal names appearing in public assertion ASTs by test path."""
    result: dict[str, list[str]] = defaultdict(list)
    items = tests.items() if isinstance(tests, Mapping) else ()
    for path, source in items:
        try:
            tree = _parse_cache.parse(source, path)
        except (SyntaxError, TypeError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            for child in ast.walk(node.test):
                if isinstance(child, ast.Dict):
                    for key in child.keys:
                        name = _literal_identifier(key)
                        if name is not None and name not in result[path]:
                            result[path].append(name)
                elif isinstance(child, ast.Subscript):
                    name = _literal_identifier(child.slice)
                    if name is not None and name not in result[path]:
                        result[path].append(name)
    return {path: tuple(names) for path, names in result.items()}


def _field_admission(brief: str, sources: Mapping[str, str],
                     examples: Sequence[_Example],
                     public_tests: Mapping[str, str] = ()) -> tuple[
                         tuple[str, ...], dict[str, str], dict[str, object]]:
    keys = set().union(*(_keys(example.fixture) for example in examples),
                       *(_keys(example.expected) for example in examples)) if examples else set()
    tokens = [token for token in _BACKTICK.findall(brief) if _IDENTIFIER.fullmatch(token)]
    subject_refs = _public_subject_refs(sources, public_tests)
    subject_by_name: dict[str, list[SubjectRef]] = defaultdict(list)
    for ref in subject_refs:
        subject_by_name[ref.name].append(ref)
    assertion_names = _public_assertion_names(public_tests)
    contract_names = set(tokens) | keys
    if assertion_names:
        contract_names |= set().union(*assertion_names.values())
    candidates: list[str] = []
    raw_candidates: list[str] = []
    optional_marker = re.compile(
        r"(?:optional|field|argument|parameter|setting|property|column)\s+[`']?([A-Za-z_]\w*)",
        re.I)
    for match in optional_marker.finditer(brief):
        if match.group(1).lower() not in {"call", "shape", "the", "value", "values"} \
                and match.group(1).lower() not in _FIELD_STOPWORDS:
            raw_candidates.append(match.group(1))
    # Keep prose diagnostics useful without letting generic prose become a
    # contract cell.  This catches shapes such as "audio channel" so the
    # rejection is journalable while admission still requires grounding.
    for match in re.finditer(
            r"\b([A-Za-z_]\w*)\s+(?:channel|mode|stream|option|setting|field|parameter)\b",
            brief, re.I):
        raw_candidates.append(match.group(1))
    for token in tokens:
        token_context = next((sentence for sentence in _field_sentences(brief, token)), "")
        if token in keys and re.search(
                r"(?:field|argument|parameter|setting|property|accepts?|accepted|"
                r"return(?:ed)?|response|result|forward|into|include(?:s|d)?)\b[^.!?`]*"
                + re.escape(token), token_context, re.I):
            if token not in {"from_payload", "http", "ValueError"} \
                    and token.lower() not in _FIELD_STOPWORDS:
                raw_candidates.append(token)
        elif re.search(r"\b[A-Za-z_]\w*-[A-Za-z_]\w*\s+`"
                       + re.escape(token) + r"`", brief, re.I):
            # Hyphenated prose such as ``audio-guide `language``` is a
            # useful rejection to journal, but is not a SubjectRef.
            raw_candidates.append(token)
        elif token in keys and re.search(r"rename|move(?:d|s)?\s+from|moving\s+from|"
                                        r"old\s+(?:names?|keys?)|to\s+`",
                                        token_context or brief, re.I):
            if token.lower() not in _FIELD_STOPWORDS:
                raw_candidates.append(token)
    # The rename/validation classes name a field without the word optional.
    quoted_tokens = set(tokens)
    for token in ("handle", "username", "days_late", "memo", "visibility", "priority", "label", "note",
                  "amount_cents", "category", "price_cents", "subtotal_cents", "fee_cents",
                  "total_cents", "tax_cents", "tip_cents", "packaging_cents", "approved"):
        if token in (quoted_tokens | keys):
            raw_candidates.append(token)
    if re.search(r"rename|move(?:d|s)?\s+from|old\s+(?:names?|keys?)", brief, re.I):
        raw_candidates.extend(key for key in sorted(keys)
                              if key.endswith("_id") and key.lower() not in _FIELD_STOPWORDS)
    rejected: dict[str, str] = {}
    certificates: dict[str, object] = {}
    for item in dict.fromkeys(raw_candidates):
        if item in _STRUCTURAL_KEYS:
            continue
        refs = subject_by_name.get(item, ())
        contract_sources: list[dict[str, object]] = []
        if item in tokens:
            contract_sources.append({"kind": "ticket_clause_ref", "ref": "brief"})
        if item in keys and not any(source.get("kind") == "public_test_ref"
                                    for source in contract_sources):
            contract_sources.append({"kind": "public_test_ref", "ref": "extracted_fixture"})
        for path, names in assertion_names.items():
            if item in names:
                contract_sources.append({"kind": "public_test_ref", "ref": path})
        if not refs or not contract_sources:
            rejected[item] = ("REJECTED_UNGROUNDED_FIELD"
                              if not refs and not contract_sources else
                              "REJECTED_NO_ROLE_EVIDENCE" if not refs else
                              "REJECTED_NO_CONTRACT_EVIDENCE")
            continue
        certificate = {
            "schema": "stateframe-admission-certificate/1",
            "field": item,
            "admission": "GATING",
            "interpretation": "CONFIRMED",
            "subject_refs": [ref.to_dict() for ref in refs],
            "contract_refs": contract_sources,
            "fixture": {"present": bool(examples), "otherwise_valid": bool(examples)},
            "observation": {"present": bool(refs), "boundary": refs[0].representation},
            "attribution": {"present": True, "authority": "public_material"},
            "evidence_complete": True,
        }
        certificates[item] = certificate
        if item not in subject_by_name or item not in contract_names:
            rejected[item] = "REJECTED_INCOMPLETE_ADMISSION"
            continue
        candidates.append(item)
    return tuple(dict.fromkeys(candidates)), rejected, certificates


def _field_candidates(brief: str, sources: Mapping[str, str], examples: Sequence[_Example]) -> tuple[str, ...]:
    return _field_admission(brief, sources, examples)[0]


def _field_sentences(brief: str, field: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|;\s*", brief)
            if re.search(r"\b" + re.escape(field) + r"s?\b", sentence, re.I)]


def _values_for_field(brief: str, field: str, examples: Sequence[_Example]) -> tuple[object, ...]:
    values: list[object] = []
    sentences = _field_sentences(brief, field)
    # Ticket prose commonly names the field in one sentence and its allowed
    # values in the following sentence.  Keep that public textual adjacency
    # instead of requiring an accidental same-sentence mention.
    if re.search(r"\b" + re.escape(field) + r"\b", brief, re.I):
        sentences = list(dict.fromkeys(sentences + [sentence.strip() for sentence in
            re.split(r"(?<=[.!?])\s+|;\s*", brief) if sentence.strip()]))
    for sentence in sentences:
        lower = sentence.lower()
        if not re.search(r"\b(?:only|valid|accepted|accepts|allowed|one of)\b|\bare\s+valid\b", lower):
            continue
        for token in _BACKTICK.findall(sentence):
            if token != field and _IDENTIFIER.fullmatch(token) and token.lower() not in {
                "http", "valueerror", "from_payload", "return", "true", "false", "none",
            }:
                if token not in values:
                    values.append(token)
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s+and\s+([A-Za-z_]\w*)\b", sentence):
            for token in match.groups():
                if token != field and token.lower() not in {
                    "must", "should", "are", "also", "accepted", "valid", "the",
                    "only", "one", "of", "with", "value", "values", "then", "or",
                    "returned", "forwarded", "selected",
                } and token not in values:
                    values.append(token)
    # A public test is evidence for one explicit value, but not for an
    # unmentioned set of allowed values.
    for example in examples:
        for value in _field_values(example.fixture, field):
            if isinstance(value, (str, int, float, bool)) and value not in values:
                if any(re.search(r"\b" + re.escape(field) + r"\b", sentence, re.I)
                       and re.search(r"selected|accepted|returned|valid|forward", sentence, re.I)
                       for sentence in _field_sentences(brief, field)):
                    values.append(value)
    return tuple(values)


def _field_values(value: object, field: str) -> tuple[object, ...]:
    found: list[object] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == field:
                found.append(item)
            found.extend(_field_values(item, field))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.extend(_field_values(item, field))
    return tuple(found)


def _default_for_field(brief: str, field: str) -> tuple[bool, object]:
    sentences = _field_sentences(brief, field)
    if re.search(r"\b" + re.escape(field) + r"\b", brief, re.I):
        sentences = list(dict.fromkeys(sentences + [sentence.strip() for sentence in
            re.split(r"(?<=[.!?])\s+|;\s*", brief) if sentence.strip()]))
    for sentence in sentences:
        lower = sentence.lower()
        quoted_default = re.search(r"[`'\"]([^`'\"]+)[`'\"]\s+is\s+the\s+default",
                                   sentence, re.I)
        if quoted_default:
            value = quoted_default.group(1).strip()
            if value.lower() in {"none", "null"}:
                return True, None
            return True, value
        normalized = re.search(r"normaliz\w*\s+to\s+[`'\"]([^`'\"]+)[`'\"]", sentence, re.I)
        if normalized:
            value = normalized.group(1)
            if value.lower() in {"none", "null"}:
                return True, None
            return True, value
        if "empty string" in lower:
            return True, ""
        if re.search(r"\b(?:none|null)\b", lower) and re.search(r"default|normaliz", lower):
            # Explicit ``None`` is a value, while “missing, None, … normalize
            # to X” is handled by the final literal below.
            match = re.search(r"(?:normalize\w*\s+to|default\w*\s+to)\s+[`']?([^`'.,;]+)", sentence, re.I)
            if match:
                value = match.group(1).strip()
                if value.lower() in {"none", "null"}:
                    return True, None
                return True, value
        match = re.search(r"(?:default\w*\s*(?:is|to|=)|is\s+the\s+default)\s*[`'=]*([^`'.,;\s]+)", sentence, re.I)
        if match:
            value = match.group(1).strip()
            if value.lower() in {"none", "null"}:
                return True, None
            if value.lower() in {"true", "false"}:
                return True, value.lower() == "true"
            try:
                return True, int(value)
            except ValueError:
                return True, value
        match = re.search(r"\b" + re.escape(field) + r"\s*=\s*[`'\"]?([^`'\".,;\s]+)", sentence, re.I)
        if match:
            value = match.group(1)
            try:
                return True, int(value)
            except ValueError:
                return True, value
        match = re.search(r"normaliz\w*\s+to\s+[`'\"]([^`'\"]+)[`'\"]", sentence, re.I)
        if match:
            return True, match.group(1)
    return False, None


def _normalization_for(brief: str, field: str) -> tuple[dict[object, object], bool]:
    sentences = _field_sentences(brief, field)
    context = " ".join(sentences)
    for sentence in sentences:
        if "normaliz" not in sentence.lower():
            continue
        found, default = _default_for_field(brief, field)
        if not found:
            continue
        result: dict[object, object] = {"__missing__": default}
        if re.search(r"\bnone\b", context, re.I):
            result[None] = default
        if "empty string" in context.lower() or "empty" in context.lower():
            result[""] = default
        if re.search(r"whitespace|blank|space", context, re.I):
            result["   "] = default
        if re.search(r"non-blank|nonblank|stripped|strip", context, re.I):
            result["  Boundary value  "] = "Boundary value"
        return result, True
    return {}, False


def _status_and_error(brief: str, field: str) -> tuple[int | str | None, str | None]:
    status: int | str | None = None
    error: str | None = None
    for sentence in _field_sentences(brief, field):
        match = _HTTP_STATUS.search(sentence)
        if match:
            status = int(match.group(1) or match.group(2))
        if re.search(r"error\s+text|with the text|text\s+`", sentence, re.I):
            quoted = _BACKTICK.findall(sentence)
            if quoted:
                error = " ".join(quoted[-1].split())
    # API mapping is often in a neighbouring sentence that omits the field.
    if status is None:
        match = _HTTP_STATUS.search(brief)
        if match:
            status = int(match.group(1) or match.group(2))
    if status is None and os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
        match = _REPAIR_ERROR_STATUS.search(brief)
        if match:
            token = match.group(1).lower()
            status = "4xx" if token == "4xx" else int(token)
    if error is None:
        matches = re.findall(
            r"(?:error\s+text|with\s+the\s+text|text)\s+[`']([^`']+)[`']",
            brief, re.I)
        error = next((" ".join(item.split()) for item in matches
                      if item != field and "(" not in item), None)
    return status, error


_ROUNDING_MODE_RE = re.compile(
    r"[`']?(?P<output>[A-Za-z_]\w*)[`']?\s+(?:as|is|=)\s*"
    r"(?P<percent>\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
    r"[`']?(?P<base>[A-Za-z_]\w*)[`']?[^.;]*?\brounded\s+"
    r"(?P<mode>half\s+up|half\s+down|to\s+even|bank(?:er|'?s)?|nearest)", re.I | re.S)


def _rounding_mode(value: str) -> str:
    lower = re.sub(r"\s+", " ", value.lower()).strip()
    if "half down" in lower:
        return "half_down"
    if "even" in lower or "bank" in lower:
        return "half_even"
    return "half_up"


def _round_exact(value: object, percent: str, mode: str) -> int | None:
    try:
        amount = Decimal(str(value)) * Decimal(percent) / Decimal("100")
        rounding = {
            "half_down": ROUND_HALF_DOWN,
            "half_even": ROUND_HALF_EVEN,
            "half_up": ROUND_HALF_UP,
            "nearest": ROUND_HALF_UP,
        }.get(mode, ROUND_HALF_UP)
        return int(amount.to_integral_value(rounding=rounding))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _field_path(value: object, field: str, prefix: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == field:
                return path
            nested = _field_path(item, field, path)
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            nested = _field_path(item, field, f"{prefix}.{index}" if prefix else str(index))
            if nested is not None:
                return nested
    return None


def _numeric_fixture_leaf(value: object, prefix: str = "") -> tuple[str, str, object] | None:
    candidates: list[tuple[int, str, str, object]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (int, float, Decimal)) and not isinstance(item, bool):
                key_text = str(key).lower()
                score = (0 if key_text in {"quantity", "count", "item_count"} else 10)
                if any(word in key_text for word in ("price", "amount", "value", "subtotal")):
                    score += 20
                if key_text.endswith("_cents"):
                    score += 10
                candidates.append((score, path, key_text, item))
            else:
                nested = _numeric_fixture_leaf(item, path)
                if nested is not None:
                    score = 5 + (20 if any(word in nested[1] for word in
                                           ("price", "amount", "value", "subtotal")) else 0)
                    candidates.append((score, nested[0], nested[1], nested[2]))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            nested = _numeric_fixture_leaf(item, f"{prefix}.{index}" if prefix else str(index))
            if nested is not None:
                candidates.append((nested[0] if len(nested) > 3 else 5,
                                   nested[0], nested[1], nested[2]))
    if not candidates:
        return None
    _, path, key, item = max(candidates, key=lambda row: row[0])
    return path, key, item


def _set_path(value: object, path: str, replacement: object) -> bool:
    parts = path.split(".") if path else []
    current = value
    for part in parts[:-1]:
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False
    if not parts:
        return False
    final = parts[-1]
    if isinstance(current, dict) and final in current:
        current[final] = copy.deepcopy(replacement)
        return True
    if isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = copy.deepcopy(replacement)
        return True
    return False


def _invalid_values(brief: str, field: str, allowed: Sequence[object]) -> tuple[object, ...]:
    values: list[object] = []
    sentences = _field_sentences(brief, field)
    text = " ".join(sentences).lower()
    if allowed and re.search(r"unknown|unsupported|invalid", text):
        values.append("unsupported")
    if re.search(r"at most\s+(\d+)", text):
        match = re.search(r"at most\s+(\d+)", text)
        limit = int(match.group(1))
        values.extend((limit + 1, "x" * (limit + 1)))
    if re.search(r"positive integer", text):
        values.extend((0, -1, 1.5))
    if re.search(r"non-negative integer|nonnegative integer", text):
        values.extend((-1, "2"))
    if re.search(r"non-empty|nonblank|non-blank", text):
        values.append("")
    if allowed and "unsupported" not in values:
        values.append("unsupported")
    return tuple(dict.fromkeys(values))


def _deep_set(value: object, field: str, item: object, *, omit: bool = False) -> bool:
    if isinstance(value, Mapping):
        if field in value:
            if omit:
                del value[field]  # type: ignore[index]
            else:
                value[field] = copy.deepcopy(item)  # type: ignore[index]
            return True
        if "json" in value and isinstance(value["json"], Mapping):
            if _deep_set(value["json"], field, item, omit=omit):
                return True
        for child in value.values():
            if isinstance(child, (Mapping, list)) and _deep_set(child, field, item, omit=omit):
                return True
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (Mapping, list)) and _deep_set(child, field, item, omit=omit):
                return True
    return False


def _fixture_field(fixture: Mapping[str, object], field: str, value: object,
                   *, omit: bool = False, signatures: Mapping[str, tuple[str, ...]] = ()) -> dict[str, object]:
    result = copy.deepcopy(dict(fixture))
    args = result.get("args")
    kwargs = result.get("kwargs")
    if isinstance(args, list) and _deep_set(args, field, value, omit=omit):
        return result
    if isinstance(kwargs, dict) and _deep_set(kwargs, field, value, omit=omit):
        return result
    # For a CLI/API function whose new parameter is positional, use the public
    # signature to append/replace exactly that parameter.  If it is not in the
    # public signature the resulting check intentionally becomes UNBOUND.
    entry = str(fixture.get("entry_point", ""))
    names = signatures.get(entry.replace(":", "."), ())
    if isinstance(args, list) and field in names:
        index = names.index(field)
        if omit:
            if index < len(args):
                args.pop(index)
        else:
            while len(args) <= index:
                args.append(copy.deepcopy(value))
            args[index] = copy.deepcopy(value)
    elif isinstance(args, list) and omit and names and len(args) > len(names):
        # An omitted newly-added positional parameter must use the genuine old
        # starter arity, rather than carrying a value appended by a public
        # example or a previous fixture transformation (check_coverage fix 3).
        del args[len(names):]
    elif (isinstance(args, list) and field not in names
          and len(args) == len(names) + 1):
        # Public tests can already exercise a newly requested positional
        # argument while the starter signature still has the old arity.  The
        # extra slot is the only defensible subject for a controlled probe.
        args[len(names)] = copy.deepcopy(value)
    elif isinstance(kwargs, dict) and not omit:
        kwargs[field] = copy.deepcopy(value)
    if not omit:
        # API fixtures conventionally carry the product payload under
        # ``request["json"]``.  If the public example omits a new field, add
        # it at that payload boundary instead of silently testing the old
        # default case.
        containers = []
        if isinstance(args, list):
            containers.extend(args)
        if isinstance(kwargs, dict):
            containers.extend(kwargs.values())
        for item in containers:
            if isinstance(item, dict) and isinstance(item.get("json"), dict):
                item["json"][field] = copy.deepcopy(value)
                return result
            if isinstance(item, dict) and field not in item:
                item[field] = copy.deepcopy(value)
                return result
    return result


def _fixture_changes_only_field(before: Mapping[str, object], after: Mapping[str, object],
                                field: str, *, entry: str | None = None,
                                signatures: Mapping[str, tuple[str, ...]] = ()) -> bool:
    """Check that an invalid probe changes exactly its named input field."""
    left = copy.deepcopy(dict(before))
    right = copy.deepcopy(dict(after))
    left.pop("entry_point", None)
    right.pop("entry_point", None)
    changes: list[tuple[object, ...]] = []
    args_before = left.get("args")
    names = signatures.get(str(entry or "").replace(":", "."), ())
    positional_index = (names.index(field) if field in names else
                        len(names) if isinstance(args_before, list)
                        and len(args_before) == len(names) + 1 else None)

    def visit(old: object, new: object, path: tuple[object, ...] = ()) -> None:
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            for key in set(old) | set(new):
                if key not in old or key not in new:
                    changes.append((*path, key))
                else:
                    visit(old[key], new[key], (*path, key))
            return
        if isinstance(old, list) and isinstance(new, list):
            for index in range(max(len(old), len(new))):
                if index >= len(old) or index >= len(new):
                    changes.append((*path, index))
                else:
                    visit(old[index], new[index], (*path, index))
            return
        if old != new:
            changes.append(path)

    visit(left, right)
    def is_target(path: tuple[object, ...]) -> bool:
        if not path:
            return False
        if str(path[-1]) == field:
            return True
        return (positional_index is not None and len(path) >= 2
                and path[-2] == "args" and path[-1] == positional_index)

    return bool(changes) and all(is_target(path) for path in changes)


def _project(value: object, field: str, projected: object) -> tuple[bool, object]:
    """Replace a named observable field without asserting unrelated changes."""
    if isinstance(value, Mapping):
        result = copy.deepcopy(dict(value))
        if field in result:
            result[field] = copy.deepcopy(projected)
            return True, result
        changed = False
        for key, child in list(result.items()):
            ok, replacement = _project(child, field, projected)
            if ok:
                result[key] = replacement
                changed = True
        return changed, result
    if isinstance(value, list):
        result = copy.deepcopy(value)
        changed = False
        for index, child in enumerate(result):
            ok, replacement = _project(child, field, projected)
            if ok:
                result[index] = replacement
                changed = True
        return changed, result
    return False, copy.deepcopy(value)


def _entry_surface(entry: str) -> str:
    module, _, attribute = entry.partition(":")
    if ".models" in module or attribute[:1].isupper():
        return "model." + attribute
    if ".service" in module:
        return "service." + attribute
    if ".merge" in module:
        return "service." + attribute
    if ".api" in module:
        return "api." + attribute
    if ".cli" in module:
        return "cli." + attribute
    return entry


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return text[:72] or "cell"


def _clause_list(brief: str) -> tuple[str, ...]:
    if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
        from kernel_requirement_alignment import extract_ticket_sentences
        return extract_ticket_sentences(brief)
    try:
        from kernel_obligation_seed import extract_obligations
        obligations = extract_obligations(brief, {"files": []})
        if obligations:
            return tuple(item.text for item in obligations)
    except (ImportError, TypeError, ValueError):
        pass
    return tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+|;\s*", brief)
                 if item.strip())


def _clause_id(clauses: Sequence[str], needle: str, default: int = 0) -> str:
    if not clauses:
        return "obl-boundary"
    words: set[str] = set()
    for word in _WORD.findall(needle):
        pieces = [piece for piece in word.lower().split("_") if len(piece) > 2]
        words.update(pieces)
        if len(word) > 2:
            words.add(word.lower())
    scores = [(len(words & {word.lower() for word in _WORD.findall(clause)}), -index)
              for index, clause in enumerate(clauses)]
    index = max(range(len(scores)), key=lambda item: scores[item]) if words else default
    return f"obl-{index:03d}"


class _Compiler:
    def __init__(self, task_id: str, brief: str, sources: Mapping[str, str],
                 tests: Mapping[str, str], *, coverage_ledger: object = None) -> None:
        self.task_id = task_id
        self.brief = brief
        self.sources = dict(sources)
        self.tests = dict(tests)
        self.coverage_ledger = coverage_ledger
        self.examples = _public_examples(self.tests)
        self.signatures = _function_signatures(self.sources)
        self.parameter_shapes = (_function_parameter_shapes(self.sources)
                                 if generator_fixes_enabled() else {})
        self.definitions = _source_definitions(self.sources)
        self.reachable = _reachable_surfaces(self.sources, self.examples)
        self.clauses = _clause_list(brief)
        self.checks: list[CheckDefinition] = []
        self.cells: list[ContractCell] = []
        self.by_obligation: dict[str, list[str]] = defaultdict(list)
        self.spec_cells: dict[str, list[ContractCell]] = defaultdict(list)
        self.counter = 0
        self._cli_forward_added = False
        self._preservation_added = False
        self._compatibility_added: set[tuple[str, str]] = set()
        self.field_rejections: dict[str, str] = {}
        self.fixture_rejections: list[dict[str, object]] = []
        self.admission_certificates: dict[str, object] = {}
        self._inferred_aliases_added = False
        self._compatibility_input_added: set[str] = set()
        self._model_shape_guarded: list[dict[str, str]] = []

    def _add_cell(self, obligation: str, entry: str | None, partition: object,
                  fixture: object, selector: object, predicate: object,
                  sources: Sequence[str], *, applies: bool = True,
                  admission: str = ADMITTED, surface: str | None = None) -> ContractCell | None:
        kind = _non_json_literal([partition, fixture, selector, predicate])
        if kind is not None:
            self._reject_non_json(f"cell:{obligation}", kind)
            return None
        surface = surface or _entry_surface(entry or "unknown:unknown")
        cell = ContractCell(
            obligation=obligation, surface=surface, case_partition=partition,
            fixture=fixture, observable_selector=selector, predicate=predicate,
            oracle_sources=tuple(sources),
            applicability=APPLIES if applies else APPLICABILITY_UNKNOWN,
            admission=admission if applies else UNBOUND,
        )
        if cell not in self.cells and len(self.cells) < MAX_CELLS:
            self.cells.append(cell)
        self.spec_cells[obligation].append(cell)
        return cell

    def _reject_non_json(self, subject: str, kind: str) -> None:
        self.field_rejections.setdefault(
            f"non_json_literal:{subject}",
            f"public material literal of type {kind} is not JSON; not bound")

    def _certificate_for(self, partition: object, *, basis: str,
                         entry: str | None, expected: object) -> dict[str, object] | None:
        field = partition.get("field") if isinstance(partition, Mapping) else None
        if isinstance(field, str) and field in self.admission_certificates:
            return copy.deepcopy(self.admission_certificates[field])
        # Non-field public examples and compatibility witnesses still need an
        # explicit admission record before their receipts can be authoritative.
        bound = bool(entry and entry.replace(":", ".") in self.definitions)
        return {
            "schema": "stateframe-admission-certificate/1",
            "field": field,
            "admission": "GATING" if bound else "ADVISORY",
            "interpretation": "CONFIRMED" if expected is not None else "OPEN",
            "subject_refs": ([{
                "boundary": "public", "representation": "PARAMETER",
                "path": str(entry or ""), "name": str(entry or ""),
                "role": "public_entry_point",
            }] if bound else []),
            "contract_refs": [{
                "kind": "public_test_ref" if basis == "PUBLIC_TEST" else "ticket_clause_ref",
                "ref": basis,
            }],
            "fixture": {"present": True, "otherwise_valid": True},
            "observation": {"present": expected is not None},
            "attribution": {"present": bound, "authority": "public_material"},
            "evidence_complete": bool(bound and expected is not None),
        }

    def _add_check(self, obligation: str, template: str, entry: str,
                   fixture: Mapping[str, object], expected: object,
                   partition: object, predicate: object, *, basis: str,
                   selection: Sequence[str], sources: Sequence[str]) -> CheckDefinition | None:
        if template not in TEMPLATES:
            raise ValueError(f"unsupported compiler template: {template}")
        kind = _non_json_literal([dict(fixture), expected, partition, predicate])
        if kind is not None:
            # Never bind a Python-only literal as a fixture or expected value.
            self._reject_non_json(f"check:{obligation}:{template}", kind)
            return None
        wire_fixture = copy.deepcopy(dict(fixture))
        wire_fixture.pop("entry_point", None)
        bound = entry.replace(":", ".") in self.definitions
        certificate = self._certificate_for(
            partition, basis=basis, entry=entry, expected=expected)
        cell_predicate = copy.deepcopy(predicate)
        if isinstance(cell_predicate, Mapping) and certificate is not None:
            cell_predicate["admission_certificate"] = copy.deepcopy(certificate)
        cell = self._add_cell(
            obligation, entry, partition, wire_fixture,
            cell_predicate.get("selector") if isinstance(cell_predicate, Mapping) else None,
            cell_predicate, sources, applies=bound,
            admission=ADMITTED if bound else UNBOUND)
        check_id = f"boundary-{_slug(self.task_id)}-{self.counter:03d}-{_slug(partition)}"
        self.counter += 1
        provenance = {
            "compiler": SCHEMA,
            "public_only": True,
            "entry_point": entry,
            "contract_cell_hash": cell.contract_cell_hash,
            "oracle_sources": list(sources),
            "admission_certificate": copy.deepcopy(certificate),
        }
        check = CheckDefinition(
            check_id, obligation, template, entry_point=entry,
            fixture=wire_fixture, expected=copy.deepcopy(expected),
            provenance=provenance, oracle_basis=basis,
            fixture_id=check_id, selection=selection,
            contract_cell=cell,
        )
        if len(self.checks) < MAX_CHECKS:
            self.checks.append(check)
            self.by_obligation[obligation].append(check_id)
        return check

    def _examples_for(self, surface: str) -> list[_Example]:
        return [item for item in self.examples
                if _entry_surface(item.entry_point).startswith(surface + ".")]

    def _targets_for_cells(self, cells: Sequence[ContractCell],
                           check_ids: Sequence[str] = ()) -> tuple[str, ...]:
        """Map each attributed check/surface to its product file."""
        surface_files = {
            "model": "app/models.py",
            "service": "app/service.py",
            "api": "app/api.py",
            "cli": "app/cli.py",
            "parser": "app/parser.py",
            "merge": "app/merge.py",
            "validation": "app/validation.py",
        }
        entry_targets = [check.entry_point.split(":", 1)[0].replace(".", "/") + ".py"
                         for check in self.checks if check.check_id in check_ids]
        targets = [path for path in entry_targets if path in self.sources]
        targets.extend(surface_files[str(cell.surface).split(".", 1)[0]]
                   for cell in cells
                   if str(cell.surface).split(".", 1)[0] in surface_files)
        bound = [path for path in dict.fromkeys(targets) if path in self.sources]
        return tuple(bound) if bound else tuple(sorted(self.sources))

    def _valid_example(self, field: str) -> _Example | None:
        candidates = [item for item in self.examples if item.expected is not None]
        with_field = [item for item in candidates if _field_values(item.fixture, field)]
        return (with_field[0] if with_field else candidates[0] if candidates else
                self.examples[0] if self.examples else None)

    def _api_example(self, field: str) -> _Example | None:
        items = [item for item in self.examples if ".api:" in item.entry_point]
        return next((item for item in items if _field_values(item.fixture, field)),
                    items[0] if items else None)

    def _service_example(self, field: str) -> _Example | None:
        items = [item for item in self.examples if ".service:" in item.entry_point]
        found = next((item for item in items if _field_values(item.fixture, field)),
                     items[0] if items else None)
        if found is not None:
            return found
        entry = self._service_entry(field)
        source = next((item for item in self.examples if item.expected is not None), None)
        if entry is None or source is None:
            return None
        fixture = self._service_fixture_from_example(source, entry, field)
        if fixture is None:
            return None
        expected = source.expected
        if isinstance(expected, Mapping) and "body" in expected:
            expected = expected["body"]
        return _Example(entry, fixture, copy.deepcopy(expected), source.selection, source.line)

    def _service_fixture_from_example(self, example: _Example, entry: str,
                                      field: str) -> dict[str, object] | None:
        """Project a public API/CLI example onto a service payload shape."""
        args = example.fixture.get("args", [])
        service_names = self.signatures.get(entry.replace(":", "."), ())
        if len(service_names) > 1:
            return {"args": copy.deepcopy(args)}
        payload: object = None
        if args and isinstance(args[0], Mapping):
            payload = args[0].get("json", args[0])
        if payload is None and ".cli:" in example.entry_point:
            names = self.signatures.get(example.entry_point.replace(":", "."), ())
            payload = {name: copy.deepcopy(value) for name, value in zip(names, args)
                       if name not in {"self", "cls"}}
        if not isinstance(payload, Mapping):
            return None
        return {"args": [copy.deepcopy(dict(payload))]}

    def _cli_entry(self, field: str) -> str | None:
        candidates = [name for name in self.signatures if name.startswith("app.cli.")]
        if not candidates:
            for name in ("run_profiles", "run"):
                if re.search(r"\b" + re.escape(name) + r"\b", self.brief):
                    return f"app.cli:{name}"
            return None
        for name in re.findall(r"\b(?:run|run_[A-Za-z_]\w*)\b", self.brief):
            if name not in {item.rsplit(".", 1)[-1] for item in candidates}:
                if name.startswith("run"):
                    return f"app.cli:{name}"
        mentioned = [name for name in candidates if re.search(
            r"\b" + re.escape(name.rsplit(".", 1)[-1]) + r"\b", self.brief)]
        return (f"{mentioned[0].rsplit('.', 1)[0]}:{mentioned[0].rsplit('.', 1)[1]}"
                if mentioned else f"{candidates[0].rsplit('.', 1)[0]}:{candidates[0].rsplit('.', 1)[1]}")

    def _service_entry(self, field: str) -> str | None:
        items = [item.entry_point for item in self.examples if ".service:" in item.entry_point]
        if items:
            return items[0]
        candidates = [name for name in self.signatures if name.startswith("app.service.")]
        if candidates:
            item = candidates[0]
            module, function = item.rsplit(".", 1)
            return f"{module}:{function}"
        return None

    def _selected_fixture(self, example: _Example, entry: str, field: str,
                          value: object, *, omit: bool = False) -> dict[str, object]:
        if entry.startswith("app.cli:") and not example.entry_point.startswith("app.cli:"):
            names = self.signatures.get(entry.replace(":", "."), ())
            shape = self.parameter_shapes.get(entry.replace(":", "."), {})
            positional_names = tuple(shape.get("posonly", ())) + tuple(shape.get("positional", ()))
            keyword_names = tuple(shape.get("keyword_only", ()))
            if not positional_names and not keyword_names:
                positional_names = tuple(names)
            source_args = example.fixture.get("args", [])
            payload = None
            if source_args and isinstance(source_args[0], Mapping):
                payload = source_args[0].get("json", source_args[0])
            if isinstance(payload, Mapping):
                args: list[object] = []
                for name in positional_names:
                    if name in {"self", "cls"}:
                        continue
                    if name in payload and (name != field or not omit):
                        args.append(copy.deepcopy(payload[name]))
                    elif name == field and not omit:
                        args.append(copy.deepcopy(value))
                    else:
                        break
                kwargs = {
                    name: copy.deepcopy(payload[name])
                    for name in keyword_names
                    if name in payload and (name != field or not omit)
                }
                if not omit and field in keyword_names:
                    kwargs[field] = copy.deepcopy(value)
                result = {"args": args, "entry_point": entry}
                if kwargs:
                    result["kwargs"] = kwargs
                if (not omit and field not in names and len(args) <= len(positional_names)
                        and self._cli_promises_field(field)):
                    args.append(copy.deepcopy(value))
                return result
        fixture = copy.deepcopy(example.fixture)
        fixture["entry_point"] = entry
        result = _fixture_field(fixture, field, value, omit=omit, signatures=self.signatures)
        if (not omit and entry.startswith("app.cli:") and isinstance(result.get("args"), list)):
            names = self.signatures.get(entry.replace(":", "."), ())
            if (field not in names and len(result["args"]) == len(names)
                    and self._cli_promises_field(field)):
                result["args"].append(copy.deepcopy(value))
        return result

    def _fixture_binds_cli_field(self, fixture: Mapping[str, object], entry: str,
                                 field: str, value: object) -> bool:
        """Require the explicit CLI fixture to carry the selected value."""
        if _field_values(fixture, field):
            return True
        args = fixture.get("args")
        if not isinstance(args, list):
            return False
        names = tuple(name for name in self.signatures.get(entry.replace(":", "."), ())
                      if name not in {"self", "cls"})
        try:
            index = names.index(field)
        except ValueError:
            return (self._cli_promises_field(field)
                    and len(args) == len(names) + 1
                    and args[-1] == value)
        return index < len(args) and args[index] == value

    def _cli_promises_field(self, field: str) -> bool:
        """Only add a new CLI positional when the ticket explicitly promises it."""
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|;\s*", self.brief)
                     if sentence.strip()]
        field_indexes = [index for index, sentence in enumerate(sentences)
                         if re.search(r"\b" + re.escape(field) + r"s?\b", sentence, re.I)]
        for index in field_indexes:
            sentence = sentences[index]
            if re.search(r"\b(?:cli|command|runner|run|positional|call\s+shape)\b",
                         sentence, re.I) and re.search(
                             r"optional|accept|forward|pass|argument|parameter|shape",
                             sentence, re.I):
                return True
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(sentences):
                    continue
                neighbor = sentences[neighbor_index]
                if (re.search(r"\b(?:cli|command|runner|public\s+entry\s+point|call\s+shape)\b",
                              neighbor, re.I)
                        and re.search(r"\b(?:carry|forward|pass|accept|expose|select|support|propagate)\b",
                                      neighbor, re.I)
                        and re.search(r"\b(?:selected|chosen|that|this|value|field|option|argument|parameter)\b",
                                      neighbor, re.I)
                        and re.search(r"\b(?:optional|new|additional|option|argument|parameter|positional|through|via)\b",
                                      sentence, re.I)):
                    return True
        return False

    def _payload_from_example(self, example: _Example) -> Mapping[str, object] | None:
        args = example.fixture.get("args", [])
        if args and isinstance(args[0], Mapping):
            payload = args[0].get("json", args[0])
            if isinstance(payload, Mapping):
                return payload
        if example.entry_point.startswith("app.cli:") and isinstance(args, list):
            names = self.signatures.get(example.entry_point.replace(":", "."), ())
            payload = {name: copy.deepcopy(value) for name, value in zip(names, args)
                       if name not in {"self", "cls"}}
            return payload
        if example.entry_point.startswith("app.api:") and isinstance(args, list):
            names = self.signatures.get(example.entry_point.replace(":", "."), ())
            payload = {name: copy.deepcopy(value) for name, value in zip(names, args)
                       if name not in {"self", "cls"}}
            return payload
        return None

    def _model_example(self, field: str, fallback: _Example) -> _Example | None:
        direct = self._examples_for("model")
        if direct:
            return direct[0]
        entries = [name for name in self.signatures
                   if name.startswith("app.models.") and name.count(".") >= 3
                   and name.rsplit(".", 1)[-1] == "from_payload"]
        payload = self._payload_from_example(fallback)
        if model_shape_guard_enabled() and payload is not None:
            self._model_shape_guarded.append({
                "field": field,
                "fallback_entry_point": fallback.entry_point,
                "reason": "serialized_fallback_is_not_model_constructor_shape",
            })
            return None
        if entries and payload is not None:
            expected = fallback.expected
            if isinstance(expected, Mapping) and "body" in expected:
                expected = expected["body"]
            module, class_name, method = entries[0].rsplit(".", 2)
            from kernel_named_layer_repair import enabled as named_layer_repair_enabled
            # r45 (autopsy-a3b-r44c-ldev01, check 001): the class lives INSIDE the module file, so the
            # entry point is module:Class.method (app/models.py), never module.Class:method (app/models/Event.py).
            entry = (f"{module}:{class_name}.{method}" if named_layer_repair_enabled()
                     else f"{module}.{class_name}:{method}")
            return _Example(
                entry,
                {"args": [copy.deepcopy(dict(payload))]},
                copy.deepcopy(expected), fallback.selection, fallback.line,
            )
        class_entries = [name for name in self.signatures
                         if name.startswith("app.models.") and name.count(".") == 2]
        if not class_entries or payload is None:
            return None
        expected = fallback.expected
        if isinstance(expected, Mapping) and "body" in expected:
            expected = expected["body"]
        names = self.signatures[class_entries[0]]
        if field not in names:
            names = (*names, field)
        return _Example(
            class_entries[0].replace("app.models.", "app.models:", 1),
            {"args": [copy.deepcopy(payload.get(name, self._sample_for_name(name, field, fallback)))
                      for name in names]},
            copy.deepcopy(expected), fallback.selection, fallback.line,
        )

    def _surface_default_examples(self, field: str, fallback: _Example) -> dict[str, _Example]:
        """Project one public example onto each callable boundary surface."""
        result: dict[str, _Example] = {}
        direct = {
            "model": self._model_example(field, fallback),
            "service": self._service_example(field),
            "api": self._api_example(field),
        }
        for surface, example in direct.items():
            if example is not None:
                result[surface] = example
        cli = self._cli_entry(field)
        base = result.get("api") or fallback
        if cli is not None and base is not None:
            result["cli"] = _Example(
                cli, self._selected_fixture(base, cli, field, None, omit=True),
                copy.deepcopy(base.expected), base.selection, base.line,
            )
        return result

    def _surface_is_bound(self, kind: str) -> bool:
        """Return whether the public AST binds the ticket's named surface."""
        definitions = self.definitions
        prefix = f"app.{kind}."
        if kind == "model":
            named = re.findall(r"\b(?:Profile|Event|Invoice|Order|Ticket|Search|Episode|"
                               r"Notification|ReturnRequest|Expense|Export)\b", self.brief)
            if "Profile" in named:
                return "app.models.Profile" in definitions
            return any(name.startswith("app.models.") and ".from_payload" in name
                       for name in definitions) or any(name.startswith("app.models.")
                                                       for name in definitions)
        if kind == "service":
            named = [name for name in re.findall(r"\b(?:find_profiles|find_users|publish|issue|"
                                                  r"checkout|receipt|search|send|deliver|return_book|"
                                                  r"submit|export|create)\b", self.brief)]
            if "find_profiles" in named:
                return "app.service.find_profiles" in definitions
            if named:
                return any(f"app.service.{name}" in definitions for name in named)
        if kind == "api":
            api_examples = [item.entry_point.replace(":", ".") for item in self.examples
                            if ".api:" in item.entry_point]
            if api_examples:
                return all(item in definitions for item in api_examples)
        if kind == "cli":
            cli_examples = [item.entry_point.replace(":", ".") for item in self.examples
                            if ".cli:" in item.entry_point]
            named = re.findall(r"\b(?:run_profiles|run)\b", self.brief)
            if "run_profiles" in named:
                return "app.cli.run_profiles" in definitions
            if cli_examples:
                return all(item in definitions for item in cli_examples)
        return any(name.startswith(prefix) for name in definitions)

    def _surface_grid(self, field: str, obligation: str) -> None:
        # The public AST supplies the surface inventory; call-graph reachability
        # is retained as provenance, while a surface is not discarded merely
        # because the public test exercises a different entry point.
        reached = sorted({name for nodes in self.reachable.values() for name in nodes})
        for kind in ("model", "service", "api", "cli"):
            applies = self._surface_is_bound(kind)
            predicate = {"kind": "surface_reachable", "field": field, "surface": kind,
                         "reachable_entries": reached}
            certificate = self.admission_certificates.get(field)
            if certificate is not None:
                predicate["admission_certificate"] = copy.deepcopy(certificate)
            self._add_cell(
                obligation, None, {"kind": "surface_grid", "field": field, "surface": kind},
                None, None, predicate,
                ("ticket:brief", "public:source_ast"), applies=applies,
                admission=ADMITTED if applies else UNBOUND, surface=f"{kind}.{field}")

    def _add_public_example(self, field: str, clauses: Sequence[str]) -> None:
        example = self._valid_example(field)
        if example is None or example.expected is None:
            return
        obligation = _clause_id(clauses, f"{field} returned selected value")
        self._add_check(
            obligation, "OUTPUT_EQUALS", example.entry_point, example.fixture,
            example.expected, {"kind": "explicit", "field": field},
            {"kind": "equals", "selector": field, "expected": _field_values(example.expected, field)[0]
             if _field_values(example.expected, field) else None},
            basis="PUBLIC_TEST", selection=(example.selection,),
            sources=("ticket:brief", f"public_test:{example.selection}"),
        )

    def _add_defaults_and_values(self, field: str, clauses: Sequence[str]) -> None:
        example = self._valid_example(field)
        if example is None:
            return
        found_default, default = _default_for_field(self.brief, field)
        normalization, normalized = _normalization_for(self.brief, field)
        service = self._service_example(field)
        if normalized and service is not None:
            cases: list[dict[str, object]] = []
            expected: list[object] = []
            for raw, projected in normalization.items():
                fixture = self._selected_fixture(service, service.entry_point, field, raw,
                                                 omit=raw == "__missing__")
                expected_value = _project(service.expected, field, projected)[1]
                cases.append(fixture)
                expected.append(expected_value)
            if cases and all(item is not None for item in expected):
                obligation = _clause_id(clauses, f"{field} normalization")
                self._add_check(
                    obligation, "OPTIONAL_INPUT", service.entry_point,
                    {"cases": cases}, expected,
                    {"kind": "normalization", "field": field},
                    {"kind": "value_transport", "selector": field},
                    basis="TICKET", selection=(service.selection,),
                    sources=("ticket:brief", f"public_test:{service.selection}"),
                )
            return
        if found_default:
            obligation = _clause_id(clauses, f"{field} default")
            for surface, candidate in self._surface_default_examples(field, example).items():
                fixture = self._selected_fixture(candidate, candidate.entry_point, field,
                                                 default, omit=True)
                changed, expected = _project(candidate.expected, field, default)
                partition = {"kind": "omitted", "field": field, "default": default,
                             "surface": surface}
                predicate = {"kind": "value_transport", "selector": field,
                             "expected": default}
                if changed:
                    self._add_check(
                        obligation, "OPTIONAL_INPUT", candidate.entry_point,
                        {"cases": [fixture]}, [expected], partition, predicate,
                        basis="TICKET", selection=(candidate.selection,),
                        sources=("ticket:brief", f"public_source:{candidate.entry_point}"),
                    )
                else:
                    wire_fixture = copy.deepcopy(fixture)
                    wire_fixture.pop("entry_point", None)
                    bound = candidate.entry_point.replace(":", ".") in self.definitions
                    self._add_cell(
                        obligation, candidate.entry_point, partition, {"cases": [wire_fixture]},
                        field, predicate, ("ticket:brief", "public_source:signature"),
                        applies=bound, admission=ADMITTED if bound else UNBOUND,
                    )
        values = _values_for_field(self.brief, field, self.examples)
        for value in values:
            changed, expected = _project(example.expected, field, value)
            if not changed:
                continue
            if _field_values(example.fixture, field) and value == _field_values(example.fixture, field)[0]:
                continue
            fixture = self._selected_fixture(example, example.entry_point, field, value)
            obligation = _clause_id(clauses, f"{field} accepted value {value}")
            self._add_check(
                obligation, "VALUE_TRANSPORT", example.entry_point, fixture, expected,
                {"kind": "explicit", "field": field, "value": value},
                {"kind": "value_transport", "selector": field, "expected": value},
                basis="TICKET", selection=(example.selection,),
                sources=("ticket:brief", f"public_source:{example.entry_point}"),
            )

    def _add_errors(self, field: str, clauses: Sequence[str]) -> None:
        allowed = _values_for_field(self.brief, field, self.examples)
        invalid = _invalid_values(self.brief, field, allowed)
        status, message = _status_and_error(self.brief, field)
        if not invalid or status is None:
            return
        example = self._api_example(field)
        if example is None:
            return
        obligation = _clause_id(clauses, f"{field} invalid input")
        for index, value in enumerate(invalid[:8]):
            fixture = self._selected_fixture(example, example.entry_point, field, value)
            if not _fixture_changes_only_field(
                    example.fixture, fixture, field,
                    entry=example.entry_point, signatures=self.signatures):
                self.fixture_rejections.append({
                    "field": field,
                    "value": copy.deepcopy(value),
                    "reason": "REJECTED_NON_TARGET_MUTATION",
                })
                continue
            expected: dict[str, object] = {"status": status}
            if message is not None:
                expected["body"] = {"error_contains": message}
            self._add_check(
                obligation, "ERROR_TRANSLATION" if message is not None else "REJECT_WITH_STATUS",
                example.entry_point, fixture, expected,
                {"kind": "invalid", "field": field, "value_index": index},
                {"kind": "error_translation", "selector": "status/body",
                 "status": status, "message": message},
                basis="TICKET", selection=(example.selection,),
                sources=("ticket:brief", f"public_source:{example.entry_point}"),
            )

    def _resolve_rounding_field(self, name: str, example: _Example) -> str | None:
        available = self._keys_from_examples()
        lowered = name.lower()
        exact = next((item for item in available if item.lower() == lowered), None)
        if exact is not None:
            return exact
        return next((item for item in available
                     if item.lower().startswith(lowered + "_")
                     or item.lower().endswith("_" + lowered)), None)

    def _discriminating_fixture(self, example: _Example, base_field: str,
                                percent: str, mode_name: str) -> tuple[dict[str, object], int] | None:
        public_base_values = _field_values(example.expected, base_field)
        if not public_base_values:
            public_base_values = _field_values(example.fixture, base_field)
        if not public_base_values:
            return None
        public_base = public_base_values[0]
        fixture = copy.deepcopy(example.fixture)
        leaf = _numeric_fixture_leaf(fixture)
        if leaf is None:
            return None
        leaf_path, _leaf_key, original = leaf
        public_result = _round_exact(public_base, percent, mode_name)
        if public_result is None:
            return None
        candidates: list[int] = [1003]
        try:
            anchor = int(Decimal(str(original)))
        except (InvalidOperation, TypeError, ValueError):
            return None
        for distance in range(1, 2048):
            for candidate in (anchor + distance, max(0, anchor - distance)):
                if candidate not in candidates:
                    candidates.append(candidate)
        chosen: int | None = None
        for candidate in candidates:
            rounded = _round_exact(candidate, percent, mode_name)
            floored = int((Decimal(str(candidate)) * Decimal(percent) / Decimal("100"))
                          .to_integral_value(rounding=ROUND_FLOOR))
            if rounded is not None and rounded != floored:
                chosen = candidate
                break
        if chosen is None or not _set_path(fixture, leaf_path, chosen):
            return None
        return fixture, chosen

    def _add_rounding_checks(self, clauses: Sequence[str]) -> None:
        seen: set[tuple[str, str, str]] = set()
        for match in _ROUNDING_MODE_RE.finditer(self.brief):
            output_name = match.group("output")
            base_name = match.group("base")
            mode_name = _rounding_mode(match.group("mode"))
            example = self._valid_example(output_name)
            if example is None or example.expected is None:
                continue
            output_field = self._resolve_rounding_field(output_name, example) or output_name
            base_field = self._resolve_rounding_field(base_name, example)
            if base_field is None:
                continue
            expected_values = _field_values(example.expected, output_field)
            public_base_values = _field_values(example.expected, base_field)
            if not expected_values or not public_base_values:
                continue
            public_result = _round_exact(public_base_values[0], match.group("percent"), mode_name)
            try:
                public_expected = int(Decimal(str(expected_values[0])))
            except (InvalidOperation, TypeError, ValueError):
                continue
            if public_result is None or public_result != public_expected:
                # The ticket oracle cannot reproduce the public example; do not
                # admit a speculative numeric cell.
                continue
            key = (output_field, base_field, mode_name)
            if key in seen:
                continue
            seen.add(key)
            discriminating = self._discriminating_fixture(
                example, base_field, match.group("percent"), mode_name)
            if discriminating is None:
                continue
            fixture, varied_value = discriminating
            fixture["entry_point"] = example.entry_point
            output_path = _field_path(example.expected, output_field)
            if output_path is None:
                continue
            expected_value = _round_exact(varied_value, match.group("percent"), mode_name)
            if expected_value is None:
                continue
            obligation = _clause_id(clauses, f"{output_field} rounding")
            self._add_check(
                obligation, "ROUNDING_DISCRIMINATION", example.entry_point, fixture,
                {"path": output_path, "value": expected_value,
                 "public_value": public_expected, "mode": mode_name,
                 "percent": match.group("percent"), "input": varied_value},
                {"kind": "rounding_discrimination", "field": output_field,
                 "base_field": base_field, "percent": match.group("percent"),
                 "mode": mode_name, "input": varied_value},
                {"kind": "rounding", "selector": output_path,
                 "expected": expected_value}, basis="TICKET",
                selection=(example.selection,),
                sources=("ticket:brief", f"public_source:{example.entry_point}"),
            )
        if re.search(r"total\w*.*(?:sum|plus).*component|components.*total|"
                     r"total\w*.*subtotal.*(?:plus|all)", self.brief, re.I | re.S):
            example = next((item for item in self.examples if item.expected is not None), None)
            if example is None:
                return
            names = [name for name in ("subtotal_cents", "tax_cents", "tip_cents",
                                       "packaging_cents", "fee_cents", "handling_cents")
                     if _field_path(example.expected, name) is not None]
            total_field = next((name for name in ("total_cents", "total")
                                if _field_path(example.expected, name) is not None), None)
            if total_field is None or len(names) < 2:
                return
            total_path = _field_path(example.expected, total_field)
            component_paths = [_field_path(example.expected, name) for name in names]
            if total_path is None or any(path is None for path in component_paths):
                return
            obligation = _clause_id(clauses, "total component consistency")
            self._add_check(
                obligation, "SUM_CONSISTENCY", example.entry_point, example.fixture,
                {"total_path": total_path, "component_paths": component_paths},
                {"kind": "sum_consistency", "total": total_field, "components": names},
                {"kind": "sum", "selector": total_path}, basis="TICKET",
                selection=(example.selection,),
                sources=("ticket:brief", f"public_source:{example.entry_point}"),
            )

    def _rename_stem(self) -> str | None:
        if not re.search(r"old\s+(?:names?|imports?|calls?)|legacy\s+names?|"
                         r"corresponding|unlisted",
                         self.brief, re.I):
            return None
        explicit = [token for token in _BACKTICK.findall(self.brief)
                    if _IDENTIFIER.fullmatch(token)]
        for token in explicit:
            if token.lower() in {"account", "profile", "customer", "user"}:
                return token.lower()
        match = re.search(r"\bto\s+([A-Za-z_]\w*)", self.brief, re.I)
        if match and match.group(1).lower() not in {"old", "new"}:
            return match.group(1).lower()
        match = re.search(r"\b(account|profile|customer)\b", self.brief, re.I)
        if match:
            return match.group(1).lower()
        return None

    def _inferred_alias_name(self, module: str, name: str, stem: str) -> str:
        lower = name.lower()
        plural = stem + "s"
        if lower == "member":
            return stem.capitalize()
        if lower.endswith("_members"):
            return name[:-len("members")] + plural
        if lower.endswith("members"):
            return name[:-len("members")] + plural
        if "member" in lower:
            return re.sub(r"members?", stem + ("s" if lower.endswith("members") else ""),
                          name, flags=re.I)
        if lower == "parse_row":
            return "parse_" + stem
        if lower in {"post", "run"}:
            return name + "_" + plural
        if module.endswith(".models") and name[:1].isupper():
            return stem.capitalize()
        return name + "_" + stem

    def _legacy_entries(self, stem: str) -> tuple[tuple[str, str, str], ...]:
        result: list[tuple[str, str, str]] = []
        for module, name in _public_symbol_entries(self.sources):
            if not module.startswith("app.") or name.startswith("_"):
                continue
            result.append((module, name, self._inferred_alias_name(module, name, stem)))
        return tuple(result)

    def _replace_input_key(self, value: object, old: str, new: str) -> object:
        if isinstance(value, Mapping):
            result: dict[object, object] = {}
            for key, item in value.items():
                target = old if key == new else key
                result[target] = self._replace_input_key(item, old, new)
            if old not in result and new not in result:
                result[old] = self._sample_for_name(old, "", None)
            return result
        if isinstance(value, list):
            return [self._replace_input_key(item, old, new) for item in value]
        return copy.deepcopy(value)

    def _add_inferred_compatibility(self, field: str, clauses: Sequence[str],
                                    example: _Example | None) -> None:
        if self._inferred_aliases_added:
            return
        stem = self._rename_stem()
        if stem is None:
            return
        self._inferred_aliases_added = True
        entries = self._legacy_entries(stem)
        old_symbols = [f"{module}.{name}" for module, name, _ in entries]
        new_symbols = [f"{module}.{new_name}" for module, _, new_name in entries]
        pairs = [{"old": old, "new": new, "alias": name}
                 for (module, name, _), old, new in zip(entries, old_symbols, new_symbols)]
        obligation = _clause_id(clauses, f"compatibility rename {stem}")
        self._add_cell(
            obligation, None,
            {"kind": "rename_coverage", "old_symbols": old_symbols,
             "new_symbols": new_symbols, "pairs": pairs, "new_stem": stem},
            {"old_symbols": old_symbols, "new_symbols": new_symbols, "pairs": pairs},
            "symbols", {"kind": "rename_coverage", "new_stem": stem},
            ("ticket:brief", "public:source_ast"), applies=True,
            admission=ADMITTED, surface=f"rename_coverage.{stem}",
        )
        selection = (example.selection,) if example is not None else ()
        for module, name, _ in entries:
            target = f"{module}:{name}"
            if (name, target) in self._compatibility_added:
                continue
            self._compatibility_added.add((name, target))
            self._add_check(
                _clause_id(clauses, f"compatibility alias {name}"),
                "COMPATIBILITY_WITNESS", target, self._compat_fixture(name, target, example), None,
                {"kind": "compatibility_alias", "alias": name},
                {"kind": "bind_and_execute", "selector": target},
                basis="PRESERVED_BEHAVIOR", selection=selection,
                sources=("ticket:brief", "public_source:signature"),
            )
        new_key = None
        for candidate in self._keys_from_examples():
            if candidate.endswith("_id") and stem in candidate:
                new_key = candidate
                break
        if new_key is None:
            new_key = next((token for token in _WORD.findall(self.brief)
                            if token.endswith("_id") and stem in token), None)
        new_key = new_key or (stem + "_id")
        old_key_candidates = set(_source_input_keys(self.sources))
        old_key_candidates.update(token for token in _WORD.findall(self.brief)
                                  if token.endswith("_id"))
        old_keys = [key for key in sorted(old_key_candidates)
                    if key not in {new_key, "json", "status", "body"}
                    and (key.endswith("_id") or key in {"username", "handle"})]
        base = self._api_example(field) or example
        if base is not None:
            for old_key in old_keys:
                if old_key in self._compatibility_input_added:
                    continue
                self._compatibility_input_added.add(old_key)
                fixture = copy.deepcopy(base.fixture)
                args = fixture.get("args")
                if isinstance(args, list):
                    fixture["args"] = [self._replace_input_key(item, old_key, new_key)
                                        for item in args]
                fixture["entry_point"] = base.entry_point
                self._add_check(
                    _clause_id(clauses, f"compatibility input {old_key}"),
                    "COMPATIBILITY_WITNESS", base.entry_point,
                    {"calls": [fixture]}, None,
                    {"kind": "compatibility_input", "alias": old_key,
                     "canonical": new_key},
                    {"kind": "bind_and_execute", "selector": base.entry_point},
                    basis="PRESERVED_BEHAVIOR", selection=selection,
                    sources=("ticket:brief", "public_source:input_keys"),
                )

    def _keys_from_examples(self) -> tuple[str, ...]:
        return tuple(sorted(set().union(*(_keys(example.fixture) for example in self.examples),
                                      *(_keys(example.expected) for example in self.examples)))) \
            if self.examples else ()

    def _add_explicit_input_aliases(self, field: str, clauses: Sequence[str],
                                    example: _Example | None) -> None:
        matches = re.findall(r"from\s+[`']([A-Za-z_]\w*)[`']\s+to\s+[`']([A-Za-z_]\w*)[`']",
                            self.brief, re.I)
        if not matches:
            return
        base = self._api_example(field) or example
        if base is None:
            return
        for old, new in matches:
            if old in self._compatibility_input_added:
                continue
            self._compatibility_input_added.add(old)
            fixture = copy.deepcopy(base.fixture)
            args = fixture.get("args")
            if isinstance(args, list):
                fixture["args"] = [self._replace_input_key(item, old, new)
                                    for item in args]
            fixture["entry_point"] = base.entry_point
            self._add_check(
                _clause_id(clauses, f"compatibility input {old}"),
                "COMPATIBILITY_WITNESS", base.entry_point,
                {"calls": [fixture]}, None,
                {"kind": "compatibility_input", "alias": old, "canonical": new},
                {"kind": "bind_and_execute", "selector": base.entry_point},
                basis="PRESERVED_BEHAVIOR", selection=(base.selection,),
                sources=("ticket:brief", "public_source:input_keys"),
            )

    def _add_cli_and_compat(self, field: str, clauses: Sequence[str]) -> None:
        entry = self._cli_entry(field)
        example = self._valid_example(field)
        if entry is not None and example is not None and not self._cli_forward_added:
            values = _values_for_field(self.brief, field, self.examples)
            value = (values[-1] if values else
                     (_field_values(example.fixture, field) or [None])[-1])
            if value is not None:
                fixture = self._selected_fixture(example, entry, field, value)
                # The CLI normally returns the API envelope unchanged.  Reuse
                # that public expected output; if it is not an envelope the
                # check remains bound to the public example's observable.
                obligation = _clause_id(clauses, f"CLI forwards {field}")
                partition = {"kind": "explicit", "surface": "cli", "field": field}
                predicate = {"kind": "invocation", "selector": field}
                sources = ("ticket:brief", "public_source:app/cli.py")
                cli_names = tuple(name for name in self.signatures.get(
                    entry.replace(":", "."), ()) if name not in {"self", "cls"})
                future_option = field not in cli_names and self._cli_promises_field(field)
                if self._fixture_binds_cli_field(fixture, entry, field, value) and not future_option:
                    self._add_check(
                        obligation,
                        "OUTPUT_EQUALS" if example.expected is not None else "INVOCATION_WITNESS",
                        entry, fixture, example.expected, partition, predicate, basis="TICKET",
                        selection=(example.selection,), sources=sources)
                elif future_option:
                    wire_fixture = copy.deepcopy(fixture)
                    wire_fixture.pop("entry_point", None)
                    coverage_predicate = {
                        **predicate, "kind": "cli_forwarding_coverage",
                        "selected_value": copy.deepcopy(value),
                    }
                    self._add_cell(
                        obligation, entry, partition, wire_fixture, field,
                        coverage_predicate, sources, applies=True,
                        admission=PROVISIONAL, surface=_entry_surface(entry))
                else:
                    wire_fixture = copy.deepcopy(fixture)
                    wire_fixture.pop("entry_point", None)
                    self.fixture_rejections.append({
                        "field": field, "surface": "cli",
                        "reason": "UNBOUND_EXPLICIT_INPUT",
                    })
                    self._add_cell(
                        obligation, entry, partition, wire_fixture, field, predicate, sources,
                        applies=False, admission=UNBOUND, surface=_entry_surface(entry))
            self._cli_forward_added = True
        # Every explicitly named old call form becomes a bind-and-execute
        # witness.  This is intentionally independent of an expected output.
        for match in re.finditer(r"`([A-Za-z_]\w*)\s*\(([^`]*)\)`", self.brief):
            name, raw_args = match.groups()
            if not re.search(r"existing|old|call|positional|integration|shape|remain|valid|alias",
                             self.brief[max(0, match.start() - 100):match.end() + 100], re.I):
                continue
            target = self._compat_entry(name)
            if target is None:
                continue
            compatibility_key = (name, target)
            if compatibility_key in self._compatibility_added:
                continue
            self._compatibility_added.add(compatibility_key)
            args, kwargs = self._compat_call_parts(raw_args, field, example)
            fixture = {"calls": [{"args": args}], "entry_point": target}
            if kwargs:
                fixture["calls"][0]["kwargs"] = kwargs
            self._add_check(
                _clause_id(clauses, f"compatibility {name}"), "COMPATIBILITY_WITNESS",
                target, fixture, None,
                {"kind": "compatibility", "call": name,
                 "arity": len(args) + len(kwargs)},
                {"kind": "bind_and_execute", "selector": target},
                basis="PRESERVED_BEHAVIOR", selection=(example.selection,) if example else (),
                sources=("ticket:brief", "public_source:signature"),
            )
        # Rename and alias clauses often list legacy symbols without spelling
        # out a call expression.  Bind-and-execute each named legacy symbol;
        # an absent promised alias is converted to a typed counterexample by
        # the host executor, while genuinely direct unbound checks remain
        # nonblocking unknowns.
        aliases: list[str] = []
        if re.search(r"moving\s+from|compatib(?:ility|le)|legacy|old\s+callers", self.brief, re.I):
            aliases.extend(name for name in _WORD.findall(self.brief)
                           if name in {"User", "parse_user", "find_users", "get_users", "run"})
        if re.search(r"service\s+alias", self.brief, re.I) and re.search(r"\bdeliver\b", self.brief):
            aliases.append("deliver")
        seen_aliases: set[str] = set()
        for name in aliases:
            if name in seen_aliases:
                continue
            seen_aliases.add(name)
            target = self._compat_entry(name)
            if target is None:
                continue
            compatibility_key = (name, target)
            if compatibility_key in self._compatibility_added:
                continue
            self._compatibility_added.add(compatibility_key)
            fixture = self._compat_fixture(name, target, example)
            self._add_check(
                _clause_id(clauses, f"compatibility alias {name}"),
                "COMPATIBILITY_WITNESS", target, fixture, None,
                {"kind": "compatibility_alias", "alias": name},
                {"kind": "bind_and_execute", "selector": target},
                basis="PRESERVED_BEHAVIOR", selection=(),
                sources=("ticket:brief", "public_source:signature"),
            )
        self._add_inferred_compatibility(field, clauses, example)
        self._add_explicit_input_aliases(field, clauses, example)

    def _compat_fixture(self, name: str, target: str,
                        example: _Example | None = None) -> dict[str, object]:
        """Build a starter-shaped alias invocation from public example values."""
        if not generator_fixes_enabled():
            # Preserve the pre-policy compiler's frozen OFF shape.  The
            # signature-aware/public-vocabulary path below is the opt-in fix.
            if name == "User":
                args: list[object] = ["legacy", "Legacy"]
            elif name in {"parse_user", "parse_profile"}:
                args = [{"username": "legacy", "name": "Legacy"}]
            elif name in {"find_users", "get_users", "run"}:
                args = [[{"username": "legacy", "name": "Legacy"}]]
            elif name == "deliver":
                args = ["n-1", "legacy@example.test", "legacy body"]
            else:
                args = [self._sample_for_name(item, "", None)
                        for item in self.signatures.get(target.replace(":", "."), ())
                        if item not in {"self", "cls"}]
            return {"calls": [{"args": args}], "entry_point": target}
        shape = self.parameter_shapes.get(target.replace(":", "."), {})
        positional_only = tuple(shape.get("posonly", ()))
        positional = tuple(shape.get("positional", ()))
        keyword_only = tuple(shape.get("keyword_only", ()))
        if shape:
            args = [self._sample_for_name(item, "", example)
                    for item in positional_only]
            kwargs = {
                item: self._sample_for_name(item, "", example)
                for item in (*positional, *keyword_only)
            }
            call: dict[str, object] = {"args": args, "kwargs": kwargs}
        else:
            names = tuple(item for item in self.signatures.get(target.replace(":", "."), ())
                          if item not in {"self", "cls"})
            call = {"args": [], "kwargs": {
                item: self._sample_for_name(item, "", example) for item in names}}
        return {"calls": [call], "entry_point": target}

    def _compat_call_parts(self, raw_args: str, field: str,
                           example: _Example | None) -> tuple[list[object], dict[str, object]]:
        args: list[object] = []
        kwargs: dict[str, object] = {}
        for raw in raw_args.split(","):
            item = raw.strip()
            if not item or item == "...":
                continue
            if "=" in item:
                name, raw_value = (part.strip() for part in item.split("=", 1))
                if _IDENTIFIER.fullmatch(name):
                    try:
                        value = ast.literal_eval(raw_value)
                    except (ValueError, SyntaxError):
                        value = self._sample_for_name(name, field, example)
                    else:
                        kind = _non_json_literal(value)
                        if kind is not None:
                            # ``name=...`` in prose is a placeholder, not a value.
                            self._reject_non_json(f"compat_kwarg:{name}", kind)
                            value = self._sample_for_name(name, field, example)
                    kwargs[name] = value
                    continue
            args.append(self._sample_for_name(item, field, example))
        return args, kwargs

    def _compat_entry(self, name: str) -> str | None:
        for module in ("app.cli", "app.api", "app.service", "app.parser", "app.merge", "app.models"):
            key = f"{module}.{name}"
            if key in self.signatures or key in _source_definitions(self.sources):
                return f"{module}:{name}"
        # The public brief can name an alias that does not exist in the starter;
        # map it to the surface named by its symbol, then let execution report
        # UNBOUND rather than claiming a product failure.
        if name in {"run", "run_profiles"}:
            return f"app.cli:{name}"
        if name == "deliver":
            return "app.service:deliver"
        if name in {"handle", "post", "get_users", "get_profiles"}:
            return f"app.api:{name}"
        if name.startswith("parse_"):
            return f"app.parser:{name}"
        if name in {"User", "Profile"}:
            return f"app.models:{name}"
        return None

    def _sample_for_name(self, name: str, field: str, example: _Example | None) -> object:
        if example is not None:
            value = _field_values(example.fixture, name)
            if value:
                return copy.deepcopy(value[0])
            value = _field_values(example.fixture, field)
            if value and name == field:
                return copy.deepcopy(value[0])
        lowered = name.lower()
        if lowered in {"rows", "items", "lines", "payload"}:
            return [] if lowered != "payload" else {}
        if lowered in {"active"}:
            return True
        if lowered.endswith("_id"):
            return "legacy-1"
        if lowered in {"amount_cents", "price_cents", "quantity", "days_late"}:
            return 1
        return "legacy"

    def _add_preservation(self, field: str, clauses: Sequence[str]) -> None:
        if self._preservation_added:
            return
        if not re.search(r"preserv|caller-owned|unchanged|idempotent|retry|logging|copy", self.brief, re.I):
            return
        example = self._service_example(field) or self._valid_example(field)
        if example is None:
            return
        fixture = copy.deepcopy(example.fixture)
        fixture["input_index"] = 0
        obligation = _clause_id(clauses, f"{field} caller preservation")
        self._add_check(
            obligation, "PRESERVE_INPUT", example.entry_point, fixture,
            {"preserved": True}, {"kind": "preserved_input", "field": field},
            {"kind": "ownership", "selector": "input"}, basis="TICKET",
            selection=(example.selection,),
            sources=("ticket:brief", f"public_test:{example.selection}"),
        )
        self._preservation_added = True
        self._add_check(
            obligation, "REPLAY_EQUIVALENT", example.entry_point, fixture,
            {"equivalent": True}, {"kind": "replay", "field": field},
            {"kind": "equivalent_replay", "selector": "output"}, basis="TICKET",
            selection=(example.selection,),
            sources=("ticket:brief", f"public_test:{example.selection}"),
        )

    def _alias_fields(self, fields: Sequence[str]) -> set[str]:
        if "handle" in fields and "username" in fields and re.search(
                r"moving\s+from\s+`username`\s+to\s+`handle`", self.brief, re.I):
            return {"username"}
        return set()

    def compile(self) -> CheckBundle | None:
        if not self.brief.strip() or not self.sources:
            return None
        fields, self.field_rejections, self.admission_certificates = _field_admission(
            self.brief, self.sources, self.examples, self.tests)
        if not fields and not self.examples and not generator_fixes_enabled():
            return None
        aliases = self._alias_fields(fields)
        public_example_added = False
        for field in fields[:16]:
            if field in aliases:
                continue
            obligation = _clause_id(self.clauses, field)
            self._surface_grid(field, obligation)
            if not public_example_added:
                self._add_public_example(field, self.clauses)
                public_example_added = True
            self._add_defaults_and_values(field, self.clauses)
            self._add_errors(field, self.clauses)
            self._add_cli_and_compat(field, self.clauses)
            self._add_preservation(field, self.clauses)
        self._add_rounding_checks(self.clauses)
        # A field-less mathematical ticket still gets the public exact
        # invocation as a contract cell; no unmentioned numeric oracle is
        # manufactured by this compiler.
        if not fields and self.examples:
            example = next((item for item in self.examples if item.expected is not None), self.examples[0])
            obligation = _clause_id(self.clauses, self.brief)
            self._add_check(
                obligation, "OUTPUT_EQUALS" if example.expected is not None else "INVOCATION_WITNESS",
                example.entry_point, example.fixture, example.expected,
                {"kind": "public_example"}, {"kind": "equals", "selector": None},
                basis="PUBLIC_TEST" if example.expected is not None else "TICKET",
                selection=(example.selection,), sources=("ticket:brief", f"public_test:{example.selection}"),
            )
            self._add_inferred_compatibility("", self.clauses, example)
            self._add_preservation("", self.clauses)
        generator_events: list[dict[str, object]] = []
        rejected_checks: list[dict[str, object]] = []
        if generator_fixes_enabled():
            original_checks = {check.check_id: check for check in self.checks}
            enforcement = enforce_generated_checks(
                self.checks, brief=self.brief, sources=self.sources, tests=self.tests,
                requirements_ledger=self.coverage_ledger)
            self.checks = list(enforcement.checks)
            cell_rewrites: dict[str, ContractCell] = {}
            for checked in enforcement.checked:
                original = original_checks.get(getattr(checked, "check_id", ""))
                old_cell = getattr(original, "contract_cell", None)
                new_cell = getattr(checked, "contract_cell", None)
                if old_cell is not None and new_cell is not None:
                    cell_rewrites[old_cell.contract_cell_hash] = new_cell
            if cell_rewrites:
                self.cells = [cell_rewrites.get(cell.contract_cell_hash, cell)
                              for cell in self.cells]
                for obligation, cells in self.spec_cells.items():
                    self.spec_cells[obligation] = [
                        cell_rewrites.get(cell.contract_cell_hash, cell) for cell in cells]
            self.by_obligation = defaultdict(list)
            for check in self.checks:
                self.by_obligation[check.obligation_id].append(check.check_id)
            generator_events = list(enforcement.events)
            rejected_checks = [dict(item) for item in enforcement.rejected]
            self.fixture_rejections.extend(rejected_checks)
        if not self.checks and not self.cells and not generator_fixes_enabled():
            return None
        specs: list[ObligationSpec] = []
        coverage_map: list[dict[str, object]] = []
        checks_by_id = {check.check_id: check for check in self.checks}
        for index, clause in enumerate(self.clauses or (self.brief,)):
            obligation = f"obl-{index:03d}"
            check_ids = tuple(self.by_obligation.get(obligation, ()))
            cells = tuple(self.spec_cells.get(obligation, ()))
            if not generator_fixes_enabled():
                if not check_ids and not cells:
                    continue
                specs.append(ObligationSpec(
                    obligation, {"path": "brief.md", "text": clause}, VALIDATED,
                    self._targets_for_cells(cells, check_ids), check_ids, clause,
                    {"compiler": SCHEMA, "source": "public brief"}, (),
                    contract_cells=cells,
                ))
                continue
            disposition = coverage_disposition(self.coverage_ledger, obligation, clause)
            coverage_required = normative_clause(clause, disposition)
            if not check_ids and not cells and not coverage_required:
                continue
            targets = self._targets_for_cells(cells, check_ids)
            check_ids = tuple(check_id for check_id in check_ids
                              if (checks_by_id.get(check_id) is not None
                                  and check_is_gating(checks_by_id[check_id])))
            coverage_map.append({
                "obligation_id": obligation,
                "clause": clause,
                "normative": coverage_required,
                "evidence_check_ids": list(check_ids),
                "covered": bool(check_ids),
                "evidence_status": "COVERED" if check_ids else "MISSING",
                "status_level": True,
                "disposition": disposition,
                "status_check_id": "coverage-{}-{}".format(
                    obligation,
                    re.sub(r"[^A-Za-z0-9]+", "-", clause).strip("-").lower()[:48]
                    or "clause"),
            })
            specs.append(ObligationSpec(
                obligation, {"path": "brief.md", "text": clause}, VALIDATED,
                targets, check_ids, clause,
                {"compiler": SCHEMA, "source": "public brief",
                 "coverage": coverage_map[-1]}, (),
                coverage_required=coverage_required,
                coverage_disposition=disposition,
                contract_cells=cells,
            ))
        # Checks can land on a clause skipped by the conservative segmenter.
        known = {spec.obligation_id for spec in specs}
        for obligation, check_ids in self.by_obligation.items():
            if obligation in known:
                continue
            if not generator_fixes_enabled():
                specs.append(ObligationSpec(
                    obligation, {"path": "brief.md", "text": self.brief}, VALIDATED,
                    tuple(sorted(self.sources)), tuple(check_ids), self.brief,
                    {"compiler": SCHEMA, "source": "public brief"}, (),
                    contract_cells=tuple(self.spec_cells.get(obligation, ())),
                ))
                continue
            disposition = coverage_disposition(self.coverage_ledger, obligation, self.brief)
            required = normative_clause(self.brief, disposition)
            check_ids = tuple(check_id for check_id in check_ids
                              if checks_by_id.get(check_id) is not None
                              and check_is_gating(checks_by_id[check_id]))
            specs.append(ObligationSpec(
                obligation, {"path": "brief.md", "text": self.brief}, VALIDATED,
                tuple(sorted(self.sources)), tuple(check_ids), self.brief,
                {"compiler": SCHEMA, "source": "public brief",
                 "coverage": {"obligation_id": obligation, "clause": self.brief,
                               "normative": required,
                               "evidence_check_ids": list(check_ids),
                               "covered": bool(check_ids),
                               "evidence_status": "COVERED" if check_ids else "MISSING",
                               "status_level": True,
                               "status_check_id": f"coverage-{obligation}-clause"}}, (),
                coverage_required=required,
                coverage_disposition=disposition,
                contract_cells=tuple(self.spec_cells.get(obligation, ())),
            ))
        provenance = {
            "schema": SCHEMA,
            "source_policy": "PUBLIC_TICKET_PUBLIC_TEST_PUBLIC_SOURCE_AST",
            "task_id": self.task_id,
            "brief_digest": digest(self.brief),
            "public_source_paths": sorted(self.sources),
            "public_test_paths": sorted(self.tests),
            "contract_cells": [cell.to_dict() for cell in self.cells],
            "field_rejections": copy.deepcopy(self.field_rejections),
            "fixture_rejections": copy.deepcopy(self.fixture_rejections),
            "admission_certificates": copy.deepcopy(self.admission_certificates),
        }
        if model_shape_guard_enabled() and self._model_shape_guarded:
            provenance["model_shape_guarded"] = copy.deepcopy(self._model_shape_guarded)
        if generator_fixes_enabled():
            generator_events.extend({
                "schema": "stateframe-check-generator-policy/1",
                "rule": "clause_coverage",
                "status": "COVERED" if item.get("covered") else "MISSING",
                **{key: copy.deepcopy(value) for key, value in item.items()
                   if key != "status"},
            } for item in coverage_map)
            provenance["generator_policy"] = {
                "schema": "stateframe-check-generator-policy/1",
                "enabled": True,
                "rejected_checks": copy.deepcopy(rejected_checks),
            }
            provenance["generator_events"] = copy.deepcopy(generator_events)
            provenance["clause_coverage"] = copy.deepcopy(coverage_map)
        return CheckBundle(
            self.task_id, tuple(specs), tuple(self.checks),
            "<dynamic-boundary-checks>", provenance, tuple(self.cells),
        )


def compile_boundary_bundle(task_id: str | Mapping[str, object], brief: str | None = None,
                            public_tests: object = None, public_sources: object = None,
                            *, row: Mapping[str, object] | None = None,
                            root: str | Path | None = None) -> CheckBundle | None:
    """Compile a dynamic public boundary bundle when the lane switch is on.

    ``task_id`` may be a row/manifest for convenience.  ``root`` is accepted
    only as a location for an explicitly supplied brief path; product source
    fallback is intentionally not performed because the live candidate is not
    public source authority.
    """
    if not enabled():
        return None
    def load_public_brief(value: object) -> str:
        if not isinstance(value, str):
            return str(value or "")
        if root is None or not (value.endswith(".md") or "/" in value):
            return value
        path = Path(root) / value
        try:
            relative = path.resolve().relative_to(Path(root).resolve())
            if (relative.suffix != ".md" or ".." in relative.parts
                    or any(part.startswith(".") or part.lower() in _FORBIDDEN_PARTS
                           or "backup" in part.lower() for part in relative.parts)):
                return ""
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            return ""

    if isinstance(task_id, Mapping):
        row = task_id
        task = str(row.get("task_id", row.get("id", "dynamic-boundary")))
        brief, sources, tests = public_material_from_row(row)
        brief = load_public_brief(brief)
    else:
        task = str(task_id)
        if row is not None:
            row_brief, row_sources, row_tests = public_material_from_row(row)
            brief = brief or row_brief
            public_sources = public_sources if public_sources is not None else row_sources
            public_tests = public_tests if public_tests is not None else row_tests
        if brief is None:
            brief = ""
        brief = load_public_brief(brief)
        sources = _collect_files(public_sources or {}, test=False)
        tests = _public_tests(public_tests or {})
    if not isinstance(brief, str):
        raise TypeError("public boundary brief must be text")
    ledger = None
    if isinstance(row, Mapping):
        ledger = row.get("requirements_ledger", row.get("coverage_dispositions"))
    key = _compile_memo_key(task, brief, sources, tests, ledger)
    if key is not None:
        with _COMPILE_MEMO_LOCK:
            if key in _COMPILE_MEMO:
                _COMPILE_MEMO.move_to_end(key)
                return copy.deepcopy(_COMPILE_MEMO[key])
    bundle = _Compiler(task, brief, sources, tests, coverage_ledger=ledger).compile()
    if key is not None:
        with _COMPILE_MEMO_LOCK:
            _COMPILE_MEMO[key] = copy.deepcopy(bundle)
            while len(_COMPILE_MEMO) > _COMPILE_MEMO_MAX:
                _COMPILE_MEMO.popitem(last=False)
    return bundle


# L112 host-time fix: the compiler is a pure function of its public inputs and the
# process environment (lane switches), yet live steps recompiled it ~13x per step.
# The memo key covers ALL of those inputs exactly (type-tagged, so a tuple never
# aliases a list); anything uncanonicalizable is simply not memoized.  Hits return a
# deep copy so no caller can alias or mutate the stored bundle.
_COMPILE_MEMO_MAX = 32
_COMPILE_MEMO: "OrderedDict[str, object]" = OrderedDict()
_COMPILE_MEMO_LOCK = threading.Lock()


def _memo_canon(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return [type(value).__name__, value]
    if isinstance(value, Mapping):
        return ["map", sorted(([_memo_canon(k), _memo_canon(v)] for k, v in value.items()),
                              key=lambda item: json.dumps(item[0], sort_keys=True))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_memo_canon(item) for item in value]]
    raise TypeError(f"not memoizable: {type(value).__name__}")


def _compile_memo_key(task: str, brief: str, sources: object, tests: object,
                      ledger: object) -> str | None:
    try:
        payload = json.dumps(_memo_canon([task, brief, sources, tests, ledger,
                                          sorted(os.environ.items())]),
                             sort_keys=True, allow_nan=True)
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def compile_public_boundary_bundle(*args, **kwargs) -> CheckBundle | None:
    return compile_boundary_bundle(*args, **kwargs)


def compile_contract_cells(*args, **kwargs) -> tuple[ContractCell, ...]:
    bundle = compile_boundary_bundle(*args, **kwargs)
    return bundle.contract_cells if bundle is not None else ()


def compile_boundary_report(*args, **kwargs) -> BoundaryCompilationReport:
    task_id = args[0] if args else kwargs.get("task_id", "dynamic-boundary")
    if isinstance(task_id, Mapping):
        task_id = task_id.get("task_id", task_id.get("id", "dynamic-boundary"))
    bundle = compile_boundary_bundle(*args, **kwargs)
    cells = tuple(bundle.contract_cells) if bundle is not None else ()
    bound = tuple(cell for cell in cells if cell.applicability == APPLIES
                  and cell.admission in {ADMITTED, PROVISIONAL})
    unknown = tuple(cell for cell in cells if cell not in bound)
    return BoundaryCompilationReport(str(task_id), bundle, bound, unknown)


__all__ = [
    "BoundaryCompilationReport", "SubjectRef", "MODE_ENV", "MODEL_SHAPE_MODE_ENV", "SCHEMA", "compile_boundary_bundle",
    "compile_boundary_report", "compile_contract_cells", "compile_public_boundary_bundle",
    "enabled", "mode", "model_shape_guard_enabled", "public_material_from_row",
]
