"""Kernel-owned FLQGATE7B typed reproduction checks.

The model supplies only values from host-owned menus.  This module builds a
bounded AST symbol packet, validates the typed specification, compiles a
unittest, and hands that generated source to the runtime's existing isolated
check runner.  No model-authored Python is accepted on this rail.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


SCHEMA = "stateframe-idealpath-reprospec/1"
SYMBOL_PACKET_SCHEMA = "stateframe-idealpath-symbol-packet/1"
TOOL_NAME = "typed_check_spec"
SCRATCH_DIR = ".repro-first-scratch"
MAX_TYPED_REPAIR_ATTEMPTS = 2
MAX_PACKET_MODULES = 24
MAX_PACKET_SYMBOLS = 256
MAX_PACKET_SOURCE_BYTES = 512_000
MAX_COMPILED_SOURCE_CHARS = 48_000
_STEP_REFERENCE = re.compile(r"^\$(?:[1-9]|1[0-6])$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEST_COUNT = re.compile(r"Ran\s+(\d+)\s+tests?", re.IGNORECASE)
_ASSERTION = re.compile(r"(?:AssertionError|FAIL(?:URE)?\s*:)", re.IGNORECASE)
_ENVIRONMENT = re.compile(
    r"(?:ImportError|ModuleNotFoundError|No module named|cannot import name|"
    r"collection error|test collection failed|environment unavailable)",
    re.IGNORECASE,
)
_RESOURCE = re.compile(r"(?:timeout|timed out|out of memory|resource limit)", re.IGNORECASE)
_TYPED_EXCEPTION_NAMES = frozenset({
    "AssertionError", "AttributeError", "KeyError", "LookupError",
    "ModuleNotFoundError", "NameError", "RuntimeError", "TimeoutError",
    "TypeError", "ValueError",
})
_TARGET_KINDS = frozenset({"function", "class"})
_SETUP_KINDS = frozenset({
    "call", "construct", "fixture", "set_attribute",
    "async_call", "call_spy", "clock_boundary",
})
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {
    "__future__", "unittest",
}


class ReproSpecError(ValueError):
    """Raised only for host-side typed specification/compiler failures."""


def _json_value(value: object) -> bool:
    try:
        encoded = json.dumps(value, allow_nan=False)
        json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return all(_json_value(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


def _literal(value: object) -> str:
    if not _json_value(value):
        raise ReproSpecError("typed value is not JSON data")
    return repr(json.loads(json.dumps(value, allow_nan=False)))


def _text(value: object, limit: int = 2000) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\x00", " ").strip()
    return value[:limit] if len(value) <= limit else ""


def _error(field: str, message: str, *, allowed: Sequence[str] = ()) -> dict[str, object]:
    result: dict[str, object] = {
        "type": "typed_check_spec_error",
        "field": field,
        "message": message,
    }
    if allowed:
        result["allowed"] = list(allowed)
    return result


def _module_for_path(path: str) -> str | None:
    normalized = path.replace("\\", "/").lstrip("./")
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] == "backend":
        parts = parts[1:]
    if not parts or not parts[-1].endswith(".py"):
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
        if not parts or not all(_IDENTIFIER.fullmatch(part) for part in parts):
            return None
        return ".".join(parts)
    if parts[-1] == "conftest.py" and len(parts) == 1:
        return "conftest"
    parts[-1] = parts[-1][:-3]
    if not all(_IDENTIFIER.fullmatch(part) for part in parts):
        return None
    return ".".join(parts)


def _ast_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        result = f"{node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            result += f" -> {ast.unparse(node.returns)}"
        return result
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    if isinstance(node, ast.AnnAssign) and node.annotation is not None:
        return ast.unparse(node.annotation)
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        return "module-level assignment"
    if isinstance(node, ast.Import):
        return "module import"
    if isinstance(node, ast.ImportFrom):
        return "re-export"
    return "module-level name"


def _assignment_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for child in target.elts for name in _assignment_names(child))
    return ()


def _public_symbols(source: str) -> tuple[dict[str, str], ...]:
    tree = ast.parse(source)
    symbols: dict[str, dict[str, str]] = {}

    def add(name: str, kind: str, signature: str) -> None:
        if name.startswith("_") or not _IDENTIFIER.fullmatch(name):
            return
        symbols.setdefault(name, {"name": name, "kind": kind, "signature": signature})

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, "function", _ast_signature(node))
        elif isinstance(node, ast.ClassDef):
            add(node.name, "class", _ast_signature(node))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.asname or alias.name.split(".")[0], "import", _ast_signature(node))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    add(alias.asname or alias.name, "re-export", _ast_signature(node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _assignment_names(target):
                    add(name, "assignment", _ast_signature(node))
        elif isinstance(node, ast.AnnAssign):
            for name in _assignment_names(node.target):
                add(name, "constant", _ast_signature(node))
        elif isinstance(node, ast.AugAssign):
            for name in _assignment_names(node.target):
                add(name, "assignment", _ast_signature(node))
    return tuple(symbols[name] for name in sorted(symbols))


def _source_revision(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def _required_parameters(node: ast.AST) -> tuple[str, ...]:
    candidate = node
    if isinstance(node, ast.ClassDef):
        candidate = next((item for item in node.body
                          if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and item.name == "__init__"), None)
    if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ()
    positional = [*candidate.args.posonlyargs, *candidate.args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    defaults = [None] * (len(positional) - len(candidate.args.defaults))
    defaults.extend(candidate.args.defaults)
    required = [arg.arg for arg, default in zip(positional, defaults) if default is None]
    required.extend(
        arg.arg for arg, default in zip(candidate.args.kwonlyargs, candidate.args.kw_defaults)
        if default is None
    )
    return tuple(required)


def _source_symbol_outline(source: str, path: str, module: str | None) -> tuple[dict[str, object], ...]:
    """Return verified source identities, including internal and nested sites."""
    tree = ast.parse(source)
    revision = _source_revision(source)
    result: list[dict[str, object]] = []
    existing = {item["name"] for item in _public_symbols(source)}
    if module is None or path.replace("\\", "/").endswith("/__init__.py"):
        result.append({
            "id": f"source:{path}:module",
            "name": "__module__",
            "kind": "module",
            "target_kind": "module_head",
            "signature": "module initialization",
            "source_identity": {
                "source_path": path, "definition": "module:1",
                "revision": revision,
            },
            "capabilities": ["module_initialization", "source_path_identity"],
        })
    def walk(body: Sequence[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = (*prefix, node.name)
                name = node.name
                public_top_level = not prefix and not name.startswith("_") and name in existing
                identity = (
                    f"{module}:{name}" if module is not None and public_top_level
                    else f"source:{path}:{'.'.join(qualified)}"
                )
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                target_kind = "nested_function" if prefix and kind == "function" else kind
                result.append({
                    "id": identity,
                    "name": name,
                    "qualified_name": ".".join(qualified),
                    "kind": kind,
                    "target_kind": target_kind,
                    "signature": _ast_signature(node),
                    "required_parameters": list(_required_parameters(node)),
                    "source_identity": {
                        "source_path": path,
                        "definition": f"{'.'.join(qualified)}:{getattr(node, 'lineno', 1)}:{getattr(node, 'end_lineno', getattr(node, 'lineno', 1))}",
                        "revision": revision,
                    },
                    "capabilities": [
                        "source_path_identity",
                        "direct_call" if not prefix else "nested_site",
                    ],
                })
                nested = getattr(node, "body", ())
                walk(nested, qualified)
    walk(tree.body)
    # Keep public top-level IDs stable and ensure the module-head target is
    # explicit without allowing source metadata to impersonate a package.
    unique: dict[str, dict[str, object]] = {}
    for item in result:
        unique.setdefault(str(item["id"]), item)
    return tuple(unique.values())


def _normalize_paths(root: Path, paths: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    root_resolved = root.resolve()
    from kernel_request_source_window import _FORBIDDEN_PARTS, _TEST_PATH
    for raw in paths:
        if not isinstance(raw, str):
            continue
        candidate_text = raw.replace("\\", "/").strip()
        if candidate_text.startswith(("src:", "path:", "file:")):
            candidate_text = candidate_text.split(":", 1)[1]
        candidate_text = candidate_text.lstrip("./")
        if (not candidate_text.endswith(".py") or ".." in Path(candidate_text).parts
                or _TEST_PATH.search(candidate_text)
                or any(part.casefold() in _FORBIDDEN_PARTS for part in candidate_text.split("/"))):
            continue
        candidate = (root / candidate_text).resolve()
        try:
            relative = candidate.relative_to(root_resolved).as_posix()
        except ValueError:
            continue
        if relative not in result:
            result.append(relative)
    return tuple(result)


def _source_for(root: Path, path: str, source_map: Mapping[str, object]) -> tuple[str | None, bool]:
    mapped = source_map.get(path)
    if isinstance(mapped, str):
        return (mapped if len(mapped.encode("utf-8")) <= MAX_PACKET_SOURCE_BYTES else None), False
    file_path = root / path
    try:
        if not file_path.is_file() or file_path.is_symlink():
            return None, False
        if file_path.stat().st_size > MAX_PACKET_SOURCE_BYTES:
            return None, True
        return file_path.read_text(encoding="utf-8"), False
    except (OSError, UnicodeError):
        return None, False


def _packet_entries(packet: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = packet.get("modules")
    if not isinstance(raw, list):
        return {}
    return {
        str(item["module"]): item
        for item in raw
        if isinstance(item, Mapping) and isinstance(item.get("module"), str)
    }


def build_symbol_packet(
        root: Path,
        reached_paths: Sequence[object],
        *,
        target_path: str | None = None,
        source_map: Mapping[str, object] | None = None,
        known_paths: Sequence[object] = (),
        max_modules: int = MAX_PACKET_MODULES,
        max_symbols: int = MAX_PACKET_SYMBOLS,
) -> dict[str, object]:
    """Build only the bounded AST outline for paths reached in S2.

    ``known_paths`` is names-only metadata used to resolve one-hop imports; no
    un-reached file is read unless it is imported by a reached module.
    """
    source_map = source_map if isinstance(source_map, Mapping) else {}
    selected = list(_normalize_paths(root, reached_paths))
    known = _normalize_paths(root, [*reached_paths, *known_paths])
    module_paths = {
        module: path for path in known
        if (module := _module_for_path(path)) is not None
    }
    modules: list[dict[str, object]] = []
    parsed: dict[str, ast.Module] = {}
    truncated = False
    cursor = 0
    while cursor < len(selected) and len(modules) < max_modules:
        path = selected[cursor]
        cursor += 1
        if path in {str(item.get("source_path")) for item in modules}:
            continue
        module = _module_for_path(path)
        source, was_truncated = _source_for(root, path, source_map)
        if was_truncated:
            truncated = True
        symbols: list[dict[str, object]] = []
        parse_error: str | None = None
        if source is not None:
            try:
                tree = ast.parse(source)
                parsed[path] = tree
                symbols = list(_source_symbol_outline(source, path, module))
            except SyntaxError as error:
                parse_error = error.msg
        entry: dict[str, object] = {
            "module": module,
            "source_path": path,
            "source_identity": {
                "source_path": path,
                "definition": "source-unit",
                "revision": _source_revision(source) if source is not None else "unreadable",
            },
            "target_capabilities": ["source_path_identity"] if module is None else ["module_import", "source_path_identity"],
            "public_names": symbols[:max(0, max_symbols - sum(len(item.get("public_names", [])) for item in modules))],
        }
        if parse_error:
            entry["parse_error"] = parse_error
        if source is None and was_truncated:
            entry["outline_status"] = "BIGFILE_UNREAD"
        modules.append(entry)
        for tree_node in parsed.get(path, ast.Module(body=[], type_ignores=[])).body:
            imported: tuple[str, ...] = ()
            if isinstance(tree_node, ast.Import):
                imported = tuple(alias.name for alias in tree_node.names)
            elif isinstance(tree_node, ast.ImportFrom) and tree_node.level == 0 and tree_node.module:
                imported = (tree_node.module,)
            for imported_module in imported:
                imported_path = module_paths.get(imported_module)
                if imported_path is not None and imported_path not in selected:
                    selected.append(imported_path)
        if sum(len(item.get("public_names", [])) for item in modules) >= max_symbols:
            truncated = True
            break
    return {
        "record_type": "symbol_packet",
        "schema": SYMBOL_PACKET_SCHEMA,
        "target_bound": bool(str(target_path or "").strip()),
        "localization_required": not bool(str(target_path or "").strip()),
        "modules": modules,
        "reached_paths": list(_normalize_paths(root, reached_paths))[:max_modules],
        "bounded": True,
        "truncated": truncated or cursor < len(selected),
    }


def _symbol_id(module: str, name: str) -> str:
    return f"{module}:{name}"


def _symbol_menus(packet: Mapping[str, object]) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    targets: dict[str, Mapping[str, object]] = {}
    fixtures: dict[str, Mapping[str, object]] = {}
    for entry in packet.get("modules", ()) if isinstance(packet.get("modules"), list) else ():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("source_path"), str):
            continue
        module = entry.get("module") if isinstance(entry.get("module"), str) else ""
        source_path = entry.get("source_path")
        is_fixture_module = isinstance(source_path, str) and source_path.endswith("conftest.py")
        for raw in entry.get("public_names", ()) if isinstance(entry.get("public_names"), list) else ():
            if not isinstance(raw, Mapping):
                continue
            name, kind = raw.get("name"), raw.get("kind")
            if not isinstance(name, str) or not isinstance(kind, str):
                continue
            target_kind = str(raw.get("target_kind", kind))
            identifier = str(raw.get("id") or (_symbol_id(module, name) if module else
                                                f"source:{source_path}:{name}"))
            symbol = {
                "id": identifier, "module": module, "name": name,
                "kind": kind, "target_kind": target_kind,
                "signature": raw.get("signature"), "source_path": source_path,
                "source_identity": copy.deepcopy(raw.get("source_identity")),
                "capabilities": list(raw.get("capabilities", entry.get("target_capabilities", ()))),
            }
            if (kind in _TARGET_KINDS or target_kind in {"module_head", "initialization"}) and not is_fixture_module:
                targets[str(symbol["id"])] = symbol
            if is_fixture_module and kind == "function":
                fixtures[str(symbol["id"])] = symbol
    return targets, fixtures


def symbol_target_menu(packet: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Expose the host-owned target menu to typed adapters.

    S3 mechanisms must share the exact REPROSPEC symbol authority.  Keeping
    this wrapper beside the private menu builder prevents a second target
    enumerator from drifting away from the frozen check compiler.
    """
    targets, _ = _symbol_menus(packet)
    return copy.deepcopy(targets)


def symbol_fixture_menu(packet: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Expose public fixture symbols to the bounded TestWriter adapter."""
    _targets, fixtures = _symbol_menus(packet)
    return copy.deepcopy(fixtures)


_BASE_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Submit one typed reproduction specification; the host writes and runs the unittest. No test source is accepted.",
        "parameters": {
            "type": "object",
            "properties": {
                "requirement_ref": {"type": "string"},
                "expected_relation": {"type": "string"},
                "plausible_wrong_impl": {"type": "string"},
                "distinguishing_scenario": {
                    "type": "object",
                    "properties": {
                        "ticket_span": {"type": "string"}, "inputs": {"type": "string"},
                        "conditions": {"type": "string"}, "expected_observation": {"type": "string"},
                        "wrong_fix_rejected": {"type": "string"},
                    },
                    "required": ["ticket_span", "inputs", "conditions", "expected_observation", "wrong_fix_rejected"],
                    "additionalProperties": False,
                },
                "check_spec": {
                    "type": "object",
                    "properties": {
                        "setup": {
                            "type": "array", "maxItems": 16, "default": [],
                            "items": {"oneOf": [
                                {"type": "object", "properties": {"kind": {"const": "call"}, "target": {"type": "string", "enum": []}, "args": {"type": ["object", "array", "null"], "default": {}}}, "required": ["kind", "target"], "additionalProperties": False},
                                {"type": "object", "properties": {"kind": {"const": "construct"}, "target": {"type": "string", "enum": []}, "args": {"type": ["object", "array", "null"], "default": {}}}, "required": ["kind", "target"], "additionalProperties": False},
                                {"type": "object", "properties": {"kind": {"const": "async_call"}, "target": {"type": "string", "enum": []}, "args": {"type": ["object", "array", "null"], "default": {}}}, "required": ["kind", "target"], "additionalProperties": False},
                                {"type": "object", "properties": {"kind": {"const": "call_spy"}, "target": {"type": "string", "enum": []}, "args": {"type": ["object", "array", "null"], "default": {}}}, "required": ["kind", "target"], "additionalProperties": False},
                                {"type": "object", "properties": {"kind": {"const": "clock_boundary"}, "target": {"type": "string", "enum": []}, "args": {"type": ["object", "array", "null"], "default": {}}}, "required": ["kind", "target"], "additionalProperties": False},
                                {"type": "object", "properties": {"kind": {"const": "fixture"}, "fixture": {"type": "string", "enum": []}}, "required": ["kind", "fixture"], "additionalProperties": False},
                                {"type": "object", "properties": {"kind": {"const": "set_attribute"}, "object": {"type": "string", "enum": []}, "attribute": {"type": "string"}, "value": {}}, "required": ["kind", "object", "attribute", "value"], "additionalProperties": False},
                            ]},
                        },
                        "observation": {"type": "object", "properties": {"source": {"type": "string", "enum": [], "default": "$last"}, "kind": {"type": "string", "enum": ["result_path", "recorded_calls", "before_after", "effect_count"]}, "path": {"type": "array", "items": {"type": ["string", "integer"]}, "maxItems": 16, "default": []}}, "required": ["kind"], "additionalProperties": False},
                        "expected": {"type": "object", "properties": {"relation": {"type": "string", "enum": ["equals", "contains", "not_called", "raises"]}, "value": {}, "exception_type": {"type": "string", "enum": []}}, "required": ["relation"], "additionalProperties": False},
                    },
                    "required": ["setup", "observation", "expected"], "additionalProperties": False,
                },
            },
            "required": ["requirement_ref", "expected_relation", "plausible_wrong_impl", "distinguishing_scenario", "check_spec"],
            "additionalProperties": False,
        },
    },
}


def typed_check_spec_schema(packet: Mapping[str, object]) -> dict[str, object]:
    """Render exact host menus into the FLQGATE7B typed schema."""
    schema = copy.deepcopy(_BASE_SCHEMA)
    targets, fixtures = _symbol_menus(packet)
    parameters = schema["function"]["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    check_spec = properties["check_spec"]
    assert isinstance(check_spec, dict)
    check_properties = check_spec["properties"]
    assert isinstance(check_properties, dict)
    setup = check_properties["setup"]
    assert isinstance(setup, dict)
    setup_items = setup["items"]
    assert isinstance(setup_items, dict)
    variants = setup_items["oneOf"]
    assert isinstance(variants, list)
    target_ids = sorted(targets)
    for variant in variants:
        assert isinstance(variant, dict)
        variant_properties = variant["properties"]
        assert isinstance(variant_properties, dict)
        kind = variant_properties["kind"]["const"]
        if kind in {"call", "construct", "async_call", "call_spy", "clock_boundary"}:
            wanted = "class" if kind == "construct" else "function"
            variant_properties["target"]["enum"] = [item for item in target_ids if targets[item].get("kind") == wanted]
        elif kind == "fixture":
            variant_properties["fixture"]["enum"] = sorted(fixtures)
        else:
            variant_properties["object"]["enum"] = [f"${index}" for index in range(1, 17)]
    # Audit 2026-10-05 #1: llama.cpp 500s on ``enum: []`` even inside oneOf.
    # Drop a setup branch whose host menu is empty; with no branch left the
    # setup list can only be empty.
    variants[:] = [variant for variant in variants
                   if all(not isinstance(spec, dict) or "enum" not in spec or spec["enum"]
                          for spec in variant["properties"].values())]
    if not variants:
        setup.pop("items", None)
        setup["maxItems"] = 0
        setup.pop("minItems", None)
    observation = check_properties["observation"]
    assert isinstance(observation, dict)
    observation["properties"]["source"]["enum"] = [*(f"${index}" for index in range(1, 17)), "$last"]
    expected = check_properties["expected"]
    assert isinstance(expected, dict)
    expected["properties"]["exception_type"]["enum"] = sorted(_TYPED_EXCEPTION_NAMES)
    return schema


def typed_check_schema(packet: Mapping[str, object]) -> dict[str, object]:
    return typed_check_spec_schema(packet)


def _step_ref(value: object, available: Sequence[str]) -> bool:
    return isinstance(value, str) and _STEP_REFERENCE.fullmatch(value) is not None and value in available


def _scenario_valid(value: object) -> bool:
    return isinstance(value, Mapping) and all(_text(value.get(key)) for key in (
        "ticket_span", "inputs", "conditions", "expected_observation", "wrong_fix_rejected",
    ))


def _validate_typed_check_spec(arguments: object, packet: Mapping[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not isinstance(arguments, Mapping):
        return None, _error("arguments", "typed check arguments must be an object")
    required = {"requirement_ref", "expected_relation", "plausible_wrong_impl", "distinguishing_scenario", "check_spec"}
    if set(arguments) != required:
        return None, _error("arguments", "typed check fields are fixed by the host schema")
    for field in ("requirement_ref", "expected_relation", "plausible_wrong_impl"):
        if not _text(arguments.get(field)):
            return None, _error(field, "field must be a non-empty string")
    if not _scenario_valid(arguments.get("distinguishing_scenario")):
        return None, _error("distinguishing_scenario", "all ticket scenario fields are required")
    spec = arguments.get("check_spec")
    if not isinstance(spec, Mapping):
        return None, _error("check_spec", "check_spec must be an object")
    if set(spec) != {"setup", "observation", "expected"}:
        return None, _error("check_spec", "setup, observation, and expected are host-owned fields")
    targets, fixtures = _symbol_menus(packet)
    setup = spec.get("setup")
    if not isinstance(setup, list) or len(setup) > 16:
        return None, _error("check_spec.setup", "setup must contain 0 through 16 typed steps")
    normalized: list[dict[str, object]] = []
    refs: list[str] = []
    callable_step = False
    allowed_fields = {
        "call": {"kind", "target", "args"}, "construct": {"kind", "target", "args"},
        "async_call": {"kind", "target", "args"}, "call_spy": {"kind", "target", "args"},
        "clock_boundary": {"kind", "target", "args"},
        "fixture": {"kind", "fixture"}, "set_attribute": {"kind", "object", "attribute", "value"},
    }
    for index, raw in enumerate(setup):
        field = f"check_spec.setup[{index}]"
        if not isinstance(raw, Mapping):
            return None, _error(field, "setup step must be an object")
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in _SETUP_KINDS:
            return None, _error(f"{field}.kind", "unknown setup step kind", allowed=sorted(_SETUP_KINDS))
        unexpected = sorted(set(raw) - allowed_fields[kind])
        if unexpected:
            return None, _error(field, "host names setup results; omit variable names and use $N references", allowed=sorted(allowed_fields[kind]))
        if kind in {"call", "construct", "async_call", "call_spy", "clock_boundary"}:
            target = raw.get("target")
            expected_kind = "class" if kind == "construct" else "function"
            if not isinstance(target, str) or target not in targets or targets[target].get("kind") != expected_kind:
                allowed = sorted(item for item in targets if targets[item].get("kind") == expected_kind)
                return None, _error(f"{field}.target", f"{kind} target must be a host menu ID", allowed=allowed)
            if not _json_value(raw.get("args")) or not isinstance(raw.get("args"), (Mapping, list, type(None))):
                return None, _error(f"{field}.args", "args must be JSON object, array, or null")
            normalized.append({"kind": kind, "target": target, "args": raw.get("args")})
            callable_step = True
        elif kind == "fixture":
            fixture = raw.get("fixture")
            if not isinstance(fixture, str) or fixture not in fixtures:
                return None, _error(f"{field}.fixture", "fixture must be a public host fixture ID", allowed=sorted(fixtures))
            normalized.append({"kind": kind, "fixture": fixture})
        else:
            obj, attribute, value = raw.get("object"), raw.get("attribute"), raw.get("value")
            if not _step_ref(obj, refs):
                return None, _error(f"{field}.object", "object must reference a prior host step result", allowed=refs)
            if not isinstance(attribute, str) or not _IDENTIFIER.fullmatch(attribute):
                return None, _error(f"{field}.attribute", "attribute must be a valid public name")
            if not _json_value(value):
                return None, _error(f"{field}.value", "value must be JSON data")
            normalized.append({"kind": kind, "object": obj, "attribute": attribute, "value": value})
        refs.append(f"${index + 1}")
    if not callable_step:
        return None, _error("check_spec.setup", "include one call or construct step")
    observation = spec.get("observation")
    if not isinstance(observation, Mapping) or set(observation) - {"source", "kind", "path"}:
        return None, _error("check_spec.observation", "observation fields are host-owned")
    if observation.get("kind") not in {"result_path", "recorded_calls", "before_after", "effect_count"}:
        return None, _error("check_spec.observation.kind", "observation kind is not in the typed menu")
    source = observation.get("source", "$last")
    source = refs[-1] if source == "$last" else source
    if not _step_ref(source, refs):
        return None, _error("check_spec.observation.source", "source must reference a setup result", allowed=refs)
    path = observation.get("path", [])
    if not isinstance(path, list) or len(path) > 16 or any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in path):
        return None, _error("check_spec.observation.path", "path must be a bounded string/integer list")
    expected = spec.get("expected")
    if not isinstance(expected, Mapping) or set(expected) - {"relation", "value", "exception_type"}:
        return None, _error("check_spec.expected", "expected fields are host-owned")
    relation = expected.get("relation")
    if relation not in {"equals", "contains", "not_called", "raises"}:
        return None, _error("check_spec.expected.relation", "relation is not in the typed menu")
    if relation in {"equals", "contains", "not_called"} and ("value" not in expected or not _json_value(expected.get("value"))):
        return None, _error("check_spec.expected.value", "this relation requires JSON value")
    if relation == "raises":
        exception_type = expected.get("exception_type")
        if not isinstance(exception_type, str) or exception_type not in _TYPED_EXCEPTION_NAMES:
            return None, _error("check_spec.expected.exception_type", "exception must come from the host menu", allowed=sorted(_TYPED_EXCEPTION_NAMES))
        if source != refs[-1] or normalized[-1]["kind"] not in {"call", "construct"}:
            return None, _error("check_spec.observation.source", "raises must observe the final call or construct")
    normalized_spec = {
        "setup": normalized,
        "observation": {"kind": observation["kind"], "source": source, "path": path},
        "expected": dict(expected),
    }
    return normalized_spec, None


def _bound_expression(value: object, variables: Sequence[str]) -> str:
    """Render JSON data plus typed prior-output references, never source text."""
    if isinstance(value, Mapping) and set(value) == {"$ref", "path"}:
        source = value.get("$ref")
        path = value.get("path")
        if not isinstance(source, str) or not _STEP_REFERENCE.fullmatch(source):
            raise ReproSpecError("binding reference is not a setup step")
        index = int(source[1:]) - 1
        if index < 0 or index >= len(variables):
            raise ReproSpecError("binding reference is forward or unresolved")
        if not isinstance(path, list) or len(path) > 16:
            raise ReproSpecError("binding path is not bounded")
        if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in path):
            raise ReproSpecError("binding path contains an unsupported segment")
        return f"_flq_path({variables[index]}, {_literal(path)})"
    if isinstance(value, Mapping):
        if not _json_value(value):
            raise ReproSpecError("typed argument mapping is not JSON data")
        return "{" + ", ".join(
            f"{_literal(key)}: {_bound_expression(item, variables)}"
            for key, item in value.items()
        ) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_bound_expression(item, variables) for item in value) + "]"
    return _literal(value)


def _call_expression(alias: str, args: object, variables: Sequence[str] = ()) -> str:
    if args is None:
        return f"{alias}()"
    if isinstance(args, Mapping):
        return f"{alias}(**{_bound_expression(args, variables)})"
    if isinstance(args, list):
        return f"{alias}(*{_bound_expression(args, variables)})"
    raise ReproSpecError("typed call args were not normalized")


def _compile_typed_check_spec(arguments: Mapping[str, object], packet: Mapping[str, object]) -> tuple[str | None, dict[str, object] | None]:
    spec, error = _validate_typed_check_spec(arguments, packet)
    if error is not None or spec is None:
        return None, error
    targets, fixtures = _symbol_menus(packet)
    used: dict[str, Mapping[str, object]] = {}
    for step in spec["setup"]:
        assert isinstance(step, Mapping)
        for key in ("target", "fixture"):
            value = step.get(key)
            if isinstance(value, str):
                used[value] = targets.get(value, fixtures.get(value, {}))
    aliases = {identifier: f"_flq_target_{index}" for index, identifier in enumerate(sorted(used))}
    lines = [
        "import asyncio as _flq_asyncio",
        "import builtins as _flq_builtins",
        "import unittest",
        "from unittest import mock as _flq_mock",
    ]
    for identifier in sorted(used):
        entry = used[identifier]
        if (not isinstance(entry.get("module"), str) or not str(entry.get("module")).strip()
                or not isinstance(entry.get("name"), str)):
            return None, _error(
                "check_spec",
                "TARGET_EXECUTION_UNSUPPORTED: source identity has no verified module loader",
            )
        if str(entry.get("target_kind", entry.get("kind", ""))) in {"module_head", "initialization", "nested_function"}:
            return None, _error(
                "check_spec",
                "TARGET_EXECUTION_UNSUPPORTED: selected source target needs a compatible host loader",
            )
        lines.append(f"from {entry['module']} import {entry['name']} as {aliases[identifier]}")
    lines.extend([
        "", "def _flq_path(value, path):", "    for _flq_segment in path:",
        "        if isinstance(value, dict):", "            value = value[_flq_segment]",
        "        elif isinstance(value, (list, tuple)):", "            value = value[_flq_segment]",
        "        else:", "            value = getattr(value, _flq_segment)",
        "    return value", "", "class HostCompiledCheck(unittest.TestCase):",
        "    def test_typed_check_spec(self):",
    ])
    variables = [f"_flq_step_{index}" for index in range(1, len(spec["setup"]) + 1)]
    observation = spec["observation"]
    expected = spec["expected"]
    assert isinstance(observation, Mapping) and isinstance(expected, Mapping)
    source_index = int(str(observation["source"])[1:]) - 1
    relation = expected["relation"]
    spy_variables: dict[int, str] = {}
    for index, step in enumerate(spec["setup"]):
        assert isinstance(step, Mapping)
        variable = variables[index]
        kind = step["kind"]
        if kind == "fixture":
            lines.append(f"        {variable} = {aliases[str(step['fixture'])]}()")
        elif kind in {"call", "construct", "async_call", "call_spy", "clock_boundary"}:
            expression = _call_expression(aliases[str(step["target"])], step.get("args"), variables)
            if kind == "call_spy":
                spy_variable = f"_flq_spy_{index + 1}"
                spy_variables[index] = spy_variable
                lines.append(f"        {spy_variable} = _flq_mock.Mock(wraps={aliases[str(step['target'])]})")
                expression = _call_expression(spy_variable, step.get("args"), variables)
            if kind == "async_call":
                expression = f"_flq_asyncio.run({expression})"
            if relation == "raises" and index == source_index:
                lines.extend([
                    f"        with self.assertRaises(_flq_builtins.{expected['exception_type']}):",
                    f"            {variable} = {expression}",
                ])
            elif kind == "clock_boundary":
                lines.extend([
                    "        with _flq_mock.patch(\"time.monotonic\", side_effect=[0.0, 300.0, 600.0, 900.0]):",
                    f"            {variable} = {expression}",
                ])
            else:
                lines.append(f"        {variable} = {expression}")
        else:
            object_index = int(str(step["object"])[1:]) - 1
            lines.append(f"        setattr({variables[object_index]}, {_literal(step['attribute'])}, {_literal(step['value'])})")
            lines.append(f"        {variable} = {variables[object_index]}")
    if relation == "raises":
        lines.append("        _flq_observed = None")
    elif observation["kind"] == "recorded_calls":
        spy_variable = spy_variables.get(source_index)
        if spy_variable is None:
            lines.append("        _flq_observed = []")
        else:
            lines.append(
                f"        _flq_observed = [(tuple(call.args), dict(call.kwargs)) for call in {spy_variable}.call_args_list]"
            )
    elif observation["kind"] == "effect_count":
        spy_variable = spy_variables.get(source_index)
        lines.append(
            f"        _flq_observed = len({spy_variable}.call_args_list)"
            if spy_variable else "        _flq_observed = 0"
        )
    elif observation["kind"] == "before_after":
        lines.extend([
            f"        _flq_result = {variables[source_index]}",
            f"        _flq_observed = _flq_path(_flq_result, {_literal(observation['path'])})",
        ])
    else:
        lines.extend([
            f"        _flq_result = {variables[source_index]}",
            f"        _flq_observed = _flq_path(_flq_result, {_literal(observation['path'])})",
        ])
    if relation == "equals":
        lines.append(f"        self.assertEqual(_flq_observed, {_literal(expected['value'])})")
    elif relation == "contains":
        lines.append(f"        self.assertIn({_literal(expected['value'])}, _flq_observed)")
    elif relation == "not_called":
        lines.append(f"        self.assertNotIn({_literal(expected['value'])}, _flq_observed)")
    else:
        lines.append("        self.assertIsNone(_flq_observed)")
    code = "\n".join(lines) + "\n"
    if len(code) > MAX_COMPILED_SOURCE_CHARS or "__flq_" in code:
        return None, _error("check_spec", "host compiler produced an unsafe or oversized check")
    return code, None


def _revision(runtime: object) -> str:
    binding = getattr(runtime, "binding", None)
    value = binding() if callable(binding) else {}
    if isinstance(value, Mapping):
        return str(value.get("product_revision") or value.get("artifact") or "UNKNOWN")
    return "UNKNOWN"


def _run_result(raw: object) -> tuple[str, int | None, str, int, str]:
    if isinstance(raw, Mapping):
        output = "\n".join(str(raw.get(key) or "") for key in ("observation", "stdout", "stderr"))
        exit_code = raw.get("exit_code", raw.get("returncode"))
        exit_value = exit_code if type(exit_code) is int else None
        status = str(raw.get("status") or raw.get("outcome") or "").upper()
    else:
        output = str(raw or "")
        exit_value = None
        status = ""
    match = _TEST_COUNT.search(output)
    tests = int(match.group(1)) if match else (1 if status in {"PASS", "FAIL", "BEHAVIOR_FAIL"} else 0)
    if exit_value is None:
        match = re.search(r"(?:pytest|unittest)\s+exit\s+(-?\d+)", output, re.IGNORECASE)
        exit_value = int(match.group(1)) if match else None
    if _RESOURCE.search(output) or status == "RESOURCE_LIMIT":
        return "RESOURCE_LIMIT", exit_value, output, tests, "TIMEOUT"
    if _ENVIRONMENT.search(output) or status in {"ENVIRONMENT", "ENV_INVALID"}:
        return "ENVIRONMENT", exit_value, output, tests, "IMPORT"
    if exit_value == 0 or status in {"PASS", "PASSED", "GREEN", "OK"}:
        return ("PASS", exit_value, output, tests, "NONE") if tests else ("INVALID_CHECK", exit_value, output, tests, "COLLECTION")
    if _ASSERTION.search(output) or status in {"FAIL", "FAILED", "RED", "BEHAVIOR_FAIL"}:
        return ("BEHAVIOR_FAIL", exit_value, output, tests, "ASSERTION") if tests else ("INVALID_CHECK", exit_value, output, tests, "COLLECTION")
    return "INVALID_CHECK", exit_value, output, tests, "EXECUTION"


def _ticket_relevant(owner: object, requirement_ref: str, scenario: Mapping[str, object]) -> bool:
    rows = getattr(owner, "rows", {})
    row = rows.get(requirement_ref) if isinstance(rows, Mapping) else None
    if not isinstance(row, Mapping):
        return False
    ticket_span = str(scenario.get("ticket_span") or "")
    refs = {str(item) for item in row.get("ticket_refs", ())} if isinstance(row.get("ticket_refs"), Sequence) and not isinstance(row.get("ticket_refs"), (str, bytes)) else set()
    host_refs = set(str(item) for item in getattr(owner, "ticket_spans", ()) or ())
    return bool(ticket_span and (ticket_span in refs or ticket_span in host_refs or ticket_span == str(row.get("ticket_span") or "")))


def _attempt(owner: object, requirement_ref: str) -> int:
    attempts = getattr(owner, "typed_repro_attempts", None)
    if not isinstance(attempts, dict):
        attempts = {}
        setattr(owner, "typed_repro_attempts", attempts)
    value = int(attempts.get(requirement_ref, 0) or 0) + 1
    attempts[requirement_ref] = value
    return value


def _result_base(*, requirement_ref: str, attempt: int, packet: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": SCHEMA, "tool": TOOL_NAME, "requirement_ref": requirement_ref,
        "attempt": attempt, "packet_modules": len(packet.get("modules", ())),
        "packet_truncated": bool(packet.get("truncated")),
    }


def execute_typed_check(
        runtime: object,
        arguments: object,
        *,
        root: Path,
        packet: Mapping[str, object],
        runner: Callable[[str, str], object] | None,
) -> dict[str, object]:
    """Validate, compile, and execute one S3 typed check in host isolation."""
    requirement_ref = str(arguments.get("requirement_ref") or "") if isinstance(arguments, Mapping) else ""
    owner = getattr(runtime, "_idealpath_owner", lambda: None)()
    if owner is None:
        owner = getattr(runtime, "ideal_path_owner", None)
    # Count malformed submissions under a stable bucket so typed repair stays bounded
    # even when the model has not supplied a usable requirement reference yet.
    attempt_key = requirement_ref or "__missing_requirement_ref__"
    attempt = _attempt(owner, attempt_key) if owner is not None else 1
    result = _result_base(requirement_ref=requirement_ref, attempt=attempt, packet=packet)
    if owner is None or getattr(owner, "stage", None) != "S3":
        return {**result, "status": "REJECTED", "code": "IDEALPATH_S3_REQUIRED", "detail": "typed reproduction is admitted only in IDEALPATH S3"}
    if attempt > MAX_TYPED_REPAIR_ATTEMPTS:
        return {**result, "status": "REJECTED", "code": "TYPED_REPAIR_EXHAUSTED", "detail": "bounded typed repair attempts are exhausted"}
    spec, validation_error = _validate_typed_check_spec(arguments, packet)
    if validation_error is not None or spec is None:
        return {**result, "status": "INVALID_CHECK", "code": "TYPED_SPEC_ERROR", "detail": validation_error["message"] if validation_error else "invalid typed specification", "typed_error": validation_error}
    scenario = arguments["distinguishing_scenario"]
    if not isinstance(scenario, Mapping) or not _ticket_relevant(owner, requirement_ref, scenario):
        error = _error("distinguishing_scenario.ticket_span", "ticket span is not provenance-bound to the IDEALPATH1 ledger row")
        return {**result, "status": "INVALID_CHECK", "code": "TICKET_PROVENANCE_REQUIRED", "detail": error["message"], "typed_error": error}
    code, compile_error = _compile_typed_check_spec(arguments, packet)
    if compile_error is not None or code is None:
        return {**result, "status": "INVALID_CHECK", "code": "TYPED_COMPILE_ERROR", "detail": compile_error["message"] if compile_error else "host compiler rejected the spec", "typed_error": compile_error}
    check_hash = "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()
    result["check_hash"] = check_hash
    result["compiled_check_code"] = code
    if runner is None:
        return {**result, "status": "ENVIRONMENT", "code": "CHECK_RUNNER_UNAVAILABLE", "detail": "existing sandbox runner is not attached"}
    base_revision = str(getattr(owner, "base_revision", ""))
    current_revision = _revision(runtime)
    scratch_rel = f"{SCRATCH_DIR}/idealpath_{uuid.uuid4().hex}.py"
    try:
        raw = runner(code, scratch_rel)
    except Exception as error:  # runner failures never count as product behavior
        raw = {"status": "ENVIRONMENT", "observation": f"{type(error).__name__}: sandbox runner unavailable"}
    outcome, exit_code, output, test_count, failure_phase = _run_result(raw)
    receipt = owner.record_check(
        requirement_id=requirement_ref, status="PASS" if outcome == "PASS" else "FAIL",
        product_revision=current_revision, current_revision=current_revision,
        check_hash=check_hash, stdout=output if outcome == "PASS" else "",
        stderr=output if outcome != "PASS" else "", reached_observation=outcome in {"PASS", "BEHAVIOR_FAIL"},
        failure_phase=failure_phase, ticket_relevant=outcome in {"PASS", "BEHAVIOR_FAIL"}, outcome=outcome,
    )
    result.update({"status": outcome, "executed": test_count > 0, "test_count": test_count, "exit_code": exit_code, "receipt": receipt})
    if outcome == "BEHAVIOR_FAIL" and current_revision == base_revision:
        admission = owner.admit_ticket_failure(str(receipt["receipt_id"]), requirement_id=requirement_ref)
        if admission.get("accepted"):
            frozen = {
                "schema": SCHEMA, "requirement_id": requirement_ref, "check_hash": check_hash,
                "compiled_check_code": code, "typed_spec": copy.deepcopy(spec),
                "scenario": copy.deepcopy(dict(scenario)), "row_provenance": copy.deepcopy(owner.rows[requirement_ref]),
                "parent_revision": base_revision, "parent_receipt_id": receipt["receipt_id"],
            }
            owner.frozen_repros[requirement_ref] = frozen
            result.update({"code": "RED_CAPTURED", "reproduced": True, "frozen_repro": frozen, "admission": admission})
            return result
        result["code"] = "PARENT_FAILURE_NOT_ADMITTED"
    elif outcome == "PASS" and current_revision == base_revision:
        result["code"] = "PARENT_CHECK_PASSED"
    elif current_revision != base_revision:
        result["code"] = "PARENT_REVISION_REQUIRED"
    else:
        result["code"] = "CHECK_NOT_ADMITTED"
    result["reproduced"] = False
    return result


def run_frozen_repros(runtime: object, *, runner: Callable[[str, str], object] | None) -> tuple[dict[str, object], ...]:
    """Run exactly the frozen S3 source on a changed candidate after public green."""
    owner = getattr(runtime, "_idealpath_owner", lambda: None)()
    if owner is None:
        owner = getattr(runtime, "ideal_path_owner", None)
    frozen = getattr(owner, "frozen_repros", {}) if owner is not None else {}
    if not isinstance(frozen, Mapping) or not frozen or runner is None:
        return ()
    current_revision = _revision(runtime)
    results: list[dict[str, object]] = []
    for requirement_ref, repro in frozen.items():
        if not isinstance(repro, Mapping) or current_revision == repro.get("parent_revision"):
            continue
        check_hash = str(repro.get("check_hash") or "")
        prior = next((receipt for receipt in reversed(getattr(owner, "receipts", {}).values())
                      if isinstance(receipt, Mapping)
                      and receipt.get("requirement_id") == requirement_ref
                      and isinstance(receipt.get("freshness"), Mapping)
                      and receipt["freshness"].get("product_revision") == current_revision
                      and isinstance(receipt.get("result"), Mapping)
                      and receipt["result"].get("outcome") == "PASS"
                      and isinstance(receipt.get("freshness"), Mapping)
                      and receipt["freshness"].get("check_hash") == check_hash), None)
        if prior is not None:
            results.append({"requirement_ref": requirement_ref, "status": "PASS", "reused": True, "receipt": prior})
            continue
        try:
            raw = runner(str(repro.get("compiled_check_code") or ""), f"{SCRATCH_DIR}/idealpath_{uuid.uuid4().hex}.py")
        except Exception as error:
            raw = {"status": "ENVIRONMENT", "observation": f"{type(error).__name__}: sandbox runner unavailable"}
        outcome, exit_code, output, test_count, failure_phase = _run_result(raw)
        receipt = owner.record_check(
            requirement_id=str(requirement_ref), status="PASS" if outcome == "PASS" else "FAIL",
            product_revision=current_revision, current_revision=current_revision,
            check_hash=check_hash, stdout=output if outcome == "PASS" else "",
            stderr=output if outcome != "PASS" else "", reached_observation=outcome in {"PASS", "BEHAVIOR_FAIL"},
            failure_phase=failure_phase, ticket_relevant=outcome in {"PASS", "BEHAVIOR_FAIL"}, outcome=outcome,
        )
        results.append({"requirement_ref": requirement_ref, "status": outcome, "reused": False, "test_count": test_count, "exit_code": exit_code, "receipt": receipt})
    return tuple(results)


__all__ = [
    "MAX_TYPED_REPAIR_ATTEMPTS", "SCHEMA", "SYMBOL_PACKET_SCHEMA", "TOOL_NAME",
    "build_symbol_packet", "execute_typed_check", "run_frozen_repros", "symbol_target_menu",
    "symbol_fixture_menu",
    "typed_check_schema", "typed_check_spec_schema", "_compile_typed_check_spec",
    "_validate_typed_check_spec",
]
