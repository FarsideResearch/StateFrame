"""Validate explicitly trusted command tools before any provider request."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
PLACEHOLDER = re.compile(r"^\{([a-z][a-z0-9_]{0,40})\}$")
NETWORK_ORDER = {"none": 0, "loopback": 1, "full": 2}
MAX_SCHEMA_TOKENS = 1500


class CustomToolError(ValueError):
    """A custom tool definition is unsafe or malformed."""


def _fail(where: str, field: str, message: str) -> None:
    raise CustomToolError(f"{where}:{field}: {message}")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _pattern(value: str, where: str) -> None:
    if len(value) > 256 or "(?" in value or re.search(r"\\[1-9]", value):
        _fail(where, "pattern", "must use a bounded, timeout-free expression")
    try:
        re.compile(value)
        repetitions = 0
        position = 0

        def walk(depth: int = 0) -> bool:
            nonlocal repetitions, position
            if depth > 8:
                _fail(where, "pattern", "nesting exceeds 8")
            contains_repeat = False
            while position < len(value) and value[position] != ")":
                char = value[position]
                position += 1
                child_repeats = False
                if char == "(":
                    child_repeats = walk(depth + 1)
                    position += 1  # closing parenthesis, validated by re.compile
                elif char == "[":
                    if value[position:position + 1] == "^":
                        position += 1
                    if value[position:position + 1] == "]":
                        position += 1
                    while value[position] != "]":
                        position += 2 if value[position] == "\\" else 1
                    position += 1
                elif char == "\\":
                    escaped = value[position]
                    if escaped not in r"AbBdDsSwWZ\\.^$*+?{}[]()|-":
                        _fail(where, "pattern", "unsupported expression")
                    position += 1
                elif char == "|":
                    _fail(where, "pattern", "alternation is not supported")
                quantifier = re.match(r"\{(\d+)(?:,(\d*))?\}|[?*+]", value[position:])
                if quantifier:
                    repetitions += 1
                    if repetitions > 2:
                        _fail(where, "pattern", "more than 2 repetitions")
                    token = quantifier.group()
                    if token == "?":
                        upper = 1
                    elif token in {"*", "+"} or quantifier.group(2) == "":
                        upper = 101
                    else:
                        upper = int(quantifier.group(2) or quantifier.group(1))
                    if child_repeats or upper > 100:
                        _fail(where, "pattern", "repetition must be bounded at 100")
                    position += len(token)
                    if value[position:position + 1] in {"?", "+"}:
                        position += 1
                    child_repeats = True
                contains_repeat |= child_repeats
            return contains_repeat

        walk()
    except re.error as error:
        _fail(where, "pattern", f"invalid expression: {error}")


def _parameters(value: object, where: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"type", "properties", "required"}:
        _fail(where, "parameters", "expected object with type, properties, required")
    if value["type"] != "object" or not isinstance(value["properties"], dict):
        _fail(where, "parameters", "type must be object with properties")
    props = value["properties"]
    if len(props) > 8:
        _fail(where, "parameters.properties", "maximum 8 properties")
    required = value["required"]
    if not isinstance(required, list) or any(not isinstance(name, str) for name in required) or len(required) != len(set(required)):
        _fail(where, "parameters.required", "expected unique property names")
    if any(not isinstance(name, str) or name not in props for name in required):
        _fail(where, "parameters.required", "unknown property")
    for name, spec in props.items():
        place = f"{where}:parameters.properties.{name}"
        if not isinstance(name, str) or not NAME.fullmatch(name):
            _fail(where, "parameters.properties", "invalid property name")
        if not isinstance(spec, dict) or spec.get("type") not in ("string", "integer", "number", "boolean"):
            _fail(place, "type", "expected string, integer, number, or boolean")
        allowed = {"type", "description", "enum"} | ({"maxLength", "pattern"} if spec["type"] == "string" else set())
        if set(spec) - allowed:
            _fail(place, sorted(set(spec) - allowed)[0], "unsupported field")
        if "description" in spec and (not isinstance(spec["description"], str) or len(spec["description"]) > 300):
            _fail(place, "description", "expected string of at most 300 characters")
        if spec["type"] == "string":
            if "maxLength" not in spec or type(spec["maxLength"]) is not int or not 1 <= spec["maxLength"] <= 1000:
                _fail(place, "maxLength", "expected integer from 1 to 1000")
            if "pattern" in spec:
                if not isinstance(spec["pattern"], str):
                    _fail(place, "pattern", "expected string")
                _pattern(spec["pattern"], place)
        if "enum" in spec:
            enum = spec["enum"]
            if not isinstance(enum, list) or not 1 <= len(enum) <= 20 or any(
                    not _valid_type(item, spec["type"]) for item in enum):
                _fail(place, "enum", "expected 1 to 20 values of the declared type")
            if len({_canonical(item) for item in enum}) != len(enum):
                _fail(place, "enum", "duplicate values")
    return value


def _valid_type(value: object, kind: str) -> bool:
    return (type(value) is str if kind == "string" else
            type(value) is int if kind == "integer" else
            type(value) in (int, float) and math.isfinite(value) if kind == "number" else type(value) is bool)


@dataclass(frozen=True)
class CustomTool:
    name: str
    description: str
    parameters: dict
    argv: tuple[str, ...]
    definition_digest: str
    executable: Path
    executable_digest: str
    network: str
    writes: str
    replay: str
    timeout_seconds: float

    def verify_executable(self) -> None:
        try:
            current = _file_digest(self.executable)
        except OSError as error:
            raise CustomToolError(f"{self.name}: executable cannot be read: {error}") from error
        if current != self.executable_digest:
            raise CustomToolError(f"{self.name}: executable changed since load")

    def build_argv(self, arguments: Mapping[str, object]) -> list[str]:
        if not isinstance(arguments, Mapping):
            raise CustomToolError(f"{self.name}:arguments: expected object")
        props = self.parameters["properties"]
        if set(arguments) - set(props) or set(self.parameters["required"]) - set(arguments):
            raise CustomToolError(f"{self.name}:arguments: missing or unknown properties")
        for name, value in arguments.items():
            spec = props[name]
            if not _valid_type(value, spec["type"]):
                raise CustomToolError(f"{self.name}:arguments.{name}: wrong type")
            if isinstance(value, str) and "\0" in value:
                raise CustomToolError(f"{self.name}:arguments.{name}: NUL byte is not allowed")
            if spec["type"] == "string" and (len(value) > spec["maxLength"] or
                    ("pattern" in spec and re.fullmatch(spec["pattern"], value) is None)):
                raise CustomToolError(f"{self.name}:arguments.{name}: violates string constraint")
            if "enum" in spec and value not in spec["enum"]:
                raise CustomToolError(f"{self.name}:arguments.{name}: outside enum")
        result = [str(self.executable)]
        for item in self.argv[1:]:
            match = PLACEHOLDER.fullmatch(item)
            if match:
                name = match.group(1)
                if name not in arguments:
                    raise CustomToolError(f"{self.name}:arguments.{name}: missing argv value")
                result.append(str(arguments[name]).lower() if type(arguments[name]) is bool else str(arguments[name]))
            else:
                result.append(item)
        return result


@dataclass(frozen=True)
class CustomToolSet:
    tools: tuple[CustomTool, ...]
    digest: str

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": tool.name,
                "description": tool.description,
                "parameters": {**tool.parameters, "additionalProperties": False}}} for tool in self.tools]


def load_tool_files(paths: Sequence[Path], *, builtin_names: frozenset[str],
                    shell_network: str, tokenizer: Callable[[str], int] | None = None) -> CustomToolSet:
    if not isinstance(shell_network, str) or shell_network not in NETWORK_ORDER:
        raise CustomToolError("shell_network: invalid network mode")
    found: list[CustomTool] = []
    names = set(builtin_names)
    for path in paths:
        path = Path(path).expanduser().resolve()
        try:
            if path.stat().st_size > 1 << 20:
                _fail(str(path), "tools", "file exceeds 1 MiB")
            raw = path.read_bytes()
            document = json.loads(raw, parse_constant=_json_constant, object_pairs_hook=_json_object)
        except CustomToolError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise CustomToolError(f"{path}: cannot read tools JSON: {error}") from error
        if not isinstance(document, dict) or set(document) != {"schema", "tools"} or document["schema"] != "stateframe-tools/1":
            _fail(str(path), "schema", "expected stateframe-tools/1 with tools")
        if not isinstance(document["tools"], list):
            _fail(str(path), "tools", "expected list")
        for index, definition in enumerate(document["tools"]):
            where = f"{path}:{definition.get('name', index) if isinstance(definition, dict) else index}"
            if not isinstance(definition, dict):
                _fail(where, "definition", "expected object")
            allowed = {"name", "description", "parameters", "argv", "network", "writes", "replay", "timeout_seconds"}
            if set(definition) - allowed:
                _fail(where, sorted(set(definition) - allowed)[0], "unsupported field")
            name = definition.get("name")
            if not isinstance(name, str) or not NAME.fullmatch(name):
                _fail(where, "name", "expected ^[a-z][a-z0-9_]{1,40}$")
            if name in names:
                _fail(where, "name", "duplicates a built-in or earlier custom tool")
            names.add(name)
            description = definition.get("description")
            if not isinstance(description, str) or not description or len(description) > 300:
                _fail(where, "description", "expected 1 to 300 characters")
            params = _parameters(definition.get("parameters"), where)
            argv = definition.get("argv")
            if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item or "\0" in item for item in argv):
                _fail(where, "argv", "expected nonempty string items")
            if len(argv) > 32 or sum(len(item) for item in argv) > 16384:
                _fail(where, "argv", "maximum 32 items and 16384 characters")
            for item in argv:
                if "{" in item or "}" in item:
                    match = PLACEHOLDER.fullmatch(item)
                    if not match or match.group(1) not in params["properties"]:
                        _fail(where, "argv", "placeholders must occupy a whole item and name a property")
                    if match.group(1) not in params["required"]:
                        _fail(where, "argv", "placeholder properties must be required")
            if PLACEHOLDER.fullmatch(argv[0]):
                _fail(where, "argv", "executable cannot be a placeholder")
            executable = shutil.which(argv[0]) if not Path(argv[0]).is_absolute() else argv[0]
            if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
                _fail(where, "argv", "executable not found")
            executable = Path(executable).resolve()
            network = definition.get("network", "none")
            if not isinstance(network, str) or network not in NETWORK_ORDER:
                _fail(where, "network", "expected none, loopback, or full")
            if NETWORK_ORDER[network] > NETWORK_ORDER[shell_network]:
                _fail(where, "network", "exceeds --shell-network")
            writes = definition.get("writes", "none")
            if writes not in ("none", "workspace"):
                _fail(where, "writes", "expected none or workspace")
            replay = definition.get("replay", "external")
            if replay not in ("local", "external"):
                _fail(where, "replay", "expected local or external")
            if network == "full" and replay == "local":
                _fail(where, "replay", "full network tools must use external replay")
            timeout = definition.get("timeout_seconds")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
                _fail(where, "timeout_seconds", "expected number from 0 to 3600")
            try:
                executable_digest = _file_digest(executable)
            except OSError as error:
                _fail(where, "argv", f"cannot hash executable: {error}")
            found.append(CustomTool(name, description, params, tuple(argv), _digest(_canonical(definition)),
                                    executable, executable_digest, network, writes, replay, float(timeout)))
    result = CustomToolSet(tuple(found), _digest(_canonical([
        (tool.definition_digest, tool.executable_digest) for tool in found])))
    rendered = json.dumps(result.schemas(), ensure_ascii=False, separators=(",", ":"))
    count = tokenizer(rendered) if tokenizer else (len(rendered.encode()) + 3) // 4
    if count > MAX_SCHEMA_TOKENS:
        raise CustomToolError(f"custom tool schemas: {count} tokens exceeds {MAX_SCHEMA_TOKENS}")
    return result
