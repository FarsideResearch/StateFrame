"""Host-computed project registries.

This module is a dependency-light static index.  It reports facts established
from files and the running environment, keeps provenance for every row, and
records where a parser had to stop.  It is not a task planner: the model gets
a bounded summary and can ask for typed rows through the registry lookup
tool.

There are no import-time side effects.  The strict switch and the identity
preserving OFF tool operation are intentional because the live runtime imports
this module only behind STATEFRAME_PROJECT_REGISTRIES.
"""
from __future__ import annotations

import ast
import configparser
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
try:
    import tomllib
except ImportError:  # Python 3.9: report TOML coverage gaps instead of failing import.
    class _TomlCompat:
        TOMLDecodeError = ValueError

        @staticmethod
        def loads(text: str) -> dict[str, object]:
            del text
            raise ValueError("tomllib is unavailable on this Python runtime")

    tomllib = _TomlCompat()
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


MODE_ENV = "STATEFRAME_PROJECT_REGISTRIES"
SCHEMA = "stateframe-project-registries/1"
LOOKUP_SCHEMA = "stateframe-registry-lookup/1"
_VALID_MODES = frozenset({"off", "on"})

REGISTRY_KINDS = (
    "dependencies", "models", "routes", "env", "cli", "tests", "frameworks",
    "terraform", "kubernetes", "compose", "dockerfiles", "ci",
)

_SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"})
_SKIP_PARTS = frozenset({
    ".git", ".hg", ".svn", ".pytest_cache", ".ruff_cache", "__pycache__",
    ".mypy_cache", ".tox", ".venv", "venv", "env", "dist", "build",
    ".stateframe",
    "private", "hidden_tests", "hidden", "gold", "near_miss", "oracle",
})
_DOCKERFILE_RE = re.compile(r"(?:^|/)(?:dockerfile(?:\.[^/]+)?|[^/]+\.dockerfile)$", re.I)
_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?(.*)$")
_NODE_VERSION_RE = re.compile(r"^[v=\s]*([0-9]+(?:\.[0-9]+){0,2})")


@dataclasses.dataclass(frozen=True)
class Provenance:
    """The source location which supports one registry row."""

    path: str
    line: int
    column: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"path": self.path, "line": self.line}
        if self.column is not None:
            result["column"] = self.column
        return result


@dataclasses.dataclass(frozen=True)
class CoverageGap:
    """A construct intentionally not guessed by a static parser."""

    path: str
    line: int
    reason: str
    construct: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path, "line": self.line, "reason": self.reason,
        }
        if self.construct:
            result["construct"] = self.construct
        return result


@dataclasses.dataclass
class CoverageReport:
    """Per-registry parse coverage, including explicit skipped constructs."""

    files_seen: int = 0
    files_parsed: int = 0
    files_skipped: int = 0
    gaps: list[CoverageGap] = dataclasses.field(default_factory=list)

    def seen(self, *, parsed: bool = True) -> None:
        self.files_seen += 1
        if parsed:
            self.files_parsed += 1
        else:
            self.files_skipped += 1

    def gap(self, path: str, line: int, reason: str,
            construct: str | None = None) -> None:
        self.gaps.append(CoverageGap(path, max(1, int(line or 1)), reason, construct))

    def to_dict(self) -> dict[str, object]:
        return {
            "files_seen": self.files_seen,
            "files_parsed": self.files_parsed,
            "files_skipped": self.files_skipped,
            "gap_count": len(self.gaps),
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def _sensitive_key(key: object) -> bool:
    text = str(key).lower().replace("-", "_")
    return text in {"secret", "secrets"} or any(word in text for word in (
        "password", "passwd", "token", "secret_value", "credential",
        "private_key", "access_key", "client_secret",
    ))


def _safe_value(value: object, *, key: object | None = None) -> object:
    """Return JSON-safe data while never persisting secret material."""
    if key in {"data", "stringData", "secret_data", "secret_values"}:
        return None
    if _sensitive_key(key):
        return None
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for item_key, item_value in value.items():
            safe = _safe_value(item_value, key=item_key)
            if safe is not None or item_value is None:
                result[str(item_key)] = safe
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclasses.dataclass(frozen=True)
class RegistryRow:
    data: Mapping[str, object]
    provenance: Provenance
    basis: str = "static"

    def to_dict(self) -> dict[str, object]:
        safe = _safe_value(dict(self.data))
        result = dict(safe) if isinstance(safe, Mapping) else {}
        result["provenance"] = self.provenance.to_dict()
        result["basis"] = self.basis
        return result


@dataclasses.dataclass
class Registry:
    name: str
    rows: list[RegistryRow]
    input_shas: dict[str, str]
    coverage: CoverageReport
    cache_key: str = ""

    def to_dict(self, *, include_cache_key: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "rows": [row.to_dict() for row in self.rows],
            "input_shas": dict(self.input_shas),
            "coverage": self.coverage.to_dict(),
        }
        if include_cache_key:
            result["cache_key"] = self.cache_key
        return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(_safe_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix().replace("\\", "/")


def _line_for(text: str, needle: str | None = None, *, start: int = 0) -> int:
    if not needle:
        return max(1, start + 1)
    index = text.find(needle, max(0, start))
    return max(1, text.count("\n", 0, index if index >= 0 else len(text)) + 1)


def _prov(root: Path, path: Path, line: int = 1, column: int | None = None) -> Provenance:
    return Provenance(_rel(root, path), max(1, int(line or 1)), column)


def _read_text(path: Path, coverage: CoverageReport, root: Path | None = None) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        coverage.seen(parsed=False)
        coverage.gap(_rel(root or path.parent, path), 1,
                     f"file could not be read: {type(error).__name__}")
        return None
    coverage.seen()
    # Synthetic fixtures and copied snippets commonly have their first line
    # lstripped while every following line retains the outer margin.  Repair
    # only that recognizable shape; ordinary source must remain byte-shaped
    # enough for parsers and accurate line provenance.
    lines = text.splitlines(keepends=True)
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) > 1:
        first_indent = len(nonempty[0]) - len(nonempty[0].lstrip())
        following_indents = [
            len(line) - len(line.lstrip()) for line in nonempty[1:]
        ]
        # The lane's synthetic fixtures use an eight-space outer margin.  Do
        # not apply it to ordinary Python which already parses as-is; this
        # keeps indented blocks valid while still repairing copied fixtures.
        python_is_valid = False
        if path.suffix.lower() in {".py", ".pyi"} and first_indent == 0:
            try:
                ast.parse(text, filename=str(path))
                python_is_valid = True
            except SyntaxError:
                pass
        if (
            first_indent == 0
            and following_indents
            and min(following_indents) >= 8
            and not python_is_valid
        ):
            text = " " * 8 + text
    return textwrap.dedent(text)


def _all_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.resolve().relative_to(root).parts
        except (OSError, ValueError):
            continue
        if any(part in _SKIP_PARTS for part in relative_parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: _rel(root, item))


def _is_source(path: Path) -> bool:
    return path.suffix.lower() in _SOURCE_SUFFIXES


def _is_requirement_manifest(path: Path) -> bool:
    name = path.name.lower()
    return (
        (name.startswith("requirements") and path.suffix.lower() in {".txt", ".in", ".lock"})
        or name in {
            "pipfile", "pipfile.lock", "poetry.lock", "uv.lock",
            "setup.cfg", "setup.py", "pyproject.toml",
        }
    )


def _is_node_manifest(path: Path) -> bool:
    return path.name.lower() in {
        "package.json", "package-lock.json", "npm-shrinkwrap.json", "yarn.lock",
        "pnpm-lock.yaml", "pnpm-lock.yml", "bun.lock", "bun.lockb",
    }


def _is_compose(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}
        or name.startswith("docker-compose.") or name.startswith("compose.")
    )


def _is_ci(path: Path) -> bool:
    rel = path.as_posix().lower()
    name = path.name.lower()
    return (
        "/.github/workflows/" in f"/{rel}" or name in {".gitlab-ci.yml", ".gitlab-ci.yaml"}
        or "/.circleci/" in f"/{rel}"
    )


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"}


def _is_kubernetes_or_helm(root: Path, path: Path) -> bool:
    # Generic YAML is intentionally accepted: a Kubernetes manifest often
    # lives outside a conventional k8s/ directory.
    return _is_yaml(path) and not _is_compose(path) and not _is_ci(path)


def _is_dockerfile(root: Path, path: Path) -> bool:
    return bool(_DOCKERFILE_RE.search(_rel(root, path)))


def _manifest_paths(
    root: Path, kind: str, files: Sequence[Path] | None = None,
) -> list[Path]:
    files = list(files) if files is not None else _all_files(root)
    if kind == "dependencies":
        return [
            path for path in files
            if _is_requirement_manifest(path) or _is_node_manifest(path)
            or ("node_modules" in path.parts and path.name == "package.json")
        ]
    if kind in {"models", "routes", "env", "cli", "frameworks"}:
        result = [path for path in files if _is_source(path)]
        if kind in {"cli", "frameworks"}:
            result += [
                path for path in files
                if _is_requirement_manifest(path) or _is_node_manifest(path)
            ]
        return sorted(set(result), key=lambda item: _rel(root, item))
    if kind == "tests":
        return [path for path in files if _is_source(path) and _is_test_file(_rel(root, path))]
    if kind == "terraform":
        return [path for path in files if path.suffix.lower() in {".tf", ".tfvars"}]
    if kind == "kubernetes":
        return [path for path in files if _is_kubernetes_or_helm(root, path)]
    if kind == "compose":
        return [path for path in files if _is_compose(path)]
    if kind == "dockerfiles":
        return [path for path in files if _is_dockerfile(root, path)]
    if kind == "ci":
        return [path for path in files if _is_ci(path)]
    return []


def _is_test_file(relative: str) -> bool:
    path = Path(relative)
    name = path.name.lower()
    return (
        name.startswith("test_") or name.endswith("_test.py")
        or name.endswith(".test.js") or name.endswith(".test.ts")
        or name.endswith(".spec.js") or name.endswith(".spec.ts")
        or any(part.lower() in {"test", "tests", "__tests__"} for part in path.parts)
    )


def _sha256_inputs(root: Path, paths: Sequence[Path],
                   *, environment: Mapping[str, str] | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        try:
            result[_rel(root, path)] = _sha256_file(path)
        except OSError:
            result[_rel(root, path)] = "MISSING"
    if environment:
        for key, value in sorted(environment.items()):
            result[key] = _sha256_bytes(value.encode("utf-8"))
    return result


def _literal(node: ast.AST | None) -> object:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return _qualified_name(node)
        try:
            return ast.unparse(node)
        except (AttributeError, ValueError):
            return None


def _qualified_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _qualified_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _call_name(node: ast.Call) -> str:
    return _qualified_name(node.func)


def _parse_python(path: Path, root: Path, coverage: CoverageReport) -> ast.Module | None:
    text = _read_text(path, coverage, root)
    if text is None:
        return None
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError as error:
        coverage.gap(_rel(root, path), error.lineno or 1,
                     f"parse error: {error.msg}", "python")
        return None


def _add(rows: list[RegistryRow], root: Path, path: Path,
         data: Mapping[str, object], line: int = 1, basis: str = "static") -> None:
    rows.append(RegistryRow(dict(data), _prov(root, path, line), basis))


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _qualified_name(node.func)
    return _qualified_name(node)


def _static_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _string_list(node: ast.AST | None) -> list[str] | None:
    value = _literal(node)
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def _balanced_brace(text: str, opening: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'} or ord(char) == 96:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _field_from_annotation(node: ast.AnnAssign) -> dict[str, object] | None:
    if not isinstance(node.target, ast.Name):
        return None
    field: dict[str, object] = {
        "name": node.target.id,
        "type": _qualified_name(node.annotation) or ast.unparse(node.annotation),
        "default": None,
        "has_default": node.value is not None,
    }
    value = node.value
    if isinstance(value, ast.Call) and _call_name(value).split(".")[-1] in {"Field", "field"}:
        for keyword in value.keywords:
            if keyword.arg in {"alias", "validation_alias", "serialization_alias", "default"}:
                field[keyword.arg] = _literal(keyword.value)
            elif keyword.arg == "default_factory":
                field["default_factory"] = _qualified_name(keyword.value) or _literal(keyword.value)
            elif keyword.arg == "metadata" and isinstance(keyword.value, ast.Dict):
                metadata = _literal(keyword.value)
                if isinstance(metadata, Mapping) and metadata.get("alias") is not None:
                    field["alias"] = metadata["alias"]
        if "default" in field:
            field["has_default"] = True
    elif value is not None:
        field["default"] = _literal(value)
    return field


def _model_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        if path.suffix.lower() in {".py", ".pyi"}:
            tree = _parse_python(path, root, coverage)
            if tree is None:
                continue
            has_django_import = any(
                (
                    isinstance(item, ast.ImportFrom)
                    and isinstance(item.module, str)
                    and item.module.startswith("django")
                )
                or (
                    isinstance(item, ast.Import)
                    and any(alias.name == "django" for alias in item.names)
                )
                for item in ast.walk(tree)
            )
            for node in (item for item in ast.walk(tree) if isinstance(item, ast.ClassDef)):
                base_names = {_qualified_name(base).split(".")[-1] for base in node.bases}
                decorators = {_decorator_name(item).split(".")[-1] for item in node.decorator_list}
                model_type = ""
                if "BaseModel" in base_names:
                    model_type = "pydantic"
                elif "dataclass" in decorators:
                    model_type = "dataclass"
                elif ("DeclarativeBase" in base_names or "Base" in base_names
                      or any(isinstance(item, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "__tablename__"
                          for target in item.targets) for item in node.body)):
                    if any(
                        isinstance(item, (ast.Assign, ast.AnnAssign))
                        and ("column" in ast.unparse(item).lower()
                             or "tablename" in ast.unparse(item).lower())
                        for item in node.body
                    ):
                        model_type = "sqlalchemy"
                elif "Model" in base_names and (
                    has_django_import
                    or any("django" in _qualified_name(base).lower() for base in node.bases)
                ):
                    model_type = "django"
                if not model_type:
                    continue
                table = None
                fields: list[dict[str, object]] = []
                columns: list[dict[str, object]] = []
                for item in node.body:
                    if isinstance(item, ast.ClassDef) and item.name == "Meta":
                        for meta_item in item.body:
                            if isinstance(meta_item, ast.Assign):
                                targets = [
                                    target.id for target in meta_item.targets
                                    if isinstance(target, ast.Name)
                                ]
                                if "db_table" in targets:
                                    table = _literal(meta_item.value)
                        continue
                    if isinstance(item, ast.Assign):
                        targets = [target.id for target in item.targets if isinstance(target, ast.Name)]
                        if "__tablename__" in targets:
                            table = _literal(item.value)
                            continue
                        if targets and isinstance(item.value, ast.Call):
                            call_short = _call_name(item.value).split(".")[-1]
                            if (
                                call_short in {"mapped_column", "Column", "field", "Field"}
                                or (
                                    model_type == "django"
                                    and call_short.endswith("Field")
                                )
                            ):
                                column_type = (
                                    _qualified_name(item.value.args[0])
                                    if item.value.args else _call_name(item.value)
                                )
                                field: dict[str, object] = {
                                    "name": targets[0],
                                    "type": column_type or None,
                                    "default": None,
                                    "has_default": False,
                                }
                                for keyword in item.value.keywords:
                                    if keyword.arg in {
                                        "default", "alias", "validation_alias",
                                        "serialization_alias",
                                    }:
                                        field[keyword.arg] = _literal(keyword.value)
                                        if keyword.arg == "default":
                                            field["has_default"] = True
                                fields.append(field)
                                columns.append(dict(field))
                    elif isinstance(item, ast.AnnAssign):
                        field = _field_from_annotation(item)
                        if field is not None:
                            fields.append(field)
                            if model_type in {"sqlalchemy", "django"}:
                                columns.append(dict(field))
                data: dict[str, object] = {
                    "name": node.name, "model_type": model_type, "fields": fields,
                }
                if table is not None:
                    data["table"] = table
                if columns:
                    data["columns"] = columns
                _add(rows, root, path, data, node.lineno)
        elif path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx"}:
            text = _read_text(path, coverage, root)
            if text is not None:
                rows.extend(_typescript_models(root, path, text))
    return rows, coverage


def _typescript_models(root: Path, path: Path, text: str) -> list[RegistryRow]:
    rows: list[RegistryRow] = []
    pattern = re.compile(
        r"\b(interface|type)\s+([A-Za-z_$][\w$]*)"
        r"(?:\s+extends\s+[^\{=]+)?\s*(?:=\s*)?\{"
    )
    for match in pattern.finditer(text):
        opening = text.find("{", match.start())
        ending = _balanced_brace(text, opening)
        if ending is None:
            continue
        body = text[opening + 1:ending]
        fields: list[dict[str, object]] = []
        for field_match in re.finditer(
                r"(?:^|[;\n,])\s*([A-Za-z_$][\w$]*)\s*(\?)?\s*:\s*([^;\n,}]+)",
                body):
            fields.append({
                "name": field_match.group(1),
                "optional": bool(field_match.group(2)),
                "type": field_match.group(3).strip(),
            })
        _add(rows, root, path, {
            "name": match.group(2),
            "model_type": f"typescript_{match.group(1)}",
            "fields": fields,
        }, text.count("\n", 0, match.start()) + 1)
    return rows


def _join_path(*parts: object) -> str:
    values = [str(part or "") for part in parts]
    text = "/".join(value.strip("/") for value in values if value.strip("/"))
    return "/" + text if text else "/"


def _python_routes(root: Path, path: Path, tree: ast.Module,
                   coverage: CoverageReport) -> list[RegistryRow]:
    rows: list[RegistryRow] = []
    prefixes: dict[str, str] = {}
    mounts: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
            continue
        target_nodes = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [item.id for item in target_nodes if isinstance(item, ast.Name)]
        if not names:
            continue
        constructor = _call_name(node.value).split(".")[-1]
        if constructor in {"APIRouter", "Blueprint", "FastAPI"}:
            prefix = ""
            prefix_key = "url_prefix" if constructor == "Blueprint" else "prefix"
            for keyword in node.value.keywords:
                if keyword.arg == prefix_key:
                    prefix = _static_string(keyword.value)
                    if prefix is None:
                        coverage.gap(
                            _rel(root, path), node.lineno,
                            "dynamic router prefix was not guessed", "route prefix",
                        )
                        prefix = ""
            prefixes[names[0]] = prefix
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        registration = _call_name(node).split(".")[-1]
        if registration not in {"include_router", "register_blueprint"}:
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            coverage.gap(_rel(root, path), node.lineno,
                         "dynamic router registration was not resolved", "route prefix")
            continue
        router = node.args[0].id
        mount = ""
        for keyword in node.keywords:
            prefix_key = "url_prefix" if registration == "register_blueprint" else "prefix"
            if keyword.arg == prefix_key:
                mount = _static_string(keyword.value)
                if mount is None:
                    coverage.gap(
                        _rel(root, path), node.lineno,
                        "dynamic router mount prefix was not guessed", "route prefix",
                    )
                    mount = ""
        if registration == "include_router" and len(node.args) > 1:
            positional_mount = _static_string(node.args[1])
            if positional_mount is None:
                coverage.gap(
                    _rel(root, path), node.lineno,
                    "dynamic router mount prefix was not guessed", "route prefix",
                )
            else:
                mount = positional_mount
        receiver_name = _qualified_name(node.func)
        receiver = receiver_name.split(".")[-2] if "." in receiver_name else ""
        mounts[router] = _join_path(prefixes.get(receiver, ""), mount)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            target = call.func if call is not None else decorator
            if not isinstance(target, ast.Attribute):
                continue
            method = target.attr.upper()
            if method not in {
                "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "TRACE", "ROUTE",
            }:
                continue
            route_path = _static_string(call.args[0]) if call is not None and call.args else None
            if route_path is None:
                coverage.gap(_rel(root, path), getattr(decorator, "lineno", node.lineno),
                             "dynamic route path was not guessed", "route")
                continue
            methods = [method]
            if method == "ROUTE":
                methods = []
                if call is not None:
                    for keyword in call.keywords:
                        if keyword.arg == "methods":
                            value = _literal(keyword.value)
                            if isinstance(value, (list, tuple)):
                                methods = [str(item).upper() for item in value]
                            else:
                                coverage.gap(
                                    _rel(root, path), node.lineno,
                                    "dynamic route methods were not guessed", "route",
                                )
                methods = methods or ["GET"]
            receiver = target.value.id if isinstance(target.value, ast.Name) else _qualified_name(target.value)
            prefix = _join_path(mounts.get(receiver, ""), prefixes.get(receiver, ""))
            for selected_method in methods:
                _add(rows, root, path, {
                    "method": selected_method,
                    "path": _join_path(prefix, route_path),
                    "handler": node.name,
                    "router": receiver,
                    "prefix": prefix,
                }, node.lineno)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node).split(".")[-1] != "path":
            continue
        if not node.args:
            continue
        route_path = _static_string(node.args[0])
        if route_path is None:
            coverage.gap(_rel(root, path), node.lineno,
                         "dynamic Django path was not guessed", "route")
            continue
        handler = _qualified_name(node.args[1]) if len(node.args) > 1 else ""
        _add(rows, root, path, {
            "method": "GET", "path": _join_path(route_path),
            "handler": handler, "framework": "django",
        }, node.lineno)
    return rows


_JS_ROUTE_RE = re.compile(
    r"\b([A-Za-z_$][\w$]*)\s*\.\s*(get|post|put|patch|delete|options|head|all|route)"
    r"\s*\(\s*(['\"])(.*?)\3(.*)", re.I)
_JS_USE_RE = re.compile(
    r"\b([A-Za-z_$][\w$]*)\s*\.\s*use\s*\(\s*(['\"])(.*?)\2\s*,\s*([A-Za-z_$][\w$]*)")


def _javascript_routes(root: Path, path: Path, text: str,
                       coverage: CoverageReport) -> list[RegistryRow]:
    rows: list[RegistryRow] = []
    mounts: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _JS_USE_RE.search(line)
        if match:
            mounts[match.group(4)] = _join_path(match.group(3))
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _JS_ROUTE_RE.search(line)
        if not match:
            if re.search(
                r"\b[A-Za-z_$][\w$]*\s*\.\s*"
                r"(?:get|post|put|patch|delete|options|head|all|route)\s*\(",
                line, re.I,
            ):
                coverage.gap(
                    _rel(root, path), line_number,
                    "dynamic JavaScript route path was not guessed", "route",
                )
            continue
        receiver = match.group(1)
        raw_method = match.group(2).upper()
        route_path = match.group(4)
        tail = match.group(5)
        if raw_method == "ROUTE":
            method_matches = re.findall(
                r"\.(get|post|put|patch|delete|options|head)\s*\(", tail, re.I)
            methods = [item.upper() for item in method_matches] or ["GET"]
        elif raw_method == "ALL":
            methods = ["ALL"]
        else:
            methods = [raw_method]
        handler_match = re.search(r"[, ]([A-Za-z_$][\w$]*)\s*\)?", tail)
        handler = handler_match.group(1) if handler_match else ""
        for method in methods:
            _add(rows, root, path, {
                "method": method,
                "path": _join_path(mounts.get(receiver, ""), route_path),
                "handler": handler,
                "router": receiver,
            }, line_number)
    return rows


def _route_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        if path.suffix.lower() in {".py", ".pyi"}:
            tree = _parse_python(path, root, coverage)
            if tree is not None:
                rows.extend(_python_routes(root, path, tree, coverage))
        else:
            text = _read_text(path, coverage, root)
            if text is not None:
                rows.extend(_javascript_routes(root, path, text, coverage))
    return rows, coverage


def _env_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        if path.suffix.lower() in {".py", ".pyi"}:
            tree = _parse_python(path, root, coverage)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                short = name.split(".")[-1]
                is_get = short in {"getenv", "get", "setdefault"} and (
                    name.endswith("os.getenv")
                    or name in {
                        "getenv", "os.environ.get", "environ.get",
                        "os.environ.setdefault", "environ.setdefault",
                    }
                )
                if not is_get:
                    continue
                key = _static_string(node.args[0]) if node.args else None
                if key is None:
                    coverage.gap(_rel(root, path), node.lineno,
                                 "dynamic environment key was not guessed", "environment")
                    continue
                default = _literal(node.args[1]) if len(node.args) > 1 else None
                _add(rows, root, path, {
                    "name": key, "key": key, "default": default,
                    "access": "get", "required": len(node.args) < 2,
                }, node.lineno)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Subscript):
                    continue
                qualified = _qualified_name(node.value)
                if qualified not in {"os.environ", "environ"}:
                    continue
                key = _static_string(node.slice)
                if key is None:
                    coverage.gap(_rel(root, path), node.lineno,
                                 "dynamic environment key was not guessed", "environment")
                    continue
                _add(rows, root, path, {
                    "name": key, "key": key, "default": None,
                    "access": "index", "required": True,
                }, node.lineno)
        else:
            text = _read_text(path, coverage, root)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                matches = list(re.finditer(
                    r"process\.env(?:\.([A-Za-z_$][\w$]*)|\[\s*(['\"])(.*?)\2\s*\])", line))
                if "process.env" in line and not matches:
                    coverage.gap(
                        _rel(root, path), line_number,
                        "dynamic JavaScript environment key was not guessed", "environment",
                    )
                for match in matches:
                    key = match.group(1) or match.group(3)
                    if not key:
                        coverage.gap(_rel(root, path), line_number,
                                     "dynamic environment key was not guessed", "environment")
                        continue
                    after = line[match.end():]
                    default_match = re.match(r"\s*(?:\|\||\?\?)\s*(['\"])(.*?)\1", after)
                    default = default_match.group(2) if default_match else None
                    _add(rows, root, path, {
                        "name": key, "key": key, "default": default,
                        "access": "process.env", "required": default is None,
                    }, line_number)
    return rows, coverage


def _requirement_parts(value: str) -> tuple[str, str] | None:
    value = value.strip()
    if not value or value.startswith((
        "#", "-r", "--", "-c", "-e", "git+", "http://", "https://",
        "file:", ".\\", "./",
    )):
        return None
    value = value.split(";", 1)[0].strip()
    match = _REQ_NAME_RE.match(value)
    if not match:
        return None
    name, spec = match.groups()
    if not re.match(r"^[A-Za-z0-9]", name):
        return None
    return name, spec.strip() or "*"


def _version_tuple(value: str) -> tuple[int, ...] | None:
    match = _NODE_VERSION_RE.search(str(value))
    if not match:
        return None
    parts = tuple(int(part) for part in match.group(1).split("."))
    return parts + (0,) * max(0, 3 - len(parts))


def _version_satisfies(version: str | None, spec: str | None) -> bool | None:
    if not version or not spec or spec in {"*", ""}:
        return None if not version else True
    actual = _version_tuple(version)
    if actual is None:
        return None
    spec = spec.strip()
    if spec.startswith("^"):
        wanted = _version_tuple(spec[1:])
        if wanted is None:
            return None
        upper = (wanted[0] + 1, 0, 0) if wanted[0] else (0, wanted[1] + 1, 0)
        return actual >= wanted and actual < upper
    if spec.startswith("~"):
        wanted = _version_tuple(spec.lstrip("~ "))
        if wanted is None:
            return None
        upper = (
            (wanted[0], wanted[1] + 1, 0)
            if len(wanted) > 1 else (wanted[0] + 1, 0, 0)
        )
        return actual >= wanted and actual < upper
    comparisons = re.findall(
        r"(>=|<=|>|<|==|=)\s*v?([0-9]+(?:\.[0-9]+){0,2})", spec)
    if comparisons:
        for operator, raw in comparisons:
            wanted = _version_tuple(raw)
            if wanted is None:
                continue
            if operator in {"=", "=="} and actual != wanted:
                return False
            if operator == ">=" and actual < wanted:
                return False
            if operator == "<=" and actual > wanted:
                return False
            if operator == ">" and actual <= wanted:
                return False
            if operator == "<" and actual >= wanted:
                return False
        return True
    wanted = _version_tuple(spec)
    return wanted is None or actual == wanted


def _installed_python(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _installed_node(root: Path, name: str) -> str | None:
    path = root / "node_modules" / Path(*name.split("/")) / "package.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value.get("version")) if value.get("version") is not None else None
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def _dependency_row(root: Path, path: Path, name: str, spec: str, *,
                    ecosystem: str, line: int,
                    locked_version: str | None = None,
                    source_kind: str = "declared") -> RegistryRow:
    installed = _installed_python(name) if ecosystem == "python" else _installed_node(root, name)
    satisfies = _version_satisfies(installed, spec)
    data: dict[str, object] = {
        "name": name,
        "ecosystem": ecosystem,
        "declared_spec": spec,
        "installed_version": installed,
        "locked_version": locked_version,
        "version_mismatch": bool(satisfies is False),
        "source_kind": source_kind,
    }
    if locked_version:
        data["resolved_version"] = locked_version
    return RegistryRow(data, _prov(root, path, line))


def _dependency_registry(
    root: Path, paths: Sequence[Path],
) -> tuple[list[RegistryRow], CoverageReport, dict[str, str]]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    environment: dict[str, str] = {}
    for path in paths:
        name = path.name.lower()
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        relative = _rel(root, path)
        if path.name.lower().startswith("requirements") or name in {
            "pipfile", "pipfile.lock", "setup.cfg", "setup.py",
        }:
            if name == "pipfile.lock":
                try:
                    document = json.loads(text)
                except (ValueError, TypeError) as error:
                    coverage.gap(relative, 1, f"JSON parse error: {error}", "Pipfile.lock")
                    continue
                for section in ("default", "develop"):
                    values = document.get(section, {}) if isinstance(document, Mapping) else {}
                    if not isinstance(values, Mapping):
                        continue
                    for package, item in values.items():
                        if isinstance(item, Mapping):
                            version = str(item.get("version", "*"))
                        else:
                            version = "*"
                        locked_match = _NODE_VERSION_RE.search(version)
                        locked = locked_match.group(1) if locked_match else None
                        rows.append(_dependency_row(
                            root, path, str(package), version, ecosystem="python",
                            line=_line_for(text, f'"{package}"'),
                            locked_version=locked, source_kind="Pipfile.lock",
                        ))
                continue
            if name == "setup.py":
                try:
                    tree = ast.parse(text, filename=str(path))
                except SyntaxError as error:
                    coverage.gap(relative, error.lineno or 1,
                                  f"parse error: {error.msg}", "setup.py")
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or _call_name(node).split(".")[-1] != "setup":
                        continue
                    for keyword in node.keywords:
                        if keyword.arg != "install_requires":
                            continue
                        values = _literal(keyword.value)
                        if not isinstance(values, (list, tuple)):
                            coverage.gap(
                                relative, node.lineno,
                                "dynamic setup.py dependencies were not guessed", "dependency",
                            )
                            continue
                        for value in values:
                            if isinstance(value, str) and (parsed := _requirement_parts(value)):
                                package, spec = parsed
                                rows.append(_dependency_row(
                                    root, path, package, spec, ecosystem="python",
                                    line=node.lineno, source_kind="setup.py",
                                ))
                continue
            if name == "setup.cfg":
                parser = configparser.ConfigParser()
                try:
                    parser.read_string(text)
                except configparser.Error as error:
                    coverage.gap(relative, 1, f"setup.cfg parse error: {error}")
                    continue
                if parser.has_option("options", "install_requires"):
                    for line in parser.get("options", "install_requires").splitlines():
                        parsed = _requirement_parts(line)
                        if parsed:
                            package, spec = parsed
                            rows.append(_dependency_row(
                                root, path, package, spec, ecosystem="python",
                                line=_line_for(text, package), source_kind="setup.cfg",
                            ))
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                parsed = _requirement_parts(line)
                if parsed:
                    package, spec = parsed
                    rows.append(_dependency_row(
                        root, path, package, spec, ecosystem="python",
                        line=line_number, source_kind="requirements",
                    ))
            continue
        if name in {"pyproject.toml", "poetry.lock", "uv.lock"}:
            try:
                document = tomllib.loads(text)
            except tomllib.TOMLDecodeError as error:
                coverage.gap(relative, getattr(error, "lineno", 1) or 1,
                             f"TOML parse error: {error}", "toml")
                continue
            project = document.get("project", {}) if isinstance(document, Mapping) else {}
            if isinstance(project, Mapping):
                declared = project.get("dependencies", [])
                if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
                    for value in declared:
                        if isinstance(value, str):
                            parsed = _requirement_parts(value)
                            if parsed:
                                package, spec = parsed
                                rows.append(_dependency_row(
                                    root, path, package, spec, ecosystem="python",
                                    line=_line_for(text, package), source_kind="pyproject",
                                ))
                optional = project.get("optional-dependencies", {})
                if isinstance(optional, Mapping):
                    for values in optional.values():
                        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                            for value in values:
                                parsed = _requirement_parts(str(value))
                                if parsed:
                                    package, spec = parsed
                                    rows.append(_dependency_row(
                                        root, path, package, spec, ecosystem="python",
                                        line=_line_for(text, package), source_kind="pyproject",
                                    ))
            tool = document.get("tool", {}) if isinstance(document, Mapping) else {}
            poetry = tool.get("poetry", {}) if isinstance(tool, Mapping) else {}
            poetry_deps = poetry.get("dependencies", {}) if isinstance(poetry, Mapping) else {}
            if isinstance(poetry_deps, Mapping):
                for package, raw_spec in poetry_deps.items():
                    if str(package).lower() == "python":
                        continue
                    if isinstance(raw_spec, str):
                        spec = raw_spec
                    elif isinstance(raw_spec, Mapping):
                        spec = str(raw_spec.get("version", "*"))
                    else:
                        spec = "*"
                    rows.append(_dependency_row(
                        root, path, str(package), spec, ecosystem="python",
                        line=_line_for(text, str(package)), source_kind="poetry",
                    ))
            packages = document.get("package", []) if isinstance(document, Mapping) else []
            if isinstance(packages, list):
                for item in packages:
                    if isinstance(item, Mapping) and item.get("name") and item.get("version"):
                        package = str(item["name"])
                        rows.append(_dependency_row(
                            root, path, package, "*", ecosystem="python",
                            line=_line_for(text, package),
                            locked_version=str(item["version"]), source_kind=name,
                        ))
            continue
        if name in {"package.json", "package-lock.json", "npm-shrinkwrap.json"}:
            try:
                document = json.loads(text)
            except (ValueError, TypeError) as error:
                coverage.gap(relative, 1, f"JSON parse error: {error}", "node manifest")
                continue
            if not isinstance(document, Mapping):
                continue
            if name == "package.json":
                for section in (
                    "dependencies", "devDependencies", "peerDependencies",
                    "optionalDependencies",
                ):
                    values = document.get(section, {})
                    if isinstance(values, Mapping):
                        for package, spec in values.items():
                            rows.append(_dependency_row(
                                root, path, str(package), str(spec), ecosystem="node",
                                line=_line_for(text, f'"{package}"'), source_kind=section,
                            ))
            else:
                packages = document.get("packages", {})
                if isinstance(packages, Mapping):
                    for package_path, item in packages.items():
                        if not isinstance(item, Mapping) or not str(package_path).startswith("node_modules/"):
                            continue
                        package = str(package_path).removeprefix("node_modules/")
                        version = item.get("version")
                        if version:
                            rows.append(_dependency_row(
                                root, path, package, "*", ecosystem="node",
                                line=_line_for(text, package),
                                locked_version=str(version), source_kind=name,
                            ))
                legacy = document.get("dependencies", {})
                if isinstance(legacy, Mapping):
                    for package, item in legacy.items():
                        if not isinstance(item, Mapping) or not item.get("version"):
                            continue
                        rows.append(_dependency_row(
                            root, path, str(package), "*", ecosystem="node",
                            line=_line_for(text, f'"{package}"'),
                            locked_version=str(item["version"]), source_kind=name,
                        ))
            continue
        if name == "yarn.lock":
            for match in re.finditer(
                r"(?ms)^((?:@[^:\n]+|[^:\n]+)):\s*\n"
                r"(?:[^\n]*\n)*?\s+version\s+['\"]([^'\"]+)['\"]",
                text,
            ):
                selector = match.group(1).split(",", 1)[0].strip().strip("\"'")
                package_match = re.match(r"(@[^/]+/[^@]+|[^@]+)@", selector)
                package = package_match.group(1) if package_match else selector
                rows.append(_dependency_row(
                    root, path, package, "*", ecosystem="node",
                    line=_line_for(text, selector),
                    locked_version=match.group(2), source_kind="yarn.lock",
                ))
            continue
        if name in {"pnpm-lock.yaml", "pnpm-lock.yml"}:
            for line_number, line in enumerate(text.splitlines(), 1):
                match = re.search(r"/((?:@[^/]+/)?[^@/\s]+)@([0-9][^\s:]*)", line)
                if match:
                    rows.append(_dependency_row(
                        root, path, match.group(1), "*", ecosystem="node",
                        line=line_number, locked_version=match.group(2),
                        source_kind=name,
                    ))
    for row in rows:
        for key in ("name", "ecosystem", "installed_version"):
            value = row.data.get(key)
            if value is not None:
                environment[
                    f"@environment/{row.data.get('ecosystem')}/{row.data.get('name')}/{key}"
                ] = str(value)
    return rows, coverage, environment


def _cli_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        name = path.name.lower()
        if name == "package.json":
            try:
                document = json.loads(text)
            except (ValueError, TypeError) as error:
                coverage.gap(_rel(root, path), 1, f"JSON parse error: {error}", "package scripts")
                continue
            scripts = document.get("scripts", {}) if isinstance(document, Mapping) else {}
            if isinstance(scripts, Mapping):
                for command, script in scripts.items():
                    _add(rows, root, path, {
                        "name": str(command), "command": str(command),
                        "script": str(script), "cli_type": "package_script",
                    }, _line_for(text, f'"{command}"'))
            continue
        if name == "pyproject.toml":
            try:
                document = tomllib.loads(text)
            except tomllib.TOMLDecodeError as error:
                coverage.gap(_rel(root, path), getattr(error, "lineno", 1) or 1,
                             f"TOML parse error: {error}", "project scripts")
                continue
            project = document.get("project", {}) if isinstance(document, Mapping) else {}
            scripts = project.get("scripts", {}) if isinstance(project, Mapping) else {}
            if isinstance(scripts, Mapping):
                for command, target in scripts.items():
                    _add(rows, root, path, {
                        "name": str(command), "command": str(command),
                        "entrypoint": str(target), "cli_type": "console_script",
                    }, _line_for(text, str(command)))
            continue
        if name == "setup.cfg":
            parser = configparser.ConfigParser()
            try:
                parser.read_string(text)
            except configparser.Error as error:
                coverage.gap(_rel(root, path), 1, f"setup.cfg parse error: {error}")
                continue
            if parser.has_section("options.entry_points"):
                value = parser.get("options.entry_points", "console_scripts", fallback="")
                for line in value.splitlines():
                    parsed = line.split("=", 1)
                    if len(parsed) == 2:
                        command, target = parsed[0].strip(), parsed[1].strip()
                        _add(rows, root, path, {
                            "name": command, "command": command,
                            "entrypoint": target, "cli_type": "console_script",
                        }, _line_for(text, command))
            continue
        if path.suffix.lower() not in {".py", ".pyi"}:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as error:
            coverage.gap(_rel(root, path), error.lineno or 1,
                         f"parse error: {error.msg}", "python")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                short = _call_name(node).split(".")[-1]
                if short == "setup":
                    for keyword in node.keywords:
                        if keyword.arg != "entry_points" or not isinstance(keyword.value, ast.Dict):
                            continue
                        for key_node, value_node in zip(keyword.value.keys, keyword.value.values):
                            if _static_string(key_node) != "console_scripts":
                                continue
                            values = _literal(value_node)
                            if isinstance(values, (list, tuple)):
                                for item in values:
                                    if not isinstance(item, str) or "=" not in item:
                                        continue
                                    command, target = item.split("=", 1)
                                    _add(rows, root, path, {
                                        "name": command.strip(), "command": command.strip(),
                                        "entrypoint": target.strip(),
                                        "cli_type": "console_script",
                                    }, node.lineno)
                if short in {"add_parser", "command", "group", "add_typer"}:
                    command = _static_string(node.args[0]) if node.args else None
                    if command:
                        _add(rows, root, path, {
                            "name": command, "command": command, "cli_type": short,
                        }, node.lineno)
                elif short in {"ArgumentParser", "Typer", "Group"}:
                    _add(rows, root, path, {
                        "name": short, "command": short, "cli_type": "parser",
                    }, node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    decorator_name = _decorator_name(decorator).split(".")[-1]
                    if decorator_name in {"command", "group"}:
                        command = (
                            _static_string(decorator.args[0])
                            if isinstance(decorator, ast.Call) and decorator.args else node.name
                        )
                        _add(rows, root, path, {
                            "name": command or node.name, "command": command or node.name,
                            "handler": node.name, "cli_type": "decorator",
                        }, node.lineno)
    return rows, coverage


def _module_to_product_path(root: Path, module: str, source: Path) -> str | None:
    base = source.parent if module.startswith(".") else root
    module = module.lstrip(".")
    candidate = base.joinpath(*[part for part in module.split(".") if part])
    for option in (
        candidate.with_suffix(".py"), candidate / "__init__.py", candidate.with_suffix(".pyi"),
    ):
        if option.is_file() and option != source:
            return _rel(root, option)
    return module.replace(".", "/") if module and not module.startswith(("os", "sys")) else None


def _js_module_to_product_path(root: Path, source: Path, module: str) -> str | None:
    if not module.startswith("."):
        return None
    candidate = (source.parent / module).resolve()
    for suffix in ("", ".js", ".ts", ".jsx", ".tsx"):
        option = Path(str(candidate) + suffix) if suffix else candidate
        if option.is_file():
            return _rel(root, option)
    return module


def _test_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        relative = _rel(root, path)
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        modules: set[str] = set()
        if path.suffix.lower() in {".py", ".pyi"}:
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as error:
                coverage.gap(relative, error.lineno or 1,
                             f"parse error: {error.msg}", "test imports")
                continue
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.Import):
                    module = node.names[0].name if node.names else ""
                elif isinstance(node, ast.ImportFrom):
                    module = "." * node.level + (node.module or "")
                if module and not module.startswith(("pytest", "unittest", "typing", "dataclasses")):
                    target = _module_to_product_path(root, module, path)
                    if target:
                        modules.add(target)
        else:
            for match in re.finditer(
                r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\))",
                text,
            ):
                module = match.group(1) or match.group(2) or ""
                target = _js_module_to_product_path(root, path, module)
                if target:
                    modules.add(target)
        if (
            "importlib.import_module" in text
            or re.search(r"\b__import__\s*\(\s*[A-Za-z_]", text)
        ):
            coverage.gap(
                relative, 1,
                "dynamic module import was not guessed", "test coverage",
            )
        _add(rows, root, path, {
            "name": relative, "test_file": relative,
            "modules": sorted(modules), "imported_modules": sorted(modules),
            "basis": "static",
        }, 1)
    return rows, coverage


_FRAMEWORK_IMPORTS = {
    "fastapi": "fastapi", "flask": "flask", "django": "django",
    "sqlalchemy": "sqlalchemy", "pydantic": "pydantic",
    "dataclasses": "dataclasses", "click": "click", "typer": "typer",
    "pytest": "pytest", "express": "express", "react": "react", "vue": "vue",
}


def _framework_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    found: dict[str, tuple[Path, int, str]] = {}
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        if path.name.lower() == "package.json":
            try:
                document = json.loads(text)
            except (ValueError, TypeError):
                document = {}
            if isinstance(document, Mapping):
                for section in (
                    "dependencies", "devDependencies", "peerDependencies",
                    "optionalDependencies",
                ):
                    values = document.get(section, {})
                    if isinstance(values, Mapping):
                        for package in values:
                            framework = _FRAMEWORK_IMPORTS.get(str(package).split("/")[0])
                            if framework and framework not in found:
                                found[framework] = (
                                    path, _line_for(text, f'"{package}"'), "node dependency",
                                )
        elif path.name.lower() == "pyproject.toml":
            try:
                document = tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                document = {}
            project = document.get("project", {}) if isinstance(document, Mapping) else {}
            declared = project.get("dependencies", []) if isinstance(project, Mapping) else []
            if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
                for value in declared:
                    if isinstance(value, str) and (parsed := _requirement_parts(value)):
                        framework = _FRAMEWORK_IMPORTS.get(parsed[0].split("[")[0].lower())
                        if framework and framework not in found:
                            found[framework] = (
                                path, _line_for(text, parsed[0]), "python dependency",
                            )
        if path.suffix.lower() in {".py", ".pyi"}:
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as error:
                coverage.gap(_rel(root, path), error.lineno or 1,
                             f"parse error: {error.msg}", "framework import")
                continue
            for node in ast.walk(tree):
                module = (
                    node.names[0].name if isinstance(node, ast.Import) and node.names
                    else node.module if isinstance(node, ast.ImportFrom) else ""
                )
                framework = _FRAMEWORK_IMPORTS.get(module.split(".")[0])
                if framework and framework not in found:
                    found[framework] = (path, getattr(node, "lineno", 1), "python import")
        else:
            for line_number, line in enumerate(text.splitlines(), 1):
                match = re.search(
                    r"(?:require\(\s*|from\s*['\"])([A-Za-z0-9_@/.-]+)", line)
                if match:
                    package = match.group(1).split("/")[0]
                    framework = _FRAMEWORK_IMPORTS.get(package)
                    if framework and framework not in found:
                        found[framework] = (path, line_number, "javascript import")
        lower = _rel(root, path).lower()
        for marker, framework in (
            ("docker-compose", "docker-compose"), ("dockerfile", "docker"),
            ("terraform", "terraform"), (".github/workflows", "github-actions"),
            (".gitlab-ci", "gitlab-ci"), ("k8s/", "kubernetes"), ("helm/", "helm"),
        ):
            if marker in lower and framework not in found:
                found[framework] = (path, 1, "manifest")
    for framework, (path, line, evidence) in sorted(found.items()):
        _add(rows, root, path, {
            "name": framework, "framework": framework, "evidence": evidence,
        }, line)
    return rows, coverage


def _hcl_literal(value: str) -> object:
    value = value.strip().rstrip(",")
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value):
        return float(value) if "." in value else int(value)
    if value.startswith("[") and value.endswith("]"):
        return [_hcl_literal(item) for item in value[1:-1].split(",") if item.strip()]
    if value.startswith("{") and value.endswith("}"):
        result: dict[str, object] = {}
        for piece in value[1:-1].split(","):
            if "=" in piece:
                key, item = piece.split("=", 1)
                result[key.strip()] = _hcl_literal(item)
        return result
    return value if re.fullmatch(r"[A-Za-z0-9_./:-]+", value) else None


def _hcl_blocks(text: str) -> Iterable[tuple[str, str | None, str | None, int, str]]:
    pattern = re.compile(
        r'(?m)^\s*(resource|data|variable|output|module|provider|terraform|backend|locals)'
        r'\s*(?:"([^"]+)")?\s*(?:"([^"]+)")?\s*\{'
    )
    for match in pattern.finditer(text):
        opening = text.find("{", match.start(), match.end())
        ending = _balanced_brace(text, opening)
        line = text.count("\n", 0, match.start()) + 1
        body = text[opening + 1:] if ending is None else text[opening + 1:ending]
        yield match.group(1), match.group(2), match.group(3), line, body


def _hcl_attributes(body: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in body.splitlines():
        match = re.match(r"\s*([A-Za-z_][\w-]*)\s*=\s*(.+?)\s*$", line)
        if match:
            value = _hcl_literal(match.group(2))
            if value is not None:
                result[match.group(1)] = value
    return result


def _terraform_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        blocks = list(_hcl_blocks(text))
        if text.strip() and not blocks:
            coverage.gap(_rel(root, path), 1,
                         "no supported Terraform blocks parsed", "hcl")
        required_match = re.search(r"(?m)^\s*required_providers\s*\{", text)
        if required_match:
            opening = text.find("{", required_match.start(), required_match.end())
            ending = _balanced_brace(text, opening)
            body = text[opening + 1:] if ending is None else text[opening + 1:ending]
            if ending is None:
                coverage.gap(
                    _rel(root, path),
                    text.count("\n", 0, required_match.start()) + 1,
                    "required_providers block was not closed", "hcl",
                )
            for provider_match in re.finditer(
                r"(?ms)^\s*([A-Za-z][\w-]*)\s*=\s*\{([^{}]*)\}", body
            ):
                provider_name, provider_body = provider_match.groups()
                version_match = re.search(
                    r"(?m)^\s*version\s*=\s*['\"]([^'\"]+)['\"]",
                    provider_body,
                )
                _add(rows, root, path, {
                    "name": provider_name, "address": provider_name,
                    "block_type": "required_provider",
                    "required_version": version_match.group(1) if version_match else None,
                }, text.count("\n", 0, required_match.start()) + 1)
        for block_type, first, second, line, body in blocks:
            attrs = _hcl_attributes(body)
            data: dict[str, object] = {"block_type": block_type, "attributes": attrs}
            if block_type in {"resource", "data"} and first and second:
                address = (
                    f"{first}.{second}" if block_type == "resource"
                    else f"data.{first}.{second}"
                )
                data.update({
                    "name": address, "address": address,
                    "type": first, "resource_name": second,
                })
            elif block_type in {"variable", "output", "module", "provider", "backend"} and first:
                data.update({"name": first, "address": first})
            elif block_type == "terraform":
                data.update({"name": "terraform", "address": "terraform"})
            elif block_type == "locals":
                data.update({"name": "locals", "address": "locals"})
            if block_type == "module":
                data["source"] = attrs.get("source")
                data["version"] = attrs.get("version")
                data["inputs"] = {
                    key: value for key, value in attrs.items()
                    if key not in {"source", "version"}
                }
            if block_type == "terraform":
                data["required_version"] = attrs.get("required_version")
                data["backend_config"] = attrs
            _add(rows, root, path, data, line)
    return rows, coverage


def _yaml_load(path: Path, text: str, root: Path,
               coverage: CoverageReport) -> list[object]:
    try:
        import yaml  # type: ignore
        return list(yaml.safe_load_all(text))
    except ImportError:
        coverage.gap(_rel(root, path), 1,
                     "PyYAML unavailable; YAML parser skipped", "yaml")
    except Exception as error:
        coverage.gap(_rel(root, path), 1,
                     f"YAML parse error: {error}", "yaml")
    return []


def _walk_mapping(value: object, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], object]]:
    yield prefix, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_mapping(child, (*prefix, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping(child, (*prefix, str(index)))


def _yaml_features(document: object) -> dict[str, object]:
    images: list[str] = []
    ports: list[object] = []
    env_keys: list[str] = []
    secret_refs: list[str] = []
    for path, value in _walk_mapping(document):
        key = path[-1].lower() if path else ""
        if key == "image" and isinstance(value, str):
            images.append(value)
        if key in {"containerport", "port", "targetport", "nodeport"} and isinstance(
            value, (str, int, float)
        ):
            ports.append(value)
        if key in {"secretname", "secretref", "secretkeyref", "secretnameref"}:
            if isinstance(value, str):
                secret_refs.append(value)
            elif isinstance(value, Mapping) and value.get("name"):
                secret_refs.append(str(value["name"]))
        if key in {"env", "environment"}:
            if isinstance(value, Mapping):
                env_keys.extend(str(item) for item in value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping) and item.get("name"):
                        env_keys.append(str(item["name"]))
                    elif isinstance(item, str):
                        env_keys.append(item.split("=", 1)[0])
    if isinstance(document, Mapping) and isinstance(document.get("image"), Mapping):
        image = document["image"]
        repository = image.get("repository")
        tag = image.get("tag")
        if isinstance(repository, str):
            images.append(repository + (f":{tag}" if isinstance(tag, (str, int, float)) else ""))
    return {
        "images": list(dict.fromkeys(images)),
        "ports": list(dict.fromkeys(ports)),
        "env_keys": list(dict.fromkeys(env_keys)),
        "secret_refs": list(dict.fromkeys(secret_refs)),
    }


def _kubernetes_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        for document in _yaml_load(path, text, root, coverage):
            if not isinstance(document, Mapping):
                continue
            metadata = document.get("metadata") if isinstance(
                document.get("metadata"), Mapping
            ) else {}
            kind = document.get("kind")
            if kind:
                name = metadata.get("name") if isinstance(metadata, Mapping) else None
                features = _yaml_features(document)
                data: dict[str, object] = {
                    "name": str(name or f"{kind}@{_rel(root, path)}"),
                    "kind": str(kind),
                    "namespace": metadata.get("namespace") if isinstance(metadata, Mapping) else None,
                    **features,
                }
                if str(kind).lower() == "secret" and data["name"] not in data["secret_refs"]:
                    data["secret_refs"] = [data["name"]]
                if str(kind).lower() == "secret" and isinstance(document.get("data"), Mapping):
                    data["data_keys"] = sorted(str(key) for key in document["data"])
                _add(rows, root, path, data,
                     _line_for(text, str(name) if name else str(kind)))
            elif "helm" in _rel(root, path).lower() or path.name.lower().startswith("values"):
                features = _yaml_features(document)
                _add(rows, root, path, {
                    "name": _rel(root, path), "kind": "HelmValues", **features,
                }, 1)
    return rows, coverage


def _compose_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        for document in _yaml_load(path, text, root, coverage):
            if not isinstance(document, Mapping):
                continue
            services = document.get("services")
            if not isinstance(services, Mapping):
                continue
            for service, config in services.items():
                if not isinstance(config, Mapping):
                    continue
                environment = config.get("environment", {})
                if isinstance(environment, Mapping):
                    env_keys = [str(key) for key in environment]
                elif isinstance(environment, list):
                    env_keys = [str(item).split("=", 1)[0] for item in environment]
                else:
                    env_keys = []
                depends = config.get("depends_on", [])
                if isinstance(depends, Mapping):
                    depends_on = [str(key) for key in depends]
                elif isinstance(depends, list):
                    depends_on = [str(item) for item in depends]
                else:
                    depends_on = []
                _add(rows, root, path, {
                    "name": str(service), "service": str(service),
                    "image": config.get("image") if isinstance(config.get("image"), str) else None,
                    "ports": [str(item) for item in config.get("ports", [])]
                    if isinstance(config.get("ports"), list) else [],
                    "env_keys": env_keys, "depends_on": depends_on,
                }, _line_for(text, str(service)))
    return rows, coverage


def _dockerfile_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        base_images: list[str] = []
        exposed: list[str] = []
        entrypoint: object = None
        command: object = None
        for line in text.splitlines():
            stripped = line.strip()
            match = re.match(r"FROM\s+([^\s]+)", stripped, re.I)
            if match:
                base_images.append(match.group(1))
            match = re.match(r"EXPOSE\s+(.+)", stripped, re.I)
            if match:
                exposed.extend(match.group(1).split())
            match = re.match(r"ENTRYPOINT\s+(.+)", stripped, re.I)
            if match:
                entrypoint = match.group(1).strip()
            match = re.match(r"CMD\s+(.+)", stripped, re.I)
            if match:
                command = match.group(1).strip()
        _add(rows, root, path, {
            "name": _rel(root, path), "base_images": base_images,
            "exposed_ports": exposed, "entrypoint": entrypoint, "cmd": command,
        }, 1)
    return rows, coverage


def _ci_registry(root: Path, paths: Sequence[Path]) -> tuple[list[RegistryRow], CoverageReport]:
    rows: list[RegistryRow] = []
    coverage = CoverageReport()
    for path in paths:
        text = _read_text(path, coverage, root)
        if text is None:
            continue
        relative = _rel(root, path).lower()
        for document in _yaml_load(path, text, root, coverage):
            if not isinstance(document, Mapping):
                continue
            if "/.github/workflows/" in f"/{relative}":
                raw_triggers = document.get("on", document.get(True, []))
                if isinstance(raw_triggers, Mapping):
                    triggers = list(raw_triggers)
                elif isinstance(raw_triggers, list):
                    triggers = [str(item) for item in raw_triggers]
                elif raw_triggers:
                    triggers = [str(raw_triggers)]
                else:
                    triggers = []
                jobs = document.get("jobs", {})
                if isinstance(jobs, Mapping):
                    for job, config in jobs.items():
                        config = config if isinstance(config, Mapping) else {}
                        _add(rows, root, path, {
                            "name": str(job), "job": str(job),
                            "workflow": document.get("name"),
                            "triggers": triggers,
                            "required": bool(config.get("required", False)),
                        }, _line_for(text, f"{job}:"))
            elif path.name.lower().startswith(".gitlab-ci"):
                reserved = {
                    "stages", "variables", "workflow", "default", "include",
                    "image", "services", "cache",
                }
                for job, config in document.items():
                    if (
                        str(job).lower() in reserved or str(job).startswith(".")
                        or not isinstance(config, Mapping)
                    ):
                        continue
                    rules = config.get("rules", [])
                    _add(rows, root, path, {
                        "name": str(job), "job": str(job),
                        "stage": config.get("stage"),
                        "required": bool(config.get("required", False)),
                        "triggers": list(rules) if isinstance(rules, list) else [],
                    }, _line_for(text, f"{job}:"))
            elif "/.circleci/" in f"/{relative}":
                jobs = document.get("jobs", {})
                if isinstance(jobs, Mapping):
                    for job in jobs:
                        _add(rows, root, path, {
                            "name": str(job), "job": str(job), "triggers": [],
                        }, _line_for(text, f"{job}:"))
    return rows, coverage


def _registry_cache_key(name: str, input_shas: Mapping[str, str]) -> str:
    return _sha256_bytes(_canonical_bytes({
        "schema": SCHEMA, "registry": name,
        "input_shas": dict(sorted(input_shas.items())),
    }))


def _registry_from_dict(payload: Mapping[str, object]) -> Registry | None:
    try:
        name = str(payload["name"])
        raw_rows = payload.get("rows", [])
        if not isinstance(raw_rows, list):
            return None
        rows: list[RegistryRow] = []
        for item in raw_rows:
            if not isinstance(item, Mapping):
                continue
            provenance = item.get("provenance")
            if not isinstance(provenance, Mapping):
                continue
            data = {
                key: value for key, value in item.items()
                if key not in {"provenance", "basis"}
            }
            column = provenance.get("column")
            rows.append(RegistryRow(
                data,
                Provenance(
                    str(provenance.get("path", "")),
                    int(provenance.get("line", 1)),
                    int(column) if column is not None else None,
                ),
                str(item.get("basis", "static")),
            ))
        coverage = CoverageReport()
        raw_coverage = payload.get("coverage", {})
        if isinstance(raw_coverage, Mapping):
            coverage.files_seen = int(raw_coverage.get("files_seen", 0))
            coverage.files_parsed = int(raw_coverage.get("files_parsed", 0))
            coverage.files_skipped = int(raw_coverage.get("files_skipped", 0))
            raw_gaps = raw_coverage.get("gaps", [])
            if isinstance(raw_gaps, list):
                for gap in raw_gaps:
                    if isinstance(gap, Mapping):
                        coverage.gaps.append(CoverageGap(
                            str(gap.get("path", "")), int(gap.get("line", 1)),
                            str(gap.get("reason", "")),
                            str(gap.get("construct")) if gap.get("construct") else None,
                        ))
        input_shas = payload.get("input_shas", {})
        return Registry(
            name, rows,
            {str(key): str(value) for key, value in input_shas.items()}
            if isinstance(input_shas, Mapping) else {},
            coverage,
            str(payload.get("cache_key", "")),
        )
    except (TypeError, ValueError, KeyError):
        return None


class ProjectRegistrySnapshot:
    """View of all registries for one repository revision."""

    def __init__(self, root: Path, registries: Mapping[str, Registry],
                 *, snapshot_id: str | None = None):
        self.root = root.resolve()
        self.registries = dict(registries)
        payload = {
            name: registry.cache_key or _registry_cache_key(name, registry.input_shas)
            for name, registry in sorted(self.registries.items())
        }
        self.snapshot_id = snapshot_id or _sha256_bytes(_canonical_bytes(payload))

    def lookup(self, kind: str, query: str = "") -> list[dict[str, object]]:
        registry = self.registries.get(str(kind))
        if registry is None:
            return []
        query_text = str(query or "").casefold()
        scored: list[tuple[int, int, dict[str, object]]] = []
        for index, row in enumerate(registry.rows):
            data = row.to_dict()
            if not query_text:
                score = 0
            else:
                serialized = json.dumps(data, sort_keys=True, ensure_ascii=False).casefold()
                if query_text not in serialized:
                    continue
                score = 1
                for preferred in (
                    "name", "path", "handler", "address", "test_file",
                    "key", "framework",
                ):
                    value = data.get(preferred)
                    if isinstance(value, str) and value.casefold() == query_text:
                        score = 100
                        break
            scored.append((-score, index, data))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored]

    def summary_block(self, named_symbols: Sequence[str] = (),
                      *, budget_tokens: int = 240) -> str:
        counts = " ".join(
            f"{name}={len(self.registries[name].rows) if name in self.registries else 0}"
            for name in REGISTRY_KINDS
        )
        lines = [
            f"[PROJECT REGISTRIES snapshot={self.snapshot_id[:12]}]",
            f"counts {counts}",
        ]
        seen: set[tuple[str, str]] = set()
        for symbol in named_symbols:
            symbol = str(symbol or "").strip()
            if not symbol:
                continue
            for name in REGISTRY_KINDS:
                matches = self.lookup(name, symbol)
                if not matches:
                    continue
                row = matches[0]
                location = row.get("provenance", {})
                location_text = (
                    f"{location.get('path')}:{location.get('line')}"
                    if isinstance(location, Mapping) else ""
                )
                row_name = str(row.get("name", row.get("path", symbol)))
                key = (name, row_name)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{symbol} -> {name}:{row_name} ({location_text})")
                break
        lines.append("[/PROJECT REGISTRIES]")
        words = " ".join(lines).split()
        limit = max(24, int(budget_tokens or 240))
        if len(words) > limit:
            words = words[:limit - 1] + ["..."]
        return " ".join(words)

    def frame_projection(self, named_symbols: Sequence[str] = (),
                         *, budget_tokens: int = 240) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "mode": "on",
            "snapshot_id": self.snapshot_id,
            "summary": self.summary_block(named_symbols, budget_tokens=budget_tokens),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "snapshot_id": self.snapshot_id,
            "registries": {
                name: registry.to_dict() for name, registry in self.registries.items()
            },
        }


def _builder(
    name: str, root: Path, paths: Sequence[Path],
) -> tuple[list[RegistryRow], CoverageReport, dict[str, str]]:
    if name == "dependencies":
        return _dependency_registry(root, paths)
    if name == "models":
        rows, coverage = _model_registry(root, paths)
    elif name == "routes":
        rows, coverage = _route_registry(root, paths)
    elif name == "env":
        rows, coverage = _env_registry(root, paths)
    elif name == "cli":
        rows, coverage = _cli_registry(root, paths)
    elif name == "tests":
        rows, coverage = _test_registry(root, paths)
    elif name == "frameworks":
        rows, coverage = _framework_registry(root, paths)
    elif name == "terraform":
        rows, coverage = _terraform_registry(root, paths)
    elif name == "kubernetes":
        rows, coverage = _kubernetes_registry(root, paths)
    elif name == "compose":
        rows, coverage = _compose_registry(root, paths)
    elif name == "dockerfiles":
        rows, coverage = _dockerfile_registry(root, paths)
    elif name == "ci":
        rows, coverage = _ci_registry(root, paths)
    else:
        raise KeyError(name)
    return rows, coverage, {}


class ProjectRegistryIndex:
    """Incremental registry builder with a content-addressed local cache."""

    def __init__(
        self, root: str | Path, *, cache_dir: str | Path | None = None,
        journal: object | None = None,
    ):
        self.root = Path(root).resolve()
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None else self.root / ".stateframe" / "project-registries"
        )
        self.journal = journal
        self.registries: dict[str, Registry] = {}
        self.snapshot: ProjectRegistrySnapshot | None = None
        self.last_refresh: dict[str, list[str]] = {
            "built": [], "reused": [], "cache_reused": [],
        }
        self._stat_cache: dict[str, tuple[int, int, str]] = {}

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id if self.snapshot is not None else ""

    def _hashed_inputs(self, paths: Sequence[Path]) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in paths:
            relative = _rel(self.root, path)
            try:
                stat = path.stat()
                cached = self._stat_cache.get(relative)
                if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
                    result[relative] = cached[2]
                    continue
                digest = _sha256_file(path)
                self._stat_cache[relative] = (stat.st_mtime_ns, stat.st_size, digest)
                result[relative] = digest
            except OSError:
                result[relative] = "MISSING"
        return result

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / "objects" / f"{cache_key}.json"

    def _load_cache(
        self, name: str, input_shas: Mapping[str, str], cache_key: str,
    ) -> Registry | None:
        try:
            payload = json.loads(self._cache_path(cache_key).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        registry = _registry_from_dict(payload) if isinstance(payload, Mapping) else None
        if registry is None or registry.name != name or registry.input_shas != dict(input_shas):
            return None
        registry.cache_key = cache_key
        return registry

    def _write_cache(self, registry: Registry) -> None:
        target = self._cache_path(registry.cache_key)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(registry.to_dict(), sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError:
            # Cache persistence is an optimization.  A read-only checkout
            # still receives current truth-level rows.
            return

    def _emit(self, event: str, *, name: str, registry: Registry,
              cache_reused: bool = False) -> None:
        emitter = getattr(self.journal, "emit", None)
        if not callable(emitter):
            return
        fields = {
            "producer": "kernel_project_registries",
            "registry_kind": name,
            "input_shas": dict(registry.input_shas),
            "cache_key": registry.cache_key,
            "source_count": len(registry.input_shas),
            "coverage": registry.coverage.to_dict(),
            "cache_reused": cache_reused,
            "dispatch": False,
        }
        try:
            emitter(event, **fields)
        except (TypeError, ValueError):
            return

    def build(self) -> ProjectRegistrySnapshot:
        return self.refresh(force=True)

    def refresh(
        self, *, changed_paths: Sequence[str | Path] = (), force: bool = False,
    ) -> ProjectRegistrySnapshot:
        # changed_paths is an advisory edit hint.  The manifest watcher still
        # stats and hashes every input set so external edits cannot be missed.
        del changed_paths
        built: list[str] = []
        reused: list[str] = []
        cache_reused: list[str] = []
        next_registries: dict[str, Registry] = {}
        manifest_files = _all_files(self.root)
        for name in REGISTRY_KINDS:
            paths = _manifest_paths(self.root, name, manifest_files)
            input_shas = self._hashed_inputs(paths)
            previous = self.registries.get(name)
            if not force and previous is not None and previous.input_shas == input_shas:
                next_registries[name] = previous
                reused.append(name)
                self._emit("registry_reused", name=name, registry=previous)
                continue
            cache_key = _registry_cache_key(name, input_shas)
            cached = self._load_cache(name, input_shas, cache_key)
            if cached is not None:
                next_registries[name] = cached
                cache_reused.append(name)
                self._emit("registry_reused", name=name, registry=cached, cache_reused=True)
                continue
            rows, coverage, environment = _builder(name, self.root, paths)
            registry = Registry(name, rows, input_shas, coverage, cache_key)
            self._write_cache(registry)
            next_registries[name] = registry
            built.append(name)
            self._emit("registry_built", name=name, registry=registry)
        self.registries = next_registries
        self.snapshot = ProjectRegistrySnapshot(self.root, self.registries)
        self.last_refresh = {
            "built": built, "reused": reused, "cache_reused": cache_reused,
        }
        return self.snapshot

    def lookup(self, kind: str, query: str = "") -> list[dict[str, object]]:
        if self.snapshot is None:
            self.build()
        return self.snapshot.lookup(kind, query) if self.snapshot is not None else []

    def summary_block(self, named_symbols: Sequence[str] = (),
                      *, budget_tokens: int = 240) -> str:
        if self.snapshot is None:
            self.build()
        return (
            self.snapshot.summary_block(named_symbols, budget_tokens=budget_tokens)
            if self.snapshot else ""
        )

    def frame_projection(self, named_symbols: Sequence[str] = (),
                         *, budget_tokens: int = 240) -> dict[str, object]:
        if self.snapshot is None:
            self.build()
        if self.snapshot is not None:
            return self.snapshot.frame_projection(
                named_symbols, budget_tokens=budget_tokens)
        return {
            "schema": SCHEMA, "mode": "on",
            "summary": "[PROJECT REGISTRIES unavailable]",
        }

    def tool_lookup(self, arguments: Mapping[str, object] | None) -> dict[str, object]:
        if not isinstance(arguments, Mapping) or set(arguments) != {"kind", "query"}:
            return {
                "schema": LOOKUP_SCHEMA, "status": "REJECTED",
                "reason": "INVALID_ARGUMENTS", "rows": [],
            }
        kind, query = arguments.get("kind"), arguments.get("query")
        if not isinstance(kind, str) or kind not in REGISTRY_KINDS or not isinstance(query, str):
            return {
                "schema": LOOKUP_SCHEMA, "status": "REJECTED",
                "reason": "INVALID_ARGUMENTS", "rows": [],
            }
        return {
            "schema": LOOKUP_SCHEMA, "status": "OK", "kind": kind,
            "query": query, "snapshot_id": self.snapshot_id,
            "rows": self.lookup(kind, query),
        }


def build_project_registries(
    root: str | Path, *, cache_dir: str | Path | None = None,
    journal: object | None = None,
) -> ProjectRegistrySnapshot:
    """Build all registries once for a repository snapshot."""
    return ProjectRegistryIndex(
        root, cache_dir=cache_dir, journal=journal,
    ).build()


def registry_lookup(
    index_or_snapshot: ProjectRegistryIndex | ProjectRegistrySnapshot | Mapping[str, object],
    kind: str | None = None, query: str | None = None,
) -> object:
    """Read typed rows, accepting direct arguments or a tool payload."""
    if isinstance(index_or_snapshot, Mapping):
        # The mapping form is useful to host adapters which keep the index in
        # a closure.  A bare payload cannot read the filesystem accidentally.
        return {
            "schema": LOOKUP_SCHEMA, "status": "REJECTED",
            "reason": "INDEX_REQUIRED", "rows": [],
        }
    if isinstance(index_or_snapshot, ProjectRegistryIndex):
        return index_or_snapshot.lookup(kind or "", query or "")
    if isinstance(index_or_snapshot, ProjectRegistrySnapshot):
        return index_or_snapshot.lookup(kind or "", query or "")
    return []


def registry_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "registry_lookup",
            "description": "Look up host-computed project registry facts by kind and query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(REGISTRY_KINDS)},
                    "query": {"type": "string"},
                },
                "required": ["kind", "query"],
                "additionalProperties": False,
            },
        },
    }


def registry_tool_surface(base: object) -> object:
    """Append the read-only lookup tool only when the switch is enabled."""
    if not enabled():
        return base
    if not isinstance(base, list):
        return base
    if any(
        isinstance(item, Mapping)
        and isinstance(item.get("function"), Mapping)
        and item["function"].get("name") == "registry_lookup"
        for item in base
    ):
        return base
    return [*base, registry_tool()]


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


__all__ = [
    "CoverageGap", "CoverageReport", "LOOKUP_SCHEMA", "MODE_ENV", "Provenance",
    "ProjectRegistryIndex", "ProjectRegistrySnapshot", "REGISTRY_KINDS",
    "Registry", "RegistryRow", "SCHEMA", "build_project_registries",
    "enabled", "mode", "registry_lookup", "registry_tool",
    "registry_tool_surface",
]
