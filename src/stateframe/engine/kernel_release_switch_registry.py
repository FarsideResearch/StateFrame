"""Derive the release's readable STATEFRAME switches from shipped Python sources."""
from __future__ import annotations

import ast
from functools import lru_cache
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


SCHEMA = "stateframe-switch-registry/1"


class StateFrameSwitchPreflightError(RuntimeError):
    """A configured STATEFRAME switch has no reader in this declared release."""

    outcome = "HARNESS_CONTRACT"
    failure_origin = "harness"
    harness_contract = True
    transport_retryable = False
    transport_class = "contract"

    def __init__(self, detail: Mapping[str, object], *, journal_path: Path):
        self.detail = dict(detail)
        self.journal_path = journal_path
        super().__init__("unconsumed STATEFRAME settings: " + ", ".join(
            str(name) for name in self.detail.get("unknown_switches", ())))


def _target_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for child in node.elts for name in _target_names(child))
    return ()


def _literal_container_switches(node: ast.AST) -> set[str]:
    """Collect switch names from static container definitions such as FLAGS maps."""
    if isinstance(node, ast.Constant):
        return ({node.value} if isinstance(node.value, str)
                and node.value.startswith("STATEFRAME_") else set())
    if isinstance(node, ast.Dict):
        children = (*node.keys, *node.values)
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        children = tuple(node.elts)
    else:
        return set()
    return set().union(*(_literal_container_switches(child) for child in children))


class _SwitchReaderVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.constants: dict[str, str] = {}
        self.container_switches: dict[str, set[str]] = {}
        self.loaded_names: set[str] = set()
        self.direct_reads: set[str] = set()
        self.definition_reads: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value.startswith("STATEFRAME_"):
                for target in node.targets:
                    for name in _target_names(target):
                        self.constants[name] = value.value
        else:
            switches = _literal_container_switches(value)
            if switches:
                for target in node.targets:
                    for name in _target_names(target):
                        self.container_switches[name] = switches
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value.startswith("STATEFRAME_"):
                for name in _target_names(node.target):
                    self.constants[name] = value.value
        else:
            switches = _literal_container_switches(value)
            if switches:
                for name in _target_names(node.target):
                    self.container_switches[name] = switches
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        function_name = (
            function.attr if isinstance(function, ast.Attribute)
            else function.id if isinstance(function, ast.Name) else ""
        )
        if function_name in {"get", "getenv", "_switch", "SwitchRef"} and node.args:
            key = node.args[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value.startswith("STATEFRAME_"):
                    self.direct_reads.add(key.value)
        if function_name == "LeanStackSlot" and len(node.args) >= 2:
            key = node.args[1]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value.startswith("STATEFRAME_"):
                    self.definition_reads.add(key.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if key.value.startswith("STATEFRAME_"):
                self.direct_reads.add(key.value)
        self.generic_visit(node)

    def switches(self) -> set[str]:
        return (
            self.direct_reads
            | self.definition_reads
            | {value for name, value in self.constants.items() if name in self.loaded_names}
            | {value for name, values in self.container_switches.items()
               if name in self.loaded_names for value in values}
        )


def _declared_python_files(root: Path) -> tuple[Path, ...]:
    declarations_path = root / "release_declarations.json"
    if not declarations_path.is_file():
        # 10-13: an ASSEMBLED runtime release does not carry the source tree's
        # release_declarations.json (release_assembly consumes it); every live run on a
        # release with this registry died at StateFrameRuntime() with FileNotFoundError
        # while source-tree smokes passed. The release's own manifest lists exactly the
        # assembled files, which already reflect the declarations. Neither file: fail closed.
        manifest_path = root / "release-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"neither release_declarations.json nor release-manifest.json under {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(listed, dict) or not listed:
            raise ValueError(f"release-manifest.json under {root} lists no files")
        return tuple(sorted(
            root / name for name in listed
            if isinstance(name, str) and name.endswith(".py") and (root / name).is_file()
        ))
    declarations = json.loads(declarations_path.read_text(encoding="utf-8"))
    patterns = declarations.get("python_modules", ())
    exclusions = tuple(str(item) for item in declarations.get("excluded_paths", ()))
    files: set[Path] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        for path in root.glob(pattern):
            if not path.is_file() or path.suffix != ".py":
                continue
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatchcase(relative, excluded) for excluded in exclusions):
                continue
            files.add(path)
    return tuple(sorted(files))


@lru_cache(maxsize=4)
def _release_registry(root_text: str) -> tuple[frozenset[str], str]:
    root = Path(root_text).resolve()
    switches: set[str] = set()
    for path in _declared_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            continue
        reader = _SwitchReaderVisitor()
        reader.visit(tree)
        switches.update(reader.switches())
    frozen = frozenset(switches)
    encoded = json.dumps(sorted(frozen), separators=(",", ":")).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return frozen, digest


def release_stateframe_switches(root: Path | None = None) -> tuple[frozenset[str], str]:
    """Return switches read by code declared in this release and its registry digest."""
    selected_root = root or Path(__file__).resolve().parent
    return _release_registry(str(selected_root.resolve()))


def unconsumed_stateframe_switches(
        environment: Mapping[str, object] | None = None, *, root: Path | None = None,
) -> dict[str, object]:
    """Report configured STATEFRAME keys absent from the release-derived reader registry."""
    selected_environment = environment if environment is not None else os.environ
    registered, registry_digest = release_stateframe_switches(root)
    unknown = sorted(
        str(name) for name in selected_environment
        if str(name).startswith("STATEFRAME_") and str(name) not in registered
    )
    return {
        "schema": SCHEMA,
        "registry_digest": registry_digest,
        "registered_switch_count": len(registered),
        "unknown_switches": unknown,
    }


def enforce_stateframe_switch_registry(
        environment: Mapping[str, object] | None = None, *, journal: object,
        root: Path | None = None,
) -> dict[str, object]:
    """Journal a typed run-start refusal before any provider request on registry drift."""
    detail = unconsumed_stateframe_switches(environment, root=root)
    unknown = detail["unknown_switches"]
    if unknown:
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit("stateframe_switch_preflight_refused",
                 producer="kernel_release_switch_registry",
                 outcome="HARNESS_CONTRACT", failure_origin="harness",
                 dispatch=False, registry_schema=detail["schema"],
                 registry_digest=detail["registry_digest"],
                 registered_switch_count=detail["registered_switch_count"],
                 unknown_switches=detail["unknown_switches"])
        raise StateFrameSwitchPreflightError(detail, journal_path=journal.path)
    return detail


__all__ = [
    "SCHEMA", "StateFrameSwitchPreflightError", "enforce_stateframe_switch_registry",
    "release_stateframe_switches", "unconsumed_stateframe_switches",
]
