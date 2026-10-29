"""Stdlib-only static checks for problems introduced by one product edit.

This is an advisory lane.  It never rejects an edit.  The caller supplies the
pre-edit bytes and this module compares them with the post-edit checkout so a
pre-existing warning is not attributed to the worker.
"""
from __future__ import annotations

import ast
import builtins
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from kernel_stub_gate import duplicate_definitions, unimplemented_functions
from kernel_test_python import test_python


MODE_ENV = "STATEFRAME_POST_EDIT_CHECKS"
MAX_FINDINGS = 5
_DEFINITION_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_PREDEFINED_MODULE_NAMES = {
    "__name__", "__file__", "__package__", "__spec__", "__loader__",
    "__cached__", "__builtins__", "__annotations__", "__class__",
}


def post_edit_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return post_edit_mode() == "on"


@dataclass(frozen=True, order=True)
class Finding:
    file: str
    line: int
    kind: str
    symbol: str
    decorator: str | None = field(default=None, compare=False)
    count: int | None = field(default=None, compare=False)

    def as_dict(self) -> dict[str, object]:
        result = {"file": self.file, "line": self.line,
                  "kind": self.kind, "symbol": self.symbol}
        if self.decorator is not None:
            result["decorator"] = self.decorator
        if self.count is not None:
            result["count"] = self.count
        return result


@dataclass(frozen=True)
class ImportStatus:
    module: str
    failed: bool
    detail: str = ""

    def finding(self, file: str) -> Finding | None:
        if not self.failed:
            return None
        return Finding(file=file, line=1, kind="import_fails", symbol=self.module)


def _source_tree(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return None


class _Scope:
    def __init__(self, kind: str, parent: "_Scope | None") -> None:
        self.kind = kind
        self.parent = parent
        self.bindings: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self.wildcard_import = False

    def lookup_parent(self) -> "_Scope | None":
        parent = self.parent
        if self.kind == "function" and parent is not None and parent.kind == "class":
            parent = parent.parent
        return parent

    def resolves(self, name: str) -> bool:
        if self.wildcard_import:
            return True
        if name in self.global_names:
            root = self
            while root.lookup_parent() is not None:
                root = root.lookup_parent()  # type: ignore[assignment]
            return name in root.bindings or name in _PREDEFINED_MODULE_NAMES
        if name in self.bindings:
            return True
        parent = self.lookup_parent()
        return parent.resolves(name) if parent is not None else name in _BUILTINS


_BUILTINS = set(dir(builtins))


def _bind_target(node: ast.AST, scope: _Scope) -> None:
    if isinstance(node, ast.Name):
        scope.bindings.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            _bind_target(item, scope)
    elif isinstance(node, ast.Starred):
        _bind_target(node.value, scope)
    elif isinstance(node, ast.Attribute):
        # Attribute stores never create a lexical name.
        return


def _collect_bindings(node: ast.AST, scope: _Scope) -> None:
    """Collect names local to one scope without entering nested lexical scopes."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        scope.bindings.add(node.name)
        return
    if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return
    if isinstance(node, ast.Global):
        scope.global_names.update(node.names)
        return
    if isinstance(node, ast.Nonlocal):
        scope.nonlocal_names.update(node.names)
        return
    if isinstance(node, ast.Import):
        for alias in node.names:
            scope.bindings.add(alias.asname or alias.name.split(".", 1)[0])
        return
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                scope.wildcard_import = True
            else:
                scope.bindings.add(alias.asname or alias.name)
        return
    if isinstance(node, ast.ExceptHandler):
        if isinstance(node.name, str):
            scope.bindings.add(node.name)
    if isinstance(node, ast.MatchAs) and node.name:
        scope.bindings.add(node.name)
    if isinstance(node, ast.MatchStar) and node.name:
        scope.bindings.add(node.name)
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        scope.bindings.add(node.id)
    for child in ast.iter_child_nodes(node):
        _collect_bindings(child, scope)


class _UndefinedNamePass:
    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.findings: set[Finding] = set()
        self.module = _Scope("module", None)
        self.module.bindings.update(_PREDEFINED_MODULE_NAMES)
        _collect_bindings(tree, self.module)

    def run(self) -> set[Finding]:
        self._visit_body(self.tree.body, self.module)
        return self.findings

    def _record_load(self, node: ast.Name, scope: _Scope) -> None:
        if not scope.resolves(node.id):
            self.findings.add(Finding("", int(getattr(node, "lineno", 1)),
                                      "undefined_name", node.id))

    def _visit_arguments_annotations(self, args: ast.arguments, scope: _Scope) -> None:
        all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg is not None:
            all_args.append(args.vararg)
        if args.kwarg is not None:
            all_args.append(args.kwarg)
        for arg in all_args:
            if arg.annotation is not None:
                self._visit(arg.annotation, scope)

    def _function_parent(self, scope: _Scope) -> _Scope:
        return scope.parent if scope.kind == "class" and scope.parent is not None else scope

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef,
                        parent: _Scope) -> None:
        for decorator in node.decorator_list:
            self._visit(decorator, parent)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self._visit(default, parent)
        self._visit_arguments_annotations(node.args, parent)
        if node.returns is not None:
            self._visit(node.returns, parent)
        scope = _Scope("function", self._function_parent(parent))
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            scope.bindings.add(arg.arg)
        if node.args.vararg is not None:
            scope.bindings.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            scope.bindings.add(node.args.kwarg.arg)
        _collect_bindings(ast.Module(body=node.body, type_ignores=[]), scope)
        self._visit_body(node.body, scope)

    def _visit_class(self, node: ast.ClassDef, parent: _Scope) -> None:
        for decorator in node.decorator_list:
            self._visit(decorator, parent)
        for base in node.bases:
            self._visit(base, parent)
        for keyword in node.keywords:
            self._visit(keyword.value, parent)
        scope = _Scope("class", parent)
        _collect_bindings(ast.Module(body=node.body, type_ignores=[]), scope)
        self._visit_body(node.body, scope)

    def _visit_comprehension(self, node: ast.AST, scope: _Scope) -> None:
        generators = list(getattr(node, "generators", ()))
        comp = _Scope("comprehension", scope)
        for generator in generators:
            self._visit(generator.iter, scope if generator is generators[0] else comp)
            _bind_target(generator.target, comp)
            for condition in generator.ifs:
                self._visit(condition, comp)
        if isinstance(node, ast.DictComp):
            self._visit(node.key, comp)
            self._visit(node.value, comp)
        else:
            self._visit(node.elt, comp)

    def _visit(self, node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._record_load(node, scope)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(node, scope)
            return
        if isinstance(node, ast.ClassDef):
            self._visit_class(node, scope)
            return
        if isinstance(node, ast.Lambda):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self._visit(default, scope)
            lambda_scope = _Scope("function", self._function_parent(scope))
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                lambda_scope.bindings.add(arg.arg)
            if node.args.vararg is not None:
                lambda_scope.bindings.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                lambda_scope.bindings.add(node.args.kwarg.arg)
            _collect_bindings(node.body, lambda_scope)
            self._visit(node.body, lambda_scope)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self._visit_comprehension(node, scope)
            return
        if isinstance(node, ast.Global):
            scope.global_names.update(node.names)
            return
        if isinstance(node, ast.Nonlocal):
            scope.nonlocal_names.update(node.names)
            return
        if isinstance(node, ast.ExceptHandler):
            if node.type is not None:
                self._visit(node.type, scope)
            for statement in node.body:
                self._visit(statement, scope)
            return
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)

    def _visit_body(self, body: Iterable[ast.stmt], scope: _Scope) -> None:
        for node in body:
            self._visit(node, scope)


def _relative_file(path: Path, checkout: Path) -> str:
    try:
        return path.resolve().relative_to(checkout.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _definition_lines(source: str, names: Iterable[str]) -> dict[str, int]:
    tree = _source_tree(source)
    if tree is None:
        return {}
    wanted = set(names)
    result: dict[str, int] = {}

    def visit(body: list[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, _DEFINITION_TYPES):
                qualified = ".".join((*prefix, node.name))
                if qualified in wanted:
                    result[qualified] = int(getattr(node, "lineno", 1))
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    return result


def _local_module_path(checkout: Path, module: str) -> Path | None:
    if not module:
        return None
    parts = module.split(".")
    candidate = checkout.joinpath(*parts)
    file_candidate = candidate.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    init_candidate = candidate / "__init__.py"
    return init_candidate if init_candidate.is_file() else None


def _resolve_import(module: str, path: Path, level: int, checkout: Path) -> str:
    if level == 0:
        return module
    try:
        package_parts = list(path.resolve().relative_to(checkout.resolve()).parts[:-1])
    except (OSError, ValueError):
        package_parts = []
    if package_parts and package_parts[-1].endswith(".py"):
        package_parts.pop()
    base = package_parts[:max(0, len(package_parts) - level + 1)]
    return ".".join([*base, *([module] if module else [])])


def _top_level_symbols(source: str) -> set[str]:
    tree = _source_tree(source)
    if tree is None:
        return set()
    names = set(_PREDEFINED_MODULE_NAMES)
    module_scope = _Scope("module", None)
    _collect_bindings(tree, module_scope)
    names.update(module_scope.bindings)
    return names


def _import_findings(source: str, path: Path, checkout: Path,
                     overrides: Mapping[str, str | None]) -> set[Finding]:
    tree = _source_tree(source)
    if tree is None:
        return set()
    findings: set[Finding] = set()
    current_file = _relative_file(path, checkout)

    def source_for(module_path: Path) -> str | None:
        relative = _relative_file(module_path, checkout)
        if relative in overrides:
            return overrides[relative]
        try:
            return module_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

    for node in ast.walk(tree):
        module = None
        level = 0
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_module = _local_module_path(checkout, alias.name)
                if local_module is not None and source_for(local_module) is None:
                    findings.add(Finding(current_file, int(node.lineno), "import_error", alias.name))
                elif local_module is not None:
                    # ``import pkg.missing`` fails at the module path itself; the
                    # local-module resolver above intentionally ignores external imports.
                    continue
                else:
                    parts = alias.name.split(".")
                    for index in range(len(parts) - 1, 0, -1):
                        prefix = ".".join(parts[:index])
                        package = _local_module_path(checkout, prefix)
                        if package is not None and package.name == "__init__.py":
                            findings.add(Finding(current_file, int(node.lineno),
                                                 "import_error", alias.name))
                            break
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = int(node.level or 0)
            resolved = _resolve_import(module, path, level, checkout)
            local_module = _local_module_path(checkout, resolved)
            if local_module is None:
                continue
            imported_source = source_for(local_module)
            if imported_source is None:
                findings.add(Finding(current_file, int(node.lineno), "import_error", resolved))
                continue
            symbols = _top_level_symbols(imported_source)
            for alias in node.names:
                submodule = _local_module_path(checkout, f"{resolved}.{alias.name}")
                if (alias.name != "*" and alias.name not in symbols
                        and submodule is None):
                    findings.add(Finding(current_file, int(node.lineno), "import_error",
                                         f"{resolved}.{alias.name}"))
    return findings


def _static_findings(source: str, path: Path, checkout: Path,
                     overrides: Mapping[str, str | None]) -> set[Finding]:
    relative = _relative_file(path, checkout)
    tree = _source_tree(source)
    findings: set[Finding] = set()
    if tree is not None:
        for finding in _UndefinedNamePass(tree).run():
            findings.add(Finding(relative, finding.line, finding.kind, finding.symbol))
        duplicate_names = [name for name in duplicate_definitions(source) if "." not in name]
        duplicate_lines = _definition_lines(source, duplicate_names)
        for name in duplicate_names:
            findings.add(Finding(relative, duplicate_lines.get(name, 1),
                                 "duplicate_def", name))
        stub_names = unimplemented_functions(source)
        stub_lines = _definition_lines(source, stub_names)
        for name in stub_names:
            findings.add(Finding(relative, stub_lines.get(name, 1), "live_stub", name))
        from kernel_frame_edit_integrity import (
            duplicate_decorator_findings,
            enabled as frame_integrity_enabled,
        )
        if frame_integrity_enabled():
            for row in duplicate_decorator_findings(source, file=relative):
                findings.add(Finding(
                    relative, int(row["line"]), "duplicate_decorator",
                    str(row["symbol"]), str(row["decorator"]), int(row["count"]),
                ))
        findings.update(_import_findings(source, path, checkout, overrides))
    return findings


def module_name_for_path(path: Path, checkout: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(checkout.resolve())
    except (OSError, ValueError):
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def import_status(checkout: Path, path: Path, *, source: str | None = None) -> ImportStatus | None:
    module = module_name_for_path(path, checkout)
    if module is None:
        return None
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    pythonpath = str(checkout.resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pythonpath if not existing else pythonpath + os.pathsep + existing
    if source is None:
        command = [test_python(), "-c", f"import {module}"]
        run_kwargs = {}
    else:
        # A direct check_edit call may be made after the caller has already
        # written ``after`` to disk.  Execute the old bytes in a throwaway
        # module namespace so an import failure that predates the edit is not
        # attributed to the new bytes.  The post-edit path above remains the
        # required literal ``python -c import <module>`` check.
        command = [test_python(), "-c", (
            "import importlib.util, sys; "
            "name, filename = sys.argv[1], sys.argv[2]; "
            "spec = importlib.util.spec_from_file_location(name, filename); "
            "module = importlib.util.module_from_spec(spec); "
            "sys.modules[name] = module; "
            "exec(compile(sys.stdin.read(), filename, 'exec'), module.__dict__)"
        ), module, str(path)]
        run_kwargs = {"input": source}
    try:
        result = subprocess.run(
            command, cwd=str(checkout), env=env, capture_output=True, text=True,
            timeout=10, **run_kwargs,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return ImportStatus(module, True, f"{type(error).__name__}: {error}")
    if result.returncode == 0:
        return ImportStatus(module, False, "")
    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    return ImportStatus(module, True, detail[:900])


def _bounded(records: Iterable[Finding], maximum: int = MAX_FINDINGS) -> tuple[list[dict], int]:
    ordered = sorted(set(records))
    return ([record.as_dict() for record in ordered[:maximum]], max(0, len(ordered) - maximum))


class PostEditTracker:
    """Per-run dedup/resolution state for edit receipts."""

    def __init__(self, checkout: Path, *, maximum: int = MAX_FINDINGS):
        self.checkout = checkout.resolve()
        self.maximum = maximum
        self.active: set[Finding] = set()
        self.resolved_emitted: set[Finding] = set()

    def baseline_import(self, path: Path) -> ImportStatus | None:
        return import_status(self.checkout, path)

    def evaluate(self, path: Path, before: str | None, after: str,
                 *, baseline_import: ImportStatus | None = None) -> dict[str, object]:
        relative = _relative_file(path, self.checkout)
        before_overrides = {relative: before}
        after_overrides = {relative: after}
        before_findings = (_static_findings(before, path, self.checkout, before_overrides)
                           if isinstance(before, str) else set())
        if baseline_import is not None:
            finding = baseline_import.finding(relative)
            if finding is not None:
                before_findings.add(finding)
        elif isinstance(before, str):
            pre_import = import_status(self.checkout, path, source=before)
            if pre_import is not None:
                finding = pre_import.finding(relative)
                if finding is not None:
                    before_findings.add(finding)
        after_findings = _static_findings(after, path, self.checkout, after_overrides)
        post_import = import_status(self.checkout, path)
        if post_import is not None:
            finding = post_import.finding(relative)
            if finding is not None:
                after_findings.add(finding)
        if not self.active:
            self.active.update(before_findings)
        introduced = after_findings - before_findings
        # A still-active issue is not repeated on every later edit.  A newly
        # introduced issue with an identical identity is likewise suppressed.
        introduced = {finding for finding in introduced if finding not in self.active}
        resolved = {finding for finding in self.active - after_findings
                    if finding not in self.resolved_emitted}
        ignored_then_failed = {
            finding for finding in after_findings - before_findings
            if any(old.file == finding.file and old.symbol == finding.symbol
                   for old in before_findings)
        }
        self.active.difference_update(resolved)
        self.active.update(introduced)
        self.resolved_emitted.update(resolved)
        introduced_rows, introduced_rest = _bounded(introduced, self.maximum)
        resolved_rows, resolved_rest = _bounded(resolved, self.maximum)
        result = {
            "schema": "kernel-post-edit-checks/1",
            "introduced": introduced_rows,
            "introduced_rest": introduced_rest,
            "resolved": [record.as_dict() for record in sorted(resolved)[:self.maximum]],
            "resolved_rest": resolved_rest,
            "ignored_then_failed": len(ignored_then_failed),
            "counts": {"introduced": len(introduced), "resolved": len(resolved),
                       "ignored_then_failed": len(ignored_then_failed)},
        }
        return result


def check_edit(checkout: Path, path: Path, before: str | None, after: str,
               *, tracker: PostEditTracker | None = None,
               baseline_import: ImportStatus | None = None) -> dict[str, object]:
    current = tracker or PostEditTracker(checkout)
    return current.evaluate(path, before, after, baseline_import=baseline_import)


__all__ = [
    "Finding", "ImportStatus", "MAX_FINDINGS", "MODE_ENV", "PostEditTracker",
    "check_edit", "enabled", "import_status", "module_name_for_path", "post_edit_mode",
]
