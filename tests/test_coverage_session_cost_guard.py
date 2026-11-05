"""Keep per-request coverage sessions independent of full proof history."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import kernel_stateframe_call_coverage as coverage


class _EmptyJournal:
    run_id = "coverage-session-cost"
    path = "memory://coverage-session-cost"

    def receipt_cursor(self):
        return SimpleNamespace(sequence=0)

    def _borrow_records_since(self, sequence: int):
        assert sequence == 0
        return ()


class _UntouchableProofs:
    def __iter__(self):
        raise AssertionError("CoverageSession must not copy proof history")

    def copy(self):
        raise AssertionError("CoverageSession must not copy proof history")


def test_session_construction_avoids_snapshot_and_proof_copy(monkeypatch):
    def reject_snapshot(_index):
        raise AssertionError("CoverageSession must not read the full coverage snapshot")

    # The 408df98 control fails here because it reads index.snapshot after sync().
    monkeypatch.setattr(
        coverage.CoverageIndex, "snapshot", property(reject_snapshot), raising=False
    )

    index = coverage.CoverageIndex(_EmptyJournal())
    index.ids = ["call-a", "call-b"]
    index.members = {"call-a", "call-b"}
    index.chains = [coverage._sha([]), "tail-chain"]
    index.frontier = ["call-b"]
    index.proofs = _UntouchableProofs()

    session = coverage.CoverageSession(index, None)

    assert session.ids == {"call-a", "call-b"}
    assert session.chain == "tail-chain"
    assert session.frontier == ["call-b"]
    assert session.previous_count == 2


def _scope_nodes(scope):
    """Walk a function/module scope without descending into nested scopes."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _coverage_scopes(tree):
    scopes = []

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.classes = []

        def visit_ClassDef(self, node):
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()

        def visit_FunctionDef(self, node):
            scopes.append((node, self.classes[-1] if self.classes else None, node.name))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    Collector().visit(tree)
    return [(tree, None, "<module>"), *scopes]


def _annotation_is_coverage_index(annotation, type_names):
    return any(
        (isinstance(node, ast.Name) and node.id in type_names)
        or (isinstance(node, ast.Attribute) and node.attr == "CoverageIndex")
        for node in ast.walk(annotation)
    )


def _coverage_index_snapshot_reads(tree, type_names):
    factories = type_names | {"context_index", "bind_index"}
    violations = []

    def is_coverage_index(expr, names, self_fields):
        if isinstance(expr, ast.Name):
            return expr.id in names
        if isinstance(expr, ast.Attribute):
            return (isinstance(expr.value, ast.Name) and expr.value.id == "self"
                    and expr.attr in self_fields)
        if isinstance(expr, ast.Call):
            func = expr.func
            return ((isinstance(func, ast.Name) and func.id in factories)
                    or (isinstance(func, ast.Attribute) and func.attr in factories))
        return False

    for scope, class_name, function_name in _coverage_scopes(tree):
        names = {"self"} if class_name in type_names else set()
        self_fields = set()
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]
            if scope.args.vararg:
                args.append(scope.args.vararg)
            if scope.args.kwarg:
                args.append(scope.args.kwarg)
            names.update(arg.arg for arg in args
                         if arg.annotation is not None
                         and _annotation_is_coverage_index(arg.annotation, type_names))

        nodes = list(_scope_nodes(scope))
        for _ in range(len(nodes) + 1):
            changed = False
            for node in nodes:
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign):
                    targets, value = [node.target], node.value
                    if _annotation_is_coverage_index(node.annotation, type_names):
                        for target in targets:
                            if isinstance(target, ast.Name) and target.id not in names:
                                names.add(target.id)
                                changed = True
                else:
                    continue
                if value is None or not is_coverage_index(value, names, self_fields):
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        changed = True
                    elif (isinstance(target, ast.Attribute)
                          and isinstance(target.value, ast.Name)
                          and target.value.id == "self"
                          and target.attr not in self_fields):
                        self_fields.add(target.attr)
                        changed = True
            if not changed:
                break

        internal_allowlist = (class_name in type_names
                              and function_name in {"snapshot", "_snapshot",
                                                    "_replace_state_from"})
        if internal_allowlist:
            continue
        for node in nodes:
            if (isinstance(node, ast.Attribute) and node.attr == "snapshot"
                    and is_coverage_index(node.value, names, self_fields)):
                violations.append(node.lineno)
    return violations


def test_engine_callers_do_not_read_coverage_index_snapshot():
    engine = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
    violations = []
    for path in sorted(engine.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in
                   ("CoverageIndex", "CoverageSession", "context_index", "bind_index")):
            continue
        tree = ast.parse(source, filename=str(path))
        type_names = {"CoverageIndex"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "CoverageIndex":
                        type_names.add(alias.asname or alias.name)
        for line in _coverage_index_snapshot_reads(tree, type_names):
            violations.append(f"{path.relative_to(engine)}:{line}")

    assert violations == [], (
        "per-request engine code must not copy CoverageIndex.snapshot; "
        "the _snapshot accessor and _replace_state_from are internal allowlisted paths: "
        + ", ".join(violations)
    )
