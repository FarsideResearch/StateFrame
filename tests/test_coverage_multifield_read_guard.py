"""Guard the contract that legacy CoverageIndex fields are not read together unlocked."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


_LEGACY_FIELDS = {"sequence", "ids", "members", "chains", "frontier", "proofs"}
_INDEX_SOURCE_NAMES = {"CoverageIndex", "context_index", "_framefit_call_coverage_index"}


def _has_index_source(tree):
    """Every index origin recognized below contains one of these AST nodes."""
    return any(
        isinstance(node, ast.Name) and node.id in _INDEX_SOURCE_NAMES
        or isinstance(node, ast.Attribute) and node.attr in _INDEX_SOURCE_NAMES
        or isinstance(node, ast.Constant) and node.value == "_framefit_call_coverage_index"
        for node in ast.walk(tree)
    )


def _receiver_key(node):
    if isinstance(node, (ast.Name, ast.Attribute)):
        return ast.unparse(node)
    return None


def _is_snapshot_value(node):
    while isinstance(node, ast.Attribute):
        if node.attr == "snapshot":
            return True
        node = node.value
    return False


def _target_names(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in _target_names(item)]
    return []


def _normalize_receiver(receiver, aliases):
    seen = set()
    while receiver:
        head, separator, tail = receiver.partition(".")
        if head in seen or head not in aliases:
            return receiver
        seen.add(head)
        receiver = aliases[head] + (f".{tail}" if separator else "")
    return receiver


def _is_index_annotation(annotation):
    return annotation is not None and any(
        isinstance(node, ast.Name) and node.id == "CoverageIndex"
        for node in ast.walk(annotation)
    )


def _is_index_source(value):
    if isinstance(value, ast.Attribute):
        return value.attr == "_framefit_call_coverage_index"
    if not isinstance(value, ast.Call):
        return False
    name = value.func.id if isinstance(value.func, ast.Name) else (
        value.func.attr if isinstance(value.func, ast.Attribute) else None
    )
    if name in {"CoverageIndex", "context_index"}:
        return True
    return (name == "getattr" and len(value.args) >= 2
            and isinstance(value.args[1], ast.Constant)
            and value.args[1].value == "_framefit_call_coverage_index")


def _class_index_attributes(class_node):
    """Find index attributes assigned from typed constructor parameters."""
    if class_node is None:
        return set()
    receivers = set()
    for method in class_node.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.name != "__init__":
            continue
        parameters = (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs)
        index_parameters = {arg.arg for arg in parameters if _is_index_annotation(arg.annotation)}
        for statement in ast.walk(method):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if not isinstance(statement.value, ast.Name) or statement.value.id not in index_parameters:
                continue
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    receivers.add(f"self.{target.attr}")
    return receivers


def _function_bindings(function, class_index_attributes):
    aliases = {}
    snapshots = set()
    indexes = set(class_index_attributes)
    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
        indexes.update(arg.arg for arg in parameters if _is_index_annotation(arg.annotation))

    class BindingsVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            if node is function:
                self.visit(node.body)

        def visit_ClassDef(self, node):
            return

        def _bind(self, targets, value):
            names = [name for target in targets for name in _target_names(target)]
            if _is_snapshot_value(value):
                snapshots.update(names)
                return
            if _is_index_source(value):
                indexes.update(names)
                source = _receiver_key(value)
                if source is not None:
                    indexes.add(source)
                return
            source = _receiver_key(value)
            if source is not None:
                aliases.update({name: source for name in names})

        def visit_Assign(self, node):
            self._bind(node.targets, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self._bind([node.target], node.value)
            self.generic_visit(node)

        def visit_NamedExpr(self, node):
            self._bind([node.target], node.value)
            self.generic_visit(node)

    visitor = BindingsVisitor()
    if isinstance(function, ast.Lambda):
        visitor.visit(function.body)
    else:
        for statement in function.body:
            visitor.visit(statement)

    changed = True
    while changed:
        changed = False
        for name, source in aliases.items():
            if source in snapshots and name not in snapshots:
                snapshots.add(name)
                changed = True
            if _normalize_receiver(source, aliases) in indexes and name not in indexes:
                indexes.add(name)
                changed = True
    return aliases, snapshots, indexes


class _ReadVisitor(ast.NodeVisitor):
    def __init__(self, function, aliases, snapshots, indexes):
        self.function = function
        self.aliases = aliases
        self.snapshots = snapshots
        self.indexes = indexes
        self.locked = set()
        self.reads = defaultdict(list)

    def visit_FunctionDef(self, node):
        if node is self.function:
            for statement in node.body:
                self.visit(statement)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        if node is self.function:
            self.visit(node.body)

    def visit_ClassDef(self, node):
        return

    def _visit_with(self, node):
        previous = set(self.locked)
        for item in node.items:
            context = item.context_expr
            if isinstance(context, ast.Attribute) and context.attr == "_lock":
                receiver = _receiver_key(context.value)
                if receiver is not None:
                    self.locked.add(_normalize_receiver(receiver, self.aliases))
        for statement in node.body:
            self.visit(statement)
        self.locked = previous

    visit_With = _visit_with
    visit_AsyncWith = _visit_with

    def visit_Attribute(self, node):
        if (isinstance(node.ctx, ast.Load) and node.attr in _LEGACY_FIELDS
                and not _is_snapshot_value(node.value)):
            receiver = _receiver_key(node.value)
            if receiver is not None:
                normalized = _normalize_receiver(receiver, self.aliases)
                snapshot_alias = normalized.partition(".")[0] in self.snapshots
                if not snapshot_alias and (normalized in self.indexes or _is_index_source(node.value)):
                    self.reads[normalized].append((node.attr, normalized in self.locked, node.lineno))
        self.generic_visit(node)


class _ScopeCollector(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []

    def visit_ClassDef(self, node):
        self.classes.append(node)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node):
        self.functions.append((node, self.classes[-1] if self.classes else None))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.functions.append((node, self.classes[-1] if self.classes else None))
        self.generic_visit(node)


def _unlocked_multifield_reads(tree):
    collector = _ScopeCollector()
    collector.visit(tree)
    violations = []
    class_attributes = {}
    for function, class_node in collector.functions:
        if class_node is not None and class_node.name == "CoverageIndex":
            continue
        if class_node not in class_attributes:
            class_attributes[class_node] = _class_index_attributes(class_node)
        aliases, snapshots, indexes = _function_bindings(
            function, class_attributes[class_node]
        )
        visitor = _ReadVisitor(function, aliases, snapshots, indexes)
        body = (function.body if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                else [function.body])
        for statement in body:
            visitor.visit(statement)
        for receiver, reads in visitor.reads.items():
            fields = {field for field, _locked, _line in reads}
            if len(fields) < 2 or all(locked for _field, locked, _line in reads):
                continue
            lines = sorted({line for _field, _locked, line in reads})
            violations.append(
                f"{function.name}:{receiver}:{','.join(sorted(fields))}@{','.join(map(str, lines))}"
            )
    return violations


def test_engine_multifield_legacy_reads_use_lock_or_snapshot():
    engine = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
    violations = []
    for path in sorted(engine.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _has_index_source(tree):
            continue
        for violation in _unlocked_multifield_reads(tree):
            violations.append(f"{path.relative_to(engine)}:{violation}")
    assert violations == [], (
        "multi-field CoverageIndex reads must be inside `with <index>._lock:` "
        "or use `snapshot`; only CoverageIndex methods are allowlisted: "
        + ", ".join(violations)
    )


def test_ast_gate_keeps_every_recognized_index_origin():
    sources = (
        "def typed(index: CoverageIndex):\n    return index.ids, index.chains\n",
        "def called(context):\n    return context_index(context).ids\n",
        "def direct(journal):\n    return journal._framefit_call_coverage_index.ids\n",
        "def dynamic(journal):\n"
        "    return getattr(journal, '\\x5fframefit_call_coverage_index').ids\n",
    )
    assert all(_has_index_source(ast.parse(source)) for source in sources)
    assert not _has_index_source(ast.parse("def unrelated(session):\n    return session.ids, session.chains\n"))


def test_synthetic_unlocked_multifield_read_is_flagged():
    synthetic = ast.parse("def bad(index: CoverageIndex):\n    return index.ids, index.chains\n")
    assert _unlocked_multifield_reads(synthetic) == ["bad:index:chains,ids@2"]


def test_snapshot_alias_and_locked_index_alias_are_allowed():
    synthetic = ast.parse(
        "def read_snapshot(index: CoverageIndex):\n"
        "    state = index.snapshot\n"
        "    return state.ids, state.chains\n"
        "def read_locked(index: CoverageIndex):\n"
        "    alias = index\n"
        "    with index._lock:\n"
        "        return alias.ids, alias.chains\n"
    )
    assert _unlocked_multifield_reads(synthetic) == []


def test_index_sources_are_tracked_but_session_fields_are_not():
    synthetic = ast.parse(
        "def from_context(context):\n"
        "    index = context_index(context)\n"
        "    return index.ids, index.chains\n"
        "def from_journal(journal):\n"
        "    index = journal._framefit_call_coverage_index\n"
        "    return index.ids, index.chains\n"
        "class Session:\n"
        "    def __init__(self, index: CoverageIndex):\n"
        "        self.index = index\n"
        "        self.ids = set(index.members)\n"
        "    def read(self):\n"
        "        return self.ids, self.frontier\n"
        "    def bad(self):\n"
        "        return self.index.ids, self.index.chains\n"
    )
    assert _unlocked_multifield_reads(synthetic) == [
        "from_context:index:chains,ids@3",
        "from_journal:index:chains,ids@6",
        "bad:self.index:chains,ids@14",
    ]


def test_direct_journal_attribute_and_getattr_index_reads_are_tracked():
    synthetic = ast.parse(
        "def direct(journal):\n"
        "    return journal._framefit_call_coverage_index.ids, journal._framefit_call_coverage_index.chains\n"
        "def via_getattr(journal):\n"
        "    index = getattr(journal, '_framefit_call_coverage_index', None)\n"
        "    return index.ids, index.chains\n"
    )
    assert _unlocked_multifield_reads(synthetic) == [
        "direct:journal._framefit_call_coverage_index:chains,ids@2",
        "via_getattr:index:chains,ids@5",
    ]
