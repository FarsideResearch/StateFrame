"""Keep StateFrame journal writers and continuation-resume policy in sync."""

from __future__ import annotations

import ast
from functools import lru_cache
from itertools import product
from pathlib import Path
import importlib.util
import re
from typing import Iterable

import pytest

import kernel_continuation_restart as restart


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "stateframe"


def _static_strings(node: ast.AST, symbols: dict[str, set[str]]) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(symbols.get(node.id, ()))
    if isinstance(node, ast.IfExp):
        return _static_strings(node.body, symbols) | _static_strings(node.orelse, symbols)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_strings(node.left, symbols)
        right = _static_strings(node.right, symbols)
        if len(left) * len(right) > 256:
            return set()
        return {first + second for first, second in product(left, right)}
    if isinstance(node, ast.JoinedStr):
        variants = [""]
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                values = {part.value}
            elif isinstance(part, ast.FormattedValue):
                values = _static_strings(part.value, symbols)
            else:
                return set()
            if not values:
                return set()
            if len(variants) * len(values) > 256:
                return set()
            variants = [prefix + value for prefix, value in product(variants, values)]
        return set(variants)
    return set()


def _unclassified(
    event_kinds: Iterable[str], safe_tail: Iterable[str], documented_reject: Iterable[str],
) -> set[str]:
    return set(event_kinds) - set(safe_tail) - set(documented_reject)


def _assert_classified(
    event_kinds: Iterable[str], safe_tail: Iterable[str], documented_reject: dict[str, str],
) -> None:
    safe_kinds = set(safe_tail)
    reject_kinds = set(documented_reject)
    missing = sorted(_unclassified(event_kinds, safe_kinds, reject_kinds))

    assert not missing, (
        "StateFrame journal event kinds must be in SAFE_TAIL or DOCUMENTED_REJECT; "
        f"unclassified ({len(missing)}): {', '.join(missing)}"
    )
    assert not (safe_kinds & reject_kinds), "resume kinds cannot be both safe and rejected"
    assert all(isinstance(reason, str) and reason.strip()
               for reason in documented_reject.values())


@lru_cache(maxsize=1)
def _journal_event_kinds() -> frozenset[str]:
    """Read emitted literal kinds and named event constants from engine source."""
    source_files = sorted(SOURCE_ROOT.rglob("*.py"))
    source_text = {path: path.read_text(encoding="utf-8") for path in source_files}
    constants_by_file: dict[Path, dict[str, set[str]]] = {}
    assignment = re.compile(
        r"^\s*([A-Za-z_]\w*(?:_EVENT(?:_KIND)?|EVENT|EVENT_KIND))"
        r"\s*=\s*(['\"])([^'\"]+)\2",
        re.MULTILINE,
    )
    for path, text in source_text.items():
        constants: dict[str, set[str]] = {}
        for match in assignment.finditer(text):
            constants.setdefault(match.group(1), set()).add(match.group(3))
        constants_by_file[path] = constants

    module_files = {path.stem: path for path in source_files}

    def journal_receiver(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "journal"
        if isinstance(node, ast.Attribute):
            return node.attr == "journal" or journal_receiver(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id == "getattr" and any(
                isinstance(arg, ast.Constant) and arg.value == "journal" for arg in node.args
            )
        return False

    event_kinds: set[str] = set()
    for path, text in source_text.items():
        if "emit(" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        local_symbols = {name: set(values) for name, values in constants_by_file[path].items()}
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def scope(node: ast.AST) -> ast.AST:
            parent = parents.get(node)
            while parent is not None and not isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                parent = parents.get(parent)
            return parent if parent is not None else tree

        journal_emit_aliases: set[tuple[ast.AST, str]] = set()
        module_name = ".".join(path.relative_to(SOURCE_ROOT.parent).with_suffix("").parts)
        if module_name.endswith(".__init__"):
            module_name = module_name[:-9]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_module = node.module
                if node.level:
                    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
                    relative = "." * node.level + (imported_module or "")
                    try:
                        imported_module = importlib.util.resolve_name(relative, package)
                    except ImportError:
                        imported_module = None
                imported_file = None
                if imported_module:
                    imported_file = module_files.get(imported_module.rsplit(".", 1)[-1])
                if imported_file is not None:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        imported_values = constants_by_file[imported_file].get(alias.name, set())
                        if imported_values:
                            local_symbols.setdefault(alias.asname or alias.name, set()).update(imported_values)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == "getattr" and len(value.args) >= 2
                        and isinstance(value.args[1], ast.Constant)
                        and value.args[1].value == "emit" and journal_receiver(value.args[0])):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            journal_emit_aliases.add((scope(node), target.id))
                if value is None:
                    continue
                event_targets = [
                    target for target in targets
                    if isinstance(target, ast.Name)
                    and (target.id.upper().endswith("_EVENT")
                         or target.id.upper().endswith("_EVENT_KIND")
                         or target.id.upper() in {"EVENT", "EVENT_KIND"})
                ]
                for target in event_targets:
                    local_symbols.setdefault(target.id, set()).update(
                        _static_strings(value, local_symbols)
                    )
                continue

        writer_kind_parameters: set[tuple[str, str, int, bool]] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            is_journal_method = (
                isinstance(function, ast.Attribute) and function.attr == "emit"
                and journal_receiver(function.value)
            )
            is_journal_alias = (
                isinstance(function, ast.Name)
                and (scope(node), function.id) in journal_emit_aliases
            )
            # Engine journal adapters also expose the writer as a callback or
            # through an object whose local name is not ``journal`` (for
            # example ``emit(...)``, ``self.emit(...)``, and ``target.emit(...)``).
            # Keep the broader rule scoped to engine modules so the CLI's
            # separate run-event stream is not mistaken for continuation journal.
            is_engine_emit_site = (
                path.is_relative_to(SOURCE_ROOT / "engine")
                and (
                    isinstance(function, ast.Name) and function.id == "emit"
                    or isinstance(function, ast.Attribute) and function.attr == "emit"
                )
            )
            if not (is_journal_method or is_journal_alias or is_engine_emit_site):
                continue
            first_arg = node.args[0]
            event_kinds.update(_static_strings(first_arg, local_symbols))
            function_scope = scope(node)
            if (isinstance(first_arg, ast.Name)
                    and isinstance(function_scope, (ast.FunctionDef, ast.AsyncFunctionDef))):
                parameters = [*function_scope.args.posonlyargs, *function_scope.args.args]
                parameter_names = [parameter.arg for parameter in parameters]
                if first_arg.id in parameter_names:
                    parameter_index = parameter_names.index(first_arg.id)
                    is_method = bool(parameter_names and parameter_names[0] in {"self", "cls"})
                    writer_kind_parameters.add((
                        function_scope.name, first_arg.id, parameter_index, is_method))

        # Resolve common journal adapters such as ``_emit(kind)`` by following
        # their statically visible call sites in the same module.
        for writer_name, parameter_name, parameter_index, is_method in writer_kind_parameters:
            call_index = parameter_index - int(is_method)
            if call_index < 0:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else (
                    function.id if isinstance(function, ast.Name) else None
                )
                if name != writer_name:
                    continue
                if call_index < len(node.args):
                    event_kinds.update(_static_strings(node.args[call_index], local_symbols))
                else:
                    for keyword in node.keywords:
                        if keyword.arg == parameter_name:
                            event_kinds.update(_static_strings(keyword.value, local_symbols))
                            break

    return frozenset(kind for kind in event_kinds if kind and kind.strip() == kind)


def test_every_static_engine_journal_kind_has_a_resume_classification() -> None:
    safe_tail = getattr(restart, "SAFE_TAIL", frozenset())
    documented_reject = getattr(restart, "DOCUMENTED_REJECT", {})
    assert isinstance(documented_reject, dict)
    _assert_classified(_journal_event_kinds(), safe_tail, documented_reject)


def test_unclassified_control_is_detected() -> None:
    planted = "planted_unclassified_resume_event"
    planted_inventory = _journal_event_kinds() | {planted}
    with pytest.raises(AssertionError, match=planted):
        _assert_classified(planted_inventory, restart.SAFE_TAIL, restart.DOCUMENTED_REJECT)
    reason = restart.resume_event_rejection_reason(planted)
    assert isinstance(reason, str) and "unclassified" in reason


def test_callsite_census_tail_events_are_safe() -> None:
    census_kinds = {"callsite_census_outcome", "callsite_census_prepared"}
    assert census_kinds <= restart.SAFE_TAIL
    reject_reason = getattr(restart, "resume_event_rejection_reason", None)
    assert callable(reject_reason)
    assert all(reject_reason(kind) is None for kind in census_kinds)


def test_every_capacity_journal_kind_has_an_explicit_resume_policy() -> None:
    observed = {kind for kind in _journal_event_kinds() if kind.startswith("capacity_")}
    policies = restart.CAPACITY_EVENT_CLASSIFICATION
    assert observed == set(policies), (
        f"capacity event classification mismatch: missing={sorted(observed - set(policies))}, "
        f"stale={sorted(set(policies) - observed)}"
    )

    for kind, (classification, reason) in policies.items():
        assert classification in {"SAFE_TAIL", "REJECT"}, kind
        assert isinstance(reason, str) and reason.strip() and "\n" not in reason, kind
        assert (kind in restart.SAFE_TAIL) is (classification == "SAFE_TAIL"), kind
        if classification == "REJECT":
            assert kind in restart.DOCUMENTED_REJECT, kind
            assert reason in restart.DOCUMENTED_REJECT[kind], kind
        else:
            assert kind not in restart.DOCUMENTED_REJECT, kind
