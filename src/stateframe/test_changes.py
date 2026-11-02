"""Describe changed test files and flag simple weakening signals.

Findings are heuristics for review, never a proof that a test is sound.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from .test_runners import is_test_file
from .workcopy import WorkCopy, changed_files


_WEAK_MARKERS = re.compile(r"(?:\.skip(?:if|Test)?\b|\.expectedFailure\b|\.importorskip\b|\.only\b|\bxfail\b|\bskip\b|\btodo\b)", re.I)
_ASSERT_LINE = re.compile(
    r"\b(?:assert|expect|should)\b|\.assert\w*\s*\(|\bt\.(?:Fail|Error|Fatal)\w*\s*\("
    r"|\bassert(?:_eq|_ne)?!\s*\(|\bif\s+\[\[?\s+[^\n]+\s+\]\]?", re.I)
_NAMED_TEST = re.compile(
    r"\b(?:test|it)\s*\(\s*['\"](?P<js>[^'\"]+)"
    r"|\bfunc\s+(?P<go>Test\w+)\s*\("
    r"|\bfn\s+(?P<rust>test_\w+)\s*\("
    r"|(?m:^\s*(?P<shell>test_\w+)\s*\(\s*\)\s*\{)"
)


def _python_tree(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, RecursionError):
        return None


def _python_tests(tree: ast.Module | None) -> dict[str, tuple[ast.AST, ast.ClassDef | None]]:
    if tree is None:
        return {}
    tests = {}
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    testcase_names = {"TestCase"}
    unittest_names = {"unittest"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            unittest_names.update(alias.asname or "unittest" for alias in node.names
                                  if alias.name == "unittest")
        elif isinstance(node, ast.ImportFrom) and node.module == "unittest":
            testcase_names.update(alias.asname or alias.name for alias in node.names
                                  if alias.name == "TestCase")

    def is_unittest(cls: ast.ClassDef, seen: set[str]) -> bool:
        if cls.name in seen:
            return False
        for base in cls.bases:
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                if base.value.id in unittest_names and base.attr == "TestCase":
                    return True
            if isinstance(base, ast.Name):
                if base.id in testcase_names:
                    return True
                parent = classes.get(base.id)
                if parent is not None and is_unittest(parent, seen | {cls.name}):
                    return True
        return False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            tests[node.name] = (node, None)
        elif isinstance(node, ast.ClassDef) and (node.name.startswith("Test") or is_unittest(node, set())):
            unittest_class = is_unittest(node, set())
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith(
                        "test" if unittest_class else "test_"):
                    tests[f"{node.name}::{child.name}"] = (child, node)
    return tests


def _assertions(node: ast.AST) -> int:
    return sum(isinstance(part, ast.Assert) or
               (isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                and part.func.attr.startswith("assert")) for part in ast.walk(node))


def _assertion_comparisons(node: ast.AST) -> list[tuple[str, ...]]:
    """Keep each assertion's operator and operand AST together, including chains."""
    result = []
    for part in ast.walk(node):
        if not isinstance(part, ast.Assert):
            continue
        result.append(tuple(ast.dump(compare, include_attributes=False)
                            for compare in ast.walk(part.test) if isinstance(compare, ast.Compare)))
    return result


def _assertion_predicates(node: ast.AST) -> list[ast.expr]:
    return [part.test for part in ast.walk(node) if isinstance(part, ast.Assert)]


def _assert_calls(node: ast.AST) -> list[ast.Call]:
    return [part for part in ast.walk(node) if isinstance(part, ast.Call)
            and isinstance(part.func, ast.Attribute) and part.func.attr.startswith("assert")]


def _terminates_block(statements: list[ast.stmt]) -> bool:
    """Recognize only exits guaranteed by the local syntax, without evaluating code."""
    for statement in statements:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, ast.If):
            if isinstance(statement.test, ast.Constant) and isinstance(statement.test.value, bool):
                branch = statement.body if statement.test.value else statement.orelse
                if _terminates_block(branch):
                    return True
            elif statement.orelse and _terminates_block(statement.body) and _terminates_block(
                    statement.orelse):
                return True
    return False


def _unreachable_assertions(node: ast.AST) -> Counter[str]:
    """Assertions after a definite top-level exit in a test body."""
    result: Counter[str] = Counter()
    body = getattr(node, "body", [])
    for index, statement in enumerate(body):
        if not _terminates_block([statement]):
            continue
        for suffix in body[index + 1:]:
            for part in ast.walk(suffix):
                if isinstance(part, ast.Assert) or (isinstance(part, ast.Call)
                        and isinstance(part.func, ast.Attribute)
                        and part.func.attr.startswith("assert")):
                    result[ast.dump(part, include_attributes=False)] += 1
        break
    return result


def _call_tautology(call: ast.Call) -> bool:
    name = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    if name in {"assertEqual", "assertIs", "assertAlmostEqual"} and len(call.args) >= 2:
        return ast.dump(call.args[0], include_attributes=False) == ast.dump(
            call.args[1], include_attributes=False)
    return name == "assertTrue" and bool(call.args) and _always_true(call.args[0])


def _always_true(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant):
        return bool(expr.value)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return isinstance(expr.operand, ast.Constant) and not bool(expr.operand.value)
    if isinstance(expr, ast.BoolOp):
        tests = [_always_true(value) for value in expr.values]
        return any(tests) if isinstance(expr.op, ast.Or) else all(tests)
    return False


def _required_checks(expr: ast.expr) -> Counter[str]:
    """Conjuncts are independent requirements; dropping one weakens an assertion."""
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        result: Counter[str] = Counter()
        for value in expr.values:
            result.update(_required_checks(value))
        return result
    return Counter([ast.dump(expr, include_attributes=False)])


def _marker_nodes(nodes: list[ast.stmt] | list[ast.expr]) -> Counter[str]:
    markers: Counter[str] = Counter()
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
                markers[ast.dump(node, include_attributes=False)] += 1
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and _WEAK_MARKERS.search(
                ast.unparse(node.value.func)):
            markers[ast.dump(node, include_attributes=False)] += 1
        elif isinstance(node, ast.expr) and _WEAK_MARKERS.search(ast.unparse(node)):
            markers[ast.dump(node, include_attributes=False)] += 1
    return markers


def _inherited_markers(tree: ast.Module | None, cls: ast.ClassDef | None) -> Counter[str]:
    if tree is None:
        return Counter()
    markers = _marker_nodes(tree.body)
    if cls is not None:
        markers.update(_marker_nodes(cls.decorator_list))
        markers.update(_marker_nodes(cls.body))
    return markers


def _test_markers(node: ast.AST) -> Counter[str]:
    markers = _marker_nodes(getattr(node, "decorator_list", []))
    for part in ast.walk(node):
        if isinstance(part, ast.Call) and _WEAK_MARKERS.search(ast.unparse(part.func)):
            markers[ast.dump(part, include_attributes=False)] += 1
    return markers


def _finding(path: str, test: str, reason: str) -> dict[str, str]:
    return {"test": f"{path}::{test}", "reason": reason, "method": "heuristic"}


def _python_findings(path: str, before: str, after: str) -> list[dict[str, str]]:
    old_tree, new_tree = _python_tree(before), _python_tree(after)
    old, new = _python_tests(old_tree), _python_tests(new_tree)
    findings = []
    for name, (original, old_class) in old.items():
        replacement = new.get(name)
        if replacement is None:
            findings.append(_finding(path, name, "test function removed"))
            continue
        changed, new_class = replacement
        if _assertions(changed) < _assertions(original):
            findings.append(_finding(path, name, "assertion removed"))
        if _unreachable_assertions(changed) - _unreachable_assertions(original):
            findings.append(_finding(path, name, "assertion made unreachable"))
        old_predicates = _assertion_predicates(original)
        new_predicates = _assertion_predicates(changed)
        tautology = any(_always_true(new_test) and not _always_true(old_test)
                        for old_test, new_test in zip(old_predicates, new_predicates))
        if tautology:
            findings.append(_finding(path, name, "assertion made tautological"))
        elif any((_required_checks(old_test) - _required_checks(new_test)
                  and _required_checks(new_test) < _required_checks(old_test))
                 for old_test, new_test in zip(old_predicates, new_predicates)):
            findings.append(_finding(path, name, "assertion predicate reduced"))
        # A retained comparison is still defeated when ``or True`` makes the
        # enclosing assertion pass regardless of that comparison.
        if ((_assertion_comparisons(changed) != _assertion_comparisons(original)
             or (tautology and any(_assertion_comparisons(original))))
                and _assertions(changed) >= _assertions(original)):
            findings.append(_finding(path, name, "comparison loosened"))
        old_calls, new_calls = _assert_calls(original), _assert_calls(changed)
        for old_call, new_call in zip(old_calls, new_calls):
            if _call_tautology(new_call) and not _call_tautology(old_call):
                findings.append(_finding(path, name, "assertion made tautological"))
            elif (ast.dump(old_call, include_attributes=False) !=
                  ast.dump(new_call, include_attributes=False)):
                findings.append(_finding(path, name, "comparison loosened"))
        old_markers = _inherited_markers(old_tree, old_class) + _test_markers(original)
        new_markers = _inherited_markers(new_tree, new_class) + _test_markers(changed)
        if new_markers - old_markers:
            findings.append(_finding(path, name, "skip/xfail/only/todo marker added"))
    return findings


def _named_blocks(source: str) -> dict[str, str]:
    matches = list(_NAMED_TEST.finditer(source))
    return {next(value for value in match.groupdict().values() if value is not None):
            source[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(source)]
            for index, match in enumerate(matches)}


def _line_findings(path: str, before: str, after: str) -> list[dict[str, str]]:
    old_blocks, new_blocks = _named_blocks(before), _named_blocks(after)
    if not old_blocks and not new_blocks:
        old_blocks, new_blocks = {"file": before}, {"file": after}
    findings = []
    for name, original in old_blocks.items():
        changed = new_blocks.get(name)
        if changed is None:
            findings.append(_finding(path, name, "test function removed"))
            continue
        if len(_ASSERT_LINE.findall(changed)) < len(_ASSERT_LINE.findall(original)):
            findings.append(_finding(path, name, "assertion removed"))
        if set(_WEAK_MARKERS.findall(changed)) - set(_WEAK_MARKERS.findall(original)):
            findings.append(_finding(path, name, "skip/xfail/only/todo marker added"))
        if (original.count("===") > changed.count("===")
                and changed.count("==") > original.count("==")) or (
                original.count("==") > changed.count("==")
                and changed.count("!=") > original.count("!=")):
            findings.append(_finding(path, name, "comparison loosened"))
        tautology = re.compile(r"expect\s*\(\s*true\s*\)\s*\.toBe\s*\(\s*true\s*\)", re.I)
        if len(tautology.findall(changed)) > len(tautology.findall(original)):
            findings.append(_finding(path, name, "assertion made tautological"))
    return findings


def inspect_test_changes(work: WorkCopy) -> list[dict[str, object]]:
    results = []
    for path, change in changed_files(work).items():
        if not is_test_file(path):
            continue
        original = work.snapshot or work.original
        before = (original / path).read_text(encoding="utf-8", errors="replace") if change != "added" else ""
        after = (work.copy / path).read_text(encoding="utf-8", errors="replace") if change != "deleted" else ""
        if change == "deleted":
            findings = [_finding(path, "file", "test file removed")]
        elif change == "added":
            findings = []
        elif Path(path).suffix == ".py":
            findings = _python_findings(path, before, after)
        else:
            findings = _line_findings(path, before, after)
        results.append({"path": path, "change": change, "weakening": findings})
    return results
