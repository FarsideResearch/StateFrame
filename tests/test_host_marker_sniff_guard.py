"""No new engine function may decide authorship by sniffing host-marker text.

A sniffer holds a host-marker literal (or a module constant bound to one) and
applies a text test to anything. It passes if it asks the host-authorship read
API, or if it is in the frozen baseline of functions that predate the ledger.
Only growth fails. A baseline entry that no longer sniffs is a warning; run
``python tests/test_host_marker_sniff_guard.py --regen`` to rewrite the baseline.
"""
from __future__ import annotations

import ast
import functools
import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
BASELINE = Path(__file__).with_name("host_marker_sniff_baseline.json")
MARKER = re.compile(
    r"\[EVIDENCEPIN\]|\[CONTINUATION LIMITS|\[(?:Current|Working|Observed|Retained|Repair|Paged) source"
    r"|\[Request budget source window\]|\[Model source window\]|\[End source window\]"
    r"|\[StateFrame\]|CURRENT_SOURCE ")
TEXT_TESTS = frozenset({"startswith", "find", "split", "rsplit", "count", "search", "match",
                        "partition", "index"})
READ_API = frozenset({"row_kind", "row_kinds", "host_spans", "host_text", "find_block",
                      "structural_class", "data_spans"})
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _has_marker(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Constant) and isinstance(child.value, str)
               and MARKER.search(child.value) for child in ast.walk(node))


def _module_constants(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) and node.value else [])
        value = getattr(node, "value", None)
        if targets and value is not None and _has_marker(value):
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _imported(tree: ast.AST) -> set[str]:
    return {alias.asname or alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names}


def scan_source(source: str, filename: str, *, shared_constants: frozenset[str] = frozenset(),
                tree: ast.Module | None = None) -> dict[str, bool]:
    """``{file::qualname: uses_read_api}`` for every sniffer in one module."""
    tree = ast.parse(source) if tree is None else tree
    constants = _module_constants(tree) | (_imported(tree) & shared_constants)
    found: dict[str, bool] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNCTIONS):
                name = f"{prefix}{child.name}"
                holds = _has_marker(child) or any(
                    isinstance(item, ast.Name) and item.id in constants for item in ast.walk(child))
                tests = any(
                    (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                     and item.func.attr in TEXT_TESTS)
                    or (isinstance(item, ast.Compare)
                        and any(isinstance(op, (ast.In, ast.NotIn)) for op in item.ops))
                    for item in ast.walk(child))
                if holds and tests:
                    found[f"{filename}::{name}"] = any(
                        isinstance(item, ast.Call)
                        and getattr(item.func, "attr", getattr(item.func, "id", None)) in READ_API
                        for item in ast.walk(child))
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


@functools.lru_cache(maxsize=1)
def _scan_engine() -> tuple[tuple[str, bool], ...]:
    trees = {path: ast.parse(path.read_text()) for path in sorted(ENGINE.rglob("*.py"))}
    shared: set[str] = set()
    for tree in trees.values():
        shared |= _module_constants(tree)
    found: dict[str, bool] = {}
    for path, tree in trees.items():
        found.update(scan_source("", path.relative_to(ENGINE).as_posix(),
                                 shared_constants=frozenset(shared), tree=tree))
    return tuple(sorted(found.items()))


def scan_engine() -> dict[str, bool]:
    return dict(_scan_engine())


def check(found: dict[str, bool], baseline: set[str]) -> tuple[list[str], list[str]]:
    """(new sniffers that must fail, baseline entries that no longer sniff)."""
    new = sorted(name for name, uses_api in found.items() if not uses_api and name not in baseline)
    gone = sorted(name for name in baseline if name not in found or found[name])
    return new, gone


def _baseline() -> set[str]:
    return set(json.loads(BASELINE.read_text())["sniffers"])


def assert_guard(found: dict[str, bool], baseline: set[str]) -> None:
    new, gone = check(found, baseline)
    if gone:
        warnings.warn("host-marker sniff baseline can shrink (run --regen): " + ", ".join(gone[:8]),
                      UserWarning, stacklevel=2)
    assert not new, (
        "new host-marker sniffers; decide authorship through kernel_host_authorship "
        "(row_kind / host_spans / find_block): " + ", ".join(new))


PLANTED = '''
def leak(content):
    return content.startswith("[Current source:")

def clean(ledger, message):
    from kernel_host_authorship import row_kind
    return row_kind(ledger, message) == "plain_source" and "[Current source:" in message["content"]

def writer(path):
    return "[Current source: " + path + "]"
'''


def test_planted_sniffer_is_reported():
    found = scan_source(PLANTED, "planted.py")
    assert found == {"planted.py::leak": False, "planted.py::clean": True}
    # A module constant bound to a marker counts the same as the literal.
    constant = 'PREFIX = ("[StateFrame]", "x")\nclass A:\n    def m(self, t):\n        return PREFIX[0] in t\n'
    assert scan_source(constant, "c.py") == {"c.py::A.m": False}
    imported = "from other import STATE_HEADER\ndef f(t):\n    return t.find(STATE_HEADER)\n"
    assert scan_source(imported, "i.py", shared_constants=frozenset({"STATE_HEADER"})) == {"i.py::f": False}
    assert scan_source(imported, "i.py") == {}


def test_new_sniffer_outside_baseline_fails():
    import pytest
    found = scan_engine()
    baseline = _baseline()
    assert len(baseline) > 100
    assert_guard(found, baseline)  # the engine as shipped is inside the frozen baseline
    planted = {**found, **scan_source(PLANTED, "planted.py")}
    new, _gone = check(planted, baseline)
    assert new == ["planted.py::leak"]
    with pytest.raises(AssertionError, match="planted.py::leak"):
        assert_guard(planted, baseline)


def test_shrunk_baseline_warns_not_fails():
    import pytest
    found = scan_engine()
    baseline = _baseline() | {"kernel_removed.py::no_longer_sniffs"}
    with pytest.warns(UserWarning, match="kernel_removed.py::no_longer_sniffs"):
        assert_guard(found, baseline)
    # A baseline function converted to the read API is reported the same way, never as a failure.
    converted = dict(found)
    name = sorted(_baseline())[0]
    converted[name] = True
    with pytest.warns(UserWarning, match=re.escape(name)):
        assert_guard(converted, _baseline())


if __name__ == "__main__":
    if "--regen" not in sys.argv:
        raise SystemExit("usage: python tests/test_host_marker_sniff_guard.py --regen")
    names = sorted(name for name, uses_api in scan_engine().items() if not uses_api)
    BASELINE.write_text(json.dumps({"sniffers": names}, indent=1) + "\n")
    print(f"{BASELINE.name}: {len(names)} sniffers")
