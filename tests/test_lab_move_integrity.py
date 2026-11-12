"""Moved lab modules must not remain in run identity or process guards."""

from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/stateframe/engine"


def test_run_identity_lists_only_present_static_modules() -> None:
    tree = ast.parse((ENGINE / "kernel_stateframe_runtime.py").read_text(encoding="utf-8"))
    module_lists = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "modules"
                for target in node.targets)
        and isinstance(node.value, ast.Call)
    ]
    assert len(module_lists) == 1
    static_paths = {
        node.value for node in ast.walk(module_lists[0])
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str) and node.value.endswith(".py")
    }
    assert static_paths
    assert not {name for name in static_paths if not (ENGINE / name).is_file()}


def test_converge_process_guard_names_moved_driver() -> None:
    source = (ROOT / "bench/lab/converge.py").read_text(encoding="utf-8")
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.List)
        and [item.value for item in node.args[0].elts[:2]
             if isinstance(item, ast.Constant)] == ["pgrep", "-f"]
    ]
    assert len(calls) == 1
    assert any(keyword.arg == "check" and isinstance(keyword.value, ast.Constant)
               and keyword.value.value is True for keyword in calls[0].keywords)
    pattern = calls[0].args[0].elts[2].value
    assert isinstance(pattern, str)
    for command in ("python bench/lab/converge.py", "/usr/bin/python /repo/bench/lab/pilot.py",
                    "python pilot.py 21 2"):
        assert re.search(pattern, command), command
