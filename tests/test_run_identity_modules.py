"""Run identity lists only engine modules that are present in the tree."""

from __future__ import annotations

import ast
from pathlib import Path


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

