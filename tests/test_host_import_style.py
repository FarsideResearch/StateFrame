"""Engine modules the host imports as `stateframe.engine.X` must not do bare `kernel_*` imports inside
functions: the engine directory is on sys.path only in the engine child, so such an import works in tests
and in the child but raises ModuleNotFoundError in the host CLI (found on Linux, where the bwrap branch ran it).
"""
import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"


def _host_importable(tree: ast.Module) -> bool:
    # The dual-import idiom `if __package__:` marks a module the host imports as part of the package.
    return any(isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "__package__"
               for node in tree.body)


def _package_guarded(function: ast.AST) -> set[int]:
    """ids of nodes inside the `else:` of an `if __package__:` (the engine-child spelling)."""
    guarded = set()
    for node in ast.walk(function):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "__package__":
            for branch in node.orelse:
                guarded |= {id(child) for child in ast.walk(branch)}
    return guarded


def _function_level_bare_imports(tree: ast.Module) -> list[str]:
    found = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        guarded = _package_guarded(function)
        for node in ast.walk(function):
            if id(node) in guarded:
                continue
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("kernel_"):
                found.append(f"{function.name}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                found += [f"{function.name}:{node.lineno}:{a.name}" for a in node.names if a.name.startswith("kernel_")]
    return found


def test_host_importable_engine_modules_use_package_imports_in_functions():
    offenders = {}
    checked = 0
    for path in sorted(ENGINE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _host_importable(tree):
            continue
        checked += 1
        bad = _function_level_bare_imports(tree)
        if bad:
            offenders[path.name] = bad
    assert checked >= 3, "guard found no host-importable modules; the detector is broken"
    assert not offenders, f"bare kernel_* imports inside functions of host-importable modules: {offenders}"


def test_guard_detects_a_planted_function_level_import():
    planted = ast.parse("if __package__:\n    pass\n\ndef f():\n    from kernel_x import y\n")
    assert _host_importable(planted) and _function_level_bare_imports(planted) == ["f:5:kernel_x"]
    dual = ast.parse("if __package__:\n    pass\n\ndef f():\n    if __package__:\n        from .kernel_x import y\n"
                     "    else:\n        from kernel_x import y\n")
    assert _function_level_bare_imports(dual) == []
