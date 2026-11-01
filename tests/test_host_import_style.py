"""Host-imported engine modules must support package and flat child imports."""
import ast
import os
from pathlib import Path
import subprocess
import sys

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
SRC = ENGINE.parent.parent
HOST_IMPORTED = {"kernel_hooks.py", "kernel_hook_exec.py"}


def _host_importable(tree: ast.Module, filename: str = "") -> bool:
    # The dual-import idiom `if __package__:` marks a module the host imports as part of the package.
    return filename in HOST_IMPORTED or any(isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "__package__"
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


def _module_level_bare_imports(tree: ast.Module) -> list[str]:
    """Find unguarded imports executed while loading a package module."""
    guarded = _package_guarded(tree)
    nested = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for child in ast.walk(node)
    }
    found = []
    for node in ast.walk(tree):
        if id(node) in guarded or id(node) in nested:
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("kernel_"):
            found.append(f"module:{node.lineno}:{node.module}")
        elif isinstance(node, ast.Import):
            found += [f"module:{node.lineno}:{alias.name}" for alias in node.names
                      if alias.name.startswith("kernel_")]
    return found


def test_host_importable_engine_modules_use_package_imports_in_functions():
    offenders = {}
    checked = 0
    for path in sorted(ENGINE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _host_importable(tree, path.name):
            continue
        checked += 1
        bad = _module_level_bare_imports(tree) + _function_level_bare_imports(tree)
        if bad:
            offenders[path.name] = bad
    assert checked >= 3, "guard found no host-importable modules; the detector is broken"
    assert not offenders, f"bare kernel_* imports in host-importable modules: {offenders}"


def test_guard_detects_a_planted_function_level_import():
    planted = ast.parse("if __package__:\n    pass\n\ndef f():\n    from kernel_x import y\n")
    assert _host_importable(planted) and _function_level_bare_imports(planted) == ["f:5:kernel_x"]
    dual = ast.parse("if __package__:\n    pass\n\ndef f():\n    if __package__:\n        from .kernel_x import y\n"
                     "    else:\n        from kernel_x import y\n")
    assert _function_level_bare_imports(dual) == []


def test_guard_detects_a_planted_module_level_import():
    planted = ast.parse("from kernel_x import y\n")
    assert _host_importable(planted, "kernel_hooks.py")
    assert _module_level_bare_imports(planted) == ["module:1:kernel_x"]
    dual = ast.parse("if __package__:\n    from .kernel_x import y\n"
                     "else:\n    from kernel_x import y\n")
    assert _module_level_bare_imports(dual) == []


def test_hook_host_imports_with_only_src_on_pythonpath(tmp_path):
    script = tmp_path / "import_hook_host.py"
    script.write_text(
        "import sys\n"
        f"assert {str(ENGINE)!r} not in sys.path\n"
        "import stateframe.hook_host\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-S", str(script)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
