"""The broker boot dir must hold every engine module its files import at load time.

Regression: kernel_web_fetch gained a module-level kernel_reason_log import that was
never copied into the boot dir, so every full-network mutation launch died with
BootError "No module named kernel_reason_log".
"""
from __future__ import annotations

import ast
from pathlib import Path

from stateframe.engine import kernel_mutate_exec

ENGINE = Path(kernel_mutate_exec.__file__).resolve().parent
# The trusted boot set every mutation launch stages, plus the broker-only additions.
BOOT_FILES = kernel_mutate_exec._BOOT_FILES
BROKER_FILES = kernel_mutate_exec._BROKER_BOOT_FILES


def _load_time_engine_imports(path: Path) -> set[str]:
    """Engine module names imported at module level (incl. module-level try blocks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def visit(nodes):
        for node in nodes:
            if isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, (ast.If, ast.With, ast.For, ast.While)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
            elif isinstance(node, ast.ImportFrom):
                # `from . import x` carries the names in aliases, not in node.module.
                names = [node.module] if node.module else [a.name for a in node.names]
                for raw in names:
                    name = raw.split(".")[0]
                    if (ENGINE / f"{name}.py").is_file() or (ENGINE / name).is_dir():
                        found.add(f"{name}.py")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if (ENGINE / f"{name}.py").is_file() or (ENGINE / name).is_dir():
                        found.add(f"{name}.py")

    visit(tree.body)
    return found


def test_broker_boot_set_is_import_closed():
    shipped = set(BOOT_FILES) | set(BROKER_FILES)
    missing = {}
    for name in sorted(shipped):
        need = _load_time_engine_imports(ENGINE / name) - shipped
        if need:
            missing[name] = sorted(need)
    assert missing == {}


def test_reason_log_is_a_broker_dependency():
    assert "kernel_reason_log.py" in BROKER_FILES
    assert "kernel_web_fetch.py" in BROKER_FILES


def test_closure_checker_sees_try_fallback_imports():
    # Guard the checker itself: the regression import sat inside a try/except fallback.
    assert "kernel_reason_log.py" in _load_time_engine_imports(ENGINE / "kernel_web_fetch.py")


def test_every_broker_dependency_exists():
    for name in BROKER_FILES:
        assert (ENGINE / name).is_file(), name


def test_checker_sees_relative_with_and_package_imports(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from contextlib import suppress\n"
        "with suppress(ImportError):\n    import kernel_reason_log\n"
        "from . import kernel_web_fetch\n",
        encoding="utf-8")
    assert {"kernel_reason_log.py", "kernel_web_fetch.py"} <= _load_time_engine_imports(probe)


def test_launch_stages_exactly_the_checked_set(tmp_path):
    # The closure test is only meaningful if the launcher stages exactly these tuples.
    boot = tmp_path / "boot"
    boot.mkdir()
    staged = kernel_mutate_exec.stage_boot_files(boot, "full")
    assert set(staged) == set(BOOT_FILES) | set(BROKER_FILES)
    assert sorted(p.name for p in boot.iterdir()) == sorted(staged)
