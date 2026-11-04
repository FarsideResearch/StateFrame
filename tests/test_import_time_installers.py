"""Every import-time installer is either pre-imported by conftest or a reviewed exemption.

A module that calls an ``install*()`` function at top level mutates process globals the moment it is first
imported. When that first import happens inside a test, the installer-attribute guard in conftest reports
the test as leaking state it never created (11-03, v0.3.29). conftest therefore imports these modules before
it snapshots the pristine baseline; this guard keeps that list complete.
"""
import ast
from pathlib import Path

from _import_time_installers import IMPORT_TIME_INSTALLER_MODULES

ROOT = Path(__file__).resolve().parents[1] / "src" / "stateframe"
# Reviewed: import-time installers that touch none of conftest's watched installer attributes.
EXEMPT = {
    "kernel_py39_compat": "Patches enum/typing/dataclasses only, and only on Python < 3.10; no watched attribute.",
}


def import_time_installers(source: str) -> list[str]:
    """Names of install*() functions called at module top level (directly or inside a top-level try)."""
    found = []
    for node in ast.parse(source).body:
        statements = node.body if isinstance(node, ast.Try) else [node]
        for statement in statements:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                func = statement.value.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name.startswith("install"):
                    found.append(name)
    return found


def _modules_with_import_time_installers() -> set[str]:
    modules = set()
    for path in ROOT.rglob("*.py"):
        if import_time_installers(path.read_text(encoding="utf-8")):
            modules.add(path.stem)
    return modules


def test_every_import_time_installer_is_preimported_or_exempt():
    found = _modules_with_import_time_installers()
    unlisted = found - set(IMPORT_TIME_INSTALLER_MODULES) - set(EXEMPT)
    assert not unlisted, (
        f"modules install at import time but conftest does not pre-import them: {sorted(unlisted)}; "
        "add them to conftest.IMPORT_TIME_INSTALLER_MODULES (or exempt them here with a reason)")


def test_listed_modules_still_install_at_import():
    found = _modules_with_import_time_installers()
    stale = (set(IMPORT_TIME_INSTALLER_MODULES) | set(EXEMPT)) - found
    assert not stale, f"listed modules no longer install at import time; drop them: {sorted(stale)}"


def test_planted_import_time_installer_is_detected():
    planted = (
        "from x import install_patch\n"
        "install_patch()\n"
        "try:\n"
        "    install_other()\n"
        "except ImportError:\n"
        "    pass\n"
        "def later():\n"
        "    install_inside_function()\n"
    )
    assert import_time_installers(planted) == ["install_patch", "install_other"]
