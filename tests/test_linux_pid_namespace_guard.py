"""Every Linux bwrap launch-argv builder must give the command its own PID namespace.

Inside its own PID namespace an orphaned descendant reparents to the namespace's init and dies with it, so on
Linux an "untracked descendant" is structurally impossible (a child subreaper would add nothing; WORK_QUEUE#23).
This guard keeps that property from regressing silently: it finds bwrap argv builders mechanically (a function
whose source builds a list literal containing "--die-with-parent", the flag every bwrap launch in this engine
passes) and requires each one to carry "--unshare-pid" or "--unshare-all", unless it is a reviewed exemption.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "stateframe"
NAMESPACE_FLAGS = frozenset({"--unshare-pid", "--unshare-all"})
BWRAP_MARKER = "--die-with-parent"

# Reviewed exemptions: builders that intentionally share the host PID namespace, with the reason.
EXEMPT = {
    ("engine/protected_tree_guard.py", "guarded_command"):
        "Write guard used only by capability_release_runner to run the project's own worker scripts read-only "
        "over protected trees; it never launches model or user commands.",
}
# Builders found when the guard was written (11-03, v0.3.26): exec_session's launcher and the two
# test-isolation argv builders. A drop below this means the scan stopped seeing a builder.
MIN_BUILDERS = 3


def _string_constants(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def bwrap_builders(source: str) -> list[tuple[str, set[str]]]:
    """(function name, string constants in its body) for each function building a bwrap argv list."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        builds_argv = any(isinstance(item, ast.List) and BWRAP_MARKER in _string_constants(item)
                          for item in ast.walk(node))
        if builds_argv:
            found.append((node.name, _string_constants(node)))
    return found


def violations(sources: dict[str, str], exempt: dict) -> tuple[list[str], int]:
    """Builders lacking a PID-namespace flag (not exempt), and the number of builders seen."""
    problems, seen = [], 0
    for relative, source in sorted(sources.items()):
        for name, constants in bwrap_builders(source):
            seen += 1
            if (relative, name) in exempt:
                continue
            if not constants & NAMESPACE_FLAGS:
                problems.append(f"{relative}:{name} builds a bwrap argv without --unshare-pid/--unshare-all")
    return problems, seen


def _engine_sources() -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(ROOT.rglob("*.py"))}


def test_every_linux_bwrap_builder_uses_a_pid_namespace():
    problems, seen = violations(_engine_sources(), EXEMPT)
    assert problems == []
    assert seen - len(EXEMPT) >= MIN_BUILDERS, (
        f"only {seen} bwrap argv builders found ({len(EXEMPT)} exempt); the scan may have lost one")


def test_exemptions_still_exist():
    sources = _engine_sources()
    for relative, name in EXEMPT:
        assert relative in sources, f"exempt file {relative} no longer exists; drop the exemption"
        assert name in {n for n, _ in bwrap_builders(sources[relative])}, (
            f"{relative}:{name} no longer builds a bwrap argv; drop the exemption")


def test_planted_builder_without_namespace_is_caught():
    planted = (
        "def launch(cmd):\n"
        "    return ['bwrap', '--die-with-parent', '--bind', '/', '/', *cmd]\n"
        "def isolated(cmd):\n"
        "    return ['bwrap', '--die-with-parent', '--unshare-all', *cmd]\n"
        "def unrelated():\n"
        "    return ['ls', '-l']\n"
    )
    problems, seen = violations({"engine/planted.py": planted}, {})
    assert seen == 2
    assert problems == ["engine/planted.py:launch builds a bwrap argv without --unshare-pid/--unshare-all"]


def test_planted_flag_added_later_in_the_same_builder_counts():
    planted = (
        "def launch(cmd, shared):\n"
        "    args = ['bwrap', '--die-with-parent']\n"
        "    if shared:\n"
        "        args += ['--unshare-pid', '--proc', '/proc']\n"
        "    return [*args, *cmd]\n"
    )
    assert violations({"engine/planted.py": planted}, {}) == ([], 1)
