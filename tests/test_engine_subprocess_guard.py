"""Guard: engine modules may not start processes outside the sandbox boundary.

The scanner walks every engine module and counts references to process-spawning
callables (``subprocess`` entry points, ``os.system``/``os.popen``/``os.exec*``/
``os.spawn*``/``os.posix_spawn*``), including names bound by ``from x import y``
or ``import x as y``. Bare references count too, so ``runner = subprocess.run``
cannot hide a spawn. Today's sites are pinned in ``ALLOWLIST``; any drift fails
until a reviewer updates it.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "src" / "stateframe" / "engine"
ZERO_SITE_MODULES = ("kernel_runner_receipts.py",)

_SUBPROCESS_CALLABLES = frozenset({
    "run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
})
_OS_CALLABLES = frozenset({"system", "popen", "posix_spawn", "posix_spawnp"})
_OS_PREFIXES = ("exec", "spawn")
_SPAWN_MODULES = frozenset({"subprocess", "os"})

# Reviewed site counts per engine module (relative posix path). Update only with review.
ALLOWLIST: dict[str, int] = {
    "blame_channel.py": 1,
    "build_tasks/swebench_adapter.py": 2,
    "build_tasks/swebench_env.py": 1,
    "capability_release_runner.py": 1,
    "check_selection.py": 3,
    "comment_facts.py": 1,
    "expectation_ledger.py": 1,
    "kernel_aha11_native_honesty.py": 1,
    "kernel_bounded_http.py": 3,
    "kernel_checkpoint.py": 1,
    "kernel_env_seed.py": 1,
    "kernel_exec_session.py": 1,
    "kernel_exec_supervisor.py": 2,
    "kernel_execution_policy.py": 2,
    "kernel_hook_driver.py": 1,
    "kernel_mcp_driver.py": 1,
    "kernel_mutate_exec.py": 1,
    "kernel_mutate_store.py": 1,
    "kernel_mutate_sweep.py": 1,
    "kernel_test_isolation.py": 3,
    "kernel_unit_policy.py": 4,
    "kernel_working_backwards.py": 1,
    "probes/react_repair.py": 3,
    "protected_tree_guard.py": 1,
    "realtask_benchmark.py": 1,
    "sf_launch.py": 2,
    "task_runners.py": 2,
}


def _is_spawn_name(module: str, attr: str) -> bool:
    if module == "subprocess":
        return attr in _SUBPROCESS_CALLABLES
    if module == "os":
        return attr in _OS_CALLABLES or attr.startswith(_OS_PREFIXES)
    return False


def _annotation_node_ids(tree: ast.AST) -> set[int]:
    """Ids of nodes living inside type annotations; an annotation names a type, never spawns."""
    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                         node.args.vararg, node.args.kwarg]
            annotations.extend(arg.annotation for arg in arguments
                               if arg is not None and arg.annotation is not None)
    ids: set[int] = set()
    for annotation in annotations:
        ids.update(id(child) for child in ast.walk(annotation))
    return ids


def _count_spawn_sites(source: str) -> list[int]:
    """Return line numbers of every spawn reference in ``source``.

    Calls and bare value references (``runner = subprocess.run``) both count, since a
    reference can be invoked later. Type annotations are ignored.
    """
    tree = ast.parse(source)
    annotation_ids = _annotation_node_ids(tree)
    module_aliases: dict[str, str] = {}
    name_bindings: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _SPAWN_MODULES:
                    module_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in _SPAWN_MODULES:
            for alias in node.names:
                if _is_spawn_name(node.module, alias.name):
                    name_bindings[alias.asname or alias.name] = (node.module, alias.name)

    lines: list[int] = []
    for node in ast.walk(tree):
        if id(node) in annotation_ids:
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = module_aliases.get(node.value.id)
            if module is not None and _is_spawn_name(module, node.attr):
                lines.append(node.lineno)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            binding = name_bindings.get(node.id)
            if binding is not None:
                lines.append(node.lineno)
    return sorted(lines)


def _scan_engine() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        sites = _count_spawn_sites(path.read_text(encoding="utf-8"))
        if sites:
            found[path.relative_to(ENGINE_DIR).as_posix()] = sites
    return found


def test_engine_process_spawn_sites_match_reviewed_allowlist():
    assert any(ENGINE_DIR.rglob("*.py")), "engine scan found no modules; path is wrong"
    found = _scan_engine()
    problems: list[str] = []
    for module, sites in found.items():
        expected = ALLOWLIST.get(module)
        if expected is None:
            problems.append(
                f"{module}: new unsandboxed process call sites at lines {sites}; route it "
                "through kernel_execution_policy.run_sandboxed or add it to ALLOWLIST with review"
            )
        elif len(sites) != expected:
            problems.append(
                f"{module}: expected {expected} process call sites, found {len(sites)} at "
                f"lines {sites}; route new sites through kernel_execution_policy.run_sandboxed "
                "or update ALLOWLIST with review"
            )
    for module, expected in ALLOWLIST.items():
        if module not in found:
            problems.append(f"{module}: allowlisted with {expected} sites but none remain; "
                            "remove the stale ALLOWLIST entry")
    assert not problems, "\n".join(problems)


def test_zero_site_modules_stay_free_of_process_spawns():
    found = _scan_engine()
    for module in ZERO_SITE_MODULES:
        assert module not in found, f"{module} must not spawn processes: {found.get(module)}"


def test_scanner_detects_direct_and_aliased_spawns():
    direct = "import subprocess\nsubprocess.run(['ls'])\n"
    aliased = "from subprocess import Popen as P\nP(['ls'])\n"
    module_alias = "import subprocess as sp\nrunner = sp.check_output\n"
    os_system = "import os\nos.system('ls')\nos.spawnv(0, 'ls', ['ls'])\n"
    unrelated = "import subprocess\nresult = subprocess.CompletedProcess\nos_path = 1\n"
    annotated = (
        "import subprocess\n"
        "def f(p: subprocess.Popen[bytes]) -> subprocess.Popen | None:\n"
        "    q: subprocess.Popen | None = None\n"
        "    return q\n"
    )

    assert _count_spawn_sites(direct) == [2]
    assert _count_spawn_sites(aliased) == [2]
    assert _count_spawn_sites(module_alias) == [2]
    assert _count_spawn_sites(os_system) == [2, 3]
    assert _count_spawn_sites(unrelated) == []
    assert _count_spawn_sites(annotated) == []
