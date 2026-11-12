"""Class guard: every parent-bound sandbox launch site is a reviewed one.

bwrap --die-with-parent (PR_SET_PDEATHSIG) kills the sandbox when the THREAD that forked it
exits. A launch site is safe only when the forking thread outlives the child: it waits for the
child itself, or it is the main thread (ExecSession._start asserts that). Any new site that
builds or launches such a command must be reviewed for that, then added here.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "stateframe"
FLAG = "--die-with-parent"
BUILDERS = {"isolated_command", "capability_command", "guarded_command"}

# Functions that put the flag on a command line.
FLAG_SITES = {
    "engine/kernel_exec_session.py::ExecSession._start",
    "engine/kernel_test_isolation.py::capability_command",
    "engine/kernel_test_isolation.py::isolated_command",
    "engine/kernel_test_isolation.py::_mutation_command",
    "engine/protected_tree_guard.py::guarded_command",
}

# Functions that use a flag-emitting builder, each reviewed: the forking thread waits for
# the child in place, or runs a supervisor that is itself started from the main thread.
BUILDER_USERS = {
    "engine/capability_release_runner.py::supervise",
    "engine/kernel_exec_supervisor.py::Supervisor.start",
    "engine/kernel_execution_policy.py::_prepared",
    "engine/kernel_execution_policy.py::sandboxed_popen",
    "engine/kernel_mcp_driver.py::converse",
    "engine/kernel_probe.py::_run_child",
    "engine/kernel_public_probe_execution.py::execute_probe",
    "engine/kernel_test_isolation.py::check_sandbox_exec_denial",
    "engine/protected_tree_guard.py::main",
}


def _scan() -> tuple[set[str], set[str]]:
    flag_sites, builder_users = set(), set()
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scope: list[str] = []

        def visit(node: ast.AST) -> None:
            named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if named:
                scope.append(node.name)
            where = f"{relative}::{'.'.join(scope)}"
            if isinstance(node, ast.Constant) and node.value == FLAG:
                flag_sites.add(where)
            reference = (node.id if isinstance(node, ast.Name) else
                         node.attr if isinstance(node, ast.Attribute) else None)
            if reference in BUILDERS and scope and not (
                    named and node.name in BUILDERS):
                if not (len(scope) == 1 and scope[0] in BUILDERS):
                    builder_users.add(where)
            for child in ast.iter_child_nodes(node):
                visit(child)
            if named:
                scope.pop()

        visit(tree)
    return flag_sites, builder_users


def test_parent_death_flag_is_set_only_by_reviewed_builders():
    flag_sites, _users = _scan()
    assert flag_sites == FLAG_SITES


def test_parent_death_builders_are_used_only_by_reviewed_launch_sites():
    _sites, builder_users = _scan()
    assert builder_users == BUILDER_USERS


def test_guard_sees_a_planted_new_launch_site(tmp_path, monkeypatch):
    planted = tmp_path / "stateframe" / "engine"
    planted.mkdir(parents=True)
    (planted / "planted.py").write_text(
        "import subprocess, threading\n"
        "def launch():\n"
        "    threading.Thread(target=lambda: subprocess.Popen(['bwrap', '--die-with-parent', 'true'])).start()\n")
    monkeypatch.setattr(__import__(__name__), "ROOT", tmp_path / "stateframe")
    flag_sites, _users = _scan()
    assert flag_sites == {"engine/planted.py::launch"}


def test_exec_session_refuses_to_start_off_the_main_thread(tmp_path):
    import threading

    from kernel_exec_capability import shell_capability
    from kernel_exec_session import ExecSession, ExecSessionError

    errors: list[BaseException] = []
    workspace = tmp_path / "work"
    workspace.mkdir()
    environment = tmp_path / "env"
    environment.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "session").mkdir()

    def start():
        session = ExecSession(tmp_path / "session")
        try:
            session.run(["true"], shell_capability(workspace=workspace, run_env_dir=environment,
                                                   scratch_home=home, network="none", writable=False))
        except ExecSessionError as error:
            errors.append(error)
        finally:
            session.close()

    thread = threading.Thread(target=start)
    thread.start()
    thread.join(timeout=60)
    assert errors and "main thread" in str(errors[0])
