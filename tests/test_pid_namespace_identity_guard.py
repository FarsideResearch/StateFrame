"""Class guard: a PID namespace is never identified by its /proc inode alone.

A namespace inode is reused as soon as the namespace dies, so comparing the readlink text of
/proc/<pid>/ns/pid is only sound while the comparing code holds that namespace open. Every
function that reads or compares namespace identity must be reviewed for a pinned handle.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "stateframe"

# Each site either defines the reader or holds an open namespace handle while comparing.
REVIEWED = {
    # Readers: they return identity; callers must compare only while it is pinned.
    "engine/kernel_exec_tracker.py::ProcessTable.pid_namespace",
    "engine/kernel_exec_tracker.py::ProcessTable.pid_namespace_inode",
    "engine/kernel_exec_tracker.py::_open_pid_namespace",  # opens the pinning handle
    # Compare against Descendants.private_pid_namespace_inode, pinned by private_pid_namespace_pin.
    "engine/kernel_exec_tracker.py::Descendants.__init__",
    "engine/kernel_exec_tracker.py::Descendants.update",
    "engine/kernel_exec_tracker.py::ProcessTable.refresh_members",
    "engine/kernel_exec_supervisor.py::Supervisor._sample_rows",
    "engine/kernel_execution_policy.py::_ResidentMemorySupervisor._rows_allowed_after_exit",
    "engine/kernel_execution_policy.py::_ResidentMemorySupervisor._run",
    # Host namespace (held by this process) and the server namespace (held by namespace_handle).
    "engine/kernel_mcp_driver.py::_ContainmentMonitor.__init__",
    "engine/kernel_mcp_driver.py::_ContainmentMonitor._sample",
    # Mutation launch identity: the namespace is one fact beside PID, start tick, boot id and
    # argv digest, never a membership test; a reused inode alone cannot match a record.
    "engine/kernel_exec_supervisor.py::Supervisor.start",
    "engine/kernel_exec_supervisor.py::Supervisor._wait_for_final_exec",
    "engine/kernel_mutate_exec.py::MutationExecutor._write_owner",
    "engine/kernel_exec_session.py::_host_mutation_identity.matching",
    # Re-verified after the pidfd pins the recorded task, with start tick and argv digest.
    "engine/kernel_mutate_sweep.py::_kill_recorded",
    # Diagnostic text only; never compared.
    "engine/kernel_exec_supervisor.py::ManagedProcess.tick",
}


def _sites(root: Path) -> set[str]:
    found = set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        scope: list[str] = []

        def visit(node):
            named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if named:
                scope.append(node.name)
            names = {"pid_namespace", "pid_namespace_inode", "private_pid_namespace_inode"}
            hit = (isinstance(node, ast.Attribute) and node.attr in names
                   or isinstance(node, ast.Constant) and isinstance(node.value, str)
                   and ("/ns/pid" in node.value or node.value in names))
            if hit and scope:
                found.add(f"{relative}::{'.'.join(scope)}")
            for child in ast.iter_child_nodes(node):
                visit(child)
            if named:
                scope.pop()

        visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def test_namespace_identity_is_read_only_at_reviewed_pinned_sites():
    assert _sites(ROOT) == REVIEWED


def test_guard_sees_a_planted_inode_comparison(tmp_path):
    planted = tmp_path / "engine"
    planted.mkdir()
    (planted / "planted.py").write_text(
        "import os\n"
        "def same(a, b):\n"
        "    return os.readlink(f'/proc/{a}/ns/pid') == os.readlink(f'/proc/{b}/ns/pid')\n")
    assert _sites(tmp_path) == {"engine/planted.py::same"}
