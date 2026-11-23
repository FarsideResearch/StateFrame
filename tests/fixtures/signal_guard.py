"""Review every source signal site and confine test signals to one helper."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re


# Each key is (source path, enclosing function, exact call). Counts detect additions
# even where an identical call is introduced inside a previously reviewed function.
# Files outside this lane are listed here for visibility; their code is not changed.
APPROVED_SOURCE_CALLS: dict[tuple[str, str, str], tuple[int, str]] = {
    ("src/stateframe/_child.py", "loop", "os.killpg(os.getpgrp(), signal.SIGKILL)"):
        (1, "child suicide on lost parent"),
    ("src/stateframe/engine/capability_host_capacity.py", "captured_process_group", "os.getpgid(pid)"):
        (1, "inspect an admitted command process"),
    ("src/stateframe/engine/kernel_check_watchdog.py", "process_snapshot", "os.kill(pid, 0)"):
        (1, "signal zero checks absence after an incomplete identity read without delivering a signal"),
    ("src/stateframe/engine/kernel_checkpoint.py", "_pid_alive", "os.kill(pid, 0)"):
        (1, "probe a checkpoint process identity"),
    ("src/stateframe/engine/kernel_env_seed.py", "_evict_to_cap", "os.kill(pid, 0)"):
        (1, "signal zero checks whether a seed temp directory's owner is alive; it delivers no signal"),
    ("src/stateframe/engine/kernel_exec_supervisor.py", "serve", "os.kill(row.pid, signal.SIGKILL)"):
        (1, "reap a tracked execution child"),
    ("src/stateframe/engine/kernel_exec_tracker.py", "signal", "signal.pidfd_send_signal(pidfd, sig)"):
        (1, "signal a descendant pinned by pidfd and checked against its recorded birth"),
    ("src/stateframe/engine/kernel_exec_tracker.py", "signal_after_recheck", "os.kill(expected.pid, sig)"):
        (1, "macOS fallback when the generation-checked signal is missing: full identity re-read first"),
    ("src/stateframe/engine/kernel_exec_tracker.py", "signal_launch_group", "os.getpgid(launch_identity.pid)"):
        (1, "inspect the recorded unreaped leader under its Popen wait lock"),
    ("src/stateframe/engine/kernel_exec_tracker.py", "signal_launch_group", "os.killpg(launch_identity.pid, sig)"):
        (1, "signal the recorded group while its leader is unreaped under the wait lock"),
    ("src/stateframe/engine/kernel_exec_tracker.py", "signal_launch_group", "signal.pidfd_send_signal(pidfd, 0)"):
        (1, "probe the pinned Linux leader without delivering a signal"),
    ("src/stateframe/engine/kernel_hook_driver.py", "_stop_group", "os.killpg(process.pid, signal.SIGKILL)"):
        (1, "signal the hook driver's own unreaped session leader"),
    ("src/stateframe/parallel_host.py", "_signal_one", "signal.pidfd_send_signal(descriptor, sig)"):
        (1, "signal one pinned Linux process after exact kernel birth verification"),
    ("src/stateframe/parallel_worker.py", "_abort_failed_launch", "os.kill(process.pid, signal.SIGKILL)"):
        (1, "signal the new unreaped direct child before reporting missing group identity"),
    ("tests/engine_run.py", "_group_exists", "os.killpg(process_group_id, 0)"):
        (1, "probe the test monitor's own child group without delivering a signal"),
    ("tests/engine_run.py", "_stop_group", "os.killpg(self.process_group_id, signal.SIGKILL)"):
        (2, "reap the isolated engine process group owned by the test monitor"),
    ("tests/engine_run.py", "_stop_group", "os.killpg(self.process_group_id, signal.SIGTERM)"):
        (1, "terminate the isolated engine process group owned by the test monitor"),
    ("tests/acceptance/test_checkpoint_write_safety.py", "<text>", "embedded process signal reference"):
        (1, "checkpoint fault injection only signals its own process"),
    ("tests/test_hook_driver.py", "<text>", "embedded process signal reference"):
        (1, "hook command tests self-signal reporting inside its own child"),
    ("tests/test_descendant_signal_platforms.py", "<text>", "embedded process signal reference"):
        (1, "monkeypatch target string asserting the kernel signal path never calls os.kill"),
    ("tests/test_launch_unit_kill.py", "<text>", "embedded process signal reference"):
        (1, "structural control contains an unexecuted direct-signal example"),
    ("tests/test_cleanup_containment.py", "_dispose", "os.kill(pid, signal.SIGKILL)"):
        (1, "test teardown stops its own observed descendant after an identity check"),
    ("tests/test_cleanup_containment.py", "test_outer_helper_loss_reports_escaped_check_as_unverified", "os.kill(int(pid_file.read_text()), signal.SIGKILL)"):
        (1, "test tears down the process it launched after its containment assertion"),
    ("tests/test_cleanup_escalation.py", "_dispose", "os.kill(child, signal.SIGKILL)"):
        (1, "test teardown stops its own observed child"),
    ("tests/test_launch_identity_signals.py", "test_exited_leader_descendant_is_stopped_before_reap", "os.kill(child, signal.SIGKILL)"):
        (1, "test finally block cleans up its own child"),
    ("tests/test_process_signal_authority.py", "<text>", "embedded process signal reference"):
        (1, "planted source text verifies the production signal guard"),
    ("tests/test_checkpoint_kill_barrier.py", "<text>", "embedded process signal reference"):
        (1, "planted direct signal is a negative example for the checkpoint-barrier audit"),
    ("tests/test_engine_run.py",
     "test_stalled_child_reports_progress_and_kills_its_process_group",
     "os.killpg(error.process_group_id, 0)"):
        (1, "probe the monitored test process group without delivering a signal"),
}


def _calls(path: Path) -> Counter[tuple[str, str, str]]:
    source = path.read_text(encoding="utf-8")
    if "kill" not in source and "getpgid" not in source and "pidfd_send_signal" not in source:
        return Counter()
    tree = ast.parse(source, filename=str(path))
    found: Counter[tuple[str, str, str]] = Counter()
    os_names = {"os", *(alias.asname or alias.name for node in ast.walk(tree)
                         if isinstance(node, ast.Import) for alias in node.names if alias.name == "os")}
    signal_names = {"signal", *(alias.asname or alias.name for node in ast.walk(tree)
                              if isinstance(node, ast.Import) for alias in node.names if alias.name == "signal")}
    importlib_names = {"importlib", *(alias.asname or alias.name for node in ast.walk(tree)
                                    if isinstance(node, ast.Import) for alias in node.names
                                    if alias.name == "importlib")}
    import_module_names = {alias.asname or alias.name for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom) and node.module == "importlib"
                           for alias in node.names if alias.name == "import_module"}
    protected = {"os": {"kill", "killpg", "getpgid"}, "signal": {"pidfd_send_signal"}}
    direct_names = {alias.asname or alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module in protected
                    for alias in node.names if alias.name in protected[node.module]}
    module_aliases: dict[str, str] = {}

    def module_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in module_aliases:
                return module_aliases[node.id]
            if node.id in os_names:
                return "os"
            if node.id in signal_names:
                return "signal"
        if (isinstance(node, ast.Call) and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in protected):
            if (isinstance(node.func, ast.Name) and node.func.id == "__import__"):
                return node.args[0].value
            if isinstance(node.func, ast.Name) and node.func.id in import_module_names:
                return node.args[0].value
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in importlib_names):
                return node.args[0].value
        return None

    def protected_target(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in direct_names
        if isinstance(node, ast.Attribute):
            module = module_name(node.value)
            return module is not None and node.attr in protected[module]
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "__dict__" and isinstance(node.slice, ast.Constant)):
            module = module_name(node.value.value)
            return module is not None and node.slice.value in protected[module]
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "vars"
                and len(node.value.args) == 1 and not node.value.keywords
                and isinstance(node.slice, ast.Constant)):
            module = module_name(node.value.args[0])
            return module is not None and node.slice.value in protected[module]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)):
            module = module_name(node.args[0])
            return module is not None and node.args[1].value in protected[module]
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            module = module_name(value) if value is not None else None
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if module is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        module_aliases[target.id] = module

    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def visit(node: ast.AST, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        if isinstance(node, ast.Call) and protected_target(node.func):
            found[(path.as_posix(), owner, ast.unparse(node))] += 1
        elif (isinstance(node, (ast.Call, ast.Subscript)) and protected_target(node)
              and not (isinstance(parents.get(id(node)), ast.Call)
                       and parents[id(node)].func is node)):
            found[(path.as_posix(), owner, ast.unparse(node))] += 1
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(tree, "<module>")
    return found


def signal_violations(root: Path, *, review_source: bool = True) -> list[str]:
    """Return every unreviewed call, including additions in an approved function."""
    found: Counter[tuple[str, str, str]] = Counter()
    for base in (root / "tests", root / "src"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            calls = _calls(path)
            if (relative.startswith("tests/") and not calls
                    and relative not in {"tests/fixtures/process_signals.py",
                                         "tests/fixtures/signal_guard.py",
                                         "tests/e2e/test_parallel_interrupt_safety.py"}
                    and re.search(r"\bos\.(?:kill|killpg|getpgid)\b", path.read_text(encoding="utf-8"))):
                found[(relative, "<text>", "embedded process signal reference")] += 1
            for (_, owner, call), count in calls.items():
                if relative.startswith("tests/") and relative != "tests/fixtures/process_signals.py":
                    found[(relative, owner, call)] += count
                elif relative.startswith("src/"):
                    found[(relative, owner, call)] += count
    expected = {key: value[0] for key, value in APPROVED_SOURCE_CALLS.items()} if review_source else {}
    violations = []
    for key in sorted(found.keys() | expected.keys()):
        if found[key] != expected.get(key, 0):
            path, owner, call = key
            violations.append(f"{path}::{owner}: {call} "
                              f"(found {found[key]}, reviewed {expected.get(key, 0)})")
    return violations
