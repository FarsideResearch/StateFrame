"""Keep every group signal tied to a launch identity or a reviewed exception."""

import ast
import sys
from collections import Counter
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src"

# Exact expressions are pinned so a new target or signal needs a fresh review.
REVIEWED_SIGNALS = {
    ("stateframe/_child.py", "os.killpg(os.getpgrp(), signal.SIGKILL)"):
        "The child signals its own group only when it is that group's leader.",
    ("stateframe/engine/kernel_checkpoint.py", "os.kill(pid, 0)"):
        "Signal zero is a liveness query and delivers no signal.",
    ("stateframe/engine/kernel_exec_supervisor.py", "os.kill(row.pid, signal.SIGKILL)"):
        "The row is this supervisor's direct child and only its serve loop reaps it.",
    ("stateframe/engine/kernel_exec_tracker.py", "os.killpg(launch_identity.pid, sig)"):
        "The captured leader is held unreaped under its Popen wait lock.",
    ("stateframe/engine/kernel_exec_tracker.py", "os.kill(expected.pid, sig)"):
        "macOS fallback only when proc_signal_with_audittoken is missing or unsupported: the "
        "full flavor-18 identity (pid, uid, start, status, PID generation) is re-read and must "
        "match immediately before the call; the residual PID-reuse window is documented in "
        "KNOWN_LIMITATIONS.",
    ("stateframe/engine/kernel_hook_driver.py", "os.killpg(process.pid, signal.SIGKILL)"):
        "The driver signals its own unreaped Popen child (session leader).",
    ("stateframe/outer_checks.py", "process.send_signal(signal.SIGTERM)"):
        "The helper is this parent's unreaped Popen child.",
}

REVIEWED_INDIRECT = {
    ("stateframe/engine/corpus_of_record.py", "getattr(os, name)"):
        "name iterates only stat, lstat, scandir and listdir in this block.",
}


def _non_stdlib_imports(source: str) -> list[tuple[int, str]]:
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports = (node.module or "",) if node.level == 0 else ("relative import",)
        else:
            continue
        for name in imports:
            if name.split(".", 1)[0] not in sys.stdlib_module_names | {"__future__"}:
                violations.append((node.lineno, name))
    return violations


def test_hook_driver_uses_only_stdlib_imports():
    driver = SOURCE / "stateframe/engine/kernel_hook_driver.py"
    assert _non_stdlib_imports(driver.read_text(encoding="utf-8")) == []
    planted = "import json, kernel_exec_tracker\nfrom stateframe import hooks\n"
    assert _non_stdlib_imports(planted) == [(1, "kernel_exec_tracker"),
                                            (2, "stateframe")]


def _calls(source: str, relative: str) -> tuple[int, list[tuple[int, str]]]:
    checked = 0
    violations = []
    tree = ast.parse(source)
    os_names = {"os"}
    signal_names = set()
    process_signal_names = set()
    unit_names = {"kill_launch_unit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_names.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "os":
                signal_names.update(alias.asname or alias.name for alias in node.names
                                    if alias.name in {"kill", "killpg"})
            elif node.module and node.module.endswith("kernel_exec_tracker"):
                unit_names.update(alias.asname or alias.name for alias in node.names
                                  if alias.name == "kill_launch_unit")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target = node.value
            if isinstance(target, ast.Name) and target.id in os_names:
                os_names.update(item.id for item in node.targets if isinstance(item, ast.Name))
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id in os_names and target.attr in {"kill", "killpg"}):
                signal_names.update(item.id for item in node.targets if isinstance(item, ast.Name))
            if isinstance(target, ast.Name) and target.id in signal_names | unit_names:
                signal_names.update(item.id for item in node.targets if isinstance(item, ast.Name))
            if isinstance(target, ast.Attribute) and target.attr == "send_signal":
                process_signal_names.update(item.id for item in node.targets
                                            if isinstance(item, ast.Name))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if (len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "send_signal"):
                checked += 1
                violations.append((node.lineno, ast.unparse(node)))
            if (len(node.args) >= 2 and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in os_names
                    and (not isinstance(node.args[1], ast.Constant)
                         or node.args[1].value in {"kill", "killpg"})):
                checked += 1
                expression = ast.unparse(node)
                if (relative, expression) not in REVIEWED_INDIRECT:
                    violations.append((node.lineno, expression))
        if isinstance(node, ast.Subscript):
            value = node.value
            indirect = (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == "vars" and len(value.args) == 1
                        and isinstance(value.args[0], ast.Name) and value.args[0].id in os_names)
            indirect |= (isinstance(value, ast.Attribute) and value.attr == "__dict__"
                         and isinstance(value.value, ast.Name) and value.value.id in os_names)
            if indirect:
                checked += 1
                violations.append((node.lineno, ast.unparse(node)))
        if not isinstance(node, ast.Call):
            continue
        if ((isinstance(node.func, ast.Name) and node.func.id in unit_names)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "kill_launch_unit")):
            checked += 1
            if not any(keyword.arg == "launch_identity" for keyword in node.keywords):
                violations.append((node.lineno, "kill_launch_unit lacks launch_identity"))
        is_os_signal = (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in os_names
                        and node.func.attr in {"kill", "killpg"})
        is_alias = (isinstance(node.func, ast.Name)
                    and node.func.id in signal_names | process_signal_names)
        is_popen_signal = isinstance(node.func, ast.Attribute) and node.func.attr == "send_signal"
        if is_os_signal or is_alias or is_popen_signal:
            checked += 1
            expression = ast.unparse(node)
            if (relative, expression) not in REVIEWED_SIGNALS:
                violations.append((node.lineno, expression))
    return checked, violations


def test_every_production_signal_has_reviewed_authority():
    total = 0
    found = Counter()
    violations = []
    for path in SOURCE.rglob("*.py"):
        relative = path.relative_to(SOURCE).as_posix()
        source = path.read_text(encoding="utf-8")
        examined, errors = _calls(source, relative)
        total += examined
        violations.extend((relative, line, detail) for line, detail in errors)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                key = (relative, ast.unparse(node))
                if key in REVIEWED_SIGNALS:
                    found[key] += 1
    assert total > 0
    assert not violations, violations
    expected = Counter({key: 2 if key[0] == "stateframe/outer_checks.py" else 1
                        for key in REVIEWED_SIGNALS})
    assert found == expected, found


def test_signal_guard_rejects_planted_calls():
    planted = "def stop(process):\n    os.killpg(process.pid, 9)\n    kill_launch_unit(process)\n"
    examined, violations = _calls(planted, "stateframe/engine/planted.py")
    assert examined == 2
    assert violations == [(2, "os.killpg(process.pid, 9)"),
                          (3, "kill_launch_unit lacks launch_identity")]


def test_signal_guard_rejects_each_indirect_form():
    planted = (
        "import os as operating\n"
        "from os import killpg as group_signal\n"
        "k = operating.killpg\n"
        "group_signal(12, 9)\n"
        "k(12, 9)\n"
        "getattr(operating, 'killpg')(12, 9)\n"
        "vars(operating)['killpg'](12, 9)\n"
        "operating.__dict__['killpg'](12, 9)\n"
        "other_process.send_signal(9)\n"
        "getattr(operating, dynamic_name)(12, 9)\n"
        "vars(operating)[dynamic_name](12, 9)\n"
        "operating.__dict__[dynamic_name](12, 9)\n"
        "tracker.kill_launch_unit(other_process)\n"
    )
    examined, violations = _calls(planted, "stateframe/engine/planted.py")
    assert examined >= 10
    assert {line for line, _ in violations} == {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}


def test_signal_guard_rejects_module_alias_and_indirect_process_signal():
    planted = ("import os\n"
               "ops = os\n"
               "ops.killpg(19, 9)\n"
               "getattr(process, 'send_signal')(9)\n")
    examined, violations = _calls(planted, "stateframe/engine/planted.py")
    assert examined == 2
    assert {line for line, _ in violations} == {3, 4}


def test_signal_guard_rejects_bound_method_alias():
    planted = "def stop(process):\n    send = process.send_signal\n    send(9)\n"
    examined, violations = _calls(planted, "stateframe/engine/planted.py")
    assert examined == 1
    assert violations == [(3, "send(9)")]


def _real_outer_check_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    tests = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "test_outer_check_reports_normal_completion_and_cleanup"]
    if len(tests) != 1:
        return ["real outer-check completion test is missing"]
    calls = [node for node in ast.walk(tests[0]) if isinstance(node, ast.Call)]
    errors = []
    if not any(isinstance(call.func, ast.Attribute)
               and isinstance(call.func.value, ast.Name)
               and call.func.value.id == "outer_checks" and call.func.attr == "run"
               for call in calls):
        errors.append("test does not run the real outer-check helper")
    if any(isinstance(node, ast.Name) and node.id in {"monkeypatch", "mock", "patch"}
           for node in ast.walk(tree)):
        errors.append("test patches a helper or cleanup boundary")
    return errors


def test_real_outer_check_regression_has_no_mocked_cleanup_boundary():
    path = Path(__file__).with_name("test_outer_check_real_cleanup.py")
    assert path.is_file()
    assert _real_outer_check_violations(path.read_text(encoding="utf-8")) == []

    planted = "def test_outer_check_reports_normal_completion_and_cleanup():\n" \
              "    monkeypatch.setattr(policy, '_kill_group', lambda process: True)\n"
    assert _real_outer_check_violations(planted) == [
        "test does not run the real outer-check helper",
        "test patches a helper or cleanup boundary",
    ]
