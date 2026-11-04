"""Real-child controls for the test subprocess progress watcher."""
from __future__ import annotations

import ast
import inspect
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import engine_run
from engine_run import (
    GROUP_VERIFY_SECONDS,
    TERMINATE_GRACE_SECONDS,
    EngineRunTimeout,
    deadline_windows,
    run_monitored,
    start_monitored,
)
from fixtures.process_signals import signal_run_owned

TESTS_ROOT = Path(__file__).resolve().parent
SUBPROCESS_APIS = {"call", "check_call", "check_output", "Popen", "run"}
REAL_ENGINE_ALLOWLIST = {
    "test_edit_scope_widen.py": "pending migration after baseline-pins/capture-isolation",
    "test_final_cache_layout_host.py": "pending migration after baseline-pins/capture-isolation",
    "test_hook_wiring_parity.py": "pending migration after baseline-pins/capture-isolation",
    "tools/compare_no_mcp_requests.py": "pending migration after baseline-pins/capture-isolation",
}


def _child(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "child.py"
    path.write_text(body)
    return path


def test_slow_progress_beats_old_fixed_limit_and_preserves_result(tmp_path):
    run_dir = tmp_path / "run"
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
run_dir.mkdir(parents=True, exist_ok=True)
for index in range(8):
    with (run_dir / "sends.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(f'{index}\\n')
        stream.flush()
    print(f"progress-{index}", flush=True)
    print(f"warning-{index}", file=sys.stderr, flush=True)
    time.sleep(0.9)
print("finished", flush=True)
raise SystemExit(7)
""")
    command = [sys.executable, str(child), str(run_dir)]

    old_limit = 3
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(command, capture_output=True, text=True, timeout=old_limit, check=False)

    started = time.monotonic()
    result = run_monitored(command, expected_seconds=old_limit, run_dir=run_dir,
                           step="slow fake engine run")
    elapsed = time.monotonic() - started

    assert elapsed > old_limit
    assert result.returncode == 7
    assert "finished" in result.stdout
    assert "warning-7" in result.stderr


def test_stalled_child_reports_progress_and_kills_its_process_group(tmp_path):
    run_dir = tmp_path / "run"
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / "sends.jsonl").open("a", encoding="utf-8") as stream:
    stream.write("started\\n")
    stream.flush()
print("stalled child stdout", flush=True)
print("stalled child stderr", file=sys.stderr, flush=True)
time.sleep(60)
""")
    expected = 2
    started = time.monotonic()

    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child), str(run_dir)], expected_seconds=expected,
                      run_dir=run_dir, step="stalled fake engine run")

    elapsed = time.monotonic() - started
    error = caught.value
    assert elapsed <= error.idle_window + TERMINATE_GRACE_SECONDS + 2.0
    assert error.stage == "send 1 in flight"
    assert error.seconds_since_progress >= error.idle_window
    assert "sends.jsonl:lines=1" in error.last_progress
    assert "stalled child stderr" in error.stderr_tail
    assert "stalled child stdout" in error.stdout_tail
    assert "send 1 in flight" in str(error)
    assert "sends.jsonl:lines=1" in str(error)
    assert "stalled child stderr" in str(error)
    assert "stalled child stdout" in str(error)
    assert error.group_gone
    with pytest.raises(ProcessLookupError):
        os.getpgid(error.process_group_id)


def test_signal_owned_uses_monitored_process_group(monkeypatch, tmp_path):
    child = _child(tmp_path, "import time\ntime.sleep(60)\n")
    original_killpg = os.killpg
    group_signals = []

    def record_group_signal(process_group_id, signal_number):
        group_signals.append((process_group_id, signal_number))
        return original_killpg(process_group_id, signal_number)

    monkeypatch.setattr(engine_run.os, "killpg", record_group_signal)
    with start_monitored([sys.executable, str(child)], expected_seconds=30,
                         step="owned monitored process") as process:
        process_group_id = process.process_group_id
        # This fake child has no checkpoint journal; exercise the group path directly.
        signal_run_owned(tmp_path / "run", process, signal.SIGKILL,
                         deadline=time.monotonic(), what="monitored group signal",
                         require_checkpoint_saved=False, group=True)
        process.communicate(timeout=5)

    assert group_signals == [(process_group_id, signal.SIGKILL)]
    assert process.returncode == -signal.SIGKILL


def test_signal_run_owned_signals_only_monitored_runner(monkeypatch, tmp_path):
    child = _child(tmp_path, "import time\ntime.sleep(60)\n")
    original_kill = os.kill
    signalled = []

    def record_signal(pid, signal_number):
        signalled.append((pid, signal_number))
        return original_kill(pid, signal_number)

    monkeypatch.setattr(os, "kill", record_signal)
    with start_monitored([sys.executable, str(child)], expected_seconds=30,
                         step="owned monitored runner") as process:
        # This fake child has no checkpoint journal; verify the single-process adapter.
        signal_run_owned(tmp_path / "run", process, signal.SIGKILL,
                         deadline=time.monotonic(), what="monitored runner signal",
                         require_checkpoint_saved=False)
        process.communicate(timeout=5)

    assert signalled == [(process.pid, signal.SIGKILL)]
    assert process.returncode == -signal.SIGKILL


def test_permission_error_checking_process_group_keeps_polling(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_run, "host_load_factor", lambda: 1.0)
    run_dir = tmp_path / "run"
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / "sends.jsonl").open("a", encoding="utf-8") as stream:
    stream.write("started\\n")
    stream.flush()
print("permission-control child", flush=True)
time.sleep(60)
    """)
    original_killpg = os.killpg
    group_checks = 0

    def permission_error_for_existence_checks(process_group_id, signal_number):
        nonlocal group_checks
        if signal_number == 0:
            group_checks += 1
            raise PermissionError("process-group probe denial")
        return original_killpg(process_group_id, signal_number)

    monkeypatch.setattr(engine_run.os, "killpg", permission_error_for_existence_checks)
    started = time.monotonic()
    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child), str(run_dir)], expected_seconds=1,
                      run_dir=run_dir, step="permission-control run")

    elapsed = time.monotonic() - started
    error = caught.value
    assert group_checks >= 2
    assert not error.group_gone
    assert error.stage == "send 1 in flight"
    assert "sends.jsonl:lines=1" in error.last_progress
    assert "idle window expired" in str(error)
    assert "permission-control child" in error.stdout_tail
    assert elapsed <= error.idle_window + TERMINATE_GRACE_SECONDS + 2 * GROUP_VERIFY_SECONDS + 2.0
    with pytest.raises(ProcessLookupError):
        original_killpg(error.process_group_id, 0)


def test_permission_error_signalling_keeps_timeout_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_run, "host_load_factor", lambda: 1.0)
    run_dir = tmp_path / "run"
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / "sends.jsonl").open("a", encoding="utf-8") as stream:
    stream.write("started\\n")
    stream.flush()
print("permission-denied signal evidence", flush=True)
time.sleep(60)
""")
    original_killpg = os.killpg
    attempted_signals = []

    def deny_process_group_signals(process_group_id, signal_number):
        if signal_number in (signal.SIGTERM, signal.SIGKILL):
            attempted_signals.append(signal_number)
            raise PermissionError("planted process-group signal denial")
        return original_killpg(process_group_id, signal_number)

    monkeypatch.setattr(engine_run.os, "killpg", deny_process_group_signals)
    processes: list[engine_run.EngineProcess] = []
    started = time.monotonic()
    try:
        with pytest.raises(EngineRunTimeout) as caught:
            with start_monitored([sys.executable, str(child), str(run_dir)], expected_seconds=1,
                                 run_dir=run_dir, step="permission-denied signal run") as process:
                processes.append(process)
                process.wait()
    finally:
        if processes:
            process = processes[0]
            if process._process.poll() is None:
                try:
                    original_killpg(process.process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process._process.wait(timeout=5)

    elapsed = time.monotonic() - started
    assert attempted_signals.count(signal.SIGTERM) >= 1
    assert attempted_signals.count(signal.SIGKILL) >= 1
    caught_error = caught.value
    assert type(caught_error) is EngineRunTimeout
    assert not caught_error.group_gone
    assert caught_error.stage == "send 1 in flight"
    assert "sends.jsonl:lines=1" in caught_error.last_progress
    assert "permission-denied signal evidence" in caught_error.stdout_tail
    assert "idle window expired" in str(caught_error)
    assert "group_gone=False" in str(caught_error)
    assert elapsed <= 1 + TERMINATE_GRACE_SECONDS + 2 * GROUP_VERIFY_SECONDS + 2.0


def test_current_send_stage_takes_precedence_over_old_check_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
(run_dir / "checks").mkdir(parents=True, exist_ok=True)
(run_dir / "checks" / "previous-check.json").write_text("{}", encoding="utf-8")
with (run_dir / "sends.jsonl").open("a", encoding="utf-8") as stream:
    stream.write("send started\\n")
    stream.flush()
time.sleep(60)
""")

    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child), str(run_dir)], expected_seconds=1,
                      run_dir=run_dir, step="send after prior check")

    assert caught.value.stage == "send 1 in flight"


def test_progressing_child_is_bounded_by_fixed_hard_ceiling(tmp_path):
    run_dir = tmp_path / "run"
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
run_dir.mkdir(parents=True, exist_ok=True)
index = 0
while True:
    with (run_dir / "sends.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(f'{index}\\n')
        stream.flush()
    print(f"tick-{index}", flush=True)
    index += 1
    time.sleep(0.1)
""")
    expected = 1
    started = time.monotonic()

    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child), str(run_dir)], expected_seconds=expected,
                      run_dir=run_dir, step="ceiling fake engine run")

    elapsed = time.monotonic() - started
    error = caught.value
    assert elapsed >= error.hard_ceiling - 0.5
    assert elapsed <= error.hard_ceiling + TERMINATE_GRACE_SECONDS + 2.0
    assert "hard ceiling reached" in str(error)
    assert error.hard_ceiling >= expected * 3
    assert "sends.jsonl:lines=" in str(error)
    assert error.group_gone


def test_output_growth_alone_resets_non_run_idle_window(tmp_path):
    child = _child(tmp_path, """
import sys
import time

for index in range(4):
    print(f"install-output-{index}", flush=True)
    print(f"install-warning-{index}", file=sys.stderr, flush=True)
    time.sleep(0.8)
""")
    result = run_monitored([sys.executable, str(child)], expected_seconds=2,
                           step="pip install")

    assert result.returncode == 0
    assert "install-output-3" in result.stdout
    assert "install-warning-3" in result.stderr


def test_caller_progress_probe_resets_idle_window(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_run, "host_load_factor", lambda: 1.0)
    child = _child(tmp_path, """
from pathlib import Path
import sys
import time

marker = Path(sys.argv[1])
for index in range(4):
    marker.write_text(str(index + 1), encoding="utf-8")
    time.sleep(0.4)
""")

    no_probe_marker = tmp_path / "without-probe.txt"
    with pytest.raises(EngineRunTimeout) as no_probe_timeout:
        run_monitored([sys.executable, str(child), str(no_probe_marker)], expected_seconds=1,
                      step="silent child without probe")
    assert "progress idle window expired" in str(no_probe_timeout.value)
    assert no_probe_timeout.value.seconds_since_progress >= no_probe_timeout.value.idle_window

    marker = tmp_path / "served-count.txt"

    def served_requests():
        return marker.read_text(encoding="utf-8") if marker.exists() else "0"

    started = time.monotonic()
    result = run_monitored([sys.executable, str(child), str(marker)], expected_seconds=1,
                           step="tape child", progress_probe=served_requests)
    elapsed = time.monotonic() - started

    assert result.returncode == 0
    assert served_requests() == "4"
    assert elapsed > no_probe_timeout.value.idle_window


def test_constant_progress_probe_expires_at_idle_window_with_value_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_run, "host_load_factor", lambda: 1.0)
    child = _child(tmp_path, "import time\ntime.sleep(60)\n")
    constant_value = "probe-value-" + "x" * 300
    started = time.monotonic()

    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child)], expected_seconds=1, step="constant probe",
                      progress_probe=lambda: constant_value)

    elapsed = time.monotonic() - started
    error = caught.value
    assert error.stage == "constant probe"
    assert error.seconds_since_progress >= error.idle_window
    assert elapsed <= error.idle_window + TERMINATE_GRACE_SECONDS + 2.0
    assert error.last_probe_value == repr(constant_value)[:200]
    assert len(error.last_probe_value) == 200
    assert error.last_probe_value in str(error)
    assert error.progress_probe_error_type is None
    assert "progress idle window expired" in str(error)


def test_raising_progress_probe_is_evidence_not_a_monitor_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_run, "host_load_factor", lambda: 1.0)
    child = _child(tmp_path, "import time\ntime.sleep(60)\n")

    def raising_probe():
        raise ValueError("sensitive probe detail")

    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child)], expected_seconds=1, step="raising probe",
                      progress_probe=raising_probe)

    error = caught.value
    assert error.stage == "progress_probe"
    assert error.progress_probe_error_type == "ValueError"
    assert error.last_probe_value is None
    assert "progress_probe_error_type=ValueError" in str(error)
    assert "sensitive probe detail" not in str(error)
    assert error.group_gone


def test_changing_progress_probe_cannot_extend_hard_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_run, "host_load_factor", lambda: 1.0)
    child = _child(tmp_path, "import time\nwhile True:\n    time.sleep(0.01)\n")
    probe_calls = 0

    def changing_probe():
        nonlocal probe_calls
        probe_calls += 1
        return probe_calls

    started = time.monotonic()
    with pytest.raises(EngineRunTimeout) as caught:
        run_monitored([sys.executable, str(child)], expected_seconds=1, step="probe hard ceiling",
                      progress_probe=changing_probe)

    elapsed = time.monotonic() - started
    error = caught.value
    assert error.stage == "probe hard ceiling"
    assert error.hard_ceiling == 3.0
    assert elapsed >= error.hard_ceiling - 0.5
    assert elapsed <= error.hard_ceiling + TERMINATE_GRACE_SECONDS + 2.0
    assert error.seconds_since_progress < error.idle_window
    assert probe_calls > error.hard_ceiling
    assert "hard ceiling reached" in str(error)
    assert error.group_gone


def test_progress_probe_is_keyword_only_on_public_monitor_interfaces():
    for callable_object in (engine_run.EngineProcess.__init__, engine_run.start_monitored,
                            engine_run.run_monitored):
        parameter = inspect.signature(callable_object).parameters["progress_probe"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_deadline_windows_scale_only_from_the_measured_expectation():
    assert deadline_windows(10, 1) == (10.0, 30.0)
    assert deadline_windows(10, 2) == (20.0, 60.0)


def _subprocess_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    module_aliases = {"subprocess"}
    function_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            function_aliases.update(alias.asname or alias.name for alias in node.names
                                    if alias.name in SUBPROCESS_APIS)
    return module_aliases, function_aliases


def _scope_bindings(scope: ast.AST) -> dict[str, ast.AST]:
    body = getattr(scope, "body", ())
    pending = list(body) if isinstance(body, list) else []
    found: dict[str, ast.AST] = {}
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
            value = node.value
        else:
            targets = []
            value = None
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                found[target.id] = value
        pending.extend(ast.iter_child_nodes(node))
    return found


def _function_returns(function: ast.AST) -> list[ast.AST]:
    pending = list(getattr(function, "body", ()))
    returns = []
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            returns.append(node.value)
        pending.extend(ast.iter_child_nodes(node))
    return returns


def _command_tokens(node: ast.AST, scopes: list[ast.AST],
                    functions: dict[str, list[ast.AST]], seen: set[str] | None = None,
                    *, shell_text: bool = True) -> list[str | None]:
    seen = set() if seen is None else seen
    if isinstance(node, ast.Name):
        if node.id in seen:
            return []
        for scope in reversed(scopes):
            value = _scope_bindings(scope).get(node.id)
            if value is not None:
                return _command_tokens(value, scopes, functions, seen | {node.id},
                                       shell_text=shell_text)
        return [None]
    if isinstance(node, (ast.List, ast.Tuple)):
        tokens: list[str | None] = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                tokens.extend(_command_tokens(item.value, scopes, functions, seen, shell_text=False))
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                tokens.append(item.value)
            elif isinstance(item, ast.Name):
                tokens.extend(_command_tokens(item, scopes, functions, seen, shell_text=False))
            else:
                tokens.append(None)
        return tokens
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if not shell_text:
            return [node.value]
        try:
            return shlex.split(node.value)
        except ValueError:
            return []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (_command_tokens(node.left, scopes, functions, seen, shell_text=shell_text) +
                _command_tokens(node.right, scopes, functions, seen, shell_text=shell_text))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_key = f"function:{node.func.id}"
        if function_key in seen:
            return [None]
        expressions = [expression for function in functions.get(node.func.id, [])
                       for expression in _function_returns(function)]
        tokens: list[str | None] = []
        for expression in expressions:
            if tokens:
                tokens.append(None)
            tokens.extend(_command_tokens(expression, scopes, functions, seen | {function_key},
                                          shell_text=shell_text))
        if tokens:
            return tokens
    return [None]


def _is_subprocess_call(call: ast.Call, module_aliases: set[str],
                        function_aliases: set[str]) -> bool:
    function = call.func
    if isinstance(function, ast.Attribute):
        return (function.attr in SUBPROCESS_APIS and isinstance(function.value, ast.Name) and
                function.value.id in module_aliases)
    return isinstance(function, ast.Name) and function.id in function_aliases


def _starts_engine_run(tokens: list[str | None]) -> bool:
    for index, token in enumerate(tokens):
        if token == "-m" and tokens[index + 1:index + 3] == ["stateframe", "run"]:
            return True
        if token == "-m" and tokens[index + 1:index + 3] == ["stateframe", "resume"]:
            return True
        if token == "stateframe" and tokens[index + 1:index + 2] in (["run"], ["resume"]):
            return True
    return False


def _engine_subprocess_calls(source: str, relative_path: str) -> list[str]:
    tree = ast.parse(source, filename=relative_path)
    module_aliases, function_aliases = _subprocess_aliases(tree)
    functions: dict[str, list[ast.AST]] = {}
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(candidate.name, []).append(candidate)
    calls: list[str] = []
    scopes: list[ast.AST] = [tree]

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for decorator in node.decorator_list:
                self.visit(decorator)
            scopes.append(node)
            for statement in node.body:
                self.visit(statement)
            scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if _is_subprocess_call(node, module_aliases, function_aliases):
                argument = node.args[0] if node.args else next(
                    (keyword.value for keyword in node.keywords if keyword.arg in {"args", "command"}),
                    None,
                )
                if argument is not None and _starts_engine_run(
                    _command_tokens(argument, scopes, functions)
                ):
                    calls.append(f"{relative_path}:{node.lineno}")
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


def _guard_scan(sources: dict[str, str], allowlist: dict[str, str]):
    engine_calls: list[str] = []
    violations: list[str] = []
    for relative_path, source in sources.items():
        matches = _engine_subprocess_calls(source, relative_path)
        engine_calls.extend(matches)
        if relative_path not in allowlist:
            violations.extend(matches)
    return engine_calls, violations


def test_real_engine_cli_subprocesses_use_the_shared_helper(capsys):
    sources = {
        path.relative_to(TESTS_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in TESTS_ROOT.rglob("*.py")
    }
    engine_calls, violations = _guard_scan(sources, REAL_ENGINE_ALLOWLIST)
    scanned_files = len(sources)
    print(f"engine subprocess guard: expected=0 violations, scanned={scanned_files} files, "
          f"engine_calls={len(engine_calls)}, violated={len(violations)}")
    assert scanned_files > 0
    assert all(path and reason.strip() for path, reason in REAL_ENGINE_ALLOWLIST.items())
    assert not violations, "real engine subprocess bypasses helper: " + ", ".join(violations)
    assert "expected=0 violations" in capsys.readouterr().out


def test_guard_flags_a_planted_bare_subprocess_run():
    source = '''
import subprocess
subprocess.run(["python", "-m", "stateframe", "run", "--repo", "repo"], timeout=240)
'''
    engine_calls, violations = _guard_scan({"planted_violation.py": source}, {})

    assert len(engine_calls) == 1
    assert violations == engine_calls


def test_guard_resolves_a_local_command_factory():
    source = '''
import subprocess
def command():
    return ["python", "-m", "stateframe", "resume", "run-dir"]
subprocess.Popen(command())
'''
    engine_calls, violations = _guard_scan({"command_factory.py": source}, {})

    assert len(engine_calls) == 1
    assert violations == engine_calls
