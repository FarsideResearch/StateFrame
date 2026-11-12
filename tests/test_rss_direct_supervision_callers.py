"""Direct sandboxed launches must verify RSS supervision before reporting completion."""
from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_environment_state as environment
import kernel_exec_tracker as tracker
import kernel_execution_policy as policy
import kernel_job_watch as jobs
import kernel_probe as probe
import kernel_public_probe_execution as public_probe


class _Supervisor:
    def __init__(self, fail: bool):
        self.fail = fail
        self.error = None
        self.finish_error = (
            "resident-memory supervisor forced post-exit verification failure" if fail else None)
        self._error_cause = RuntimeError("forced supervisor error") if fail else None
        self.reap_notifications = 0
        self.finish_calls = 0
        self._leader_reaped = False

    def notify_leader_reaped(self):
        if not self._leader_reaped:
            self.reap_notifications += 1
            self._leader_reaped = True

    def finish(self, timeout=3.0):
        self.finish_calls += 1
        return not self.fail


def _assert_chained_containment_error(error):
    assert isinstance(error.value.__cause__, RuntimeError)


class _Process:
    pid = 41218
    returncode = 0
    stdout = None

    def __init__(self, fail: bool):
        self._stateframe_memory_supervisor = _Supervisor(fail)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.mark.parametrize("fail", [True, False], ids=["verification-fails", "clean"])
def test_environment_state_fails_closed_on_supervision_failure(monkeypatch, tmp_path, fail):
    process = _Process(fail)
    monkeypatch.setattr(environment, "sandboxed_popen", lambda *_args, **_kwargs: (process, None))
    owner = environment.EnvironmentState(tmp_path)

    launched = owner.launch([sys.executable, "-c", "pass"])

    if fail:
        with pytest.raises(tracker.ContainmentUnverified, match="resident-memory supervisor") as error:
            owner.refresh()
        _assert_chained_containment_error(error)
        assert process._stateframe_memory_supervisor.reap_notifications == 1
    else:
        assert owner.refresh()[0].state == "exited"
        assert process._stateframe_memory_supervisor.finish_calls == 1
        assert process._stateframe_memory_supervisor.reap_notifications == 1
    assert launched.state == "running"


@pytest.mark.parametrize("fail", [True, False], ids=["verification-fails", "clean"])
def test_job_watch_fails_closed_on_supervision_failure(monkeypatch, tmp_path, fail):
    process = _Process(fail)
    monkeypatch.setattr(policy, "sandboxed_popen", lambda *_args, **_kwargs: (process, None))
    monkeypatch.setattr(jobs.threading.Thread, "start", lambda _thread: None)
    owner = jobs.JobWatch(tmp_path)

    launched = owner.launch_job("python -c pass", 1, "DONE")
    owner._reader_loop(launched["job_id"], process)

    if fail:
        with pytest.raises(tracker.ContainmentUnverified, match="resident-memory supervisor") as error:
            owner.job_status(launched["job_id"])
        _assert_chained_containment_error(error)
        assert process._stateframe_memory_supervisor.reap_notifications == 1
    else:
        assert owner.job_status(launched["job_id"])["state"] == "completed"
        assert process._stateframe_memory_supervisor.finish_calls == 1
        assert process._stateframe_memory_supervisor.reap_notifications == 1


@pytest.mark.parametrize("fail", [True, False], ids=["verification-fails", "clean"])
def test_probe_child_fails_closed_on_supervision_failure(monkeypatch, tmp_path, fail):
    process = _Process(fail)
    monkeypatch.setattr(policy, "sandboxed_popen", lambda *_args, **_kwargs: (process, None))
    monkeypatch.setattr(probe, "_collect_process", lambda *_args: (b"ok", b"", False, False, False))
    monkeypatch.setattr(probe, "_read_policy_metadata", lambda _fd: ())

    if fail:
        with pytest.raises(tracker.ContainmentUnverified, match="resident-memory supervisor") as error:
            probe._run_child(
                tmp_path, "pass", executable=sys.executable, wall_seconds=1,
                cpu_seconds=1, memory_bytes=64 << 20)
        _assert_chained_containment_error(error)
        assert process._stateframe_memory_supervisor.reap_notifications == 1
    else:
        result = probe._run_child(
            tmp_path, "pass", executable=sys.executable, wall_seconds=1,
            cpu_seconds=1, memory_bytes=64 << 20)
        assert result.returncode == 0
        assert process._stateframe_memory_supervisor.finish_calls == 1
        assert process._stateframe_memory_supervisor.reap_notifications == 1


@pytest.mark.parametrize("fail", [True, False], ids=["verification-fails", "clean"])
def test_public_probe_context_fails_closed_on_supervision_failure(
        monkeypatch, tmp_path, fail):
    process = _Process(fail)
    root = tmp_path / "workspace"
    output = tmp_path / "output"
    root.mkdir()
    output.mkdir()
    product = root / "module.py"
    product.write_text("value = 1\n", encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(product.read_bytes()).hexdigest()
    monkeypatch.setattr(public_probe, "locked_test_process", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(public_probe, "child_exited_unreaped", lambda _process: True)
    monkeypatch.setattr(public_probe, "kill_launch_unit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(public_probe, "recorded_launch_identity", lambda _process: object())
    monkeypatch.setattr(policy, "sandboxed_popen", lambda *_args, **_kwargs: (process, None))

    if fail:
        with pytest.raises(tracker.ContainmentUnverified, match="resident-memory supervisor") as error:
            public_probe.execute_probe(
                root, {"module.py": digest}, "def test_ok(): pass\n", output)
        _assert_chained_containment_error(error)
        assert process._stateframe_memory_supervisor.reap_notifications == 1
    else:
        result = public_probe.execute_probe(
            root, {"module.py": digest}, "def test_ok(): pass\n", output)
        assert result["exit_code"] == 0
        assert process._stateframe_memory_supervisor.finish_calls == 1
        assert process._stateframe_memory_supervisor.reap_notifications == 1


def test_finish_supervision_is_noop_without_rss_supervisor():
    policy.finish_supervision(SimpleNamespace(returncode=None))


def test_finish_supervision_verifies_attached_supervisor_once():
    process = _Process(False)

    policy.finish_supervision(process)
    policy.finish_supervision(process)

    assert process._stateframe_memory_supervisor.finish_calls == 1
    assert process._stateframe_memory_supervisor.reap_notifications == 1


def test_finish_supervision_fails_on_a_recorded_error_even_if_finish_returns_true():
    process = _Process(False)
    supervisor = process._stateframe_memory_supervisor
    supervisor.error = "recorded RSS scan error"
    supervisor._error_cause = RuntimeError("recorded cause")

    with pytest.raises(tracker.ContainmentUnverified, match="recorded RSS scan error") as error:
        policy.finish_supervision(process)

    _assert_chained_containment_error(error)
    assert supervisor.finish_calls == 1


def test_finish_supervision_does_not_retry_failed_completion():
    process = _Process(True)

    for _ in range(2):
        with pytest.raises(tracker.ContainmentUnverified):
            policy.finish_supervision(process)

    assert process._stateframe_memory_supervisor.finish_calls == 1
    assert process._stateframe_memory_supervisor.reap_notifications == 1


def _function_map(tree: ast.Module) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)
    return functions


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent
        parent = parents.get(parent)
    return None


def _under_supervised_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.With, ast.AsyncWith)):
            if any(
                    isinstance(candidate, ast.Call)
                    and _call_name(candidate) == "supervised_process"
                    for item in parent.items
                    for candidate in ast.walk(item.context_expr)):
                return True
        parent = parents.get(parent)
    return False


def _function_calls(function, expected: str) -> bool:
    return any(isinstance(node, ast.Call) and _call_name(node) == expected
               for node in ast.walk(function))


def _function_call_count(function, expected: str) -> int:
    return sum(isinstance(node, ast.Call) and _call_name(node) == expected
               for node in ast.walk(function))


def test_every_engine_sandboxed_popen_call_has_completion_coverage():
    engine = Path(__file__).parents[1] / "src/stateframe/engine"
    async_callers = {
        "kernel_environment_state.py": ("launch", "_reconcile_one"),
        "kernel_job_watch.py": ("launch_job", "_reconcile"),
    }
    same_function_callers = {
        "kernel_probe.py": "_run_child",
    }
    policy_path = engine / "kernel_execution_policy.py"
    policy_tree = ast.parse(policy_path.read_text(encoding="utf-8"), filename=str(policy_path))
    wrappers = [node for node in ast.walk(policy_tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "supervised_process"]
    assert len(wrappers) == 1 and _function_calls(wrappers[0], "finish_supervision")
    violations = []
    for path in sorted(engine.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "sandboxed_popen" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        functions = _function_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "sandboxed_popen":
                continue
            caller = _enclosing_function(node, parents)
            if path.name == "kernel_execution_policy.py" and caller is not None \
                    and caller.name == "run_sandboxed" \
                    and _function_call_count(caller, "sandboxed_popen") == 1 \
                    and _function_calls(caller, "finish_supervision"):
                continue
            if _under_supervised_context(node, parents) and _function_call_count(
                    wrappers[0], "finish_supervision") == 1:
                continue
            if caller is not None and path.name in same_function_callers \
                    and caller.name == same_function_callers[path.name] \
                    and _function_call_count(caller, "sandboxed_popen") == 1 \
                    and _function_calls(caller, "finish_supervision"):
                continue
            async_spec = async_callers.get(path.name)
            completion_functions = functions.get(async_spec[1], []) if async_spec else []
            if (async_spec is not None and caller is not None and caller.name == async_spec[0]
                    and _function_call_count(caller, "sandboxed_popen") == 1
                    and len(completion_functions) == 1
                    and _function_calls(completion_functions[0], "finish_supervision")):
                continue
            violations.append(f"{path.relative_to(engine)}:{node.lineno}")

    assert violations == []
