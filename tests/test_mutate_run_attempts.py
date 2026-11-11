import json
import sys
import types
from dataclasses import dataclass

import pytest

from kernel_mutate_build import BuildError, _run, stage_unit


def _manifest():
    return {
        "schema": "stateframe-mutation-unit/1",
        "name": "x",
        "revision": 1,
        "parent": None,
        "entry": "main.py",
        "summary": "x",
        "kind": "reader",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "requires": {"network": False, "writes": False},
        "smoke": [{"input": {}, "expect": {}}],
        "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256},
    }


def _install_executor_types(monkeypatch):
    @dataclass
    class MutationLimits:
        wall_s: float
        cpu_s: int
        mem_mb: int

    @dataclass
    class LaunchSpec:
        role: str
        call: str
        launch: str
        unit_dir: object
        inputs: dict
        responses: dict
        stdin_doc: bytes
        limits: MutationLimits

    module = types.ModuleType("kernel_mutate_exec")
    module.MutationLimits = MutationLimits
    module.LaunchSpec = LaunchSpec
    monkeypatch.setitem(sys.modules, "kernel_mutate_exec", module)


def _success(spec):
    result = {"ok": True}
    body = json.dumps({"wire": 1, "call": spec.call, "kind": "result",
                       "result": result}).encode()
    return types.SimpleNamespace(
        stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
        stdout_overflow=False,
        timed_out=False,
        exit_code=0,
        cleanup=types.SimpleNamespace(ok=True),
    )


def _timeout(_spec):
    return types.SimpleNamespace(
        stdout=b"",
        stdout_overflow=False,
        timed_out=True,
        exit_code=None,
        cleanup=types.SimpleNamespace(ok=True),
    )


def _failure(_spec):
    return types.SimpleNamespace(
        stdout=b"",
        stdout_overflow=False,
        timed_out=False,
        exit_code=137,
        cleanup=types.SimpleNamespace(ok=True),
    )


class SequenceExecutor:
    def __init__(self, *outcomes):
        self.outcomes = outcomes
        self.specs = []

    def run(self, spec):
        self.specs.append(spec)
        outcome = self.outcomes[min(len(self.specs) - 1, len(self.outcomes) - 1)]
        return outcome(spec)


def _unit(tmp_path):
    return stage_unit(
        {"main.py": b"def handle(input, ctx): return {}\n"}, _manifest(), tmp_path
    )


def test_run_success_executes_once_with_timeout_retry_enabled(tmp_path, monkeypatch):
    _install_executor_types(monkeypatch)
    executor = SequenceExecutor(_success)

    launch, result = _run(_unit(tmp_path), executor, "validator", {}, timeout_retries=1)

    assert result == {"ok": True}
    assert launch == executor.specs[0].launch
    assert len(executor.specs) == 1


def test_run_genuine_failure_executes_once_and_is_not_retried(tmp_path, monkeypatch):
    _install_executor_types(monkeypatch)
    executor = SequenceExecutor(_failure)

    with pytest.raises(BuildError) as caught:
        _run(_unit(tmp_path), executor, "validator", {}, timeout_retries=1)

    assert caught.value.code == "EXECUTION"
    assert len(executor.specs) == 1


def test_run_retries_timeout_then_succeeds_on_fresh_launch(tmp_path, monkeypatch):
    _install_executor_types(monkeypatch)
    executor = SequenceExecutor(_timeout, _success)

    launch, result = _run(_unit(tmp_path), executor, "validator", {}, timeout_retries=1)

    assert result == {"ok": True}
    assert len(executor.specs) == 2
    assert executor.specs[0].launch != executor.specs[1].launch
    assert launch == executor.specs[1].launch


def test_run_two_timeouts_raise_named_timeout_reason(tmp_path, monkeypatch):
    _install_executor_types(monkeypatch)
    executor = SequenceExecutor(_timeout, _timeout)

    with pytest.raises(BuildError) as caught:
        _run(_unit(tmp_path), executor, "validator", {}, timeout_retries=1)

    assert caught.value.code == "TIMEOUT"
    assert "validator: timed out" in caught.value.detail
    assert len(executor.specs) == 2
