"""The engine wall sits a finalize reserve before the run deadline; the watchdog kill does not."""
import json
import math
from types import SimpleNamespace

import pytest

from stateframe import runner

BUDGETS = [1, 5, 30, 60, 600, 1199, 1200, 1201, 2400, 5400, 86400]


@pytest.mark.parametrize("remaining", BUDGETS)
def test_engine_wall_leaves_finalize_reserve(remaining):
    now = 1000.0
    run_deadline = now + remaining
    engine_deadline, kill_at = runner._engine_wall(now, float(remaining), run_deadline)
    reserve = min(120.0, 0.1 * remaining)
    assert engine_deadline < kill_at <= run_deadline
    assert kill_at - engine_deadline == pytest.approx(reserve)
    engine_budget = engine_deadline - now
    assert engine_budget > 0
    for default in (1800.0, 600.0):  # per-request caps as drive() applies them
        assert min(default, engine_budget) <= engine_budget


def test_engine_wall_without_run_deadline_uses_remaining():
    engine_deadline, kill_at = runner._engine_wall(10.0, 100.0, None)
    assert kill_at == pytest.approx(110.0)
    assert engine_deadline == pytest.approx(100.0)


def test_no_timeout_is_infinite():
    assert runner._engine_wall(5.0, math.inf, None) == (math.inf, math.inf)


def _drive_spec(tmp_path, monkeypatch, remaining, deadline):
    class Engine:
        returncode = 0
        stdout = iter(())

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Engine())
    monkeypatch.setattr(runner, "_remaining", lambda _o: remaining)
    monkeypatch.setattr(runner, "_child_env", lambda _o: {})
    monkeypatch.setattr(runner, "engine_reads_test_python", lambda: False)
    monkeypatch.setattr(runner, "_stop_group", lambda _p: None)
    (tmp_path / "spec.json").write_text(json.dumps({"environment": {}}))
    options = SimpleNamespace(timeout_s=None if math.isinf(remaining) else remaining,
                              _deadline=deadline, python=__import__("sys").executable,
                              shell_confirm=None, quiet=True)
    try:
        runner.drive(tmp_path, options)
    except Exception:
        pass  # only the written spec matters here; the fake engine is minimal
    return json.loads((tmp_path / "spec.json").read_text())["environment"]


def test_drive_writes_engine_budget_into_spec(tmp_path, monkeypatch):
    remaining = 600.0
    env = _drive_spec(tmp_path, monkeypatch, remaining, runner.time.monotonic() + remaining)
    reserve = min(120.0, 0.1 * remaining)
    assert 0 < float(env["STATEFRAME_RUN_WALL_SECONDS"]) <= remaining - reserve + 1e-6
    assert float(env["STATEFRAME_PER_REQUEST_TIMEOUT_S"]) <= float(env["STATEFRAME_RUN_WALL_SECONDS"])
    assert float(env["REACT_REQUEST_TIMEOUT_S"]) <= float(env["STATEFRAME_RUN_WALL_SECONDS"])
    assert float(env["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"]) <= runner.time.monotonic() + remaining - reserve + 1e-3


def test_drive_no_timeout_sets_no_wall_env(tmp_path, monkeypatch):
    env = _drive_spec(tmp_path, monkeypatch, math.inf, None)
    assert "STATEFRAME_RUN_WALL_SECONDS" not in env
    assert "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC" not in env
