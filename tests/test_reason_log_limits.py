"""Arbitrary time limits name themselves when they fire: limit, value and override env (or none)."""
import io
import subprocess
from pathlib import Path

import pytest

import kernel_reason_log as rl


@pytest.fixture()
def out(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    stream = io.StringIO()
    rl.configure(level="WARN", stream=stream)
    yield stream
    rl.reset_for_tests()


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append(kind)
        return fields


def test_model_request_timeout_names_both_env_vars(out):
    import kernel_measurement as km
    journal = _Journal()
    km.journal_provider_request_timeout(journal, request_id="r", step=1, timeout_seconds=300,
                                        attempt=1, endpoint="http://127.0.0.1:1/v1", retry_scheduled=True)
    text = out.getvalue()
    assert "300s" in text and "REACT_REQUEST_TIMEOUT_S" in text and "STATEFRAME_PER_REQUEST_TIMEOUT_S" in text
    assert journal.events == ["provider_request_timeout"]  # the log adds no journal event


def test_plain_wall_limit_names_env(out):
    import kernel_plain_loop as loop
    journal = _Journal()
    journal.write = lambda *a, **k: None
    ticks = iter([0.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0])
    limits = loop.Limits(model="m", wall_s=10.0)
    loop.run_loop("t", provider=lambda *a, **k: None, execute=lambda *a, **k: None, limits=limits,
                  journal=journal, clock=lambda: next(ticks, 5000.0))
    assert "STATEFRAME_PLAIN_WALL_S" in out.getvalue() and "plain_loop" in out.getvalue()


def test_edit_nturn_budget_names_constant_and_no_override(out):
    import kernel_step_contract as sc
    journal = _Journal()
    sc.emit_edit_nturn_skipped_budget(journal, n_sites=2, turns_used=1, remaining_seconds=3.0,
                                      required_seconds=12.0)
    text = out.getvalue()
    assert "EDIT_NTURN_RUN_BUDGET_SECONDS=1800s" in text and "no override" in text
    assert journal.events == ["edit_nturn_skipped_budget"]


def test_git_query_timeout_is_logged_not_silent(out, tmp_path, monkeypatch):
    import kernel_git_view as gv
    import kernel_safe_probe as sp
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(gv.GitView, "build", classmethod(lambda cls, *a, **k: type(
        "V", (), {"env": lambda s: {}, "git_dir": tmp_path, "object_dirs": ()})()))
    monkeypatch.setattr(sp, "run_probe", lambda *a, **k: type(
        "R", (), {"timed_out": True, "exit_code": None, "output_truncated": False})())
    assert gv.legacy_git_query(tmp_path, ["log", "-1"], timeout=3) is None
    assert gv.legacy_git_query(tmp_path, ["log", "-1"], timeout=3) is None
    lines = [l for l in out.getvalue().splitlines() if "git_view" in l]
    assert len(lines) == 1 and "3s" in lines[0] and "no override" in lines[0]


def test_env_preflight_release_and_runner_limits_name_override(out):
    import kernel_env_preflight as ep
    ev = ep._run(("sleep", "5"), Path("."), 0.2, runner=lambda *a: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("sleep", 0.2)))
    assert ev.timed_out and "no env override" in out.getvalue()
