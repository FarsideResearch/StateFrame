"""Regression coverage for directory selectors passed to the run_tests tool."""

import subprocess
from types import SimpleNamespace

import pytest


def _tiny_repo(tmp_path):
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (tests / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    return repo


def _stub_test_dispatch(monkeypatch):
    import kernel_check_receipts
    import kernel_execution_policy
    from probes import react_repair

    dispatched = []

    def fake_run_sandboxed(command, root, **kwargs):
        dispatched.append((tuple(command), root))
        completed = subprocess.CompletedProcess(command, 0, "1 passed in 0.01s\n", "")
        return SimpleNamespace(completed=completed, timed_out=False, output_truncated=False)

    monkeypatch.setenv("STATEFRAME_TEST_RUN_SPANS", "off")
    monkeypatch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", fake_run_sandboxed)
    monkeypatch.setattr(kernel_execution_policy, "policy_read_grants", lambda _result: ())
    monkeypatch.setattr(kernel_check_receipts, "sandbox_open_denial", lambda _text, _grants: None)
    return react_repair, dispatched


@pytest.mark.parametrize("selector", ["tests/", "tests", "./tests/"])
def test_run_tests_dispatches_existing_directory_selectors(tmp_path, monkeypatch, selector):
    repo = _tiny_repo(tmp_path)
    react_repair, dispatched = _stub_test_dispatch(monkeypatch)

    observation = react_repair.run_tests(repo, "test_x", selector, row={"_skip_db": True})

    assert "REFUSED_PREDISPATCH" not in observation, observation
    assert observation.startswith("pytest exit 0"), observation
    assert len(dispatched) == 1
    command, dispatched_root = dispatched[0]
    assert dispatched_root == repo
    assert command[-1] == selector


def test_run_tests_still_refuses_a_missing_path(tmp_path, monkeypatch):
    repo = _tiny_repo(tmp_path)
    react_repair, dispatched = _stub_test_dispatch(monkeypatch)

    observation = react_repair.run_tests(repo, "test_x", "nope/", row={"_skip_db": True})

    assert observation.startswith("[selector] REFUSED_PREDISPATCH"), observation
    assert "nope/" in observation
    assert dispatched == []
