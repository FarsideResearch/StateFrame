"""Directory selector normalization stays local to run_tests predispatch lookup."""

import subprocess
from types import SimpleNamespace

import pytest


def _tiny_repo(tmp_path):
    repo = tmp_path / "repo"
    tests = repo / "tests"
    (tests / "sub").mkdir(parents=True)
    (tests / "test_x.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8"
    )
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


@pytest.mark.parametrize("selector", [
    "tests//",
    "./tests",
    "tests/sub/",
    "tests/test_x.py::test_a",
])
def test_existing_selectors_dispatch_unchanged(tmp_path, monkeypatch, selector):
    repo = _tiny_repo(tmp_path)
    react_repair, dispatched = _stub_test_dispatch(monkeypatch)

    observation = react_repair.run_tests(repo, "test_x", selector, row={"_skip_db": True})

    assert observation.startswith("pytest exit 0"), observation
    assert len(dispatched) == 1
    command, dispatched_root = dispatched[0]
    assert dispatched_root == repo
    assert command[-1] == selector


@pytest.mark.parametrize("selector, expected", [
    (
        "/abs/path",
        "pytest exit 4 — invalid test target list; provide file paths or selectors, not options.",
    ),
    (
        "nope/",
        "[selector] REFUSED_PREDISPATCH — public selector does not resolve to an existing "
        "repository test: nope/; [selector] MISSING: nope/. Closest real paths: "
        "none found; use list_files to inspect the workspace.",
    ),
])
def test_invalid_selectors_keep_existing_refusal(tmp_path, monkeypatch, selector, expected):
    repo = _tiny_repo(tmp_path)
    react_repair, dispatched = _stub_test_dispatch(monkeypatch)

    observation = react_repair.run_tests(repo, "test_x", selector, row={"_skip_db": True})

    assert observation == expected
    assert dispatched == []


@pytest.mark.parametrize("selector", ["tests/sub/::test_a", "tests/*"])
def test_directory_node_ids_and_globs_remain_refused(tmp_path, monkeypatch, selector):
    repo = _tiny_repo(tmp_path)
    react_repair, dispatched = _stub_test_dispatch(monkeypatch)

    observation = react_repair.run_tests(repo, "test_x", selector, row={"_skip_db": True})

    assert observation.startswith("[selector] REFUSED_PREDISPATCH"), observation
    assert dispatched == []
