"""Model/corpus-supplied selectors and shas must never reach argv as options."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import task_runners
from kernel_test_python import selected_test_python
from probes import react_repair
from stateframe.engine import check_selection

BAD_SELECTORS = ["-p evil", "--rootdir=/", "-c evil.ini", "--import-mode=append", "-x", " --co", "../../etc/passwd", "/etc/passwd"]
BAD_SHAS = ["--output=/tmp/x", "-1", "abc", "zzzzzzz", "abc123; rm", "HEAD", "", 123, None]


def _recorder(calls):
    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout="1 passed in 0.01s\n", stderr="", returncode=0)
    return fake_run


def _flat(calls):
    return [arg for argv in calls for arg in argv]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_a.py").write_text("def test_x(): pass\n")
    return tmp_path


@pytest.mark.parametrize("bad", BAD_SELECTORS)
def test_pytest_unsafe_selector_never_reaches_argv(repo, bad):
    calls: list = []
    result = task_runners.run_pytest(
        repo, "fn", selectors=[bad, "tests/test_a.py::test_x"],
        requires_db=False, run=_recorder(calls))
    flat = _flat(calls)
    tail = flat[flat.index("no:cacheprovider") + 2:]  # everything after the fixed runner flags
    assert tail == ["tests/test_a.py::test_x"]
    assert "evil" not in flat and "--rootdir=/" not in flat
    assert bad in result.output_tail


@pytest.mark.parametrize("bad", BAD_SELECTORS)
def test_pytest_only_unsafe_selector_runs_nothing(repo, bad):
    calls: list = []
    for kwargs in ({"selectors": [bad]}, {"explicit": bad}):
        result = task_runners.run_pytest(
            repo, "fn", requires_db=False, run=_recorder(calls), **kwargs)
        assert result.status == "HARNESS_ERROR"
    assert calls == []  # no subprocess, not even the grep fallback


def test_pytest_valid_argv_is_unchanged(repo):
    calls: list = []
    result = task_runners.run_pytest(
        repo, "fn", selectors=["tests/test_a.py::test_x"],
        requires_db=False, run=_recorder(calls))
    assert calls == [[selected_test_python(), "-m", "pytest", "-x", "-q", "--no-header",
                      "-p", "no:cacheprovider", "--", "tests/test_a.py::test_x"]]
    assert result.status != "HARNESS_ERROR" and "rejected" not in result.output_tail


@pytest.mark.parametrize("bad", BAD_SELECTORS)
def test_vitest_unsafe_selector_never_reaches_argv(repo, bad):
    calls: list = []
    task_runners.run_vitest(
        repo, [bad, "src/a.test.ts"], requires_db=False, run=_recorder(calls))
    assert calls == [[task_runners.VITEST_BINARY, "run", "--reporter=json", "--", "src/a.test.ts"]]


def test_vitest_only_unsafe_selector_runs_nothing(repo):
    calls: list = []
    result = task_runners.run_vitest(
        repo, ["--root=/"], requires_db=False, run=_recorder(calls))
    assert result.status == "HARNESS_ERROR" and calls == []


def test_vitest_valid_argv_is_unchanged(repo):
    calls: list = []
    task_runners.run_vitest(
        repo, ["src/a.test.ts", "name filter"], requires_db=False, run=_recorder(calls))
    assert calls == [[task_runners.VITEST_BINARY, "run", "--reporter=json",
                      "--", "src/a.test.ts", "name filter"]]


def test_vitest_no_selectors_unchanged(repo):
    calls: list = []
    task_runners.run_vitest(repo, [], requires_db=False, run=_recorder(calls))
    assert calls == [[task_runners.VITEST_BINARY, "run", "--reporter=json"]]


def _git_recorder(calls):
    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=1, stdout="", stderr="")
    return fake_run


@pytest.mark.parametrize("bad", BAD_SHAS)
def test_bad_sha_never_reaches_git_argv(monkeypatch, tmp_path, bad):
    monkeypatch.setitem(react_repair.REPOS, "mc", tmp_path / "backend")
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _git_recorder(calls))
    row = {"repo": "mc", "sha": bad, "fn": "argv_guard_unique_fn"}
    assert check_selection.fail_to_pass_ids(row) == []
    assert check_selection.overlay_fix_tests(tmp_path / "g", row) == 0
    git_calls = [argv for argv in calls if argv[:1] == ["git"]]
    assert git_calls == []


def test_good_sha_still_reaches_git(monkeypatch, tmp_path):
    monkeypatch.setitem(react_repair.REPOS, "mc", tmp_path / "backend")
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _git_recorder(calls))
    row = {"repo": "mc", "sha": "abc1234", "fn": "argv_guard_unique_fn"}
    check_selection.fail_to_pass_ids(row)
    assert any("abc1234^" in argv for argv in calls)


def test_is_commit_id():
    assert check_selection.is_commit_id("abcdef1") and check_selection.is_commit_id("A" * 40)
    assert not any(check_selection.is_commit_id(v) for v in BAD_SHAS)


def test_react_repair_grounded_brief_guards_sha():
    """The brief's `git log ... d["sha"]` sits behind the hex-id check (huge fn; source-level pin)."""
    src = Path(react_repair.__file__).read_text()
    idx = src.index('"--format=%s"')
    window = src[idx - 1500:idx]
    assert 'GROUNDED_BRIEF' in window
    assert re.search(r're\.fullmatch\(r"\[0-9a-fA-F\]\{7,64\}", str\(d\["sha"\]\)\)', window)
