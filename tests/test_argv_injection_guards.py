"""Model/corpus-supplied selectors and shas must never reach argv as options."""
from __future__ import annotations

import re
import shutil
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


# --- Resolution must not rewrite a screened selector into an option-shaped argv entry ---
# A file named `-p<module>` inside the repo (the patch-writing model can create one) is
# reachable through `./-pevil`, `tests/../-pevil`, a basename match or an absolute path;
# every one of those passes the raw screen and used to resolve to a bare `-pevil`, which
# pytest's plugin pre-scan honours even after `--`.

DASH_SELECTORS = ["./-pevil", "tests/../-pevil", "nope/-pevil", "{root}/-pevil"]


def _tail(calls):
    flat = _flat(calls)
    return flat[flat.index("no:cacheprovider") + 2:]


@pytest.fixture
def dash_repo(repo: Path) -> Path:
    (repo / "-pevil").write_text("")
    (repo / "evil.py").write_text("raise SystemExit('plugin loaded')\n")
    return repo


@pytest.mark.parametrize("sel", DASH_SELECTORS)
def test_pytest_resolved_option_target_is_dropped(dash_repo, sel):
    sel = sel.format(root=dash_repo)
    calls: list = []
    result = task_runners.run_pytest(
        dash_repo, "fn", selectors=[sel, "tests/test_a.py::test_x"],
        requires_db=False, run=_recorder(calls))
    assert _tail(calls) == ["tests/test_a.py::test_x"]
    assert not any(arg.startswith("-p") and arg != "-p" for arg in _flat(calls))
    assert "rejected" in result.output_tail and "-pevil" in result.output_tail


@pytest.mark.parametrize("sel", DASH_SELECTORS)
def test_pytest_only_resolved_option_target_runs_nothing(dash_repo, sel):
    sel = sel.format(root=dash_repo)
    calls: list = []
    result = task_runners.run_pytest(
        dash_repo, "fn", selectors=[sel], requires_db=False, run=_recorder(calls))
    assert result.status == "HARNESS_ERROR"
    assert calls == []


@pytest.mark.parametrize("target, dropped", [
    ("-pevil", True), ("./-pevil", True), ("-pevil::test_x", True), (" -x", True),
    ("tests/test_a.py", False), ("tests/test_a.py::test_x", False), (".", False),
    ("tests/../-pevil", False), ("/abs/tests/test_a.py", False),
])
def test_screen_resolved_targets(target, dropped):
    kept, rejected = task_runners.screen_resolved_targets([target])
    assert (rejected == (target,)) is dropped
    assert (kept == (target,)) is not dropped


def test_pytest_symlinked_root_basename_fallback_does_not_raise(repo, tmp_path):
    """relative_to(root.resolve()) on an unresolved rglob match raised ValueError."""
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)
    calls: list = []
    result = task_runners.run_pytest(
        link, "fn", selectors=["nope/test_a.py"], requires_db=False, run=_recorder(calls))
    assert result.status != "HARNESS_ERROR"
    assert _tail(calls) == ["tests/test_a.py"]


def test_pytest_symlink_escaping_root_is_dropped(repo, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_far.py").write_text("def test_y(): pass\n")
    (repo / "tests" / "escape").symlink_to(outside, target_is_directory=True)
    calls: list = []
    result = task_runners.run_pytest(
        repo, "fn", selectors=["nope/test_far.py"], requires_db=False, run=_recorder(calls))
    assert result.status == "HARNESS_ERROR" and calls == []


# --- The corpus `fn` used by the grep fallback can never be a grep option ---

def test_grep_fallback_fn_is_a_pattern_operand(repo):
    calls: list = []
    task_runners.run_pytest(repo, "--include=*", requires_db=False, run=_recorder(calls))
    grep = calls[0]
    assert grep == ["grep", "-rl", "--include=test_*.py", "-e", "--include=*", "--", str(repo)]


@pytest.mark.skipif(shutil.which("grep") is None, reason="grep not installed")
def test_grep_fallback_option_like_fn_yields_no_targets(repo):
    """Real grep: fn='-V' must be searched for, not print the version banner into the targets."""
    calls: list = []
    real_run = subprocess.run

    def spy(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == "grep":
            return real_run(argv, **kwargs)
        return SimpleNamespace(stdout="", stderr="", returncode=4)

    result = task_runners.run_pytest(repo, "-V", requires_db=False, run=spy)
    assert result.status == "NO_TESTS"
    assert [argv[0] for argv in calls] == ["grep"]


# --- The React loop's `test <selector>` surface uses the same final-target screen ---

def _react_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_x.py").write_text("def test_a():\n    assert True\n")
    return repo


def _stub_react_dispatch(monkeypatch):
    import kernel_check_receipts
    import kernel_execution_policy

    dispatched: list = []

    def fake_run_sandboxed(command, root, **_kwargs):
        dispatched.append((tuple(command), root))
        completed = subprocess.CompletedProcess(command, 0, "1 passed in 0.01s\n", "")
        return SimpleNamespace(completed=completed, timed_out=False, output_truncated=False)

    monkeypatch.setenv("STATEFRAME_TEST_RUN_SPANS", "off")
    monkeypatch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", fake_run_sandboxed)
    monkeypatch.setattr(kernel_execution_policy, "policy_read_grants", lambda _result: ())
    monkeypatch.setattr(kernel_check_receipts, "sandbox_open_denial", lambda _text, _grants: None)
    return dispatched


def test_react_repair_test_argv_separates_targets_with_double_dash(tmp_path, monkeypatch):
    repo = _react_repo(tmp_path)
    dispatched = _stub_react_dispatch(monkeypatch)
    observation = react_repair.run_tests(
        repo, "test_x", "tests/test_x.py::test_a", row={"_skip_db": True})
    assert observation.startswith("pytest exit 0"), observation
    (command, _root), = dispatched
    assert command[-2:] == ("--", "tests/test_x.py::test_a")


@pytest.mark.parametrize("selector", ["./-pevil", "-pevil"])
def test_react_repair_option_shaped_target_is_refused(tmp_path, monkeypatch, selector):
    repo = _react_repo(tmp_path)
    (repo / "-pevil").write_text("")
    dispatched = _stub_react_dispatch(monkeypatch)
    observation = react_repair.run_tests(repo, "test_x", selector, row={"_skip_db": True})
    assert dispatched == [], observation
    assert "-pevil" not in " ".join(" ".join(cmd) for cmd, _ in dispatched)
    assert observation.startswith(("[selector] REFUSED_PREDISPATCH", "pytest exit 4")), observation
