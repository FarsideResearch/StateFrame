"""Opt-in edit scope widening (STATEFRAME_EDIT_SCOPE_WIDEN).

A scoped run (repo over the editable limit, or --target/--edit) keeps the model to the
EditAuthority's files. With the switch on, an edit to another existing product file widens the
scope to that one file, journals a scope_widen event and proceeds; every refusal keeps today's
behavior. Off by default.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from probes import react_repair as rr
from probes.react_repair import (EDIT_SCOPE_WIDEN_CAP, EDIT_SCOPE_WIDENED_KEY, FileSet,
                                 _edit_scope_widen_refusal, _widen_edit_scope)
from kernel_effects import effect_policy_for_run
from kernel_scope_widen_guard import admitted, write as guarded_write

from _journal import measurement_events
from acceptance._support import FIXTURES, Endpoint, PYTHON, environment, read_requests, run_case
from tools.immediate_baseline import baseline_checkout  # noqa: F401 (pytest fixture)

PRICING = "from app.rates import rate\n\ndef total(a, p):\n    return a * rate(p)\n"
BROKEN_RATES = "def rate(p):\n    return 1 - p / 10\n"
FIXED_RATES = "def rate(p):\n    return 1 - p / 100\n"


class Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


def _workspace(tmp_path: Path) -> tuple[Path, dict, FileSet]:
    root = tmp_path / "ws"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "pricing.py").write_text(PRICING)
    (root / "app" / "rates.py").write_text(BROKEN_RATES)
    (root / "app" / "test_helpers.py").write_text("X = 1\n")
    (root / "tests" / "test_pricing.py").write_text("def test_x():\n    pass\n")
    row = {"tier": "USER-REPO", "scoped_edits": True, "runner_kind": "generic",
           "files": {"app/pricing.py": {"parent_src": PRICING}},
           "protected_test_paths": ["tests/test_pricing.py"],
           "allowed_product_paths": ["app"], "new_file_roots": [],
           "entry": {"fn": "", "path": "app/pricing.py"}}
    return root, row, FileSet.from_row(row)


@pytest.fixture
def widen_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EDIT_SCOPE_WIDEN", "on")


def test_existing_product_file_widens_every_editability_source(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    policy = effect_policy_for_run(root, row, set())
    assert not policy.edit_scope("app/rates.py")
    journal = Journal()
    note = _widen_edit_scope(root, "app/rates.py", row, fileset, set(), journal=journal)
    assert note and "app/rates.py" in note and "widened" in note
    assert "app/rates.py" in fileset.paths
    assert fileset.parent["app/rates.py"] == BROKEN_RATES
    assert row["files"]["app/rates.py"] == {"parent_src": BROKEN_RATES}
    assert row[EDIT_SCOPE_WIDENED_KEY] == ["app/rates.py"]
    # The shell effect policy built before the widening sees it on the next effect.
    assert policy.edit_scope("app/rates.py")
    assert journal.events == [("scope_widen", {
        "producer": "react_repair", "dispatch": False, "path": "app/rates.py",
        "pre_edit_sha256": hashlib.sha256(BROKEN_RATES.encode()).hexdigest(),
        "reason": "edit outside the editable set", "widened_count": 1,
        "cap": EDIT_SCOPE_WIDEN_CAP})]


def test_switch_off_never_widens(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_EDIT_SCOPE_WIDEN", raising=False)
    root, row, fileset = _workspace(tmp_path)
    before = json.dumps(row, sort_keys=True)
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set()) is None
    assert json.dumps(row, sort_keys=True) == before
    assert "app/rates.py" not in fileset.paths
    monkeypatch.setenv("STATEFRAME_EDIT_SCOPE_WIDEN", "off")
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set()) is None


def test_unscoped_run_is_untouched(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    row["scoped_edits"] = False
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set()) is None
    assert EDIT_SCOPE_WIDENED_KEY not in row


def test_existing_file_can_widen_across_starting_directories(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    (root / "pkg").mkdir()
    (root / "pkg/rates.py").write_text(BROKEN_RATES)
    assert row["allowed_product_paths"] == ["app"]
    assert _widen_edit_scope(root, "pkg/rates.py", row, fileset, set())
    assert "pkg/rates.py" in fileset.paths


def test_widening_requires_starting_baseline(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(root, baseline)
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set(),
                             baseline_root=baseline)
    second_root, second_row, second_fileset = _workspace(tmp_path / "second")
    (second_root / "app/rates.py").write_text(FIXED_RATES)
    assert _widen_edit_scope(second_root, "app/rates.py", second_row, second_fileset,
                             set(), baseline_root=baseline) is None
    (baseline / "app/rates.py").unlink()
    assert _widen_edit_scope(second_root, "app/rates.py", second_row, second_fileset,
                             set(), baseline_root=baseline) is None


def test_failed_journal_append_does_not_admit(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    class BrokenJournal:
        def emit(self, _kind, **_fields):
            raise OSError("journal unavailable")
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set(),
                             journal=BrokenJournal()) is None
    assert "app/rates.py" not in fileset.paths
    assert not admitted(root / "app/rates.py")


@pytest.mark.parametrize("path,reason", [
    ("tests/test_pricing.py", "not a product source"),
    ("app/test_helpers.py", "not a product source"),
    ("app/missing.py", "not an existing file"),
    ("/etc/passwd", "not relative"),
    ("../outside.py", "normalized"),
    ("app/./rates.py", "normalized"),
    ("app", "not a regular file"),
    ("app/pricing.py", "already editable"),
])
def test_refusals(tmp_path, widen_on, path, reason):
    root, row, fileset = _workspace(tmp_path)
    (tmp_path / "outside.py").write_text("X = 1\n")
    assert reason in (_edit_scope_widen_refusal(root, path, row, fileset, set()) or "")
    assert _widen_edit_scope(root, path, row, fileset, set()) is None or path == "app/pricing.py"
    assert EDIT_SCOPE_WIDENED_KEY not in row


def test_symlink_and_hardlink_refused(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    (tmp_path / "secret.py").write_text("S = 1\n")
    os.symlink(tmp_path / "secret.py", root / "app" / "link.py")
    os.symlink(root / "app" / "rates.py", root / "app" / "inner_link.py")
    os.link(root / "app" / "pricing.py", root / "app" / "alias.py")
    for path in ("app/link.py", "app/inner_link.py", "app/alias.py"):
        assert _widen_edit_scope(root, path, row, fileset, set()) is None, path
    assert EDIT_SCOPE_WIDENED_KEY not in row


def test_protected_paths_refused(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    assert _widen_edit_scope(root, "app/rates.py", row, fileset,
                             {str(root / "app" / "rates.py")}) is None
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set(),
                             protected=lambda path: path == "app/rates.py") is None
    assert EDIT_SCOPE_WIDENED_KEY not in row


def test_named_check_and_hook_input_refused(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    row["public_tests"] = {"app/rates.py": BROKEN_RATES}
    fileset.public_tests["app/rates.py"] = BROKEN_RATES
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set()) is None
    fileset.public_tests.clear()
    row["public_tests"] = {}
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set(),
                             protected=lambda path: path == "app/rates.py") is None
    assert not admitted(root / "app/rates.py")


def test_cap(tmp_path, widen_on):
    root, row, fileset = _workspace(tmp_path)
    for index in range(EDIT_SCOPE_WIDEN_CAP + 1):
        (root / "app" / f"m{index:02d}.py").write_text(f"V = {index}\n")
    admission_order = [f"app/m{index:02d}.py"
                       for index in reversed(range(1, EDIT_SCOPE_WIDEN_CAP + 1))]
    for path in admission_order:
        assert _widen_edit_scope(root, path, row, fileset, set())
    last = "app/m00.py"
    assert "cap" in _edit_scope_widen_refusal(root, last, row, fileset, set())
    assert _widen_edit_scope(root, last, row, fileset, set()) is None
    assert len(row[EDIT_SCOPE_WIDENED_KEY]) == EDIT_SCOPE_WIDEN_CAP
    assert row[EDIT_SCOPE_WIDENED_KEY] == admission_order


def test_file_operations_on_admitted_source(tmp_path, widen_on):
    from kernel_file_ops import file_delete, file_move
    root, row, fileset = _workspace(tmp_path)
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set())
    def allowed(relative):
        return relative in fileset.paths
    moved = file_move(root, "app/rates.py", "app/renamed.py",
                      protected=lambda _: False, editable=allowed,
                      new_file_allowed=lambda path: path == "app/renamed.py")
    assert moved.changed_paths == ("app/rates.py", "app/renamed.py")
    assert not (root / "app/rates.py").exists()
    assert (root / "app/renamed.py").read_text() == BROKEN_RATES
    # A fresh admitted source can also be removed through the guarded delete route.
    delete_root, delete_row, delete_fileset = _workspace(tmp_path / "delete")
    assert _widen_edit_scope(delete_root, "app/rates.py", delete_row, delete_fileset, set())
    effect = file_delete(delete_root, "app/rates.py", protected=lambda _: False,
                         editable=lambda relative: relative in delete_fileset.paths)
    assert effect.changed_paths == ("app/rates.py",)
    assert not (delete_root / "app/rates.py").exists()


def test_decision_is_deterministic(tmp_path, widen_on):
    """Replay re-executes tools from a fresh row; the same calls give the same scope."""
    outcomes = []
    for attempt in ("a", "b"):
        root, row, fileset = _workspace(tmp_path / attempt)
        _widen_edit_scope(root, "app/rates.py", row, fileset, set())
        _widen_edit_scope(root, "tests/test_pricing.py", row, fileset, set())
        outcomes.append((fileset.paths, row[EDIT_SCOPE_WIDENED_KEY]))
    assert outcomes[0] == outcomes[1]


def test_host_widened_scope_paths():
    from stateframe.task_row import EDIT_SCOPE_WIDEN_CAP as HOST_CAP, widened_scope_paths
    assert HOST_CAP == EDIT_SCOPE_WIDEN_CAP
    row = {"scoped_edits": True, "files": {"app/pricing.py": {}}, "allowed_product_paths": ["app"],
           "protected_test_paths": ["app/check.py"], "test_file_paths": ["app/spec_x.py"]}
    baseline = {path: "x" for path in ("app/pricing.py", "app/rates.py", "app/check.py",
                                       "app/spec_x.py", "app/test_a.py", "app/tests/u.py",
                                       "lib/other.py", "app/conftest.py")}
    assert widened_scope_paths(row, [*baseline, "app/new.py"], baseline) == [
        "app/rates.py", "lib/other.py"]
    assert widened_scope_paths({**row, "scoped_edits": False}, ["app/rates.py"], baseline) == []
    many = {f"app/m{index:02d}.py": "x" for index in range(HOST_CAP + 3)}
    assert len(widened_scope_paths(row, list(many), many)) == HOST_CAP
    order = list(reversed(many))
    assert widened_scope_paths(row, order, many) == order[:HOST_CAP]
    assert widened_scope_paths(row, ["app/rates.py"], baseline,
                               hook_paths={"app/rates.py"}) == []


def _host_work(tmp_path):
    from stateframe.workcopy import WorkCopy
    root, row, _ = _workspace(tmp_path)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    baseline = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*.py")}
    return WorkCopy(root, copy, baseline), row


def _host_journal(run_dir, path, digest):
    journal = run_dir / "worker" / "measurement-runs" / "one" / "events.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"kind": "scope_widen", "path": path,
                                   "pre_edit_sha256": digest}) + "\n")


def test_host_rejects_unjournaled_and_protected_candidate(tmp_path, monkeypatch):
    from stateframe import runner
    work, row = _host_work(tmp_path)
    spec = {"public_row": row}
    candidate = {"app/rates.py": FIXED_RATES.encode()}
    with pytest.raises(runner.RunError, match="journal"):
        runner._admitted_candidate_paths(tmp_path / "run", spec, work, candidate)
    _host_journal(tmp_path / "run", "app/pricing.py", work.baseline["app/pricing.py"])
    with pytest.raises(runner.RunError, match="inadmissible|unadmitted"):
        runner._admitted_candidate_paths(tmp_path / "run", spec, work, candidate)
    _host_journal(tmp_path / "run", "app/rates.py", work.baseline["app/rates.py"])
    assert runner._admitted_candidate_paths(tmp_path / "run", spec, work, candidate) == ["app/rates.py"]
    row["public_check_paths"] = ["app/rates.py"]
    with pytest.raises(runner.RunError, match="inadmissible"):
        runner._admitted_candidate_paths(tmp_path / "run", spec, work, candidate)
    row["public_check_paths"] = []
    monkeypatch.setattr(runner, "_widen_hook_paths", lambda _spec: frozenset({"app/rates.py"}))
    with pytest.raises(runner.RunError, match="inadmissible"):
        runner._admitted_candidate_paths(tmp_path / "run", spec, work, candidate)


@pytest.mark.parametrize("destination", [None, "app/renamed.py"])
def test_host_carries_widened_deletion_and_move(tmp_path, destination):
    from stateframe import runner, workcopy
    work, row = _host_work(tmp_path)
    run_dir = tmp_path / "run"
    _host_journal(run_dir, "app/rates.py", work.baseline["app/rates.py"])
    candidate = {"app/rates.py": None}
    if destination:
        candidate[destination] = BROKEN_RATES.encode()
    admitted_paths = runner._admitted_candidate_paths(run_dir, {"public_row": row}, work, candidate)
    written = workcopy.apply_files(work, candidate, declared=set(row["files"]) | set(admitted_paths))
    assert "app/rates.py" in written
    assert not (work.copy / "app/rates.py").exists()
    if destination:
        assert (work.copy / destination).read_text() == BROKEN_RATES


def test_symlink_swap_between_admission_and_write_is_refused(tmp_path, widen_on, monkeypatch):
    from kernel_scope_widen_guard import _open
    root, row, fileset = _workspace(tmp_path)
    target = root / "app/rates.py"
    secret = tmp_path / "secret.py"
    secret.write_text("SECRET = 1\n")
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set())
    def swap_then_open(*args, **kwargs):
        target.rename(root / "app/displaced.py")
        target.symlink_to(secret)
        return _open(*args, **kwargs)
    monkeypatch.setattr("kernel_scope_widen_guard._open", swap_then_open)
    with pytest.raises(OSError):
        guarded_write(target, FIXED_RATES.encode())
    assert secret.read_text() == "SECRET = 1\n"


def test_shell_first_write_outside_scope_is_restored(tmp_path, widen_on):
    from kernel_effects import ContentStore, diff, reconcile, scan
    root, row, _ = _workspace(tmp_path)
    policy = effect_policy_for_run(root, row, set())
    before = scan(root, exclude=lambda _: False)
    store = ContentStore(tmp_path / "shell-store")
    store.capture(root, before)
    (root / "app/rates.py").write_text(FIXED_RATES)
    after = scan(root, exclude=lambda _: False)
    effect = next(item for item in diff(before, after) if item.path == "app/rates.py")
    assert policy.authorize(effect) == (False, "outside edit scope")
    reconciled = reconcile(root, before, after, policy, store, store)
    assert (root / "app/rates.py").read_text() == BROKEN_RATES
    assert any(item.path == "app/rates.py" for item, _ in reconciled.restored)


def test_batch_publishes_only_to_the_admitted_inode(tmp_path, widen_on, monkeypatch):
    from kernel_coordinated_change_ledger import atomic_apply
    root, row, fileset = _workspace(tmp_path)
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    assert _widen_edit_scope(root, "app/rates.py", row, fileset, set())
    result = atomic_apply(root, {"edits": [{"path": "app/rates.py",
                                             "old": "p / 10", "new": "p / 100"}]})
    assert result.status == "APPLIED", result.to_dict()
    assert (root / "app/rates.py").read_text() == FIXED_RATES


def test_runner_and_cli_surface():
    from stateframe import cli, runner
    assert runner.OPT_IN_SWITCHES["STATEFRAME_EDIT_SCOPE_WIDEN"] == ("on", "off")
    assert runner.edit_scope_widened({"environment": {"STATEFRAME_EDIT_SCOPE_WIDEN": "on"}})
    assert not runner.edit_scope_widened({"environment": {}})
    registry = json.loads((Path(rr.__file__).parents[1] / "switch-registry.json").read_text())
    assert "STATEFRAME_EDIT_SCOPE_WIDEN" in json.dumps(registry)
    base = ["run", "--repo", ".", "--task", "x", "--endpoint", "http://127.0.0.1:1/v1", "--model", "m"]
    parser = cli.build_parser()
    assert parser.parse_args(base).edit_scope is None
    assert parser.parse_args([*base, "--edit-scope", "widen"]).edit_scope == "widen"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--edit-scope", "everything"])


# Acceptance: the public CLI against a scripted endpoint on a repo over the editable limit.

def _large_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    for index in range(210):
        (repo / "pkg" / f"mod_{index:03d}.py").write_text(f"VALUE = {index}\n")
    (repo / "app").mkdir()
    (repo / "app" / "__init__.py").write_text("")
    (repo / "app" / "pricing.py").write_text(PRICING)
    (repo / "app" / "rates.py").write_text(BROKEN_RATES)
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "test_pricing.py").write_text(
        "import unittest\nfrom app.pricing import total\n\n\n"
        "class T(unittest.TestCase):\n    def test_total(self):\n"
        "        self.assertEqual(total(80, 25), 60)\n")
    return repo


def _script(tmp_path: Path) -> Path:
    steps = [
        {"tool": "edit", "arguments": {"path": "app/rates.py", "function_source": FIXED_RATES}},
        {"tool": "done", "arguments": {"summary": "Rate fixed."}},
        {"tool": "done", "arguments": {"summary": "Rate fixed."}},
        {"tool": "done", "arguments": {"summary": "Rate fixed."}},
    ]
    path = tmp_path / "widen.script.json"
    path.write_text(json.dumps({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}}))
    return path


def _events(run_dir: Path, kind: str) -> list[dict[str, object]]:
    return [event for event in measurement_events(run_dir)
            if kind in (event.get("kind"), event.get("event"), event.get("type"))]


def _run(tmp_path: Path, widen: bool):
    repo = _large_repo(tmp_path)
    case = run_case(tmp_path, str(_script(tmp_path)), repo=repo,
                    task="Fix the discount math in app/pricing.py.",
                    flags=("--edit", "app/pricing.py", *(("--edit-scope", "widen") if widen else ())))
    return repo, case


def test_acceptance_switch_off_refuses_unlisted_file(tmp_path):
    repo, case = _run(tmp_path, widen=False)
    case.assert_reply('\\"code\\":\\"NOT_EDITABLE\\",\\"path\\":\\"app/rates.py\\"')
    assert (repo / "app" / "rates.py").read_text() == BROKEN_RATES
    assert "rates.py" not in case.diff
    assert not _events(case.run_dir, "scope_widen")
    spec = json.loads((case.run_dir / "spec.json").read_text())
    assert "STATEFRAME_EDIT_SCOPE_WIDEN" not in spec["environment"]


def test_acceptance_switch_on_widens_and_carries_the_change(tmp_path):
    repo, case = _run(tmp_path, widen=True)
    assert "NOT_EDITABLE" not in case.transcript, case.transcript[-3000:]
    case.assert_reply("[edit scope widened] app/rates.py")
    assert "app/rates.py" in case.diff and "p / 100" in case.diff, (case.process.stdout, case.diff)
    events = _events(case.run_dir, "scope_widen")
    assert events and events[0]["path"] == "app/rates.py"
    # A later request lists the widened file among the editable paths.
    listed = 'editable_paths\\":[\\"app/pricing.py\\",\\"app/rates.py\\"]'
    assert listed not in json.dumps(case.requests[0])
    assert any(listed in json.dumps(request) for request in case.requests[1:])
    assert case.summary and case.summary["check_outcome"] == "pass", case.summary


def test_acceptance_batch_str_replace_is_refused_before_dispatch(tmp_path):
    """The shipped action contract rejects the batch form before any tool runs.

    Widening in the batch path itself is covered by the unit test above; here the
    point is that a refused batch never widens the scope or changes a file.
    """
    script = tmp_path / "batch.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                  "steps": [
        {"tool": "str_replace", "arguments": {"edits": [{
            "path": "app/rates.py", "old": "p / 10", "new": "p / 100"}]}},
        *({"tool": "done", "arguments": {"summary": "Rate fixed."}}
          for _ in range(3))]}))
    repo = _large_repo(tmp_path)
    case = run_case(tmp_path, str(script), repo=repo,
                    task="Fix the discount math in app/pricing.py.",
                    flags=("--edit", "app/pricing.py", "--edit-scope", "widen"),
                    env_overrides={"STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY": "on"})
    assert "app/rates.py" not in case.diff, (
        case.process.stdout, case.process.stderr, case.transcript[-3000:])
    assert "REJECTED before dispatch" in case.transcript
    assert not _events(case.run_dir, "scope_widen")


def _remaining_script(tmp_path: Path) -> Path:
    path = tmp_path / "remaining.script.json"
    path.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "done", "arguments": {"summary": "Rate fixed."}} for _ in range(3)]}))
    return path


def test_acceptance_widened_run_resumes_by_replay(tmp_path):
    """Kill the engine after the widened edit; resume replays the tape (never a checkpoint)."""
    from acceptance.test_resume_completion import _engine_child
    from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified

    repo = _large_repo(tmp_path)
    run_dir, marker = tmp_path / "run", tmp_path / "paused"
    endpoint = subprocess.Popen(
        [PYTHON, str(FIXTURES / "pause_endpoint.py"), "--script", str(_script(tmp_path)),
         "--marker", str(marker), "--pause-at", "2", "--pause-seconds", "600"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment())
    runner = None
    try:
        url = endpoint.stdout.readline().strip()
        launched_after = time.time()
        runner = subprocess.Popen(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Fix the discount math in app/pricing.py.", "--endpoint", url,
             "--model", "scripted", "--run-dir", str(run_dir), "--test-python", PYTHON,
             "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "32",
             "--quiet", "--shell", "off", "--edit", "app/pricing.py", "--edit-scope", "widen"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**environment(), "STATEFRAME_CHECKPOINT": "save", "STATEFRAME_CHECKPOINT_EVERY": "1"},
            start_new_session=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline and not marker.exists():
            if runner.poll() is not None:
                raise AssertionError("run ended before interruption: " + str(runner.communicate()))
            time.sleep(0.1)
        assert marker.exists(), "paused model request was not reached"
        expected = read_requests(run_dir)[1]
        child = _engine_child(runner.pid)
        if child is None:
            rows = subprocess.run(("ps", "-ww", "-eo", "pid=,ppid=,args="), capture_output=True,
                                  text=True, check=True).stdout.splitlines()
            related = [row.strip() for row in rows
                       if len((parts := row.strip().split(maxsplit=2))) == 3
                       and (parts[1] == str(runner.pid)
                            or ("_child.py" in parts[2] and str(run_dir) in parts[2]
                                and parts[2].startswith(("/usr/bin/python", PYTHON))))]
        else:
            related = []
        assert child is not None, (
            f"engine child missing at paused request; runner_exit={runner.poll()} "
            f"endpoint_exit={endpoint.poll()} related_processes={related[:5]}")
        signal_run_verified(run_dir, child, record_identity(
            child, launched_after=launched_after, command_fragment="_child.py"),
            signal.SIGKILL, deadline=deadline, what="widened edit replay")
        runner.communicate(timeout=30)
    finally:
        if runner is not None and runner.poll() is None:
            # Failure cleanup may run before a checkpoint exists.
            signal_run_owned(run_dir, runner, signal.SIGKILL, deadline=time.monotonic(),
                             what="widened run cleanup", require_checkpoint_saved=False)
            runner.wait(timeout=5)
        if endpoint.poll() is None:
            endpoint.kill()
            endpoint.wait(timeout=5)
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["environment"]["STATEFRAME_EDIT_SCOPE_WIDEN"] == "on"
    with Endpoint(_remaining_script(tmp_path)) as remaining:
        resumed = subprocess.run(
            [PYTHON, "-m", "stateframe", "resume", str(run_dir), "--endpoint", remaining.url,
             "--model", "scripted", "--quiet"], capture_output=True, text=True,
            env=environment(), timeout=300, check=False)
    output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0, output
    assert "checkpoint: engine state restored" not in output
    actual = read_requests(run_dir)[1]
    # The replayed widening rebuilds the same scope, so the next request is byte-identical.
    assert actual == expected
    assert "[edit scope widened] app/rates.py" in json.dumps(actual)
    diff = (run_dir / "final.diff").read_text()
    assert "app/rates.py" in diff and "p / 100" in diff, output
    assert (repo / "app" / "rates.py").read_text() == BROKEN_RATES


def test_engine_child_lookup_requests_untruncated_ps_arguments(monkeypatch):
    from acceptance.test_resume_completion import _engine_child

    observed = []

    def fake_run(argv, **_kwargs):
        observed.append(argv)
        return SimpleNamespace(stdout=(
            " 25 4321 /usr/bin/python3 -B /very/long/source/path/stateframe/_child.py /tmp/run\n"
        ))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _engine_child(4321) == 25
    assert observed == [("ps", "-ww", "-eo", "pid=,ppid=,args=")]


@pytest.mark.host_sandbox
def test_switch_off_request_bytes_match_base_release(tmp_path, baseline_checkout):
    from test_hook_wiring_parity import _assert_only_wire_ids_moved, _run_checkout, _write_launchers
    current = Path(__file__).resolve().parents[1]
    baseline = baseline_checkout
    assert (baseline / "src/stateframe/runner.py").is_file()
    repo = tmp_path / "repo"
    shutil.copytree(current / "examples/discount_bug", repo)
    launcher, wrapper = tmp_path / "launcher.py", tmp_path / "wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    base_requests = _run_checkout(baseline, repo, run, launcher, wrapper,
                                  tmp_path / "base_modules.json", stable_tool_head="off", final_cache_layout="off")[0]
    run.rename(tmp_path / "base_run")
    # v0.4.0's final cache layout changes wire order; pin it off for cross-release parity.
    current_requests = _run_checkout(current, repo, run, launcher, wrapper,
                                     tmp_path / "current_modules.json",
                                     stable_tool_head="off", final_cache_layout="off")[0]
    # Only request_id and the receipt IDs derived like it moved (receipt determinism, 11-03).
    _assert_only_wire_ids_moved(base_requests, current_requests, run)
