"""A sealed candidate that could not be delivered must never end as SUBMITTED.

Covers the candidate apply helper (in-scope edits still land when out-of-scope baseline files
differ) and the post-summary refusal that rewrites engine_result when delivery failed.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import sys
from pathlib import Path

import pytest

from stateframe import outcomes, runner, workcopy
from stateframe.endpoint import Endpoint
from stateframe.engine.kernel_mutate_records import Journal


@pytest.fixture()
def work(tmp_path: Path) -> workcopy.WorkCopy:
    original = tmp_path / "repo"
    (original / "data").mkdir(parents=True)
    (original / "a.py").write_bytes(b"A = 1\n")
    (original / "b.py").write_bytes(b"B = 1\n")
    (original / "data" / "x.dbf").write_bytes(b"\x03binary\x00")
    return workcopy.create(original, tmp_path / "copy")


def _collect() -> tuple[list[str], object]:
    lines: list[str] = []
    return lines, lines.append


def _prepare_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: dict[str, bytes | None],
    *,
    task: str = "update sample",
) -> tuple[runner.RunOptions, Path, workcopy.WorkCopy, dict[str, object], Path]:
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "outside.py").write_text("outside = 1\n", encoding="utf-8")
    options = runner.RunOptions(
        repo=repo, task=task, in_place=True, run_dir=tmp_path / "run",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1, test_cmd="true", runner_kind="generic", check_source="command",
        target="sample.py", edit_paths=["sample.py"], protect_tests="off", no_project_hooks=True,
    )
    run, work, spec = runner.prepare(options, printer=lambda *_args: None)
    digest = "sha256:" + "a" * 64
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": digest, "artifact_digest": digest,
    }}

    def drive(*_args):
        (run / "engine-result.json").write_text(json.dumps({"result": terminal}))
        return 0

    monkeypatch.setattr(runner, "drive", drive)
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "engine_terminal": terminal, "engine_result": "SUBMITTED", "cached_share_display": "cache",
    })
    monkeypatch.setattr(runner, "final_candidate", lambda _run: candidate)
    monkeypatch.setattr(runner, "_artifact_digest", lambda _candidate: digest)
    monkeypatch.setattr(runner, "_tests_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "_outer_test_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "run_test_cmd", lambda *_args, **_kwargs: {
        "command": "true", "exit_code": 0, "passed": True, "tail": "ok",
    })
    monkeypatch.setattr(runner, "finish_run_storage", lambda *_args, **_kwargs: {
        "checkpoints": None, "env": None,
    })
    return options, run, work, spec, repo


def _execute_prepared(
    options: runner.RunOptions,
    run: Path,
    work: workcopy.WorkCopy,
    spec: dict[str, object],
) -> dict[str, object]:
    return runner._execute(options, 0.0, lambda *_args: None, run, work, spec)


def _mutate_after_verification(monkeypatch: pytest.MonkeyPatch, mutation) -> None:
    project = outcomes.project
    mutated = False

    def project_then_mutate(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            mutation()
        return project(*args, **kwargs)

    monkeypatch.setattr(outcomes, "project", project_then_mutate)


def _assert_in_place_refused(summary: dict[str, object], run: Path, repo: Path) -> None:
    assert summary["engine_result"] == "SUBMITTED"
    assert str(summary["apply_error"]).startswith("candidate changed after host verification:")
    assert summary["exit_code"] == 2
    assert summary.get("applied_in_place", []) == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert not (repo / "created.py").exists()
    assert not (run / "apply-receipt.json").exists()


def test_in_place_late_delete_of_verified_addition_refuses_all_delivery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n", "created.py": b"created\n"})
    spec["public_row"]["task_file_scope"] = None
    _mutate_after_verification(monkeypatch, lambda: (work.copy / "created.py").unlink())

    summary = _execute_prepared(options, run, work, spec)

    _assert_in_place_refused(summary, run, repo)


@pytest.mark.parametrize("task", ["Update sample.py.", "Only edit sample.py."])
def test_in_place_reverted_verified_edit_refuses_scoped_and_unscoped_delivery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task=task)
    spec["public_row"]["task_file_scope"] = ["sample.py"] if task.startswith("Only") else None
    _mutate_after_verification(
        monkeypatch, lambda: (work.copy / "sample.py").write_bytes(b"value = 1\n"))

    summary = _execute_prepared(options, run, work, spec)

    _assert_in_place_refused(summary, run, repo)


def test_in_place_recreated_verified_deletion_refuses_all_delivery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n", "outside.py": None})
    spec["public_row"]["task_file_scope"] = None
    spec["public_row"]["files"]["outside.py"] = {"parent_src": "outside = 1\n"}
    _mutate_after_verification(
        monkeypatch, lambda: (work.copy / "outside.py").write_bytes(b"outside = 1\n"))

    summary = _execute_prepared(options, run, work, spec)

    _assert_in_place_refused(summary, run, repo)


@pytest.mark.parametrize("before_diff", [False, True], ids=["after-scan", "before-scan"])
def test_in_place_undoing_every_verified_change_still_enters_apply(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_diff: bool) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch,
        {"sample.py": b"value = 2\n", "outside.py": None, "created.py": b"created\n"})
    spec["public_row"]["task_file_scope"] = None
    spec["public_row"]["files"]["outside.py"] = {"parent_src": "outside = 1\n"}

    def undo_all() -> None:
        (work.copy / "sample.py").write_bytes(b"value = 1\n")
        (work.copy / "outside.py").write_bytes(b"outside = 1\n")
        (work.copy / "created.py").unlink()

    if before_diff:
        export = runner._export_final_diff

        def undo_then_export(*args):
            undo_all()
            return export(*args)

        monkeypatch.setattr(runner, "_export_final_diff", undo_then_export)
    else:
        _mutate_after_verification(monkeypatch, undo_all)
    summary = _execute_prepared(options, run, work, spec)

    _assert_in_place_refused(summary, run, repo)


@pytest.mark.parametrize("phase", ["final-diff", "summary-scan", "test-inspection"])
def test_in_place_addition_disappearing_during_reporting_uses_apply_refusal(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n", "test_created.py": b"created\n"})
    spec["public_row"]["task_file_scope"] = None
    path = work.copy / "test_created.py"
    scan = workcopy.changed_files
    export = runner._export_final_diff
    inspect = runner.inspect_test_changes
    hash_file = workcopy._hash_file
    read_text = Path.read_text

    def scan_then_delete(target: workcopy.WorkCopy) -> dict[str, str]:
        changes = scan(target)
        if "test_created.py" in changes:
            path.unlink()
        return changes

    def export_then_delete_on_read(*args):
        error = export(*args)

        def hash_after_export(target: Path) -> str:
            if target == path and path.exists():
                path.unlink()
            return hash_file(target)

        monkeypatch.setattr(workcopy, "_hash_file", hash_after_export)
        return error

    def inspect_with_late_deletion(target: workcopy.WorkCopy):
        def read_during_inspection(source: Path, **kwargs) -> str:
            if source == path:
                path.unlink()
            return read_text(source, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", read_during_inspection)
            return inspect(target)

    if phase == "final-diff":
        monkeypatch.setattr(workcopy, "changed_files", scan_then_delete)
    elif phase == "summary-scan":
        monkeypatch.setattr(runner, "_export_final_diff", export_then_delete_on_read)
    else:
        monkeypatch.setattr(runner, "inspect_test_changes", inspect_with_late_deletion)

    summary = _execute_prepared(options, run, work, spec)

    _assert_in_place_refused(summary, run, repo)
    assert not path.exists()
    assert not (repo / "test_created.py").exists()


@pytest.mark.parametrize("placeholder", ["directory", "symlink"])
def test_in_place_verified_deletion_with_ignored_placeholder_promotes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placeholder: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": None})
    digest = runner.candidate_digest

    def digest_with_placeholder(root: Path, **kwargs) -> str:
        path = root / "sample.py"
        if not path.is_dir() and not path.is_symlink():
            assert not path.exists()
            if placeholder == "directory":
                path.mkdir()
            else:
                path.symlink_to("outside.py")
        return digest(root, **kwargs)

    monkeypatch.setattr(runner, "candidate_digest", digest_with_placeholder)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert summary.get("apply_error") is None
    assert summary["exit_code"] == 0
    assert summary["applied_in_place"] == ["sample.py"]
    assert not (repo / "sample.py").exists()
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert (run / "apply-receipt.json").is_file()


def test_in_place_late_new_path_is_not_promoted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"})
    _mutate_after_verification(
        monkeypatch, lambda: (work.copy / "late.py").write_bytes(b"unverified\n"))

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert summary.get("apply_error") is None
    assert summary["applied_in_place"] == ["sample.py"]
    assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert not (repo / "late.py").exists()
    assert (run / "apply-receipt.json").is_file()


def test_in_place_refuses_fifo_replacing_verified_addition(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n", "created.py": b"created\n"})
    spec["public_row"]["task_file_scope"] = None
    apply = workcopy.apply_in_place

    def fail_on_blocking_read(_signum, _frame):
        pytest.fail("applying tried to read a FIFO with no writer")

    def replace_with_fifo(target: workcopy.WorkCopy, **kwargs):
        path = target.copy / "created.py"
        path.unlink()
        os.mkfifo(path)
        previous = signal.signal(signal.SIGALRM, fail_on_blocking_read)
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            return apply(target, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    monkeypatch.setattr(workcopy, "apply_in_place", replace_with_fifo)

    summary = _execute_prepared(options, run, work, spec)

    _assert_in_place_refused(summary, run, repo)


@pytest.mark.parametrize("candidate", [
    {"sample.py": b"value = 2\n"},
    {"sample.py": b"value = 1\n"},
])
def test_in_place_without_check_keeps_delivery_disabled(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: dict[str, bytes]) -> None:
    options, run, work, spec, repo = _prepare_execution(tmp_path, monkeypatch, candidate)
    options.test_cmd = None

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert summary.get("applied_in_place", []) == []
    assert summary["exit_code"] == 1
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert not (run / "apply-receipt.json").exists()


@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
@pytest.mark.parametrize("change", ["modify", "delete"])
def test_no_check_late_workcopy_change_refuses_frozen_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoped: bool, change: str) -> None:
    task = "Only edit sample.py." if scoped else "update sample"
    candidate = {"sample.py": b"value = 2\n" if change == "modify" else None}
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, candidate, task=task)
    options.test_cmd = None
    options.check_source = "none"
    spec["public_row"]["task_file_scope"] = ["sample.py"] if scoped else None

    def change_after_candidate_scan(target, *_args):
        (target.copy / "sample.py").write_bytes(b"value = 1\n")
        return None

    monkeypatch.setattr(runner, "_export_final_diff", change_after_candidate_scan)

    summary = _execute_prepared(options, run, work, spec)

    assert not summary["succeeded"]
    assert summary.get("applied_in_place", []) == []
    assert summary["apply_error"] == "candidate changed after host verification: sample.py"
    assert summary["exit_code"] == 2
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert not (run / "apply-receipt.json").exists()


def test_no_check_promotion_does_not_silently_deliver_a_subset(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch,
        {"sample.py": b"value = 2\n", "created.py": b"created\n"})
    options.test_cmd = None
    options.check_source = "none"

    def revert_one_approved_file(target, *_args):
        (target.copy / "sample.py").write_bytes(b"value = 1\n")
        return None

    monkeypatch.setattr(runner, "_export_final_diff", revert_one_approved_file)

    summary = _execute_prepared(options, run, work, spec)

    assert not summary["succeeded"] and not summary.get("applied_in_place", []), (
        f"no-check promotion reported succeeded={summary['succeeded']} and "
        f"applied={summary.get('applied_in_place')!r} after sample.py changed post-scan")
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert not (repo / "created.py").exists()
    assert summary.get("apply_error") == "candidate changed after host verification: sample.py"
    assert summary["exit_code"] == 2
    assert not (run / "apply-receipt.json").exists()


def test_clean_no_check_candidate_is_delivered(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch,
        {"sample.py": b"value = 2\n", "created.py": b"created\n"})
    options.test_cmd = None
    options.check_source = "none"

    summary = _execute_prepared(options, run, work, spec)

    assert summary["succeeded"]
    assert summary["applied_in_place"] == ["created.py", "sample.py"]
    assert summary["exit_code"] == 0
    assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert (repo / "created.py").read_bytes() == b"created\n"
    assert (run / "apply-receipt.json").is_file()


def test_out_of_scope_drop_still_applies_in_scope_edits(work: workcopy.WorkCopy) -> None:
    candidate: dict[str, bytes | None] = {"a.py": b"A = 2\n", "data/x.dbf": None}
    lines, printer = _collect()

    written, dropped, error = runner._apply_sealed_candidate(
        work, candidate, {"a.py"}, printer)

    assert dropped == ["data/x.dbf"]
    assert error == "sealed candidate changed existing files outside edit scope"
    assert "a.py" in written
    assert (work.copy / "a.py").read_bytes() == b"A = 2\n"
    assert (work.copy / "data" / "x.dbf").read_bytes() == b"\x03binary\x00"
    assert any(line.startswith("warning: candidate changes outside edit scope were dropped")
               and "data/x.dbf" in line for line in lines)


def test_unchanged_outside_task_scope_in_seal_is_not_a_violation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n", "outside.py": b"outside = 1\n"}
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, candidate, task="Only edit sample.py.")
    spec["public_row"]["files"]["outside.py"] = {"parent_src": "outside = 1\n"}

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_files_written"] == ["sample.py"]
    assert summary["applied_in_place"] == ["sample.py"]
    assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"


@pytest.mark.parametrize(("outside_path", "outside_data"), [
    ("outside.py", b"outside = 2\n"),
    ("created.py", b"created = True\n"),
    ("outside.py", None),
])
def test_outside_task_scope_edit_create_delete_refuses_without_in_place_delivery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        outside_path: str, outside_data: bytes | None) -> None:
    candidate = {"sample.py": b"value = 2\n", outside_path: outside_data}
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, candidate, task="Only edit sample.py.")
    spec["public_row"]["files"][outside_path] = {"parent_src": "outside = 1\n"}

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == [outside_path]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert not (repo / "created.py").exists()


def test_outside_task_scope_change_in_retry_workcopy_is_refused_even_when_unsealed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n"}
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, candidate, task="Only edit sample.py.")
    spec["public_row"]["files"]["outside.py"] = {"parent_src": "outside = 1\n"}
    (work.copy / "outside.py").write_bytes(b"outside = 2\n")

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == ["outside.py"]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"


@pytest.mark.parametrize("change", ["edit", "create", "delete"])
def test_in_place_promotes_only_frozen_changes_after_check(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"})
    candidate_digest = runner.candidate_digest
    digest_calls = 0

    def digest_with_late_workcopy_change(root: Path, **kwargs) -> str:
        nonlocal digest_calls
        result = candidate_digest(root, **kwargs)
        digest_calls += 1
        if digest_calls == 2:
            if change == "edit":
                (work.copy / "sample.py").write_bytes(b"value = 3\n")
            elif change == "create":
                (work.copy / "created.py").write_bytes(b"created = True\n")
            else:
                (work.copy / "outside.py").unlink()
        return result

    monkeypatch.setattr(runner, "candidate_digest", digest_with_late_workcopy_change)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    if change == "edit":
        assert summary["apply_error"] == "candidate changed after host verification: sample.py"
        assert summary.get("applied_in_place", []) == []
        assert summary["exit_code"] == 2
        assert not (run / "apply-receipt.json").exists()
        assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    else:
        assert "apply_error" not in summary
        assert summary["applied_in_place"] == ["sample.py"]
        assert summary["exit_code"] == 0
        assert (run / "apply-receipt.json").is_file()
        assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert not (repo / "created.py").exists()


def test_in_place_all_candidate_hashes_are_checked_before_any_write(
        work: workcopy.WorkCopy) -> None:
    first, second = b"A = 2\n", b"B = 2\n"
    (work.copy / "a.py").write_bytes(first)
    (work.copy / "b.py").write_bytes(second)
    changed = workcopy.changed_files(work)
    file_digests = {
        "a.py": hashlib.sha256(first).hexdigest(),
        "b.py": hashlib.sha256(b"unexpected").hexdigest(),
    }

    with pytest.raises(workcopy.WorkCopyError, match="candidate changed after host verification: b.py"):
        workcopy.apply_in_place(work, changed=changed, candidate_file_digests=file_digests)

    assert (work.original / "a.py").read_bytes() == b"A = 1\n"
    assert (work.original / "b.py").read_bytes() == b"B = 1\n"


def test_in_place_reports_candidate_deleted_after_promotion_scan(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task="Only edit sample.py.")
    apply = workcopy.apply_in_place

    def delete_before_read(target: workcopy.WorkCopy, **kwargs):
        assert kwargs["changed"] == {"sample.py": "modified"}
        (target.copy / "sample.py").unlink()
        return apply(target, **kwargs)

    monkeypatch.setattr(workcopy, "apply_in_place", delete_before_read)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["check_outcome"] == "pass"
    assert summary["apply_error"] == "candidate changed after host verification: sample.py"
    assert summary["exit_code"] == 2
    assert summary["succeeded"] is False
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert not (run / "apply-receipt.json").exists()
    assert json.loads((run / "summary.json").read_text())["apply_error"] == summary["apply_error"]


def test_in_place_ignores_check_litter_and_promotes_verified_create_and_delete(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": None, "created.py": b"created = True\n"})
    check = runner.run_test_cmd

    def check_with_litter(command, cwd, **kwargs):
        if (cwd / "created.py").exists():
            (cwd / ".pytest_cache").mkdir()
            (cwd / ".pytest_cache" / "status").write_bytes(b"cached")
            (cwd / "sample.pyc").write_bytes(b"cache")
        return check(command, cwd, **kwargs)

    monkeypatch.setattr(runner, "run_test_cmd", check_with_litter)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["exit_code"] == 0
    assert summary["applied_in_place"] == ["created.py", "sample.py"]
    assert not (repo / "sample.py").exists()
    assert (repo / "created.py").read_bytes() == b"created = True\n"
    assert not (repo / ".pytest_cache").exists()
    assert not (repo / "sample.pyc").exists()


def test_in_place_mode_only_drift_does_not_change_candidate_byte_approval(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"})
    original_mode = stat.S_IMODE((repo / "sample.py").stat().st_mode)
    project = outcomes.project

    def project_with_mode_change(*args, **kwargs):
        result = project(*args, **kwargs)
        (work.copy / "sample.py").chmod(0o600)
        return result

    monkeypatch.setattr(outcomes, "project", project_with_mode_change)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert "apply_error" not in summary
    assert summary["applied_in_place"] == ["sample.py"]
    assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert stat.S_IMODE((repo / "sample.py").stat().st_mode) == original_mode


@pytest.mark.parametrize("change", ["edit", "create", "delete"])
def test_in_place_late_outside_scope_changes_are_not_promoted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task="Only edit sample.py.")
    outside_path = "created.py" if change == "create" else "outside.py"
    project = outcomes.project

    def project_with_late_workcopy_change(*args, **kwargs):
        # This change arrives after the frozen set was collected.
        monkeypatch.setattr(outcomes, "project", project)
        path = work.copy / outside_path
        if change == "delete":
            path.unlink()
        else:
            path.write_bytes(b"outside = 2\n")
        return project(*args, **kwargs)

    monkeypatch.setattr(outcomes, "project", project_with_late_workcopy_change)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "SUBMITTED"
    assert "apply_error" not in summary
    assert summary["applied_in_place"] == ["sample.py"]
    assert summary["artifact"] == "saved"
    assert summary["exit_code"] == 0
    assert (repo / "sample.py").read_bytes() == b"value = 2\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert not (repo / "created.py").exists()


def test_delivery_refuses_outside_scope_change_after_initial_scan(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, _ = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n"}, task="Only edit sample.py.")
    options.in_place = False
    candidate_digest = runner.candidate_digest

    def digest_with_late_workcopy_change(root: Path, **kwargs) -> str:
        # This change is outside the seal but present in the final exported workcopy.
        (root / "outside.py").write_bytes(b"outside = 2\n")
        return candidate_digest(root, **kwargs)

    monkeypatch.setattr(runner, "candidate_digest", digest_with_late_workcopy_change)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED", summary["files_changed"]
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == ["outside.py"]


@pytest.mark.parametrize(("path", "retry_data", "sealed_data"), [
    pytest.param("outside.py", b"outside = 2\n", b"outside = 1\n", id="edit"),
    pytest.param("created.py", b"created = True\n", None, id="create"),
    pytest.param("outside.py", None, b"outside = 1\n", id="delete"),
])
def test_retry_scope_violation_cannot_be_erased_by_unchanged_seal(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
        retry_data: bytes | None, sealed_data: bytes | None) -> None:
    candidate = {"sample.py": b"value = 2\n", path: sealed_data}
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, candidate, task="Only edit sample.py.")
    spec["public_row"]["files"][path] = {"parent_src": "outside = 1\n"}
    retry_path = work.copy / path
    if retry_data is None:
        retry_path.unlink()
    else:
        retry_path.write_bytes(retry_data)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == [path]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == b"outside = 1\n"
    assert not (repo / "created.py").exists()
    assert (retry_path.read_bytes() if retry_path.exists() else None) == retry_data


def test_sealed_scope_violation_survives_committed_effect_reconciliation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = b"outside = 2\n"
    baseline = b"outside = 1\n"
    options, run, work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch, {"sample.py": b"value = 2\n", "outside.py": changed},
        task="Only edit sample.py.")
    # The most recent committed effect has already restored the baseline bytes.
    journal = Journal(run)
    journal.append("EFFECT_COMMITTED", call="restored", files=[{
        "path": "outside.py",
        "pre_sha256": hashlib.sha256(changed).hexdigest(),
        "pre_blob": journal.put_blob(changed),
        "post_sha256": hashlib.sha256(baseline).hexdigest(),
        "post_blob": journal.put_blob(baseline),
    }])

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["candidate_paths_dropped"] == ["outside.py"]
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"
    assert (repo / "outside.py").read_bytes() == baseline


def test_in_scope_only_candidate_applies_without_error(work: workcopy.WorkCopy) -> None:
    candidate: dict[str, bytes | None] = {"a.py": b"A = 3\n"}
    lines, printer = _collect()

    written, dropped, error = runner._apply_sealed_candidate(
        work, candidate, {"a.py"}, printer)

    assert dropped == []
    assert error is None
    assert written == ["a.py"]
    assert (work.copy / "a.py").read_bytes() == b"A = 3\n"
    assert lines == []


def _submitted_summary(*, dropped: list[str], error: str | None) -> dict[str, object]:
    return {
        "engine_result": "SUBMITTED",
        "engine_terminal": {"result": "SUBMITTED", "reason_code": "SUBMITTED_OK"},
        "candidate_captured": True,
        "candidate_error": error,
        "candidate_paths_dropped": dropped,
        "candidate_files_written": ["a.py"],
        "child_exit_code": 0,
        "candidate_sha256": None,
        "candidate_artifact_digest": None,
    }


def test_refusal_rewrites_submitted_when_out_of_scope_dropped() -> None:
    summary = _submitted_summary(
        dropped=["data/x.dbf"],
        error="sealed candidate changed existing files outside edit scope")
    terminal_before = dict(summary["engine_terminal"])

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == runner.DELIVERY_REFUSED == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"] == {
        "engine_result": "SUBMITTED",
        "reason_code": "CANDIDATE_OUT_OF_SCOPE",
        "detail": "sealed candidate changed existing files outside edit scope",
    }
    assert summary["engine_terminal"] == terminal_before


def test_refusal_reason_code_when_not_delivered_without_drops() -> None:
    summary = _submitted_summary(dropped=[], error="disk full while applying")

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_NOT_DELIVERED"
    assert summary["delivery_refusal"]["detail"] == "disk full while applying"


def test_no_refusal_without_candidate_error() -> None:
    summary = _submitted_summary(dropped=[], error=None)

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == "SUBMITTED"
    assert "delivery_refusal" not in summary


@pytest.mark.parametrize("result", ["HOOK_BLOCKED", "CONTINUATION_BLOCKED", "ENGINE_ERROR"])
def test_no_refusal_for_non_submitted_results(result: str) -> None:
    summary = _submitted_summary(dropped=["data/x.dbf"], error="sealed candidate changed")
    summary["engine_result"] = result

    runner._refuse_submitted_delivery(summary)

    assert summary["engine_result"] == result
    assert "delivery_refusal" not in summary


def test_exit_code_stays_nonzero_after_refusal() -> None:
    summary = _submitted_summary(
        dropped=["data/x.dbf"],
        error="sealed candidate changed existing files outside edit scope")
    summary.update(outcomes.project(summary))
    summary["exit_code"] = outcomes.exit_code(summary)
    assert summary["exit_code"] == 2

    runner._refuse_submitted_delivery(summary)

    assert summary["exit_code"] == 2
    assert summary["engine_result"] == "DELIVERY_REFUSED"


def test_budget_stop_hint_keeps_turn_limit_and_object_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options, run, work, spec, _repo = _prepare_execution(tmp_path, monkeypatch, {})
    terminal = {"result": "BUDGET_EXHAUSTED", "error": {"message": "step limit reached"}}

    def drive(*_args):
        (run / "engine-result.json").write_text(json.dumps({"result": terminal}))
        return 0

    monkeypatch.setattr(runner, "drive", drive)
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "engine_terminal": terminal,
        "engine_result": "BUDGET_EXHAUSTED",
        "cached_share_display": "cache",
    })
    output: list[str] = []

    summary = runner._execute(options, 0.0, output.append, run, work, spec)

    assert summary["runtime_outcome"] == "budget_stop"
    assert (f"stopped before finishing; continue it with: stateframe resume {run} "
            "--max-turns <more>") in output
    saved = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert saved["engine_terminal"]["error"] == {"message": "step limit reached"}


def test_apply_failure_preserves_out_of_scope_evidence(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n", "outside.py": b"outside = 2\n"}
    options, run, work, spec, _repo = _prepare_execution(tmp_path, monkeypatch, candidate)

    def fail_apply(*_args, **_kwargs):
        raise OSError("disk full while applying")

    monkeypatch.setattr(workcopy, "apply_files", fail_apply)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["candidate_paths_dropped"] == ["outside.py"]
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["check_outcome"] == "not_run"
    assert summary["exit_code"] == 2


def test_stale_host_check_has_distinct_refusal_and_skips_in_place(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n"}
    options, run, work, spec, repo = _prepare_execution(tmp_path, monkeypatch, candidate)
    check_calls = 0

    def mutate_during_check(*args, **_kwargs):
        nonlocal check_calls
        check_calls += 1
        if check_calls == 2:
            (Path(args[1]) / "sample.py").write_bytes(b"value = 3\n")
        return {"command": "true", "exit_code": 0, "passed": True, "tail": "ok"}

    monkeypatch.setattr(runner, "run_test_cmd", mutate_during_check)

    summary = _execute_prepared(options, run, work, spec)

    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["reason_code"] == "CHECKS_STALE"
    assert summary["check_outcome"] == "not_run"
    assert summary["exit_code"] == 2
    assert summary["applied_in_place"] == []
    assert (repo / "sample.py").read_bytes() == b"value = 1\n"


def test_out_of_scope_classification_on_genuine_submission_has_no_resume_hint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {"sample.py": b"value = 2\n"}
    options, run, work, spec, _repo = _prepare_execution(tmp_path, monkeypatch, candidate)

    def fail_apply(*_args, **_kwargs):
        raise OSError("disk full while applying")

    monkeypatch.setattr(workcopy, "apply_files", fail_apply)
    natural_refusal = runner._refuse_submitted_delivery

    def inject_out_of_scope_classification(summary: dict[str, object]) -> None:
        natural_refusal(summary)
        assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_NOT_DELIVERED"
        # No natural route in this scenario produces this code; inject it at the runner boundary.
        summary["delivery_refusal"]["reason_code"] = "CANDIDATE_OUT_OF_SCOPE"

    monkeypatch.setattr(
        runner, "_refuse_submitted_delivery", inject_out_of_scope_classification,
    )
    output: list[str] = []

    summary = runner._execute(options, 0.0, output.append, run, work, spec)

    assert summary["runtime_outcome"] == "engine_error"
    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert summary["delivery_refusal"]["engine_result"] == "SUBMITTED"
    assert summary["delivery_refusal"]["reason_code"] == "CANDIDATE_OUT_OF_SCOPE"
    assert summary["engine_terminal"]["result"] == "SUBMITTED"
    engine_result = json.loads(
        (run / "engine-result.json").read_text(encoding="utf-8"),
    )["result"]
    assert engine_result["result"] == "SUBMITTED"
    assert not any("stopped before finishing; continue it with:" in line for line in output)

    saved_summary = (run / "summary.json").read_bytes()
    assert not outcomes.resume_hint_applies(summary)
    assert (run / "summary.json").read_bytes() == saved_summary
    assert json.loads(saved_summary)["engine_terminal"]["result"] == "SUBMITTED"
