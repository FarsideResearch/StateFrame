from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import outcomes, workcopy


def _summary(*, files_changed: list[str] | None = None, runner_kind: str = "pytest",
             check_source: str = "tests", command: str = "pytest -q",
             captured: bool = True, authored_check: bool = False) -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    summary: dict[str, object] = {
        "runner_kind": runner_kind,
        "protect_tests": "off",
        "check_source": check_source,
        "child_exit_code": 0,
        "engine_terminal": {"result": "SUBMITTED", "submission": {
            "stored": True, "seal_id": digest, "artifact_digest": digest}},
        "candidate_artifact_digest": digest,
        "candidate_captured": captured,
        "candidate_sha256": digest,
        "files_changed": [] if files_changed is None else files_changed,
        "tests_after": {"passed": True, "exit_code": 0,
                        "candidate_sha256": digest, "command": command},
    }
    if authored_check:
        summary["authored_check_receipt_id"] = "receipt-1"
        summary["tests_after"] = {"passed": None, "exit_code": None}
    return summary


@pytest.mark.parametrize(
    ("runner_kind", "check_source", "command", "output", "authored", "expected"),
    [
        ("pytest", "tests", "pytest -q", "1 passed in 0.01s\n", False,
         "no files changed; existing tests pass"),
        ("generic", "command", "sh verify.sh", "ok example/pkg 0.01s\n", False,
         "no files changed; existing check passes"),
        ("generic", "build", "npm run build", "build complete\n", False,
         "no files changed; existing check passes"),
        ("generic", "none", "", "", True,
         "no files changed; agent-written check passes"),
    ],
    ids=["tests", "check-command", "build", "agent-written-check"],
)
def test_captured_no_change_uses_check_specific_label(
        runner_kind: str, check_source: str, command: str, output: str,
        authored: bool, expected: str) -> None:
    summary = _summary(runner_kind=runner_kind, check_source=check_source,
                       command=command, authored_check=authored)

    projected = outcomes.project(summary, check_output=output,
                                 declared_test_command=check_source == "tests")

    assert projected["no_change"] is True
    assert projected["verification"] == expected
    assert "verified by" not in str(projected["verification"])


@pytest.mark.parametrize(
    "files_changed",
    [
        [],
        [],
        [],
        [],
        [],
        [],
        ["app.py"],
        ["obsolete.py"],
        ["old.py", "new.py"],
        ["tool.sh"],
    ],
    ids=["edit-task-with-no-edit", "ask-only", "already-satisfied",
         "resume-no-edits-before-or-after", "in-place-no-edit", "chat-turn-no-edit",
         "resume-keeps-edit-from-before-stop", "deletion-recorded-as-change",
         "rename-recorded-as-change", "mode-only-recorded-as-change"],
)
def test_run_context_does_not_replace_captured_file_facts(
        files_changed: list[str]) -> None:
    projected = outcomes.project(_summary(files_changed=files_changed),
                                 check_output="1 passed in 0.01s\n",
                                 declared_test_command=True)

    assert projected["no_change"] is (not files_changed)
    if files_changed:
        assert projected["verification"] == "verified by tests"
    else:
        assert projected["verification"] == "no files changed; existing tests pass"


def test_test_file_only_change_keeps_verification_and_weakening_text() -> None:
    summary = _summary(files_changed=["tests/test_app.py"])
    summary["test_weakening_findings"] = [{
        "test": "tests/test_app.py::test_value", "reason": "assertion removed"}]

    projected = outcomes.project(summary, check_output="1 passed in 0.01s\n",
                                 declared_test_command=True)

    assert projected["no_change"] is False
    assert projected["verification"] == (
        "verified by tests, but 1 test file weakened: "
        "tests/test_app.py::test_value (assertion removed; heuristic)")


def test_reverted_bytes_are_no_change(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_bytes(b"VALUE = 1\n")
    work = workcopy.create(repo, tmp_path / "copy")
    path = work.copy / "app.py"
    path.write_bytes(b"VALUE = 2\n")
    path.write_bytes(b"VALUE = 1\n")

    changed = workcopy.changed_files(work)
    projected = outcomes.project(_summary(files_changed=sorted(changed)),
                                 check_output="1 passed in 0.01s\n",
                                 declared_test_command=True)

    assert changed == {}
    assert projected["no_change"] is True
    assert projected["verification"] == "no files changed; existing tests pass"


def test_unknown_or_uncaptured_candidate_is_not_no_change() -> None:
    uncaptured = outcomes.project(_summary(captured=False), declared_test_command=True)
    missing_paths = _summary()
    missing_paths.pop("files_changed")
    unknown = outcomes.project(missing_paths, check_output="1 passed in 0.01s\n",
                               declared_test_command=True)

    assert uncaptured["no_change"] is None
    assert unknown["no_change"] is None
    assert "verification" not in uncaptured


def test_incomplete_check_keeps_incomplete_label_without_verified_word() -> None:
    summary = _summary()
    summary["tests_after"] = {**summary["tests_after"], "execution_complete": False}

    projected = outcomes.project(summary, declared_test_command=True)

    assert projected["no_change"] is True
    assert projected["verification"] == "verification incomplete"
    assert "verified" not in str(projected["verification"])


def test_pytest_no_change_label_applies_with_all_test_protection() -> None:
    summary = _summary()
    summary["protect_tests"] = "all"

    projected = outcomes.project(summary, check_output="1 passed in 0.01s\n",
                                 declared_test_command=True)

    assert projected["no_change"] is True
    assert projected["verification"] == "no files changed; existing tests pass"


def test_chat_no_edit_uses_projection_without_promoting_files(monkeypatch, tmp_path: Path) -> None:
    from stateframe import chat
    from stateframe.runner import RunOptions

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n")
    session_home = tmp_path / "session"
    base = RunOptions(repo=repo, task="", endpoint=None)
    session = chat.Session(repo=repo, home=session_home, base=base)
    session._project_hooks_resolved = True
    observed: dict[str, object] = {}

    def run_without_edits(options, printer):
        options.run_dir.mkdir(parents=True)
        (options.run_dir / "final.diff").write_text("")
        summary = _summary()
        summary.update(outcomes.project(summary, check_output="1 passed in 0.01s\n",
                                        declared_test_command=True))
        observed.update(summary)
        return summary

    monkeypatch.setattr(chat, "run", run_without_edits)
    record = session.run_task("Explain app.py without changing it.", printer=lambda _line: None)

    assert observed["verification"] == "no files changed; existing tests pass"
    assert observed["no_change"] is True
    assert record.files == []
    assert record.kept is False


@pytest.mark.host_sandbox
def test_cli_no_edit_in_place_json_reports_no_change(tmp_path: Path) -> None:
    from tests.acceptance._support import run_case

    script = tmp_path / "no-edit.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                  "steps": [
                                      {"tool": "read", "arguments": {"path": "app.py"}},
                                      {"tool": "run_tests", "arguments": {
                                          "target": "tests/test_app.py"}},
                                      {"tool": "done", "arguments": {
                                          "summary": "The existing implementation is ready."}},
                                  ]}))
    case = run_case(tmp_path, str(script), task="Add behavior to app.py.",
                    flags=("--in-place", "--json", "--protect-tests", "all",
                           "--test-cmd", "python -m pytest -q tests/test_app.py"),
                    default_test_cmd=False)

    assert case.summary is not None
    assert case.process.returncode == case.summary["exit_code"] == 0
    assert case.summary["engine_result"] == "SUBMITTED"
    assert case.summary["files_changed"] == []
    assert case.summary["no_change"] is True
    assert case.summary["runtime_outcome"] == "completed"
    assert case.summary["check_outcome"] == "pass"
    assert case.summary["check_source"] == "tests"
    assert case.summary["artifact"] == "saved"
    assert case.summary["verification"] == "no files changed; existing tests pass"
    assert json.loads(case.process.stdout) == case.summary


@pytest.mark.host_sandbox
def test_cli_real_edit_keeps_verified_by_tests_label(tmp_path: Path) -> None:
    from tests.acceptance._support import run_case

    script = tmp_path / "edit.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                  "steps": [
                                      {"tool": "read", "arguments": {"path": "app.py"}},
                                      {"tool": "str_replace", "arguments": {
                                          "path": "app.py", "old_str": 'VALUE = "before"',
                                          "new_str": 'VALUE = "after"'}},
                                      {"tool": "run_tests", "arguments": {
                                          "target": "tests/test_app.py"}},
                                      {"tool": "done", "arguments": {
                                          "summary": "Updated app.py and checked it."}},
                                  ]}))
    case = run_case(tmp_path, str(script), task="Change VALUE to after.",
                    flags=("--test-cmd", "python -m pytest -q tests/test_app.py",
                           "--protect-tests", "off"),
                    default_test_cmd=False)

    assert case.summary is not None
    assert case.process.returncode == case.summary["exit_code"] == 0
    assert case.summary["files_changed"] == ["app.py"]
    assert case.summary["no_change"] is False
    assert case.summary["verification"] == "verified by tests"
    assert "result: verified by tests" in case.process.stdout


def test_test_command_without_executed_test_evidence_is_only_a_check() -> None:
    projected = outcomes.project(
        _summary(runner_kind="generic", command="go test ./..."),
        check_output="ok example/pkg 0.01s\n", declared_test_command=True)

    assert projected["verification"] == "no files changed; existing check passes"


def test_unfinished_run_with_an_authored_receipt_gets_no_passing_label() -> None:
    summary = _summary(runner_kind="generic", check_source="none", command="",
                       authored_check=True)
    projected = outcomes.project(summary, cancelled=True)

    assert projected["no_change"] is True
    assert "pass" not in str(projected.get("verification") or "")
