"""Offline whole-command checks through the public CLI and sandboxed engine."""
from __future__ import annotations

import json
import shutil

import pytest

from ._support import environment, run_case


pytestmark = pytest.mark.host_sandbox


def _script(tmp_path, path: str, old: str, new: str, *, protected: str | None = None):
    steps = [{"tool": "read", "arguments": {"path": path}},
             {"tool": "run_tests", "arguments": {"target": ""}}]
    if protected is not None:
        steps.append({"tool": "str_replace", "arguments": {
            "path": protected, "old_str": "good", "new_str": "bad"}})
    steps.extend([
        {"tool": "str_replace", "arguments": {
            "path": path, "old_str": old, "new_str": new}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Fixed the product and ran its check."}},
    ])
    script = tmp_path / "generic.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                  "steps": steps}))
    return script


def _steps(tmp_path, steps):
    script = tmp_path / "case.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                  "steps": steps}))
    return script


def _journal(case):
    paths = list((case.run_dir / "worker" / "measurement-runs").glob("*/events.jsonl"))
    assert len(paths) == 1
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def _assert_shipped_spec(case, protection: str):
    spec = json.loads((case.run_dir / "spec.json").read_text())
    assert spec["profile"] == "r2hc3"
    assert spec["environment"]["STATEFRAME_F4"] == "on"
    assert spec["environment_sources"]["STATEFRAME_F4"] == "profile:r2hc3"
    assert spec["protect_tests"] == protection
    assert case.summary["profile"] == "r2hc3"
    assert case.summary["protect_tests"] == protection
    return spec


def _edit_results(case, *paths: str):
    wanted = set(paths)
    return [(receipt["attempt"]["target"], receipt["result_class"])
            for event in _journal(case) if event.get("kind") == "action_receipt"
            for receipt in (event["receipt"],)
            if isinstance(receipt.get("attempt"), dict)
            and receipt["attempt"].get("target") in wanted]


def test_acceptance_environment_ignores_inherited_stateframe_switches(monkeypatch):
    monkeypatch.setenv("STATEFRAME_F4", "off")
    monkeypatch.setenv("STATEFRAME_PROTECT_TESTS", "all")
    monkeypatch.setenv("STATEFRAME_ENDPOINT", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("ACCEPTANCE_KEEP", "yes")
    env = environment()
    assert not any(key.startswith("STATEFRAME_") for key in env)
    assert env["ACCEPTANCE_KEEP"] == "yes"
    assert env["PYTHONPATH"]


def _assert_red_green(case, product: str, expected: str):
    assert case.summary is not None, case.process.stdout + case.process.stderr
    after_tail = case.run_dir / "tests-after.txt"
    diagnostic = {
        "tests_before": case.summary["tests_before"],
        "tests_after": case.summary["tests_after"],
        "engine_terminal": case.summary.get("engine_terminal"),
        "candidate_error": case.summary.get("candidate_error"),
        "after_tail": after_tail.read_text() if after_tail.is_file() else None,
    }
    assert case.summary["tests_before"]["passed"] is False, case.summary
    assert case.summary["tests_after"]["passed"] is True, diagnostic
    assert case.summary["succeeded"] is True, diagnostic
    assert expected in (case.run_dir / "workspace" / product).read_text()
    case.assert_reply("check exit 0")


def test_shell_check_fixes_bug_and_protects_check_script(tmp_path):
    repo = tmp_path / "shell-repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "scripts" / "check.sh").write_text(
        "#!/bin/sh\nset -eu\n. ./app.sh\n"
        "if [ \"$VALUE\" != good ]; then echo 'expected good' >&2; exit 1; fi\n")
    script = _script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good",
                     protected="scripts/check.sh")
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "sh scripts/check.sh", "--target", "app.sh",
                           "--protect-tests", "all"))
    _assert_red_green(case, "app.sh", "VALUE=good")
    spec = _assert_shipped_spec(case, "all")
    assert "scripts/check.sh" in spec["public_row"]["protected_test_paths"]
    assert case.summary["verification"] == "verified by check command"
    case.assert_reply("REJECTED")
    case.assert_reply("expected good")
    assert (case.run_dir / "workspace" / "scripts" / "check.sh").read_text() == (
        repo / "scripts" / "check.sh").read_text()


def test_shipped_profile_all_restores_shell_test_edit(tmp_path):
    case = run_case(tmp_path, "shell_policy.script.json",
                    substitutions={"COMMAND": "echo tampered > tests/test_app.py"},
                    flags=("--protect-tests", "all"))
    spec = _assert_shipped_spec(case, "all")
    assert "tests/test_app.py" in spec["public_row"]["protected_test_paths"]
    assert case.summary["files_changed"] == []
    assert (case.run_dir / "workspace/tests/test_app.py").read_bytes() == (
        case.repo / "tests/test_app.py").read_bytes()
    case.assert_reply("restored (not allowed): tests/test_app.py")


def test_shell_done_without_check_is_refused_before_success(tmp_path):
    repo = tmp_path / "shell-repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "scripts" / "check.sh").write_text(
        "#!/bin/sh\nset -eu\n. ./app.sh\n"
        "if [ \"$VALUE\" != good ]; then exit 1; fi\n")
    script = tmp_path / "missing-check.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                  "steps": [
                                      {"tool": "read", "arguments": {"path": "app.sh"}},
                                      {"tool": "done", "arguments": {"summary": "Check passed."}},
                                      {"tool": "str_replace", "arguments": {
                                          "path": "app.sh", "old_str": "VALUE=bad",
                                          "new_str": "VALUE=good"}},
                                      {"tool": "run_tests", "arguments": {"target": ""}},
                                      {"tool": "done", "arguments": {"summary": "Check passed."}},
                                  ]}))
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "sh scripts/check.sh", "--target", "app.sh"))

    _assert_red_green(case, "app.sh", "VALUE=good")
    case.assert_reply("RESULT_NEEDS_WORK")
    case.assert_reply("no check has passed on the current revision")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_test_runner_fixes_bug(tmp_path):
    repo = tmp_path / "node-repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.js").write_text("module.exports = { value: 'bad' };\n")
    (repo / "tests" / "app.test.js").write_text(
        "const test = require('node:test');\n"
        "const assert = require('node:assert/strict');\n"
        "test('value', () => assert.equal(require('../app').value, 'good'));\n")
    script = _script(tmp_path, "app.js", "value: 'bad'", "value: 'good'",
                     protected="tests/app.test.js")
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "node --test", "--target", "app.js",
                           "--protect-tests", "all"))
    _assert_red_green(case, "app.js", "value: 'good'")
    case.assert_reply("REJECTED")
    assert case.summary["verification"] == "verified by tests"


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is not installed")
def test_npm_script_without_tests_reports_check_command(tmp_path):
    repo = tmp_path / "npm-repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.js").write_text("module.exports = { value: 'bad' };\n")
    (repo / "tests" / "app.test.js").write_text("test('value', () => {});\n")
    (repo / "package.json").write_text('{"scripts":{"test":"echo ok"}}\n')
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "app.js"}},
        {"tool": "str_replace", "arguments": {"path": "app.js", "old_str": "'bad'",
                                             "new_str": "'good'"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Fixed the value and ran the check."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "npm test", "--target", "app.js",
                           "--protect-tests", "all"))
    assert case.summary["tests_before"]["passed"] is True
    assert case.summary["tests_after"]["passed"] is True
    assert case.summary["verification"] == "verified by check command"
    assert "result: verified by check command" in case.process.stdout


@pytest.mark.skipif(shutil.which("go") is None, reason="go is not installed")
def test_go_test_runner_fixes_bug(tmp_path):
    repo = tmp_path / "go-repo"
    repo.mkdir()
    (repo / "go.mod").write_text("module example.com/check\n\ngo 1.21\n")
    (repo / "app.go").write_text("package check\n\nfunc Value() string { return \"bad\" }\n")
    (repo / "app_test.go").write_text(
        "package check\n\nimport \"testing\"\n\n"
        "func TestValue(t *testing.T) { if Value() != \"good\" { t.Fail() } }\n")
    script = _script(tmp_path, "app.go", 'return "bad"', 'return "good"',
                     protected="app_test.go")
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "go test ./...", "--target", "app.go",
                           "--protect-tests", "all"), timeout=360)
    _assert_red_green(case, "app.go", 'return "good"')
    case.assert_reply("REJECTED")


@pytest.mark.skipif(shutil.which("node") is None or shutil.which("npm") is None,
                    reason="node and npm are required for package build discovery")
def test_no_tests_package_build_is_verified_by_build_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.js").write_text("module.exports = { value: 'bad' };\n")
    (repo / "verify.js").write_text("if (require('./app').value !== 'good') process.exit(1);\n")
    (repo / "package.json").write_text('{"scripts":{"build":"node verify.js"}}\n')
    case = run_case(tmp_path, str(_script(tmp_path, "app.js", "'bad'", "'good'")),
                    repo=repo, flags=("--target", "app.js"), default_test_cmd=False)
    assert case.summary["tests_before"]["exit_code"] == 1
    assert case.summary["tests_after"]["exit_code"] == 0
    assert case.summary["verification"] == "verified by build only"
    assert case.summary["protect_tests"] == "off"
    assert case.summary["succeeded"] is True


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is required for package build discovery")
def test_build_printing_test_count_keeps_build_result(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "verify.sh").write_text(
        '. ./app.sh\n[ "$VALUE" = good ] || exit 1\necho "Tests: 1 passed, 1 total"\n')
    (repo / "package.json").write_text('{"scripts":{"build":"sh verify.sh"}}\n')
    case = run_case(tmp_path, str(_script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good")),
                    repo=repo, flags=("--target", "app.sh"), default_test_cmd=False)
    assert case.summary["check_source"] == "build"
    assert case.summary["test_execution_evidence"] is False
    assert case.summary["verification"] == "verified by build only"
    assert "result: verified by build only" in case.process.stdout


def test_check_command_printing_go_receipt_keeps_command_result(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "verify.sh").write_text(
        '. ./app.sh\n[ "$VALUE" = good ] || exit 1\necho "ok example/pkg 0.01s"\n')
    case = run_case(tmp_path, str(_script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good")),
                    repo=repo, flags=("--check-cmd", "sh verify.sh", "--target", "app.sh"),
                    default_test_cmd=False)
    assert case.summary["check_source"] == "command"
    assert case.summary["test_execution_evidence"] is False
    assert case.summary["verification"] == "verified by check command"
    assert "result: verified by check command" in case.process.stdout


def test_no_tests_make_build_is_verified_by_build_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "verify.sh").write_text('. ./app.sh\n[ "$VALUE" = good ]\n')
    (repo / "Makefile").write_text("build:\n\tsh verify.sh\n")
    case = run_case(tmp_path, str(_script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good")),
                    repo=repo, flags=("--target", "app.sh"), default_test_cmd=False)
    assert case.summary["tests_before"]["passed"] is False
    assert case.summary["tests_after"]["passed"] is True
    assert case.summary["verification"] == "verified by build only"
    assert case.summary["succeeded"] is True


def test_no_tests_no_build_is_unverified(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.sh").write_text("VALUE=bad\n")
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "app.sh"}},
        {"tool": "str_replace", "arguments": {"path": "app.sh", "old_str": "VALUE=bad",
                                               "new_str": "VALUE=good"}},
        {"tool": "done", "arguments": {"summary": "Changed the requested value."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--target", "app.sh"), default_test_cmd=False)
    assert case.summary["verification"] == "unverified: review the diff"
    assert case.summary["tests_after"]["passed"] is None
    assert case.summary["succeeded"] is True


def test_no_tests_agent_written_check_is_verified_by_red_green_pair(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 'bad'\n")
    path = "scratch/authored-check/check_value.py"
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "str_replace", "arguments": {"path": "app.py",
                                               "old_str": "VALUE = 'bad'", "new_str": "VALUE = 'good'"}},
        {"tool": "authored_check", "arguments": {
            "check_id": "value-check", "requirement_id": "value-good",
            "clause": "value should be good", "requirement_quote": "value should be good",
            "path": path, "selector": path, "language": "python",
            "source": "print('__STATEFRAME_AUTHORED_OBSERVED__')\n"
                      "from app import VALUE\nassert VALUE == 'good'\n",
            "write_authority": {"authority_ref": "task", "allowed_paths": [path]},
        }},
        {"tool": "done", "arguments": {"summary": "Fixed the value."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo, task="The value should be good",
                    flags=("--target", "app.py"), default_test_cmd=False)
    assert case.summary["tests_after"]["passed"] is None
    assert case.summary["verification"] == "verified by an agent-written check"
    assert case.summary["authored_check_receipt_id"].startswith("authored-lifecycle:")
    assert case.summary["succeeded"] is True


def test_no_tests_explicit_check_command(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "verify.sh").write_text('. ./app.sh\n[ "$VALUE" = good ]\n')
    case = run_case(tmp_path, str(_script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good")),
                    repo=repo, flags=("--check-cmd", "sh verify.sh", "--target", "app.sh"),
                    default_test_cmd=False)
    assert case.summary["tests_before"]["passed"] is False
    assert case.summary["tests_after"]["passed"] is True
    assert case.summary["verification"] == "verified by check command"


def test_no_tests_test_cmd_still_reports_check_command(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "verify.sh").write_text('. ./app.sh\n[ "$VALUE" = good ]\n')
    case = run_case(tmp_path, str(_script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good")),
                    repo=repo, flags=("--test-cmd", "sh verify.sh", "--target", "app.sh"),
                    default_test_cmd=False)
    assert case.summary["check_source"] == "command"
    assert case.summary["verification"] == "verified by check command"


def test_non_test_command_with_existing_tests_reports_check_command(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "tests/test_unrelated.py").write_text("def test_unrelated():\n    assert True\n")
    (repo / "verify.sh").write_text('. ./app.sh\n[ "$VALUE" = good ]\n')
    case = run_case(tmp_path, str(_script(tmp_path, "app.sh", "VALUE=bad", "VALUE=good")),
                    repo=repo, flags=("--test-cmd", "sh verify.sh", "--target", "app.sh"),
                    default_test_cmd=False)
    assert case.summary["tests_after"]["passed"] is True
    assert case.summary["check_source"] == "command"
    assert case.summary["verification"] == "verified by check command"
    assert "result: verified by check command" in case.process.stdout


def test_off_mode_reports_test_weakening_by_file_and_test(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    assertion = 'if [ "$VALUE" != good ]; then exit 1; fi'
    (repo / "tests/check.sh").write_text(f"#!/bin/sh\n. ./app.sh\n{assertion}\n")
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/check.sh"}},
        {"tool": "str_replace", "arguments": {"path": "tests/check.sh",
                                               "old_str": assertion, "new_str": "true"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Check finished."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "sh tests/check.sh", "--target", "app.sh"))
    spec = _assert_shipped_spec(case, "off")
    assert spec["public_row"]["protected_test_paths"] == []
    assert case.summary["test_files_changed"] == [{
        "path": "tests/check.sh", "change": "modified",
        "weakening": [{"test": "tests/check.sh::file", "reason": "assertion removed",
                       "method": "heuristic"}]}]
    assert case.summary["test_weakening_findings"] == [{
        "test": "tests/check.sh::file", "reason": "assertion removed", "method": "heuristic"}]
    assert case.summary["verification"] == (
        "verified by check command, but 1 test file weakened: "
        "tests/check.sh::file (assertion removed; heuristic)")
    assert (case.run_dir / "workspace/tests/check.sh").read_text().endswith("true\n")
    assert "tests/check.sh::file" in case.process.stdout
    assert "1 test file changed: tests/check.sh" in case.process.stdout


def test_task_mode_locks_judging_test_but_allows_other_test(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "tests/check.sh").write_text(
        'echo tests/check.sh\n. ./app.sh\n[ "$VALUE" = good ]\n')
    (repo / "tests/other.sh").write_text("assert good\n")
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/other.sh"}},
        {"tool": "str_replace", "arguments": {"path": "tests/other.sh",
                                               "old_str": "assert good", "new_str": "assert bad"}},
        {"tool": "str_replace", "arguments": {"path": "tests/check.sh",
                                               "old_str": "good", "new_str": "bad"}},
        {"tool": "str_replace", "arguments": {"path": "app.sh",
                                               "old_str": "VALUE=bad", "new_str": "VALUE=good"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Fixed app value."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "sh tests/check.sh", "--target", "app.sh",
                           "--protect-tests", "task"))
    spec = _assert_shipped_spec(case, "task")
    assert spec["public_row"]["protected_test_paths"] == ["tests/check.sh"]
    assert "tests/other.sh" in case.summary["files_changed"]
    assert (case.run_dir / "workspace/tests/other.sh").read_text() == "assert bad\n"
    assert (case.run_dir / "workspace/tests/check.sh").read_bytes() == (repo / "tests/check.sh").read_bytes()
    assert [item["path"] for item in case.summary["test_files_changed"]] == ["tests/other.sh"]
    assert _edit_results(case, "tests/other.sh", "tests/check.sh") == [
        ("tests/other.sh", "APPLIED"), ("tests/check.sh", "REJECTED")]
    case.assert_reply("REJECTED")


def test_task_mode_locks_test_reported_failing_at_start(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "tests/check.sh").write_text('. ./app.sh\n[ "$VALUE" = good ]\n')
    (repo / "verify.sh").write_text("echo tests/check.sh\nsh tests/check.sh\n")
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/check.sh"}},
        {"tool": "str_replace", "arguments": {"path": "tests/check.sh",
                                               "old_str": "good", "new_str": "bad"}},
        {"tool": "str_replace", "arguments": {"path": "app.sh",
                                               "old_str": "VALUE=bad", "new_str": "VALUE=good"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Fixed product value."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "sh verify.sh", "--target", "app.sh",
                           "--protect-tests", "task"))
    assert case.summary["verification"] == "verified by check command"
    assert (case.run_dir / "workspace/tests/check.sh").read_bytes() == (repo / "tests/check.sh").read_bytes()
    case.assert_reply("REJECTED")


def test_task_lock_uses_full_starting_output_before_long_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    (repo / "tests/check.sh").write_text('. ./app.sh\n[ "$VALUE" = good ]\n')
    (repo / "tests/other.sh").write_text("true\n")
    (repo / "verify.sh").write_text(
        "echo tests/check.sh\n"
        "python3 -c 'print(\"x\" * 5000)'\n"
        "sh tests/check.sh\n")
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/check.sh"}},
        {"tool": "str_replace", "arguments": {"path": "tests/check.sh",
                                             "old_str": "good", "new_str": "bad"}},
        {"tool": "str_replace", "arguments": {"path": "app.sh",
                                             "old_str": "VALUE=bad", "new_str": "VALUE=good"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Fixed value."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--check", "tests/other.sh", "--test-cmd", "sh verify.sh",
                           "--target", "app.sh", "--protect-tests", "task"),
                    default_test_cmd=False)
    assert case.summary["tests_before"]["passed"] is False
    assert case.summary["verification"] == "verified by check command"
    assert (case.run_dir / "workspace/tests/check.sh").read_bytes() == (
        repo / "tests/check.sh").read_bytes()
    case.assert_reply("REJECTED")


def test_off_mode_module_skip_qualifies_passing_pytest_result(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 'bad'\n")
    (repo / "tests/test_app.py").write_text(
        "from app import VALUE\n\ndef test_value():\n    assert VALUE == 'good'\n")
    (repo / "tests/test_other.py").write_text("def test_other():\n    assert True\n")
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/test_app.py"}},
        {"tool": "str_replace", "arguments": {"path": "tests/test_app.py",
                                             "old_str": "from app import VALUE",
                                             "new_str": "import pytest\npytest.skip('later', allow_module_level=True)\nfrom app import VALUE"}},
        {"tool": "str_replace", "arguments": {"path": "app.py",
                                             "old_str": "'bad'", "new_str": "'good'"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "Fixed value."}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "python -m pytest -q tests",
                           "--check", "tests/test_app.py", "--check", "tests/test_other.py",
                           "--target", "app.py",
                           "--protect-tests", "off"), default_test_cmd=False)
    assert case.summary["tests_after"]["passed"] is True
    assert case.summary["verification"] == (
        "verified by tests, but 1 test file weakened: tests/test_app.py::test_value "
        "(skip/xfail/only/todo marker added; heuristic)")
    assert case.summary["test_weakening_findings"] == [{
        "test": "tests/test_app.py::test_value", "reason": "skip/xfail/only/todo marker added",
        "method": "heuristic"}]
    assert case.summary["protect_tests"] == "off"


@pytest.mark.parametrize("mode", ("off", "all"))
def test_wrong_judging_test_stops_with_evidence(tmp_path, mode):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.sh").write_text("VALUE=bad\n")
    assertion = 'if [ "$VALUE" != good ]; then exit 1; fi'
    (repo / "tests/check.sh").write_text(f"#!/bin/sh\n. ./app.sh\n{assertion}\n")
    evidence = {"path": "tests/check.sh", "test": "check value", "assertion": assertion,
                "why": "The requested value is bad.", "observed": "The check exits 1 on bad."}
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/check.sh"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "The test is wrong.", "test_wrong": evidence}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "sh tests/check.sh", "--target", "app.sh",
                           "--protect-tests", mode),
                    task="The current bad value is correct")
    assert case.summary["engine_result"] == "TEST_REPORTED_WRONG"
    assert case.summary["protect_tests"] == mode
    assert case.summary["verification"] == "stopped: test reported wrong, see evidence"
    assert case.summary["wrong_test_evidence"] == evidence
    assert case.summary["succeeded"] is False
    events = [item for item in _journal(case) if item.get("kind") == "test_reported_wrong"]
    assert len(events) == 1 and events[0]["evidence"] == evidence


@pytest.mark.parametrize("mode", ("task", "off"))
def test_pytest_editable_modes_wrong_judging_test_stops_with_evidence(tmp_path, mode):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 'bad'\n")
    assertion = "assert VALUE == 'good'"
    (repo / "tests/test_app.py").write_text(
        f"from app import VALUE\n\ndef test_value():\n    {assertion}\n")
    evidence = {"path": "tests/test_app.py", "test": "test_value", "assertion": assertion,
                "why": "The requested bad value is correct.",
                "observed": "The assertion fails because VALUE is bad."}
    script = _steps(tmp_path, [
        {"tool": "read", "arguments": {"path": "tests/test_app.py"}},
        {"tool": "run_tests", "arguments": {"target": ""}},
        {"tool": "done", "arguments": {"summary": "The test is wrong.", "test_wrong": evidence}},
    ])
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "python -m pytest -q tests/test_app.py",
                           "--check", "tests/test_app.py", "--target", "app.py",
                           "--protect-tests", mode),
                    task="The current bad value is correct", default_test_cmd=False)
    assert case.summary["engine_result"] == "TEST_REPORTED_WRONG"
    assert case.summary["verification"] == "stopped: test reported wrong, see evidence"
    assert case.summary["wrong_test_evidence"] == evidence
    assert case.summary["succeeded"] is False
    assert [event["evidence"] for event in _journal(case)
            if event.get("kind") == "test_reported_wrong"] == [evidence]
