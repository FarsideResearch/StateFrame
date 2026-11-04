"""Sandbox launch refusals cannot be mistaken for successful checks."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import kernel_generic_check
import kernel_test_isolation as isolation

from kernel_check_receipts import (classify_check_outcome, record_check_output,
                                   sandbox_exec_denial)
from kernel_test_isolation import SandboxReadGrants


def _grants(tmp_path: Path) -> SandboxReadGrants:
    root = tmp_path / "allowed"
    root.mkdir()
    return SandboxReadGrants(frozenset({root}), frozenset({root.parent, Path("/")}))


def test_existing_sandbox_exec_refusal_forms_remain_detected():
    assert "sandbox blocked" in sandbox_exec_denial(
        "sandbox-exec: execvp() of '/missing/python' failed: Operation not permitted")
    assert "sandbox blocked" in sandbox_exec_denial(
        "bwrap: execvp /missing/python: Permission denied")


def test_ungranted_error_opening_is_a_refusal(tmp_path):
    grants = _grants(tmp_path)
    blocked = tmp_path / "hidden" / "interpreter"
    message = f"Error opening {blocked}: Operation not permitted\n"
    assert sandbox_exec_denial(message) is None
    assert sandbox_exec_denial(message, read_grants=grants) == (
        f"the sandbox blocked the project interpreter at {blocked}: this is an "
        "environment error, not a test result (no test ran)")


def test_granted_path_and_realpath_are_not_refusals(tmp_path):
    grants = _grants(tmp_path)
    granted = tmp_path / "allowed" / "program"
    assert sandbox_exec_denial(
        f"Error opening {granted}: Operation not permitted", read_grants=grants) is None
    assert sandbox_exec_denial(
        f"Error opening {granted.parent.parent}: Operation not permitted",
        read_grants=grants) is None
    assert sandbox_exec_denial("PermissionError: Operation not permitted",
                               read_grants=grants) is None
    target = tmp_path / "hidden"
    target.write_text("x")
    granted.symlink_to(target)
    assert sandbox_exec_denial(
        f"Error opening {granted}: Operation not permitted", read_grants=grants)


def test_profile_builder_exports_its_exact_read_grants(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    grants: list[SandboxReadGrants] = []
    argv, _env = isolation.isolated_command(
        [sys.executable, "-I", "-B", "-c", "pass"], tmp_path,
        {"PATH": os.defpath}, read_grants_out=grants)
    assert len(grants) == 1
    assert str(tmp_path.resolve()) in argv[2]
    assert tmp_path.resolve() in grants[0].subpaths
    assert tmp_path.resolve().parent in grants[0].literals
    assert grants[0].allows(tmp_path / "module.py")
    assert not grants[0].allows(tmp_path.parent / "private" / "module.py")


def test_lookalike_in_normal_pytest_output_is_not_reclassified(tmp_path):
    grants = _grants(tmp_path)
    text = (f"test prints: Error opening {tmp_path / 'hidden'}: Operation not permitted\n"
            "1 passed in 0.01s")
    assert sandbox_exec_denial(text, read_grants=grants) is None
    assert classify_check_outcome("PASS", stdout=text, exit_code=0,
                                  read_grants=grants)[0] == "PASS"


def test_zero_exit_refusal_cannot_be_a_pass_receipt(tmp_path):
    grants = _grants(tmp_path)
    blocked = tmp_path / "hidden" / "interpreter"
    runtime = SimpleNamespace(owner=SimpleNamespace(current_step=1), revision="revision-1")
    row = record_check_output(
        runtime, selector="tests/test_example.py", status="PASS", outcome="PASS",
        stdout="", stderr=f"Error opening {blocked}: Operation not permitted\n",
        exit_code=0, collected_tests=1, read_grants=grants)
    assert row["outcome"] == "ENVIRONMENT"
    assert row["status"] != "PASS"
    assert row["environment_error"] == (
        f"the sandbox blocked the project interpreter at {blocked}: this is an "
        "environment error, not a test result (no test ran)")


def test_ungranted_open_refusal_uses_existing_execvp_status(tmp_path):
    grants = _grants(tmp_path)
    runtime = SimpleNamespace(owner=SimpleNamespace(current_step=1), revision="revision-1")
    execvp = record_check_output(
        runtime, selector="tests/test_example.py", status="HOST_REFUSED",
        outcome="ENVIRONMENT", execution_id="execvp-refusal",
        stderr="sandbox-exec: execvp() of '/missing/python' failed: Operation not permitted",
        exit_code=126)
    opening = record_check_output(
        runtime, selector="tests/test_example.py", status="PASS", outcome="PASS",
        execution_id="opening-refusal", read_grants=grants, exit_code=0,
        stderr=f"Error opening {tmp_path / 'hidden' / 'interpreter'}: Operation not permitted\n")
    assert execvp["outcome"] == opening["outcome"] == "ENVIRONMENT"
    assert execvp["status"] == opening["status"] == "HOST_REFUSED"


def test_later_ungranted_open_refusal_replaces_same_execution_pass(tmp_path):
    grants = _grants(tmp_path)
    runtime = SimpleNamespace(owner=SimpleNamespace(current_step=1), revision="revision-1")
    passed = record_check_output(
        runtime, selector="tests/test_example.py", status="PASS", outcome="PASS",
        execution_id="same-execution", exit_code=0)
    assert passed["status"] == "PASS"
    runtime.test_receipts = [{"execution_id": "same-execution", "status": "PASS", "outcome": "PASS"}]
    refused = record_check_output(
        runtime, selector="tests/test_example.py", status="PASS", outcome="PASS",
        execution_id="same-execution", read_grants=grants, exit_code=0,
        stderr=f"Error opening {tmp_path / 'hidden' / 'interpreter'}: Operation not permitted\n")
    assert refused["outcome"] == "ENVIRONMENT"
    assert refused["status"] == "HOST_REFUSED"
    assert runtime.host_check_receipts[-1]["status"] == "HOST_REFUSED"
    assert runtime.test_receipts[0]["outcome"] == "ENVIRONMENT"
    assert runtime.test_receipts[0]["status"] == "HOST_REFUSED"


def test_zero_exit_generic_check_reports_environment(monkeypatch, tmp_path):
    grants = _grants(tmp_path)
    blocked = tmp_path / "hidden" / "interpreter"
    result = SimpleNamespace(
        completed=subprocess.CompletedProcess(
            ["check"], 0, "", f"Error opening {blocked}: Operation not permitted\n"),
        timed_out=False, output_truncated=False, read_grants=grants)
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", lambda *_args, **_kwargs: result)
    observation, code = kernel_generic_check.check_result(
        tmp_path, {"test_command": ["check"], "protected_test_paths": []})
    assert code == 126
    assert "sandbox blocked" in observation
    assert "check passed" not in observation


def test_legacy_generic_result_without_grants_keeps_opening_as_check_output(
        monkeypatch, tmp_path):
    message = f"Error opening {tmp_path / 'hidden' / 'interpreter'}: Operation not permitted\n"
    result = SimpleNamespace(
        completed=subprocess.CompletedProcess(["check"], 0, "", message),
        timed_out=False, output_truncated=False)
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", lambda *_args, **_kwargs: result)

    observation, code = kernel_generic_check.check_result(
        tmp_path, {"test_command": ["check"], "protected_test_paths": []})

    assert code == 0
    assert "check passed" in observation
    assert message.strip() in observation


def test_legacy_generic_result_without_grants_still_detects_execvp_refusal(
        monkeypatch, tmp_path):
    result = SimpleNamespace(
        completed=subprocess.CompletedProcess(
            ["check"], 126, "", "sandbox-exec: execvp() of '/missing/python' failed: "
            "Operation not permitted"),
        timed_out=False, output_truncated=False)
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", lambda *_args, **_kwargs: result)

    observation, code = kernel_generic_check.check_result(
        tmp_path, {"test_command": ["check"], "protected_test_paths": []})

    assert code == 126
    assert "could not run" in observation
    assert "check passed" not in observation


def _policy_results_without_grants(source: str) -> list[int]:
    """Line numbers of PolicyResult(...) calls that do not pass read_grants."""
    import ast
    return [node.lineno for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "PolicyResult"
            and not any(keyword.arg == "read_grants" for keyword in node.keywords)]


def test_production_policy_results_always_record_read_grants():
    # The getattr accessor exists only for old-shape results and test doubles; every
    # PolicyResult the engine builds must carry its grants so refusals stay detectable.
    import kernel_execution_policy
    source = Path(kernel_execution_policy.__file__).read_text(encoding="utf-8")
    assert source.count("PolicyResult(") >= 1
    assert _policy_results_without_grants(source) == []


def test_policy_result_grants_guard_flags_a_planted_omission():
    planted = "PolicyResult(completed=done, timed_out=False)\n"
    assert _policy_results_without_grants(planted) == [1]
    assert _policy_results_without_grants(
        "PolicyResult(completed=done, read_grants=None)\n") == []
