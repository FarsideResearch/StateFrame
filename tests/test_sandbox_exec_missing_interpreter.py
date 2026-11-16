"""A sandboxed check whose interpreter cannot start is an environment error, never RED."""
from __future__ import annotations

from types import SimpleNamespace
import subprocess

import pytest

import kernel_generic_check

MISSING = "bwrap: execvp /repo/.venv/bin/python: No such file or directory\n"


def _run(monkeypatch, tmp_path, code: int, stdout: str, stderr: str) -> tuple[str, int]:
    result = SimpleNamespace(
        completed=subprocess.CompletedProcess(["check"], code, stdout, stderr),
        timed_out=False, output_truncated=False, read_grants=None)
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", lambda *_args, **_kwargs: result)
    return kernel_generic_check.check_result(
        tmp_path, {"test_command": ["check"], "protected_test_paths": []})


def test_bwrap_missing_interpreter_exit_one_is_an_environment_error(monkeypatch, tmp_path):
    # bwrap reports a failed exec with exit 1 and empty stdout: no check ran.
    observation, code = _run(monkeypatch, tmp_path, 1, "", MISSING)
    assert code == 126
    assert "could not run" in observation
    assert "not found inside the check sandbox" in observation
    assert "check failed" not in observation


@pytest.mark.parametrize("stderr", [
    "bwrap: execvp /repo/.venv/bin/python: Permission denied\n",
    "sandbox-exec: execvp() of '/repo/.venv/bin/python' failed: Operation not permitted\n",
])
def test_exec_refusal_with_ordinary_exit_is_an_environment_error(monkeypatch, tmp_path, stderr):
    observation, code = _run(monkeypatch, tmp_path, 1 if "bwrap" in stderr else 71, "", stderr)
    assert code == 126
    assert "sandbox blocked" in observation
    assert "check failed" not in observation


@pytest.mark.parametrize("stderr", [
    "FileNotFoundError: [Errno 2] No such file or directory: 'fixtures/data.json'\n",
    "sh: 1: ./build.sh: not found\nls: cannot access 'out': No such file or directory\n",
    "execvp ./tool: No such file or directory\n",
])
def test_genuine_failure_mentioning_missing_file_stays_red(monkeypatch, tmp_path, stderr):
    observation, code = _run(monkeypatch, tmp_path, 1, "1 failed in 0.01s\n", stderr)
    assert code == 1
    assert "check failed" in observation


def test_bwrap_lookalike_on_stdout_is_check_output(monkeypatch, tmp_path):
    # Only the sandbox's own stderr launch line is a refusal; a program printing the
    # text on stdout is still the program's result.
    observation, code = _run(monkeypatch, tmp_path, 1, MISSING, "")
    assert code == 1
    assert "check failed" in observation
