"""The startup check probe must mount the interpreter it actually tries to run."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import tempfile

import pytest

import kernel_test_isolation as isolation
from stateframe import runner
from stateframe.engine import kernel_test_isolation as runner_isolation
from kernel_check_receipts import sandbox_open_denial


def _require_working_sandbox(tmp_path: Path) -> None:
    if platform.system() not in {"Linux", "Darwin"}:
        pytest.skip("requires bwrap or sandbox-exec")
    try:
        argv, env = isolation.isolated_command(
            [sys.executable, "-I", "-B", "-c", "pass"], tmp_path, dict(os.environ))
        result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True,
                                text=True, timeout=30, check=False)
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        pytest.skip("check sandbox is unavailable on this host")
    if result.returncode != 0:
        pytest.skip("check sandbox cannot start the host interpreter")


@pytest.mark.host_sandbox
def test_probe_accepts_venv_python_without_host_selection(monkeypatch, tmp_path):
    _require_working_sandbox(tmp_path)
    monkeypatch.delenv("STATEFRAME_TEST_PYTHON", raising=False)
    venv = tmp_path / "v"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).resolve().parent}\n", encoding="utf-8")
    python = venv / "bin" / "python"
    python.symlink_to(sys.executable)

    assert isolation.check_sandbox_exec_denial(str(python)) is None


def _make_venv_python(venv: Path) -> Path:
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).resolve().parent}\n", encoding="utf-8")
    python = venv / "bin" / "python"
    python.symlink_to(sys.executable)
    return python


def _shell_python_wrapper(path: Path) -> Path:
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n",
                    encoding="utf-8")
    path.chmod(0o700)
    return path


def test_darwin_check_profile_admits_system_shebang_but_not_private_interpreter(
        monkeypatch, tmp_path, tmp_path_factory):
    wrapper = _shell_python_wrapper(tmp_path / "crash-python")
    # Outside the check root (tmp_path), which the profile grants as the workspace.
    hidden = tmp_path_factory.mktemp("private-interpreter") / "interpreter"
    hidden.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hidden.chmod(0o700)
    monkeypatch.setattr(isolation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(isolation, "backend", lambda: "/usr/bin/sandbox-exec")
    grants = []
    isolation.isolated_command([str(wrapper), "-I", "-B", "-c", "pass"], tmp_path,
                               dict(os.environ), selected_python=str(wrapper),
                               read_grants_out=grants)

    assert len(grants) == 1
    assert grants[0].allows(Path("/bin/sh"))
    if Path("/private/var/select").is_dir():
        assert grants[0].allows(Path("/private/var/select/sh"))
    assert grants[0].allows(wrapper)
    assert not grants[0].allows(hidden)
    assert sandbox_open_denial(
        f"Error opening {hidden}: Operation not permitted", grants[0]) is not None


@pytest.mark.host_sandbox
def test_check_sandbox_probe_executes_shell_python_wrapper(tmp_path):
    wrapper = _shell_python_wrapper(tmp_path / "crash-python")
    assert isolation.check_sandbox_exec_denial(str(wrapper)) is None
    command, env = isolation.isolated_command(
        [str(wrapper), "-I", "-B", "-c", "pass"], tmp_path, dict(os.environ),
        selected_python=str(wrapper))
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def _assert_probe_and_runner_startup(repo: Path, python: Path, monkeypatch) -> None:
    assert runner_isolation.check_sandbox_exec_denial(str(python)) is None
    with tempfile.TemporaryDirectory(prefix="stateframe-check-probe-") as directory:
        command, env = runner_isolation.isolated_command(
            [str(python), "-I", "-B", "-c", "pass"], Path(directory), dict(os.environ),
            selected_python=str(python))
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                                text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stderr
    checked = []
    actual_probe = runner_isolation.check_sandbox_exec_denial

    def record_probe(selected: str) -> str | None:
        checked.append(selected)
        return actual_probe(selected)

    monkeypatch.setattr(runner_isolation, "check_sandbox_exec_denial", record_probe)
    monkeypatch.setattr(runner, "_startup_probe", lambda *_args, **_kwargs: b"ok")
    options = runner.RunOptions(repo=repo, task="test", endpoint=None,
                                test_python=str(python))
    assert runner._select_test_python(options, printer=lambda _line: None) == str(python)
    assert checked == [str(python)]


@pytest.mark.host_sandbox
def test_probe_and_startup_accept_project_venv(monkeypatch, tmp_path):
    _require_working_sandbox(tmp_path)
    monkeypatch.delenv("STATEFRAME_TEST_PYTHON", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    python = _make_venv_python(repo / ".venv")

    _assert_probe_and_runner_startup(repo, python, monkeypatch)


@pytest.mark.host_sandbox
def test_probe_and_startup_accept_system_temp_venv(monkeypatch, tmp_path):
    _require_working_sandbox(tmp_path)
    monkeypatch.delenv("STATEFRAME_TEST_PYTHON", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    temp_dir = Path(tempfile.mkdtemp(prefix="stateframe-probe-"))
    try:
        python = _make_venv_python(temp_dir / "venv")
        _assert_probe_and_runner_startup(repo, python, monkeypatch)
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.host_sandbox
def test_probe_still_reports_a_real_sandbox_exec_refusal(monkeypatch, tmp_path, tmp_path_factory):
    _require_working_sandbox(tmp_path)
    monkeypatch.delenv("STATEFRAME_TEST_PYTHON", raising=False)
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    if platform.system() == "Darwin":
        # A private link to a system shell, outside every granted root: the sandbox refuses
        # it. (Before 11-04 this test passed only because /private/var/select/sh was ungranted.)
        hidden = tmp_path_factory.mktemp("private-interpreter") / "interpreter"
        hidden.symlink_to("/bin/sh")
        python.write_text(f"#!{hidden}\n", encoding="utf-8")
        python.chmod(0o755)
    else:
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o644)

    denial = isolation.check_sandbox_exec_denial(str(python))
    assert denial == (f"the sandbox blocked the project interpreter at {python}: this is an "
                      "environment error, not a test result (no test ran)")


def test_isolated_command_default_keeps_selected_interpreter_and_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("STATEFRAME_TEST_PYTHON", raising=False)
    command = [sys.executable, "-I", "-B", "-c", "pass"]
    supplied_env = {"PATH": os.defpath, "LANG": "C", "SECRET": "not-forwarded"}
    selected_calls = []
    selected = isolation.selected_test_python()
    monkeypatch.setattr(isolation, "selected_test_python",
                        lambda: selected_calls.append(True) or selected)
    try:
        default_argv, default_env = isolation.isolated_command(command, tmp_path, supplied_env)
    except RuntimeError:
        pytest.skip("sandbox executable is unavailable on this host")

    assert selected_calls == [True]
    assert default_argv[0] == isolation.backend()
    assert default_argv[-len(command):] == command
    assert default_env["PATH"] == os.defpath
    assert default_env["HOME"] == str(tmp_path / ".test-tmp")
    assert "SECRET" not in default_env
