"""A check program the sandbox cannot see is refused before the repository is copied."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from stateframe import preflight, runner

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sandbox lookup")


def _tool(directory, name="sf-home-tool"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _ask(command, probe, env=None, user=False):
    return preflight.missing_program_message(
        command, test_env=env, repo=Path("/nonexistent-repo"), user_test_command=user,
        run_probe=probe)


def _exit(code):
    return lambda _command, _cwd: {"exit_code": code}


def _never(_command, _cwd):
    raise AssertionError("probe not expected")


@linux
def test_absent_program_is_refused():
    message = _ask("sf-no-such-program test", _exit(preflight.MISSING_PROGRAM_EXIT))
    assert message.startswith("check command not found: sf-no-such-program;")


@linux
def test_invisible_program_names_the_absolute_form(tmp_path):
    tool = _tool(tmp_path / "bin")
    message = _ask("sf-home-tool run 'a b'", _exit(preflight.MISSING_PROGRAM_EXIT),
                   {"PATH": f"{tool.parent}:/usr/bin"})
    assert str(tool) in message and "--test-cmd" in message and "a b" in message


@linux
@pytest.mark.parametrize("code", [0, 1, 127, None])
def test_only_the_probe_verdict_refuses(code):
    assert _ask("sf-no-such-program", _exit(code)) is None


@linux
def test_probe_failure_is_left_to_the_check():
    def broken(_command, _cwd):
        raise runner.RunError("sandbox unavailable")
    assert _ask("sf-no-such-program", broken) is None


@linux
@pytest.mark.host_sandbox
def test_real_sandbox_sees_a_system_program_through_the_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_SANDBOX_VISIBLE", ())
    monkeypatch.setattr(sys, "prefix", "/nonexistent")
    calls = []

    def probe(command, cwd):
        calls.append(command)
        return runner.run_test_cmd(command, cwd, runner_kind="generic")

    assert _ask("env true", probe) is None
    assert calls


@linux
@pytest.mark.parametrize("command", ["/usr/bin/env true", "./tool", "", "'unclosed", "sh -c true"])
def test_commands_that_need_no_probe(command):
    assert _ask(command, _never, {"PATH": "/usr/bin:/bin"}) is None


@linux
def test_relative_path_entry_is_left_to_the_check():
    assert _ask("sf-no-such-program", _never, {"PATH": "bin:/usr/bin"}) is None
    assert _ask("sf-no-such-program", _never, {"PATH": ":/usr/bin"}) is None


@linux
def test_selected_python_is_left_to_the_runner():
    assert _ask("python -m pytest", _never, {"PATH": "/nowhere"}) is None


@linux
@pytest.mark.host_sandbox
def test_real_sandbox_agrees_with_the_check(tmp_path):
    tool = _tool(tmp_path / "home" / "bin")
    env = {"PATH": f"{tool.parent}:{os.environ['PATH']}"}

    def probe(command, cwd):
        return runner.run_test_cmd(command, cwd, test_env=env, runner_kind="generic")

    assert "absolute path" in _ask("sf-home-tool", probe, env)
    assert _ask("sf-no-such-program", probe, env) is not None
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README").write_text("x\n")
    with pytest.raises(runner.RunError, match="check command not found"):
        runner.run_test_cmd("sf-home-tool", repo, test_env=env, runner_kind="generic")


@linux
def test_prepare_refuses_before_any_copy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README").write_text("x\n")
    options = runner.RunOptions(task="t", repo=repo, endpoint=None, test_cmd="sf-no-such-program test",
                                runner_kind="generic", run_dir=tmp_path / "run")
    options._test_python = sys.executable
    monkeypatch.setattr(runner, "run_test_cmd", lambda *_a, **_k: {
        "exit_code": preflight.MISSING_PROGRAM_EXIT})
    monkeypatch.setattr(runner.workcopy, "create", lambda *_a, **_k: pytest.fail("copied"))
    with pytest.raises(runner.RunError, match="check command not found: sf-no-such-program"):
        runner.prepare(options)
    assert not (tmp_path / "run").exists()
