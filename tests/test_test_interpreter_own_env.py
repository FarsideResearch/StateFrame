"""Only a virtual environment is excluded as StateFrame's own test interpreter."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import runner, test_interpreter
from stateframe.engine import kernel_safe_probe, kernel_test_isolation


@pytest.mark.parametrize(("prefix", "base_prefix", "candidate", "expected"), [
    ("/sysroot/usr", "/sysroot/usr", "/sysroot/usr/bin/python3", False),
    ("/sysroot/usr", "/sysroot/usr", "/sysroot/usr/local/bin/python3", False),
    ("/opt/stateframe-env", "/sysroot/usr", "/opt/stateframe-env/bin/python3", True),
    ("/opt/stateframe-env", "/sysroot/usr", "/sysroot/usr/bin/python3", False),
    ("/opt/stateframe-env", "/sysroot/usr", "/opt/stateframe-env-other/bin/python3", False),
])
def test_own_environment_requires_a_virtual_environment(
    monkeypatch, prefix, base_prefix, candidate, expected,
):
    monkeypatch.setattr(test_interpreter.sys, "prefix", prefix)
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", base_prefix)

    assert test_interpreter._is_own_environment(Path(candidate), Path(prefix)) is expected


@pytest.mark.parametrize("in_venv", [False, True])
def test_path_fallback_excludes_only_the_own_virtual_environment(
    tmp_path, monkeypatch, in_venv,
):
    repo = tmp_path / "project"
    repo.mkdir()
    prefix = tmp_path / "runtime"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "python3"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(prefix))
    monkeypatch.setattr(
        test_interpreter.sys, "base_prefix", str(tmp_path / "system" if in_venv else prefix),
    )

    if in_venv:
        with pytest.raises(test_interpreter.TestInterpreterError, match="own environment"):
            test_interpreter.choose(repo, test_python=None, python=None, path_env=str(bin_dir))
    else:
        assert test_interpreter.choose(
            repo, test_python=None, python=None, path_env=str(bin_dir),
        ) == (str(executable), "python3 on PATH (no project environment found)")


def test_own_virtual_environment_detected_through_interpreter_symlink(tmp_path, monkeypatch):
    system_python = tmp_path / "system" / "bin" / "python3"
    system_python.parent.mkdir(parents=True)
    system_python.write_text("#!/bin/sh\nexit 0\n")
    prefix = tmp_path / "runtime"
    candidate = prefix / "bin" / "python3"
    candidate.parent.mkdir(parents=True)
    candidate.symlink_to(system_python)
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(prefix))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))

    assert test_interpreter._is_own_environment(candidate, prefix)


@pytest.mark.parametrize(
    ("python3_in_own_env", "python_in_own_env", "selected"),
    [(False, False, True), (True, False, True), (False, True, False), (True, True, False)],
)
def test_runner_selects_non_own_path_python_after_python3(
    tmp_path, monkeypatch, python3_in_own_env, python_in_own_env, selected,
):
    repo = tmp_path / "project"
    repo.mkdir()
    own_prefix = tmp_path / "stateframe"
    own_bin = own_prefix / "bin"
    own_bin.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(own_prefix))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))
    if python3_in_own_env:
        python3 = own_bin / "python3"
        python3.write_text("#!/bin/sh\nexit 0\n")
        python3.chmod(0o755)
    python_dir = own_bin if python_in_own_env else bin_dir
    python = python_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(own_bin), str(bin_dir))))
    monkeypatch.setattr(kernel_test_isolation, "check_sandbox_exec_denial", lambda *_args: None)

    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")

    if not selected:
        with pytest.raises(runner.RunError, match="own environment"):
            runner._select_test_python(options, printer=lambda _message: None)
        return
    assert runner._select_test_python(options, printer=lambda _message: None) == str(python)


def test_runner_skips_own_python_and_finds_later_external_python(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    own_prefix = tmp_path / "stateframe"
    own_bin = own_prefix / "bin"
    own_bin.mkdir(parents=True)
    external_bin = tmp_path / "external" / "bin"
    external_bin.mkdir(parents=True)
    own_python = own_bin / "python"
    external_python = external_bin / "python"
    for executable in (own_python, external_python):
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(own_prefix))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setenv("PATH", os.pathsep.join((str(own_bin), str(external_bin))))
    monkeypatch.setattr(kernel_test_isolation, "check_sandbox_exec_denial", lambda *_args: None)
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")

    assert runner._select_test_python(options, printer=lambda _message: None) == str(external_python)
    assert options._capabilities["test_python"] == str(external_python)
    assert options._capabilities["test_python_reason"] == "python on PATH (no project environment found)"


def test_runner_diagnostic_names_both_missing_path_interpreters(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(tmp_path / "system"))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setenv("PATH", "")
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")

    with pytest.raises(runner.RunError) as error:
        runner._select_test_python(options, printer=lambda _message: None)

    assert "python3 on PATH" in str(error.value)
    assert "python on PATH" in str(error.value)


def test_runner_searches_current_directory_for_empty_path_entry_in_order(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    local_python = repo / "python"
    local_python.write_text("#!/bin/sh\nexit 0\n")
    local_python.chmod(0o755)
    later_bin = tmp_path / "later"
    later_bin.mkdir()
    later_python = later_bin / "python"
    later_python.write_text("#!/bin/sh\nexit 0\n")
    later_python.chmod(0o755)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(tmp_path / "stateframe"))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setenv("PATH", os.pathsep + str(later_bin))
    monkeypatch.setattr(kernel_test_isolation, "check_sandbox_exec_denial", lambda *_args: None)
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")

    assert runner._select_test_python(options, printer=lambda _message: None) == str(local_python)


def test_empty_path_does_not_search_current_directory(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    local_python = repo / "python"
    local_python.write_text("#!/bin/sh\nexit 0\n")
    local_python.chmod(0o755)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(tmp_path / "stateframe"))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setenv("PATH", "")
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")

    with pytest.raises(runner.RunError, match="python on PATH was not found"):
        runner._select_test_python(options, printer=lambda _message: None)


def test_python_fallback_resolves_path_parent_before_ownership_check(tmp_path, monkeypatch):
    work = tmp_path / "work"
    own_prefix = work / "own"
    external_own = work / "external" / "own"
    external_bin = external_own / "bin"
    external_bin.mkdir(parents=True)
    target = work / "external" / "system-python"
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o755)
    python = external_bin / "python"
    python.symlink_to(target)
    alias = work / "alias"
    alias.symlink_to(external_own, target_is_directory=True)
    path_entry = alias / ".." / "own" / "bin"
    expected = external_bin.resolve() / "python"
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(own_prefix))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(work / "system"))
    monkeypatch.setenv("PATH", str(path_entry))
    checked = []
    monkeypatch.setattr(
        kernel_test_isolation,
        "check_sandbox_exec_denial",
        lambda path: checked.append(path),
    )
    repo = work / "project"
    repo.mkdir(parents=True)
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")

    selected = runner._select_test_python(options, printer=lambda _message: None)

    assert selected == str(expected)
    assert options._capabilities["test_python"] == selected
    assert Path(selected).is_symlink()
    assert checked == [selected]
    assert not test_interpreter._is_own_environment(Path(checked[0]), own_prefix)


def _install_python_fallback(
    tmp_path, monkeypatch, *, version, version_status=0, reject_isolated=True,
):
    repo = tmp_path / "project"
    repo.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    version_line = f"printf '%s\\n' '{version}'" if version else "printf 'version unavailable\\n' >&2"
    isolated_case = (
        "printf 'Unknown option: -I\\n' >&2; exit 2"
        if reject_isolated else "exit 0"
    )
    script = (
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        f"  --version) {version_line}; exit {version_status} ;;\n"
        f"  -I) {isolated_case} ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    python.write_text(script)
    python.chmod(0o755)
    (tmp_path / "system").mkdir()
    (tmp_path / "stateframe").mkdir()
    monkeypatch.setattr(test_interpreter.sys, "prefix", str(tmp_path / "stateframe"))
    monkeypatch.setattr(test_interpreter.sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(kernel_test_isolation, "check_sandbox_exec_denial", lambda *_args: None)
    calls = []

    # The managed host denies Bubblewrap, so model its result without running this fake outside containment.
    def record_probe(argv, workspace, **kwargs):
        assert argv[0] == str(python)
        arguments = tuple(argv[1:])
        calls.append(arguments)
        if arguments == ("--version",):
            output, exit_code = ((version + "\n").encode() if version else b""), version_status
        elif arguments[:1] == ("-I",) and reject_isolated:
            output, exit_code = b"Unknown option: -I\n", 2
        else:
            output, exit_code = b"", 0
        return SimpleNamespace(
            output=output, exit_code=exit_code, output_truncated=False,
            timed_out=False, cleanup=SimpleNamespace(killed=0), probe_failures=(),
        )

    monkeypatch.setattr(kernel_safe_probe, "run_probe", record_probe)
    return repo, python, calls


@pytest.mark.parametrize("runner_kind", ["pytest", "generic"])
def test_python2_path_fallback_is_refused_with_python3_guidance(
    tmp_path, monkeypatch, runner_kind,
):
    repo, python, calls = _install_python_fallback(
        tmp_path, monkeypatch, version="Python 2.7.18",
    )
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind=runner_kind)

    with pytest.raises(runner.RunError) as error:
        runner._select_test_python(options, printer=lambda _message: None)

    message = str(error.value)
    assert str(python) in message
    assert "Python 2" in message and "Python 3" in message
    assert "--test-python" in message
    assert calls == [("--version",)]


@pytest.mark.parametrize("runner_kind", ["pytest", "generic"])
def test_python3_path_fallback_passes_version_probe(tmp_path, monkeypatch, runner_kind):
    repo, python, calls = _install_python_fallback(
        tmp_path, monkeypatch, version="Python 3.12.1", reject_isolated=False,
    )
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind=runner_kind)

    assert runner._select_test_python(options, printer=lambda _message: None) == str(python)
    assert calls[0] == ("--version",)
    if runner_kind == "pytest":
        assert ("-I", "-B", "-c", "import pytest") in calls


@pytest.mark.parametrize("runner_kind", ["pytest", "generic"])
def test_python_version_probe_error_preserves_existing_interpreter_behavior(
    tmp_path, monkeypatch, runner_kind,
):
    repo, python, calls = _install_python_fallback(
        tmp_path, monkeypatch, version="", version_status=2,
    )
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind=runner_kind)

    if runner_kind == "pytest":
        with pytest.raises(runner.RunError, match="cannot import pytest") as error:
            runner._select_test_python(options, printer=lambda _message: None)
        assert "Python 2" not in str(error.value)
        assert calls == [("--version",), ("-I", "-B", "-c", "import pytest")]
    else:
        assert runner._select_test_python(options, printer=lambda _message: None) == str(python)
        assert calls == [("--version",)]


def test_explicit_test_python_does_not_get_path_fallback_version_probe(tmp_path, monkeypatch):
    repo, python, calls = _install_python_fallback(
        tmp_path, monkeypatch, version="Python 2.7.18",
    )
    options = runner.RunOptions(repo=repo, task="", endpoint=None, runner_kind="generic")
    options.test_python = str(python)

    assert runner._select_test_python(options, printer=lambda _message: None) == str(python)
    assert calls == []
