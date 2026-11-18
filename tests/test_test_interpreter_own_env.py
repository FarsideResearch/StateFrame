"""Only a virtual environment is excluded as StateFrame's own test interpreter."""
from __future__ import annotations

from pathlib import Path

import pytest

from stateframe import test_interpreter


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
