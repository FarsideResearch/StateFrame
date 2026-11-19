import os
import subprocess
import sys
import time
import tokenize
from pathlib import Path

import pytest

from stateframe import runner
from stateframe.endpoint import Endpoint
from stateframe.task_row import TaskRowError, build_row
from stateframe.test_runners import verification_evidence_for_tests
from stateframe.workcopy import create


class _Selected(Exception):
    pass


def _stop_after_selection(_options):
    raise _Selected


def _run_pytest(repo, *selectors):
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *selectors],
        cwd=repo, env=env, capture_output=True, text=True, check=False)


def test_default_checks_use_declared_python_encoding(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pytest.ini").write_text("", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_first.py").write_text("def test_first(): pass\n", encoding="utf-8")
    (repo / "test_second.py").write_bytes(
        b"# coding: latin-1\n# caf\xe9\ndef test_second(): assert False\n")

    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix app")

    assert row["public_check_paths"] == ["test_first.py", "test_second.py"]
    result = _run_pytest(work.copy, *row["public_check_paths"])
    assert result.returncode == 1


@pytest.mark.parametrize("checks", [None, ["test_second.py"]], ids=["discovery", "selected"])
def test_default_checks_report_python_read_error(tmp_path, monkeypatch, checks):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_first.py").write_text("def test_first(): pass\n", encoding="utf-8")
    (repo / "test_second.py").write_text("def test_second(): assert False\n", encoding="utf-8")
    work = create(repo, tmp_path / "copy")
    target = work.copy / "test_second.py"
    original_path_open = Path.open
    original_tokenize_open = tokenize.open

    def guarded_path_open(path, *args, **kwargs):
        if path == target:
            raise PermissionError("read denied")
        return original_path_open(path, *args, **kwargs)

    def guarded_tokenize_open(file, *args, **kwargs):
        try:
            matches = Path(file) == target
        except (TypeError, ValueError):
            matches = False
        if matches:
            raise PermissionError("read denied")
        return original_tokenize_open(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_path_open)
        patch.setattr(tokenize, "open", guarded_tokenize_open)
        with pytest.raises(
            TaskRowError, match="cannot prepare repository text.*test_second.py"):
            build_row(work, task="Fix app", checks=checks)


@pytest.mark.parametrize("checks", [None, ["test_second.py"]], ids=["discovery", "selected"])
def test_default_checks_report_python_decode_error(tmp_path, checks):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_first.py").write_text("def test_first(): pass\n", encoding="utf-8")
    (repo / "test_second.py").write_bytes(
        b"# coding: codec_that_does_not_exist\n"
        b"def test_second(): assert False\n")

    work = create(repo, tmp_path / "copy")
    with pytest.raises(TaskRowError, match="test_second.py"):
        build_row(work, task="Fix app", checks=checks)


@pytest.mark.parametrize(
    ("config_name", "config_text"),
    [
        ("pytest.ini", "[pytest]\npython_files = spec_*.py\ntestpaths = tests\n"),
        ("pyproject.toml",
         '[tool.pytest.ini_options]\npython_files = ["spec_*.py"]\ntestpaths = ["tests"]\n'),
        ("tox.ini", "[pytest]\npython_files = spec_*.py\ntestpaths = tests\n"),
        ("setup.cfg", "[tool:pytest]\npython_files = spec_*.py\ntestpaths = tests\n"),
    ],
    ids=["pytest-ini", "pyproject", "tox-ini", "setup-cfg"],
)
def test_configured_pytest_patterns_keep_pytest_runner(
    tmp_path, monkeypatch, config_name, config_text
):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / config_name).write_text(config_text, encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "spec_value.py").write_text(
        "def test_value(): assert False\n", encoding="utf-8")
    options = runner.RunOptions(
        repo, "Fix value", Endpoint("http://127.0.0.1:8080", "local"))
    monkeypatch.setattr(runner, "_check_engine_python", _stop_after_selection)

    with pytest.raises(_Selected):
        runner._run(options, time.monotonic(), printer=lambda _message: None)

    assert options.runner_kind == "pytest"
    assert options.check_source == "tests"
    assert options.checks == ["tests/spec_value.py"]
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix value", checks=options.checks)
    assert row["public_check_paths"] == ["tests/spec_value.py"]
    result = _run_pytest(work.copy, *row["public_check_paths"])
    assert result.returncode == 1


def test_explicit_pytest_selector_keeps_pytest_without_discovery_match(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "selected.py").write_text("def test_value(): assert False\n", encoding="utf-8")
    options = runner.RunOptions(
        repo, "Fix value", Endpoint("http://127.0.0.1:8080", "local"), checks=["selected.py"])
    monkeypatch.setattr(runner, "_check_engine_python", _stop_after_selection)

    with pytest.raises(_Selected):
        runner._run(options, time.monotonic(), printer=lambda _message: None)

    assert options.runner_kind == "pytest"
    assert options.check_source == "tests"
    assert options.checks == ["selected.py"]
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Fix value", checks=options.checks)
    assert row["public_check_paths"] == ["selected.py"]


def test_configured_pytest_receipt_is_test_evidence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npython_files = ["spec_*.py"]\n', encoding="utf-8")
    (repo / "spec_value.py").write_text("def test_value(): pass\n", encoding="utf-8")

    command = ("python", "-m", "pytest", "-q", "spec_value.py")
    result = _run_pytest(repo, "spec_value.py")
    assert result.returncode == 0
    assert verification_evidence_for_tests(
        "tests", "pytest", command, result.stdout,
        passed=True, declared=True, repo=repo,
    ) == ("tests", "verified by tests", True)
