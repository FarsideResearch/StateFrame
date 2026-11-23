from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from stateframe.test_runners import (command_runs_tests, declared_test_evidence,
                                    discover_check, parse_command,
                                    verification_evidence_for_tests)


NODE_PASS = (
    "# tests 1\n"
    "# pass 1\n"
    "# fail 0\n"
    "# cancelled 0\n"
    "# skipped 0\n"
    "# todo 0\n"
    "# duration_ms 1\n"
)


def _write_package(repo, *, script="node --test", package_manager=None):
    repo.mkdir(parents=True, exist_ok=True)
    package = {"scripts": {"test": script}}
    if package_manager is not None:
        package["packageManager"] = package_manager
    (repo / "package.json").write_text(json.dumps(package), encoding="utf-8")


def _write_node_test(repo, relative="tests/example.test.js"):
    test_file = repo / relative
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "require('node:test')('works', () => require('node:assert/strict').equal(1, 1));\n",
        encoding="utf-8",
    )


def test_discovers_direct_node_script_and_certifies_positive_receipt(tmp_path):
    repo = tmp_path / "repo"
    _write_package(repo)
    _write_node_test(repo)

    command, source, found = discover_check(repo)

    assert command == "node --test"
    assert found
    kind, argv = parse_command(command)
    evidence = verification_evidence_for_tests(
        source, kind, argv, NODE_PASS, passed=True,
        declared=declared_test_evidence(command, None, source, kind), repo=repo)
    assert evidence == ("tests", "verified by tests", True)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_discovery_accepts_supported_separate_reporter_argument(tmp_path):
    repo = tmp_path / "repo"
    script = "node --test --test-reporter tap"
    _write_package(repo, script=script)
    _write_node_test(repo)

    run = subprocess.run(shlex.split(script), cwd=repo, capture_output=True,
                         text=True, check=False)
    assert run.returncode == 0, run.stderr

    command, source, found = discover_check(repo)
    assert found
    assert command == script
    kind, argv = parse_command(command)
    assert verification_evidence_for_tests(
        source, kind, argv, run.stdout, passed=True, declared=True, repo=repo
    ) == ("tests", "verified by tests", True)


def test_does_not_discover_node_script_without_test_files(tmp_path):
    repo = tmp_path / "repo"
    _write_package(repo)

    assert discover_check(repo) == (None, "none", False)


def test_does_not_use_unsupported_package_manager_for_node_script(tmp_path):
    repo = tmp_path / "repo"
    _write_package(repo, package_manager="bun@1.2.0")
    (repo / "bun.lock").write_text("", encoding="utf-8")
    _write_node_test(repo)

    assert discover_check(repo) == (None, "none", False)


def test_does_not_discover_shell_compound_package_script(tmp_path):
    repo = tmp_path / "repo"
    _write_package(repo, script="node --test && echo complete")
    _write_node_test(repo)

    assert discover_check(repo) == (None, "none", False)


@pytest.mark.parametrize("script", [
    "node --test\nprintf complete",
    "env NODE_OPTIONS=--no-warnings node --test",
    "node tests/example.test.js --test",
    "node --test --help",
    "node --test ../example.test.js",
])
def test_only_direct_runnable_node_scripts_are_discovered(tmp_path, script):
    _write_package(tmp_path, script=script)
    _write_node_test(tmp_path)

    assert discover_check(tmp_path) == (None, "none", False)


@pytest.mark.parametrize("relative", ["tests/helper.js", "example.spec.js"])
def test_implicit_node_check_requires_a_file_node_discovers(tmp_path, relative):
    _write_package(tmp_path)
    _write_node_test(tmp_path, relative)

    assert discover_check(tmp_path) == (None, "none", False)


@pytest.mark.parametrize("relative", ["test/test.js", "test/example-test.cjs", "example.test.cjs"])
def test_discovery_recognizes_protected_node_default_file_names(tmp_path, relative):
    _write_package(tmp_path)
    _write_node_test(tmp_path, relative)

    command, source, found = discover_check(tmp_path)

    assert found
    kind, argv = parse_command(command)
    assert verification_evidence_for_tests(
        source, kind, argv, NODE_PASS, passed=True, declared=True, repo=tmp_path
    ) == ("tests", "verified by tests", True)


@pytest.mark.parametrize(("script", "unprotected"), [
    ("node --test", "test.js"),
    ("node --test", "test.ts"),
    ("node --test verify.js", "verify.js"),
])
def test_discovery_does_not_select_tests_outside_the_protected_inventory(
        tmp_path, script, unprotected):
    _write_package(tmp_path, script=script)
    _write_node_test(tmp_path)
    _write_node_test(tmp_path, unprotected)

    assert discover_check(tmp_path) == (None, "none", False)


def test_existing_make_check_keeps_precedence(tmp_path):
    _write_package(tmp_path)
    _write_node_test(tmp_path)
    (tmp_path / "Makefile").write_text("check:\n\t@true\n", encoding="utf-8")

    assert discover_check(tmp_path) == ("make -f Makefile check", "build", False)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_discovered_node_command_executes_quoted_file_without_package_manager(tmp_path, monkeypatch):
    node = Path(shutil.which("node")).resolve()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "node").symlink_to(node)
    monkeypatch.setenv("PATH", str(bin_dir))
    repo = tmp_path / "repo"
    relative = "tests/sample case.spec.js"
    _write_package(repo, script=f'node --test "{relative}"')
    _write_node_test(repo, relative)

    command, source, found = discover_check(repo)
    assert found
    kind, argv = parse_command(command)
    run = subprocess.run(argv, cwd=repo, capture_output=True, text=True, check=False)

    assert run.returncode == 0, run.stderr
    assert verification_evidence_for_tests(
        source, kind, argv, run.stdout, passed=True, declared=True, repo=repo
    ) == ("tests", "verified by tests", True)


def test_malformed_package_cannot_certify_node_receipt(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text("{ malformed", encoding="utf-8")
    _write_node_test(repo)

    with pytest.raises(ValueError, match=r"cannot inspect package\.json scripts"):
        discover_check(repo)
    assert not command_runs_tests(
        "generic", ("npm", "test"), set(), output=NODE_PASS, repo=repo)


def test_python_tests_keep_precedence_over_node_package_script(tmp_path):
    repo = tmp_path / "repo"
    _write_package(repo)
    _write_node_test(repo)
    python_test = repo / "tests" / "test_python.py"
    python_test.write_text("def test_value():\n    assert True\n", encoding="utf-8")

    assert discover_check(repo) == (None, "tests", True)


@pytest.mark.parametrize("receipt", [
    "",
    "# tests 1\n# pass 0\n# fail 1\n",
    "# tests 0\n# pass 0\n# fail 0\n",
    "# pass 1\n",
    "# pass 1\n# fail 0\n",
    "# tests 0\n# pass 1\n# fail 0\n",
])
def test_empty_failing_or_nonexecuted_receipt_does_not_verify(tmp_path, receipt):
    repo = tmp_path / "repo"
    _write_package(repo)
    _write_node_test(repo)

    command, source, _ = discover_check(repo)
    kind, argv = parse_command(command)
    evidence = verification_evidence_for_tests(
        source, kind, argv, receipt, passed=True, declared=True, repo=repo)

    assert evidence == ("command", None, False)
