from __future__ import annotations

import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import re
from pathlib import Path

import pytest

from stateframe import outer_checks

ROOT = Path(__file__).resolve().parents[1]


def _console_environment(root):
    """Create an environment with a stock pytest console script beside its interpreter."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True)
    interpreter = root / "bin" / "python"
    site = subprocess.run(
        [str(interpreter), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # The check sandbox reads only this environment, so it must hold pytest itself.
    for name in ("pytest", "_pytest", "py", "pluggy", "iniconfig", "packaging", "pygments"):
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        origin = Path(spec.origin)
        if spec.submodule_search_locations:
            shutil.copytree(origin.parent, Path(site, name))
        else:
            shutil.copy2(origin, site)
    script = root / "bin" / "pytest"
    script.write_text(
        f"#!{interpreter}\nimport sys\nfrom pytest import console_main\n"
        "if __name__ == '__main__':\n    sys.exit(console_main())\n", encoding="utf-8")
    script.chmod(0o755)
    return interpreter


def _check(repo, *args, launch=None, capture_output=False, interpreter=None):
    python = str(interpreter or sys.executable)
    return outer_checks.run(
        shlex.join([*(launch or [sys.executable, "-m", "pytest"]), "-q", *args]),
        repo,
        python=python,
        test_python=python,
        runner_kind="pytest",
        capture_output=capture_output,
        required_test_ledger=True,
    )


def _normalized_output(value: bytes) -> bytes:
    return re.sub(rb"\bin \d+(?:\.\d+)?s\b", b"in <duration>s", value)


def _snapshot_check_artifacts(repo):
    files = {}
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo)
        if relative.parts[:1] == (".test-tmp",) and path.name.startswith(".stateframe-pytest-"):
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _run_direct_check_pair(tmp_path, files, pytest_args, expected_reason):
    source = tmp_path / "source"
    source.mkdir()
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    repo = tmp_path / "checked-repo"
    outputs = []
    report_reason = None
    for required in (False, True):
        if repo.exists():
            shutil.rmtree(repo)
        shutil.copytree(source, repo)
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--basetemp=.test-tmp/pytest-base", *pytest_args]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if required:
            report_directory = repo / ".test-tmp"
            report_directory.mkdir(exist_ok=True)
            report_path = report_directory / ".stateframe-pytest-report"
            report_path.write_bytes(b"")
            wrapped = outer_checks._pytest_ledger_command(command, repo, report_path, environment)
            assert wrapped is not None
            command = wrapped
        completed = subprocess.run(command, cwd=repo, env=environment, capture_output=True,
                                   timeout=60, check=False)
        outputs.append((completed.stdout, completed.stderr, completed.returncode,
                        _snapshot_check_artifacts(repo)))
        if required:
            from kernel_required_tests import (PYTEST_REPORT_NOT_PRODUCED,
                                               parse_pytest_report_with_reason)
            if report_path.is_file():
                _, report_reason = parse_pytest_report_with_reason(
                    report_path.read_bytes(), completed.returncode)
            else:
                report_reason = PYTEST_REPORT_NOT_PRODUCED

    plain, observed = outputs
    assert (_normalized_output(observed[0]), _normalized_output(observed[1]),
            observed[2], observed[3]) == (
        _normalized_output(plain[0]), _normalized_output(plain[1]), plain[2], plain[3],
    )
    assert report_reason == expected_reason

    from kernel_required_tests import RequiredTests
    from stateframe.runner import _observe_required_test_run, _required_test_ledger_summary

    ledger = RequiredTests()
    reasons = {}
    incomplete = {"complete": False, "outcomes": {}}
    _, observation_reason = _observe_required_test_run(
        ledger, "baseline", {
            "required_test_report": incomplete,
            "required_test_report_reason": report_reason,
        }, reasons=reasons)
    assert observation_reason == expected_reason
    summary = _required_test_ledger_summary(ledger, reasons)
    assert summary["state"] == "unknown"
    assert summary["reason"] == f"baseline: {expected_reason}"


@pytest.mark.parametrize(("name", "test_source", "pytest_args", "reason"), [
    ("fail-fast",
     "def test_first():\n    assert False\n\ndef test_second():\n    pass\n",
     ["-x", "test_signal.py"], "pytest did not finish every collected test"),
    ("collect-only", "def test_signal():\n    pass\n",
     ["--collect-only", "test_signal.py"], "pytest was run in collection-only mode"),
    ("import-failure", "def test_signal():\n    pass\n",
     ["test_signal.py"], "pytest could not be imported"),
    ("oversized", "import pytest\n\n"
     "@pytest.mark.parametrize('value', range(1700), ids=lambda _: 'x' * 1000)\n"
     "def test_signal(value):\n    pass\n",
     ["test_signal.py"], "pytest report exceeded its size bound"),
    ("unfinished", "import os\n\ndef test_signal():\n    os._exit(7)\n",
     ["test_signal.py"], "pytest report was not produced"),
    ("write-failure", "from pathlib import Path\n\ndef test_signal():\n"
     "    reports = list((Path.cwd() / '.test-tmp').glob('.stateframe-pytest-*'))\n"
     "    if reports:\n        reports[0].unlink()\n        reports[0].mkdir()\n",
     ["test_signal.py"], "pytest report was not produced"),
])
@pytest.mark.host_sandbox
def test_incomplete_ledger_reasons_do_not_change_check_streams_or_artifacts(
        tmp_path, name, test_source, pytest_args, reason):
    files = {
        "conftest.py": "from pathlib import Path\n\ndef pytest_sessionfinish(session, exitstatus):\n"
        "    Path('check-artifact.txt').write_text(str(exitstatus), encoding='utf-8')\n",
        "test_signal.py": test_source,
    }
    if name == "import-failure":
        files["pytest.py"] = "raise RuntimeError('synthetic pytest import failure')\n"
    _run_direct_check_pair(tmp_path, files, pytest_args, reason)


def _mutated_runner_function(tmp_path, old, new, name):
    from stateframe import runner

    source = ROOT / "src/stateframe/runner.py"
    text = source.read_text(encoding="utf-8")
    assert text.count(old) == 1
    copied = tmp_path / f"runner_mutated_{name}.py"
    copied.write_text(text.replace(old, new, 1), encoding="utf-8")
    tree = ast.parse(copied.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        "_required_test_ledger_reason": runner._required_test_ledger_reason,
        "_required_test_has_observation": runner._required_test_has_observation,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(copied), "exec"), namespace)
    return namespace[name]


def test_mutation_proofs_reject_stderr_stale_reason_and_dropped_ledger(tmp_path, monkeypatch):
    source = ROOT / "src/stateframe/engine/kernel_required_tests.py"
    original = source.read_text(encoding="utf-8")
    needle = "def write_incomplete(reason):\n"
    assert original.count(needle) == 1
    copied = tmp_path / "kernel_required_tests_stderr_mutation.py"
    copied.write_text(original.replace(
        needle, needle + "    print('ledger diagnostic', file=sys.stderr, flush=True)\n", 1),
        encoding="utf-8")
    spec = importlib.util.spec_from_file_location("kernel_required_tests_stderr_mutation", copied)
    assert spec is not None and spec.loader is not None
    mutated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated_module)
    monkeypatch.setitem(sys.modules, "kernel_required_tests", mutated_module)
    parity_root = tmp_path / "stderr-parity"
    parity_root.mkdir()
    with pytest.raises(AssertionError):
        _run_direct_check_pair(
            parity_root, {"test_signal.py": "def test_signal():\n    pass\n"},
            ["--collect-only", "test_signal.py"], "pytest was run in collection-only mode")
    monkeypatch.delitem(sys.modules, "kernel_required_tests")

    stale = _mutated_runner_function(
        tmp_path,
        "        if complete or _required_test_has_observation(ledger, kind):\n"
        "            reasons.pop(kind, None)\n",
        "        if False:\n            reasons.pop(kind, None)\n",
        "_observe_required_test_run")
    from kernel_required_tests import RequiredTests

    stale_reasons = {"baseline": "old incomplete report"}
    stale(RequiredTests(), "baseline", {"required_test_report": {
        "complete": True, "outcomes": {"case-a": "passed"},
    }}, reasons=stale_reasons)
    with pytest.raises(AssertionError):
        assert stale_reasons == {}

    dropped = _mutated_runner_function(
        tmp_path,
        "                reasons[kind] = reason\n"
        "        return ledger, reason\n"
        "    if reasons is not None:\n",
        "                reasons[kind] = reason\n"
        "        return None, reason\n"
        "    if reasons is not None:\n",
        "_observe_required_test_run")

    class RejectedLedger:
        def observe(self, *_args, **_kwargs):
            raise ValueError("synthetic rejected observation")

    preserved = RejectedLedger()
    result, _ = dropped(preserved, "current", {"required_test_report": {
        "complete": True, "outcomes": {},
    }})
    with pytest.raises(AssertionError):
        assert result is preserved


@pytest.mark.host_sandbox
def test_dot_in_filename_does_not_alias_directory_in_pytest_node_ids(tmp_path):
    repo = tmp_path / "repo"
    for name in ("tests/test_group/test_value.py", "tests/test_group.test_value.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("class TestCase:\n    def test_value(self):\n        pass\n")

    result = _check(repo, "--import-mode=importlib", "tests")

    assert result["passed"] is True
    assert result["required_test_report"] == {"complete": True, "outcomes": {
        "tests/test_group/test_value.py::TestCase::test_value": "passed",
        "tests/test_group.test_value.py::TestCase::test_value": "passed",
    }}


@pytest.mark.host_sandbox
def test_pytest_rootdir_does_not_alias_tests_from_different_directories(tmp_path):
    repo = tmp_path / "repo"
    for directory in ("left", "right"):
        path = repo / directory / "test_value.py"
        path.parent.mkdir(parents=True)
        path.write_text("def test_value():\n    pass\n")

    for directory in ("left", "right"):
        result = _check(repo, f"--rootdir={directory}", f"{directory}/test_value.py")

        assert result["passed"] is True
        assert result["required_test_report"] == {"complete": True, "outcomes": {
            f"{directory}/test_value.py::test_value": "passed",
        }}


@pytest.mark.host_sandbox
def test_ledger_preserves_check_with_junit_plugin_disabled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_value.py").write_text("def test_value():\n    pass\n")

    result = _check(repo, "-p", "no:junitxml", "test_value.py")

    assert result["passed"] is True, result["tail"]
    assert result["required_test_report"] == {"complete": True, "outcomes": {
        "test_value.py::test_value": "passed",
    }}


@pytest.mark.host_sandbox
def test_fail_fast_is_an_incomplete_pytest_run(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_values.py").write_text(
        "def test_first():\n    assert False\n\ndef test_second():\n    pass\n")

    result = _check(repo, "-x", "test_values.py")

    assert result["exit_code"] == 1
    assert result["required_test_report"] == {"complete": False, "outcomes": {}}
    assert result["required_test_report_reason"] == "pytest did not finish every collected test"


@pytest.mark.host_sandbox
def test_complete_empty_pytest_run_can_prove_last_test_was_removed(tmp_path):
    from kernel_required_tests import RequiredTests

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "test_value.py"
    source.write_text("def test_value():\n    pass\n")
    ledger = RequiredTests()
    baseline = _check(repo, "test_value.py")["required_test_report"]
    ledger.observe("baseline", **baseline)

    source.write_text("def helper():\n    pass\n")
    after = _check(repo, "test_value.py")
    ledger.observe("current", **after["required_test_report"])

    assert after["exit_code"] == 5
    assert ledger.report() == {
        "state": "regressed", "missing": ["test_value.py::test_value"], "passed": [], "failed": [],
    }


@pytest.mark.host_sandbox
@pytest.mark.parametrize("wrapper", [False, True], ids=["ordinary-hook", "tryfirst-wrapper"])
def test_pytest_unconfigure_cannot_forge_a_passing_ledger_report(tmp_path, wrapper):
    repo = tmp_path / "repo"
    repo.mkdir()
    decorator = ("@pytest.hookimpl(hookwrapper=True, tryfirst=True)\n"
                 if wrapper else "@pytest.hookimpl(trylast=True)\n")
    yield_hook = "    yield\n" if wrapper else ""
    (repo / "conftest.py").write_text(
        "import json\n"
        "import pytest\n"
        "from pathlib import Path\n\n"
        "import sys\n\n"
        + decorator
        + "def pytest_unconfigure(config):\n"
        + yield_hook
        + "    report = Path(sys.modules['__main__'].report_path)\n"
        "    try:\n"
        "        report.write_text(json.dumps({\n"
        "            'exit_code': 0,\n"
        "            'outcomes': {'test_value.py::test_value': 'passed'},\n"
        "        }))\n"
        "    except OSError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    (repo / "test_value.py").write_text(
        "import pytest\n\n"
        "def test_value():\n"
        "    pytest.skip('intentionally skipped')\n",
        encoding="utf-8",
    )

    result = _check(repo, "test_value.py")

    assert result["exit_code"] == 0
    assert result["required_test_report"] == {
        "complete": True,
        "outcomes": {"test_value.py::test_value": "skipped"},
    }, result


@pytest.mark.host_sandbox
def test_large_ledger_cannot_stop_the_check(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_values.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('value', range(2350), ids=lambda i: 'a' * 600 + str(i))\n"
        "def test_value(value):\n    if value == 0:\n        print('a' * 220000)\n")

    result = _check(repo, "-s", "test_values.py", capture_output=True)

    assert result["passed"] is True
    assert result["output_bytes"] > 220000
    assert result["required_test_report"] == {"complete": False, "outcomes": {}}


@pytest.mark.host_sandbox
def test_console_script_of_another_interpreter_reports_unknown(tmp_path, monkeypatch):
    interpreter = _console_environment(tmp_path / "environment")
    other = _console_environment(tmp_path / "other")
    (interpreter.parent / "pytest").unlink()
    monkeypatch.setenv("PATH", f"{other.parent}{os.pathsep}{os.environ['PATH']}")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_value.py").write_text("def test_value():\n    assert 7 == 7\n", encoding="utf-8")
    result = _check(repo, "test_value.py", launch=["pytest"], interpreter=interpreter)
    assert result["passed"] is True
    assert result["required_test_report"] == {"complete": False, "outcomes": {}}
    assert result["required_test_report_reason"]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("launch", [
    [sys.executable, "-m", "pytest"],
    [sys.executable, "-I", "-m", "pytest"],
    ["pytest"],
])
def test_outer_check_returns_structured_pytest_case_ids(tmp_path, launch, monkeypatch):
    interpreter = None
    if launch == ["pytest"]:
        # The interpreter running this suite may have no console script beside it.
        interpreter = _console_environment(tmp_path / "environment")
        monkeypatch.setenv("PATH", f"{interpreter.parent}{os.pathsep}{os.environ['PATH']}")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_value.py").write_text(
        "def test_value_is_present():\n    assert 7 == 7\n",
        encoding="utf-8",
    )
    result = _check(repo, "test_value.py", launch=launch, interpreter=interpreter)

    report = result.get("required_test_report")
    assert result["passed"] is True
    assert isinstance(report, dict), result
    assert report["complete"] is True, {
        "report": report,
        "exit_code": result.get("exit_code"),
        "tail": result.get("tail"),
    }
    outcomes = report["outcomes"]
    assert isinstance(outcomes, dict)
    assert outcomes == {"test_value.py::test_value_is_present": "passed"}
