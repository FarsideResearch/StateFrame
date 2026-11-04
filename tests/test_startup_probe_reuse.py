"""The test-interpreter probe is skipped only when the engine probe already proved pytest from that exact path."""
import sys

import pytest

from stateframe import runner
from stateframe.engine import kernel_test_isolation


@pytest.fixture
def probes(monkeypatch):
    calls = []

    def probe(python, code, *, label):
        calls.append((python, code, label))
        return b"[3, 12, 0]"
    monkeypatch.setattr(runner, "_startup_probe", probe)
    return calls


def options_for(tmp_path, *, runner_kind="pytest", test_python=sys.executable):
    options = runner.RunOptions(repo=tmp_path, task="test", endpoint=None, test_python=test_python)
    options.runner_kind = runner_kind
    return options


def test_engine_check_proves_pytest_then_test_probe_is_skipped(tmp_path, probes):
    options = options_for(tmp_path)
    runner._check_engine_python(options)
    assert options._pytest_proven == options.python
    assert [code for _, code, _ in probes] == ["import sys, json, pytest; print(json.dumps(list(sys.version_info[:3])))"]
    assert runner._select_test_python(options, printer=lambda _: None) == sys.executable
    assert len(probes) == 1


def test_generic_runner_proves_nothing_and_probes_nothing(tmp_path, probes):
    options = options_for(tmp_path, runner_kind="generic")
    runner._check_engine_python(options)
    assert options._pytest_proven is None
    assert "pytest" not in probes[0][1]
    runner._select_test_python(options, printer=lambda _: None)
    assert len(probes) == 1


def test_different_test_interpreter_path_is_still_probed(tmp_path, probes):
    # A symlink to the same binary is a different interpreter (its own pyvenv.cfg and site-packages).
    link = tmp_path / "bin" / "python3"
    link.parent.mkdir()
    link.symlink_to(sys.executable)
    options = options_for(tmp_path, test_python=str(link))
    runner._check_engine_python(options)
    selected = runner._select_test_python(options, printer=lambda _: None)
    assert selected == str(link) != options._pytest_proven
    assert [(python, code) for python, code, _ in probes[1:]] == [(str(link), "import pytest")]


def test_without_engine_proof_the_test_probe_runs(tmp_path, probes):
    options = options_for(tmp_path)
    assert options._pytest_proven is None
    runner._select_test_python(options, printer=lambda _: None)
    assert [code for _, code, _ in probes] == ["import pytest"]


def test_failed_engine_check_leaves_no_proof(tmp_path, monkeypatch):
    def broken(python, code, *, label):
        raise runner.RunError("engine interpreter cannot import pytest")
    monkeypatch.setattr(runner, "_startup_probe", broken)
    options = options_for(tmp_path)
    with pytest.raises(runner.RunError):
        runner._check_engine_python(options)
    assert options._pytest_proven is None


def test_missing_interpreter_inside_check_sandbox_has_relevant_startup_advice(
        tmp_path, monkeypatch, probes):
    link = tmp_path / "bin" / "python"
    link.parent.mkdir()
    link.symlink_to(sys.executable)
    message = (f"test interpreter {link} not found inside the check sandbox: this is an "
               "environment error, not a test result (no test ran)")
    monkeypatch.setattr(kernel_test_isolation, "check_sandbox_exec_denial",
                        lambda selected: message if selected == str(link) else None)
    options = options_for(tmp_path, test_python=str(link))

    with pytest.raises(runner.RunError) as error:
        runner._select_test_python(options, printer=lambda _line: None)

    assert str(error.value) == (
        f"test interpreter {link} (--test-python): {message}. Every check would fail to start; "
        "check that the interpreter and its links are available and executable inside the "
        "check sandbox, or report this as a StateFrame bug")
    assert [python for python, _, _ in probes] == [str(link)]
