"""The interpreter startup probe retries once, with a doubled wall, only when it timed out."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateframe import runner
from stateframe.engine import kernel_safe_probe


def _result(*, timed_out=False, exit_code=0, output=b"[3, 12, 0]", truncated=False):
    return SimpleNamespace(timed_out=timed_out, exit_code=exit_code, output=output,
                           output_truncated=truncated)


@pytest.fixture
def probes(monkeypatch):
    """Replace the sandboxed probe with a script of results; record the wall each call was given."""
    calls, script = [], []
    monkeypatch.delenv("STATEFRAME_STARTUP_PROBE_SECONDS", raising=False)

    def fake(argv, workspace, *, timeout, **_kwargs):
        calls.append(timeout)
        return script.pop(0)

    monkeypatch.setattr(kernel_safe_probe, "run_probe", fake)
    return calls, script


def test_a_timeout_is_retried_once_with_double_the_wall(probes):
    calls, script = probes
    script += [_result(timed_out=True), _result()]
    assert runner._startup_probe("/usr/bin/python3", "import sys", label="engine interpreter x") == b"[3, 12, 0]"
    assert calls == [runner.STARTUP_PROBE_SECONDS, 2 * runner.STARTUP_PROBE_SECONDS]


def test_shell_setting_changes_both_probe_walls(probes, monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_STARTUP_PROBE_SECONDS", "2.5")
    calls, script = probes
    script += [_result(timed_out=True), _result()]
    runner._startup_probe("/usr/bin/python3", "import sys", label="engine interpreter x")
    assert calls == [2.5, 5.0]
    assert "STATEFRAME_STARTUP_PROBE_SECONDS" not in runner.ignored_shell_settings()
    options = runner.RunOptions(repo=tmp_path, task="test", endpoint=None)
    assert "STATEFRAME_STARTUP_PROBE_SECONDS" not in runner._child_env(options)


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf", "", "301", "1e308"])
def test_invalid_shell_setting_is_refused_before_probe(probes, monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_STARTUP_PROBE_SECONDS", value)
    calls, _script = probes
    with pytest.raises(runner.RunError, match="STATEFRAME_STARTUP_PROBE_SECONDS"):
        runner._startup_probe("/usr/bin/python3", "import sys", label="engine interpreter x")
    assert calls == []


def test_invalid_setting_is_refused_at_run_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_STARTUP_PROBE_SECONDS", "bad")
    options = runner.RunOptions(repo=tmp_path, task="test", endpoint=None)
    with pytest.raises(runner.RunError, match="STATEFRAME_STARTUP_PROBE_SECONDS"):
        runner.run(options, printer=lambda _line: None)


def test_two_timeouts_say_timed_out_under_load(probes):
    calls, script = probes
    script += [_result(timed_out=True), _result(timed_out=True)]
    with pytest.raises(runner.RunError) as error:
        runner._startup_probe("/usr/bin/python3", "import sys", label="engine interpreter x")
    message = str(error.value)
    assert "timed out under load" in message
    assert f"{runner.STARTUP_PROBE_SECONDS} s and then {2 * runner.STARTUP_PROBE_SECONDS} s" in message
    assert "broken" not in message
    assert len(calls) == 2


@pytest.mark.parametrize("result", [_result(exit_code=1, output=b"ModuleNotFoundError: No module named 'pytest'"),
                                    _result(exit_code=127, output=b""),
                                    _result(truncated=True)])
def test_a_real_failure_is_not_retried_and_is_not_called_a_timeout(probes, result):
    calls, script = probes
    script += [result, _result()]
    with pytest.raises(runner.RunError) as error:
        runner._startup_probe("/usr/bin/python3", "import pytest", label="test interpreter x")
    assert len(calls) == 1
    assert "timed out" not in str(error.value)


def test_a_failure_after_a_timeout_reports_the_failure(probes):
    calls, script = probes
    script += [_result(timed_out=True), _result(exit_code=1, output=b"boom")]
    with pytest.raises(runner.RunError, match="interpreter is broken.*boom"):
        runner._startup_probe("/usr/bin/python3", "import sys", label="engine interpreter x")
    assert len(calls) == 2


def test_a_nonzero_exit_says_the_interpreter_is_broken(probes):
    _calls, script = probes
    script += [_result(exit_code=1, output=b"boom")]
    with pytest.raises(runner.RunError, match="interpreter is broken"):
        runner._startup_probe("/usr/bin/python3", "import sys", label="engine interpreter x")
