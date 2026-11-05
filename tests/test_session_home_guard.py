"""The session-home guard sees powerup/skill state a test leaves behind, and nothing else."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

TESTS = Path(__file__).resolve().parent
# These independently inventoried subprocess outputs are visible to pytest callers.
_OFF_MODE_CHANNELS = ("return_code", "stdout", "stderr")


def _inner_environment(pytester, mode: str) -> None:
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        value for value in (str(pytester.path), str(TESTS), inherited_pythonpath) if value
    )
    pytester._monkeypatch.setenv("PYTHONPATH", pythonpath)
    pytester._monkeypatch.setenv("SF_GATE_MUST_RUN", mode)
    pytester._monkeypatch.delenv("SF_GATE_MANIFEST", raising=False)


def _run_inner(pytester, mode: str, *plugins: str):
    _inner_environment(pytester, mode)
    arguments = []
    for plugin in plugins:
        arguments.extend(("-p", plugin))
    arguments.extend(("-p", "no:cacheprovider", "-q", "-rA"))
    return pytester.runpytest_subprocess(*arguments)


def _make_inner_session(pytester) -> None:
    pytester.makepyfile(test_inner="def test_inner_passes():\n    assert True\n")
    assert not (pytester.path / "tests" / "must_run.json").exists()


def _session_channels(result) -> dict[str, object]:
    status = r"(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)"
    stdout = re.sub(
        rf"(?m)^(\d+ {status}(?:, \d+ {status})*) in \d+(?:\.\d+)?s$",
        r"\1",
        result.stdout.str(),
    )
    return {
        "return_code": result.ret,
        "stdout": stdout,
        "stderr": result.stderr.str(),
    }


def _assert_off_mode_inert(reference, candidate) -> None:
    reference_channels = _session_channels(reference)
    candidate_channels = _session_channels(candidate)
    scanned = tuple(
        channel for channel in _OFF_MODE_CHANNELS
        if channel in reference_channels and channel in candidate_channels
    )
    violated = tuple(
        channel for channel in scanned
        if reference_channels[channel] != candidate_channels[channel]
    )
    summary = (
        f"GUARD must_run_off_inertness: expected={len(_OFF_MODE_CHANNELS)} "
        f"scanned={len(scanned)} violated={len(violated)} "
        f"channels={','.join(scanned)}"
    )
    print(summary)
    assert len(_OFF_MODE_CHANNELS) == len(scanned), (
        f"{summary}; scanned channels={scanned}"
    )
    assert not violated, f"{summary}; changed channels={violated}"

INNER = '''
import os
from pathlib import Path


def test_leaves_a_skill_behind():
    skill = Path(os.environ["STATEFRAME_HOME"]) / "skills" / "leaked"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("x")


def test_uses_its_own_home(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    (tmp_path / "powerups.json").write_text("{}")


def test_touches_only_unrelated_home_files():
    (Path(os.environ["STATEFRAME_HOME"]) / "env-seeds").mkdir(exist_ok=True)
'''


def test_guard_names_a_leaking_test_and_spares_isolated_ones(pytester):
    pytester.makepyfile(test_inner=INNER)
    # Load the real tests/conftest.py as a plugin so its session home and guard run as in the suite.
    pytester._monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")]))
    pytester._monkeypatch.setenv("SF_GATE_MUST_RUN", "off")
    pytester._monkeypatch.delenv("SF_GATE_MANIFEST", raising=False)
    pytester._monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    result = pytester.runpytest_subprocess("-p", "conftest", "-p", "no:cacheprovider", "-q", "-rA")
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(["*test_leaves_a_skill_behind left session STATEFRAME_HOME state changed: skills*"])
    assert "ERROR test_inner.py::test_uses_its_own_home" not in result.stdout.str()
    assert "ERROR test_inner.py::test_touches_only_unrelated_home_files" not in result.stdout.str()
    assert "GUARD must_run" not in result.stdout.str()


def test_must_run_plugin_off_mode_is_inert_without_manifest(pytester):
    _make_inner_session(pytester)
    reference = _run_inner(pytester, "off")
    candidate = _run_inner(pytester, "off", "must_run_plugin")

    _assert_off_mode_inert(reference, candidate)
    candidate.assert_outcomes(passed=1)
    assert "GUARD must_run" not in candidate.stdout.str()


def test_must_run_plugin_on_mode_still_requires_manifest(pytester):
    _make_inner_session(pytester)
    result = _run_inner(pytester, "on", "must_run_plugin")

    output = result.stdout.str() + result.stderr.str()
    assert result.ret != 0
    assert "must-run manifest cannot be read:" in output
    assert "must_run.json" in output


def test_must_run_off_mode_guard_rejects_planted_manifest_read(pytester):
    _make_inner_session(pytester)
    reference = _run_inner(pytester, "off")
    pytester.makepyfile(planted_manifest_read="""
from pathlib import Path

import must_run_plugin


def pytest_configure(config):
    if not must_run_plugin._must_run_enabled():
        must_run_plugin._read_manifest(Path(str(config.rootpath)).resolve())
""")
    candidate = _run_inner(pytester, "off", "must_run_plugin", "planted_manifest_read")

    with pytest.raises(AssertionError, match="changed channels"):
        _assert_off_mode_inert(reference, candidate)
