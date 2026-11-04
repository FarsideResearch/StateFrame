"""The session-home guard sees powerup/skill state a test leaves behind, and nothing else."""
from __future__ import annotations

import os
from pathlib import Path

pytest_plugins = ["pytester"]

TESTS = Path(__file__).resolve().parent

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
    pytester._monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    result = pytester.runpytest_subprocess("-p", "conftest", "-p", "no:cacheprovider", "-q", "-rA")
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(["*test_leaves_a_skill_behind left session STATEFRAME_HOME state changed: skills*"])
    assert "ERROR test_inner.py::test_uses_its_own_home" not in result.stdout.str()
    assert "ERROR test_inner.py::test_touches_only_unrelated_home_files" not in result.stdout.str()
