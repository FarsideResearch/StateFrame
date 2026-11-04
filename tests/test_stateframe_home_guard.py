"""Test subprocesses never use the developer's real ~/.stateframe (11-03: 52 GB of env seeds in a day)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import _stateframe_home
from _stateframe_home import isolated_stateframe_home

TESTS = Path(__file__).resolve().parent
_STRIP = re.compile(r"""not\s+\w+\.startswith\(\s*["']STATEFRAME_["']\s*\)""")


def _stripping_helpers() -> list[Path]:
    return sorted(path for path in TESTS.rglob("*.py")
                  if path.name != Path(__file__).name and _STRIP.search(path.read_text(errors="replace")))


def test_every_env_helper_that_strips_stateframe_vars_restores_an_isolated_home():
    helpers = _stripping_helpers()
    assert helpers, "pattern no longer matches anything; update _STRIP rather than letting this guard go vacuous"
    missing = [str(path.relative_to(TESTS)) for path in helpers
               if "isolated_stateframe_home(" not in path.read_text(errors="replace")]
    assert missing == [], f"strip STATEFRAME_* but never set STATEFRAME_HOME=isolated_stateframe_home(): {missing}"


def test_planted_helper_without_home_is_caught(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text('env = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}\n')
    assert _STRIP.search(planted.read_text())
    assert "isolated_stateframe_home(" not in planted.read_text()


@pytest.fixture
def fresh_fallback(monkeypatch):
    monkeypatch.setattr(_stateframe_home, "_fallback", None)


def _real() -> Path:
    return (Path.home() / ".stateframe").resolve()


@pytest.mark.parametrize("value", [None, "~/.stateframe", "real"])
def test_real_or_missing_home_is_replaced_by_a_private_one(monkeypatch, fresh_fallback, value):
    if value is None:
        monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_HOME", str(_real()) if value == "real" else value)
    home = Path(isolated_stateframe_home())
    assert home.resolve() != _real()
    assert home.is_dir()
    assert isolated_stateframe_home() == str(home)  # one private home per process, not one per call


def test_an_already_isolated_home_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    assert isolated_stateframe_home() == str(tmp_path)


def test_acceptance_environment_never_points_at_the_real_home(monkeypatch, fresh_fallback):
    from acceptance._support import environment

    monkeypatch.setenv("STATEFRAME_HOME", str(_real()))
    env = environment()
    assert Path(env["STATEFRAME_HOME"]).resolve() != _real()
