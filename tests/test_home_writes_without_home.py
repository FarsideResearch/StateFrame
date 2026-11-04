"""Writing commands keep the clear no-home error, and a no-home run writes nothing to a guessed home."""
import pwd
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import powerups, storage_gc
from stateframe.powerup_manifest import PowerupError
from stateframe.powerup_prepare import prepare_run_inputs


@pytest.fixture
def no_home(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)


def test_install_without_home_is_the_clear_error(no_home):
    source = Path(__file__).resolve().parents[1] / "examples" / "powerups" / "repo-grep"
    with pytest.raises(PowerupError, match="set HOME or STATEFRAME_HOME"):
        powerups.install(source)


def test_enable_without_home_is_the_clear_error(no_home):
    with pytest.raises(PowerupError, match="set HOME or STATEFRAME_HOME"):
        powerups.enable("sample")


def test_gc_without_home_is_the_clear_error(no_home):
    with pytest.raises(ValueError, match="set HOME or STATEFRAME_HOME"):
        storage_gc.run_gc()


def test_nothing_is_written_to_a_guessed_home(no_home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = SimpleNamespace(no_powerups=False, repo=tmp_path, powerup_requests=(), powerup_sets=(), skills=())
    prepare_run_inputs(options, {}, printer=lambda _line: None)
    powerups.resolve_units(tmp_path, "", "default", (), (), False)
    assert not any(path.name == ".stateframe" for path in tmp_path.rglob("*"))
