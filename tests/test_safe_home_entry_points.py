"""Missing-home behavior at powerup, storage, rehydrate, and seed entry points."""
from __future__ import annotations

import pwd
from pathlib import Path

import pytest

from stateframe import powerup_state, rehydrate, storage_gc
from stateframe.engine import kernel_env_seed
from stateframe.engine.kernel_run_env import create_run_env
from stateframe.powerup_manifest import PowerupError
from test_env_seed_store import _source


@pytest.fixture
def no_home(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)

    def missing_passwd_entry(_uid):
        raise KeyError("no passwd entry")

    monkeypatch.setattr(pwd, "getpwuid", missing_passwd_entry)


def test_powerup_home_without_home(no_home):
    with pytest.raises(PowerupError, match=r"HOME or STATEFRAME_HOME"):
        powerup_state.home()


def test_powerup_home_with_home(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    assert powerup_state.home() == Path(home) / ".stateframe"


def test_default_runs_home_without_home_matches_the_runner(no_home):
    from stateframe import runner

    expected = runner.default_run_dir(Path("repo")).parent
    assert storage_gc.default_runs_home() == expected


def test_default_runs_home_with_home(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "override"))
    assert storage_gc.default_runs_home() == Path(home) / ".stateframe" / "runs"


def test_stateframe_home_without_home(no_home):
    with pytest.raises(ValueError, match=r"HOME or STATEFRAME_HOME"):
        storage_gc.stateframe_home()


def test_stateframe_home_with_home(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    assert storage_gc.stateframe_home() == Path(home) / ".stateframe"


def test_wheel_directories_without_home(no_home, tmp_path):
    with pytest.raises(rehydrate.RehydrateError, match=r"HOME or STATEFRAME_HOME"):
        rehydrate._wheel_directories(tmp_path / "run", {})


def test_wheel_directories_with_home(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    cache = Path(home) / ".cache" / "pip"
    cache.mkdir(parents=True)
    (cache / "package.whl").write_bytes(b"wheel")
    assert rehydrate._wheel_directories(tmp_path / "run", {}) == [cache]


def test_env_seed_without_home_falls_back_to_copy(no_home, tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_ENV_SEED", "auto")
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    result = create_run_env(run_dir, source / "bin/python")
    assert result.kind == "clone"
    assert (run_dir / "env" / "bin" / "python").exists()
    assert not (run_dir / "env-seed.json").exists()


def test_env_seed_with_home(monkeypatch, tmp_path):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    source = _source(tmp_path)
    seed = kernel_env_seed.clone(source, tmp_path / "clone", source / "bin/python", None, ())
    assert seed.parent == Path(home) / ".stateframe" / "env-seeds"


def test_stateframe_home_override_without_user_home(no_home, monkeypatch, tmp_path):
    configured = tmp_path / "stateframe"
    monkeypatch.setenv("STATEFRAME_HOME", str(configured))
    assert powerup_state.home() == configured
    assert storage_gc.stateframe_home() == configured
    assert rehydrate._wheel_directories(tmp_path / "run", {}) == []
