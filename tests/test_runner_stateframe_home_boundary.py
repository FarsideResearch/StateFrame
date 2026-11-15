"""Runner refuses StateFrame homes nested in the edited repository or run tree."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import runner


@pytest.mark.parametrize(
    "layout",
    [
        "run",
        "repo",
        "run-path-symlink",
        "repo-path-symlink",
        "home-symlink-to-run",
        "home-symlink-to-repo",
    ],
)
def test_runner_refuses_stateframe_home_inside_run_or_repo(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "run"
    state_home: Path

    if layout == "run":
        state_home = run_dir / "state"
    elif layout == "repo":
        state_home = repo / "state"
    elif layout == "run-path-symlink":
        resolved_run = tmp_path / "resolved-run"
        resolved_run.mkdir()
        run_dir.symlink_to(resolved_run, target_is_directory=True)
        state_home = resolved_run / "state"
    elif layout == "repo-path-symlink":
        state_home = repo / "state"
        repo_alias = tmp_path / "repo-alias"
        repo_alias.symlink_to(repo, target_is_directory=True)
        repo = repo_alias
    elif layout == "home-symlink-to-run":
        state_home = run_dir / "state"
        state_home.mkdir(parents=True)
        home_alias = tmp_path / "state-home-alias"
        home_alias.symlink_to(state_home, target_is_directory=True)
        state_home = home_alias
    else:
        state_home = repo / "state"
        state_home.mkdir()
        home_alias = tmp_path / "state-home-alias"
        home_alias.symlink_to(state_home, target_is_directory=True)
        state_home = home_alias

    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(state_home))
    options = runner.RunOptions(repo=repo, task="", endpoint=None, run_dir=run_dir)

    with pytest.raises(runner.RunError) as raised:
        runner.run(options, printer=lambda _message: None)
    assert "STATEFRAME_HOME_BOUNDARY" in str(raised.value)


def test_stateframe_home_parent_of_run_remains_allowed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_home = tmp_path / ".stateframe"
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(state_home))
    options = runner.RunOptions(
        repo=repo, task="", endpoint=None, run_dir=state_home / "runs" / "one",
    )

    runner._validate_stateframe_home_boundary(options)


@pytest.mark.parametrize("source", ["profile", "extra-env"])
def test_runner_refuses_stateframe_home_from_run_environment_overlays(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "run"
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    options = runner.RunOptions(repo=repo, task="", endpoint=None, run_dir=run_dir)
    nested_home = str(run_dir / "state")
    if source == "profile":
        monkeypatch.setattr(runner.profiles, "load", lambda _profile: {"STATEFRAME_HOME": nested_home})
    else:
        options.extra_env["STATEFRAME_HOME"] = nested_home

    with pytest.raises(runner.RunError) as raised:
        runner.run(options, printer=lambda _message: None)
    assert "STATEFRAME_HOME_BOUNDARY" in str(raised.value)


def test_direct_prepare_establishes_startup_home_authority(tmp_path, monkeypatch):
    from stateframe.endpoint import Endpoint

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    home = tmp_path / "installation"
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    options = runner.RunOptions(
        repo=repo, task="inspect", endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None),
        run_dir=tmp_path / "run", tool_settings=runner.ToolSettings(shell="off", run_env=False))
    _run, _work, spec = runner.prepare(options, printer=lambda _message: None)
    assert options._stateframe_home_identity is not None
    from stateframe.engine import kernel_checkpoint

    assert kernel_checkpoint.recorded_home_identity(spec) == options._stateframe_home_identity
    assert "stateframe_home_identity" not in spec
    startup_pin = runner._child_env(options)[runner.RUNNER_HOME_IDENTITY_ENV]
    assert json.loads(startup_pin) == options._stateframe_home_identity
    spec["environment"][runner.RUNNER_HOME_IDENTITY_ENV] = "null"
    assert runner._child_env(options)[runner.RUNNER_HOME_IDENTITY_ENV] == startup_pin
    assert (home / "identity.key").is_file()


def test_resume_verifies_powerups_before_home_work(tmp_path, monkeypatch):
    from stateframe import powerups

    spec = {"powerups": {}}
    monkeypatch.setattr(runner.resume_module, "assess", lambda *_args, **_kwargs: {"spec": spec})

    def changed(_records):
        raise powerups.PowerupError("POWERUP_CHANGED")

    def unexpected_home_work(*_args, **_kwargs):
        pytest.fail("HOME validation ran before the powerup refusal")

    monkeypatch.setattr(powerups, "verify_records", changed)
    monkeypatch.setattr(runner, "_validate_stateframe_home_paths", unexpected_home_work)
    with pytest.raises(runner.resume_module.ResumeRefused, match="POWERUP_CHANGED"):
        runner.resume_run(tmp_path / "run")
    assert spec == {"powerups": {}}
