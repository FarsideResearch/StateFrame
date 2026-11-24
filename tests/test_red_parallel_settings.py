"""Reproduce profile parallel settings rejected by the public CLI path."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stateframe import cli, profiles, runner
from stateframe.engine import kernel_parallel_records as parallel_records


PROFILE_SETTINGS = (
    ("STATEFRAME_PARALLEL_INSPECT_SECONDS", "17.5"),
    ("STATEFRAME_PARALLEL_CLEANUP_SECONDS", "18.5"),
    ("STATEFRAME_PARALLEL_WATCHDOG_GRACE_SECONDS", "19.5"),
    ("STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS", "20.5"),
    ("STATEFRAME_PARALLEL_PREPARE_SECONDS", "21.5"),
    ("STATEFRAME_PARALLEL_FANOUT_CONTINUATION", "off"),
)
PARALLEL_PROFILE_SWITCHES = (*runner.PARALLEL_TIMING_SETTINGS,
                             "STATEFRAME_PARALLEL_FANOUT_CONTINUATION")
PROFILE_VALIDATION_VALUES = (
    "", "-1", "0", "text", "nan", "inf", "1e100", True, False,
    "2.5", "12.5", " ON ", "off",
)


def _run_cli_through_prepare(
        tmp_path: Path, monkeypatch, capsys, setting: tuple[str, str] | None):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text("def test_sample():\n    assert True\n", encoding="utf-8")

    switches = profiles.load(profiles.DEFAULT)
    expected = None
    if setting is not None:
        key, expected = setting
        switches[key] = expected
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}),
                       encoding="utf-8")

    for name in tuple(os.environ):
        if name.startswith("STATEFRAME_PARALLEL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    prepared: dict[str, object] = {}

    def prepare_only(options, printer=print):
        _run_dir, _work, spec = runner.prepare(options, printer=lambda *_: None)
        prepared["environment"] = spec["environment"]
        prepared["parallel_limits"] = spec.get("parallel_limits")
        return {"exit_code": 0}

    # Keep the public CLI parser/error boundary and the real preparation path, while avoiding
    # provider traffic and engine execution unrelated to profile validation.
    monkeypatch.setattr(cli, "run", prepare_only)
    args = [
        "run", "--repo", str(repo), "--task", "Inspect sample.py",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "local-model",
        "--profile", str(profile), "--run-dir", str(tmp_path / "run"),
        "--max-turns", "1", "--parallel", "2", "--json",
    ]
    exit_code = cli.main(args)
    output = capsys.readouterr()
    return exit_code, output.out + output.err, prepared, expected


@pytest.mark.parametrize("setting", PROFILE_SETTINGS)
def test_cli_parallel_two_accepts_profile_parallel_settings(tmp_path, monkeypatch, capsys, setting):
    exit_code, output, prepared, expected = _run_cli_through_prepare(
        tmp_path, monkeypatch, capsys, setting)

    assert exit_code == 0, output
    key, _value = setting
    assert prepared["environment"][key] == expected
    assert prepared["parallel_limits"]["max_concurrency"] >= 2


def test_cli_parallel_two_without_special_profile_settings_stays_green(
        tmp_path, monkeypatch, capsys):
    exit_code, output, prepared, _expected = _run_cli_through_prepare(
        tmp_path, monkeypatch, capsys, None)

    assert exit_code == 0, output
    assert prepared["parallel_limits"]["max_concurrency"] >= 2


def test_profile_parallel_tuning_switches_match_runner_and_engine():
    registered = {
        switch for switch in parallel_records.PARALLEL_TUNING_SWITCHES
        if switch.endswith("_SECONDS") or switch == "STATEFRAME_PARALLEL_FANOUT_CONTINUATION"
    }
    assert registered == set(PARALLEL_PROFILE_SWITCHES)


@pytest.mark.parametrize("switch", PARALLEL_PROFILE_SWITCHES)
@pytest.mark.parametrize("value", PROFILE_VALIDATION_VALUES)
def test_parallel_profile_tuning_matches_runner_ambient_validation(switch, value, monkeypatch):
    ambient_settings = (set(runner.OPT_IN_SWITCHES) | set(runner.CHECK_SETTINGS)
                        | set(runner.PARALLEL_TIMING_SETTINGS))
    for name in ambient_settings:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(switch, str(value))

    try:
        if switch in runner.PARALLEL_TIMING_SETTINGS:
            runner.resolve_parallel_timing_env(parallel_enabled=True)
        else:
            runner.resolve_opt_in_switches(parallel_enabled=True)
        ambient_accepted = True
    except runner.RunError:
        ambient_accepted = False

    try:
        parallel_records.ParallelLimits.from_settings({switch: value}, {}, prompt_budget=33344)
        profile_accepted = True
    except parallel_records.ParallelConfigError as error:
        profile_accepted = False
        reason = ("must be a positive finite number of seconds"
                  if switch in runner.PARALLEL_TIMING_SETTINGS
                  else "must be on or off; unset it to keep the default")
        assert (error.setting, error.source, error.reason) == (switch, switch, reason)
        assert str(error) == f"invalid parallel setting {switch}={value!r} from {switch}: {reason}"

    assert profile_accepted is ambient_accepted, (switch, value)


@pytest.mark.parametrize("switch", PARALLEL_PROFILE_SWITCHES)
def test_cli_parallel_two_rejects_invalid_profile_tuning(tmp_path, monkeypatch, capsys, switch):
    invalid = "text" if switch in runner.PARALLEL_TIMING_SETTINGS else "maybe"
    exit_code, output, _prepared, _expected = _run_cli_through_prepare(
        tmp_path, monkeypatch, capsys, (switch, invalid))

    assert exit_code != 0
    assert switch in output
