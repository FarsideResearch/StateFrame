"""Mutation L0 leaves full requests, tool order, imports and the run spec untouched."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from stateframe import runner
from stateframe.scripted_endpoint import Script, serve
from tools.compare_mutation_off_requests import capture, write_launcher


CURRENT = Path(__file__).resolve().parents[1]
_baseline = os.environ.get("SF_TEST_PARITY_BASELINE")
BASELINE = Path(_baseline).expanduser().resolve() if _baseline else None


def test_default_level_is_off_and_omits_spec_key(monkeypatch):
    monkeypatch.delenv("STATEFRAME_MUTATIONS", raising=False)
    before = {name for name in sys.modules if name.startswith("kernel_mutate")}
    assert runner.mutation_spec_options(None, {}) is None
    assert {name for name in sys.modules if name.startswith("kernel_mutate")} == before


def test_explicit_off_overrides_ambient_and_profile(monkeypatch):
    from stateframe.endpoint import Endpoint

    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    options = runner.RunOptions(repo=Path("."), task="parity", mutations="off",
                                endpoint=Endpoint("http://127.0.0.1:1/v1", "scripted"))
    assert runner.mutation_spec_options(options, {"STATEFRAME_MUTATIONS": "authored"}) is None
    assert "STATEFRAME_MUTATIONS" not in runner.profiles.load("r2hc3")


def test_opt_in_switch_accepts_only_literal_levels(monkeypatch):
    assert runner.OPT_IN_SWITCHES["STATEFRAME_MUTATIONS"] == (
        "off", "prebuilt", "authored")
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "prebuilt")
    assert runner.resolve_opt_in_switches()["STATEFRAME_MUTATIONS"] == "prebuilt"


@pytest.mark.host_sandbox
def test_ambient_level_reaches_run_spec_not_just_tool_menu(monkeypatch, tmp_path):
    from stateframe.endpoint import Endpoint

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample(): assert True\n")
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    options = runner.RunOptions(repo=repo, task="Read sample.py", target="sample.py",
                                run_dir=tmp_path / "run",
                                endpoint=Endpoint("http://127.0.0.1:1/v1", "scripted"),
                                max_turns=3)
    _, _, spec = runner.prepare(options, printer=lambda message: None)
    assert spec["mutations"]["level"] == "authored"
    assert spec["environment"]["STATEFRAME_MUTATIONS"] == "authored"


def _remove_run(path: Path) -> None:
    """A finished run may leave read-only directories (protected snapshots); make them writable first."""
    os.chmod(path, 0o700)
    for directory, children, _files in os.walk(path):
        for child in children:
            child_path = os.path.join(directory, child)
            if not os.path.islink(child_path):
                os.chmod(child_path, 0o700)
    shutil.rmtree(path)


def _stable_spec(value: dict) -> dict:
    """Drop only fields tied to the capture's endpoint, clock or absolute paths."""
    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()
                    if key not in {"chat_url", "engine_dir", "created"}
                    and not (key in {"file_count", "shipped_hash", "release_manifest", "source_hash"}
                             and item is value.get("engine_source"))
                    and not (key == "DS_URL")}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, str) and item.startswith("/"):
            return "<absolute-path>"
        return item
    return normalize(value)


@pytest.mark.host_sandbox
@pytest.mark.skipif(BASELINE is None, reason="set SF_TEST_PARITY_BASELINE to an untouched base checkout")
@pytest.mark.parametrize("example", ["discount_bug", "tiny_site", "brief_to_report"])
def test_l0_full_run_capture_parity_against_base(tmp_path, example):
    assert BASELINE is not None
    launcher = tmp_path / "launch.py"
    write_launcher(launcher)
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    base_request, base_tools, base_spec = capture(BASELINE, sys.executable, launcher, example, repo, run,
                                                  stable_tool_head="on")
    _remove_run(run)
    current_request, current_tools, current_spec = capture(CURRENT, sys.executable, launcher, example, repo, run,
                                                            stable_tool_head="on")
    assert current_request == base_request
    assert current_tools == base_tools
    assert _stable_spec(current_spec) == _stable_spec(base_spec)
