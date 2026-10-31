import json

import pytest

from stateframe.cli import build_parser, _run_options
from stateframe.endpoint import Endpoint
from stateframe.engine.kernel_tool_settings import ToolSettings, ToolSettingsError, load_tool_settings
from stateframe.engine.kernel_release_switch_registry import release_stateframe_switches
from stateframe.runner import RunOptions, prepare, resume_run
from stateframe.resume import ResumeRefused
from stateframe import profiles


def test_cli_modes_and_resume_flags():
    parser = build_parser()
    run = parser.parse_args(["run", "--repo", ".", "--task", "fix", "--endpoint", "http://localhost:8080/v1",
                             "--model", "x", "--shell", "read-only", "--no-run-env", "--shell-allow", "rg"])
    assert run.shell == "read-only" and run.run_env is False and run.shell_allow == ["rg"]
    chat = parser.parse_args(["chat", "--repo", ".", "--endpoint", "http://localhost:8080/v1", "--model", "x"])
    assert chat.shell_ask is True
    resume = parser.parse_args(["resume", "run", "--replay-external"])
    assert resume.replay_external is True
    with pytest.raises(SystemExit):
        parser.parse_args(["resume", "run", "--shell", "off"])


def test_host_mode_requires_unrestricted_and_records_authorization(capsys):
    parser = build_parser()
    base = ["run", "--repo", ".", "--task", "fix", "--endpoint", "http://localhost:8080/v1",
            "--model", "x", "--shell-sandbox", "off"]
    endpoint = Endpoint.from_args("http://localhost:8080/v1", "x", None)
    with pytest.raises(ToolSettingsError, match="requires --shell unrestricted"):
        _run_options(parser.parse_args(base), endpoint)
    options = _run_options(parser.parse_args(base + ["--shell", "unrestricted"]), endpoint)
    assert options.tool_settings.host_authorization["flags"] == [
        "--shell unrestricted", "--shell-sandbox off"]
    assert "WARNING: SHELL SANDBOX OFF" in capsys.readouterr().err


def test_settings_file_round_trip_and_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    original = ToolSettings(shell="unrestricted", shell_network="none", shell_allow=("python",))
    path.write_text(json.dumps(original.as_dict()))
    monkeypatch.setenv("STATEFRAME_TOOL_SETTINGS_FILE", str(path))
    assert load_tool_settings() == original
    path.write_text('{"schema":"stateframe-tool-settings/1"}')
    with pytest.raises(ToolSettingsError, match="missing or unknown fields"):
        load_tool_settings()
    path.write_text("{")
    with pytest.raises(ToolSettingsError, match="STATEFRAME_TOOL_SETTINGS_FILE"):
        load_tool_settings()
    path.write_text('{"schema":"stateframe-tool-settings/1","schema":"stateframe-tool-settings/1"}')
    with pytest.raises(ToolSettingsError, match="duplicate JSON key schema"):
        load_tool_settings()


def test_settings_persist_in_run_spec_and_child_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    run_dir = tmp_path / "run"
    settings = ToolSettings(shell="off", shell_network="none", run_env=False, git_tool=False)
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    options = RunOptions(repo=repo, task="inspect sample", endpoint=endpoint,
                         run_dir=run_dir, tool_settings=settings)
    run, _, spec = prepare(options)
    assert spec["tool_settings"] == settings.as_dict()
    assert spec["environment"]["STATEFRAME_TOOL_SETTINGS_FILE"] == str(run / "tool-settings.json")
    monkeypatch.setenv("STATEFRAME_TOOL_SETTINGS_FILE", str(run / "tool-settings.json"))
    assert load_tool_settings() == settings


def test_release_registry_sees_host_file_reader():
    switches, _ = release_stateframe_switches()
    assert "STATEFRAME_TOOL_SETTINGS_FILE" in switches


@pytest.mark.parametrize("change", ["definition", "executable"])
def test_resume_refuses_changed_custom_tool(tmp_path, change):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text("def test_sample():\n    assert True\n")
    executable = tmp_path / "command"
    executable.write_text("#!/bin/sh\necho one\n")
    executable.chmod(0o755)
    tool_file = tmp_path / "tools.json"
    definition = {"name": "sample_tool", "description": "Print",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "argv": [str(executable)], "timeout_seconds": 2}
    tool_file.write_text(json.dumps({"schema": "stateframe-tools/1", "tools": [definition]}))
    endpoint = Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None)
    settings = ToolSettings(tools_files=({"path": str(tool_file), "digest": "pending"},))
    run_dir, _, _ = prepare(RunOptions(repo=repo, task="inspect sample", endpoint=endpoint,
                                      run_dir=tmp_path / "run", tool_settings=settings))
    if change == "definition":
        tool_file.write_text(tool_file.read_text().replace("Print", "Changed"))
        reason = "custom tool definition changed"
    else:
        executable.write_text("#!/bin/sh\necho two\n")
        reason = "custom tool executable or definition changed"
    with pytest.raises(ResumeRefused, match=reason):
        resume_run(run_dir)


@pytest.mark.parametrize("name", ["STATEFRAME_SHELL", "STATEFRAME_SHELL_NETWORK", "STATEFRAME_WEB",
                                  "STATEFRAME_TOOLS", "STATEFRAME_HOST_MODE", "STATEFRAME_RUN_ENV",
                                  "STATEFRAME_GIT_TOOL", "STATEFRAME_TOOL_SETTINGS_FILE"])
def test_privileged_profile_key_refused(tmp_path, name):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": {name: "on"}}))
    with pytest.raises(ValueError, match="privileged tool settings"):
        profiles.load(str(path))
