"""JSON-mode stdout remains one document through warnings and startup failures."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe import cli, mutations_cli, shell_host
from stateframe.scripted_endpoint import Script, serve


@pytest.mark.parametrize("failure", [
    "run_env", "bad_profile", "missing_repo", "bad_endpoint", "bad_tool_settings",
    "missing_resume", "mutation_platform", "bad_argument",
])
def test_json_cli_failure_paths_are_one_document(failure, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("STATEFRAME_ALLOW_REMOTE_ENDPOINT", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\nassert VALUE == 7\n", encoding="utf-8")
    script = Script({"model": "scripted", "steps": [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "done", "arguments": {"summary": "app.py checks the expected value."}},
    ]})
    server = serve(script)
    run_dir = tmp_path / "run"
    args = [
        "run", "--repo", str(repo), "--task", "Inspect app.py and verify its assertion.",
        "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
        "--run-dir", str(run_dir), "--test-python", sys.executable,
        "--check-cmd", f"{sys.executable} app.py", "--max-turns", "4", "--json",
    ]
    if failure == "run_env":
        def unavailable_environment(*_args, **_kwargs):
            raise shell_host.RunEnvError("injected environment provisioning failure")

        monkeypatch.setattr(shell_host, "create_run_env", unavailable_environment)
        expected_error = "run environment off: injected environment provisioning failure"
    elif failure == "bad_profile":
        args += ["--profile", "no-such-json-test-profile"]
        expected_error = "unknown profile"
    elif failure == "missing_repo":
        args[args.index("--repo") + 1] = str(tmp_path / "missing")
        expected_error = "repository unavailable"
    elif failure == "bad_endpoint":
        args[args.index("--endpoint") + 1] = "http://remote.invalid/v1"
        expected_error = "local-only"
    elif failure == "bad_tool_settings":
        args += ["--shell-sandbox", "off", "--shell", "read-only"]
        expected_error = "requires --shell unrestricted"
    elif failure == "missing_resume":
        args = ["resume", str(tmp_path / "missing-run"), "--json"]
        expected_error = "cannot resume"
    elif failure == "bad_argument":
        args = ["run", "--json", "--max-turns", "invalid"]
        expected_error = "--max-turns"
    else:
        monkeypatch.setattr(cli.sys, "platform", "darwin")
        args += ["--mutations", "prebuilt"]
        expected_error = "refused"

    try:
        status = cli.main(args)
    finally:
        server.shutdown()
        server.server_close()
    captured = capsys.readouterr()
    # json.loads rejects both leading diagnostics and a second JSON document.
    result = json.loads(captured.out)
    assert isinstance(result, dict)
    assert result["exit_code"] == status
    assert expected_error in captured.err
    if failure == "run_env":
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert result == summary
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        assert spec["tool_settings"]["run_env"] is False
        assert result["succeeded"] is True
        assert status == 0
        script.assert_consumed()
    else:
        assert status == 2
        assert result["succeeded"] is False
        assert expected_error in result["error"]
        if failure == "missing_repo":
            assert str(tmp_path / "missing") in result["error"]
            assert str(tmp_path / "missing") in captured.err


@pytest.mark.parametrize("argv,error", [
    (["run", "--json", "--max-turns", "invalid"], "--max-turns"),
    (["run", "--max-turns", "invalid", "--json"], "--max-turns"),
    (["run", "--json"], "required"),
    (["resume", "missing", "--json", "--unknown"], "unrecognized"),
    (["resume", "missing", "--json", "--max-turns"], "expected one argument"),
    (["run", "--shell", "invalid", "--json"], "invalid choice"),
    (["mutations", "list", "--unknown", "--json"], "unrecognized"),
    (["gc", "--older-than", "invalid", "--json"], "nonnegative"),
    (["run", "--max-turns", "invalid", "--js"], "--max-turns"),
])
@pytest.mark.parametrize("process_argv", [False, True])
def test_json_argument_errors_are_one_document(argv, error, process_argv, monkeypatch, capsys):
    def unexpected_dispatch(*_args, **_kwargs):
        pytest.fail("argument errors must not dispatch a command")

    monkeypatch.setattr(cli, "_mutation_platform_refusal", unexpected_dispatch)
    if process_argv:
        monkeypatch.setattr(sys, "argv", ["stateframe", *argv])
    status = cli.main(None if process_argv else argv)
    captured = capsys.readouterr()
    assert status == 2
    result = json.loads(captured.out)
    assert result == {"error": captured.err.splitlines()[-1], "exit_code": 2, "succeeded": False}
    assert "usage:" in captured.err
    assert error in result["error"]


def test_json_argument_error_process_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(cli.__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-m", "stateframe.cli", "run", "--json", "--max-turns", "invalid"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    document = json.loads(result.stdout)
    assert document == {"error": result.stderr.splitlines()[-1], "exit_code": 2, "succeeded": False}
    assert "--max-turns" in document["error"]


@pytest.mark.parametrize("argv", [
    ["run", "--max-turns", "invalid"],
    ["run", "--max-turns", "invalid", "--", "--json"],
    ["run", "--task=--json", "--max-turns", "invalid"],
])
def test_text_argument_errors_keep_argparse_exit(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(argv)
    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "--max-turns" in captured.err


@pytest.mark.parametrize("argv", [["run", "--json", "--help"], ["--version", "--json"]])
def test_json_does_not_turn_help_or_version_into_errors(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(argv)
    assert stop.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(("usage:", "stateframe "))
    assert captured.err == ""


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("error", [ValueError("injected startup failure"), KeyboardInterrupt()])
def test_cli_error_output_preserves_text_mode(error, json_mode, monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(cli, "run", fail)
    flags = ["--json"] if json_mode else []
    status = cli.main([
        "run", "--repo", ".", "--task", "Inspect", "--endpoint", "http://127.0.0.1:8080/v1",
        "--model", "scripted", *flags,
    ])
    captured = capsys.readouterr()
    assert status == (130 if isinstance(error, KeyboardInterrupt) else 2)
    assert captured.err.startswith("stateframe: ")
    if json_mode:
        result = json.loads(captured.out)
        assert result == {"error": captured.err.strip(), "exit_code": status, "succeeded": False}
    else:
        assert captured.out == ""


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("error,status", [
    (mutations_cli._Refusal("injected refusal"), 2),
    (OSError("injected store failure"), 1),
])
def test_mutations_failure_json_is_one_document(error, status, json_mode, monkeypatch, capsys):
    def fail(_args):
        raise error

    monkeypatch.setattr(mutations_cli, "_command", fail)
    flags = ["--json"] if json_mode else []
    assert cli.main(["mutations", "list", *flags]) == status
    captured = capsys.readouterr()
    assert str(error) in captured.err
    if json_mode:
        result = json.loads(captured.out)
        assert result == {"error": captured.err.strip(), "exit_code": status, "succeeded": False}
    else:
        assert captured.out == ""
