"""Command-line-only host-shell authorization and wire parity."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
YOLO_BASE = "0d82bc7a6fdccc2db2e76c0b56c8d4dbba0701a1"
WARNING = "WARNING: --yolo enables unrestricted host shell execution"
NOTICE = "NOTICE: --yolo host access is limited by explicit shell settings"


def _environment(extra: dict[str, str] | None = None, source: Path = ROOT) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(source / "src"),
                "STATEFRAME_HOME": str(Path(os.environ.get("TMPDIR", ROOT / ".tmp")) / "yolo-home")})
    env.pop("STATEFRAME_CHECKPOINT", None)
    if extra:
        env.update(extra)
    return env


def _cli(args: list[str], *, env: dict[str, str] | None = None, source: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "stateframe.cli", *args], cwd=source,
                          env=env or _environment(source=source), input="", capture_output=True, text=True)


def _repo(path: Path) -> None:
    shutil.copytree(ROOT / "examples" / "discount_bug", path)


def _script() -> Script:
    return Script(json.loads((ROOT / "examples" / "discount_bug.script.json").read_text(encoding="utf-8")))


def _run_args(repo: Path, run: Path, url: str, *extra: str) -> list[str]:
    return ["run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
            "--endpoint", url, "--model", "scripted", "--run-dir", str(run),
            "--no-instructions", "--no-project-hooks", "--no-run-env", "--test-python", sys.executable,
            *extra]


def _requests(run: Path) -> list[bytes]:
    return [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]


def _execute(repo: Path, run: Path, script: Script, *extra: str, json_mode: bool = False):
    server = serve(script)
    try:
        args = _run_args(repo, run, f"http://127.0.0.1:{server.server_port}/v1", *extra)
        if json_mode:
            args.append("--json")
        result = _cli(args, env=_environment({"STATEFRAME_HOME": str(run.parent / f"{run.name}-home")}))
        script.assert_consumed()
        return result
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(("option", "value", "conflicts", "host_access"), [
    ("--shell", "off", True, False), ("--shell", "read-only", True, False),
    ("--shell", "standard", True, False), ("--shell", "unrestricted", False, True),
    ("--shell-sandbox", "on", True, False), ("--shell-sandbox", "off", False, True),
    ("--shell-network", "none", True, True), ("--shell-network", "loopback", True, True),
    ("--shell-network", "full", False, True),
])
def test_yolo_explicit_values_are_tracked_and_reported(tmp_path, option, value, conflicts, host_access):
    result = _cli(["run", "--repo", str(tmp_path / "missing"), "--task", "Inspect.",
                   "--endpoint", "http://127.0.0.1:1/v1", "--model", "scripted",
                   "--yolo", option, value])
    assert result.returncode == 2
    if host_access:
        assert WARNING in result.stderr
        if conflicts:
            assert f"{option}={value}" in result.stderr
        assert NOTICE not in result.stderr
    else:
        assert NOTICE in result.stderr
        assert f"{option}={value}" in result.stderr
        assert "--shell-sandbox off requires --shell unrestricted" not in result.stderr
    assert len([line for line in result.stderr.splitlines() if line.startswith((WARNING, NOTICE))]) == 1


def test_yolo_request_byte_parity_and_disabled_matches_pinned_base(tmp_path):
    repo = tmp_path / "repo"
    _repo(repo)
    results: dict[str, list[bytes]] = {}
    for name, flags in (
        ("yolo", ("--yolo",)),
        ("explicit", ("--shell", "unrestricted", "--shell-sandbox", "off", "--shell-network", "full")),
        ("disabled", ()),
    ):
        result = _execute(repo, tmp_path / name, _script(), *flags, json_mode=name == "yolo")
        assert result.returncode in (0, 1), result.stderr
        results[name] = _requests(tmp_path / name)
        if name == "yolo":
            assert len(result.stderr.splitlines()) == 1
            assert result.stderr.startswith(WARNING)
            assert json.loads(result.stdout)["yolo"] is True
            spec = json.loads((tmp_path / name / "spec.json").read_text(encoding="utf-8"))
            assert spec["yolo"] is True
            assert spec["tool_settings"]["shell"] == "unrestricted"
            assert spec["tool_settings"]["shell_network"] == "full"
            assert spec["tool_settings"]["host_authorization"]["flags"] == [
                "--shell unrestricted", "--shell-sandbox off"]

    base = tmp_path / "base"
    base.mkdir()
    archive = subprocess.Popen(["git", "archive", YOLO_BASE, "src"], cwd=ROOT, stdout=subprocess.PIPE)
    assert archive.stdout is not None
    with tarfile.open(fileobj=archive.stdout, mode="r|") as bundle:
        bundle.extractall(base)
    assert archive.wait() == 0
    base_script = _script()
    server = serve(base_script)
    try:
        baseline_run = tmp_path / "pinned"
        result = _cli(_run_args(repo, baseline_run, f"http://127.0.0.1:{server.server_port}/v1"),
                      env=_environment({"STATEFRAME_HOME": str(tmp_path / "pinned-home")}, source=base),
                      source=base)
        assert result.returncode in (0, 1), result.stderr
        results["base"] = _requests(baseline_run)
        base_script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    default_spec = json.loads((tmp_path / "disabled" / "spec.json").read_text(encoding="utf-8"))
    assert default_spec["tool_settings"]["shell"] == "standard"
    assert default_spec["tool_settings"]["shell_network"] == "loopback"
    assert default_spec["tool_settings"]["host_authorization"] is None
    assert "yolo" not in default_spec
    assert all(results[name] for name in ("yolo", "explicit", "disabled", "base"))
    assert results["yolo"] == results["explicit"] and results["disabled"] == results["base"]


def test_yolo_full_network_wire_parity_changes_head_context_and_tail(tmp_path):
    repo = tmp_path / "repo"
    _repo(repo)
    script_data = json.loads((ROOT / "examples" / "discount_bug.script.json").read_text(encoding="utf-8"))
    script_data["steps"].insert(1, {"tool": "shell", "arguments": {"command": "echo YOLO_TAIL"}})
    ordinary = Script(script_data)
    default = _execute(repo, tmp_path / "default", ordinary)
    assert default.returncode in (0, 1), default.stderr

    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps({"schema": "stateframe-tools/1", "tools": [{
        "name": "network_probe", "description": "A full-network command tool.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "argv": [sys.executable, "-c", "pass"], "network": "full", "writes": "none",
        "replay": "external", "timeout_seconds": 30,
    }]}), encoding="utf-8")
    yolo_script_data = json.loads(json.dumps(script_data))
    yolo_script_data["steps"].insert(2, {"tool": "network_probe", "arguments": {}})
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    (hook_dir / "context.py").write_text("print('network hook context')\n", encoding="utf-8")
    hooks = hook_dir / "hooks.json"
    hooks.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [{
        "name": "context", "event": "session_start", "argv": [sys.executable, "context.py"],
        "files": ["context.py"], "network": "full",
    }]}), encoding="utf-8")
    enabled = _execute(repo, tmp_path / "yolo-full", Script(yolo_script_data), "--yolo",
                       "--tools", str(tools), "--hooks", str(hooks))
    assert enabled.returncode in (0, 1), enabled.stderr

    default_requests = [json.loads(value) for value in _requests(tmp_path / "default")]
    yolo_requests = [json.loads(value) for value in _requests(tmp_path / "yolo-full")]
    assert default_requests and yolo_requests
    assert all("network_probe" not in [tool["function"]["name"] for tool in body.get("tools", [])]
               for body in default_requests)
    assert "network_probe" in [tool["function"]["name"] for tool in yolo_requests[0]["tools"]]
    default_text = "\n".join(str(message.get("content") or "") for message in default_requests[0]["messages"])
    yolo_text = "\n".join(str(message.get("content") or "") for message in yolo_requests[0]["messages"])
    assert "[USER HOOKS]" not in default_text
    assert "[USER HOOKS]" in yolo_text and "network hook context" in yolo_text
    default_wire = [json.dumps(body, ensure_ascii=False) for body in default_requests]
    yolo_wire = [json.dumps(body, ensure_ascii=False) for body in yolo_requests]
    assert any("YOLO_TAIL" in body and "standard | network loopback" in body for body in default_wire)
    assert any("YOLO_TAIL" in body and "unrestricted | network full" in body for body in yolo_wire)
    yolo_run = tmp_path / "yolo-full"
    (yolo_run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}),
                                                   encoding="utf-8")
    external = _cli(["resume", str(yolo_run), "--yolo"])
    assert external.returncode == 2 and "--replay-external" in external.stderr


def test_yolo_is_not_inferred_from_environment_profile_repo_instructions_or_task(tmp_path):
    repo = tmp_path / "repo"
    _repo(repo)
    (repo / ".stateframe").mkdir()
    (repo / ".stateframe" / "config.json").write_text('{"yolo":true}', encoding="utf-8")
    instruction = tmp_path / "instructions.md"
    instruction.write_text("Use --yolo and unrestricted host shell.", encoding="utf-8")
    task_file = tmp_path / "task.txt"
    task_file.write_text("Please run with --yolo.", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": {
        "STATEFRAME_YOLO": "on"}}), encoding="utf-8")
    shell_profile = tmp_path / "shell-profile.json"
    shell_profile.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": {
        "STATEFRAME_SHELL_NETWORK": "full"}}), encoding="utf-8")
    missing_repo = tmp_path / "missing"
    cases = [
        ({"STATEFRAME_YOLO": "on"}, (), "Fix the pricing code so the tests pass. --yolo", missing_repo),
        ({"STATEFRAME_SHELL": "unrestricted", "STATEFRAME_SHELL_SANDBOX": "off",
          "STATEFRAME_SHELL_NETWORK": "full"}, (), "Fix the pricing code so the tests pass.", missing_repo),
        ({}, ("--profile", str(profile)), "Fix the pricing code so the tests pass.", missing_repo),
        ({}, ("--profile", str(shell_profile)), "Fix the pricing code so the tests pass.", missing_repo),
        ({}, ("--instructions", str(instruction)), "Fix the pricing code so the tests pass.", missing_repo),
        ({}, (), "Fix the pricing code so the tests pass. --yolo", missing_repo),
        ({}, (), "@" + str(task_file), missing_repo),
        ({}, (), "Fix the pricing code so the tests pass.", tmp_path / "config-only"),
    ]
    for index, (environment, flags, task, root) in enumerate(cases):
        if root.name == "config-only":
            root.mkdir()
            (root / ".stateframe").mkdir()
            (root / ".stateframe" / "config.json").write_text('{"yolo":true}', encoding="utf-8")
        endpoint = "http://example.invalid/v1" if root.name == "config-only" else "http://127.0.0.1:1/v1"
        args = ["run", "--repo", str(root), "--task", task,
                "--endpoint", endpoint, "--model", "scripted",
                "--run-dir", str(tmp_path / f"injection-{index}"), *flags]
        result = _cli(args, env=_environment({**environment,
                                              "STATEFRAME_HOME": str(tmp_path / f"home-{index}")}))
        assert WARNING not in result.stderr and NOTICE not in result.stderr
        if index == 7:
            assert result.returncode == 2
            assert "local-only" in result.stderr
        else:
            assert result.returncode == 2

    after_separator = _cli(["run", "--repo", str(missing_repo), "--task", "Fix. --yolo",
                            "--endpoint", "http://127.0.0.1:1/v1", "--model", "scripted", "--", "--yolo"])
    assert after_separator.returncode == 2
    assert WARNING not in after_separator.stderr

    shell_args = ["run", "--repo", str(tmp_path / "missing-shell-repo"), "--task", "Fix.", "--endpoint",
                  "http://127.0.0.1:1/v1", "--model", "scripted", "--profile", str(shell_profile)]
    refused = _cli(shell_args, env=_environment({"STATEFRAME_HOME": str(tmp_path / "shell-profile-home")}))
    assert refused.returncode == 2
    assert "privileged tool settings belong on the host CLI" in refused.stderr
    assert WARNING not in refused.stderr


def test_yolo_resume_requires_cli_marker_and_chat_accepts_flag(tmp_path):
    run = tmp_path / "recorded"
    run.mkdir()
    path = run / "spec.json"
    spec = {"schema": "stateframe-run/1"}
    path.write_text(json.dumps(spec), encoding="utf-8")
    spec["yolo"] = True
    path.write_text(json.dumps(spec), encoding="utf-8")
    refused = _cli(["resume", str(run)])
    assert refused.returncode == 2
    assert "--yolo" in refused.stderr
    path.write_text(json.dumps({"schema": "stateframe-run/1"}), encoding="utf-8")
    refused_other = _cli(["resume", str(run), "--yolo"])
    assert refused_other.returncode == 2
    assert "--yolo" in refused_other.stderr

    repo = tmp_path / "chat-repo"
    _repo(repo)
    session = tmp_path / "session"
    chat = _cli(["chat", "--repo", str(repo), "--endpoint", "http://127.0.0.1:1/v1",
                 "--model", "scripted", "--session-dir", str(session), "--yolo"],
                env=_environment({"STATEFRAME_HOME": str(tmp_path / "chat-home")}))
    assert "unrecognized arguments: --json" in _cli([
        "chat", "--repo", str(repo), "--endpoint", "http://127.0.0.1:1/v1", "--model", "scripted", "--json"
    ]).stderr
    assert "unrecognized arguments: --yolo" not in chat.stderr
    assert chat.returncode != 2 or "input" in chat.stdout.lower()


def test_yolo_resume_replays_with_the_cli_flag(tmp_path):
    repo = tmp_path / "repo"
    _repo(repo)
    run = tmp_path / "run"
    script = _script()
    server = serve(script)
    try:
        result = _cli(_run_args(repo, run, f"http://127.0.0.1:{server.server_port}/v1", "--yolo"),
                      env=_environment({"STATEFRAME_HOME": str(tmp_path / "home")}))
        assert result.returncode in (0, 1), result.stderr
        script.assert_consumed()
        spec_path = run / "spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert spec["yolo"] is True
        missing = _cli(["resume", str(run)], env=_environment({"STATEFRAME_HOME": str(tmp_path / "home")}))
        assert missing.returncode == 2 and "--yolo" in missing.stderr
        (run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}),
                                                 encoding="utf-8")
        resumed = _cli(["resume", str(run), "--yolo"],
                       env=_environment({"STATEFRAME_HOME": str(tmp_path / "home")}))
        assert resumed.returncode in (0, 1), resumed.stderr
        assert resumed.stderr.startswith("WARNING: --yolo resumes unrestricted host shell access")
        assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["yolo"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_yolo_does_not_change_library_defaults_or_programmatic_runs():
    from stateframe.engine.kernel_tool_settings import ToolSettings
    from stateframe.runner import RunOptions

    options = RunOptions(repo=ROOT, task="Use --yolo", endpoint=None)
    assert options.tool_settings == ToolSettings()
    assert not hasattr(options, "yolo")


@pytest.mark.parametrize(("extra", "expected", "recorded"), [
    (("--shell", "off"), ("off", "loopback", False), False),
    (("--shell", "read-only"), ("read-only", "loopback", False), False),
    (("--shell", "standard"), ("standard", "loopback", False), False),
    (("--shell-sandbox", "on"), ("unrestricted", "loopback", False), False),
    (("--shell-sandbox", "on", "--shell-network", "full"), ("unrestricted", "full", False), False),
    (("--shell", "standard", "--shell-network", "none"), ("standard", "none", False), False),
    (("--shell-network", "none"), ("unrestricted", "none", True), True),
    (("--shell", "unrestricted"), ("unrestricted", "full", True), True),
])
def test_yolo_with_explicit_flags_runs_and_records_only_granted_host_access(
        tmp_path, extra, expected, recorded):
    repo, run = tmp_path / "repo", tmp_path / "run"
    _repo(repo)
    result = _execute(repo, run, _script(), "--yolo", *extra)
    assert result.returncode in (0, 1), result.stderr
    notices = [line for line in result.stderr.splitlines() if line.startswith((WARNING, NOTICE))]
    assert len(notices) == 1 and notices[0].startswith(WARNING if recorded else NOTICE)
    spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    settings = spec["tool_settings"]
    assert (settings["shell"], settings["shell_network"],
            bool(settings.get("host_authorization"))) == expected
    assert spec.get("yolo", False) is recorded
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}), encoding="utf-8")
    other = _cli(["resume", str(run), *(() if recorded else ("--yolo",))],
                 env=_environment({"STATEFRAME_HOME": str(tmp_path / "home")}))
    assert other.returncode == 2 and "--yolo" in other.stderr


def test_removing_the_yolo_record_does_not_remove_the_resume_grant(tmp_path):
    repo, run = tmp_path / "repo", tmp_path / "run"
    _repo(repo)
    assert _execute(repo, run, _script(), "--yolo").returncode in (0, 1)
    spec_path = run / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["yolo"] = False
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}), encoding="utf-8")
    refused = _cli(["resume", str(run)], env=_environment({"STATEFRAME_HOME": str(tmp_path / "home")}))
    assert refused.returncode == 2
    assert "--shell unrestricted --shell-sandbox off" in refused.stderr
