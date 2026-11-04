"""Powerups through the existing child tool and hook paths."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe import powerups
from stateframe.engine.kernel_effects import WorkspaceEffectPolicy
from stateframe.engine.kernel_hook_exec import HookRunner
from stateframe.engine.kernel_hooks import create_bundle, load_hook_files
from stateframe.scripted_endpoint import Script, serve
from engine_run import run_monitored

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "powerups"
pytestmark = pytest.mark.host_sandbox


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_value.py").write_text(
        "from pkg.value import VALUE\ndef test_value():\n    assert VALUE in (1, 3)\n"
    )
    return repo


def _run(tmp_path, steps, *flags):
    repo = _repo(tmp_path)
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "STATEFRAME_HOME": str(tmp_path / "home"),
    }
    try:
        result = run_monitored(
            [
                sys.executable,
                "-m",
                "stateframe",
                "run",
                "--repo",
                str(repo),
                "--task",
                "Search, edit and check VALUE.",
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "scripted",
                "--run-dir",
                str(run),
                "--test-python",
                sys.executable,
                "--max-turns",
                "8",
                "--no-run-env",
                "--quiet",
                *flags,
            ],
            env=env,
            capture_output=True,
            text=True,
            expected_seconds=600,
        )
        try:
            script.assert_consumed()
        except AssertionError as error:
            raise AssertionError(
                f"{error}\nCLI {result.returncode}: {result.stdout}{result.stderr}"
            ) from error
        wire = [
            path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
        ]
        return result, run, wire
    finally:
        server.shutdown()
        server.server_close()


def step(tool, **arguments):
    return {"tool": tool, "required": True, "arguments": arguments}


def test_repo_grep_command_runs_in_sandbox(tmp_path):
    result, run, wire = _run(
        tmp_path,
        [
            step("grep_repo", pattern="VALUE", glob="*.py"),
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Search and checks passed."),
        ],
        "--powerup",
        str(EXAMPLES / "repo-grep"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert b"grep_repo" in wire[0]
    assert any(b"pkg/value.py:1" in item for item in wire)
    assert (
        json.loads((run / "spec.json").read_text())["powerups"]["active"][0]["name"]
        == "repo-grep"
    )


def test_offer_has_no_command_schema_or_catalog_name(tmp_path):
    result, run, wire = _run(
        tmp_path,
        [
            step("skill_load", name="repo-grep"),
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Checks passed."),
        ],
        "--powerup",
        str(EXAMPLES / "repo-grep") + "=offer",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads(wire[0])
    names = [tool["function"]["name"] for tool in first["tools"]]
    assert "grep_repo" not in names
    assert b"grep_repo" not in wire[0]
    assert b"repo-grep" in wire[0]


def test_py_format_missing_ruff_is_noop_with_once_notice(tmp_path):
    result, run, wire = _run(
        tmp_path,
        [
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Checks passed."),
        ],
        "--powerup",
        str(EXAMPLES / "py-format"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "workspace/pkg/value.py").read_text() == "VALUE = 1\n"
    assert sum(b"ruff is unavailable" in item for item in wire) >= 1
    probe = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "py-format/format.py"),
            "--line-length",
            "100",
            str(run / "workspace/pkg/value.py"),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0 and not probe.stdout


def test_py_format_uses_bundled_fake_ruff(tmp_path):
    source = tmp_path / "py-format"
    shutil.copytree(EXAMPLES / "py-format", source)
    fake = source / "ruff"
    fake.write_text(
        f"#!{sys.executable}\nimport pathlib,sys\n"
        "p=pathlib.Path(sys.argv[-1]);p.write_text(p.read_text().replace('VALUE=3','VALUE = 3'))\n"
    )
    fake.chmod(0o755)
    config = json.loads((source / "hooks.json").read_text())
    for hook in config["hooks"]:
        hook["files"].append("ruff")
    (source / "hooks.json").write_text(json.dumps(config))
    result, run, wire = _run(
        tmp_path,
        [
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Checks passed."),
        ],
        "--powerup",
        str(source),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(b"ruff is unavailable" in item for item in wire)
    sample = tmp_path / "sample.py"
    sample.write_text("VALUE=3\n")
    probe = subprocess.run(
        [
            sys.executable,
            str(source / "format.py"),
            "--line-length",
            "100",
            str(sample),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert sample.read_text() == "VALUE = 3\n"


@pytest.mark.parametrize("ruff_present", [False, True])
def test_py_format_post_tool_sandbox_write_scope(tmp_path, monkeypatch, ruff_present):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = tmp_path / "py-format"
    shutil.copytree(EXAMPLES / "py-format", source)
    if ruff_present:
        fake = source / "ruff"
        fake.write_text(
            f"#!{sys.executable}\nimport pathlib,sys\n"
            "p=pathlib.Path(sys.argv[-1]);p.write_text(p.read_text().replace('VALUE=3','VALUE = 3'))\n"
        )
        fake.chmod(0o755)
        config = json.loads((source / "hooks.json").read_text())
        for hook in config["hooks"]:
            hook["files"].append("ruff")
        (source / "hooks.json").write_text(json.dumps(config))
    repo = tmp_path / "repo"
    repo.mkdir()
    selected = powerups.resolve_units(repo, "", "default", (str(source),), (), False)
    assert len(selected.active) == 1, selected.skipped
    hooks = load_hook_files(
        (selected.active[0].materialized_dir / "hooks.json",),
        shell_network="none",
        repo=repo,
    )
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg/value.py").write_text("VALUE=3\n")
    (workspace / "pkg/other.py").write_text("OTHER=2\n")
    bundle = run / "hook-bundle"
    create_bundle(hooks, bundle)
    runner = HookRunner(
        run,
        hooks,
        bundle_dir=bundle,
        workspace=workspace,
        shell_network="none",
        run_seed="seed",
        run_id="run",
        version="0.3",
        effect_policy=WorkspaceEffectPolicy(
            lambda _: False, lambda _: True, lambda _: True
        ),
    )
    try:
        started = runner.run_event(
            "session_start", boundary="session_start", data={}, match={}
        )
        result = runner.run_event(
            "post_tool",
            boundary="post_tool:edit",
            data={"tool": "str_replace", "changed_paths": ["pkg/value.py"]},
            match={"tool": "str_replace", "changed_paths": ["pkg/value.py"]},
        )
        assert result.runs[0].outcome.exit_code == 0
    finally:
        runner.close()
    assert (workspace / "pkg/value.py").read_text() == (
        "VALUE = 3\n" if ruff_present else "VALUE=3\n"
    )
    assert (workspace / "pkg/other.py").read_text() == "OTHER=2\n"
    assert ("unavailable" in started.runs[0].outcome.context) is not ruff_present


def test_test_gate_runs_check(tmp_path):
    result, run, _ = _run(
        tmp_path,
        [
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Checks passed."),
        ],
        "--powerup",
        str(EXAMPLES / "test-gate"),
        "--powerup-set",
        "test-gate.check_command=true",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line)
        for line in (run / "hooks/records.jsonl").read_text().splitlines()
    ]
    assert any(row.get("hook") == "test-gate-stop" for row in records)


def test_test_gate_missing_program_refuses_submission(tmp_path):
    script = EXAMPLES / "test-gate/gate.py"
    absent = "stateframe-check-program-that-does-not-exist"
    notice = subprocess.run(
        [sys.executable, str(script), "--check-program", absent],
        capture_output=True,
        text=True,
    )
    check = subprocess.run(
        [sys.executable, str(script), "--run", absent],
        capture_output=True,
        text=True,
    )
    assert notice.returncode == 0
    assert notice.stdout.count("unavailable") == 1
    assert check.returncode != 0 and "not installed" in check.stderr


def test_no_powerups_keeps_spec_empty(tmp_path):
    result, run, _ = _run(
        tmp_path,
        [
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Checks passed."),
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "powerups" not in json.loads((run / "spec.json").read_text())
