"""Platform gates and qualification-boundary tests for mutations."""
from __future__ import annotations

import json
import sys

import pytest

from stateframe import cli

PLATFORM_REFUSAL = (
    "stateframe: refused: MUTATION_PLATFORM_UNSUPPORTED: mutations are Linux-only in this release; "
    "macOS is planned"
)


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="real mutation qualification requires Linux containment")
def test_linux_qualification_persists_success_and_limits(tmp_path):
    from kernel_mutate_exec import MutationLimits
    from kernel_mutate_qualify import qualify

    home = tmp_path / "stateframe-home"
    home.mkdir()
    result = qualify(home)

    assert result.ok
    assert result.platform == "linux"
    assert result.record
    checks = {check["name"]: check["ok"] for check in result.record["report"]["checks"]}
    assert checks and all(checks.values())
    qualified_path = home / "mutations" / "qualified.json"
    assert qualified_path.is_file()
    assert json.loads(qualified_path.read_text(encoding="utf-8")) == result.record
    assert result.record["schema"] == "stateframe-mutation-qualified/1"
    assert result.record["report"]["ok"] is True

    default_limits = MutationLimits()
    assert default_limits.wall_s == 10
    assert default_limits.cpu_s == 5
    assert default_limits.mem_mb == 256
    assert default_limits.output_bytes == 1 << 20

    reduced_limits = MutationLimits(wall_s=0.1, cpu_s=0.1, mem_mb=32, output_bytes=1024)
    for limits in (reduced_limits, default_limits):
        assert 0 < limits.wall_s <= 10
        assert 0 < limits.cpu_s <= 5
        assert 0 < limits.mem_mb <= 256
        assert 0 < limits.output_bytes <= 1 << 20


def test_mutation_platform_gate_refuses_non_linux_run_before_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    run_dir = tmp_path / "run"
    code = cli.main([
        "run", "--repo", str(tmp_path), "--task", "report a result",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "fixture",
        "--mutations", "prebuilt", "--run-dir", str(run_dir),
    ])
    error = capsys.readouterr().err
    assert code == 2
    assert error.strip() == PLATFORM_REFUSAL
    assert not run_dir.exists()


def test_mutation_platform_gate_refuses_environment_and_selftest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    run_dir = tmp_path / "environment-run"
    code = cli.main([
        "run", "--repo", str(tmp_path), "--task", "report a result",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "fixture",
        "--run-dir", str(run_dir),
    ])
    error = capsys.readouterr().err
    assert code == 2
    assert error.strip() == PLATFORM_REFUSAL
    assert not run_dir.exists()

    assert cli.main(["mutations", "selftest"]) == 2
    assert capsys.readouterr().err.strip() == PLATFORM_REFUSAL.removeprefix("stateframe: ")
    assert cli.main(["mutations", "enable-level", "prebuilt"]) == 2
    assert capsys.readouterr().err.strip() == PLATFORM_REFUSAL.removeprefix("stateframe: ")


def test_mutation_platform_gate_refuses_hook_attachments(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    run_dir = tmp_path / "hook-run"
    code = cli.main([
        "run", "--repo", str(tmp_path), "--task", "report a result",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "fixture",
        "--mutations", "off", "--mutation-hook", "observer:session_end",
        "--run-dir", str(run_dir),
    ])
    error = capsys.readouterr().err
    assert code == 2
    assert error.strip() == PLATFORM_REFUSAL
    assert not run_dir.exists()


def test_mutations_off_is_unchanged_on_non_linux(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    captured = {}

    def run_without_mutations(options, *, printer):
        captured["options"] = options
        return {"exit_code": 0}

    monkeypatch.setattr(cli, "run", run_without_mutations)
    code = cli.main([
        "run", "--repo", str(tmp_path), "--task", "report a result",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "fixture",
        "--mutations", "off", "--run-dir", str(tmp_path / "off-run"), "--json",
    ])
    from stateframe.runner import mutation_spec_options

    assert code == 0
    assert "MUTATION_PLATFORM_UNSUPPORTED" not in capsys.readouterr().err
    assert mutation_spec_options(captured["options"], {"STATEFRAME_MUTATIONS": "authored"}) is None


def test_non_linux_read_only_commands_report_platform_limit_without_selftest(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe"))
    from stateframe import mutations_cli

    monkeypatch.setattr(mutations_cli, "_selftest_report",
                        lambda _home: pytest.fail("doctor must not launch qualification probes"))
    for command in (["mutations", "list", "--json"],
                    ["mutations", "show", "file_facts", "--json"],
                    ["mutations", "doctor", "--json"]):
        assert cli.main(command) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["platform_limit"].startswith("MUTATION_PLATFORM_UNSUPPORTED")


def test_mutation_resume_refuses_non_linux_before_reconciliation(tmp_path, monkeypatch):
    from stateframe import runner
    from stateframe.resume import ResumeRefused

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({"mutations": {"level": "authored"}}))
    monkeypatch.setattr(runner.sys, "platform", "darwin")

    with pytest.raises(ResumeRefused, match="MUTATION_PLATFORM_UNSUPPORTED: mutations are Linux-only"):
        runner.resume_run(run_dir)


def test_macos_qualification_platform_identifier(tmp_path, monkeypatch):
    import kernel_mutate_qualify as module
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module, "MutationExecutor", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe unavailable")))
    assert module.selftest(tmp_path)["platform"] == "macos"
