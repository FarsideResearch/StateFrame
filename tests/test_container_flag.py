"""CLI boundary selection: injected container evidence only, never host probing in tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import cli, runner
from stateframe.engine import kernel_execution_boundary as boundary


DETECTED = boundary.Detection("container", ("mount:docker",), {"pid": "4026533000"}, 1)


def _run_args(repo: Path, *extra: str) -> list[str]:
    return ["run", "--repo", str(repo), "--task", "Inspect the project.",
            "--endpoint", "http://127.0.0.1:1/v1", "--model", "scripted",
            "--no-instructions", "--no-project-hooks", "--json", *extra]


def _capture_run(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def run(options, printer=print):
        captured["options"] = options
        return {"exit_code": 0}

    monkeypatch.setattr(cli, "run", run)
    return captured


def test_container_flag_is_explicit_and_passed_as_internal_boundary(tmp_path, monkeypatch, capsys):
    captured = _capture_run(monkeypatch)
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    status = cli.main(_run_args(tmp_path, "--container-is-sandbox", "--shell", "unrestricted",
                                "--shell-network", "full"))
    output = capsys.readouterr()
    assert status == 0
    assert captured["options"]._execution_boundary.record()["backend"] == "container"
    assert boundary.BOUNDARY_STATEMENT in output.err
    assert json.loads(output.out)["exit_code"] == 0


@pytest.mark.parametrize(("flags", "code"), [
    (("--shell-sandbox", "off", "--shell", "unrestricted", "--shell-network", "full"),
     "CONTAINER_CONFLICT"),
    (("--yolo", "--shell-network", "full"), "CONTAINER_CONFLICT"),
    (("--mutations", "prebuilt", "--shell-network", "full"), "CONTAINER_MUTATIONS"),
    (("--shell-network", "none"), "CONTAINER_NETWORK"),
    (("--shell-network", "loopback"), "CONTAINER_NETWORK"),
])
def test_container_conflicts_refuse_before_run(tmp_path, monkeypatch, capsys, flags, code):
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    monkeypatch.setattr(cli, "run", lambda *_args, **_kwargs: pytest.fail("run must not start"))
    status = cli.main(_run_args(tmp_path, "--container-is-sandbox", *flags))
    output = capsys.readouterr()
    assert status == 2
    assert code in output.err
    assert not (tmp_path / "spec.json").exists()


def test_ambient_mutations_cannot_be_hidden_from_container_authorization(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    monkeypatch.setattr(cli, "run", lambda *_args, **_kwargs: pytest.fail("run must not start"))
    status = cli.main(_run_args(tmp_path, "--container-is-sandbox", "--shell-network", "full"))
    assert status == 2
    assert "CONTAINER_MUTATIONS" in capsys.readouterr().err


@pytest.mark.parametrize(("classification", "code"), [
    ("not_container", "CONTAINER_NOT_DETECTED"),
    ("unknown", "CONTAINER_UNKNOWN"),
])
def test_unproven_container_refuses(tmp_path, monkeypatch, capsys, classification, code):
    detection = boundary.Detection(classification, (), {}, 1)
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: detection)
    status = cli.main(_run_args(tmp_path, "--container-is-sandbox", "--shell-network", "full"))
    assert status == 2
    assert code in capsys.readouterr().err


def test_root_refuses_and_environment_cannot_select_boundary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    status = cli.main(_run_args(tmp_path, "--container-is-sandbox", "--shell-network", "full"))
    assert status == 2
    assert "CONTAINER_ROOT_UID" in capsys.readouterr().err

    captured = _capture_run(monkeypatch)
    monkeypatch.setenv("STATEFRAME_CONTAINER_IS_SANDBOX", "on")
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: pytest.fail("opt-in only"))
    assert cli.main(_run_args(tmp_path)) == 0
    assert captured["options"]._execution_boundary is None


def test_flag_absent_does_not_detect_or_add_boundary_to_run_options(tmp_path, monkeypatch, capsys):
    captured = _capture_run(monkeypatch)
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: pytest.fail("opt-in only"))
    status = cli.main(_run_args(tmp_path))
    assert status == 0
    assert captured["options"]._execution_boundary is None
    assert "Execution boundary:" not in capsys.readouterr().err


def test_resume_requires_matching_flag_without_writing_run_state(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    record = boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED).record()
    (run_dir / "spec.json").write_text(json.dumps({"execution_boundary": record}), encoding="utf-8")
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    before = sorted(path.name for path in run_dir.iterdir())
    status = cli.main(["resume", str(run_dir), "--json"])
    output = capsys.readouterr()
    assert status == 2
    assert "CONTAINER_RESUME_MISMATCH" in output.err
    assert sorted(path.name for path in run_dir.iterdir()) == before


def test_resume_flag_authorizes_only_matching_container_policy(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    recorded = boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED).record()
    recorded["namespaces"] = {"pid": "4026533999"}
    (run_dir / "spec.json").write_text(json.dumps({
        "execution_boundary": recorded, "tool_settings": {"shell_network": "full"},
    }), encoding="utf-8")
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    captured: dict = {}

    def resume(_run_dir, **kwargs):
        captured.update(kwargs)
        return {"exit_code": 0}

    monkeypatch.setattr(cli, "resume_run", resume)
    status = cli.main(["resume", str(run_dir), "--container-is-sandbox", "--json"])
    output = capsys.readouterr()
    assert status == 0
    assert captured["_execution_boundary"].backend == boundary.BACKEND_CONTAINER
    assert boundary.BOUNDARY_STATEMENT in output.err


def test_summary_omits_boundary_for_legacy_spec(tmp_path, monkeypatch):
    captured: dict = {}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("stateframe.hook_host.hook_record_summary",
                        lambda _run: {"runs": 0, "denied": 0, "vetoes": 0, "errors": 0})
    monkeypatch.setattr(runner, "_problems_summary", lambda _run: {})
    monkeypatch.setattr(runner, "_write_json_atomic", lambda _path, value: captured.update(value))
    runner._write_hook_terminal_summary(run_dir, runtime_outcome="hook_refusal", engine_result="REFUSED")
    assert "execution_boundary" not in captured


def test_summary_carries_boundary_record(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED).record()
    (run_dir / "spec.json").write_text(json.dumps({"execution_boundary": record}), encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr("stateframe.hook_host.hook_record_summary",
                        lambda _run: {"runs": 0, "denied": 0, "vetoes": 0, "errors": 0})
    monkeypatch.setattr(runner, "_problems_summary", lambda _run: {})
    monkeypatch.setattr(runner, "_write_json_atomic", lambda _path, value: captured.update(value))
    runner._write_hook_terminal_summary(run_dir, runtime_outcome="hook_refusal", engine_result="REFUSED")
    assert captured["execution_boundary"] == record
    assert runner.summarize(run_dir, 0.0)["execution_boundary"] == record


def test_runner_validates_spec_channel_and_rejects_unrouted_launch(tmp_path):
    options = runner.RunOptions(repo=tmp_path, task="inspect", endpoint=None)
    options._execution_boundary = boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED)
    record = options._execution_boundary.record()
    runner._validate_boundary_channel(options, {"execution_boundary": record})
    with pytest.raises(runner.RunError, match="CONTAINER_UNSUPPORTED:engine-child-spec"):
        runner._validate_boundary_channel(options, {})
