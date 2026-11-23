"""The CLI requires a stored artifact before reporting a delivered submission."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel_done_is_submit_authority import capture_and_seal, terminal_from_seal

from stateframe import cli, runner
from stateframe.scripted_endpoint import Script, serve


_SOURCE = "sample.py"
_BEFORE = "value = 1\n"
_AFTER = "value = 2\n"


@pytest.fixture(scope="module")
def endpoint():
    with serve(Script({"model": "scripted"})) as server:
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys, endpoint):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / _SOURCE).write_text(_BEFORE, encoding="utf-8")
    run = tmp_path / "run"
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")

    def invoke(script):
        # Script the child output at the delivery boundary. CLI preparation, summary
        # projection, in-place application, and final diff export all run normally.
        def drive(run_dir, _options, _printer):
            terminal = script(run_dir)
            (run_dir / "engine-result.json").write_text(
                json.dumps({"result": terminal}), encoding="utf-8")
            return 0

        monkeypatch.setattr(runner, "drive", drive)
        exit_code = cli.main([
            "run", "--repo", str(repo), "--task", "Review the sample", "--target", _SOURCE,
            "--run-dir", str(run), "--endpoint", endpoint, "--model", "scripted",
            "--shell", "off", "--no-project-hooks", "--max-turns", "1", "--in-place", "--json",
        ])
        output = capsys.readouterr()
        summary = json.loads(output.out)
        assert "error" not in summary, (summary, output.err)
        return exit_code, summary

    return invoke, repo, run


def _store_submission(run: Path, source: str) -> tuple[dict[str, object], Path]:
    destination = run / "worker/results/react_done_submissions"
    runtime = SimpleNamespace()
    seal = capture_and_seal(
        runtime, destination=destination, task_id="sample",
        sources={_SOURCE: source}, parent_sources={_SOURCE: _BEFORE},
    )
    terminal = terminal_from_seal(runtime)
    assert terminal is not None
    terminal["submission"] = terminal.pop("done_seal")
    return terminal, destination / "sample" / f"{seal.artifact_digest.removeprefix('sha256:')}.json"


def _write_partial_capture(run: Path) -> None:
    path = run / "worker/candidates/0001.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "files_base64": {_SOURCE: base64.b64encode(b"value = 3\n").decode("ascii")},
        "deleted_files": [],
    }), encoding="utf-8")


@pytest.mark.parametrize("partial_capture", [False, True], ids=["no-capture", "unrelated-capture"])
def test_cli_refuses_missing_sealed_artifact(run_cli, partial_capture):
    invoke, repo, run = run_cli

    def script(run_dir):
        terminal, artifact = _store_submission(run_dir, _AFTER)
        artifact.unlink()
        if partial_capture:
            _write_partial_capture(run_dir)
        return terminal

    exit_code, summary = invoke(script)

    assert (repo / _SOURCE).read_bytes() == _BEFORE.encode()
    assert (run / "final.diff").read_bytes() == b""
    assert summary["files_changed"] == []
    assert summary["engine_result"] == "DELIVERY_REFUSED"
    assert exit_code == summary["exit_code"] == 2
    assert summary["succeeded"] is False
    assert summary["candidate_captured"] is False
    assert summary["artifact"] == "partial"
    assert summary["check_outcome"] == "not_run"
    assert summary["engine_terminal"]["result"] == "SUBMITTED"
    assert summary["delivery_refusal"] == {
        "engine_result": "SUBMITTED",
        "reason_code": "CANDIDATE_NOT_DELIVERED",
        "detail": "sealed candidate artifact missing or ambiguous; candidate remains partial",
    }
    assert summary["candidate_error"] == summary["delivery_refusal"]["detail"]


@pytest.mark.parametrize("source", [_AFTER, _BEFORE], ids=["matching-artifact", "no-change-artifact"])
def test_cli_accepts_verified_sealed_artifact(run_cli, source):
    invoke, repo, run = run_cli

    def script(run_dir):
        terminal, _artifact = _store_submission(run_dir, source)
        return terminal

    exit_code, summary = invoke(script)

    assert exit_code == summary["exit_code"] == 0
    assert summary["succeeded"] is True
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["artifact"] == "saved"
    assert summary["candidate_error"] is None
    seal = summary["engine_terminal"]["submission"]
    assert summary["candidate_artifact_digest"] == seal["artifact_digest"]
    assert (repo / _SOURCE).read_bytes() == source.encode()
    diff = (run / "final.diff").read_bytes()
    if source == _BEFORE:
        assert seal["label"] == "no_change"
        assert summary["files_changed"] == []
        assert diff == b""
    else:
        assert seal["label"] == "submitted"
        assert summary["files_changed"] == [_SOURCE]
        assert b"+" + _AFTER.encode() in diff


def test_cli_preserves_unsealed_partial_capture(run_cli):
    invoke, repo, run = run_cli

    def script(run_dir):
        _write_partial_capture(run_dir)
        return {"result": "RUN_INCOMPLETE"}

    exit_code, summary = invoke(script)

    assert exit_code == summary["exit_code"] == 1
    assert summary["succeeded"] is False
    assert summary["engine_result"] == "RUN_INCOMPLETE"
    assert summary["runtime_outcome"] == "incomplete"
    assert summary["candidate_captured"] is True
    assert summary["candidate_error"] is None
    assert summary["artifact"] == "partial"
    assert summary["files_changed"] == [_SOURCE]
    assert (repo / _SOURCE).read_bytes() == _BEFORE.encode()
    assert b"+value = 3\n" in (run / "final.diff").read_bytes()
