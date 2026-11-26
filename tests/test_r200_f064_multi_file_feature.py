"""Regression coverage for schedule-witness scratch during CLI in-place promotion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from stateframe import cli, runner
from stateframe.scripted_endpoint import Script, serve


def test_schedule_witness_opt_in(monkeypatch):
    switch = "STATEFRAME_SCHEDULE_WITNESS"
    monkeypatch.delenv(switch, raising=False)
    assert switch not in runner.resolve_opt_in_switches()
    for value in ("off", "on"):
        monkeypatch.setenv(switch, value)
        assert runner.resolve_opt_in_switches()[switch] == value
    monkeypatch.setenv(switch, "invalid")
    with pytest.raises(runner.RunError, match=switch):
        runner.resolve_opt_in_switches()


@pytest.mark.parametrize("engine_reader", [False, True], ids=["shipped", "with-engine-reader"])
def test_cli_in_place_preserves_preexisting_scratch_when_witness_is_enabled(tmp_path, monkeypatch, engine_reader):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("STATEFRAME_SCHEDULE_WITNESS", "on")
    if engine_reader:
        switches = runner.engine_switches() | {"STATEFRAME_SCHEDULE_WITNESS"}
        monkeypatch.setattr(runner, "engine_switches", lambda *_args: switches)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "selection.py").write_text("value = 'before'\n", encoding="utf-8")
    (repo / "check.py").write_text(
        "from selection import value\n"
        "assert value == 'after'\n",
        encoding="utf-8",
    )
    scratch = repo / ".test-tmp" / "prior-run.txt"
    scratch.parent.mkdir()
    scratch.write_text("scratch from a prior pytest run\n", encoding="utf-8")

    run = tmp_path / "run"
    contents = b"value = 'after'\n"
    artifact_digest = "sha256:" + "a" * 64
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": artifact_digest, "artifact_digest": artifact_digest,
    }}

    def drive(run_dir: Path, *_args):
        # Model a witness workspace that omits scratch before both host digest scans.
        (run_dir / "workspace" / scratch.relative_to(repo)).unlink()
        (run_dir / "engine-result.json").write_text(json.dumps({"result": terminal}), encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "drive", drive)
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "engine_terminal": terminal, "engine_result": "SUBMITTED", "cached_share_display": "cache",
    })
    monkeypatch.setattr(runner, "final_candidate", lambda _run: {"selection.py": contents})
    monkeypatch.setattr(runner, "_artifact_digest", lambda _candidate: artifact_digest)
    monkeypatch.setattr(runner, "finish_run_storage", lambda *_args, **_kwargs: {
        "checkpoints": None, "env": None,
    })

    server = serve(Script({"model": "scripted", "steps": []}))
    try:
        status = cli.main([
            "run", "--repo", str(repo), "--task", "Update selection.py.",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run), "--in-place", "--no-project-hooks", "--no-instructions",
            "--no-run-env", "--test-python", sys.executable,
            "--check-cmd", "python check.py", "--target", "selection.py", "--max-turns", "1", "--quiet",
        ])
    finally:
        server.shutdown()
        server.server_close()

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    assert status == 0
    workspace = run / "workspace"
    candidate_digests = {}
    digest = runner.candidate_digest(workspace, file_digests=candidate_digests)
    assert digest == summary["tests_after"]["candidate_sha256"]
    assert scratch.relative_to(repo).as_posix() not in candidate_digests
    assert "_schedule_witness" not in spec["public_row"]
    assert summary["succeeded"] is True
    assert summary["applied_in_place"] == ["selection.py"]
    assert (repo / "selection.py").read_bytes() == contents
    assert scratch.read_text(encoding="utf-8") == "scratch from a prior pytest run\n"
    if "STATEFRAME_SCHEDULE_WITNESS" in runner.engine_switches():
        assert spec["environment"]["STATEFRAME_SCHEDULE_WITNESS"] == "on"
    else:
        assert spec["schedule_witness"] == "on"
        assert "STATEFRAME_SCHEDULE_WITNESS" not in spec["environment"]
