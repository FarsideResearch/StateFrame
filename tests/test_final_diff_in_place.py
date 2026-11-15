"""In-place promotion depends on candidate changes, independently of diff export."""
from __future__ import annotations

import json
import sys

import pytest

from stateframe import runner
from stateframe.endpoint import Endpoint


@pytest.mark.parametrize("refused,changed", [(False, True), (True, True), (False, False)])
def test_in_place_promotion_uses_candidate_changes(tmp_path, monkeypatch, refused, changed):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("value = 1\n", encoding="utf-8")
    options = runner.RunOptions(
        repo=repo, task="update sample", in_place=True, run_dir=tmp_path / "run",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1, test_cmd="true", runner_kind="generic", check_source="command",
        target="sample.py", no_project_hooks=True,
    )
    run, work, spec = runner.prepare(options, printer=lambda *_args: None)
    digest = "sha256:" + "a" * 64
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": digest, "artifact_digest": digest,
    }}

    def drive(*_args):
        (run / "engine-result.json").write_text(json.dumps({"result": terminal}))
        return 0

    monkeypatch.setattr(runner, "drive", drive)
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "engine_terminal": terminal, "engine_result": "SUBMITTED", "cached_share_display": "cache",
    })
    contents = b"value = 2\n" if changed else b"value = 1\n"
    monkeypatch.setattr(runner, "final_candidate", lambda _run: {"sample.py": contents})
    monkeypatch.setattr(runner, "_artifact_digest", lambda _candidate: digest)
    monkeypatch.setattr(runner, "_tests_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "_outer_test_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "run_test_cmd", lambda *_args, **_kwargs: {
        "command": "true", "exit_code": 0, "passed": True, "tail": "ok",
    })
    monkeypatch.setattr(runner, "finish_run_storage", lambda *_args, **_kwargs: {
        "checkpoints": None, "env": None,
    })
    if refused:
        monkeypatch.setattr(runner, "_export_final_diff", lambda *_args: "STATEFRAME_HOME_IDENTITY_CHANGED")

    summary = runner._execute(options, 0.0, lambda *_args: None, run, work, spec)

    assert summary["runtime_outcome"] == "completed"
    assert summary["succeeded"] is True
    assert summary.get("applied_in_place", []) == (["sample.py"] if changed else [])
    assert (repo / "sample.py").read_bytes() == contents
    if refused:
        assert summary["final_diff_error"] == "STATEFRAME_HOME_IDENTITY_CHANGED"
        assert not (run / "final.diff").exists()
