"""Startup metering recovery at every durable archive boundary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_agent_schemas import measure_schemas
from kernel_agent_tools import AgentTools
from kernel_continuation_journal import ContinuationJournal
from kernel_custom_tools import CustomToolSet
from kernel_tool_settings import ToolSettings
from stateframe import profiles, resume


ARCHIVE_BOUNDARIES = ["mkdir", "copy-spec", *resume.ATTEMPT_FILES, "checkpoints", "publish-spec"]


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "parent").mkdir()
    spec = {"schema": "stateframe-run/1", "environment": {"REACT_STEPS": "8"},
            "public_row": {"msg": "task"}, "parent_snapshot": str(run / "parent"),
            "chat_url": "http://127.0.0.1:9/v1/chat/completions", "model": "scripted",
            "max_turns": 8, "repo": str(tmp_path), "run_identity_seed": "original-attempt",
            "engine_source": profiles.engine_source_summary()}
    (run / "spec.json").write_text(json.dumps(spec))
    directories = {"worker", "wire", "requests", "workspace", "checks", "env", "shell-home",
                   "shell-cache", "hooks"}
    for name in resume.ATTEMPT_FILES:
        path = run / name
        if name in directories:
            path.mkdir()
        else:
            path.write_text("" if name == "sends.jsonl" else "{}")
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}))
    (run / "resume-divergence.json").unlink()
    (run / "checkpoints").mkdir()
    budget = {"reason": "deadline", "counts": measure_schemas()}
    journals = run / "worker" / "measurement-runs"
    journals.mkdir()
    journal = ContinuationJournal(journals, "task", {"startup_schema_budget": budget})
    journal.close()
    wire = b'{"messages": [], "model": "scripted"}'
    (run / "wire" / "0001.request.json").write_bytes(wire)
    (run / "wire" / "0001.response.json").write_text('{"choices": [{"message": {"tool_calls": []}}]}')
    return run, budget, wire


@pytest.mark.parametrize("boundary", ARCHIVE_BOUNDARIES)
def test_archive_interruption_preserves_schema_record(tmp_path, monkeypatch, boundary):
    run, budget, wire = make_run(tmp_path)
    # This marker is meaningful only to carried replay. Include it to exercise its move.
    if boundary == "resume-divergence.json":
        (run / boundary).write_text("{}")
    real_replace = resume.os.replace
    real_copy = resume.shutil.copy2
    real_mkdir = Path.mkdir
    hit = []

    def stop(label):
        if boundary == label:
            hit.append(label)
            raise KeyboardInterrupt(label)

    def replace(source, target):
        real_replace(source, target)
        source, target = Path(source), Path(target)
        if source.parent == run and target.parent == run / "attempts" / "1":
            stop(source.name)
        if target == run / "spec.json":
            stop("publish-spec")

    def copy(source, target, *args, **kwargs):
        result = real_copy(source, target, *args, **kwargs)
        if Path(target) == run / "attempts" / "1" / "spec.json":
            stop("copy-spec")
        return result

    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path == run / "attempts" / "1":
            stop("mkdir")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(resume.os, "replace", replace)
        patch.setattr(resume.shutil, "copy2", copy)
        patch.setattr(Path, "mkdir", mkdir)
        with pytest.raises(KeyboardInterrupt, match=boundary):
            resume.archive_attempt(resume.assess(run), tmp_path, use_checkpoint=False)
    assert hit == [boundary]
    state = resume.assess(run)
    assert state["tape_count"] == 1
    resume.archive_attempt(state, tmp_path, use_checkpoint=False)
    spec = json.loads((run / "spec.json").read_text())
    request = Path(spec["resume"]["tape"][0]["request"])
    assert request.read_bytes() == wire
    shell = SimpleNamespace(spec=spec, run_dir=run, settings=ToolSettings())
    tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
    tools.bind()
    assert tools.startup_schema_budget == budget


def test_live_restart_interrupted_after_tool_tape_move_keeps_external_authorization(tmp_path, monkeypatch):
    from kernel_tool_tape import ToolTape

    run, _, _ = make_run(tmp_path)
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "BUDGET_EXHAUSTED"}}))
    tape = ToolTape(run)
    tape.intent(call_id="call-1", tool="remote", args={}, replay_class="external",
                capability_digest="cap", definition_digest="definition", env_identity="env",
                pre_manifest_digest="before")
    tape.complete(call_id="call-1", exit_code=0, observation="effect completed",
                  post_manifest_digest="after", env_identity="env")
    real_replace = resume.os.replace
    stopped = []

    def crash(source, target):
        real_replace(source, target)
        if Path(source) == run / "tool-tape.jsonl":
            stopped.append("tool-tape")
            raise KeyboardInterrupt("tool-tape")

    with monkeypatch.context() as patch:
        patch.setattr(resume.os, "replace", crash)
        with pytest.raises(KeyboardInterrupt, match="tool-tape"):
            resume.archive_attempt(resume.assess(run, max_turns=9, replay_external=True),
                                   tmp_path, use_checkpoint=False, replay_external=True)
    assert stopped == ["tool-tape"]
    # The interrupted move is still the active attempt, so its external records
    # need the same authorization they needed before archival.
    state = resume.assess(run, max_turns=9, replay_external=True)
    with pytest.raises(resume.ResumeRefused, match="requires --replay-external"):
        resume.assess(run, max_turns=9)
    resume.archive_attempt(state, tmp_path, use_checkpoint=False, replay_external=True)
    saved = json.loads((run / "spec.json").read_text())["resume"]
    assert saved["restart_live"] is True
    assert saved["archived_external_operations"] == [{"call_id": "call-1", "tool": "remote"}]
    assert not (run / "tool-tape.jsonl").exists()
