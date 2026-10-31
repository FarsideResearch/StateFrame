import pytest

from kernel_tool_tape import ToolTape, ToolTapeError, replay_decision, verify_intent, verify_reexecution
from stateframe.resume import ResumeRefused, archive_attempt, assess
from stateframe import profiles
import json


@pytest.mark.parametrize("replay_class,allowed,serve,reexecute,refused", [
    ("observation", False, True, False, False),
    ("local", False, True, True, False),
    ("external", False, False, False, True),
    ("external", True, True, True, False),
])
def test_replay_classes(tmp_path, replay_class, allowed, serve, reexecute, refused):
    tape = ToolTape(tmp_path)
    tape.intent(call_id="stable", tool="shell", args={"command": "echo hi"},
                replay_class=replay_class, capability_digest="cap", definition_digest=None,
                env_identity="before", pre_manifest_digest="pre")
    tape.complete(call_id="stable", exit_code=0, observation="hi", post_manifest_digest="post",
                  env_identity="after")
    record = ToolTape(tmp_path).lookup("stable")
    decision = replay_decision(record, replay_external_allowed=allowed)
    assert (decision.serve_taped, decision.reexecute, bool(decision.refuse_reason)) == (
        serve, reexecute, refused)
    assert verify_reexecution(record, exit_code=0, post_manifest_digest="post", env_identity="after") is None
    identity = dict(tool="shell", args={"command": "echo hi"}, replay_class=replay_class,
                    capability_digest="cap", definition_digest=None, env_identity="before",
                    pre_manifest_digest="pre")
    assert verify_intent(record, **identity) is None
    assert "arguments" in verify_intent(record, **{**identity, "args": {"command": "other"}})
    assert "exit code" in verify_reexecution(record, exit_code=1, post_manifest_digest="post", env_identity="after")
    assert "workspace manifest" in verify_reexecution(record, exit_code=0, post_manifest_digest="other", env_identity="after")
    assert "environment identity" in verify_reexecution(record, exit_code=0, post_manifest_digest="post", env_identity="other")


def test_intent_without_completion_refuses(tmp_path):
    tape = ToolTape(tmp_path)
    tape.intent(call_id="stable", tool="deploy", args={}, replay_class="external",
                capability_digest="cap", definition_digest="def", env_identity="env",
                pre_manifest_digest="pre")
    record = ToolTape(tmp_path).lookup("stable")
    assert "may or may not have happened" in replay_decision(record, replay_external_allowed=True).refuse_reason
    assert ToolTape(tmp_path).pending() == (record,)
    with pytest.raises(ToolTapeError, match="duplicate"):
        tape.intent(call_id="stable", tool="deploy", args={}, replay_class="external",
                    capability_digest="cap", definition_digest="def", env_identity="env",
                    pre_manifest_digest="pre")


def test_corrupt_intent_cannot_claim_completion(tmp_path):
    (tmp_path / "tool-tape.jsonl").write_text(
        '{"kind":"intent","call_id":"x","tool":"shell","args":{},"replay_class":"local",'
        '"capability_digest":"c","definition_digest":null,"env_identity":"e",'
        '"pre_manifest_digest":"p","exit_code":null,"observation":null,'
        '"post_manifest_digest":null,"post_env_identity":null,"completed":true}\n')
    with pytest.raises(ToolTapeError, match="invalid intent"):
        ToolTape(tmp_path)


def test_resume_refuses_uncertain_operation_before_replay(tmp_path):
    (tmp_path / "spec.json").write_text(json.dumps({"schema": "stateframe-run/1",
        "environment": {}, "public_row": {}, "parent_snapshot": str(tmp_path / "parent"),
        "chat_url": "http://127.0.0.1/v1/chat/completions", "model": "x", "max_turns": 1,
        "repo": str(tmp_path)}))
    tape = ToolTape(tmp_path)
    tape.intent(call_id="stable", tool="deploy", args={}, replay_class="external",
                capability_digest="cap", definition_digest="def", env_identity="env",
                pre_manifest_digest="pre")
    with pytest.raises(ResumeRefused, match="operation stable may or may not have happened"):
        assess(tmp_path)


def _limited_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "parent").mkdir()
    spec = {"schema": "stateframe-run/1", "environment": {"REACT_STEPS": "1"},
            "public_row": {}, "parent_snapshot": str(run / "parent"),
            "chat_url": "http://127.0.0.1/v1/chat/completions", "model": "scripted",
            "max_turns": 1, "repo": str(tmp_path), "engine_source": profiles.engine_source_summary()}
    (run / "spec.json").write_text(json.dumps(spec))
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "RESPONSE_BUDGET_EXHAUSTED"}}))
    return run, spec


def _operation(run, replay_class, *, complete=True):
    tape = ToolTape(run)
    tape.intent(call_id="same-id", tool="publish" if replay_class == "external" else "shell",
                args={}, replay_class=replay_class, capability_digest="cap", definition_digest="def",
                env_identity="env", pre_manifest_digest="pre")
    if complete:
        tape.complete(call_id="same-id", exit_code=0, observation="ok",
                      post_manifest_digest="post", env_identity="env")


def test_limit_restart_refuses_incomplete_before_archiving(tmp_path):
    run, spec = _limited_run(tmp_path)
    _operation(run, "external", complete=False)
    with pytest.raises(ResumeRefused, match="may or may not have happened"):
        assess(run, max_turns=8, replay_external=True)
    with pytest.raises(ResumeRefused, match="may or may not have happened"):
        archive_attempt({"run": run, "spec": spec, "attempt": 1, "max_turns": 8}, tmp_path,
                        replay_external=True)
    assert not (run / "attempts").exists()
    assert (run / "tool-tape.jsonl").is_file()


def test_limit_restart_requires_consent_and_preserves_external_history(tmp_path):
    run, spec = _limited_run(tmp_path)
    _operation(run, "external")
    with pytest.raises(ResumeRefused, match=r"same-id \(publish\).*--replay-external"):
        assess(run, max_turns=8)
    with pytest.raises(ResumeRefused, match="--replay-external"):
        archive_attempt({"run": run, "spec": spec, "attempt": 1, "max_turns": 8}, tmp_path)
    assert not (run / "attempts").exists()
    state = assess(run, max_turns=8, replay_external=True)
    archive = archive_attempt(state, tmp_path, replay_external=True)
    assert ToolTape(archive).lookup("same-id").replay_class == "external"
    assert state["spec"]["resume"]["archived_external_operations"] == [
        {"call_id": "same-id", "tool": "publish"}]
    assert not (run / "tool-tape.jsonl").exists()
    _operation(run, "local")  # A reused id belongs to the new attempt, not the archived tape.
    with pytest.raises(ResumeRefused, match="--replay-external"):
        assess(run)
    assert assess(run, replay_external=True)["external_operations"] == [
        {"call_id": "same-id", "tool": "publish"}]


def test_limit_restart_allows_completed_local_and_observation_ops(tmp_path):
    for replay_class in ("local", "observation"):
        run, spec = _limited_run(tmp_path / replay_class)
        _operation(run, replay_class)
        state = assess(run, max_turns=8)
        archive = archive_attempt(state, tmp_path)
        assert ToolTape(archive).lookup("same-id").replay_class == replay_class
        assert spec["resume"]["archived_external_operations"] == []
