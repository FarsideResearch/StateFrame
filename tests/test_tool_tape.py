import pytest

from kernel_tool_tape import ToolTape, ToolTapeError, replay_decision, verify_intent, verify_reexecution
from stateframe.resume import ResumeRefused, assess
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
