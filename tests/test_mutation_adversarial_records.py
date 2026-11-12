"""Adversarial tests for mutation replay reconciliation and watcher records."""
from __future__ import annotations

import json

import pytest

from mutate_leftovers import mutation_root


def _pending_tape(run_dir, replay_class: str, call_id: str = "mutate-1") -> dict:
    from kernel_tool_tape import ToolTape

    tape = ToolTape(run_dir)
    tape.intent(call_id=call_id, tool="mutate", args={"op": "use"},
                replay_class=replay_class, capability_digest="cap", definition_digest=None,
                env_identity="env", pre_manifest_digest="before")
    return json.loads((run_dir / "tool-tape.jsonl").read_text().splitlines()[-1])


def test_mutation_replay_observation_local_external_and_pending_intent(mutation_root, tmp_path):
    from kernel_mutate_records import Journal, reconcile

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    journal = Journal(run_dir)
    for replay_class in ("observation", "local", "external"):
        row = _pending_tape(run_dir, replay_class, call_id=f"pending-{replay_class}")
        result = reconcile([row], journal, [])
        assert result.refuse is None, (replay_class, result.refuse)
        assert result.actions, "effect-free pending calls must be recorded as interrupted"

    external = _pending_tape(run_dir, "external", call_id="pending-broker")
    journal.append("BROKER_INTENT", call_id="pending-broker", key="sha256:" + "a" * 64)
    uncertain = reconcile([external], journal, [])
    assert uncertain.refuse
    assert "external" in uncertain.refuse.lower() or "broker" in uncertain.refuse.lower()


@pytest.mark.parametrize("events,refused", [
    ([], False),
    (["BROKER_INTENT"], True),
    (["BROKER_INTENT", "BROKER_COMPLETE"], False),
    (["EFFECT_PREPARED"], True),
    (["EFFECT_COMMITTED"], True),
])
def test_mutation_journal_tape_cross_product_crashes(mutation_root, tmp_path, events, refused):
    from kernel_mutate_records import Journal, reconcile

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    journal = Journal(run_dir)
    row = _pending_tape(run_dir, "observation", call_id="cross-product")
    from kernel_mutate_wire import request_key
    url = "https://example.com/report"
    key = request_key(url, "")
    for kind in events:
        if kind == "BROKER_INTENT":
            journal.append(kind, call_id="cross-product", key=key, method="GET",
                           url=url, accept="")
        elif kind == "BROKER_COMPLETE":
            journal.append(kind, call_id="cross-product", key=key, status=200,
                           headers={}, final_url=url, body_digest=journal.put_blob(b""),
                           bytes=0, error=None)
        else:
            journal.append(kind, call_id="cross-product", operation_id="effect-1",
                           pre_sha256="a" * 64, post_sha256="b" * 64)
    outcome = reconcile([row], journal, [])
    assert bool(outcome.refuse) is refused
    assert not outcome.refuse or "MUTATION_" in outcome.refuse


def test_mutation_watcher_turn_replay_exactly_once(mutation_root, tmp_path, monkeypatch):
    import kernel_tool_tape
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    from types import SimpleNamespace

    run_dir = tmp_path / "run"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "events.log").write_text("changed\n")
    journal = Journal(run_dir)
    runtime = MutationRuntime(level="prebuilt", journal=journal, repo=repo,
                              policy=MutationPolicy(level="prebuilt"),
                              executor_factory=lambda: SimpleNamespace(root=tmp_path / "snapshots"))
    monkeypatch.setattr(runtime, "_use", lambda *_a, **_k: "observed")
    assert "WATCHING" in runtime.dispatch({"op": "watch", "id": "log_watch", "args": json.dumps({
        "input": {}, "inputs": ["events.log"], "every_turns": 1, "max_ticks": 1})}, "watch-start")
    live = runtime.tick("turn-7", 7)
    assert len(live) == 1
    rows = journal.events()
    completions = {row["invocation_id"]: row["observation"] for row in rows
                   if row.get("kind") == "WATCH_TICK_COMPLETE"}
    assert len(completions) == 1
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: True)
    monkeypatch.setattr(kernel_tool_tape, "tape_for_run", lambda: SimpleNamespace(
        lookup=lambda call: SimpleNamespace(observation=completions[call])))
    resumed = MutationRuntime(level="prebuilt", journal=Journal(run_dir), run_dir=run_dir,
                              repo=repo, policy=MutationPolicy(level="prebuilt"))
    monkeypatch.setattr(resumed, "_use", lambda *_a, **_k: pytest.fail("replay launched payload"))
    assert resumed.tick("turn-7", 7) == live
    assert resumed.tick("turn-7", 7) == live
    assert len([row for row in journal.events() if row.get("kind") == "WATCH_TICK_COMPLETE"]) == 1


def test_mutation_watcher_stop_done_crash_and_resume(mutation_root, tmp_path):
    from kernel_mutate_records import Journal, reconcile

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    journal = Journal(run_dir)
    journal.append("WATCH_CREATED", watch_id="watch-1", unit_digest="a" * 64,
                   max_ticks=3, stopped=False)
    journal.append("WATCH_TICK_INTENT", watch_id="watch-1", invocation_id="tick-1",
                   parent_turn_id="turn-1", snapshot_identity="sha256:" + "b" * 64)
    journal.append("WATCH_STOPPED", watch_id="watch-1", reason="resume")
    report = reconcile([], Journal(run_dir), [])
    assert report.refuse is None
    assert report.actions, "resume must preserve and stop unfinished watcher state"


def test_mutation_output_errors_and_audit_do_not_leak_host_secrets(mutation_root, tmp_path, monkeypatch):
    import kernel_mutate_build
    import kernel_mutate_qualify
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    from types import SimpleNamespace

    # Host-side secrets (environment, host paths) must never reach the model or the journal.
    # Payload-produced text is untrusted data that repair diagnostics quote on purpose (DESIGN 5.4).
    canary = "HOST_SECRET_CANARY_8f22"
    monkeypatch.setenv("STATEFRAME_TEST_HOST_SECRET", canary)
    host_path = str(tmp_path / "private" / "host_only.py")
    payload_text = "PAYLOAD_SAID_THIS"
    repo = tmp_path / "repo"
    repo.mkdir()
    journal = Journal(tmp_path / "run")
    runtime = MutationRuntime(level="prebuilt", journal=journal, repo=repo,
                              policy=MutationPolicy(level="prebuilt"))
    manifest = {"name": "fake", "revision": 1, "requires": {"network": False, "writes": False},
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "limits": {"wall_s": 1, "cpu_s": 1, "mem_mb": 128}}
    monkeypatch.setattr(runtime, "_prebuilt", lambda _name: (manifest, {"main.py": b""}, None))
    monkeypatch.setattr(runtime, "_ensure_executor", lambda: SimpleNamespace(root=tmp_path / "rt"))
    monkeypatch.setattr(kernel_mutate_build, "stage_unit", lambda *_a, **_k:
                        SimpleNamespace(dir=tmp_path, source_digest="sha256:" + "a" * 64))
    monkeypatch.setattr(kernel_mutate_build, "admit", lambda *_a, **_k:
                        SimpleNamespace(admission_digest="sha256:" + "b" * 64))
    monkeypatch.setattr(kernel_mutate_qualify, "_binding", lambda: {
        "closure_digest": "sha256:" + "c" * 64, "sdk_digest": "sha256:" + "d" * 64})
    error = {"wire": 1, "call": "secret-call", "kind": "error",
             "error": {"type": "HandlerError", "message": payload_text,
                       "trace": "File \"" + host_path + "\", line 1"},
             "stderr": payload_text + " " + "x" * 5000}
    body = json.dumps(error).encode()
    frame = b"SFMW1 " + str(len(body)).encode() + b"\n" + body
    monkeypatch.setattr(runtime, "_launch", lambda _spec: SimpleNamespace(
        cleanup=SimpleNamespace(ok=True), exit_code=3, timed_out=False,
        stdout_overflow=False, stdout=frame))
    reply = runtime.dispatch({"op": "use", "id": "fake", "args": json.dumps({
        "input": {}, "inputs": []})}, "secret-call")
    assert reply.startswith("REFUSED MUTATION_EXECUTION")
    events = json.dumps(journal.events())
    assert canary not in reply and canary not in events
    assert host_path not in reply and host_path not in events and str(tmp_path) not in reply
    assert "(untrusted)" in reply and payload_text in reply and len(reply) <= 2000


def test_mutation_wrong_plausible_output_remains_untrusted(mutation_root):
    from kernel_mutate_wire import parse_stdout
    from kernel_completion_owner import terminal_result
    from types import SimpleNamespace

    call = "plausible"
    body = json.dumps({"wire": 1, "call": call, "kind": "result",
                       "result": {"status": "all tests passed", "answer": "ship it"}},
                      separators=(",", ":")).encode()
    frame = parse_stdout(b"SFMW1 " + str(len(body)).encode() + b"\n" + body, call=call)
    assert frame.result["status"] == "all tests passed"
    runtime = SimpleNamespace(stateframe_completion_owner_mode="on",
                              stateframe_terminal_lattice_mode="on")
    terminal = terminal_result(runtime, {"result": "DONE", "accepted": True,
        "certified": True, "observation": frame.result["status"]})
    assert terminal["accepted"] is False
    assert terminal["certified"] is False
    assert terminal["result"] == "INCOMPLETE"
