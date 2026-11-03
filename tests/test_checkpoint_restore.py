"""Checkpoint state must preserve task data without replacing resume control."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
import zlib
from dataclasses import dataclass

import pytest

import kernel_bolt
import kernel_checkpoint as live_checkpoint
import kernel_continuation_journal
from kernel_continuation_workspace import ContinuationWorkspace, WorkspaceError
import kernel_request_diet
import kernel_wire_identity
import kernel_vacuity_truth
from stateframe import _child, replay_tape, resume, runner
from stateframe.engine import kernel_checkpoint, kernel_tool_tape
from stateframe.profiles import load
from scripts import checkpoint_restore_probe


def test_checkpoint_defaults_on_and_off_opts_out(monkeypatch):
    monkeypatch.delenv("STATEFRAME_CHECKPOINT", raising=False)
    assert live_checkpoint.mode() == "save"
    assert runner.checkpoint_env()["STATEFRAME_CHECKPOINT"] == "save"
    assert load("r2hc3")["STATEFRAME_CHECKPOINT"] == "on"
    assert runner.checkpoint_env(load("r2hc3"))["STATEFRAME_CHECKPOINT"] == "on"

    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "")
    assert live_checkpoint.mode() == "save"
    assert runner.checkpoint_env()["STATEFRAME_CHECKPOINT"] == "save"

    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    assert live_checkpoint.mode() == "off"
    assert runner.checkpoint_env()["STATEFRAME_CHECKPOINT"] == "off"
    assert runner.checkpoint_env(load("r2hc3"))["STATEFRAME_CHECKPOINT"] == "off"


@pytest.mark.parametrize("strict", [False, True])
def test_request_hex_task_change_diverges_on_recorded_tail(tmp_path, strict):
    recorded = b'{"model": "old", "task": "' + b"a" * 32 + b'"}'
    changed = recorded.replace(b"a" * 32, b"b" * 32)
    request = tmp_path / "0001.request.json"
    response = tmp_path / "0001.response.json"
    request.write_bytes(recorded)
    response.write_bytes(b'{"choices":[{}]}')
    tape = [{"send": 1, "request": str(request), "response": str(response),
             "request_digest": replay_tape.normalized_digest(recorded),
             "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest()}]
    pending = {"send": 2, "request": str(request),
               "request_digest": replay_tape.normalized_digest(recorded)}
    player = replay_tape.Replayer(tape, strict=strict, pending=pending)
    assert player.answer(changed) is None
    assert player.diverged is not None
    assert player.pending is None
    assert player.diverged["first_difference"]["recorded_window"] != player.diverged["first_difference"]["replayed_window"]


@pytest.mark.parametrize("strict", [False, True])
def test_model_name_swap_changes_only_top_level_field(tmp_path, strict):
    recorded = b'{"model": "old", "task": "' + b"a" * 32 + b'"}'
    served = recorded.replace(b'"old"', b'"new"', 1)
    request = tmp_path / "0001.request.json"
    response = tmp_path / "0001.response.json"
    request.write_bytes(recorded)
    response.write_bytes(b'{"choices":[{}]}')
    tape = [{"send": 1, "request": str(request), "response": str(response),
             "request_digest": replay_tape.normalized_digest(recorded),
             "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest()}]
    player = replay_tape.Replayer(tape, {"served": "new", "recorded": ["old"]}, strict=strict)
    assert player.answer(served) == response.read_bytes()
    assert player.model_swapped == 1
    assert player.diverged is None
    pending = {"send": 2, "request": str(request), "request_digest": tape[0]["request_digest"]}
    pending_player = replay_tape.Replayer([], {"served": "new", "recorded": ["old"]},
                                          strict=strict, pending=pending)
    assert pending_player.answer(served) is None
    assert pending_player.diverged is None
    assert pending_player.pending is None


def test_carry_forward_checks_archived_bytes_across_digest_change(tmp_path):
    current = tmp_path / "current.json"
    previous = tmp_path / "previous.json"
    current.write_bytes(b'{"model": "new", "task": "same"}')
    previous.write_bytes(b'{"model": "old", "task": "same"}')
    own = [{"send": 1, "request": str(current), "request_digest": "sha256:new"}]
    old = [{"send": 1, "request": str(previous), "request_digest": "sha256:legacy"},
           {"send": 2, "request": str(previous), "request_digest": "sha256:remaining"}]
    spec = {"resume": {"tape": old, "model_swap": {"served": "new", "recorded": ["old"]}}}
    assert resume.carry_forward(spec, own, False) == own + old[1:]
    current.write_bytes(current.read_bytes().replace(b"same", b"different"))
    assert resume.carry_forward(spec, own, False) == own


def test_carry_forward_checks_pending_bytes_before_reusing_older_response(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    pending_path = tmp_path / "pending.json"
    first.write_bytes(b'{"model": "scripted", "request": 1}')
    second.write_bytes(b'{"model": "scripted", "request": 2}')
    pending_path.write_bytes(second.read_bytes())
    own = [{"send": 1, "request": str(first)}]
    previous = own + [{"send": 2, "request": str(second)}]
    spec = {"resume": {"tape": previous}}
    pending = {"send": 2, "request": str(pending_path)}
    assert resume.carry_forward(spec, own, False, pending) == previous
    pending_path.write_bytes(b'{"model": "scripted", "request": 3}')
    assert resume.carry_forward(spec, own, False, pending) == own


def test_checkpoint_request_mismatch_refuses_with_operation_tape(tmp_path, monkeypatch):
    import kernel_bounded_http
    import kernel_tool_tape as live_tool_tape

    run = tmp_path / "run"
    run.mkdir()
    operation_tape = live_tool_tape.ToolTape(run)
    operation_tape.intent(call_id="call-1", tool="read", args={"path": "app.py"},
                          replay_class="observation", capability_digest="capability",
                          definition_digest=None, env_identity="environment",
                          pre_manifest_digest="manifest")
    operation_tape.complete(call_id="call-1", exit_code=0, observation="contents",
                            post_manifest_digest="manifest", env_identity="environment")
    assert live_tool_tape.ToolTape(run).lookup("call-1").completed
    recorded = b'{"model": "old", "task": "' + b"a" * 32 + b'"}'
    request = run / "old-request.json"
    request.write_bytes(recorded)
    pending = {"send": 1, "request": str(request),
               "request_digest": replay_tape.normalized_digest(recorded)}
    player = replay_tape.Replayer([], strict=True, pending=pending)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *_args, **_kwargs: b"unused")
    _child.install_wire_recorder(run, "http://127.0.0.1:1/v1/chat/completions", player)
    with pytest.raises(replay_tape.CheckpointUnfaithful):
        kernel_bounded_http.request_bytes(recorded.replace(b"a" * 32, b"b" * 32),
                                          "http://127.0.0.1:1/v1/chat/completions", "")
    assert json.loads((run / "checkpoint-refused.json").read_text())["error"] == "CheckpointUnfaithful"
    assert (run / "resume-divergence.json").is_file()
    assert "CheckpointUnfaithful" in runner._checkpoint_refusal(run)

    response = run / "old-response.json"
    response.write_bytes(b'{"choices":[{}]}')
    tape = [{"send": 1, "request": str(request), "response": str(response),
             "request_digest": replay_tape.normalized_digest(recorded),
             "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest()}]
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *_args, **_kwargs: b"unused")
    _child.install_wire_recorder(run, "http://127.0.0.1:1/v1/chat/completions",
                                 replay_tape.Replayer(tape))
    with pytest.raises(live_tool_tape.ToolReplayRefused):
        kernel_bounded_http.request_bytes(recorded.replace(b"a" * 32, b"b" * 32),
                                          "http://127.0.0.1:1/v1/chat/completions", "")


def test_full_replay_pending_mismatch_refuses_before_live_send(tmp_path, monkeypatch):
    import kernel_bounded_http

    recorded = b'{"model": "scripted", "task": "original"}'
    request = tmp_path / "old-request.json"
    request.write_bytes(recorded)
    pending = {"send": 1, "request": str(request),
               "request_digest": replay_tape.normalized_digest(recorded)}
    player = replay_tape.Replayer([], pending=pending)
    live_calls = []
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda *_args, **_kwargs: live_calls.append(True))
    _child.install_wire_recorder(tmp_path, "http://127.0.0.1:1/v1/chat/completions", player)
    with pytest.raises(replay_tape.ReplayUnfaithful, match="request in flight"):
        kernel_bounded_http.request_bytes(
            recorded.replace(b"original", b"different"),
            "http://127.0.0.1:1/v1/chat/completions", "")
    assert not live_calls
    assert json.loads((tmp_path / "resume-divergence.json").read_text())["checkpoint"] is False


def test_tokenizer_fallback_does_not_advance_journal_position():
    class Journal:
        def __init__(self):
            self._sequence = 4
            self.events = []

        def emit(self, kind, **fields):
            self.events.append((kind, fields))
            self._sequence += 1

        def subscribe(self, _name, _observer, *, kinds):
            assert "request_diet_applied" in kinds

    journal = Journal()
    meter = kernel_request_diet.RequestTokenMeter(tokenizer=lambda _text: (_ for _ in ()).throw(OSError()))
    meter.attach_journal(journal)
    meter.begin_request("request")
    _, method = meter.measure({"model": "scripted", "messages": [{"role": "user", "content": "hello"}]})
    assert method == "calibrated_bytes"
    assert journal._sequence == 4
    assert journal.events == []
    assert meter.journal_fields(method)["tokenize_availability"] == "unavailable"


def test_resume_child_environment_preserves_checkpoint_opt_out(monkeypatch):
    previous = os.environ.copy()
    try:
        monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
        _child.install_environment({"environment": {"STATEFRAME_CHECKPOINT": "save"}})
        assert live_checkpoint.mode() == "off"
    finally:
        os.environ.clear()
        os.environ.update(previous)


def test_resume_checkpoint_opt_out_controls_selection(monkeypatch):
    spec = {"environment": {"STATEFRAME_CHECKPOINT": "save"}}
    monkeypatch.delenv("STATEFRAME_CHECKPOINT", raising=False)
    assert not runner.checkpoint_resume_off(spec)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    assert runner.checkpoint_resume_off(spec)
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "")
    spec["environment"]["STATEFRAME_CHECKPOINT"] = "off"
    assert runner.checkpoint_resume_off(spec)


def test_archived_attempts_keep_diagnostics_without_checkpoint_payloads(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    spec = {"environment": {}, "max_turns": 10}
    state = {"run": run, "spec": spec, "max_turns": 10, "attempt": 1}
    for number in (1, 2, 3):
        (run / "spec.json").write_text(json.dumps(spec))
        root = run / "checkpoints"
        snapshot = root / f"000{number}"
        (snapshot / "run").mkdir(parents=True)
        (snapshot / "checkpoint.json").write_text(json.dumps({"step": number}))
        (snapshot / "state.pkl").write_bytes(b"payload")
        (snapshot / "run" / "data").write_bytes(b"payload")
        (root / "blob-pool").mkdir()
        (root / "blob-pool" / "data").write_bytes(b"payload")
        (root / "journal-cache").mkdir()
        (root / "journal-cache" / "data").write_bytes(b"payload")
        (root / "events.jsonl").write_text("saved\n")
        (root / "LATEST").write_text(snapshot.name)
        resume.archive_attempt(state, tmp_path, use_checkpoint=False)
        archived = run / "attempts" / str(number) / "checkpoints"
        assert (archived / snapshot.name / "checkpoint.json").is_file()
        assert (archived / "events.jsonl").read_text() == "saved\n"
        assert not any(archived.rglob("state.pkl"))
        assert not any(archived.rglob("data"))
        state["attempt"] = number + 1


@pytest.mark.parametrize("damage", ("truncated", "changed"))
def test_corrupt_state_preflight_records_restore_refusal(tmp_path, damage):
    root = tmp_path / "run" / "checkpoints"
    snapshot = root / "0002"
    snapshot.mkdir(parents=True)
    original = b"complete checkpoint state"
    (snapshot / "checkpoint.json").write_text(json.dumps({
        "schema": resume.CHECKPOINT_SCHEMA, "sends": 1, "state_bytes": len(original),
        "state_sha256": hashlib.sha256(original).hexdigest(), "seq": 1}))
    (snapshot / "state.pkl").write_bytes(original[:-1] if damage == "truncated"
                                          else original[:-1] + b"x")
    assert resume.select_checkpoint(root.parent, [], root.parent / "wire") is None
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "checkpoint_restore_refused"
    assert events[0]["source"] == str(snapshot)
    assert events[0]["detail"] == "checkpoint state checksum mismatch"


def test_checkpoint_writer_flushes_snapshot_and_new_pool_files(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    nested = snapshot / "run" / "wire"
    nested.mkdir(parents=True)
    files = [snapshot / "state.pkl", snapshot / "checkpoint.json", snapshot / "blobs.json",
             nested / "0001.request.json"]
    for path in files:
        path.write_bytes(b"data")
    pool = tmp_path / "blob-pool" / "wire"
    pool.mkdir(parents=True)
    pooled = pool / "0001.request.json"
    pooled.write_bytes(b"data")
    durable = tmp_path / "prior"
    durable.write_bytes(b"already flushed")
    linked = nested / "prior.json"
    os.link(durable, linked)
    synced = []
    monkeypatch.setattr(live_checkpoint, "_fsync_file", synced.append)
    monkeypatch.setattr(live_checkpoint, "_fsync_dir", lambda _path: None)
    prior_stat = durable.stat()
    live_checkpoint._fsync_snapshot(snapshot, [pooled], {(prior_stat.st_dev, prior_stat.st_ino)})
    assert set(synced) == {*files, pooled}
    assert len(synced) == len(files) + 1


def test_checkpoint_writer_flushes_pool_link_left_by_failed_writer(tmp_path):
    root = tmp_path / "checkpoints"
    previous = root / "saved"
    previous.mkdir(parents=True)
    (previous / "blobs.json").write_text(json.dumps({"wire": [["old", 3]]}))
    files = live_checkpoint._unpublished_pool_files(
        previous, root, {"wire": [["old", 3], ["new", 3]]})
    assert files == [root / "blob-pool" / "wire" / "new"]


def test_restore_probe_defaults_to_product_cadence_and_honors_override(monkeypatch):
    monkeypatch.delenv("STATEFRAME_CHECKPOINT_EVERY", raising=False)
    assert "STATEFRAME_CHECKPOINT_EVERY" not in checkpoint_restore_probe.child_env(True)
    assert checkpoint_restore_probe.child_env(True, 7)["STATEFRAME_CHECKPOINT_EVERY"] == "7"
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_EVERY", "4")
    assert checkpoint_restore_probe.child_env(True)["STATEFRAME_CHECKPOINT_EVERY"] == "4"


def test_signing_key_is_loaded_by_default_but_not_when_off(monkeypatch):
    key = b"checkpoint-test-key-0123456789abcdef"
    registered = []
    monkeypatch.setattr(_child, "checkpoint_key", lambda: key)
    monkeypatch.setattr(live_checkpoint, "register_signing_key", registered.append)
    monkeypatch.setattr(live_checkpoint, "register_participant", lambda *_args: None)
    monkeypatch.delenv("STATEFRAME_CHECKPOINT", raising=False)

    _child.register_host_state(required=True)
    assert registered == [key]

    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    _child.register_host_state(required=True)
    assert registered == [key]


def test_module_snapshot_keeps_new_typed_state_but_not_attempt_replay_control():
    module = types.ModuleType("kernel_tool_tape")
    module.__file__ = kernel_tool_tape.__file__
    module._active_tape = object()
    module._replay_active = False
    module.future_typed_state = {"bookmark": {"path": "src/app.py", "line": 12}}

    saved = kernel_checkpoint._module_state(module.__name__, module)

    assert saved == {"future_typed_state": module.future_typed_state}


def test_restore_keeps_complete_tool_tape_for_later_resume(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "saved"
    (checkpoint / "run").mkdir(parents=True)
    tape = run / "tool-tape.jsonl"
    tape.write_text('checkpoint intent\nrecorded tail\n')
    (run / "sends.jsonl").write_text("new attempt\n")
    (checkpoint / "run" / "sends.jsonl").write_text("checkpoint prefix\n")

    kernel_checkpoint._restore_files(checkpoint, run, {"entries": ["sends.jsonl"]})

    assert tape.read_text() == 'checkpoint intent\nrecorded tail\n'
    assert (run / "sends.jsonl").read_text() == "checkpoint prefix\n"
    assert "tool-tape.jsonl" in kernel_checkpoint.SKIP_RUN_ENTRIES


@dataclass(frozen=True)
class _CheckResult:
    completed: subprocess.CompletedProcess[str]


def test_restored_check_count_uses_new_attempt_timing_substitution(monkeypatch, tmp_path):
    policy = types.ModuleType("kernel_execution_policy")
    policy.run_sandboxed = lambda *_args, **_kwargs: _CheckResult(
        subprocess.CompletedProcess(["check"], 0, "1 passed in 0.04s", ""))
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    monkeypatch.setattr(_child, "emit", lambda *_args, **_kwargs: None)
    old_state = _child.HOST_STATE.copy()
    try:
        previous = tmp_path / "previous"
        previous.mkdir()
        record = {"check": 8, "argv": ["check"], "returncode": 0,
                  "stdout": "1 passed in 0.03s", "stderr": ""}
        (previous / "0008.json.z").write_bytes(zlib.compress(json.dumps(record).encode()))
        current = tmp_path / "current"
        current.mkdir()
        _child.install_check_recorder(current, types.SimpleNamespace(active=True), previous)
        _child.HOST_STATE["checks"].update(n=7, on=True)  # the checkpoint's host participant

        result = policy.run_sandboxed(["check"], tmp_path)

        assert result.completed.stdout == "1 passed in 0.03s"
        assert _child.HOST_STATE["checks"] == {"n": 8, "on": True}
    finally:
        _child.HOST_STATE.clear()
        _child.HOST_STATE.update(old_state)


def test_restored_meter_does_not_shift_edit_and_check_ids(monkeypatch, tmp_path):
    run = tmp_path / "run"
    parent = run / "worker" / "measurement-runs"
    parent.mkdir(parents=True)
    repo = run / "worker" / "repo"
    repo.mkdir()
    journal = kernel_continuation_journal.ContinuationJournal(parent, "task", {})
    meter = kernel_request_diet.RequestTokenMeter()
    meter.attach_journal(journal)
    frame = types.SimpleNamespace(f_code=test_restored_meter_does_not_shift_edit_and_check_ids.__code__,
                                  f_locals={"journal": journal, "meter": meter, "step": 1})
    old_state = live_checkpoint._STATE.copy()
    old_key = live_checkpoint._SIGNING_KEY.copy()
    old_secrets = live_checkpoint._SECRETS.copy()
    old_scans = live_checkpoint._SCANS.copy()
    try:
        monkeypatch.setattr(live_checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(live_checkpoint, "_context_items", lambda: [])
        monkeypatch.setattr(live_checkpoint.contextvars, "copy_context", contextvars.Context)
        monkeypatch.setattr(live_checkpoint, "install_continuous_clock", lambda _saved: None)
        live_checkpoint._STATE["setup"] = {"code": frame.f_code, "line": 0, "funcs": frozenset()}
        live_checkpoint.register_signing_key(b"checkpoint-test-key-0123456789abcdef")
        checkpoint = live_checkpoint.save(frame, 1, run)

        def edit_and_check(active_journal, active_meter):
            active_meter.attach_journal(active_journal)
            (repo / "step_016.txt").write_text("completed 16\n")
            owner = types.SimpleNamespace(
                audit_journal=active_journal, current_step=15, workspace_revision=15,
                current_request_id="request", run_id=active_journal.run_id, _receipts=[])
            kernel_bolt.KernelBolt.receipt(
                owner, "edit", "create step_016.txt", "completed 16\n", "APPLIED",
                target_path="step_016.txt")
            edit_id = owner.last_receipt_id
            projected = kernel_vacuity_truth.journal_applied_edit_rows(active_journal.records())
            assert projected[-1]["edit_receipt"] == edit_id
            check_id = kernel_wire_identity.next_wire_id(
                active_journal, "check-execution", 15, "tests/test_app.py", 1)
            effect_id = kernel_wire_identity.next_wire_id(
                active_journal, "effect-application", "file_create", "APPLIED", "body-hash")
            figure_id = kernel_wire_identity.next_wire_id(
                active_journal, "figure-it-out", "READ", "input-hash", "result-hash")
            request_id = kernel_wire_identity.next_wire_id(active_journal, "request", 16)
            return edit_id, check_id, effect_id, figure_id, request_id, active_journal._sequence

        expected = edit_and_check(journal, meter)
        journal.close()

        live_checkpoint.restore(frame, checkpoint, run)
        assert not (repo / "step_016.txt").exists()
        restored_journal = frame.f_locals["journal"]
        restored_meter = frame.f_locals["meter"]
        actual = edit_and_check(restored_journal, restored_meter)

        assert actual == expected
        assert not any(row["kind"] == "request_token_meter_observer_failed"
                       for row in restored_journal.records())
    finally:
        active = frame.f_locals.get("journal")
        if active is not None:
            active.close()
        live_checkpoint._STATE.clear()
        live_checkpoint._STATE.update(old_state)
        live_checkpoint._SIGNING_KEY[:] = old_key
        live_checkpoint._SECRETS[:] = old_secrets
        live_checkpoint._SCANS.clear()
        live_checkpoint._SCANS.update(old_scans)


def _restore_relocated_workspace(source_run, destination_run, relocation, replacement):
    task_digest = "sha256:" + "a" * 64
    workspace = ContinuationWorkspace.create(
        source_run / "worker" / "continuation-workspaces", task_digest=task_digest)
    workspace_name = workspace.directory.name
    live_checkpoint._STATE["setup"] = {
        "code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    checkpoint = live_checkpoint.save(sys._getframe(), 1, source_run)
    workspace.close()
    if relocation == "move":
        source_run.rename(destination_run)
    else:
        shutil.copytree(source_run, destination_run)
    checkpoint = destination_run / "checkpoints" / checkpoint.name
    workspace = ContinuationWorkspace.open_existing(
        destination_run / "worker" / "continuation-workspaces" / workspace_name,
        task_digest=task_digest)
    setup_workspace = workspace
    try:
        live_checkpoint.restore(sys._getframe(), checkpoint, destination_run)
        assert workspace is not setup_workspace
        assert workspace.directory == destination_run / "worker" / "continuation-workspaces" / workspace_name
        with pytest.raises(WorkspaceError, match="workspace ownership changed"):
            setup_workspace.assert_owner()
        workspace.assert_owner()
        if replacement == "directory":
            workspace.directory.rename(workspace.directory.with_name(workspace.directory.name + ".previous"))
            workspace.directory.mkdir()
        else:
            lock = workspace.directory / "owner.lock"
            lock.rename(workspace.directory / "owner.previous")
        (workspace.directory / "owner.lock").write_bytes(b"")
        with pytest.raises(WorkspaceError, match="workspace ownership changed"):
            workspace.assert_owner()
    finally:
        workspace.close()
        setup_workspace.close()


@pytest.mark.parametrize("relocation", ["move", "copy"])
@pytest.mark.parametrize("replacement", ["directory", "lock"])
def test_checkpoint_workspace_lease_rebinds_after_run_relocation(
        monkeypatch, tmp_path, relocation, replacement):
    old_state = live_checkpoint._STATE.copy()
    old_key = live_checkpoint._SIGNING_KEY.copy()
    old_scans = live_checkpoint._SCANS.copy()
    try:
        monkeypatch.setattr(live_checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(live_checkpoint, "_context_items", lambda: [])
        monkeypatch.setattr(live_checkpoint.contextvars, "copy_context", contextvars.Context)
        monkeypatch.setattr(live_checkpoint, "install_continuous_clock", lambda _saved: None)
        live_checkpoint.register_signing_key(b"checkpoint-test-key-0123456789abcdef")
        source_run = tmp_path / "source"
        source_run.mkdir()
        _restore_relocated_workspace(source_run, tmp_path / "destination", relocation, replacement)
    finally:
        live_checkpoint._STATE.clear()
        live_checkpoint._STATE.update(old_state)
        live_checkpoint._SIGNING_KEY[:] = old_key
        live_checkpoint._SCANS.clear()
        live_checkpoint._SCANS.update(old_scans)


def _restore_live_local(run):
    value = "saved"
    live_checkpoint._STATE["setup"] = {
        "code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    checkpoint = live_checkpoint.save(sys._getframe(), 1, run)
    value = "changed"
    live_checkpoint.restore(sys._getframe(), checkpoint, run)
    return value


def test_checkpoint_restores_live_frame_locals(monkeypatch, tmp_path):
    old_state = live_checkpoint._STATE.copy()
    old_key = live_checkpoint._SIGNING_KEY.copy()
    old_scans = live_checkpoint._SCANS.copy()
    try:
        monkeypatch.setattr(live_checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(live_checkpoint, "_context_items", lambda: [])
        monkeypatch.setattr(live_checkpoint.contextvars, "copy_context", contextvars.Context)
        monkeypatch.setattr(live_checkpoint, "install_continuous_clock", lambda _saved: None)
        live_checkpoint.register_signing_key(b"checkpoint-test-key-0123456789abcdef")
        assert _restore_live_local(tmp_path / "run") == "saved"
    finally:
        live_checkpoint._STATE.clear()
        live_checkpoint._STATE.update(old_state)
        live_checkpoint._SIGNING_KEY[:] = old_key
        live_checkpoint._SCANS.clear()
        live_checkpoint._SCANS.update(old_scans)


def test_probe_waits_for_new_wire_inode_without_losing_pending_request(tmp_path):
    case = tmp_path / "case"
    wire = case / "run" / "wire"
    wire.mkdir(parents=True)
    original = b'{"model":"scripted"}'
    request = wire / "0003.request.json"
    request.write_bytes(original)
    (wire / "0003.error.txt").write_text("interrupted\n")

    saved = checkpoint_restore_probe.set_aside_request(case, case / "run", 3, original)

    assert saved.read_bytes() == original
    assert request.samefile(saved)
    assert (case / "original-wire" / "0003.error.txt").read_text() == "interrupted\n"
    archived = case / "attempt" / "wire"
    archived.parent.mkdir()
    wire.replace(archived)
    wire.mkdir()
    request.write_bytes(original)
    assert not request.samefile(saved)


def test_probe_records_real_event_fields_and_requested_script_length(tmp_path):
    engine = tmp_path / "engine.log"
    engine.write_text('[sf] {"kind":"checkpoint_verified"}\nplain output\n'
                      '[sf] {"kind":"checkpoint_unfaithful"}\n')
    checkpoint = tmp_path / "events.jsonl"
    checkpoint.write_text('{"event":"checkpoint_restored"}\n'
                          '{"event":"checkpoint_restore_refused"}\n')

    assert checkpoint_restore_probe.event_rows(engine, "kind") == [
        "checkpoint_verified", "checkpoint_unfaithful"]
    assert checkpoint_restore_probe.event_rows(checkpoint, "event") == [
        "checkpoint_restored", "checkpoint_restore_refused"]
    steps = checkpoint_restore_probe.script_steps(120)
    assert len(steps) == 120
    assert steps[109]["tool"] == "file_create"
    assert steps[-1]["tool"] == "done"
