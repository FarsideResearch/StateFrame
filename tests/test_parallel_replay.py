"""A completed parent subframe call replays its recorded local commits only."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import stateframe.engine.kernel_parallel as parallel
from stateframe.engine.kernel_parallel import ParallelContext, RecoveryRequired, execute_parallel
from stateframe.engine.kernel_tool_tape import ToolReplayRefused
from stateframe.engine.kernel_parallel_txn import TxnStore
from stateframe.engine.kernel_parallel_records import RequirementClause
from stateframe.engine.kernel_tool_tape import configure_replay
from stateframe.engine.kernel_stateframe_framefit5 import _opaque_actions

from test_parallel_dispatch import Bolt, Runtime, ScriptedHost, restore_snapshot_permissions, setup_context, unit  # noqa: F401


class NoHost:
    def submit(self, *_args, **_kwargs):
        raise AssertionError("completed replay launched or queried a subframe")

    def poll(self, *_args, **_kwargs):
        raise AssertionError("completed replay polled the host")

    def cancel(self, *_args, **_kwargs):
        raise AssertionError("completed replay cancelled a subframe")


def _receipt_bytes(result, run, monkeypatch):
    import kernel_parallel as flat_parallel

    monkeypatch.setattr(flat_parallel, "_service", (run, {"parallel_mode": "2"}))
    turns = (({"role": "assistant", "content": "", "tool_calls": [{
        "id": result.call_id, "type": "function", "function": {
            "name": "parallel", "arguments": json.dumps({"subframes": []}),
        },
    }]}, {"role": "tool", "tool_call_id": result.call_id,
          "content": result.observation}),)
    rows, ids = _opaque_actions(turns)
    assert ids == {result.call_id}
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


def test_completed_replay_preserves_observation_bytes_and_txn_order(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    initial = tmp_path / "initial"
    shutil.copytree(root, initial)
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    first = execute_parallel(context, args, host=host,
                             store_root=run / "parallel" / "parallel_1")
    receipt_before = _receipt_bytes(first, run, monkeypatch)
    replay_root = tmp_path / "replay-workspace"
    shutil.copytree(initial, replay_root)
    configure_replay(run, active=True)
    replay = ParallelContext(Runtime(replay_root, "parallel_1"), Bolt(replay_root), replay_root,
                             set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    second = execute_parallel(replay, args, host=NoHost(),
                              store_root=run / "parallel" / "parallel_1")
    assert second.observation.encode() == first.observation.encode()
    assert _receipt_bytes(second, run, monkeypatch) == receipt_before
    assert second.txn_order == (1, 2)
    assert replay_root.joinpath("a.py").read_bytes() == root.joinpath("a.py").read_bytes()
    assert replay_root.joinpath("b.py").read_bytes() == root.joinpath("b.py").read_bytes()
    assert [kwargs["changed_paths"] for args, kwargs in replay.bolt.receipts
            if args[0] == "shell_effect"] == [("a.py",), ("b.py",)]


def test_completed_partial_failure_replays_identical_f4_receipt(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    initial = tmp_path / "initial"
    shutil.copytree(root, initial)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    first = execute_parallel(context, args,
                             host=ScriptedHost({"b": {"b.py": "OTHER = 2\n"}},
                                               terminal={"a": "BLOCKED"}),
                             store_root=run / "parallel" / "parallel_1")
    assert [(item.unit_id, item.status) for item in first.outcomes] == [
        ("a", "BLOCKED"), ("b", "COMMITTED_UNVERIFIED")]
    receipt_before = _receipt_bytes(first, run, monkeypatch)
    replay_root = tmp_path / "replay-workspace"
    shutil.copytree(initial, replay_root)
    configure_replay(run, active=True)
    replay = ParallelContext(Runtime(replay_root, "parallel_1"), Bolt(replay_root), replay_root,
                             set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    second = execute_parallel(replay, args, host=NoHost(),
                              store_root=run / "parallel" / "parallel_1")
    assert second.observation.encode() == first.observation.encode()
    assert _receipt_bytes(second, run, monkeypatch) == receipt_before


def test_decided_transaction_recovers_before_launching_remaining_unit(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    original_publish = parallel._publish_txn
    failed = False

    def crash_after_decision(store, record, parent_context):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("planted crash after DECIDED")
        return original_publish(store, record, parent_context)

    monkeypatch.setattr(parallel, "_publish_txn", crash_after_decision)
    with pytest.raises(RecoveryRequired, match="RECOVERY_REQUIRED"):
        execute_parallel(context, args, host=ScriptedHost({"a": {"a.py": "VALUE = 2\n"},
                                                          "b": {"b.py": "OTHER = 2\n"}}),
                         store_root=run / "parallel" / "parallel_1")
    store = TxnStore(run / "parallel" / "parallel_1" / "txn")
    assert [(record.unit_id, state) for record, state in store.records()] == [("a", "DECIDED")]
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"

    monkeypatch.setattr(parallel, "_publish_txn", original_publish)
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(root, "parallel_1"), Bolt(root), root, set(),
                              context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    host = ScriptedHost({"b": {"b.py": "OTHER = 2\n"}})
    result = execute_parallel(resumed, args, host=host,
                              store_root=run / "parallel" / "parallel_1")
    assert [scope for op, scope, _ in host.calls if op == "launch"] == ["b"]
    assert result.txn_order == (1, 2)
    assert root.joinpath("a.py").read_text() == "VALUE = 2\n"
    assert root.joinpath("b.py").read_text() == "OTHER = 2\n"
    assert [(record.txn_id, state) for record, state in store.records() if state == "PUBLISHED"] == [
        (1, "PUBLISHED"), (2, "PUBLISHED")]


def test_sealed_proposal_recovers_without_relaunching_worker(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})
    original_decide = TxnStore.decide

    class PlantedCrash(BaseException):
        pass

    def crash_before_decision(self, record, blobs):
        raise PlantedCrash

    monkeypatch.setattr(TxnStore, "decide", crash_before_decision)
    with pytest.raises(PlantedCrash):
        execute_parallel(context, args, host=host,
                         store_root=run / "parallel" / "parallel_1")
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    assert (run / "parallel" / "parallel_1" / "proposals" / "a" / "proposal.json").is_file()
    for unit_id in ("a", "b"):
        record = run / "parallel" / "parallel_1" / "host" / "units" / unit_id / "1.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"incarnation": 1, "unit_id": unit_id}))
    monkeypatch.setattr(TxnStore, "decide", original_decide)
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(root, "parallel_1"), Bolt(root), root, set(),
                              context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    launches_before = len([1 for op, _, _ in host.calls if op == "launch"])
    result = execute_parallel(resumed, args, host=host,
                              store_root=run / "parallel" / "parallel_1")
    launches_after = [scope for op, scope, _ in host.calls if op == "launch"]
    assert len(launches_after) == launches_before + 1
    assert launches_after[-1] == "b"
    assert result.txn_order == (1, 2)


def test_blocked_terminal_outcome_is_durable_and_not_relaunched(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    initial = tmp_path / "initial"
    shutil.copytree(root, initial)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    host = ScriptedHost({"b": {"b.py": "OTHER = 2\n"}}, terminal={"a": "BLOCKED"})
    original_final = parallel._final_check

    class PlantedCrash(BaseException):
        pass

    def crash_before_result(*_args, **_kwargs):
        raise PlantedCrash

    monkeypatch.setattr(parallel, "_final_check", crash_before_result)
    with pytest.raises(PlantedCrash):
        execute_parallel(context, args, host=host, store_root=run / "parallel" / "parallel_1")
    logged = [json.loads(line) for line in (run / "parallel" / "parallel_1" / "outcomes.jsonl").read_text().splitlines()]
    assert [(row["unit_id"], row["status"]) for row in logged] == [
        ("a", "BLOCKED"), ("b", "COMMITTED_UNVERIFIED")]

    monkeypatch.setattr(parallel, "_final_check", original_final)
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    replay_root = tmp_path / "replay"
    shutil.copytree(initial, replay_root)
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(replay_root, "parallel_1"), Bolt(replay_root), replay_root,
                              set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    result = execute_parallel(resumed, args, host=NoHost(),
                              store_root=run / "parallel" / "parallel_1")
    assert [(item.unit_id, item.status) for item in result.outcomes] == [
        ("a", "BLOCKED"), ("b", "COMMITTED_UNVERIFIED")]
    assert replay_root.joinpath("a.py").read_text() == "VALUE = 1\n"
    assert replay_root.joinpath("b.py").read_text() == "OTHER = 2\n"


def test_pending_recovery_rejects_changed_pre_manifest_without_txn(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    original_start = parallel._record_start

    class PlantedCrash(BaseException):
        pass

    def crash_before_launch(*_args):
        raise PlantedCrash

    monkeypatch.setattr(parallel, "_record_start", crash_before_launch)
    with pytest.raises(PlantedCrash):
        execute_parallel(context, args, host=NoHost(), store_root=run / "parallel" / "parallel_1")
    monkeypatch.setattr(parallel, "_record_start", original_start)
    assert TxnStore(run / "parallel" / "parallel_1" / "txn").records() == []
    root.joinpath("a.py").write_text("TAMPERED = 1\n")
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(root, "parallel_1"), Bolt(root), root,
                              set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    with pytest.raises(ToolReplayRefused, match="pre-manifest"):
        execute_parallel(resumed, args, host=NoHost(),
                         store_root=run / "parallel" / "parallel_1")


def test_pending_recovery_rejects_changed_parent_requirements(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    context.requirements = (RequirementClause("R1", "Keep a.py compatible", "intersect", ("a.py",)),)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    original_start = parallel._record_start

    class PlantedCrash(BaseException):
        pass

    def crash_before_launch(*_args):
        raise PlantedCrash

    monkeypatch.setattr(parallel, "_record_start", crash_before_launch)
    with pytest.raises(PlantedCrash):
        execute_parallel(context, args, host=NoHost(), store_root=run / "parallel" / "parallel_1")
    monkeypatch.setattr(parallel, "_record_start", original_start)
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(root, "parallel_1"), Bolt(root), root,
                              set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    with pytest.raises(ToolReplayRefused, match="requirements ledger changed"):
        execute_parallel(resumed, args, host=NoHost(),
                         store_root=run / "parallel" / "parallel_1")


@pytest.mark.parametrize("tamper,expected", [
    ("spec_digest", "spec digest"),
    ("execution_id", "execution identity"),
    ("projection", "projection differs"),
])
def test_recovered_spec_must_match_frozen_execution_and_projection(
        tmp_path, monkeypatch, tamper, expected):
    run, root, context = setup_context(tmp_path, monkeypatch)
    args = {"subframes": [unit("a", "a.py"), unit("b", "b.py")]}
    original_answer = parallel._host_answer

    class PlantedCrash(BaseException):
        pass

    def crash_on_launch(host, op, scope, payload, deadline):
        if op == "launch":
            raise PlantedCrash
        return original_answer(host, op, scope, payload, deadline)

    monkeypatch.setattr(parallel, "_host_answer", crash_on_launch)
    with pytest.raises(PlantedCrash):
        execute_parallel(context, args, host=NoHost(), store_root=run / "parallel" / "parallel_1")
    monkeypatch.setattr(parallel, "_host_answer", original_answer)
    call_root = run / "parallel" / "parallel_1"
    spec_path = call_root / "units" / "a" / "unit-spec.json"
    spec_row = json.loads(spec_path.read_text())
    intent_path = call_root / "intent.json"
    intent = json.loads(intent_path.read_text())
    if tamper == "execution_id":
        intent["executions"][0]["execution_id"] = "0" * 32
    else:
        spec_row["failure_projection"] = "tampered"
        spec_path.write_text(json.dumps(spec_row))
        if tamper == "projection":
            from stateframe.engine.kernel_parallel_records import UnitSpec
            intent["executions"][0]["spec_digest"] = UnitSpec.from_json(spec_row).digest()
    intent_path.write_text(json.dumps(intent))
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(root, "parallel_1"), Bolt(root), root,
                              set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    with pytest.raises(ToolReplayRefused, match=expected):
        execute_parallel(resumed, args, host=NoHost(), store_root=call_root)


@pytest.mark.parametrize("crash_point,disposition,changed", [
    ("after_a", "DECIDED", ("a.py",)),
    ("after_c", "DECIDED", ("a.py", "c.py")),
    ("after_materialized", "MATERIALIZED", ("a.py", "c.py")),
    ("after_published", "PUBLISHED", ("a.py", "c.py")),
])
def test_recovery_after_each_publication_write_is_single_commit(
        tmp_path, monkeypatch, crash_point, disposition, changed):
    run, root, context = setup_context(
        tmp_path, monkeypatch,
        files={"a.py": "VALUE = 1\n", "b.py": "OTHER = 1\n", "c.py": "EXTRA = 1\n"})
    args_a = unit("a", "a.py")
    args_a["owns"].append("c.py")
    args = {"subframes": [args_a, unit("b", "b.py")]}
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n", "c.py": "EXTRA = 2\n"},
                         "b": {"b.py": "OTHER = 2\n"}})
    original_replace = os.replace
    original_materialize = TxnStore.materialize
    original_published = TxnStore.mark_published
    hit = False

    def replace_then_crash(source, target):
        nonlocal hit
        original_replace(source, target)
        if not hit and crash_point in {"after_a", "after_c"} and Path(target) == root / f"{crash_point[-1]}.py":
            hit = True
            raise OSError("planted crash after file replacement")

    def materialize_then_crash(self, txn_id, workspace):
        nonlocal hit
        original_materialize(self, txn_id, workspace)
        if not hit and crash_point == "after_materialized":
            hit = True
            raise OSError("planted crash after MATERIALIZED")

    def publish_then_crash(self, txn_id, publication_id):
        nonlocal hit
        original_published(self, txn_id, publication_id)
        if not hit and crash_point == "after_published":
            hit = True
            raise OSError("planted crash after PUBLISHED")

    monkeypatch.setattr(os, "replace", replace_then_crash)
    monkeypatch.setattr(TxnStore, "materialize", materialize_then_crash)
    monkeypatch.setattr(TxnStore, "mark_published", publish_then_crash)
    with pytest.raises(RecoveryRequired, match="RECOVERY_REQUIRED"):
        execute_parallel(context, args, host=host, store_root=run / "parallel" / "parallel_1")
    assert hit
    store = TxnStore(run / "parallel" / "parallel_1" / "txn")
    assert {record.unit_id for record, _ in store.records()} == {"a"}
    assert store.records()[-1][1] == disposition
    assert tuple(path for path in ("a.py", "c.py") if root.joinpath(path).read_text().endswith("2\n")) == changed
    monkeypatch.setattr(os, "replace", original_replace)
    monkeypatch.setattr(TxnStore, "materialize", original_materialize)
    monkeypatch.setattr(TxnStore, "mark_published", original_published)
    spec = parallel._service[1].copy()
    spec["parallel_recovery"] = {"call_id": "parallel_1"}
    monkeypatch.setattr(parallel, "_service", (run, spec))
    configure_replay(run, active=True)
    resumed = ParallelContext(Runtime(root, "parallel_1"), Bolt(root), root,
                              set(), context.parent_seed, 2, (), context.limits, "2", lambda _path: False)
    recovery_host = ScriptedHost({"b": {"b.py": "OTHER = 2\n"}})
    result = execute_parallel(resumed, args, host=recovery_host,
                              store_root=run / "parallel" / "parallel_1")
    assert result.txn_order == (1, 2)
    assert [(record.txn_id, record.unit_id, state) for record, state in store.records()
            if state == "PUBLISHED"] == [(1, "a", "PUBLISHED"), (2, "b", "PUBLISHED")]
    assert [scope for op, scope, _ in recovery_host.calls if op == "launch"] == ["b"]
    assert root.joinpath("a.py").read_text() == "VALUE = 2\n"
    assert root.joinpath("b.py").read_text() == "OTHER = 2\n"
    assert root.joinpath("c.py").read_text() == "EXTRA = 2\n"
