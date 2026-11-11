"""Fixed mutation tool surface and fail-closed host dispatch."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest


def test_off_switch_does_not_import_mutation_kernel():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "STATEFRAME_MUTATIONS": "off",
                   "PYTHONPATH": os.pathsep.join((str(root / "src"),
                                                   str(root / "src/stateframe/engine")))}
    script = ("import sys\nfrom probes import react_repair\n"
              "assert all(t['function']['name'] != 'mutate' for t in react_repair.TOOLS)\n"
              "assert not any(n.startswith('kernel_mutate') for n in sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_l0_imports_no_mutation_host_or_engine_module():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "STATEFRAME_MUTATIONS": "off", "PYTHONPATH": str(root / "src")}
    script = ("import sys\nfrom stateframe import runner\n"
              "assert runner.mutation_spec_options(None, {}) is None\n"
              "assert 'stateframe.mutate_host' not in sys.modules\n"
              "assert not any(n.startswith('kernel_mutate') for n in sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_fourth_revision_of_one_unit_is_refused_before_validation(tmp_path, monkeypatch):
    import kernel_mutate_build
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    journal = Journal(tmp_path)
    for index in range(3):
        journal.append("BUDGET_RESERVE", name="revisions_per_unit", amount=1, scope="reader",
                       id=f"revision-{index}")
    runtime = MutationRuntime(level="authored", journal=journal,
                              policy=MutationPolicy(level="authored"))
    runtime.episode = {"id": "episode", "sends_left": 1}
    runtime.ready["reader"] = {"manifest": {"revision": 3}, "source_digest": "sha256:previous"}
    monkeypatch.setattr(kernel_mutate_build, "stage_unit",
                        lambda *_args: pytest.fail("over-limit revision was staged"))
    reply = runtime._submit({"manifest": {"name": "reader", "revision": 4,
                                           "parent": "sha256:previous", "requires": {}},
                             "files": {"main.py": "def handle(input, ctx): return {}\n"}}, "call-4")
    assert reply.startswith("REFUSED MUTATION_BUDGET")


def test_broker_launch_reserves_and_settles_its_own_time_budgets(tmp_path):
    from kernel_mutate_policy import Budget
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    journal = Journal(tmp_path)
    outcome = SimpleNamespace(wall_s=2.0, cpu_s=None)
    runtime = MutationRuntime(level="authored", journal=journal,
                              executor_factory=lambda: SimpleNamespace(
                                  root=tmp_path, run=lambda _spec: outcome))
    spec = SimpleNamespace(role="broker", launch="broker-1",
                           limits=SimpleNamespace(wall_s=12.0, cpu_s=5.0))
    assert runtime.run(spec) is outcome
    remaining = Budget().fold(journal.events()).remaining()
    assert remaining["broker_wall_s"] == Budget().limits["broker_wall_s"] - 2.0
    assert remaining["broker_cpu_s"] == Budget().limits["broker_cpu_s"] - 5.0
    assert remaining["payload_wall_s"] == Budget().limits["payload_wall_s"]
    assert remaining["payload_cpu_s"] == Budget().limits["payload_cpu_s"]


def test_fixed_schema_when_enabled(monkeypatch):
    from kernel_mutate_runtime import schema_for_level

    schema = schema_for_level("authored")
    assert schema["function"]["name"] == "mutate"
    params = schema["function"]["parameters"]
    assert params["required"] == ["op"]
    assert params["properties"]["op"]["enum"] == [
        "catalog", "use", "enter", "submit", "leave", "watch", "unwatch", "inspect", "retain"]
    assert params["properties"]["args"]["maxLength"] == 16000


def test_mutation_engine_modules_are_release_declared_with_current_hashes():
    import hashlib

    root = Path(__file__).resolve().parents[1] / "src/stateframe/engine"
    manifest = json.loads((root / "release-manifest.json").read_text())["files"]
    for path in (*root.glob("kernel_mutate_*.py"), root / "probes/react_repair.py"):
        assert manifest.get(path.relative_to(root).as_posix()) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_retained_hook_stages_its_complete_admitted_bundle(tmp_path, monkeypatch):
    import kernel_mutate_build
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path),
                              executor_factory=lambda: SimpleNamespace(root=tmp_path / "rt"))
    manifest = {"name": "observer", "entry": "main.py", "requires": {
        "network": False, "writes": False}, "output_schema": {"type": "object"},
        "limits": {"wall_s": 1, "cpu_s": 1, "mem_mb": 128}}
    files = {"main.py": b"from helper import value\ndef handle(input, ctx): return value()\n",
             "helper.py": b"def value(): return {'ok': True}\n"}
    runtime.ready["observer"] = {"manifest": manifest, "files": files}
    runtime.attach_hook("observer", event="post_tool")
    staged = []

    def capture(bundle, *_args, **_kwargs):
        staged.append(bundle)
        raise RuntimeError("staged")

    monkeypatch.setattr(kernel_mutate_build, "stage_unit", capture)
    result = runtime.invoke_hook(event="post_tool", facts={"tool": "read"})
    assert not result.enabled
    assert staged == [files]


def test_observing_hook_run_budget_uses_durable_elapsed_seconds(tmp_path, monkeypatch):
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    journal = Journal(tmp_path)
    runtime = MutationRuntime(level="authored", journal=journal)
    unit = {"name": "observer", "requires": {"network": False, "writes": False},
            "source": "def handle(input, ctx): return {}\n"}
    runtime.attach_hook(unit, event="post_tool")
    journal.append("HOOK_TRIGGER_COMPLETE", invocation_id="old", name="observer",
                   event="post_tool", parent_turn_id="old", text="{}", duration_s=10.0)
    monkeypatch.setattr(runtime._hook_manager(), "_run",
                        lambda *_args, **_kwargs: pytest.fail("over-budget hook launched"))
    result = runtime.invoke_hook(event="post_tool", facts={"tool": "read", "call_id": "new"})
    assert not result.enabled
    assert runtime.hook_failures("observer") == 0
    assert any(row["kind"] == "HOOK_DISABLED" and row["reason"] == "run_wall_budget"
               for row in journal.events())


def test_observing_hook_launch_uses_remaining_trigger_time(tmp_path, monkeypatch):
    import time
    import kernel_mutate_build
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path))
    runtime._ensure_executor = lambda: SimpleNamespace(root=tmp_path)
    observed = []

    def slow_stage(*_args):
        time.sleep(0.06)
        return SimpleNamespace(dir=tmp_path)

    def launch(spec):
        observed.append(spec.limits.wall_s)
        return SimpleNamespace(cleanup=SimpleNamespace(ok=True), timed_out=True,
                               stdout_overflow=False, exit_code=0)

    monkeypatch.setattr(kernel_mutate_build, "stage_unit", slow_stage)
    runtime._launch = launch
    result = runtime._hook_manager()._run({"name": "observer", "entry": "main.py",
                                           "source": "def handle(input, ctx): return {}\n"},
                                          "trigger", {}, limit_s=1.0)
    assert not result.enabled
    assert observed and observed[0] < 0.99


def test_observing_hook_runtime_error_is_non_gating(tmp_path, monkeypatch):
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path))
    runtime.attach_hook({"name": "observer", "requires": {"network": False, "writes": False},
                         "source": "def handle(input, ctx): return {}\n"}, event="post_tool")
    monkeypatch.setattr(runtime._hook_manager(), "_run",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
    for index in range(3):
        result = runtime.invoke_hook(event="post_tool", facts={
            "tool": "read", "call_id": f"call-{index}"})
        assert not result.enabled
    assert runtime.hook_failures("observer") == 3
    assert not runtime._hook_manager().attached


def test_watch_ticks_record_snapshot_before_invocation_and_coalesce(tmp_path, monkeypatch):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "events.log"
    source.write_text("first\n")
    run = tmp_path / "run"
    run.mkdir()
    runtime = MutationRuntime(level="prebuilt", journal=Journal(run), repo=repo,
                              policy=MutationPolicy(level="prebuilt"),
                              executor_factory=lambda: SimpleNamespace(root=tmp_path / "rt"))
    monkeypatch.setattr(runtime, "_use", lambda *_a, **_k: source.read_text())
    registered = runtime.dispatch({"op": "watch", "id": "log_watch", "args": json.dumps({
        "input": {"input_id": "in0"}, "inputs": ["events.log"],
        "every_turns": 1, "max_ticks": 3})}, "register")
    assert "WATCHING" in registered
    first = runtime.tick("parent-1", 1)
    assert len(first) == 1 and "first" in first[0]
    second = runtime.tick("parent-2", 2)
    assert second == []
    source.write_text("second " + "x" * 200 + "\n")
    third = runtime.tick("parent-3", 3)
    assert len(third) == 1 and "coalesced 1" in third[0]
    assert "coalesced 1" in third[0][:80]
    rows = runtime.journal.events()
    decisions = [row for row in rows if row["kind"] == "WATCH_TICK_INTENT"]
    assert len(decisions) == 3
    assert all("snapshot_identity" in row for row in decisions)
    assert not runtime.watches


def test_watch_replay_serves_completed_tick_without_invocation(tmp_path, monkeypatch):
    import kernel_tool_tape
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    journal = Journal(tmp_path)
    journal.append("WATCH_TICK_INTENT", invocation_id="watch-1", parent_turn_id="turn-1",
                   watch="w1", snapshot_identity="snapshot-1", facts=[])
    journal.append("WATCH_TICK_COMPLETE", invocation_id="watch-1", parent_turn_id="turn-1",
                   observation="changed")
    journal.append("WATCH_TICK_INTENT", invocation_id="watch-2", parent_turn_id="turn-2",
                   watch="w1", snapshot_identity="snapshot-2", facts=[])
    journal.append("WATCH_TICK_COMPLETE", invocation_id="watch-2", parent_turn_id="turn-2",
                   observation="changed")
    journal.append("watch_stop", watch="w1", invocation_id="watch-2", reason="max_ticks")
    tape = SimpleNamespace(lookup=lambda call: SimpleNamespace(
        call_id=call, observation="changed"))
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: True)
    monkeypatch.setattr(kernel_tool_tape, "tape_for_run", lambda: tape)
    runtime = MutationRuntime(level="prebuilt", journal=journal, run_dir=tmp_path)
    monkeypatch.setattr(runtime, "_use", lambda *_a, **_k: pytest.fail("replay invoked payload"))
    assert runtime.tick("turn-1", 1) == ["mutation watch w1: changed"]
    assert runtime.tick("turn-2", 2) == ["mutation watch w1: coalesced 1 unchanged ticks"]


def test_resumed_runtime_durably_stops_unclosed_watches(tmp_path):
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    journal = Journal(tmp_path)
    journal.append("watch_start", call="register", watch="w1", unit="log_watch",
                   args={}, every_turns=1, max_ticks=3)
    runtime = MutationRuntime(level="prebuilt", journal=journal, run_dir=tmp_path)
    assert runtime.watches == {}
    assert any(row["kind"] == "watch_stop" and row.get("watch") == "w1"
               and row.get("reason") == "resume" for row in journal.events())


def test_watch_schedule_is_relative_to_registration_and_boundary_id(tmp_path, monkeypatch):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "events.log").write_text("entry\n")
    runtime = MutationRuntime(level="prebuilt", journal=Journal(tmp_path / "run"), repo=repo,
                              policy=MutationPolicy(level="prebuilt"),
                              executor_factory=lambda: SimpleNamespace(root=tmp_path / "rt"))
    monkeypatch.setattr(runtime, "_use", lambda *_a, **_k: "result")
    runtime.dispatch({"op": "watch", "id": "log_watch", "args": json.dumps({
        "input": {}, "inputs": ["events.log"], "every_turns": 2, "max_ticks": 2})}, "register")
    assert runtime.tick("parent-4", 4) == []
    assert runtime.tick("parent-4", 4) == []
    assert len(runtime.tick("parent-5", 5)) == 1
    assert runtime.tick("parent-4", 4) == []
    assert runtime.tick("parent-6", 6) == []
    assert len([row for row in runtime.journal.events()
                if row["kind"] == "WATCH_TICK_INTENT"]) == 1


def test_watch_snapshot_refusal_is_recorded_without_gating_parent(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "events.log"
    source.write_text("entry\n")
    runtime = MutationRuntime(level="prebuilt", journal=Journal(tmp_path / "run"), repo=repo,
                              policy=MutationPolicy(level="prebuilt"),
                              executor_factory=lambda: SimpleNamespace(root=tmp_path / "rt"))
    runtime.dispatch({"op": "watch", "id": "log_watch", "args": json.dumps({
        "input": {}, "inputs": ["events.log"], "every_turns": 1, "max_ticks": 1})}, "register")
    source.unlink()
    source.symlink_to("elsewhere")
    result = runtime.tick("parent-1", 1)
    assert len(result) == 1 and "REFUSED MUTATION_WATCH" in result[0]
    kinds = [row["kind"] for row in runtime.journal.events()]
    assert "WATCH_TICK_INTENT" in kinds and "WATCH_TICK_COMPLETE" in kinds


def test_edit_proposals_reject_set_on_conflict_scope_and_protected_test(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    first = repo / "a.py"
    second = repo / "b.py"
    protected = repo / "test_guard.py"
    for path in (first, second, protected):
        path.write_text("old\n")
    run = tmp_path / "run"
    run.mkdir()
    runtime = MutationRuntime(level="authored", journal=Journal(run), repo=repo,
        policy=MutationPolicy(level="authored", write_globs=("*.py",)),
        edit_scope={"a.py", "b.py"})
    edits = [{"op": "replace", "path": name, "base_sha256": runtime.file_digest(repo / name),
              "text": "new\n"} for name in ("a.py", "b.py")]
    second.write_text("other\n")
    with pytest.raises(ValueError):
        runtime.apply_edit_proposals("conflict", edits, cleanup_ok=True)
    assert first.read_text() == "old\n"
    with pytest.raises(ValueError):
        runtime.apply_edit_proposals("protected", [{"op": "replace", "path": "test_guard.py",
            "base_sha256": runtime.file_digest(protected), "text": "new\n"}], cleanup_ok=True)
    with pytest.raises(ValueError):
        runtime.apply_edit_proposals("scope", [{"op": "create", "path": "new.py",
            "base_sha256": None, "text": "new\n"}], cleanup_ok=True)


def test_edit_proposals_roll_back_whole_set_on_ordinary_write_failure(tmp_path, monkeypatch):
    import kernel_mutate_edits
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("a.py", "b.py"):
        (repo / name).write_text("old\n")
    run = tmp_path / "run"
    run.mkdir()
    runtime = MutationRuntime(level="authored", journal=Journal(run), repo=repo,
                              policy=MutationPolicy(level="authored", write_globs=("*.py",)))
    proposals = [{"op": "replace", "path": name,
                  "base_sha256": runtime.file_digest(repo / name), "text": "new\n"}
                 for name in ("a.py", "b.py")]
    original = kernel_mutate_edits._write_at
    count = 0

    def fail_second(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected second write failure")
        return original(*args)

    monkeypatch.setattr(kernel_mutate_edits, "_write_at", fail_second)
    with pytest.raises(OSError, match="second write failure"):
        runtime.apply_edit_proposals("write-failure", proposals, cleanup_ok=True)
    assert [(repo / name).read_text() for name in ("a.py", "b.py")] == ["old\n", "old\n"]
    assert runtime.edit_state("write-failure") == "REFUSED"


def test_committed_edit_paths_supply_native_receipt_bridge(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "value.py"
    target.write_text("old\n")
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "run"), repo=repo,
                              policy=MutationPolicy(level="authored", write_globs=("*.py",)))
    proposal = {"op": "replace", "path": "value.py",
                "base_sha256": runtime.file_digest(target), "text": "new\n"}
    runtime.apply_edit_proposals("edit-call", [proposal], cleanup_ok=True)
    assert runtime.committed_edit_paths("edit-call") == ("value.py",)
    assert runtime.committed_edit_paths("missing") == ()


@pytest.mark.host_sandbox
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux bubblewrap/seccomp isolate watcher payloads",
)
def test_log_watch_ticks_use_isolated_payload_and_report_rotation(tmp_path):
    from kernel_mutate_exec import MutationExecutor
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "events.log"
    source.write_text("ERROR first\n")
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "rt" / ("a" * 32)
    root.mkdir(parents=True)
    runtime = MutationRuntime(level="prebuilt", journal=Journal(run), repo=repo,
                              policy=MutationPolicy(level="prebuilt"),
                              executor_factory=lambda: MutationExecutor(root, network_envelope="none"))
    try:
        registered = runtime.dispatch({"op": "watch", "id": "log_watch", "args": json.dumps({
            "input": {"input_id": "in0", "patterns": ["ERROR"]},
            "inputs": [{"path": "events.log", "tail_bytes": 4096}],
            "every_turns": 1, "max_ticks": 2})}, "register-watch")
        assert "WATCHING" in registered
        first = runtime.tick("turn-one", 1)
        assert '"count": 1' in first[0]
        replacement = repo / "next.log"
        replacement.write_text("ERROR second\n")
        replacement.replace(source)
        second = runtime.tick("turn-two", 2)
        assert '"snapshot_change": "rotated"' in second[0]
        assert not runtime.watches
    finally:
        runtime.close()
    assert not root.exists()


def test_authoring_send_exhaustion_closes_episode_with_failed_receipt(tmp_path):
    from kernel_mutate_runtime import MutationRuntime
    from kernel_mutate_records import Journal
    from kernel_mutate_projection import reserve_send

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path))
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read an odd file"})},
                     "enter", remaining_turns=6)
    for ordinal in range(4):
        assert reserve_send(runtime, f"request-{ordinal}", 1, 1)
    assert runtime.episode["sends_left"] == 0
    assert runtime.expire_episode() == "mutation episode e1 FAILED authoring sends exhausted"
    assert runtime.episode is None
    assert runtime.last.endswith("FAILED authoring sends exhausted")
    assert any(row["kind"] == "episode_failed" for row in runtime.journal.events())


def test_network_grant_cannot_exceed_shell_ceiling():
    from stateframe.mutate_host import validate_ceiling

    with pytest.raises(ValueError, match="shell-network full"):
        validate_ceiling({"network": "public-web"}, {"shell_network": "none"})
    validate_ceiling({"network": "public-web"}, {"shell_network": "full"})


def test_mutate_name_is_reserved_only_for_enabled_runs():
    from stateframe.runner import mutation_builtin_names

    assert mutation_builtin_names({}) == frozenset()
    assert mutation_builtin_names({"mutations": {"level": "authored"}}) == frozenset({"mutate"})


def test_host_sweeps_at_start_and_defers_executor(tmp_path, monkeypatch):
    from stateframe import mutate_host
    import kernel_mutate_runtime

    monkeypatch.setattr(mutate_host.sys, "platform", "linux")
    policy_module = ModuleType("kernel_mutate_policy")
    policy_module.policy_from = lambda _options: SimpleNamespace(
        enabled=True, level="prebuilt", network="none")
    records_module = ModuleType("kernel_mutate_records")
    records_module.Journal = lambda _run: SimpleNamespace(events=lambda: [])
    calls = []
    sweep_module = ModuleType("kernel_mutate_sweep")
    sweep_module.sweep = lambda root, *, run_locks: (
        calls.append((root, run_locks(tmp_path / ("a" * 32)),
                      run_locks(tmp_path / "other"))) or SimpleNamespace(unresolved=[]))
    monkeypatch.setitem(sys.modules, "kernel_mutate_policy", policy_module)
    monkeypatch.setitem(sys.modules, "kernel_mutate_records", records_module)
    monkeypatch.setitem(sys.modules, "kernel_mutate_sweep", sweep_module)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    before = "kernel_mutate_exec" in sys.modules
    runtime = mutate_host.start({"mutations": {"level": "prebuilt", "network": "none"},
                                 "run_identity_seed": "a" * 32,
                                 "tool_settings": {"shell_network": "none"}}, tmp_path)
    try:
        assert runtime.executor is None
        assert ("kernel_mutate_exec" in sys.modules) is before
        assert calls and calls[0][1:] == (False, True)
    finally:
        kernel_mutate_runtime.install(None)


def test_enter_and_leave_are_durable(tmp_path):
    from kernel_mutate_runtime import MutationRuntime

    journal = SimpleNamespace(rows=[], append=lambda *a, **k: journal.rows.append((a, k)))
    runtime = MutationRuntime(level="authored", journal=journal)
    enter = runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "parse report"})},
                             "call-1", remaining_turns=5)
    assert "episode" in enter and runtime.episode is not None
    leave = runtime.dispatch({"op": "leave"}, "call-2", remaining_turns=4)
    assert "LEFT" in leave and runtime.episode is None
    assert [kind[0] for kind, _ in journal.rows] == ["episode_enter", "episode_leave"]


def test_enter_rejects_ungranted_network_and_bad_inputs():
    from kernel_mutate_runtime import MutationRuntime

    journal = SimpleNamespace(rows=[], append=lambda *a, **k: journal.rows.append((a, k)))
    policy = SimpleNamespace(network="none", write_globs=())
    runtime = MutationRuntime(level="authored", journal=journal, policy=policy)
    assert runtime.dispatch({"op": "enter", "args": '{"goal":"fetch","needs":{"network":true}}'},
                            "c1").startswith("REFUSED MUTATION_GRANT")
    assert runtime.dispatch({"op": "enter", "args": '{"goal":"read","inputs":[1]}'},
                            "c2").startswith("REFUSED MUTATION_BAD_ARGUMENTS")
    assert runtime.episode is None and journal.rows == []


def test_replayed_enter_reconstructs_episode_without_duplicate_journal(tmp_path, monkeypatch):
    import kernel_tool_tape
    from kernel_mutate_runtime import MutationRuntime

    event = {"kind": "episode_enter", "episode": "e1", "goal": "parse report",
             "inputs": [], "unit_kind": "reader", "needs": {}, "call": "c1"}
    journal = SimpleNamespace(events=lambda: [event],
                              append=lambda *_a, **_k: pytest.fail("replay appended a journal event"))
    record = SimpleNamespace(observation="mutation episode e1 OPEN; submit a unit or leave")
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: True)
    monkeypatch.setattr(kernel_tool_tape, "tape_for_run",
                        lambda: SimpleNamespace(lookup=lambda _call: record))
    monkeypatch.setattr(kernel_tool_tape, "replay_decision",
                        lambda *_a, **_k: SimpleNamespace(refuse_reason=None))
    monkeypatch.setattr(kernel_tool_tape, "verify_intent", lambda *_a, **_k: None)
    runtime = MutationRuntime(level="authored", journal=journal, run_dir=tmp_path)
    runtime.dispatch({"op": "enter", "args": '{"goal":"parse report"}'}, "c1")
    assert runtime.episode is not None and runtime.episode["id"] == "e1"


def test_replayed_refused_enter_does_not_require_episode_evidence(tmp_path, monkeypatch):
    import kernel_tool_tape
    from kernel_mutate_runtime import MutationRuntime

    journal = SimpleNamespace(events=lambda: [], append=lambda *_a, **_k: None)
    record = SimpleNamespace(observation="REFUSED MUTATION_BUDGET no turns")
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: True)
    monkeypatch.setattr(kernel_tool_tape, "tape_for_run",
                        lambda: SimpleNamespace(lookup=lambda _call: record))
    monkeypatch.setattr(kernel_tool_tape, "replay_decision",
                        lambda *_a, **_k: SimpleNamespace(refuse_reason=None))
    monkeypatch.setattr(kernel_tool_tape, "verify_intent", lambda *_a, **_k: None)
    runtime = MutationRuntime(level="authored", journal=journal, run_dir=tmp_path)
    assert runtime.dispatch({"op": "enter", "args": '{"goal":"parse report"}'},
                            "c1") == record.observation


def test_replayed_submit_restores_ready_bundle(tmp_path, monkeypatch):
    import base64
    import kernel_tool_tape
    from kernel_mutate_build import source_digest
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    manifest = {"schema": "stateframe-mutation-unit/1", "name": "odd", "revision": 1,
                "parent": None, "entry": "main.py", "summary": "odd", "kind": "reader",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "requires": {"network": False, "writes": False},
                "smoke": [{"input": {}, "expect": {}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    files = {"main.py": b"def handle(input, ctx): return {}\n"}
    journal = Journal(tmp_path)
    blob = journal.put_blob(json.dumps({"manifest": manifest, "files": {
        "main.py": base64.b64encode(files["main.py"]).decode()}}, sort_keys=True).encode())
    journal.append("ADMISSION", call="submit-1", blob=blob, name="odd", revision=1,
                   source_digest=source_digest(files, manifest),
                   admission_digest="sha256:" + "a" * 64)
    record = SimpleNamespace(observation="mutation odd r1 READY smoke 1/1 [e1]")
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: True)
    monkeypatch.setattr(kernel_tool_tape, "tape_for_run",
                        lambda: SimpleNamespace(lookup=lambda _call: record))
    monkeypatch.setattr(kernel_tool_tape, "replay_decision",
                        lambda *_a, **_k: SimpleNamespace(refuse_reason=None))
    monkeypatch.setattr(kernel_tool_tape, "verify_intent", lambda *_a, **_k: None)
    runtime = MutationRuntime(level="authored", journal=journal, run_dir=tmp_path)
    runtime.episode = {"id": "e1", "goal": "odd", "sends_left": 2}
    runtime.dispatch({"op": "submit", "args": "{}"}, "submit-1")
    assert runtime.episode is None
    assert runtime.ready["odd"]["files"] == files


def test_catalog_pages_pinned_prebuilt_names(tmp_path):
    from kernel_mutate_runtime import MutationRuntime

    catalog = tmp_path / "prebuilt"
    catalog.mkdir()
    for name in ("alpha", "beta"):
        unit = catalog / name
        unit.mkdir()
        (unit / "unit.json").write_text(json.dumps({"name": name, "summary": f"{name} summary"}))
    (catalog / "manifest.json").write_text(json.dumps({
        "schema": "stateframe-mutation-prebuilt/1",
        "units": {name: "sha256:" + index * 64 for name, index in
                  (("alpha", "a"), ("beta", "b"))}}))
    journal = SimpleNamespace(append=lambda *_a, **_k: None)
    runtime = MutationRuntime(level="prebuilt", journal=journal, catalog_root=catalog)
    reply = runtime.dispatch({"op": "catalog", "args": '{"offset":1}'}, "c1")
    assert "beta summary" in reply
    assert "alpha summary" not in reply


def test_retain_keeps_ready_source_in_run_audit_only(tmp_path):
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    run = tmp_path / "run"
    run.mkdir()
    runtime = MutationRuntime(level="authored", journal=Journal(run),
                              catalog_root=tmp_path / "missing")
    runtime.ready["odd_reader"] = {
        "manifest": {"name": "odd_reader", "revision": 2, "summary": "Read odd rows"},
        "source_digest": "sha256:" + "a" * 64, "files": {"main.py": b"pass\n"}}
    refused = runtime.dispatch({"op": "retain", "id": "odd_reader",
                                "args": '{"scope":"user"}'}, "retain-user")
    assert refused.startswith("REFUSED MUTATION_GRANT")
    accepted = runtime.dispatch({"op": "retain", "id": "odd_reader"}, "retain-run")
    assert "RETAINED" in accepted
    rows = [row for row in runtime.journal.events() if row["kind"] == "retention"]
    assert len(rows) == 1
    assert rows[0]["call"] == "retain-run" and rows[0]["scope"] == "run"
    assert rows[0]["source_digest"] == "sha256:" + "a" * 64


def test_inspect_ready_revision_from_durable_admission(tmp_path):
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path),
                              catalog_root=tmp_path / "missing")
    runtime.ready["odd_reader"] = {
        "manifest": {"name": "odd_reader", "revision": 2, "summary": "Read odd rows",
                     "kind": "reader", "requires": {"network": False, "writes": False}},
        "source_digest": "sha256:" + "b" * 64, "files": {"main.py": b"pass\n"}}
    result = runtime.dispatch({"op": "inspect", "id": "odd_reader"}, "inspect-odd")
    assert '"revision": 2' in result
    assert '"source_digest": "sha256:' in result


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"a":NaN}',
                                      '{"a":1e999}', '{"a":123456789012345678901234567890123456789012345}',
                                      '[1]'])
def test_args_refuse_noncanonical_json(payload):
    from kernel_mutate_runtime import MutationRuntime

    journal = SimpleNamespace(append=lambda *_a, **_k: None)
    runtime = MutationRuntime(level="prebuilt", journal=journal)
    assert runtime.dispatch({"op": "catalog", "args": payload}, "c1").startswith(
        "REFUSED MUTATION_BAD_ARGUMENTS")


@pytest.mark.parametrize("bad", ["", "unknown", "OFF", "prebuilt "])
def test_invalid_level_refuses(bad):
    from kernel_mutate_runtime import schema_for_level

    with pytest.raises(ValueError):
        schema_for_level(bad)


def _usage_args(reply: str) -> dict:
    line = next(part for part in reply.splitlines() if part.startswith("usage: "))
    raw = line.split("args=", 1)[1]
    raw = raw[:raw.index('")') + 1] if '")' in raw else raw
    return json.loads(json.loads(raw))


@pytest.mark.parametrize("op", ["enter", "use", "watch", "submit"])
def test_malformed_call_returns_exact_usage(tmp_path, op):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "run"), repo=repo,
                              policy=MutationPolicy(level="authored"))
    fields = {"op": op, "id": "file_facts"} if op in {"use", "watch"} else {"op": op}
    if op == "submit":
        runtime.episode = {"id": "episode", "sends_left": 2}
    reply = runtime.dispatch(fields, f"call-{op}", remaining_turns=10)
    assert reply.startswith("REFUSED MUTATION_BAD_ARGUMENTS")
    assert isinstance(_usage_args(reply), dict)


def test_enter_usage_example_is_accepted_verbatim(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path),
                              policy=MutationPolicy(level="authored"))
    refused = runtime.dispatch({"op": "enter", "id": "events_reader"}, "call-1", remaining_turns=10)
    args = _usage_args(refused)
    accepted = runtime.dispatch({"op": "enter", "id": "events_reader", "args": json.dumps(args)},
                                "call-2", remaining_turns=10)
    assert not accepted.startswith("REFUSED"), accepted
    assert runtime.episode is not None


def _authoring_runtime(tmp_path, monkeypatch):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    import kernel_mutate_qualify

    monkeypatch.setattr(kernel_mutate_qualify, "_binding", lambda: {
        "closure_digest": "closure", "sdk_digest": "sdk"})
    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "runtime"
    root.mkdir()

    class Executor:
        def __init__(self):
            self.root = root
            self.calls = []

        def run(self, spec):
            self.calls.append(spec)
            result = ({"ok": True, "errors": [], "hints": []}
                      if spec.role == "validator" else {"bytes": 4})
            body = json.dumps({"wire": 1, "call": spec.call,
                               "kind": "result", "result": result}).encode()
            return SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                                   stdout_overflow=False, timed_out=False, exit_code=0,
                                   wall_s=0.01, cpu_s=0.01, cleanup=SimpleNamespace(ok=True))

    executor = Executor()
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "run"), repo=repo,
                              policy=MutationPolicy(level="authored"),
                              executor_factory=lambda: executor)
    return runtime, repo, executor


def _live_submit(tag: str) -> dict:
    """A real model `mutate submit` call from a live run, stored without server metadata."""
    path = Path(__file__).resolve().parent / "fixtures" / "mutations" / f"{tag}_submit.json"
    return json.loads(path.read_text())["call"]


def test_r5_authored_smoke_inputs_are_admitted_before_staging(tmp_path, monkeypatch):
    from kernel_mutate_wire import strict_loads

    submitted = json.loads(_live_submit("r5")["args"])
    runtime, repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    (repo / "data").mkdir()
    (repo / "data/events.b64").write_bytes(b"YWJj")
    submitted["manifest"]["smoke"][0]["inputs"] = ["data/events.b64"]
    submitted["files"]["main.py"] = (
        "def handle(input, ctx):\n"
        "    return {'bytes': len(ctx.input_bytes(input['input_id']))}\n")
    assert "OPEN" in runtime.dispatch({"op": "enter", "id": "events_reader",
        "args": json.dumps({"goal": "Read events", "inputs": ["data/events.b64"]})}, "enter-r5")
    reply = runtime.dispatch({"op": "submit", "id": "events_reader",
                              "args": json.dumps(submitted)}, "submit-r5")
    assert "READY" in reply, reply
    smoke = next(spec for spec in executor.calls if spec.role == "smoke")
    assert list(smoke.inputs) == ["in0"]
    assert smoke.inputs["in0"].read_bytes() == b"YWJj"
    assert strict_loads(smoke.stdin_doc)["inputs"][0]["name"] == "in0"


def test_authored_smoke_secret_path_refuses_before_unit_staging(tmp_path, monkeypatch):
    import kernel_mutate_build

    runtime, repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    (repo / "public.txt").write_text("safe")
    (repo / ".env.secret").write_text("private")
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read data"})}, "enter-secret")
    monkeypatch.setattr(kernel_mutate_build, "stage_unit",
                        lambda *_args, **_kwargs: pytest.fail("unit was staged before input admission"))
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "unit", "revision": 1,
                "parent": None, "entry": "main.py", "summary": "Read data", "kind": "reader",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "requires": {"network": False, "writes": False},
                "smoke": [{"input": {"input_id": "in0"},
                           "inputs": ["public.txt"], "expect": {}},
                          {"input": {"input_id": "in0"},
                           "inputs": [".env.secret"], "expect": {}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "manifest": manifest, "files": {"main.py": "def handle(input, ctx): return {}\n"}})},
        "submit-secret")
    assert reply.startswith("REFUSED MUTATION_ADMISSION"), reply
    assert ".env.secret" in reply
    assert executor.calls == []
    assert not list((executor.root / "in").iterdir())
    assert not (executor.root / "units").exists()


def test_input_admission_refusal_escapes_path_control_characters(tmp_path):
    from kernel_mutate_inputs import admit_inputs
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError) as refused:
        admit_inputs(repo, [".env\nfake reply"], MutationPolicy(level="authored"),
                     snapshot_root=tmp_path / "snapshot")
    assert "\\n" in str(refused.value)
    assert "\n" not in str(refused.value)


def test_submit_short_form_usage_is_accepted_verbatim(tmp_path, monkeypatch):
    from kernel_mutate_runtime import USAGE

    runtime, repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    (repo / "data").mkdir()
    (repo / "data/x.bin").write_bytes(b"data")
    assert "OPEN" in runtime.dispatch({"op": "enter", "id": "events_reader",
        "args": json.dumps({"goal": "parse data/x.bin into totals", "inputs": ["data/x.bin"],
                           "kind": "reader"})}, "enter-example")
    short = USAGE["submit"].split("; full form:", 1)[0]
    args = json.loads(json.loads(short.split("args=", 1)[1][:-1]))
    reply = runtime.dispatch({"op": "submit", "args": json.dumps(args)}, "submit-example")
    assert "events_reader r1 READY" in reply, reply
    assert "def handle(input, ctx)" in args["code"]
    assert "ctx.input_bytes(input_id)" in USAGE["submit"]
    assert executor.calls[-1].role == "smoke"


def test_enter_reply_includes_unit_contract_and_short_submit_usage(tmp_path):
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime, USAGE

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path))
    reply = runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read input"})},
                             "enter-contract")
    assert "unit contract: main.py defines handle(input, ctx) -> dict" in reply
    assert "ctx.input_bytes(input['input_id']) with ids in0, in1" in reply
    assert reply.endswith("\nsubmit: " + USAGE["submit"].split("; full form:", 1)[0])


def test_submit_short_form_without_inputs_and_repair_revision(tmp_path, monkeypatch):
    runtime, _repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Compute"})}, "enter-one")
    args = {"code": "def handle(input, ctx): return {'bytes': 4}\n"}
    assert "unit r1 READY" in runtime.dispatch({"op": "submit", "args": json.dumps(args)}, "submit-one")
    assert executor.calls[-1].inputs == {}
    first = runtime.ready["unit"]["source_digest"]
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Repair"})}, "enter-two")
    assert "unit r2 READY" in runtime.dispatch({"op": "submit", "args": json.dumps(args)}, "submit-two")
    assert runtime.ready["unit"]["manifest"]["parent"] == first


def test_frame_refusal_in_episode_includes_short_submit_guidance(tmp_path):
    from kernel_mutate_runtime import MutationRuntime, USAGE

    journal = SimpleNamespace(rows=[], append=lambda *a, **k: journal.rows.append((a, k)))
    runtime = MutationRuntime(level="authored", journal=journal)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read"})}, "enter-first")
    refused = runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Again"})}, "enter-second")
    assert refused.startswith("REFUSED MUTATION_FRAME"), refused
    assert refused.endswith("\nusage: " + USAGE["submit"].split("; full form:", 1)[0])


def test_submit_missing_smoke_input_names_rule_and_repair_budget(tmp_path, monkeypatch):
    runtime, _repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read records"})}, "enter-rule")
    runtime.episode["sends_left"] = 3
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "unit",
                "revision": 1, "parent": None, "entry": "main.py",
                "summary": "Read records", "kind": "reader",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "requires": {"network": False, "writes": False},
                "smoke": [{"expect": {}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "manifest": manifest, "files": {"main.py": "def handle(input, ctx): return {}\n"}})},
        "submit-rule")
    assert "REFUSED MUTATION_ADMISSION SMOKE:" in reply
    assert "smoke[0]: needs input (object) and expect (object)" in reply
    assert "repair: fix the unit and submit again" in reply
    assert "revision 1" in reply
    assert "sends left 3" in reply
    assert executor.calls == []


def test_direct_submit_returns_repairable_build_refusal(tmp_path, monkeypatch):
    runtime, _repo, _executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read records"})}, "enter-direct")
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "unit",
                "revision": 1, "parent": None, "entry": "main.py",
                "summary": "Read records", "kind": "reader",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "requires": {"network": False, "writes": False},
                "smoke": [{"expect": {}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    reply = runtime._submit({"manifest": manifest,
                             "files": {"main.py": "def handle(input, ctx): return {}\n"}},
                            "submit-direct")
    assert reply.startswith("REFUSED MUTATION_ADMISSION SMOKE:")
    assert "sends left 4" in reply


def test_submit_handler_error_frame_reaches_model_reply(tmp_path, monkeypatch):
    runtime, _repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read records"})}, "enter-error")
    original_run = executor.run

    def run(spec):
        if spec.role == "validator":
            return original_run(spec)
        body = json.dumps({"wire": 1, "call": spec.call, "kind": "error",
                           "error": {"type": "HandlerError", "message": "ValueError: bad row",
                                     "trace": '  File "/sf/unit/main.py", line 2, in handle\nValueError: bad row'},
                           "stderr": "last parser note"}).encode()
        return SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                               stdout_overflow=False, timed_out=False, exit_code=3,
                               wall_s=0.01, cpu_s=0.01, cleanup=SimpleNamespace(ok=True))

    executor.run = run
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "code": "def handle(input, ctx):\n    raise ValueError('bad row')\n"})}, "submit-error")
    assert "REFUSED MUTATION_ADMISSION EXECUTION:" in reply
    assert "HandlerError" in reply and "ValueError: bad row" in reply
    assert "/sf/unit/main.py" in reply and "last parser note" in reply


def test_submit_smoke_mismatch_reaches_model_reply(tmp_path, monkeypatch):
    runtime, _repo, _executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Count bytes"})}, "enter-diff")
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "code": "def handle(input, ctx): return {'bytes': 4}\n",
        "expect": {"bytes": 5}})}, "submit-diff")
    assert "REFUSED MUTATION_ADMISSION SMOKE_MISMATCH:" in reply
    assert 'expected: {"bytes":5}' in reply
    assert 'actual: {"bytes":4}' in reply
    assert "first difference: bytes" in reply


def test_submit_validator_syntax_location_reaches_model_reply(tmp_path, monkeypatch):
    runtime, _repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Count bytes"})}, "enter-syntax")

    def run(spec):
        assert spec.role == "validator"
        body = json.dumps({"wire": 1, "call": spec.call, "kind": "result",
                           "result": {"ok": False, "errors": [
                               {"file": "main.py", "line": 2, "col": 5, "msg": "invalid syntax"}],
                                      "hints": []}}).encode()
        return SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                               stdout_overflow=False, timed_out=False, exit_code=0,
                               wall_s=0.01, cpu_s=0.01, cleanup=SimpleNamespace(ok=True))

    executor.run = run
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "code": "def handle(input, ctx):\n    broken =\n"})}, "submit-syntax")
    assert "REFUSED MUTATION_ADMISSION STATIC:" in reply
    assert "main.py:2:5: invalid syntax" in reply


def test_submit_bad_arguments_names_both_accepted_forms(tmp_path):
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path),
                              policy=MutationPolicy(level="authored"))
    runtime.episode = {"id": "e1", "sends_left": 2, "goal": "Read"}
    reply = runtime.dispatch({"op": "submit", "args": "{}"}, "submit-bad")
    assert "submit requires code (short form) or manifest and files" in reply


def test_submit_malformed_requires_is_repairable(tmp_path, monkeypatch):
    runtime, _repo, _executor = _authoring_runtime(tmp_path, monkeypatch)
    runtime.dispatch({"op": "enter", "args": json.dumps({"goal": "Read records"})}, "enter-requires")
    manifest = {"schema": "stateframe-mutation-unit/1", "name": "unit",
                "revision": 1, "parent": None, "entry": "main.py",
                "summary": "Read records", "kind": "reader",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
                "requires": [], "smoke": [{"input": {}, "expect": {}}],
                "limits": {"wall_s": 10, "cpu_s": 5, "mem_mb": 256}}
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "manifest": manifest, "files": {"main.py": "def handle(input, ctx): return {}\n"}})},
        "submit-requires")
    assert reply.startswith("REFUSED MUTATION_ADMISSION REQUIRES:"), reply
    assert "requires: needs boolean network and writes" in reply


def test_r6_live_submit_gets_a_repairable_rule_level_diagnostic(tmp_path, monkeypatch):
    """r6: the real model's full-form submit was answered with a bare SMOKE code and abandoned."""
    runtime, repo, _executor = _authoring_runtime(tmp_path, monkeypatch)
    (repo / "data").mkdir()
    (repo / "data/events.b64").write_bytes(b"YWJj")
    assert "OPEN" in runtime.dispatch({"op": "enter", "id": "reader_unit",
        "args": json.dumps({"goal": "Read events", "inputs": ["data/events.b64"]})}, "enter-r6")
    call = _live_submit("r6")
    reply = runtime.dispatch({"op": "submit", "args": call["args"]}, "submit-r6")
    assert reply.startswith("REFUSED MUTATION_ADMISSION"), reply
    code_and_detail = reply.splitlines()[0]
    assert ":" in code_and_detail and len(code_and_detail) > len("REFUSED MUTATION_ADMISSION SMOKE")
    assert "repair:" in reply and "sends left" in reply


def test_repair_diagnostic_never_leaks_secrets_or_host_paths(tmp_path, monkeypatch):
    """The error frame and stderr are untrusted; secrets and host paths must not reach the reply."""
    import kernel_mutate_build
    from kernel_mutate_build import BuildError

    runtime, repo, _executor = _authoring_runtime(tmp_path, monkeypatch)
    canary = "SFCANARY-" + "7f3a9c"
    host_path = str(tmp_path / "home" / "user" / ".ssh" / "id_ed25519")
    monkeypatch.setenv("STATEFRAME_TEST_SECRET", canary)

    def failing_run(*_args, **_kwargs):
        raise BuildError("EXECUTION", "EXECUTION exit=3 HandlerError: boom " + host_path + " " + canary
                         + "\n" + "x" * 5000)

    monkeypatch.setattr(kernel_mutate_build, "_run", failing_run)
    assert "OPEN" in runtime.dispatch({"op": "enter", "id": "leaky",
        "args": json.dumps({"goal": "Read events"})}, "enter-leak")
    reply = runtime.dispatch({"op": "submit", "args": json.dumps(
        {"code": "def handle(input, ctx):\n    raise RuntimeError('boom')\n"})}, "submit-leak")
    assert reply.startswith("REFUSED MUTATION_ADMISSION"), reply
    assert str(tmp_path) not in reply and host_path not in reply
    assert len(reply) <= 2000


def test_use_accepts_id_inside_args_or_the_single_ready_unit(tmp_path):
    """r8: the model put `id` inside args, then dropped it; both resolve when unambiguous."""
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "run"), repo=repo,
                              policy=MutationPolicy(level="authored"))
    seen = []
    runtime._use = lambda name, args, call_id, **_kw: seen.append((name, dict(args))) or "ok"
    runtime.ready["reader"] = {"manifest": {"revision": 1}}
    assert runtime._dispatch("use", None, {"id": "reader", "input": {}}, "c1", 10) == "ok"
    assert runtime._dispatch("use", None, {"input": {}}, "c2", 10) == "ok"
    assert seen == [("reader", {"input": {}}), ("reader", {"input": {}})]
    runtime.ready["other"] = {"manifest": {"revision": 1}}
    assert runtime._dispatch("use", None, {"input": {}}, "c3", 10).startswith("REFUSED MUTATION_BAD_ARGUMENTS")


def test_r9_episode_without_inputs_states_it_and_submit_can_name_inputs(tmp_path, monkeypatch):
    """r9: the episode opened with no inputs, so the short-form smoke ran on {} and KeyError'd."""
    runtime, repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    (repo / "data").mkdir()
    (repo / "data/events.b64").write_bytes(b"YWJj")
    opened = runtime.dispatch({"op": "enter", "args": json.dumps(
        {"goal": "decode events", "id": "reader"})}, "enter-r9")
    assert "OPEN" in opened and "inputs: none" in opened
    assert runtime.episode["unit_id"] == "reader"
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({
        "code": "def handle(input, ctx):\n    return {'n': len(ctx.input_bytes(input['input_id']))}\n",
        "inputs": ["data/events.b64"]})}, "submit-r9")
    assert "READY" in reply, reply
    smoke = next(spec for spec in executor.calls if spec.role == "smoke")
    assert list(smoke.inputs) == ["in0"]


def test_enter_names_unknown_fields(tmp_path, monkeypatch):
    runtime, _repo, _executor = _authoring_runtime(tmp_path, monkeypatch)
    reply = runtime.dispatch({"op": "enter", "args": json.dumps(
        {"goal": "decode events", "sources": ["x"]})}, "enter-unknown")
    assert reply.startswith("REFUSED MUTATION_BAD_ARGUMENTS unknown enter field sources"), reply


def _ready_reader(runtime, repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data/events.b64").write_bytes(b"YWJj")
    assert "OPEN" in runtime.dispatch({"op": "enter", "id": "events_reader", "args": json.dumps(
        {"goal": "decode events", "inputs": ["data/events.b64"]})}, "enter-ready")
    reply = runtime.dispatch({"op": "submit", "args": json.dumps({"code": (
        "def handle(input, ctx):\n    return {'bytes': len(ctx.input_bytes(input['input_id']))}\n")})},
        "submit-ready")
    assert "READY" in reply, reply


def test_r10_use_without_inputs_reads_the_files_it_was_smoke_tested_on(tmp_path, monkeypatch):
    """r10: `use` with input but no inputs list failed; it now mounts the admitted smoke inputs."""
    runtime, repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    _ready_reader(runtime, repo)
    reply = runtime.dispatch({"op": "use", "id": "events_reader",
                              "args": json.dumps({"input": {"input_id": "in0"}})}, "use-r10")
    assert not reply.startswith("REFUSED"), reply
    payload = [spec for spec in executor.calls if spec.role == "payload"][-1]
    assert list(payload.inputs) == ["in0"]


def test_r10_use_failure_carries_the_error_frame_diagnostic(tmp_path, monkeypatch):
    """r10: a failed `use` said only "payload did not complete"; it now shows the handler error."""
    runtime, repo, executor = _authoring_runtime(tmp_path, monkeypatch)
    _ready_reader(runtime, repo)
    original = executor.run

    def failing(spec):
        if spec.role != "payload":
            return original(spec)
        body = json.dumps({"wire": 1, "call": spec.call, "kind": "error", "stderr": "",
                           "error": {"type": "HandlerError", "message": "unknown input id",
                                     "trace": "File \"/sf/unit/main.py\", line 2, in handle"}}).encode()
        return SimpleNamespace(stdout=b"SFMW1 " + str(len(body)).encode() + b"\n" + body,
                               stdout_overflow=False, timed_out=False, exit_code=3,
                               wall_s=0.01, cpu_s=0.01, cleanup=SimpleNamespace(ok=True))

    executor.run = failing
    reply = runtime.dispatch({"op": "use", "id": "events_reader", "args": json.dumps(
        {"input": {"input_id": "in0"}})}, "use-fail")
    assert reply.startswith("REFUSED MUTATION_EXECUTION"), reply
    assert "HandlerError" in reply and "unknown input id" in reply and "/sf/unit/main.py" in reply
