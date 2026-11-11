"""Regression coverage for mutation receipts against the current engine."""

import errno
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from kernel_frame_optional_allocation import (
    FrameAllocationReplanNeeded, validate_optional_dependencies,
)
from kernel_post_edit_refresh import refresh_after_applied_edit
from kernel_stateframe_framefit5 import _update_headcarry
from kernel_stateframe_headcarry import render_headcarry


def _ordinary_launch_harness(tmp_path, monkeypatch, *, startup_s, fail_spawn=False,
                             registration_s=0, arm_s=0):
    import kernel_mutate_exec as execution
    from kernel_exec_session import CleanupReport, ExecResult, ExecSessionError

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: clock.now, monotonic_ns=lambda: int(clock.now * 1e9)))
    monkeypatch.setattr(execution, "rpc_timeout_seconds", lambda: 15.0)
    root = tmp_path / "executor"
    for name in ("stage", "boot", "session"):
        (root / name).mkdir(parents=True)
    executor = object.__new__(execution.MutationExecutor)
    executor.root = root
    executor._stage = root / "stage"
    executor._boot = root / "boot"
    executor._session_dir = root / "session"
    executor._ledger = (root / "launches.jsonl").open("ab", buffering=0)
    executor._closed = False
    executor._seen = set()
    events = []
    def stage_hook(state):
        events.append((state, None))
        if state == "REGISTERED":
            clock.now += registration_s
    executor._launch_stage_hook = stage_hook
    executor._roots = ()
    executor._broker_roots = ()
    executor._hidden = ()
    executor.network_envelope = "none"
    executor._write_owner = lambda: None
    def stage_spec(spec):
        stage = executor._stage / spec.launch
        for name in ("ctl", "in", "resp", "unit"):
            (stage / name).mkdir(parents=True)
        return stage, {name: str(stage / name) for name in ("ctl", "in", "resp", "unit")}
    executor._stage_spec = stage_spec

    control = SimpleNamespace(timeout=15.0)
    control.gettimeout = lambda: control.timeout
    control.settimeout = lambda value: setattr(control, "timeout", value)
    class Session:
        _control = control

        def _spawn(self, command, cap, **kwargs):
            events.append(("spawn", kwargs, cap))
            assert not (executor._stage / spec.launch / "ctl" / "release.json").exists()
            assert cap.limits.memory_bytes == spec.limits.mem_mb << 20
            assert cap.limits.cpu_seconds == spec.limits.cpu_s
            clock.now += startup_s
            if fail_spawn:
                raise ExecSessionError("execution supervisor did not answer")
            return SimpleNamespace(id="job", identity={"pid": 99999999, "start": [0, 0]})

        def _rpc(self, request):
            events.append(("arm", request))
            assert not (executor._stage / spec.launch / "ctl" / "release.json").exists()
            clock.now += arm_s
            return {"started": clock.now, "expired": False}

        def _collect(self, handle, cap):
            events.append(("payload", None))
            release = executor._stage / spec.launch / "ctl" / "release.json"
            assert json.loads(release.read_text())["deadline"] > clock.now
            clock.now += 2.0
            return ExecResult(0, False, b"ok", False, 2, 2.0,
                              CleanupReport(True, 0, (), ""))

        def stop_all(self, grace):
            events.append(("stop_all", grace))
            return CleanupReport(True, 0, (), "")

        def close(self):
            events.append(("socket_closed", None))

    executor._session = Session()
    launch = uuid.uuid4().hex
    doc = {"wire": 1, "call": "case", "launch": launch, "role": "payload",
           "grants": {"network": False, "writes": False}}
    spec = execution.LaunchSpec("payload", "case", launch, None, {}, {},
                                json.dumps(doc).encode(), execution.MutationLimits(wall_s=3))
    return executor, spec, events, clock


def test_ordinary_payload_wall_excludes_slow_startup(tmp_path, monkeypatch):
    executor, spec, events, _ = _ordinary_launch_harness(tmp_path, monkeypatch, startup_s=4)
    try:
        outcome = executor.run(spec)
        assert outcome.exit_code == 0
        assert not outcome.timed_out
        assert [event[0] for event in events].index("arm") < [event[0] for event in events].index("payload")
        assert next(event for event in events if event[0] == "spawn")[1]["armable_wall_s"] == spec.limits.wall_s
    finally:
        executor.close()


@pytest.mark.parametrize("startup_s,fail_spawn,registration_s,expected_state", [
    (11, False, 0, "PREPARED"),
    (11, True, 0, "PREPARED"),
    (1, False, 10, "REAPED"),
])
def test_ordinary_startup_timeout_refuses_and_closes_executor(
        tmp_path, monkeypatch, startup_s, fail_spawn, registration_s, expected_state):
    from kernel_mutate_exec import MutationStartupTimeout

    executor, spec, events, _ = _ordinary_launch_harness(
        tmp_path, monkeypatch, startup_s=startup_s, fail_spawn=fail_spawn,
        registration_s=registration_s)
    with pytest.raises(MutationStartupTimeout, match="startup exceeded bound"):
        executor.run(spec)
    assert "payload" not in [event[0] for event in events]
    assert "stop_all" in [event[0] for event in events]
    assert "socket_closed" in [event[0] for event in events]
    assert executor._ledger.closed
    assert not (executor._stage / spec.launch).exists()
    assert not executor._session_dir.exists()
    assert not executor._boot.exists()
    rows = [json.loads(line) for line in (executor.root / "launches.jsonl").read_text().splitlines()]
    assert rows[-1]["state"] == expected_state


def test_ordinary_payload_cannot_run_before_wall_is_armed(tmp_path, monkeypatch):
    executor, spec, events, _ = _ordinary_launch_harness(tmp_path, monkeypatch, startup_s=1)
    try:
        executor.run(spec)
        order = [event[0] for event in events]
        assert order.index("arm") < order.index("RELEASED") < order.index("payload")
    finally:
        executor.close()


def test_release_record_stays_valid_after_slow_arm_reply(tmp_path, monkeypatch):
    executor, spec, _, _ = _ordinary_launch_harness(
        tmp_path, monkeypatch, startup_s=1, arm_s=6)
    try:
        assert executor.run(spec).exit_code == 0
    finally:
        executor.close()


def test_ordinary_boot_cannot_import_payload_before_release(monkeypatch):
    import sf_mutate_boot as boot

    marker = []
    doc = {"wire": 1, "call": "case", "launch": "a" * 32,
           "nonce": "nonce", "role": "payload"}
    monkeypatch.setattr(boot, "_read_doc", lambda: doc)
    monkeypatch.setattr(boot, "wait_for_release", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(boot, "_unit_files", lambda *_args: marker.append("loaded"))
    monkeypatch.setattr(boot, "os", SimpleNamespace(O_WRONLY=os.O_WRONLY,
                                                     open=lambda *_args: 3,
                                                     dup2=lambda *_args: None,
                                                     close=lambda *_args: None))
    monkeypatch.setattr(boot.sys, "argv", ["sf_mutate_boot.py"])
    monkeypatch.setattr(boot.sys, "stderr", boot.sys.stderr)
    assert boot.main() == 4
    assert marker == []


def test_release_publish_is_never_followed_by_startup_refusal(tmp_path, monkeypatch):
    import kernel_mutate_exec as execution

    executor, spec, events, clock = _ordinary_launch_harness(
        tmp_path, monkeypatch, startup_s=1)
    replace = execution.os.replace
    def delayed_replace(source, target):
        assert any(event[0] == "arm" for event in events)
        replace(source, target)
        clock.now += 11
    monkeypatch.setattr(execution.os, "replace", delayed_replace)

    try:
        assert executor.run(spec).exit_code == 0
        assert "payload" in [event[0] for event in events]
    finally:
        executor.close()


def test_boot_release_wait_covers_scaled_startup(monkeypatch):
    import sf_mutate_boot as boot

    timeouts = []
    doc = {"wire": 1, "call": "case", "launch": "a" * 32,
           "nonce": "nonce", "role": "payload", "startup_wait_s": 82.0}
    monkeypatch.setattr(boot, "_read_doc", lambda: doc)
    monkeypatch.setattr(boot, "wait_for_release", lambda *_args, **kwargs:
                        timeouts.append(kwargs["timeout"]) or None)
    monkeypatch.setattr(boot, "os", SimpleNamespace(O_WRONLY=os.O_WRONLY,
                                                     open=lambda *_args: 3,
                                                     dup2=lambda *_args: None,
                                                     close=lambda *_args: None))
    monkeypatch.setattr(boot.sys, "argv", ["sf_mutate_boot.py"])
    monkeypatch.setattr(boot.sys, "stderr", boot.sys.stderr)
    assert boot.main() == 4
    assert timeouts == [82.0]


def test_broker_boot_release_wait_covers_scaled_startup(monkeypatch):
    import sf_mutate_boot as boot

    timeouts = []
    monkeypatch.setattr(boot, "wait_for_release", lambda *_args, **kwargs:
                        timeouts.append(kwargs["timeout"]) or None)
    monkeypatch.setattr(boot, "os", SimpleNamespace(O_WRONLY=os.O_WRONLY,
                                                     open=lambda *_args: 3,
                                                     dup2=lambda *_args: None,
                                                     close=lambda *_args: None))
    monkeypatch.setattr(boot.sys, "argv", ["sf_mutate_boot.py", "--broker",
                                            "a" * 32, "nonce", "82.0"])
    monkeypatch.setattr(boot.sys, "stderr", boot.sys.stderr)
    assert boot.main() == 4
    assert timeouts == [82.0]


def test_ordinary_startup_window_uses_existing_timeout_scale(tmp_path, monkeypatch):
    import kernel_mutate_exec as execution

    executor, spec, events, _ = _ordinary_launch_harness(tmp_path, monkeypatch, startup_s=31)
    monkeypatch.setattr(execution, "rpc_timeout_seconds", lambda: 60.0)
    try:
        assert executor.run(spec).exit_code == 0
        stdin = next(event for event in events if event[0] == "spawn")[1]["stdin"]
        assert json.loads(stdin)["startup_wait_s"] == 42.0
    finally:
        executor.close()


def test_startup_timeout_refusal_is_journaled(tmp_path, monkeypatch):
    import kernel_tool_tape
    from kernel_mutate_exec import MutationStartupTimeout
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    journal = Journal(tmp_path)
    runtime = MutationRuntime(level="authored", journal=journal, run_dir=tmp_path)
    def fail_dispatch(*_args):
        raise MutationStartupTimeout("mutation startup exceeded bound")
    runtime._dispatch = fail_dispatch
    tape = SimpleNamespace(intent=lambda **_fields: None, complete=lambda **_fields: None)
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: False)
    monkeypatch.setattr(kernel_tool_tape, "tape_for_run", lambda: tape)

    result = runtime._dispatch_call({"op": "catalog"}, "call")
    assert result == "REFUSED MUTATION_STARTUP_TIMEOUT mutation startup exceeded bound"
    assert journal.events()[-1] == {"kind": "REFUSAL", "call": "call", "observation": result}


def test_mutation_receipt_retires_unavailable_generated_path(tmp_path):
    path = "reports/daily-summary.md"
    stale = {"path": path, "revision": "sha256:old", "start_line": 1,
             "end_line": 1, "content": "old output\n"}
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path),
        _read_reuse_visibility_candidates=[stale],
        _empirical_frame_sources={path: "old output\n"},
        _empirical_current_revisions={path: "sha256:old"},
    )
    receipt = {"kind": "edit", "result_class": "APPLIED",
               "target_path": path, "changed_files": [{"path": path}], "turn": 2}
    binding = {"product_files": {"invoices.csv": "sha256:input"},
               "product_revision": "sha256:current"}

    assert refresh_after_applied_edit(runtime, receipt, binding, (path,)) == {}
    assert runtime._read_reuse_visibility_candidates == []
    assert runtime._empirical_frame_sources == {}
    assert runtime._empirical_current_revisions == {}
    assert runtime._post_edit_refresh["paths"] == {path: "UNAVAILABLE"}


def test_optional_dependency_replans_when_source_disappears(tmp_path):
    source = tmp_path / "app.py"
    raw = b"VALUE = 1\n"
    source.write_bytes(raw)
    optional = {"dependency_revisions": {
        "app.py": "sha256:" + hashlib.sha256(raw).hexdigest()}}
    validate_optional_dependencies(optional, root=tmp_path)

    source.unlink()
    with pytest.raises(FrameAllocationReplanNeeded, match="optional dependency changed"):
        validate_optional_dependencies(optional, root=tmp_path)


def test_missing_applied_edit_head_does_not_block_next_request(tmp_path, monkeypatch):
    """A receipt may name a generated path absent from the current source view."""
    (tmp_path / "reports").mkdir()
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))

    panel = render_headcarry(
        tmp_path, (), [{"file": "reports/daily-summary.md"}], journal=journal,
    )

    assert panel["heads"] == []
    assert panel["omitted_count"] == 1
    assert events == []

    monkeypatch.setenv("STATEFRAME_HEADCARRY", "on")
    runtime = SimpleNamespace(journal=journal)
    frame = {}
    _update_headcarry(
        frame, runtime=runtime, root=tmp_path, open_sites=(),
        applied_edits=[{"file": "reports/daily-summary.md"}], messages=(),
    )
    assert frame["headcarry"]["omitted_count"] == 1
    assert runtime._framefit5_headcarry_failed is False
    assert runtime._framefit5_headcarry_unavailable is None
    assert events == []


def test_host_identity_uses_verified_launcher_tree_before_full_scan(tmp_path, monkeypatch):
    import kernel_exec_session as execution

    rows = {
        100: SimpleNamespace(pid=100, start=(1, 0), zombie=False, pgid=100, sid=100),
        200: SimpleNamespace(pid=200, start=(2, 0), zombie=False, pgid=100, sid=100),
    }

    class Table:
        def identity(self, pid):
            return rows.get(pid)

        def refresh_members(self, members):
            assert set(members) == {100}
            return rows

        def snapshot(self):
            pytest.fail("full host process scan should not be needed")

        def pid_namespace(self, pid):
            return "pid:[7]" if pid == 200 else "pid:[1]"

    proc = tmp_path / "proc"
    (proc / "200").mkdir(parents=True)
    (proc / "200" / "status").write_text("NSpid:\t200\t5\n")
    real_path = execution.Path
    monkeypatch.setattr(execution, "Path", lambda value, *rest: real_path(
        str(proc) if value == "/proc" else value, *rest))
    monkeypatch.setattr(execution, "ProcessTable", Table)
    monkeypatch.setattr(execution, "process_cmdline_digest", lambda pid: "a" * 64)
    launcher = SimpleNamespace(pid=100)
    monkeypatch.setattr(execution, "recorded_launch_identity", lambda process:
                        SimpleNamespace(pid=100, start=(1, 0)) if process is launcher else None)

    result = execution._host_mutation_identity(
        {"pid": 5, "start": [2, 0], "ns": "pid:[7]", "cmdline_sha256": "a" * 64},
        launcher=launcher)
    assert result["pid"] == 200
    assert result["ns_pid"] == 5


def test_hook_arm_deadline_excludes_launch_and_readiness_delay():
    from kernel_mutate_exec import _hook_arm_deadline

    assert _hook_arm_deadline(2.5, launch_started=1.6, ready_at=7.6) == pytest.approx(8.5)


def test_hook_startup_failure_retires_executor(tmp_path, monkeypatch):
    import kernel_mutate_build
    import kernel_mutate_wire
    from kernel_mutate_exec import MutationContainmentError
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "journal"))
    runtime._ensure_executor = lambda: SimpleNamespace(root=tmp_path)
    def fail_start(_spec):
        raise MutationContainmentError("mutation hook startup exceeded bound")

    runtime._launch = fail_start
    retired = []
    runtime.retire_hook_executor = lambda: retired.append(True)
    monkeypatch.setattr(kernel_mutate_build, "stage_unit", lambda *_args:
                        SimpleNamespace(dir=tmp_path))
    monkeypatch.setattr(kernel_mutate_wire, "build_stdin_doc", lambda **_kwargs: b"{}")

    with pytest.raises(MutationContainmentError, match="startup exceeded bound"):
        runtime._hook_manager()._run({"name": "observer", "entry": "main.py",
                                      "source": "def handle(input, ctx): return {}\n"},
                                     "trigger", {})
    assert retired == [True]


def test_memory_join_report_cannot_block_supervisor_forever():
    from kernel_exec_supervisor import _read_memory_join_result

    reader, writer = os.pipe()
    try:
        with pytest.raises(RuntimeError, match="memory join result timed out"):
            _read_memory_join_result(reader, timeout=0.01)
    finally:
        os.close(reader)
        os.close(writer)


def test_cgroup_leaf_lock_contention_has_bounded_fallback(tmp_path, monkeypatch):
    import fcntl
    import sf_launch

    (tmp_path / "cgroup.procs").write_text("")

    def busy(_fd, operation):
        assert operation & fcntl.LOCK_NB
        raise BlockingIOError(errno.EWOULDBLOCK, "held by another supervisor")

    monkeypatch.setattr(fcntl, "flock", busy)
    with pytest.raises(TimeoutError, match="cgroup leaf setup lock timed out"):
        with sf_launch._cgroup_leaf_pattern_process_lock(tmp_path, timeout=0.01):
            pytest.fail("contended setup must not enter")


def test_cgroup_leaf_lock_timeout_keeps_memory_fallback_available(tmp_path, monkeypatch):
    import sf_launch

    @contextmanager
    def contended(_path):
        raise TimeoutError("cgroup leaf setup lock timed out")
        yield

    monkeypatch.setattr(sf_launch, "_current_cgroup", lambda: tmp_path)
    monkeypatch.setattr(sf_launch, "_cgroup_leaf_pattern_process_lock", contended)
    for name in ("_CGROUP_LEAF_PATTERN_INITIALIZED", "_CGROUP_LEAF_PATTERN_ORIGIN",
                 "_CGROUP_LEAF_PATTERN_PARENT", "_CGROUP_LEAF_PATTERN_APPLIED",
                 "_CGROUP_LEAF_PATTERN_STATUS", "_CGROUP_LEAF_PATTERN_CREATED_PARENT",
                 "_CGROUP_LEAF_PATTERN_CREATOR_PID"):
        monkeypatch.setattr(sf_launch, name, getattr(sf_launch, name))
    sf_launch._CGROUP_LEAF_PATTERN_INITIALIZED = False
    parent, applied = sf_launch._command_cgroup_parent()

    assert parent == tmp_path
    assert applied is False
    assert sf_launch._cgroup_leaf_pattern_status() == "failed:leaf_lock:TimeoutError"
