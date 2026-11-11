"""Regression coverage for mutation receipts against the current engine."""

import errno
import hashlib
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from kernel_frame_optional_allocation import (
    FrameAllocationReplanNeeded, validate_optional_dependencies,
)
from kernel_post_edit_refresh import refresh_after_applied_edit
from kernel_stateframe_framefit5 import _update_headcarry
from kernel_stateframe_headcarry import render_headcarry


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
