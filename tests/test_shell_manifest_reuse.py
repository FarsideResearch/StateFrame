"""Shell post-manifest reuse preserves effects and exact wire records."""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENGINE = ROOT / "src" / "stateframe" / "engine"
for import_root in (SRC, ENGINE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import kernel_effects
import kernel_shell_tool
from kernel_exec_session import CleanupReport, ExecResult
from kernel_measurement import _serialize_final_request_body
from kernel_shell_tool import ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import ToolReplayRefused, configure_replay, tape_for_run


class ProcessBoundary:
    def __init__(self, actions=None, *, on_run=None):
        self.actions = actions or {}
        self.on_run = on_run
        self.calls = []

    def configure(self, capability):
        self.capability = capability

    def run(self, argv, capability, *, cwd):
        command = argv[-1]
        self.calls.append(command)
        if self.on_run is not None:
            self.on_run(cwd)
        action = self.actions.get(command)
        if action is not None:
            action(cwd)
        return ExecResult(0, False, b"ok\n", False, 3, 0.01,
                          CleanupReport(True, 0, (), "test cleanup"))


def _make_tool(base: Path, *, files=None, actions=None, on_run=None, reuse=True):
    base.mkdir(parents=True, exist_ok=True)
    root = base / "workspace"
    root.mkdir(parents=True)
    initial = {"app.py": "VALUE = 1\n", **(files or {})}
    for relative, contents in initial.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    run = base / "run"
    run.mkdir()
    row = {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": True,
           "allowed_product_paths": ["app.py"], "new_file_roots": []}
    process = ProcessBoundary(actions, on_run=on_run)
    tool = ShellTool(run, ToolSettings(run_env=False), {}, process)
    tool.bind(root, row, set())
    # This private seam lets parity tests exercise both paths without adding a
    # user-facing switch for an implementation detail.
    tool._post_manifest_reuse_enabled = reuse
    configure_replay(run, active=False)
    return tool, process, root, run


def test_second_noop_call_reuses_settled_manifest_and_known_objects(tmp_path, monkeypatch):
    tool, process, _, _ = _make_tool(tmp_path)
    scan_count = 0
    capture_count = 0
    scan_at_dispatch = []
    capture_at_dispatch = []
    real_effects_scan = kernel_effects.scan
    real_capture = tool.store.capture

    def counted_scan(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return real_effects_scan(*args, **kwargs)

    def counted_capture(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(kernel_shell_tool, "scan", counted_scan)
    monkeypatch.setattr(kernel_effects, "scan", counted_scan)
    monkeypatch.setattr(tool.store, "capture", counted_capture)
    process.on_run = lambda _: (scan_at_dispatch.append(scan_count),
                                capture_at_dispatch.append(capture_count))

    tool.execute({"command": "echo first"}, "call-1")
    first_boundary = (scan_count, capture_count)
    tool.execute({"command": "echo second"}, "call-2")

    assert scan_at_dispatch[1] - first_boundary[0] == 0
    assert capture_at_dispatch[1] - first_boundary[1] == 0
    assert scan_count > scan_at_dispatch[1]


def test_bound_workspace_consumers_keep_the_post_tree_for_the_next_call(tmp_path, monkeypatch):
    from kernel_workspace_consumers import bind_run_consumers
    from kernel_workspace_refs import bind_run_store

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    root = run / "workspace"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n")
    bind_run_store(run, "d" * 64)
    consumers = bind_run_consumers(run, root, parent=root)
    state = consumers.state
    scan_count = 0
    scan_at_dispatch = []
    real_index_scan = state._index._scan

    def counted_index_scan(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return real_index_scan(*args, **kwargs)

    monkeypatch.setattr(state._index, "_scan", counted_index_scan)
    process = ProcessBoundary(on_run=lambda _: scan_at_dispatch.append(scan_count))
    tool = ShellTool(run, ToolSettings(run_env=False), {"parent_snapshot": str(root)}, process)
    row = {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": True,
           "allowed_product_paths": ["app.py"], "new_file_roots": []}
    tool.bind(root, row, set())
    configure_replay(run, active=False)
    try:
        tool.execute({"command": "echo first"}, "call-1")
        after_first = scan_count
        tool.execute({"command": "echo second"}, "call-2")
        assert scan_at_dispatch[1] - after_first == 0
    finally:
        if state._feed is not None:
            state._feed.close()
        consumers.private_directory.cleanup()


def test_reused_baseline_still_reports_an_allowed_file_edit(tmp_path):
    tool, _, root, _ = _make_tool(
        tmp_path, actions={"python edit.py": lambda cwd: (cwd / "app.py").write_text("VALUE = 2\n")},
    )
    tool.execute({"command": "echo first"}, "call-1")
    reply = tool.execute({"command": "python edit.py"}, "call-2")

    assert [(effect.path, effect.change) for effect in reply.applied] == [("app.py", "modified")]
    assert (root / "app.py").read_text() == "VALUE = 2\n"


def test_changed_post_manifest_is_captured_before_later_reuse(tmp_path):
    tool, _, root, _ = _make_tool(
        tmp_path, actions={"python edit.py": lambda cwd: (cwd / "app.py").write_text("VALUE = 2\n")},
    )
    tool.execute({"command": "echo first"}, "call-1")
    tool.execute({"command": "python edit.py"}, "call-2")
    changed = kernel_effects.scan(root, exclude=lambda _: False)["app.py"]

    tool.execute({"command": "echo third"}, "call-3")

    assert changed.digest is not None
    assert tool.store.get(changed.digest) == b"VALUE = 2\n"


def test_reused_baseline_catches_and_restores_an_out_of_scope_creation(tmp_path):
    tool, _, root, _ = _make_tool(
        tmp_path, actions={"python create.py": lambda cwd: (cwd / "unlisted.py").write_text("x = 1\n")},
    )
    tool.execute({"command": "echo first"}, "call-1")
    reply = tool.execute({"command": "python create.py"}, "call-2")

    assert "restored (not allowed): unlisted.py (outside edit scope)" in reply.observation
    assert not (root / "unlisted.py").exists()
    assert "unlisted.py (outside edit scope)" in reply.observation


def test_reused_baseline_catches_and_restores_an_out_of_scope_deletion(tmp_path):
    tool, _, root, _ = _make_tool(
        tmp_path, files={"outside.py": "VALUE = 9\n"},
        actions={"python delete.py": lambda cwd: (cwd / "outside.py").unlink()},
    )
    tool.execute({"command": "echo first"}, "call-1")
    reply = tool.execute({"command": "python delete.py"}, "call-2")

    assert reply.observation.startswith("REFUSED OUTSIDE_EDIT_SCOPE:")
    assert (root / "outside.py").read_text() == "VALUE = 9\n"
    assert "outside.py (outside edit scope)" in reply.observation


def test_replay_discards_reuse_after_an_external_workspace_change(tmp_path):
    tool, process, root, run = _make_tool(tmp_path)
    tool.execute({"command": "echo first"}, "call-1")
    (root / "app.py").write_text("VALUE = 2\n")
    configure_replay(run, active=True)

    with pytest.raises(ToolReplayRefused, match="pre-state"):
        tool.execute({"command": "echo first"}, "call-1")
    assert process.calls == ["echo first"]


def test_native_edit_scope_invalidates_the_legacy_shell_baseline(tmp_path, monkeypatch):
    tool, process, root, _ = _make_tool(tmp_path)
    tool.execute({"command": "echo first"}, "call-1")
    monkeypatch.setattr(kernel_shell_tool, "_active", tool)
    from probes.react_repair import _workspace_write_scope

    with _workspace_write_scope(root / "app.py"):
        (root / "app.py").write_text("VALUE = 2\n")

    scan_count = 0
    scan_at_dispatch = []
    real_scan = kernel_effects.scan

    def counted_scan(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(kernel_shell_tool, "scan", counted_scan)
    monkeypatch.setattr(kernel_effects, "scan", counted_scan)
    process.on_run = lambda _: scan_at_dispatch.append(scan_count)
    tool.execute({"command": "echo second"}, "call-2")

    assert scan_at_dispatch == [1]


def test_native_file_delete_invalidates_the_legacy_shell_baseline(tmp_path, monkeypatch):
    import kernel_file_ops

    tool, _, root, _ = _make_tool(tmp_path)
    tool.execute({"command": "echo first"}, "call-1")
    monkeypatch.setattr(kernel_shell_tool, "_active", tool)
    effect = kernel_file_ops.file_delete(
        root, "app.py", protected=lambda _: False, editable=lambda _: True,
    )

    assert effect.changed_paths == ("app.py",)
    assert tool._post_manifest is None


def test_native_file_move_invalidates_the_legacy_shell_baseline(tmp_path, monkeypatch):
    import kernel_file_ops

    tool, _, root, _ = _make_tool(tmp_path)
    tool.execute({"command": "echo first"}, "call-1")
    monkeypatch.setattr(kernel_shell_tool, "_active", tool)
    effect = kernel_file_ops.file_move(
        root, "app.py", "moved.py", protected=lambda _: False,
        editable=lambda _: True, new_file_allowed=lambda _: True,
    )

    assert effect.changed_paths == ("app.py", "moved.py")
    assert tool._post_manifest is None


def test_mutation_edit_invalidates_the_legacy_shell_baseline(tmp_path, monkeypatch):
    import os
    import kernel_mutate_edits

    tool, _, root, _ = _make_tool(tmp_path)
    tool.execute({"command": "echo first"}, "call-1")
    monkeypatch.setattr(kernel_shell_tool, "_active", tool)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    parent_fd = os.dup(root_fd)
    identity = kernel_mutate_edits._identity(parent_fd)
    try:
        with kernel_mutate_edits._bound_write_context(
                root_fd, ("app.py",), identity, b"VALUE = 1\n"):
            kernel_mutate_edits._write_at(parent_fd, "app.py", b"VALUE = 2\n", 0o644)
    finally:
        os.close(parent_fd)
        os.close(root_fd)

    assert tool._post_manifest is None


def test_unsandboxed_commands_do_not_keep_a_reusable_baseline(tmp_path):
    from dataclasses import replace

    tool, _, _, _ = _make_tool(tmp_path)
    tool.capability = replace(tool.capability, sandboxed=False)
    tool.execute({"command": "echo first"}, "call-1")

    assert tool._post_manifest is None


def _parity_run(base: Path, *, reuse: bool) -> tuple[bytes, bytes]:
    tool, _, _, run = _make_tool(base, reuse=reuse)
    replies = [tool.execute({"command": command}, call_id)
               for command, call_id in (("echo first", "call-1"), ("echo second", "call-2"))]
    journal = tape_for_run().path.read_bytes()
    messages = [{"role": "user", "content": "Run two no-op shell calls."}]
    for (command, call_id), reply in zip(
            (("echo first", "call-1"), ("echo second", "call-2")), replies):
        messages.extend((
            {"role": "assistant", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": command})},
            }]},
            {"role": "tool", "tool_call_id": call_id, "content": reply.observation},
        ))
    request = _serialize_final_request_body(
        {"model": "fixture", "messages": messages, "tools": [kernel_shell_tool.SHELL_SCHEMA]},
        None, "request-1",
    )
    assert tape_for_run().path.parent == run
    return journal, request


def test_manifest_reuse_preserves_tool_journal_and_request_bytes(tmp_path):
    enabled = _parity_run(tmp_path / "enabled", reuse=True)
    disabled = _parity_run(tmp_path / "disabled", reuse=False)
    assert enabled == disabled


def benchmark_second_noop_call(*, file_count: int = 3000, repetitions: int = 5) -> float:
    """Median second-call latency over a synthetic Python workspace."""
    with tempfile.TemporaryDirectory(prefix="shell-manifest-reuse-") as temporary:
        base = Path(temporary)
        workspace = base / "workspace"
        workspace.mkdir()
        body = "VALUE = 1\n"
        for index in range(file_count):
            (workspace / f"module_{index:04d}.py").write_text(body)

        measurements = []
        for iteration in range(repetitions):
            run = base / f"run-{iteration}"
            run.mkdir()
            process = ProcessBoundary()
            row = {"files": {}, "scoped_edits": False,
                   "allowed_product_paths": [], "new_file_roots": []}
            tool = ShellTool(run, ToolSettings(run_env=False), {}, process)
            tool.bind(workspace, row, set())
            tool._post_manifest_reuse_enabled = True
            configure_replay(run, active=False)
            tool.execute({"command": "echo warmup"}, "warmup")
            started = time.perf_counter()
            tool.execute({"command": "echo measured"}, "measured")
            measurements.append(time.perf_counter() - started)
        return statistics.median(measurements)


if __name__ == "__main__" and "--benchmark" in sys.argv:
    print(f"median second no-op shell call (3000 Python files, 5 runs): "
          f"{benchmark_second_noop_call() * 1000:.3f} ms")
