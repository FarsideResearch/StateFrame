"""Work counts for startup baselines and foreground shell projections."""

import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

import kernel_workspace_consumers as consumers
from kernel_workspace_refs import bind_run_store
from kernel_workspace_state import WorkspaceState, _PayloadSpool
from kernel_shell_tool import ShellExecutionFailed, ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import configure_replay, tape_for_run
from probes import react_repair
from test_shell_tool import ProcessBoundary

_TURN_SWITCHES = (
    "ATOMIC_PUBLICATION", "BOUNDED_QUALIFICATION", "DELIVERY_PROOFS",
    "GENERATION_OVERLAP", "INDEXED_READS", "MINIMAL_LAUNCH", "PANELS",
    "REPO_INTEL", "TIMING",
)


def _switches(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", value)
    for name in _TURN_SWITCHES:
        monkeypatch.setenv(f"STATEFRAME_TURN_{name}", value)


def _run(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    parent = run / "parent"
    parent.mkdir()
    return run, parent, run / "workspace"


@pytest.mark.skipif(sys.platform != "linux", reason="unchanged work uses the Linux workspace feed")
def test_navigation_and_shell_work(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    run, parent, root = _run(tmp_path, monkeypatch)
    paths = tuple(f"pkg{index // 100:02d}/source{index:04d}.py" for index in range(4500))
    for relative in paths:
        path = parent / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"VALUE = 1\r\n")
    shutil.copytree(parent, root)
    root.chmod(0o700)
    bind_run_store(run, "d" * 64)
    configure_replay(run, active=False)

    counts = Counter()
    original_observe = WorkspaceState._observe
    original_publish = _PayloadSpool.publish
    original_stored = consumers.effect_manifest
    original_live = consumers._live_manifest
    original_read_text = Path.read_text

    def observe(state, *args, **kwargs):
        tree = original_observe(state, *args, **kwargs)
        key = "parent" if state.fence.root == parent else "live"
        counts[key + "_sweeps"] += int(state._index.full)
        counts[key + "_reads"] += state.work.content_reads
        return tree

    def publish(spool, lease, digests):
        counts["payloads_published"] += len(digests)
        return original_publish(spool, lease, digests)

    def stored(*args, **kwargs):
        result = original_stored(*args, **kwargs)
        counts["stored_projections"] += 1
        counts["stored_entries_read"] += len(result)
        return result

    def live(*args, **kwargs):
        counts["live_projections"] += 1
        return original_live(*args, **kwargs)

    def read_text(path, *args, **kwargs):
        if path.is_relative_to(root) and path.suffix == ".py":
            counts["source_text_reads"] += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(WorkspaceState, "_observe", observe)
    monkeypatch.setattr(_PayloadSpool, "publish", publish)
    monkeypatch.setattr(consumers, "effect_manifest", stored)
    monkeypatch.setattr(consumers, "_live_manifest", live)
    monkeypatch.setattr(Path, "read_text", read_text)

    process = ProcessBoundary()
    tool = ShellTool(run, ToolSettings(shell="read-only", run_env=False),
                     {"parent_snapshot": str(parent)}, process)
    started = time.perf_counter()
    tool.bind(root, {"files": {}, "scoped_edits": False}, set())
    snapshot = getattr(react_repair, "_initial_source_snapshot", react_repair._source_snapshot)
    assert snapshot(root, paths) == dict.fromkeys(paths, "VALUE = 1\n")
    navigation_seconds = time.perf_counter() - started
    navigation = counts.copy()

    steps = []
    for call_id in ("first", "second"):
        counts.clear()
        started = time.perf_counter()
        reply = tool.execute({"command": "pwd"}, call_id)
        elapsed = time.perf_counter() - started
        assert not reply.applied and not reply.fatal
        steps.append(counts.copy())
        print(f"{call_id}: {elapsed:.3f}s {dict(counts)}")
    assert len(process.calls) == 2
    print(f"navigation: {navigation_seconds:.3f}s {dict(navigation)}")
    assert navigation["parent_reads"] == 0
    assert navigation["parent_sweeps"] == 0
    assert navigation["payloads_published"] == navigation["stored_entries_read"] == 0
    assert navigation["source_text_reads"] == 0
    assert navigation["live_reads"] <= len(paths)
    first, second = steps
    assert first["stored_projections"] + first["live_projections"] <= 1
    assert second["parent_reads"] == second["live_reads"] == 0
    assert second["parent_sweeps"] == second["live_sweeps"] == 0
    assert second["stored_projections"] == second["live_projections"] == 0
    assert navigation_seconds < 10


def _shell(run, parent, root, process, *, mode="standard"):
    tool = ShellTool(run, ToolSettings(shell=mode, run_env=False), {
        "parent_snapshot": str(parent),
    }, process)
    tool.bind(root, {"files": {}, "scoped_edits": False, "tier": "USER-REPO"},
              {str(root / "test_app.py")} if (root / "test_app.py").exists() else set())
    configure_replay(run, active=False)
    return tool


def test_shell_effects_match_with_objects_and_turn_switches(tmp_path, monkeypatch):
    def exercise(case, switches):
        case.mkdir()
        _switches(monkeypatch, switches)
        run, parent, root = _run(case, monkeypatch)
        (parent / "café.py").write_bytes("VALUE = 'é'\r\n".encode())
        (parent / "test_app.py").write_bytes(b"assert True\n")
        shutil.copytree(parent, root)
        if switches == "on":
            bind_run_store(run, "e" * 64)

        def change(cwd):
            (cwd / "café.py").rename(cwd / "thé.py")
            (cwd / "alias.py").symlink_to("thé.py")
            (cwd / "test_app.py").write_bytes(b"assert False\n")

        tool = _shell(run, parent, root, ProcessBoundary(change))
        reply = tool.execute({"command": "python change.py"}, "effect")
        assert {(row.path, row.change) for row in reply.applied} == {
            ("café.py", "deleted"), ("thé.py", "created"),
        }
        assert not (root / "alias.py").is_symlink()
        assert (root / "test_app.py").read_bytes() == b"assert True\n"
        assert (root / "thé.py").read_bytes() == "VALUE = 'é'\r\n".encode()
        record = tape_for_run().lookup("effect")
        return reply, record.pre_manifest_digest, record.post_manifest_digest

    assert exercise(tmp_path / "objects", "on") == exercise(tmp_path / "legacy", "off")


@pytest.mark.parametrize("change", ["replace-content", "remove-directory"])
def test_unretained_parent_changes_fail_closed(tmp_path, monkeypatch, change):
    _switches(monkeypatch, "on")
    run, parent, root = _run(tmp_path, monkeypatch)
    (parent / "test_app.py").write_bytes(b"assert True\n")
    shutil.copytree(parent, root)
    bind_run_store(run, "f" * 64)
    process = ProcessBoundary()
    tool = _shell(run, parent, root, process)
    if change == "replace-content":
        (parent / "test_app.py").write_bytes(b"assert None\n")
    else:
        parent.rename(run / "removed-parent")
    with pytest.raises(ShellExecutionFailed, match="parent|workspace observation"):
        tool.execute({"command": "python run.py"}, "effect")
    assert not process.calls


@pytest.mark.parametrize("change", ["replace-content", "remove-directory"])
def test_retained_parent_remains_rollback_authority(tmp_path, monkeypatch, change):
    _switches(monkeypatch, "on")
    run, parent, root = _run(tmp_path, monkeypatch)
    (parent / "test_app.py").write_bytes(b"assert True\n")
    shutil.copytree(parent, root)
    bind_run_store(run, "f" * 64)
    initial = _shell(run, parent, root, ProcessBoundary())
    initial.execute({"command": "python run.py"}, "establish")
    if change == "replace-content":
        (parent / "test_app.py").write_bytes(b"assert None\n")
    else:
        parent.rename(run / "removed-parent")

    restarted = run / "restarted"
    shutil.copytree(root, restarted)
    (restarted / "test_app.py").write_bytes(b"assert 2\n")

    def damage(cwd):
        (cwd / "test_app.py").write_bytes(b"assert False\n")

    tool = _shell(run, parent, restarted, ProcessBoundary(damage))
    reply = tool.execute({"command": "python run.py"}, "effect")
    assert not reply.applied
    assert (restarted / "test_app.py").read_bytes() == b"assert True\n"


@pytest.mark.skipif(sys.platform != "linux", reason="requires the Linux workspace feed")
def test_feed_overflow_rechecks_external_rename(tmp_path, monkeypatch):
    from kernel_effects import scan
    from kernel_workspace_feed import _CoverageLost
    from kernel_shell_tool import workspace_identity

    _switches(monkeypatch, "on")
    run, parent, root = _run(tmp_path, monkeypatch)
    (parent / "café.py").write_bytes(b"VALUE = 1\n")
    shutil.copytree(parent, root)
    bind_run_store(run, "f" * 64)
    tool = _shell(run, parent, root, ProcessBoundary(), mode="read-only")
    tool.execute({"command": "pwd"}, "first")
    (root / "café.py").rename(root / "changed.py")
    feed = tool.workspace_effects.state._feed
    original_drain = feed._drain

    def overflow():
        monkeypatch.setattr(feed, "_drain", original_drain)
        raise _CoverageLost("overflow")

    monkeypatch.setattr(feed, "_drain", overflow)
    tool.execute({"command": "pwd"}, "second")
    expected = workspace_identity(scan(root, exclude=lambda _: False))
    record = tape_for_run().lookup("second")
    assert record.pre_manifest_digest == record.post_manifest_digest == expected
