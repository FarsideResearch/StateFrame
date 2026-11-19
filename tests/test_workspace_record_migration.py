"""Storage and work budgets for proposal and checkpoint boundaries."""
from __future__ import annotations

import ctypes
import io
import json
import os
import pickle
import random
import stat
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import kernel_bridge_evidence as evidence
import pytest

from stateframe import _child, runner
from stateframe.engine import kernel_checkpoint as checkpoint


@pytest.fixture
def run(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_KEEP", "4")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"workspace-record-test-key-0123456789"])
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    return run


def _bind(run, work, monkeypatch):
    # Base still executes the historical call sites. A stitched tree supplies the
    # same fenced fixture binding the product host supplies to its consumers.
    if not hasattr(checkpoint, "_workspace_checkpoint_codec"):
        return None
    import kernel_workspace_index as index
    from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
    from kernel_workspace_refs import bind_run_store
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    bind_run_store(run, "a" * 64)
    monkeypatch.setattr(index, "time_ns", lambda: time.time_ns() + 5_000_000_000)
    consumers = WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(work, run.parent / f"fence-{work.name}"), trust_stat=True))
    bind_consumers(work, consumers)
    return consumers


def test_sparse_proposal_does_not_copy_or_read_the_workspace(run, monkeypatch):
    parent = run / "parent"
    work = run / "workspace"
    candidates = run / "worker" / "candidates"
    parent.mkdir()
    work.mkdir()
    paths = [f"item-{n:05d}.bin" for n in range(5_000)]
    generator = random.Random(1953)
    for name in paths:
        data = generator.randbytes(1024)
        (parent / name).write_bytes(data)
        (work / name).write_bytes(data)
    consumers = _bind(run, work, monkeypatch)
    row = {"path": paths[0], "files": dict.fromkeys(paths)}
    monkeypatch.setattr(evidence, "capture_candidate", evidence.capture_candidate)
    _child.install_compact_capture(parent)
    evidence.capture_candidate(str(candidates), work, row, 1)

    reads = 0
    read_bytes = Path.read_bytes

    def count(path):
        nonlocal reads
        if path.is_relative_to(work) or path.is_relative_to(parent):
            reads += 1
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count)
    binary = b"\x00\xffchanged\x80"
    edit = consumers.state.begin_write(paths[1:3]) if consumers is not None else nullcontext()
    with edit:
        (work / paths[1]).write_bytes(binary)
        (work / paths[2]).unlink()
    started = time.perf_counter()
    evidence.capture_candidate(str(candidates), work, row, 2)
    elapsed = time.perf_counter() - started
    boundary = candidates / "0002.json"
    if consumers is not None:
        reads += consumers.state.work.content_reads
    print(f"proposal: {elapsed:.6f}s, {reads} workspace reads, {boundary.stat().st_size} record bytes")
    assert reads <= 2
    assert boundary.stat().st_size <= 1024
    assert elapsed < 2
    record = json.loads(boundary.read_bytes())
    assert record["parent_tree"] != record["new_tree"]
    assert set(record) == {"schema", "proposals", "namespace", "parent_tree", "new_tree", "projection"}
    exported = runner.final_candidate(run)
    assert exported is not None
    assert exported[paths[0]] == read_bytes(parent / paths[0])
    assert exported[paths[1]] == binary and exported[paths[2]] is None


def test_proposal_capture_makes_private_boundaries_with_permissive_umask(run, monkeypatch):
    parent, work = run / "parent", run / "workspace"
    parent.mkdir()
    work.mkdir()
    (parent / "sample.py").write_text("value = 1\n")
    (work / "sample.py").write_text("value = 1\n")
    consumers = _bind(run, work, monkeypatch)
    assert consumers is not None
    monkeypatch.setattr(evidence, "capture_candidate", evidence.capture_candidate)
    _child.install_compact_capture(parent)
    previous = os.umask(0o002)
    try:
        evidence.capture_candidate(str(run / "worker" / "candidates"), work,
                                   {"path": "sample.py", "files": {"sample.py": None}}, 1)
    finally:
        os.umask(previous)
    for path in (run / "worker", run / "worker" / "candidates"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert runner.final_candidate(run)["sample.py"] == b"value = 1\n"


def test_object_checkpoint_reuses_current_child_tool_bindings(run, monkeypatch):
    import kernel_agent_tools
    import kernel_shell_tool

    root = run / "workspace"
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    shell = kernel_shell_tool.ShellTool.__new__(kernel_shell_tool.ShellTool)
    shell.root = root
    shell.native_pointer = ctypes.c_void_p(1)
    agent_tools = kernel_agent_tools.AgentTools.__new__(kernel_agent_tools.AgentTools)
    agent_tools.shell = shell
    agent_tools.native_pointer = ctypes.c_void_p(2)
    monkeypatch.setattr(kernel_shell_tool, "_active", shell)
    monkeypatch.setattr(kernel_agent_tools, "_active", agent_tools)
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    stream = io.BytesIO()
    checkpoint._Pickler(stream, context).dump((shell, agent_tools))
    stream.seek(0)
    restored = checkpoint._Unpickler(stream, context).load()
    assert restored == (shell, agent_tools)
    assert "_active" not in checkpoint._module_state("kernel_shell_tool", kernel_shell_tool)
    assert "_active" not in checkpoint._module_state("kernel_agent_tools", kernel_agent_tools)
    stale_shell = kernel_shell_tool.ShellTool.__new__(kernel_shell_tool.ShellTool)
    stale_shell.root = root
    with pytest.raises(pickle.PicklingError, match="shell checkpoint has no current host binding"):
        checkpoint._Pickler(io.BytesIO(), context).dump(stale_shell)
    shell.root = run / "other"
    stream.seek(0)
    with pytest.raises(checkpoint.CheckpointError, match="shell workspace binding is unavailable"):
        checkpoint._Unpickler(stream, context).load()


def test_object_checkpoint_reuses_fenced_workspace_binding(run, monkeypatch):
    import weakref
    import kernel_workspace_consumers as bindings

    root = run / "workspace"
    root.mkdir()
    consumers = _bind(run, root, monkeypatch)
    assert consumers is not None
    consumers.state._secrets = (b"registered-workspace-secret",)
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    stream = io.BytesIO()
    checkpoint._Pickler(stream, context).dump((consumers, consumers.state))
    assert b"registered-workspace-secret" not in stream.getvalue()
    stream.seek(0)
    assert checkpoint._Unpickler(stream, context).load() == (consumers, consumers.state)
    monkeypatch.setattr(bindings, "_bindings", weakref.WeakValueDictionary())
    stream.seek(0)
    with pytest.raises(checkpoint.CheckpointError, match="restored workspace binding is unavailable"):
        checkpoint._Unpickler(stream, context).load()


def test_object_checkpoint_drops_prior_step_candidate_binding(monkeypatch):
    def capture():
        capture_candidate = object()
        return sys._getframe(), capture_candidate

    frame, previous = capture()
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    assert "capture_candidate" not in checkpoint._frame_state(frame)
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    assert checkpoint._frame_state(frame)["capture_candidate"] is previous


def test_proposal_coverage_does_not_rescan_after_checkpoint_coverage(run, monkeypatch):
    import kernel_workspace_index as index

    parent, work = run / "parent", run / "workspace"
    parent.mkdir()
    work.mkdir()
    for path in (parent / "sample.py", work / "sample.py"):
        path.write_text("value = 1\n")
    consumers = _bind(run, work, monkeypatch)
    assert consumers is not None
    monkeypatch.setattr(evidence, "capture_candidate", evidence.capture_candidate)
    _child.install_compact_capture(parent)
    row = {"path": "sample.py", "files": {"sample.py": None}}
    destination = str(run / "worker" / "candidates")
    evidence.capture_candidate(destination, work, row, 1)
    with consumers.state.pin("checkpoint-workspace"):
        pass
    scans = []
    original = index.WorkspaceIndex._scan

    def count(self, *args, **kwargs):
        scans.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(index.WorkspaceIndex, "_scan", count)
    evidence.capture_candidate(destination, work, row, 2)
    assert scans == []


def test_second_checkpoint_emits_no_file_payloads_in_state_stream(run, monkeypatch):
    work = run / "workspace"
    work.mkdir()
    target = work / "data.bin"
    target.write_bytes(b"before")
    consumers = _bind(run, work, monkeypatch)
    payload_bytes = random.Random(12345).randbytes(1 << 20) * 144
    # A single reachable object is aliased through two owners and the list cycles back to itself.
    # It must round-trip as one instance while its immutable bytes are only published once.
    shared = [payload_bytes]
    shared.append(shared)
    state = {"frame": {"first": shared, "second": shared}, "modules": {}, "classes": {},
             "patched": {}, "participants": {}, "context": [], "random": random.getstate()}
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: (state, context))
    first = checkpoint.save(sys._getframe(), 1, run)
    edit = consumers.state.begin_write(("data.bin",)) if consumers is not None else nullcontext()
    with edit:
        target.write_bytes(b"after")

    emitted = 0
    write = checkpoint._StateWriter.write

    def count(writer, data):
        nonlocal emitted
        emitted += len(data)
        return write(writer, data)

    monkeypatch.setattr(checkpoint._StateWriter, "write", count)
    started = time.perf_counter()
    second = checkpoint.save(sys._getframe(), 2, run)
    elapsed = time.perf_counter() - started
    print(f"checkpoint: {elapsed:.6f}s, {emitted} serialized bytes, "
          f"{(second / 'checkpoint.json').stat().st_size} record bytes")
    assert emitted <= 64 << 10
    assert elapsed < 2
    for saved in (first, second):
        assert {entry.name for entry in saved.iterdir()} == {"checkpoint.json"}
        assert (saved / "checkpoint.json").stat().st_size <= 1024
    assert checkpoint.validate(second)["step"] == 2

    restored = {}
    monkeypatch.setattr(checkpoint, "_apply_restored_state",
                        lambda _frame, payload: restored.update(payload))
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _clocks: None)
    target.write_bytes(b"uncommitted setup")
    checkpoint.restore(sys._getframe(), second, run)
    assert target.read_bytes() == b"after"
    assert restored["frame"]["first"] is restored["frame"]["second"]
    assert restored["frame"]["first"][1] is restored["frame"]["first"]
    assert restored["frame"]["first"][0] == payload_bytes


def test_workspace_seal_round_trips_legacy_digest_and_explicit_writes(run, monkeypatch):
    from kernel_done_is_submit_authority import artifact_digest, store_candidate

    old = {"same.py": "unchanged\n", "gone.py": "old\n"}
    proposed = {"same.py": "unchanged\n", "binary.bin": b"\x00\xff", "gone.py": None}
    root = run / "worker" / "results" / "react_done_submissions"
    digest = artifact_digest(proposed)
    expected = {"same.py": b"unchanged\n", "binary.bin": b"\x00\xff", "gone.py": None}
    monkeypatch.setattr(runner.outcomes, "terminal_receipt",
                        lambda _run: {"submission": {"stored": True, "artifact_digest": digest,
                                                     "seal_id": digest}})
    if hasattr(checkpoint, "_workspace_checkpoint_codec"):
        from kernel_workspace_refs import bind_run_store

        bind_run_store(run, "a" * 64)
        assert store_candidate(root, "task", proposed, parent_sources=old)["artifact_digest"] == digest
        artifact = next(root.glob("*/*.json"))
        document = json.loads(artifact.read_bytes())
        assert document["schema"] == "stateframe-candidate-trees/1"
        assert b"unchanged" not in artifact.read_bytes()
        assert runner.final_candidate(run) == expected
        document["new_tree"] = document["parent_tree"]
        artifact.write_text(json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        with pytest.raises(runner.RunError, match="sealed candidate workspace record"):
            runner.final_candidate(run)
    else:
        assert store_candidate(root, "task", proposed)["artifact_digest"] == digest
        assert runner.final_candidate(run) == expected


def test_interrupted_checkpoint_activation_restores_previous_setup(run):
    if not hasattr(checkpoint, "_workspace_checkpoint_codec"):
        pytest.skip("workspace activation records are introduced by this change")
    codec = checkpoint._workspace_checkpoint_codec()
    root = run / "checkpoints"
    incoming = root / ".workspace-restore-test" / "run"
    prior = root / "setup-discard.test"
    root.mkdir(mode=0o700)
    incoming.parent.mkdir(mode=0o700)
    incoming.mkdir(mode=0o700)
    prior.mkdir(mode=0o700)
    previous = run / "workspace"
    previous.write_bytes(b"fresh setup")
    (incoming / "workspace").write_bytes(b"captured")
    with pytest.raises(RuntimeError, match="interrupted"), codec._activate(run, incoming, prior):
        assert previous.read_bytes() == b"captured"
        raise RuntimeError("interrupted")
    assert previous.read_bytes() == b"fresh setup"
    assert (incoming / "workspace").read_bytes() == b"captured"
    assert not (root / codec.ACTIVATION).exists()
