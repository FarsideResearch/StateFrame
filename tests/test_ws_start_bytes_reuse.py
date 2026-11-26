from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_host_done_facts as facts
from kernel_delivery_manifest import DeliveryManifestTracker
from kernel_stateframe_runtime import StateFrameRuntime


class Journal:
    def emit(self, *_args, **_kwargs):
        pass


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(offset=0)
    monkeypatch.setattr(facts, "time", SimpleNamespace(
        time_ns=lambda: time.time_ns() + clock.offset))
    return clock


@pytest.fixture
def file_opens(monkeypatch):
    opened = []
    original = facts._open_regular_nofollow

    def traced(root, path):
        opened.append((root, path))
        return original(root, path)

    monkeypatch.setattr(facts, "_open_regular_nofollow", traced)
    return opened


def _prepare(tracker, request_id="request"):
    return tracker.prepare_message([], [], request_id=request_id)


def _note(tracker, request_id="request"):
    tracker.note_prepared({"request_id": request_id}, [], [])
    return tracker.last_prepared["manifest"]


def test_runtime_shares_one_start_capture_and_reuses_worker_bytes(
        tmp_path, monkeypatch, clock, file_opens, isolated_pilot_context):
    monkeypatch.setenv("STATEFRAME_HOST_DONE_FACTS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVERY_CONTRACT", "on")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    import kernel_release_switch_registry

    monkeypatch.setattr(kernel_release_switch_registry, "enforce_stateframe_switch_registry",
                        lambda *_args, **_kwargs: {})
    root = tmp_path / "worker"
    root.mkdir()
    target = root / "app.py"
    target.write_text("VALUE = 1\n")
    (root / "large.dat").write_bytes(b"x" * (128 << 10))
    clock.offset = 2 * facts.RACY_NS
    owner = SimpleNamespace(root=root, task_id="start-reuse", task_row={}, f2p_ids=[],
                            authorize=lambda *_args, **_kwargs: None)
    runtime = StateFrameRuntime(
        owner, output=tmp_path / "journal", target=target, model="scripted",
        tools=[], temperature=0, source_resolver=lambda _path, _symbol: None,
    )
    runtime.binding = lambda: {"product_revision": "product"}
    runtime.required_public_selectors = lambda: []
    tracker = runtime.delivery_contract
    baseline = runtime._host_done_facts_baseline
    assert isinstance(baseline, facts.WorkspaceBaseline)
    assert tracker.baseline is runtime._delivery_contract_baseline is baseline
    assert file_opens.count((root, "app.py")) == file_opens.count((root, "large.dat")) == 1
    tracker.journal = Journal()
    expected = DeliveryManifestTracker(root, dict(baseline), tracker.journal, runtime)
    expected_message = _prepare(expected)
    expected_prepared = _note(expected)
    expected_final = expected.finalize({})
    expected_completion = facts.compute_done_facts(root, dict(baseline), current_revision="product")

    file_opens.clear()
    assert _prepare(tracker) == expected_message
    assert _note(tracker) == expected_prepared
    assert tracker.finalize({}) == expected_final
    assert runtime.compute_done_facts().to_dict() == expected_completion.to_dict()
    assert file_opens == []


def test_snapshot_parity_after_same_mtime_edit_and_membership_changes(
        tmp_path, clock, file_opens):
    root = tmp_path / "worker"
    root.mkdir()
    target = root / "edited.py"
    target.write_bytes(b"before\r\n")
    (root / "deleted.py").write_bytes(b"removed\n")
    (root / "stable.dat").write_bytes(b"\x00\xff")
    (root / "was_file.py").write_bytes(b"replacement\n")
    (root / "was_link.py").symlink_to("stable.dat")
    (root / "directory").mkdir()
    (root / "directory" / "nested.py").write_text("original")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("ignored")
    (root / ".pytest_cache").write_text("also ignored")
    clock.offset = 2 * facts.RACY_NS
    baseline = facts.WorkspaceBaseline.capture(root)
    starting = dict(baseline)

    before = target.stat()
    # Coarse filesystem timestamps can give consecutive writes the same ctime.
    deadline = time.monotonic() + 2
    while target.stat().st_ctime_ns == before.st_ctime_ns and time.monotonic() < deadline:
        target.write_bytes(b"edited\r\n")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = target.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode) == (
        before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode)
    assert after.st_ctime_ns != before.st_ctime_ns
    (root / "deleted.py").unlink()
    (root / "made.py").write_bytes(b"new\n")
    (root / "was_file.py").unlink()
    (root / "was_file.py").symlink_to("stable.dat")
    (root / "was_link.py").unlink()
    (root / "was_link.py").write_bytes(b"now regular\n")
    outside = tmp_path / "outside"
    (root / "directory").rename(outside)
    (root / "directory").symlink_to(outside, target_is_directory=True)
    (root / ".git" / "config").write_text("still ignored")
    (root / ".pytest_cache").write_text("still ignored")

    eager = facts.workspace_snapshot(root, live=True)
    file_opens.clear()
    current = facts.current_workspace_snapshot(root, baseline)
    assert dict(current) == dict(eager)
    deltas = facts.diff_snapshots(baseline, current)
    assert deltas == facts.diff_snapshots(starting, eager)
    assert {delta.path: delta.status for delta in deltas} == {
        "deleted.py": "deleted", "directory/nested.py": "deleted",
        "edited.py": "modified", "made.py": "added", "was_file.py": "deleted",
        "was_link.py": "added",
    }
    assert file_opens == [(root, name) for name in ("edited.py", "made.py", "was_link.py")]


def test_recent_capture_requires_fresh_proof_even_after_racy_window(tmp_path, clock, file_opens):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    baseline = facts.WorkspaceBaseline.capture(tmp_path)
    file_opens.clear()
    for _ in range(2):
        assert facts.current_workspace_snapshot(tmp_path, baseline)["app.py"] == b"VALUE = 1\n"
    assert file_opens == [(tmp_path, "app.py")] * 2

    clock.offset = 2 * facts.RACY_NS
    facts.current_workspace_snapshot(tmp_path, baseline)
    assert file_opens == [(tmp_path, "app.py")] * 3
    assert facts.current_workspace_snapshot(tmp_path, baseline)["app.py"] == b"VALUE = 1\n"
    assert file_opens == [(tmp_path, "app.py")] * 3


def test_delivery_handoff_reuses_a_recent_scan_for_the_same_request(tmp_path, file_opens):
    (tmp_path / "app.py").write_text("start")
    baseline = facts.WorkspaceBaseline.capture(tmp_path)
    tracker = DeliveryManifestTracker(tmp_path, baseline, Journal())
    file_opens.clear()
    _prepare(tracker)
    assert _note(tracker)["files_changed"] == []
    assert file_opens == [(tmp_path, "app.py")]


def test_fresh_read_rejects_replacement_after_open(tmp_path, monkeypatch):
    target = tmp_path / "app.py"
    target.write_bytes(b"old")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new")
    original = facts._open_regular_nofollow

    def replace_after_open(root, relative):
        fd = original(root, relative)
        replacement.replace(target)
        return fd

    monkeypatch.setattr(facts, "_open_regular_nofollow", replace_after_open)
    with pytest.raises(facts.SnapshotChanged, match="file identity changed during read"):
        facts._read_regular(tmp_path, "app.py")


def test_atomic_replacement_during_snapshot_is_not_reported_as_deleted(tmp_path, monkeypatch):
    root = tmp_path / "worker"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"before")
    baseline = facts.WorkspaceBaseline.capture(root)
    target.write_bytes(b"intermediate")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"after")
    original = facts._open_regular_nofollow

    def replace_after_first_open(root, relative):
        fd = original(root, relative)
        if replacement.exists():
            replacement.replace(target)
        return fd

    monkeypatch.setattr(facts, "_open_regular_nofollow", replace_after_first_open)
    current = facts.current_workspace_snapshot(root, baseline)
    assert [(delta.path, delta.status) for delta in facts.diff_snapshots(baseline, current)] == [
        ("app.py", "modified"),
    ]
    assert current["app.py"] == b"after"


@pytest.mark.parametrize("boundary", ["mutation", "different_request", "different_root", "replaced_root"])
def test_prepare_handoff_expires_at_request_root_and_mutation_boundaries(
        tmp_path, clock, file_opens, boundary):
    root = tmp_path / "worker"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"before\n")
    clock.offset = 2 * facts.RACY_NS
    baseline = facts.WorkspaceBaseline.capture(root)
    tracker = DeliveryManifestTracker(root, baseline, Journal())
    _prepare(tracker)
    file_opens.clear()
    request_id = "request"
    if boundary == "mutation":
        StateFrameRuntime.note_mutation(SimpleNamespace(delivery_contract=tracker), ["app.py"])
    elif boundary == "different_request":
        request_id = "next"
    elif boundary == "different_root":
        # Even matching file signatures (a hard link) cannot cross roots.
        tracker.root = tmp_path / "other"
        tracker.root.mkdir()
        os.link(target, tracker.root / "app.py")
    else:
        root.rename(tmp_path / "old")
        root.mkdir()
    (tracker.root / "app.py").write_bytes(b"after\n")
    assert _note(tracker, request_id)["files_changed"] == ["app.py"]
    assert tracker.finalize({})["files_changed"] == ["app.py"]
    assert facts.compute_done_facts(tracker.root, baseline).changed_files == ("app.py",)
    assert file_opens == [(tracker.root, "app.py")] * 3


def test_mutation_during_prepare_prevents_a_stale_handoff(tmp_path, clock, monkeypatch):
    target = tmp_path / "app.py"
    target.write_text("start")
    clock.offset = 2 * facts.RACY_NS
    tracker = DeliveryManifestTracker(tmp_path, facts.WorkspaceBaseline.capture(tmp_path), Journal())
    capture = tracker._capture_workspace_snapshot

    def capture_then_mutate():
        snapshot = capture()
        target.write_text("edited")
        tracker.invalidate_workspace_snapshot()
        return snapshot

    monkeypatch.setattr(tracker, "_capture_workspace_snapshot", capture_then_mutate)
    _prepare(tracker)
    assert tracker._pending_workspace_snapshot is None
    monkeypatch.setattr(tracker, "_capture_workspace_snapshot", capture)
    assert _note(tracker)["files_changed"] == ["app.py"]


def test_completion_falls_back_to_shared_reuse_when_pin_is_unavailable(
        tmp_path, clock, file_opens, monkeypatch):
    from kernel_workspace_state import WorkspaceUnavailable

    (tmp_path / "app.py").write_text("VALUE = 1\n")
    clock.offset = 2 * facts.RACY_NS
    baseline = facts.WorkspaceBaseline.capture(tmp_path)
    expected = facts.compute_done_facts(tmp_path, dict(baseline))
    file_opens.clear()
    pin_calls = []

    def pin(root, before, **_kwargs):
        pin_calls.append((root, before))
        raise WorkspaceUnavailable("pin failed")

    monkeypatch.setattr(facts, "_pinned_done_workspace", pin)
    assert facts.compute_done_facts(tmp_path, baseline, pinned_baseline=True) == expected
    assert pin_calls == [(tmp_path, baseline)]
    assert file_opens == []


def test_start_cache_budget_and_rebind_preserve_cold_bytes(tmp_path, clock, file_opens, monkeypatch):
    worker, parent = tmp_path / "worker", tmp_path / "parent"
    worker.mkdir()
    parent.mkdir()
    files = {
        "a.txt": b"abcd", "b.txt": b"efgh", "large.txt": b"y" * (128 << 10),
        "overlay.txt": b"x" * (128 << 10), "tiny-overlay.txt": b"ab",
    }
    for name, content in files.items():
        (worker / name).write_bytes(content)
        if "overlay" not in name:
            (parent / name).write_bytes(content)
    monkeypatch.setattr(facts.SnapshotFiles, "BODY_CACHE_LIMIT", 4)
    clock.offset = 2 * facts.RACY_NS
    baseline = facts.WorkspaceBaseline.capture(worker, start_snapshot=parent)
    assert sum(map(len, baseline._bodies.values())) == 4
    assert set(baseline._spooled_paths) == {"overlay.txt", "tiny-overlay.txt"}

    file_opens.clear()
    assert dict(facts.current_workspace_snapshot(worker, baseline)) == files
    assert file_opens == [(parent, "b.txt"), (parent, "large.txt")]
    assert sum(map(len, baseline._bodies.values())) == 4
    tracker = DeliveryManifestTracker(worker, baseline, Journal())
    starting_message = _prepare(tracker)
    blob = pickle.dumps((baseline, tracker))
    baseline, tracker = pickle.loads(blob)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=worker),
                              _host_done_facts_baseline=baseline, delivery_contract=tracker)
    facts.rebind_workspace_snapshots(runtime, parent)
    assert tracker.baseline is baseline
    assert tracker._pending_workspace_snapshot is None
    file_opens.clear()
    assert _prepare(tracker) == starting_message
    assert file_opens == [(worker, name) for name in files]
    file_opens.clear()
    assert _prepare(tracker) == starting_message
    assert file_opens == []

    (worker / "overlay.txt").unlink()
    (worker / "large.txt").write_bytes(b"changed")
    assert dict(baseline) == files
    assert facts.diff_snapshots(baseline, facts.current_workspace_snapshot(worker, baseline)) == (
        facts.diff_snapshots(files, facts.workspace_snapshot(worker, live=True)))
    assert sum(map(len, baseline._bodies.values())) == 4


def test_unavailable_cold_copy_falls_back_only_for_current_bytes(tmp_path, clock, file_opens):
    worker, parent = tmp_path / "worker", tmp_path / "parent"
    worker.mkdir()
    parent.mkdir()
    content = b"start" * (32 << 10)
    (worker / "app.py").write_bytes(content)
    (parent / "app.py").write_bytes(content)
    clock.offset = 2 * facts.RACY_NS
    baseline = facts.WorkspaceBaseline.capture(worker, start_snapshot=parent)
    (parent / "app.py").write_bytes(b"corrupted")

    file_opens.clear()
    current = facts.current_workspace_snapshot(worker, baseline)
    assert file_opens == []
    assert current["app.py"] == content
    assert file_opens == [(parent, "app.py"), (worker, "app.py")]
    with pytest.raises(facts.SnapshotChanged):
        baseline["app.py"]


@pytest.mark.parametrize("offset", [0, (1 << 20) - 20], ids=["within-block", "across-blocks"])
def test_checkpoint_rejects_registered_secret_in_spooled_baseline(tmp_path, monkeypatch, offset):
    import kernel_checkpoint as checkpoint

    original = tmp_path / "overlay.txt"
    secret = b"workspace-overlay-credential-for-test"
    original.write_bytes(b"x" * offset + secret + b"x" * (128 << 10))
    baseline = facts.WorkspaceBaseline.capture(tmp_path)
    assert set(baseline._spooled_paths) == {"overlay.txt"}
    if offset:
        # The spool and pickle headers place the first four bytes before the boundary.
        spool = baseline._spool
        assert spool._read_exact((1 << 20) - 4, 4) == secret[:4]
        assert spool._read_exact(1 << 20, len(secret) - 4) == secret[4:]
    original.unlink()

    root = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoint, "_SAVE", {"root": root})
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    checkpoint.register_secret(secret)
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    writer = checkpoint._StateWriter(io.BytesIO())
    with pytest.raises(checkpoint.CheckpointError, match="a registered secret would be written"):
        checkpoint.dump_to(writer, baseline, context)
        writer.finish()
    assert not list((root / checkpoint.CACHE_DIR).glob(f"*{checkpoint.CHUNK_SUFFIX}"))


def test_engine_identity_seals_shipped_files():
    engine = Path(facts.__file__).parent
    source = json.loads((engine / "ENGINE_SOURCE.json").read_bytes())
    assert source["file_count"] == len(source["files"])
    assert source["files"] == {
        name: hashlib.sha256((engine / name).read_bytes()).hexdigest() for name in source["files"]}
    recorded = "".join(f"{name}\0{digest}\n" for name, digest in sorted(source["files"].items()))
    assert source["shipped_hash"] == hashlib.sha256(recorded.encode()).hexdigest()
