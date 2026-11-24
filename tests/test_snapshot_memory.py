from __future__ import annotations

import gc
import hashlib
import os
import json
import pickle
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_host_done_facts as host_done_facts
from kernel_coordinated_change_ledger import _compute_tree_revision
from kernel_delivery_manifest import DeliveryManifestTracker, _applied_edit_rows, build_manifest
from kernel_green_lock import GreenLock
from kernel_host_done_facts import (
    SnapshotChanged,
    SnapshotFiles,
    compute_done_facts,
    diff_snapshots,
    workspace_snapshot,
)
from kernel_tree_digest import compatible_revision


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _readable_files(root: Path) -> dict[str, bytes]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            result[path.relative_to(root).as_posix()] = path.read_bytes()
        except OSError:
            continue
    return {path: result[path] for path in sorted(result, key=Path)}


def _peak_bytes(operation):
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_snapshot_mapping_matches_plain_dict_for_file_kinds(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "binary.bin").write_bytes(b"\x00\xff\x01")
    (root / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    (root / "raw.dat").write_bytes(b"\xff\xfe")
    (root / "empty").write_bytes(b"")
    executable = root / "run.sh"
    executable.write_bytes(b"#!/bin/sh\r\necho ok\r\n")
    executable.chmod(0o755)
    (root / "link.txt").symlink_to("crlf.txt")
    unreadable = root / "unreadable.txt"
    unreadable.write_bytes(b"private")
    unreadable.chmod(0)
    try:
        with unreadable.open("rb"):
            unreadable_readable = True
    except OSError:
        unreadable_readable = False

    captured = workspace_snapshot(root)
    expected = _readable_files(root)
    assert isinstance(captured, SnapshotFiles)
    assert dict(captured.items()) == expected
    assert captured == expected
    assert list(captured) == list(expected)
    assert len(captured) == len(expected)
    assert "link.txt" not in captured
    if not unreadable_readable:
        assert "unreadable.txt" not in captured
    assert compute_done_facts(root, captured).to_dict() == compute_done_facts(
        root, expected).to_dict()

    before = dict(captured.items())
    (root / "crlf.txt").write_bytes(b"one\r\ntwo changed\r\n")
    after = workspace_snapshot(root, live=True)
    assert diff_snapshots(captured, after) == diff_snapshots(before, _readable_files(root))

    paths = ("binary.bin", "crlf.txt", "empty", "link.txt", "raw.dat", "run.sh",
             "unreadable.txt")
    rows = []
    for relative in paths:
        path = root / relative
        if path.is_symlink():
            rows.append((relative, os.fsencode(os.readlink(path)), "symlink"))
        else:
            try:
                rows.append((relative, path.read_bytes(), "file"))
            except OSError:
                rows.append((relative, None, "unreadable"))
    assert _compute_tree_revision(root, paths) == "sha256:" + compatible_revision(
        "stateframe.coordinated_tree_revision/2", rows)


def test_capture_diff_and_tree_revision_stay_below_memory_bound(tmp_path: Path):
    parent = tmp_path / "parent"
    worker = tmp_path / "worker"
    parent.mkdir()
    worker.mkdir()
    payload = b"A" * (1 << 20)
    paths = tuple(f"pkg/file_{index:03d}.py" for index in range(60))
    for relative in paths:
        for root in (parent, worker):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    baseline, peak = _peak_bytes(
        lambda: workspace_snapshot(worker, start_snapshot=parent))
    assert peak < 25 * 1024 * 1024
    assert len(baseline) == len(paths)
    assert all(baseline.digest_for_path(path) == _sha(payload) for path in paths)
    checkpoint_blob = pickle.dumps(baseline, protocol=5)
    assert len(checkpoint_blob) < 100_000

    (worker / paths[0]).write_bytes(b"B" * (1 << 20))
    after, capture_peak = _peak_bytes(lambda: workspace_snapshot(worker, live=True))
    assert capture_peak < 25 * 1024 * 1024
    deltas, diff_peak = _peak_bytes(lambda: diff_snapshots(baseline, after))
    assert diff_peak < 25 * 1024 * 1024
    assert [delta.path for delta in deltas] == [paths[0]]

    revision, revision_peak = _peak_bytes(lambda: _compute_tree_revision(worker, paths))
    assert revision_peak < 25 * 1024 * 1024
    expected_rows = ((path, (worker / path).read_bytes(), "file") for path in paths)
    assert revision == "sha256:" + compatible_revision(
        "stateframe.coordinated_tree_revision/2", expected_rows)


def test_snapshot_pickle_keeps_rows_and_small_bodies_only(tmp_path: Path):
    parent = tmp_path / "parent"
    worker = tmp_path / "worker"
    parent.mkdir()
    worker.mkdir()
    large = b"x" * (1 << 20)
    for root in (parent, worker):
        (root / "large.py").write_bytes(large)
        (root / "small.py").write_bytes(b"VALUE = 1\n")

    snapshot = workspace_snapshot(worker, start_snapshot=parent)
    blob = pickle.dumps(snapshot, protocol=5)
    assert len(blob) < 100_000
    restored = pickle.loads(blob)
    with pytest.raises(SnapshotChanged):
        restored["large.py"]
    restored.rebind_start_snapshot(parent)
    assert restored["large.py"] == large
    assert restored["small.py"] == b"VALUE = 1\n"

    changed_parent = tmp_path / "changed-parent"
    changed_parent.mkdir()
    (changed_parent / "large.py").write_bytes(b"y" * (1 << 20))
    (changed_parent / "small.py").write_bytes(b"VALUE = 2\n")
    restored.rebind_start_snapshot(changed_parent)
    with pytest.raises(SnapshotChanged):
        restored["large.py"]
    assert restored["small.py"] == b"VALUE = 1\n"  # kept in the pickle, proven by its own digest
    restored = pickle.loads(blob)
    restored._bodies["small.py"] = b"VALUE = 2\n"
    with pytest.raises(SnapshotChanged):
        restored["small.py"]

    old = pickle.loads(pickle.dumps({"old.py": b"old"}, protocol=5))
    assert diff_snapshots(old, {"old.py": b"new"})[0].path == "old.py"


@pytest.mark.parametrize("parent_change", ["changed", "removed", "file_link", "directory_link"])
def test_changed_start_copy_has_no_completion_delivery_or_green_credit(
        tmp_path: Path, parent_change: str):
    parent = tmp_path / "parent"
    worker = tmp_path / "worker"
    parent.mkdir()
    worker.mkdir()
    relative = "pkg/app.py"
    original = b"VALUE = 1\n" + b"# start\n" * 10_000
    for root in (parent, worker):
        target = root / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
    baseline = workspace_snapshot(worker, start_snapshot=parent)

    events = []
    journal = SimpleNamespace(emit=lambda event, **fields: events.append((event, fields)))
    current = {"revision": "start", "product_revision": "start",
               "product_files": {relative: "sha256:" + _sha(original)}}
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=str(worker), task_row={"editable_paths": [relative]},
                              f2p_ids=["check"], _receipts=[]),
        _host_done_facts_baseline=baseline,
        _delivery_contract_baseline=baseline,
        journal=journal,
        test_receipts=[],
        restart_enabled=True,
    )
    runtime.binding = lambda: current
    tracker = DeliveryManifestTracker(worker, baseline, journal, runtime)
    lock = GreenLock(runtime, mode="terminal")

    edited = b"VALUE = 2\n" + b"# worker edit\n" * 10_000
    (worker / relative).write_bytes(edited)
    current.update(revision="edited", product_revision="edited",
                   product_files={relative: "sha256:" + _sha(edited)})
    parent_file = parent / relative
    if parent_change == "changed":
        parent_file.write_bytes(b"VALUE = 3\n" + b"# changed\n" * 10_000)
    elif parent_change == "removed":
        parent_file.unlink()
    elif parent_change == "file_link":
        replacement = parent / "replacement.py"
        replacement.write_bytes(original)
        parent_file.unlink()
        parent_file.symlink_to(replacement)
    else:
        folder = parent / "pkg"
        moved = parent / "pkg-real"
        folder.rename(moved)
        folder.symlink_to(moved, target_is_directory=True)

    with pytest.raises(SnapshotChanged):
        baseline[relative]
    facts = compute_done_facts(worker, baseline, journal=journal)
    assert not facts.accepted
    assert f"snapshot_unavailable:{relative}" in facts.unresolved_gating_items
    assert facts.file_deltas[0].status == "unavailable"

    changed, deltas = tracker._workspace_delta()
    assert changed == [relative]
    assert deltas[0]["status"] == "unavailable"
    applied, open_items = _applied_edit_rows(
        [{"kind": "edit", "path": relative, "result": "APPLIED",
          "receipt_id": "edit-1"}], deltas)
    assert applied == []
    assert f"applied_edit_snapshot_unavailable:{relative}" in open_items
    journal_manifest = build_manifest(
        phase="prepared", files_changed=changed, claims={}, receipts=[],
        open_items=[], payload={}, file_deltas=deltas,
        journal_applied_edits=[{"file": relative, "hunk": "-old\n+new\n"}],
    )
    assert journal_manifest["applied_edits"] == []
    assert f"applied_edit_snapshot_unavailable:{relative}" in journal_manifest["open_items"]
    assert ("workspace_snapshot_changed", "delivery_manifest") in {
        (event, fields.get("consumer")) for event, fields in events
    }
    runtime.owner._receipts.append({"kind": "edit", "path": relative,
                                    "result": "APPLIED", "receipt_id": "edit-1"})
    tracker.prepare_message([], [], request_id="snap-change")
    manifest = tracker._pending_manifest
    assert manifest is not None
    assert f"snapshot_unavailable:{relative}" in manifest["open_items"]
    assert not any(row.get("file") == relative for row in manifest["applied_edits"])

    assert not lock.sync()
    assert "product source unavailable for this file" in lock._terminal_change_diff()


def test_live_file_changed_after_hash_is_diffed_from_bytes_read(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "large.py"
    old = b"A" * (100 * 1024)
    observed = b"B" * (100 * 1024)
    actual = b"C" * (100 * 1024)
    target.write_bytes(old)
    baseline = workspace_snapshot(root)
    target.write_bytes(observed)
    after = workspace_snapshot(root, live=True)
    target.write_bytes(actual)

    (delta,) = diff_snapshots(baseline, after)
    assert delta.after_sha256 == _sha(actual)
    assert delta.status == "modified"
    assert after.digest_for_path("large.py") == _sha(actual)
    assert after.changed_during_read("large.py")


@pytest.mark.parametrize("size", [8, 100 * 1024], ids=["small", "large"])
def test_file_that_reads_back_equal_is_an_empty_diff(tmp_path: Path, monkeypatch, size):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "a.py"
    target.write_bytes(b"A" * size)
    baseline = workspace_snapshot(root)
    real = host_done_facts.workspace_snapshot

    def racing(path, **options):
        target.write_bytes(b"B" * size)
        snapshot = real(path, **options)
        target.write_bytes(b"A" * size)
        return snapshot

    monkeypatch.setattr(host_done_facts, "workspace_snapshot", racing)
    facts = compute_done_facts(root, baseline)
    deltas = facts.file_deltas
    assert all(delta.before_sha256 != delta.after_sha256 for delta in deltas)
    assert facts.empty_diff is (not deltas)
    if size > 64 * 1024:
        assert facts.empty_diff is True and facts.changed_files == ()


def _terminal_lock(worker, baseline, files):
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=str(worker), task_row={}, f2p_ids=["check"], _receipts=[]),
        _host_done_facts_baseline=baseline, _delivery_contract_baseline=baseline,
        journal=None, test_receipts=[], restart_enabled=True,
    )
    binding = {"revision": "start", "product_revision": "start",
               "product_files": {name: "sha256:" + _sha(raw) for name, raw in files.items()}}
    runtime.binding = lambda: binding
    return GreenLock(runtime, mode="terminal"), binding


def _two_trees(tmp_path, files):
    for name in ("parent", "worker"):
        for relative, raw in files.items():
            target = tmp_path / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    return tmp_path / "parent", tmp_path / "worker"


def test_unchanged_undecodable_source_does_not_block(tmp_path: Path):
    files = {"app.py": b"VALUE = 1\r\n", "legacy.py": b"\xff" * 100}
    parent, worker = _two_trees(tmp_path, files)
    lock, binding = _terminal_lock(worker, workspace_snapshot(worker, start_snapshot=parent), files)
    edited = b"VALUE = 2\n"
    (worker / "app.py").write_bytes(edited)
    binding.update(revision="edited", product_revision="edited",
                   product_files={**binding["product_files"], "app.py": "sha256:" + _sha(edited)})
    diff = lock._terminal_change_diff()
    assert "-VALUE = 1" in diff and "+VALUE = 2" in diff
    assert "legacy.py" not in diff and "unavailable" not in diff


def test_only_changed_sources_are_read(tmp_path: Path, monkeypatch):
    from kernel_host_done_facts import SnapshotFiles

    files = {f"pkg/m{index}.py": b"# %d\n" % index + b"x" * 70_000 for index in range(20)}
    parent, worker = _two_trees(tmp_path, files)
    baseline = workspace_snapshot(worker, start_snapshot=parent)
    lock, binding = _terminal_lock(worker, baseline, files)
    edited = b"changed\n"
    (worker / "pkg/m3.py").write_bytes(edited)
    binding["product_files"] = {**binding["product_files"], "pkg/m3.py": "sha256:" + _sha(edited)}
    seen = []
    load = SnapshotFiles.__getitem__
    monkeypatch.setattr(SnapshotFiles, "__getitem__", lambda self, path: seen.append(path) or load(self, path))
    assert "+changed" in lock._terminal_change_diff()
    assert seen == ["pkg/m3.py"]


def test_file_without_a_proven_start_copy_survives_a_pickle(tmp_path: Path):
    parent, worker = _two_trees(tmp_path, {"same.py": b"s" * 100_000})
    (worker / "overlay.py").write_bytes(b"o" * 100_000)
    (worker / "small-overlay.py").write_bytes(b"o")
    (worker / "differs.py").write_bytes(b"w" * 100_000)
    (parent / "differs.py").write_bytes(b"p" * 100_000)
    restored = pickle.loads(pickle.dumps(workspace_snapshot(worker, start_snapshot=parent)))
    restored.rebind_start_snapshot(parent)
    for name in ("same.py", "overlay.py", "small-overlay.py", "differs.py"):
        assert restored[name] == (worker / name).read_bytes()


def test_source_that_changed_before_capture_keeps_its_live_text(tmp_path: Path):
    files = {"app.py": b"VALUE = 1\n"}
    parent, worker = _two_trees(tmp_path, files)
    baseline = workspace_snapshot(worker, start_snapshot=parent)
    (worker / "app.py").write_bytes(b"VALUE = 9\n")
    lock, _ = _terminal_lock(worker, baseline, {"app.py": b"VALUE = 9\n"})
    assert lock._starting_product_sources.get("app.py") == "VALUE = 9\n"


def test_green_lock_owner_snapshot_round_trip(tmp_path: Path):
    from kernel_stateframe_resume import _green_lock_snapshot, _json_value, _restore_green_lock

    files = {"app.py": b"VALUE = 1\n" + b"#" * 70_000}
    parent, worker = _two_trees(tmp_path, files)
    baseline = workspace_snapshot(worker, start_snapshot=parent)
    lock, _ = _terminal_lock(worker, baseline, files)
    state = _json_value(_green_lock_snapshot(lock))
    assert len(json.dumps(state)) < 20_000
    other, _ = _terminal_lock(worker, baseline, files)
    _restore_green_lock(other, state)
    assert other._starting_product_sources.get("app.py") == files["app.py"].decode()
    state["state"]["_starting_product_sources"] = {"app.py": "old text"}
    _restore_green_lock(other, state)
    assert other._starting_product_sources == {"app.py": "old text"}


def test_worker_root_reached_through_a_link(tmp_path: Path):
    parent, worker = _two_trees(tmp_path, {"app.py": b"VALUE = 1\n"})
    (tmp_path / "link").symlink_to(worker, target_is_directory=True)
    assert dict(workspace_snapshot(tmp_path / "link", start_snapshot=parent)) == {"app.py": b"VALUE = 1\n"}


def test_tree_digest_names_a_file_that_fails_while_read():
    from kernel_tree_digest import compatible_revision

    def broken():
        yield b"first"
        raise OSError("read failed")

    whole = [("a.py", b"first", "file"), ("b.py", None, "unreadable")]
    assert compatible_revision("t/1", whole) == compatible_revision(
        "t/1", iter([("a.py", iter([b"fi", b"rst"]), "file"), ("b.py", broken(), "file")]))


def test_restored_runtime_reads_from_the_current_start_copy(tmp_path: Path):
    from kernel_host_done_facts import rebind_workspace_snapshots

    files = {"app.py": b"VALUE = 1\n" + b"#" * 70_000}
    parent, worker = _two_trees(tmp_path, files)
    baseline = workspace_snapshot(worker, start_snapshot=parent)
    lock, _ = _terminal_lock(worker, baseline, files)
    tracker = DeliveryManifestTracker(worker, baseline, None, None)
    saved = pickle.dumps((baseline, lock._starting_product_sources, tracker.baseline))
    baseline, sources, tracked = pickle.loads(saved)
    assert sources.get("app.py") is None  # nothing is proven before the rebind
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=str(worker)), _host_done_facts_baseline=baseline,
        _delivery_contract_baseline=baseline, delivery_contract=SimpleNamespace(baseline=tracked),
        green_lock=SimpleNamespace(_starting_product_sources=sources))
    moved = tmp_path / "attempt-2-parent"
    parent.rename(moved)
    rebind_workspace_snapshots(runtime, moved)
    assert baseline["app.py"] == tracked["app.py"] == files["app.py"]
    assert sources.get("app.py") == files["app.py"].decode()
