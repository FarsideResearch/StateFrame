"""Work and audit contracts for repeated compact workspace proposals."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import kernel_bridge_evidence as evidence
import kernel_workspace_index as index
import pytest
from kernel_workspace_candidate import checkpoint, export_candidate, read_candidate
from kernel_workspace_consumers import bind_run_consumers, write_scope
from kernel_workspace_objects import ObjectLease
from kernel_workspace_refs import Ref, RootKey, bind_run_store, read_root, retire_root, workspace_run_id
from kernel_workspace_state import WorkspaceUnavailable
from kernel_workspace_tree_io import lookup_ref, read_entry

from stateframe import _child


@pytest.mark.skipif(sys.platform != "linux", reason="the production change feed uses inotify")
def test_compact_proposals_do_only_changed_workspace_work(tmp_path, monkeypatch):
    home, run = tmp_path / "home", tmp_path / "run"
    home.mkdir(mode=0o700)
    run.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    parent, work = run / "parent", run / "workspace"
    parent.mkdir()
    work.mkdir()
    payloads = {}
    for directory in range(50):
        group = f"group-{directory:02d}"
        (parent / group).mkdir()
        (work / group).mkdir()
        for file in range(100):
            name = f"{group}/source-{file:02d}.py"
            # Keep object-store setup small while exercising all 5,000 paths.
            data = f"value = {file}\n".encode()
            payloads[name] = data
            (parent / name).write_bytes(data)
            (work / name).write_bytes(data)

    bind_run_store(run, "e" * 64)
    consumers = bind_run_consumers(run, work, parent=parent)
    monkeypatch.setattr(evidence, "capture_candidate", evidence.capture_candidate)
    _child.install_compact_capture(parent)
    folder = run / "worker" / "candidates"
    target = "group-00/source-00.py"
    edited = "group-49/source-99.py"
    new_body = b"value = 'edited'\n"
    row = {"path": target, "files": {target: payloads[target]}}
    old_blobs = {hashlib.sha256(data).hexdigest() for data in payloads.values()}
    changed_blob = hashlib.sha256(new_body).hexdigest()
    counts: Counter[str] = Counter()
    refresh = index.WorkspaceIndex.refresh
    publish_many = ObjectLease.publish_many

    def observe(self, *args, **kwargs):
        result = refresh(self, *args, **kwargs)
        root = Path(os.fsdecode(b"/".join((b"", *self._components))))
        if root in (parent, work):
            counts[f"{root.name}_reads"] += self.work.content_reads
            counts["directories_scanned"] += self.work.directories_scanned
        return result

    def publish(self, entries):
        entries = tuple(entries)
        for ref, _source in entries:
            if ref.digest in old_blobs:
                counts["unchanged_blob_stages"] += 1
            if ref.digest == changed_blob:
                counts["changed_blob_stages"] += 1
        return publish_many(self, entries)

    monkeypatch.setattr(index.WorkspaceIndex, "refresh", observe)
    monkeypatch.setattr(ObjectLease, "publish_many", publish)
    # Prime the unfiltered navigation view and the initial proposal boundary.
    assert consumers.digests([target]) == {target: "sha256:" + hashlib.sha256(payloads[target]).hexdigest()}

    def capture(step):
        started = perf_counter()
        evidence.capture_candidate(str(folder), work, row, step)
        return perf_counter() - started

    counts.clear()
    admission = capture(1)
    admitted = counts.copy()
    counts.clear()
    clean = []
    for step in (2, 3, 4):
        assert consumers.digests([target])
        clean.append(capture(step))
    clean_counts = counts.copy()
    counts.clear()
    with write_scope(work / edited):
        (work / edited).write_bytes(new_body)
    dirty = capture(5)
    print(f"admission={admission:.6f}s {dict(admitted)}; "
          f"clean={[round(seconds, 6) for seconds in clean]} {dict(clean_counts)}; "
          f"dirty={dirty:.6f}s {dict(counts)}")

    records = [json.loads((folder / f"{step:04d}.json").read_bytes()) for step in range(1, 6)]
    assert [record["proposals"] for record in records] == list(range(1, 6))
    for step, record in enumerate(records, 1):
        assert record["schema"] == "stateframe-candidate-trees/1"
        assert set(record) == {"schema", "proposals", "namespace", "parent_tree", "new_tree", "projection"}
        assert record["parent_tree"] == records[0]["parent_tree"]
        assert record["projection"] == records[0]["projection"]
        expected = {target: payloads[target]}
        if step == 5:
            expected[edited] = new_body
        else:
            assert record["new_tree"] == records[0]["parent_tree"]
        assert export_candidate(run, folder / f"{step:04d}.json", parent=parent) == expected

    assert not any(clean_counts.values()), f"clean workspace work: {dict(clean_counts)}"
    assert counts["parent_reads"] == counts["unchanged_blob_stages"] == counts["directories_scanned"] == 0
    assert counts["workspace_reads"] == counts["changed_blob_stages"] == 1
    assert admitted["parent_reads"] == 0, f"admitted parent was read again: {dict(admitted)}"
    assert max(*clean, dirty) < 2.0


@pytest.fixture
def proposal(tmp_path, monkeypatch):
    home, run = tmp_path / "home", tmp_path / "run"
    home.mkdir(mode=0o700)
    run.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    parent, work = run / "parent", run / "workspace"
    for root in (parent, work):
        (root / ".pytest_cache").mkdir(parents=True)
        (root / "tracked.py").write_bytes(b"original\n")
        (root / ".pytest_cache" / "target").write_text("ordinary data")
    (work / "view").symlink_to("tracked.py")
    store = bind_run_store(run, "e" * 64)
    consumers = bind_run_consumers(run, work, parent=parent)
    monkeypatch.setattr(evidence, "capture_candidate", evidence.capture_candidate)
    _child.install_compact_capture(parent)
    folder = run / "worker" / "candidates"

    def capture(step):
        evidence.capture_candidate(str(folder), work, {"path": "tracked.py", "files": {"tracked.py": ""}}, step)
        return folder / f"{step:04d}.json"

    return home, run, parent, work, store, consumers, capture


def test_policy_change_requires_current_payload_admission(proposal, monkeypatch):
    _home, run, parent, work, _store, _consumers, capture = proposal
    token = b"private-fixture-value-4910823745"
    with write_scope(work / "tracked.py"):
        (work / "tracked.py").write_bytes(token)
    first = capture(1)
    monkeypatch.setattr(checkpoint, "_SECRETS", [*checkpoint._SECRETS, token])
    with pytest.raises(WorkspaceUnavailable, match="retention policy"):
        capture(2)
    assert not (first.parent / "0002.json").exists()
    with write_scope(work / "tracked.py"):
        (work / "tracked.py").write_bytes(b"replacement\n")
    assert export_candidate(run, capture(2), parent=parent) == {"tracked.py": b"replacement\n"}


def test_coverage_and_symlink_targets_are_checked_again(proposal, monkeypatch):
    home, run, parent, work, store, consumers, capture = proposal
    first = capture(1)
    with consumers.state.begin_write(("view",)):
        (work / "view").unlink()
        (work / "view").symlink_to(".pytest_cache/target")
    second = capture(2)
    original = read_candidate(run, first, parent=parent)
    changed = read_candidate(run, second, parent=parent)
    assert original.new_tree != changed.new_tree
    with store.lease() as lease:
        assert lookup_ref(lease, changed.new_tree, (b".pytest_cache",)) is None
        link = lookup_ref(lease, changed.new_tree, (b"view",))
        assert link is not None
        assert read_entry(lease, link).target == b".pytest_cache/target"
    assert export_candidate(run, second, parent=parent) == {"tracked.py": b"original\n"}

    # The included symlink itself is unchanged. Its excluded target can still
    # change the result of resolving the link against the retention policy.
    with consumers.state.begin_write((".pytest_cache/target",)):
        (work / ".pytest_cache" / "target").unlink()
        (work / ".pytest_cache" / "target").symlink_to(home)
    with pytest.raises(WorkspaceUnavailable, match="retention policy"):
        capture(3)
    assert not (first.parent / "0003.json").exists()
    monkeypatch.setattr(_child, "CAPTURE_SKIP_PARTS", {*_child.CAPTURE_SKIP_PARTS, "view"})
    third = read_candidate(run, capture(3), parent=parent)
    assert third.new_tree == third.parent_tree


def test_proposal_covers_multiple_writes_and_an_interrupted_writer(proposal):
    _home, run, parent, work, _store, consumers, capture = proposal
    first = capture(1)
    with write_scope(work / "extra.py"):
        (work / "extra.py").write_bytes(b"extra\n")
    assert consumers.digests(["extra.py"])
    with write_scope(work / "tracked.py"):
        (work / "tracked.py").unlink()
    assert export_candidate(run, capture(2), parent=parent) == {
        "extra.py": b"extra\n", "tracked.py": None,
    }
    with pytest.raises(RuntimeError, match="writer interrupted"), consumers.state.begin_write(()):
        (work / "extra.py").unlink()
        raise RuntimeError("writer interrupted")
    assert export_candidate(run, capture(3), parent=parent) == {"tracked.py": None}
    assert export_candidate(run, first, parent=parent) == {"tracked.py": b"original\n"}


@pytest.mark.parametrize("lost", ["record", "parent-admission", "tree", "projection"])
def test_missing_retained_root_blocks_next_proposal(proposal, lost):
    _home, run, _parent, _work, store, _consumers, capture = proposal
    boundary = capture(1)
    if lost in ("record", "parent-admission"):
        kind, slot = ("proposal", "0001") if lost == "record" else ("shell-baseline", "parent")
        with store.lease() as lease:
            key = RootKey(workspace_run_id(run), kind, slot)
            ref = read_root(lease, key)
            assert isinstance(ref, Ref)
            retire_root(lease, key, expected=ref)
    else:
        record = json.loads(boundary.read_bytes())
        oid = record["new_tree"] if lost == "tree" else record["projection"][1]
        (store._path / "objects" / oid).unlink()
    with pytest.raises((OSError, ValueError)):
        capture(2)
    assert not (boundary.parent / "0002.json").exists()
