"""Checkpoint publication and shared storage invariants."""
from __future__ import annotations

import json
import os
import hashlib
import sys
from pathlib import Path

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint


def _complete(path: Path, seq: int) -> None:
    path.mkdir()
    (path / checkpoint.META_FILE).write_text(json.dumps({"seq": seq, "created": seq}))
    (path / checkpoint.MANIFEST_FILE).write_text("{}")


def test_publish_crash_never_points_latest_at_missing_directory(tmp_path, monkeypatch):
    run = tmp_path / "run"
    root = run / "checkpoints"
    root.mkdir(parents=True)
    old = root / "0001-1"
    _complete(old, 1)
    (root / "LATEST").write_text(old.name + "\n")
    pending = root / ".0002-2.tmp"
    _complete(pending, 2)
    original = checkpoint.os.replace

    def crash(source, target):
        if Path(target).name == "LATEST":
            raise OSError("simulated crash")
        return original(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        checkpoint._publish_checkpoint(root, pending, root / "0002-2", run)
    assert checkpoint.latest(run) == old
    assert old.is_dir()
    assert (root / "0002-2" / checkpoint.META_FILE).is_file()


def test_late_writer_cannot_prune_latest(tmp_path, monkeypatch):
    run = tmp_path / "run"
    root = run / "checkpoints"
    root.mkdir(parents=True)
    current = root / "0003-3"
    _complete(current, 3)
    (root / "LATEST").write_text(current.name + "\n")
    pending = root / ".0002-2.tmp"
    _complete(pending, 2)
    monkeypatch.setattr(checkpoint, "_keep", lambda: 1)
    checkpoint._publish_checkpoint(root, pending, root / "0002-2", run)
    assert checkpoint.latest(run) == current
    assert current.is_dir()


def test_crash_after_latest_before_prune_keeps_its_target(tmp_path, monkeypatch):
    run = tmp_path / "run"
    root = run / "checkpoints"
    root.mkdir(parents=True)
    previous = root / "0001-1"
    _complete(previous, 1)
    (root / "LATEST").write_text(previous.name + "\n")
    pending = root / ".0002-2.tmp"
    _complete(pending, 2)

    def crash(*_args):
        raise OSError("simulated prune crash")

    monkeypatch.setattr(checkpoint, "_prune_chunks", crash)
    with pytest.raises(OSError, match="simulated prune crash"):
        checkpoint._publish_checkpoint(root, pending, root / "0002-2", run)
    assert checkpoint.latest(run) == root / "0002-2"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="async writer requires fork")
def test_async_writer_publishes_completed_directory(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    root.mkdir(parents=True)
    pending = root / ".0002-2.tmp"
    final = root / "0002-2"
    _complete(pending, 2)
    before = checkpoint._STATE.copy()

    def write():
        checkpoint._publish_checkpoint(root, pending, final, run)
        return {"sends": 1, "state_bytes": 1, "chunks": []}

    try:
        checkpoint._fork_writer(run, 2, final, write, pending)
        pid = checkpoint._STATE["child"]
        waited, status = os.waitpid(pid, 0)
        assert waited == pid and os.waitstatus_to_exitcode(status) == 0
        assert checkpoint.latest(run) == final
    finally:
        checkpoint._STATE.clear()
        checkpoint._STATE.update(before)


@pytest.mark.skipif(os.name != "posix", reason="inode sharing requires POSIX")
def test_snapshot_links_only_seed_linked_readonly_files(tmp_path):
    seed = tmp_path / "seed"
    run = tmp_path / "run"
    out = tmp_path / "snapshot"
    for root in (seed, run):
        (root / "lib/python3/site-packages").mkdir(parents=True)
    shared = seed / "lib/python3/site-packages/package.py"
    shared.write_text("shared\n")
    shared.chmod(0o444)
    os.link(shared, run / shared.relative_to(seed))
    private = run / "pyvenv.cfg"
    private.write_text("home = run\n")
    counts = checkpoint._copy_tree(run, out, None, set(), seed=seed)
    assert (out / shared.relative_to(seed)).stat().st_ino == shared.stat().st_ino
    assert (out / "pyvenv.cfg").stat().st_ino != private.stat().st_ino
    assert counts["linked"] == 1


def test_pool_gc_keeps_every_manifest_reference(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    store = root / checkpoint.POOL_DIR / "wire"
    store.mkdir(parents=True)
    (store / "kept").write_text("keep")
    (store / "unused").write_text("drop")
    saved = root / "0001-1"
    _complete(saved, 1)
    (saved / checkpoint.MANIFEST_FILE).write_text(json.dumps({"wire": [["kept", 4]]}))
    checkpoint._prune_pool(root, run)
    assert (store / "kept").is_file()
    assert not (store / "unused").exists()


def test_pool_gc_keeps_reference_from_unpublished_manifest(tmp_path):
    run = tmp_path / "run"
    root = run / "checkpoints"
    store = root / checkpoint.POOL_DIR / "wire"
    store.mkdir(parents=True)
    (store / "pending").write_text("data")
    pending = root / ".0002-2.tmp"
    pending.mkdir()
    (pending / checkpoint.MANIFEST_FILE).write_text(json.dumps({"wire": [["pending", 4]]}))
    checkpoint._prune_pool(root, run)
    assert (store / "pending").is_file()


def test_linux_pool_snapshot_links_write_once_files(tmp_path):
    source = tmp_path / "run/wire"
    source.mkdir(parents=True)
    (source / "0001.request.json").write_text('{"model":"test"}')
    target = tmp_path / "snapshot/wire"
    target.parent.mkdir()
    pool = tmp_path / "checkpoints/blob-pool"
    manifests = {}
    created = []
    checkpoint._snapshot_tree(source, target, "wire", pool, manifests, created)
    pooled = pool / "wire/0001.request.json"
    assert pooled.is_file()
    assert (source / "0001.request.json").stat().st_ino == pooled.stat().st_ino
    assert target.is_dir() and not list(target.iterdir())
    assert len(manifests["wire"][0]) == 2


def test_pool_checksum_detects_same_size_tampering(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    root.mkdir()
    saved = root / "0001-1"
    _complete(saved, 1)
    state = b"state"
    (saved / checkpoint.STATE_FILE).write_bytes(state)
    store = root / checkpoint.POOL_DIR / "wire"
    store.mkdir(parents=True)
    blob = store / "0001.request.json"
    blob.write_bytes(b"good")
    manifest = json.dumps({"wire": [[blob.name, 4]]}).encode()
    (saved / checkpoint.MANIFEST_FILE).write_bytes(manifest)
    meta = {"schema": checkpoint.SCHEMA, "state_bytes": len(state),
            "state_sha256": hashlib.sha256(state).hexdigest(),
            "blob_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "pool_sha256": {f"wire/{blob.name}": hashlib.sha256(b"good").hexdigest()},
            "signature": "test"}
    (saved / checkpoint.META_FILE).write_text(json.dumps(meta))
    monkeypatch.setattr(checkpoint, "_owned_private", lambda *_args: True)
    monkeypatch.setattr(checkpoint, "_signature", lambda *_args: "test")
    blob.write_bytes(b"evil")
    with pytest.raises(checkpoint.CheckpointError, match="checksum mismatch"):
        checkpoint.authenticate(saved)


def test_missing_pooled_blob_refuses_with_its_name(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    root.mkdir()
    saved = root / "0001-1"
    _complete(saved, 1)
    state = b"state"
    (saved / checkpoint.STATE_FILE).write_bytes(state)
    manifest = json.dumps({"wire": [["missing.request.json", 10]]}).encode()
    (saved / checkpoint.MANIFEST_FILE).write_bytes(manifest)
    meta = {"schema": checkpoint.SCHEMA, "state_bytes": len(state),
            "state_sha256": hashlib.sha256(state).hexdigest(),
            "blob_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "signature": "test"}
    (saved / checkpoint.META_FILE).write_text(json.dumps(meta))
    monkeypatch.setattr(checkpoint, "_owned_private", lambda *_args: True)
    monkeypatch.setattr(checkpoint, "_signature", lambda *_args: "test")
    with pytest.raises(checkpoint.CheckpointError, match="missing pooled blob wire/missing.request.json"):
        checkpoint.authenticate(saved)


def test_restore_point_records_missing_blob_refusal(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "unusable-checkpoint"
    monkeypatch.setenv(checkpoint.RESTORE_ENV, str(source))

    def refuse(*_args):
        raise checkpoint.CheckpointError("missing pooled blob wire/0001.request.json")

    monkeypatch.setattr(checkpoint, "restore", refuse)
    before = checkpoint._STATE.copy()
    try:
        with pytest.raises(checkpoint.RestoreRefused, match="missing pooled blob wire/0001.request.json"):
            checkpoint.restore_point(sys._getframe(), run / "poc")
    finally:
        checkpoint._STATE.clear()
        checkpoint._STATE.update(before)
    detail = json.loads((run / checkpoint.REFUSAL_FILE).read_text())["detail"]
    assert "missing pooled blob wire/0001.request.json" in detail
