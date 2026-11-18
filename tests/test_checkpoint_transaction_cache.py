"""Verified pool identities survive transaction publication, but never mutation."""
from __future__ import annotations

import json
import os
import signal
import sys
import time

import pytest
from fixtures.process_signals import record_identity, signal_run_verified

from stateframe.engine import kernel_checkpoint as checkpoint


@pytest.fixture
def staged_pool_file(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    pending = root / ".0001-test.tmp"
    artifacts = pending / checkpoint.ARTIFACT_STAGE_DIR
    source = tmp_path / "source.json"
    source.write_bytes(b'{"value":"safe"}')
    target = root / checkpoint.POOL_DIR / "wire" / "0001.request.json"
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})
    monkeypatch.setattr(checkpoint, "_SAVE", {
        "artifact_root": artifacts, "pool_root": root / checkpoint.POOL_DIR,
    })
    created = []
    identity, _ = checkpoint._capture_snapshot_file(source, target, new_pool_files=created)
    checkpoint._POOL_KNOWN[str(target.parent)] = {target.name: identity}
    return root, pending, source, target, created[0], identity


def _cached(path, identity):
    return checkpoint._snapshot_file_cache_key(
        path.stat(), checkpoint._SECRET_GENERATION, identity[1]
    ) in checkpoint._VERIFIED_POOL_IDENTITIES


def test_publication_rescans_then_reuses_verified_pool_identity(
        staged_pool_file, monkeypatch):
    root, pending, source, target, staged, identity = staged_pool_file
    assert _cached(staged, identity)

    original_check = checkpoint._check_snapshot_file
    scans = []

    def observe_scan(path, **kwargs):
        scans.append((path, kwargs.get("force_scan", False)))
        return original_check(path, **kwargs)

    monkeypatch.setattr(checkpoint, "_check_snapshot_file", observe_scan)

    checkpoint._publish_staged_artifacts(root, pending)

    assert not staged.exists()
    assert target.read_bytes() == source.read_bytes()
    assert _cached(target, identity)
    assert scans == [(target, True)]

    def unexpected_scan(*_args, **_kwargs):
        pytest.fail("unchanged published pool file was scanned again")

    monkeypatch.setattr(checkpoint, "_check_snapshot_file", unexpected_scan)
    assert checkpoint._capture_snapshot_file(
        source, target, known=identity, new_pool_files=[])[0] == identity


def test_publication_refuses_mutated_staged_bytes(staged_pool_file):
    root, pending, _source, target, staged, identity = staged_pool_file
    before = staged.stat()
    # Pooled files are read-only by design; this tamperer bypasses that with chmod.
    staged.chmod(0o600)
    staged.write_bytes(b'{"value":"evil"}')
    os.utime(staged, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(checkpoint.CheckpointError, match="pooled file changed"):
        checkpoint._publish_staged_artifacts(root, pending)

    assert not _cached(target, identity)


def test_publication_rescans_with_new_secret_generation(staged_pool_file):
    root, pending, _source, target, _staged, identity = staged_pool_file
    checkpoint.register_secret(b"newly-registered-secret")

    checkpoint._publish_staged_artifacts(root, pending)

    assert _cached(target, identity)


def test_published_pool_mutation_invalidates_transferred_verification(staged_pool_file):
    root, pending, source, target, _staged, identity = staged_pool_file
    checkpoint._publish_staged_artifacts(root, pending)
    assert _cached(target, identity)
    before = target.stat()
    # Pooled files are read-only by design; this tamperer bypasses that with chmod.
    target.chmod(0o600)
    target.write_bytes(b'{"value":"evil"}')
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(checkpoint.CheckpointError, match="pooled file changed"):
        checkpoint._capture_snapshot_file(source, target, known=identity, new_pool_files=[])


def test_staged_capture_refuses_pool_file_linked_to_live_source(staged_pool_file):
    _root, _pending, source, target, _staged, identity = staged_pool_file
    target.parent.mkdir(parents=True)
    os.link(source, target)

    with pytest.raises(checkpoint.CheckpointError, match="checkpoint artifact identity conflict"):
        checkpoint._capture_snapshot_file(source, target, known=identity, new_pool_files=[])


def test_rollback_removes_published_pool_and_invalidates_verification(staged_pool_file):
    root, pending, _source, target, _staged, identity = staged_pool_file
    checkpoint._publish_staged_artifacts(root, pending)
    assert _cached(target, identity)

    checkpoint._cleanup_checkpoint_transaction(root, pending)

    assert not pending.exists()
    assert not target.exists()
    assert not checkpoint._VERIFIED_POOL_IDENTITIES
    assert not checkpoint._POOL_KNOWN


def test_rollback_preserves_committed_pool_reference_and_latest(staged_pool_file, tmp_path):
    root, pending, source, target, _staged, identity = staged_pool_file
    checkpoint._publish_staged_artifacts(root, pending)
    kept = root / "0001-kept"
    (kept / "run" / "wire").mkdir(parents=True)
    (kept / checkpoint.META_FILE).write_text(json.dumps({"seq": 1, "chunks": []}))
    (kept / checkpoint.MANIFEST_FILE).write_text(json.dumps({
        "wire": [[target.name, identity[0]]],
    }))
    (root / "LATEST").write_text(kept.name + "\n")

    checkpoint._cleanup_checkpoint_transaction(root, pending)

    assert target.read_bytes() == source.read_bytes()
    assert checkpoint.latest(root.parent) == kept
    restored = tmp_path / "restored"
    restored.mkdir()
    checkpoint._restore_files(kept, restored, {"entries": ["wire"], "file_chunks": {}})
    assert (restored / "wire" / target.name).read_bytes() == source.read_bytes()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="background writer requires fork")
@pytest.mark.parametrize("failure", ["abrupt-exit", "alarm"])
def test_reap_rolls_back_artifacts_when_writer_cannot_run_cleanup(staged_pool_file, failure):
    root, pending, _source, target, _staged, _identity = staged_pool_file
    checkpoint._STATE.update(failed=0, consecutive_failed=0, disabled=None)

    def write():
        checkpoint._publish_staged_artifacts(root, pending)
        if failure == "alarm":
            pid = os.getpid()
            identity = record_identity(pid, launched_after=0, command_fragment=sys.executable)
            # This writer intentionally dies before it can publish a checkpoint.
            signal_run_verified(root.parent, pid, identity, signal.SIGALRM,
                                deadline=time.monotonic(), what="writer alarm fault",
                                require_checkpoint_saved=False)
        os._exit(1)

    checkpoint._fork_writer(root.parent, 1, root / "0001-unpublished", write, pending)
    try:
        deadline = time.monotonic() + 5
        while not checkpoint._reap() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert checkpoint._STATE.get("child") is None
        assert checkpoint.failure_summary()["checkpoint_failures"] == 1
        assert not target.exists()
        assert not pending.exists()
    finally:
        pid = checkpoint._STATE.get("child")
        if pid is not None:
            try:
                identity = record_identity(pid, launched_after=0, command_fragment=sys.executable)
                signal_run_verified(root.parent, pid, identity, signal.SIGKILL,
                                    deadline=time.monotonic(), what="writer timeout cleanup",
                                    require_checkpoint_saved=False)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
            checkpoint._cleanup_checkpoint_transaction(root, pending)
