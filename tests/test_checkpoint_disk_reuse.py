"""A checkpoint adds only the bytes that changed: unchanged content is shared, never stored twice.

Every test drives the real ``checkpoint.save`` / ``_validate`` / ``_restore_files`` path. Only the
in-memory capture (``collect``/``dump_to``) is stubbed, as in test_checkpoint_stale_copy.py.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import pickle
import random
import stat
import sys
from pathlib import Path

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint

posix_only = pytest.mark.skipif(os.name != "posix", reason="inode sharing requires POSIX")
SEED_KEY = hashlib.sha256(b"disk-reuse-seed").hexdigest()
# checkpoint.json, state.pkl, manifest, LATEST and the events log: a few 4 KiB blocks each.
OVERHEAD = 256 << 10


@pytest.fixture(autouse=True)
def _checkpoint_environment(monkeypatch):
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"disk-reuse-signing-key-0123456789ab"])
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    state = {"frame": {}, "modules": {}, "classes": {}, "patched": {}, "participants": {},
             "context": [], "random": random.getstate()}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: (dict(state), context))
    monkeypatch.setattr(checkpoint, "dump_to",
                        lambda stream, payload, _ctx, protocol=5: pickle.dump(payload, stream))


def _inodes(root: Path) -> dict[tuple[int, int], int]:
    """Every regular inode under ``root`` with its allocated bytes (st_blocks, not apparent size)."""
    found: dict[tuple[int, int], int] = {}
    if not root.is_dir():
        return found
    for directory, _dirs, names in os.walk(root):
        for name in names:
            info = os.lstat(os.path.join(directory, name))
            if stat.S_ISREG(info.st_mode):
                found[(info.st_dev, info.st_ino)] = info.st_blocks * 512
    return found


class _Ledger:
    """Bytes newly allocated by one save: inodes under the root that did not exist before it."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, run: Path, step: int) -> int:
        before = _inodes(self.root)
        checkpoint.save(sys._getframe(), step, run)
        after = _inodes(self.root)
        return sum(blocks for key, blocks in after.items() if key not in before)


def _seeded_run(base: Path) -> Path:
    """A run whose env/ files are read-only hard links of an env-seed (the production layout)."""
    seed = base / "seeds" / "env-seeds" / SEED_KEY
    seed.mkdir(parents=True)
    (seed / "seed.json").write_text(json.dumps({"key": SEED_KEY}))
    for index in range(20):
        (seed / f"small{index:02d}.py").write_bytes(f"x{index}\n".encode() * 2000)
    (seed / "model.bin").write_bytes(os.urandom(4 << 20))
    run = base / "run"
    (run / "env").mkdir(parents=True)
    (run / "env-seed.json").write_text(json.dumps({"path": str(seed)}))
    for source in seed.iterdir():
        if source.name != "seed.json":
            os.link(source, run / "env" / source.name)
            os.chmod(source, 0o444)
    (run / "worker" / "extra").mkdir(parents=True)
    (run / "worker" / "extra" / "blob.bin").write_bytes(os.urandom(2 << 20))
    (run / "payload.json").write_text('{"turn": 0}' * 50)
    return run


def _plain_run(base: Path) -> Path:
    run = base / "run"
    (run / "worker" / "extra").mkdir(parents=True)
    (run / "worker" / "extra" / "blob.bin").write_bytes(os.urandom(3 << 20))
    (run / "env").mkdir()
    (run / "env" / "model.bin").write_bytes(os.urandom(2 << 20))
    (run / "payload.json").write_text('{"turn": 0}' * 50)
    return run


@posix_only
def test_recently_changed_file_is_not_memoized(tmp_path, monkeypatch):
    # A coarse clock can give a rewrite right after a scan the same ctime, so the digest of a
    # recently changed file must not be remembered (it is hashed again on the next save).
    live = tmp_path / "state.bin"
    live.write_bytes(b"one")
    monkeypatch.setattr(checkpoint, "_DIGEST_MEMO", {})
    checkpoint._live_digest(live, os.lstat(live))
    assert checkpoint._DIGEST_MEMO == {}
    monkeypatch.setattr(checkpoint, "_RACY_WINDOW_NS", 0)
    checkpoint._live_digest(live, os.lstat(live))
    assert len(checkpoint._DIGEST_MEMO) == 1


@posix_only
def test_second_checkpoint_adds_only_changed_bytes(tmp_path):
    run = _seeded_run(tmp_path)
    ledger = _Ledger(run / "checkpoints")
    first = ledger.save(run, 1)
    assert first >= 6 << 20  # the first checkpoint stores the whole run once

    (run / "payload.json").write_text('{"turn": 1}' * 50)  # the only changed bytes
    second = ledger.save(run, 2)

    changed = len(((run / "payload.json").read_bytes()))
    assert second <= changed + OVERHEAD, f"second checkpoint allocated {second} bytes"
    assert (checkpoint.latest(run) / "run" / "payload.json").read_bytes() == (run / "payload.json").read_bytes()


@posix_only
def test_same_bytes_new_mtime_is_not_stored_twice(tmp_path):
    run = _plain_run(tmp_path)
    ledger = _Ledger(run / "checkpoints")
    ledger.save(run, 1)
    blob = run / "worker" / "extra" / "blob.bin"
    before = blob.read_bytes()
    os.utime(blob, ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000))

    second = ledger.save(run, 2)

    assert second <= OVERHEAD, f"unchanged bytes were stored again: {second}"
    assert (checkpoint.latest(run) / "run" / "worker" / "extra" / "blob.bin").read_bytes() == before


@posix_only
def test_append_after_checkpoint_leaves_old_checkpoint_identical(tmp_path):
    run = _plain_run(tmp_path)
    journal = run / "journal.txt"
    journal.write_bytes(b"a" * 1000)
    ledger = _Ledger(run / "checkpoints")
    ledger.save(run, 1)
    old = checkpoint.latest(run) / "run" / "journal.txt"
    recorded = old.read_bytes()

    # Append, then truncate and rewrite in place: the live inode is mutated, never a checkpoint.
    with journal.open("ab") as stream:
        stream.write(b"b" * 500)
    with journal.open("r+b") as stream:
        stream.truncate(0)
        stream.write(b"c" * 10)
    assert old.read_bytes() == recorded

    ledger.save(run, 2)
    newest = checkpoint.latest(run)
    assert (newest / "run" / "journal.txt").read_bytes() == b"c" * 10
    assert old.read_bytes() == recorded

    meta, _whole = checkpoint._validate(newest)
    checkpoint._restore_files(newest, run, meta)
    restored = run / "journal.txt"
    assert not os.path.samestat(restored.stat(), (newest / "run" / "journal.txt").stat())
    with restored.open("ab") as stream:
        stream.write(b"after restore")
    assert (newest / "run" / "journal.txt").read_bytes() == b"c" * 10


@posix_only
def test_pool_files_are_read_only(tmp_path):
    run = _plain_run(tmp_path)
    blobs = run / "worker" / "measurement-runs" / "r1" / "blobs"
    blobs.mkdir(parents=True)
    for index in range(3):
        (blobs / f"b{index}.bin").write_bytes(os.urandom(4096))
    _Ledger(run / "checkpoints").save(run, 1)

    stored = []
    for store in ("env-objects", "blob-pool"):
        for directory, _dirs, names in os.walk(run / "checkpoints" / store):
            stored.extend(Path(directory, name) for name in names)
    assert stored, "the content and pool stores were not populated"
    writable = [path for path in stored if os.lstat(path).st_mode & 0o222]
    assert writable == []


@posix_only
@pytest.mark.parametrize("error", [errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP])
def test_link_exdev_falls_back_to_copy(tmp_path, monkeypatch, error):
    run = _plain_run(tmp_path)
    ledger = _Ledger(run / "checkpoints")
    ledger.save(run, 1)
    payload = run / "payload.json"
    payload.write_text('{"turn": 9}' * 50)

    def refuse(source, destination, *args, **kwargs):
        raise OSError(error, os.strerror(error))

    monkeypatch.setattr(os, "link", refuse)
    ledger.save(run, 2)

    newest = checkpoint.latest(run)
    assert (newest / "run" / "payload.json").read_bytes() == payload.read_bytes()
    meta = json.loads((newest / checkpoint.META_FILE).read_text())
    assert meta["link_fallback"] > 0
    assert meta["files"]["link_fallback"] == meta["link_fallback"]


@posix_only
def test_restore_round_trip_after_reuse(tmp_path):
    run = _plain_run(tmp_path)
    script = run / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n")
    script.chmod(0o755)
    ledger = _Ledger(run / "checkpoints")
    ledger.save(run, 1)
    payload_mode = stat.S_IMODE(os.lstat(run / "payload.json").st_mode)

    (run / "payload.json").write_text('{"turn": 2}' * 50)
    script.write_text("#!/bin/sh\necho changed\n")
    script.chmod(0o755)
    ledger.save(run, 2)
    expected = {name: (run / name).read_bytes() for name in ("payload.json", "run.sh")}
    newest = checkpoint.latest(run)

    (run / "payload.json").write_text("live drifted")
    (run / "run.sh").write_text("drift")
    meta, _whole = checkpoint._validate(newest)
    checkpoint._restore_files(newest, run, meta)

    for name, data in expected.items():
        assert (run / name).read_bytes() == data
        assert os.lstat(run / name).st_mode & stat.S_IWUSR  # the restored copy is writable
    assert stat.S_IMODE(os.lstat(run / "run.sh").st_mode) == 0o755
    assert stat.S_IMODE(os.lstat(run / "payload.json").st_mode) == payload_mode
