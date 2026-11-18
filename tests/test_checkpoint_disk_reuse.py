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


T1_NS = 1_700_000_000_123_456_789
T2_NS = 1_750_000_000_987_654_321


def _checkpoints_in_order(run: Path) -> list[Path]:
    """Checkpoint roots of ``run`` sorted by their save sequence (oldest first)."""
    roots = [p for p in (run / "checkpoints").iterdir()
             if p.is_dir() and (p / checkpoint.META_FILE).is_file()]
    return sorted(roots, key=lambda p: json.loads((p / checkpoint.META_FILE).read_text())["seq"])


def _restore(root: Path, run: Path) -> None:
    """Restore ``root`` into ``run``, first moving aside the previous restore's setup-discard dir."""
    previous = run / "checkpoints" / f"setup-discard.{os.getpid()}"
    if previous.exists():
        os.rename(previous, run / "checkpoints" / f"setup-discard.prev-{os.urandom(4).hex()}")
    meta, _whole = checkpoint._validate(root)
    checkpoint._restore_files(root, run, meta)


def _set_mtime(path: Path, ns: int) -> int:
    os.utime(path, ns=(ns, ns))
    return os.lstat(path).st_mtime_ns


@posix_only
def test_restore_gives_back_live_mtime_when_bytes_were_shared(tmp_path):
    run = _plain_run(tmp_path)
    tune = run / "env" / "tune.bin"
    tune.write_bytes(b"X" * 4096)
    ledger = _Ledger(run / "checkpoints")
    t1 = _set_mtime(tune, T1_NS)
    ledger.save(run, 1)
    t2 = _set_mtime(tune, T2_NS)  # same bytes, new mtime: checkpoint 2 shares the stored object
    ledger.save(run, 2)

    _restore(checkpoint.latest(run), run)

    restored = os.lstat(run / "env" / "tune.bin")
    assert restored.st_mtime_ns == t2, f"restored mtime {restored.st_mtime_ns} != live {t2} (object carries {t1})"
    assert (run / "env" / "tune.bin").read_bytes() == b"X" * 4096


@posix_only
def test_restore_utime_does_not_touch_shared_object(tmp_path):
    run = _plain_run(tmp_path)
    tune = run / "env" / "tune.bin"
    tune.write_bytes(b"X" * 4096)
    ledger = _Ledger(run / "checkpoints")
    t1 = _set_mtime(tune, T1_NS)
    ledger.save(run, 1)
    _set_mtime(tune, T2_NS)
    ledger.save(run, 2)
    first, second = _checkpoints_in_order(run)
    ckpt2_copy = second / "run" / "env" / "tune.bin"
    ckpt1_copy = first / "run" / "env" / "tune.bin"
    assert os.lstat(ckpt2_copy).st_ino == os.lstat(ckpt1_copy).st_ino  # bytes really are shared
    before = os.lstat(ckpt2_copy)

    _restore(second, run)
    after = os.lstat(ckpt2_copy)
    assert (after.st_mtime_ns, after.st_ino, after.st_ctime_ns) == (
        before.st_mtime_ns, before.st_ino, before.st_ctime_ns)
    assert os.lstat(ckpt1_copy).st_mtime_ns == t1

    _restore(first, run)
    assert os.lstat(run / "env" / "tune.bin").st_mtime_ns == t1
    assert os.lstat(ckpt1_copy).st_mtime_ns == t1


@posix_only
def test_two_restores_sharing_a_file_each_get_their_mtime(tmp_path):
    run = _plain_run(tmp_path)
    tune = run / "env" / "tune.bin"
    tune.write_bytes(b"Y" * 4096)
    ledger = _Ledger(run / "checkpoints")
    t1 = _set_mtime(tune, T1_NS)
    ledger.save(run, 1)
    t2 = _set_mtime(tune, T2_NS)
    ledger.save(run, 2)
    ckpt1, ckpt2 = _checkpoints_in_order(run)
    object_before = os.lstat(ckpt1 / "run" / "env" / "tune.bin")
    object_snapshot = (object_before.st_ino, object_before.st_mtime_ns, object_before.st_ctime_ns)

    for root, expected in ((ckpt1, t1), (ckpt2, t2), (ckpt1, t1)):
        _restore(root, run)
        assert os.lstat(run / "env" / "tune.bin").st_mtime_ns == expected
        assert (run / "env" / "tune.bin").read_bytes() == b"Y" * 4096
        shared = os.lstat(ckpt1 / "run" / "env" / "tune.bin")
        assert (shared.st_ino, shared.st_mtime_ns, shared.st_ctime_ns) == object_snapshot


@posix_only
def test_restore_without_mtime_record_still_works(tmp_path):
    run = _plain_run(tmp_path)
    tune = run / "env" / "tune.bin"
    tune.write_bytes(b"Z" * 4096)
    ledger = _Ledger(run / "checkpoints")
    _set_mtime(tune, T1_NS)
    ledger.save(run, 1)
    newest = checkpoint.latest(run)
    stored_mtime = os.lstat(newest / "run" / "env" / "tune.bin").st_mtime_ns
    meta, _whole = checkpoint._validate(newest)
    meta.pop("file_mtime_ns", None)
    tune.write_bytes(b"drift")

    checkpoint._restore_files(newest, run, meta)

    assert (run / "env" / "tune.bin").read_bytes() == b"Z" * 4096
    assert os.lstat(run / "env" / "tune.bin").st_mtime_ns == stored_mtime  # old behavior: no utime


@posix_only
@pytest.mark.parametrize("record", ["nope", ["env/tune.bin"], {"env/tune.bin": "1"},
                                    {"env/tune.bin": True}, {"env/tune.bin": 1 << 70},
                                    {"../../etc/passwd": 1}, {"/etc/passwd": 1}, {"other/x": 1}])
def test_invalid_mtime_record_does_not_crash_restore(tmp_path, record):
    run = _plain_run(tmp_path)
    tune = run / "env" / "tune.bin"
    tune.write_bytes(b"Z" * 4096)
    _set_mtime(tune, T1_NS)
    _Ledger(run / "checkpoints").save(run, 1)
    newest = checkpoint.latest(run)
    meta, _whole = checkpoint._validate(newest)
    meta["file_mtime_ns"] = record
    tune.write_bytes(b"drift")

    checkpoint._restore_files(newest, run, meta)

    assert (run / "env" / "tune.bin").read_bytes() == b"Z" * 4096


@posix_only
def test_mtime_restore_never_utimes_a_linked_inode(tmp_path):
    # Guard for the case restore itself never produces: a restored file still hard-linked to the
    # stored copy must be replaced by a private inode before its mtime is set.
    run = _plain_run(tmp_path)
    tune = run / "env" / "tune.bin"
    tune.write_bytes(b"Q" * 4096)
    _set_mtime(tune, T1_NS)
    _Ledger(run / "checkpoints").save(run, 1)
    newest = checkpoint.latest(run)
    stored = newest / "run" / "env" / "tune.bin"
    stored_before = os.lstat(stored)
    live = run / "env" / "tune.bin"
    live.unlink()
    os.link(stored, live)
    meta, _whole = checkpoint._validate(newest)
    meta["file_mtime_ns"] = {"env/tune.bin": T2_NS}

    checkpoint._restore_mtimes(newest, run, meta)

    after_live = os.lstat(live)
    assert after_live.st_nlink == 1
    assert not os.path.samestat(after_live, stored_before)
    assert after_live.st_mtime_ns == T2_NS
    assert os.lstat(stored).st_mtime_ns == stored_before.st_mtime_ns
    assert live.read_bytes() == b"Q" * 4096


def _pooled_blob_run(base: Path) -> tuple[Path, Path]:
    """A run whose measurement blobs live in the per-run blob pool (worker/measurement-runs/<run>/blobs)."""
    run = _plain_run(base)
    blobs = run / "worker" / "measurement-runs" / "r1" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "b0.bin").write_bytes(b"P" * 4096)
    return run, blobs / "b0.bin"


@posix_only
def test_pooled_blob_restores_its_own_mtime_per_checkpoint(tmp_path):
    run, blob = _pooled_blob_run(tmp_path)
    ledger = _Ledger(run / "checkpoints")
    t1 = _set_mtime(blob, T1_NS)
    ledger.save(run, 1)
    t2 = _set_mtime(blob, T2_NS)  # same bytes, same pool file: checkpoint 2 reuses the pooled inode
    ledger.save(run, 2)
    ckpt1, ckpt2 = _checkpoints_in_order(run)
    pool_file = ckpt1.parent / checkpoint.POOL_DIR / "worker__measurement-runs__r1__blobs" / "b0.bin"
    pool_before = os.lstat(pool_file)

    live = run / "worker" / "measurement-runs" / "r1" / "blobs" / "b0.bin"
    for root, expected in ((ckpt2, t2), (ckpt1, t1), (ckpt2, t2)):
        _restore(root, run)
        restored = os.lstat(live)
        assert restored.st_mtime_ns == expected, f"pooled blob mtime {restored.st_mtime_ns} != {expected}"
        assert live.read_bytes() == b"P" * 4096
        assert not os.path.samestat(restored, pool_before)  # a private inode, never the pool file
        pool_now = os.lstat(pool_file)
        assert (pool_now.st_ino, pool_now.st_mtime_ns) == (pool_before.st_ino, pool_before.st_mtime_ns)


@posix_only
def test_run_file_fallback_capture_records_mode_and_mtime(tmp_path, monkeypatch):
    source = tmp_path / "payload.json"
    source.write_bytes(b"{}")
    os.chmod(source, 0o640)
    expected_mtime = _set_mtime(source, T1_NS)
    monkeypatch.setattr(checkpoint, "_SAVE", {"copy_modes": {}, "copy_mtimes": {}})

    checkpoint._capture_run_file(source, tmp_path / "out.json", "payload.json")

    assert checkpoint._SAVE["copy_mtimes"] == {"payload.json": expected_mtime}
    assert checkpoint._SAVE["copy_modes"] == {"payload.json": 0o640}
    assert (tmp_path / "out.json").read_bytes() == b"{}"
