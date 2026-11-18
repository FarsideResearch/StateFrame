"""Checkpoint copies must capture live bytes; a prior copy is reused only when its content is proven."""
from __future__ import annotations

import hashlib
import pickle
import random
import os
import sys
from pathlib import Path

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint

posix_only = pytest.mark.skipif(os.name != "posix", reason="inode sharing requires POSIX")
MTIME_NS = 1_700_000_000_123_456_789


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_at_fixed_mtime(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    os.utime(path, ns=(MTIME_NS, MTIME_NS))


def _rewrite_same_size_keep_mtime(path: Path, data: bytes) -> None:
    """Rewrite in place (same inode), then restore the old mtime, as a stale-mtime tool would."""
    assert len(data) == path.stat().st_size
    with path.open("r+b") as handle:
        handle.write(data)
    os.utime(path, ns=(MTIME_NS, MTIME_NS))


@posix_only
@pytest.mark.parametrize("use_objects", [False, True], ids=["plain-copy", "object-store"])
def test_equal_size_same_mtime_rewrite_is_captured(tmp_path, use_objects):
    run = tmp_path / "env"
    run.mkdir()
    live = run / "state.txt"
    _write_at_fixed_mtime(live, b"before")
    objects = tmp_path / "objects" if use_objects else None
    first = tmp_path / "first"
    checkpoint._copy_tree(run, first, None, set(), objects=objects)

    _rewrite_same_size_keep_mtime(live, b"after!")
    second = tmp_path / "second"
    counts = checkpoint._copy_tree(run, second, first, set(), objects=objects)

    assert (first / "state.txt").read_bytes() == b"before"
    assert (second / "state.txt").read_bytes() == b"after!"
    assert counts["linked"] == 0


@posix_only
def test_unchanged_file_is_still_linked(tmp_path):
    run = tmp_path / "env"
    run.mkdir()
    _write_at_fixed_mtime(run / "state.txt", b"stable")
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set())

    counts = checkpoint._copy_tree(run, second, first, set())

    assert counts == {"linked": 1, "copied": 0, "cloned": 0}
    assert os.path.samestat(os.lstat(first / "state.txt"), os.lstat(second / "state.txt"))


@posix_only
@pytest.mark.parametrize(("live_bytes", "expected_linked"), [(b"same!", 1), (b"diff!", 0)])
def test_prior_snapshot_without_recorded_hash_is_hashed_not_trusted(
        tmp_path, live_bytes, expected_linked):
    # No record in the prior snapshot: the prior file's own bytes decide, never size+mtime alone.
    run = tmp_path / "env"
    run.mkdir()
    live = run / "state.txt"
    _write_at_fixed_mtime(live, b"same!")
    first = tmp_path / "first"
    checkpoint._copy_tree(run, first, None, set())
    _rewrite_same_size_keep_mtime(live, live_bytes)

    second = tmp_path / "second"
    counts = checkpoint._copy_tree(run, second, first, set())

    assert counts["linked"] == expected_linked
    assert (second / "state.txt").read_bytes() == live_bytes


@posix_only
def test_recorded_digest_must_match_live_bytes_before_linking(tmp_path, monkeypatch):
    run = tmp_path / "env"
    run.mkdir()
    _write_at_fixed_mtime(run / "state.txt", b"stable")
    first = tmp_path / "first"
    checkpoint._copy_tree(run, first, None, set())

    wrong = {"state.txt": _sha256(b"not the bytes")}
    refused: dict[str, str] = {}
    monkeypatch.setitem(checkpoint._SAVE, "copy_digests", (refused, wrong))
    counts = checkpoint._copy_tree(run, tmp_path / "refused", first, set())
    assert counts["linked"] == 0
    assert refused == {"state.txt": _sha256(b"stable")}  # the captured copy is recorded

    right = {"state.txt": _sha256(b"stable")}
    recorded: dict[str, str] = {}
    monkeypatch.setitem(checkpoint._SAVE, "copy_digests", (recorded, right))
    counts = checkpoint._copy_tree(run, tmp_path / "linked", first, set())
    assert counts["linked"] == 1
    assert recorded == {"state.txt": _sha256(b"stable")}


@posix_only
def test_restore_after_stale_rewrite_returns_new_bytes(tmp_path, monkeypatch):
    # Bare run dir without a spec: opt into the legacy direct-engine home check, as the restore tests do.
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"stale-copy-signing-key-0123456789"])
    run = tmp_path / "run"
    env = run / "env"
    env.mkdir(parents=True)
    live = env / "state.txt"
    _write_at_fixed_mtime(live, b"before")
    # The frame/module state is not under test: stub the in-memory capture with a plain pickle so
    # the real save (run-file snapshot, manifest, signature) and restore (file moves) still run.
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    state = {"frame": {}, "modules": {}, "classes": {}, "patched": {}, "participants": {},
             "context": [], "random": random.getstate()}
    monkeypatch.setattr(checkpoint, "collect", lambda _frame: (dict(state), context))
    monkeypatch.setattr(checkpoint, "dump_to",
                        lambda stream, payload, _ctx, protocol=5: pickle.dump(payload, stream))

    first = checkpoint.save(sys._getframe(), 1, run)
    assert (first / "run" / "env" / "state.txt").read_bytes() == b"before"

    _rewrite_same_size_keep_mtime(live, b"after!")
    second = checkpoint.save(sys._getframe(), 2, run)
    assert (second / "run" / "env" / "state.txt").read_bytes() == b"after!"

    # restore() authenticates the snapshot and then restores run files before its payload replay.
    # That replay (context vars, frame state) only makes sense in a fresh process, so the file step
    # is driven directly here with the same two calls restore() makes first.
    _rewrite_same_size_keep_mtime(live, b"later!")
    meta, _whole = checkpoint._validate(second)
    checkpoint._restore_files(second, run, meta)
    assert (run / "env" / "state.txt").read_bytes() == b"after!"


@posix_only
def test_darwin_seed_match_captures_live_bytes(tmp_path, monkeypatch):
    # The darwin seed branch only chooses a clone preference; it must never publish the seed's bytes.
    monkeypatch.setattr(checkpoint.sys, "platform", "darwin")
    monkeypatch.setattr(checkpoint, "_CLONEFILE", None)
    run, seed = tmp_path / "env", tmp_path / "seed"
    run.mkdir()
    seed.mkdir()
    _write_at_fixed_mtime(run / "state.txt", b"after!")
    _write_at_fixed_mtime(seed / "state.txt", b"before")
    target = tmp_path / "target"

    counts = checkpoint._copy_tree(run, target, None, set(), seed=seed)

    assert (target / "state.txt").read_bytes() == b"after!"
    assert counts == {"linked": 0, "copied": 1, "cloned": 0}
