"""Checkpoint save cost: scandir snapshot walk, content-addressed object store, one save per send."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint

posix_only = pytest.mark.skipif(os.name != "posix", reason="inode sharing requires POSIX")


def _tree(root: Path, directories: int = 4, files: int = 5) -> None:
    for d in range(directories):
        (root / f"d{d}").mkdir(parents=True)
        for f in range(files):
            (root / f"d{d}" / f"f{f}.py").write_bytes(f"{d}-{f}\n".encode() * 50)


@posix_only
def test_unchanged_snapshot_costs_one_lstat_per_file(tmp_path, monkeypatch):
    run = tmp_path / "env"
    _tree(run)
    first, second = tmp_path / "first", tmp_path / "second"
    # ctime cannot be aged in a test, so the racy-timestamp window is lifted here; that guard has its
    # own test in test_checkpoint_disk_reuse.py.
    monkeypatch.setattr(checkpoint, "_RACY_WINDOW_NS", 0)
    checkpoint._copy_tree(run, first, None, set())
    calls = []
    real = os.lstat
    monkeypatch.setattr(os, "lstat", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    counts = checkpoint._copy_tree(run, second, first, set())
    assert counts == {"linked": 20, "copied": 0, "cloned": 0}
    assert len(calls) <= 20  # was 3 per file: lstat, islink, previous copy


@posix_only
def test_changed_file_is_stored_once_in_the_object_store(tmp_path):
    run, objects = tmp_path / "env", tmp_path / "objects"
    _tree(run)
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set(), objects=objects)
    # The first snapshot fills the store: every distinct file content is stored once.
    assert len(list(objects.iterdir())) == 20
    target = run / "d0" / "f0.py"
    target.write_bytes(b"changed\n" * 10)
    counts = checkpoint._copy_tree(run, second, first, set(), objects=objects)
    assert counts["copied"] == 1 and counts["linked"] == 19
    stored = list(objects.iterdir())
    assert len(stored) == 21
    assert (second / "d0" / "f0.py").stat().st_ino == next(
        item for item in stored if item.stat().st_size == len(b"changed\n" * 10)).stat().st_ino
    assert (second / "d0" / "f0.py").read_bytes() == b"changed\n" * 10
    assert (second / "d0" / "f0.py").stat().st_mtime_ns == target.stat().st_mtime_ns


@posix_only
def test_object_key_covers_content_and_mode_not_mtime(tmp_path):
    """Stale-value guard: a file differing in bytes or mode never reuses an object; an mtime-only change does."""
    run, objects = tmp_path / "env", tmp_path / "objects"
    run.mkdir()
    base = tmp_path / "base"
    checkpoint._copy_tree(run, base, None, set())
    target = run / "a.txt"
    seen = []
    for index, (data, mode, mtime) in enumerate([
            (b"aaaa", 0o644, 1_000_000_000), (b"bbbb", 0o644, 1_000_000_000),
            (b"aaaa", 0o755, 1_000_000_000), (b"aaaa", 0o644, 2_000_000_000)]):
        target.write_bytes(data)
        target.chmod(mode)
        os.utime(target, ns=(mtime, mtime))
        out = tmp_path / f"out{index}"
        checkpoint._copy_tree(run, out, base, set(), objects=objects)
        copy = (out / "a.txt").stat()
        assert (out / "a.txt").read_bytes() == data
        assert (copy.st_mode & 0o777) == mode & ~0o222  # stored read-only
        seen.append(copy.st_ino)
    # the mtime-only change (index 3) shares the object of the first capture of these bytes and mode
    assert seen[3] == seen[0]
    assert len(set(seen)) == 3
    assert len(list(objects.iterdir())) == 3


@posix_only
def test_prune_objects_keeps_linked_and_drops_orphans_and_temporaries(tmp_path):
    root = tmp_path / "checkpoints"
    objects = root / checkpoint.OBJECT_DIR
    objects.mkdir(parents=True)
    kept, orphan = objects / ("a" * 64 + "-600"), objects / ("b" * 64 + "-644")
    kept.write_text("k")
    orphan.write_text("o")
    os.link(kept, root / "snapshot-link")
    (objects / ".123-1.tmp").write_text("partial")
    checkpoint._prune_objects(root)
    assert sorted(p.name for p in objects.iterdir()) == [kept.name]


def test_save_point_saves_at_most_once_per_send(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    saves = []
    sends = {"n": 0}
    before = checkpoint._STATE.copy()
    monkeypatch.setattr(checkpoint, "mode", lambda: "save")
    monkeypatch.setattr(checkpoint, "_every", lambda: 1)
    monkeypatch.setattr(checkpoint, "_sends", lambda _run: sends["n"])
    monkeypatch.setattr(checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(checkpoint, "_low_disk_space", lambda _run: None)
    monkeypatch.setattr(checkpoint, "save",
                        lambda _f, step, _r, background=False: saves.append(step) or run / f"{step:04d}-1")
    try:
        checkpoint._STATE.update(disabled=None, restored_step=None, last_sends=None, saved=0)
        poc = run / "poc"
        for step, n in [(1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (7, 3)]:
            sends["n"] = n
            checkpoint.save_point(None, step, poc)
        assert saves == [2, 5, 7]  # boundaries without a new model send never save
    finally:
        checkpoint._STATE.clear()
        checkpoint._STATE.update(before)


# ---------------------------------------------------------------- kill / resume replay

import subprocess  # noqa: E402
import sys  # noqa: E402

CHILD = Path(__file__).resolve().parent / "fixtures" / "checkpoint_kill_child.py"
TURNS = 6


def _child(action: str, run: Path, spec: str = "none") -> subprocess.CompletedProcess:
    # The child saves outside a runner, so it has no spec.json identity record to verify.
    env = dict(os.environ, STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE="1", PYTHONPATH=os.pathsep.join(
        [str(Path(checkpoint.__file__).parents[2]), str(CHILD.parent.parent)]))
    return subprocess.run([sys.executable, str(CHILD), action, str(run), str(TURNS), spec],
                          env=env, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def control_digest(tmp_path_factory) -> str:
    done = _child("run", tmp_path_factory.mktemp("control") / "run")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@posix_only
@pytest.mark.parametrize("spec", ["mid:3", "mid:1", "prepublish:4", "between:2", "between:5"])
def test_sigkill_then_resume_is_byte_identical_to_no_kill_run(tmp_path, control_digest, spec):
    run = tmp_path / "run"
    killed = _child("run", run, spec)
    assert killed.returncode == -9, killed.stderr  # SIGKILL, not a clean exit or a crash
    root = run / "checkpoints"
    pointer = (root / "LATEST").read_text().strip() if (root / "LATEST").exists() else None
    if pointer is not None:  # LATEST only ever names a complete, published checkpoint
        assert (root / pointer / checkpoint.META_FILE).is_file()
    resumed = _child("resume", run)
    if pointer is None:  # killed before the first publish: nothing to resume, replay from scratch
        assert resumed.stdout.strip() == "NOCHECKPOINT"
        resumed = _child("run", tmp_path / "fresh")
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.strip() == control_digest


@posix_only
def test_torn_state_file_is_rejected_not_loaded(tmp_path):
    run = tmp_path / "run"
    assert _child("run", run).returncode == 0
    newest = checkpoint.latest(run)
    state = newest / checkpoint.STATE_FILE
    state.write_bytes(state.read_bytes()[:-7])  # a partially written state image
    resumed = _child("resume", run)
    assert resumed.returncode != 0
    assert "CheckpointError" in resumed.stderr


@posix_only
def test_object_link_limit_falls_back_to_a_private_copy(tmp_path, monkeypatch):
    """Identical tiny files share one object inode; EMLINK must not fail the save."""
    import errno as _errno
    real_link = os.link
    refused = []

    def full_link(source, destination, *args, **kwargs):
        if "objects" in str(source):
            refused.append(destination)
            raise OSError(_errno.EMLINK, "Too many links")
        return real_link(source, destination, *args, **kwargs)

    run, objects = tmp_path / "env", tmp_path / "objects"
    _tree(run)
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set(), objects=objects)
    target = run / "d0" / "f0.py"
    target.write_bytes(b"changed\n")
    monkeypatch.setattr(checkpoint.os, "link", full_link)
    checkpoint._copy_tree(run, second, first, set(), objects=objects)
    assert refused, "the object-store link path was not exercised"
    assert (second / "d0" / "f0.py").read_bytes() == b"changed\n"
    assert (second / "d0" / "f0.py").stat().st_mtime_ns == target.stat().st_mtime_ns


@posix_only
def test_copy_tree_applies_home_exclusions_and_still_dedups_through_object_store(tmp_path):
    run, home = tmp_path / "env", tmp_path / "env" / "home"
    _tree(run, directories=1, files=2)
    home.mkdir()
    (home / "identity.key").write_bytes(b"installation-key-material")
    (run / "link").symlink_to(home / "identity.key")
    objects = tmp_path / "objects"
    first, second = tmp_path / "first", tmp_path / "second"
    checkpoint._copy_tree(run, first, None, set(), objects=objects, home=home,
                          home_subtree=home, excluded=set())
    checkpoint._copy_tree(run, second, first, set(), objects=objects, home=home,
                          home_subtree=home, excluded=set())
    for snapshot in (first, second):
        assert not (snapshot / "home").exists()  # boundary filter wins over the storage path
        assert not (snapshot / "link").exists() and not (snapshot / "link").is_symlink()
        assert (snapshot / "d0" / "f0.py").read_bytes() == b"0-0\n" * 50


@posix_only
def test_copy_tree_unchanged_file_is_rescanned_only_after_a_new_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_DIGEST_MEMO", {})
    monkeypatch.setattr(checkpoint, "_RACY_WINDOW_NS", 0)  # ctime of a fresh file cannot be aged here
    run = tmp_path / "env"
    _tree(run, directories=1, files=2)
    first, second, third = tmp_path / "first", tmp_path / "second", tmp_path / "third"
    checkpoint._copy_tree(run, first, None, set())
    scans = []
    real = checkpoint._check_snapshot_file
    monkeypatch.setattr(checkpoint, "_check_snapshot_file", lambda *a, **k: scans.append(a[0]) or real(*a, **k))
    checkpoint._copy_tree(run, second, first, set())
    assert scans == []  # captured clean once under this secret set: linked without another read
    checkpoint.register_secret(b"0-1\n" * 3)  # a secret that appears inside f1.py
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._copy_tree(run, third, second, set())
    assert scans  # the generation bump invalidated the memo, so the file was read again
