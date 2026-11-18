"""Public-behavior contract for the GreenLock editable-file digest.

Any stat-identity cache in front of _workspace_product_file_digests must return, for every
scenario below, the digest of the bytes on disk right now. Today's function re-hashes every
file, so everything here passes on the uncached code except the read-budget test at the end,
which pins the cache's purpose (a one-file change must not re-read the whole workspace).
"""
from __future__ import annotations

import hashlib
import os
import time
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_green_lock as green_lock

DIGEST = green_lock._workspace_product_file_digests
PKG = "pkg"
NS_PER_S = 1_000_000_000


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _runtime(root: Path) -> SimpleNamespace:
    return SimpleNamespace(owner=SimpleNamespace(root=str(root)))


def _rel(index: int) -> str:
    return f"{PKG}/f{index:04d}.py"


def _write(root: Path, rel: str, data: bytes) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _reference(root: Path, rel: str) -> str | None:
    """Digest of the file as it is on disk now, or None when absent (today's contract)."""
    target = root / rel
    if not os.path.lexists(target):
        return None
    return _sha(target.read_bytes())


def digests(root: Path, paths: list[str]) -> dict[str, object] | None:
    """Call the function twice so any cache in front of it is exercised on the second call."""
    DIGEST(_runtime(root), paths)
    return DIGEST(_runtime(root), paths)


def _single(root: Path, rel: str) -> object:
    result = digests(root, [rel])
    assert result is not None
    return result[rel]


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def restart(monkeypatch):
    """Clear the shared digest memo and legacy module-local dictionaries."""
    def _reset() -> None:
        shared = getattr(green_lock, "kernel_file_digest_memo", None)
        if shared is not None:
            shared.clear()
        for name in dir(green_lock):
            lowered = name.lower()
            if "cache" not in lowered and "memo" not in lowered:
                continue
            value = getattr(green_lock, name)
            if isinstance(value, dict):
                monkeypatch.setitem(vars(green_lock), name, {})
    return _reset


# Edge cases -----------------------------------------------------------------

def test_same_size_rewrite_with_restored_mtime_reflects_new_bytes(ws):
    rel = _rel(1)
    _write(ws, rel, b"AAAAAAAA")
    stamp = 1_700_000_000 * NS_PER_S
    os.utime(ws / rel, ns=(stamp, stamp))
    assert _single(ws, rel) == _sha(b"AAAAAAAA")

    _write(ws, rel, b"BBBBBBBB")
    os.utime(ws / rel, ns=(stamp, stamp))
    assert _single(ws, rel) == _sha(b"BBBBBBBB")


def test_rewrite_inside_racy_window_same_size_reflects_new_bytes(ws):
    rel = _rel(2)
    _write(ws, rel, b"first-bytes")
    assert _single(ws, rel) == _sha(b"first-bytes")

    # Both writes land in the same second with an identical size and a fresh mtime.
    _write(ws, rel, b"other-bytz")
    assert _single(ws, rel) == _sha(b"other-bytz")


def test_rename_over_same_size_same_mtime_reflects_new_bytes(ws):
    rel = _rel(3)
    stamp = 1_700_000_100 * NS_PER_S
    _write(ws, rel, b"old-content")
    os.utime(ws / rel, ns=(stamp, stamp))
    assert _single(ws, rel) == _sha(b"old-content")

    staged = ws / PKG / "staged.tmp"
    staged.write_bytes(b"new-content")
    os.utime(staged, ns=(stamp, stamp))
    os.replace(staged, ws / rel)
    assert _single(ws, rel) == _sha(b"new-content")


def test_touch_only_mtime_change_keeps_same_digest(ws):
    rel = _rel(4)
    _write(ws, rel, b"steady bytes")
    before = _single(ws, rel)
    os.utime(ws / rel, ns=(1_600_000_000 * NS_PER_S, 1_600_000_000 * NS_PER_S))
    assert _single(ws, rel) == before == _sha(b"steady bytes")


def test_chmod_only_keeps_bytes_digest(ws):
    # Today's digest is content-only, so mode bits do not move it. A mode-aware cache must
    # still return this value because the bytes did not change.
    rel = _rel(5)
    _write(ws, rel, b"mode test")
    before = _single(ws, rel)
    os.chmod(ws / rel, 0o755)
    assert _single(ws, rel) == before == _sha(b"mode test")


def test_symlink_in_place_of_file_refuses_without_stale_digest(ws):
    rel = _rel(6)
    _write(ws, rel, b"real file")
    assert _single(ws, rel) == _sha(b"real file")

    target = ws / PKG / "elsewhere.py"
    target.write_bytes(b"real file")
    (ws / rel).unlink()
    os.symlink(target, ws / rel)
    assert digests(ws, [rel]) is None


def test_delete_then_recreate_reflects_missing_then_new_bytes(ws):
    rel = _rel(7)
    _write(ws, rel, b"before delete")
    assert _single(ws, rel) == _sha(b"before delete")

    (ws / rel).unlink()
    assert _single(ws, rel) is None

    _write(ws, rel, b"after recreate")
    assert _single(ws, rel) == _sha(b"after recreate")


def test_case_only_rename_is_a_different_path(ws):
    upper = f"{PKG}/Mixed.py"
    lower = f"{PKG}/mixed.py"
    _write(ws, upper, b"case bytes")
    if os.path.exists(ws / lower):
        pytest.skip("filesystem is case-insensitive; case-only rename is not observable")
    assert _single(ws, upper) == _sha(b"case bytes")

    os.rename(ws / upper, ws / lower)
    assert _single(ws, upper) is None
    assert _single(ws, lower) == _sha(b"case bytes")


@pytest.mark.parametrize("mutate_before_read", [True, False])
def test_file_modified_during_hash_never_records_stale_bytes_for_new_stat(
        ws, monkeypatch, mutate_before_read):
    rel = _rel(8)
    old, new = b"old-old-old", b"new-new-new"
    _write(ws, rel, old)
    target = (ws / rel).resolve()
    real_read = Path.read_bytes
    fired: list[bool] = []

    def racing_read(self: Path) -> bytes:
        if self.resolve() != target or fired:
            return real_read(self)
        fired.append(True)  # the racing writer fires exactly once
        if mutate_before_read:
            target.write_bytes(new)
            return real_read(self)
        data = real_read(self)
        target.write_bytes(new)
        return data

    monkeypatch.setattr(Path, "read_bytes", racing_read)
    racing = DIGEST(_runtime(ws), [rel])
    assert fired, "racing writer never ran; the scenario did not exercise the read"

    allowed = {None, _sha(old), _sha(new)} if not mutate_before_read else {None, _sha(new)}
    assert racing is None or racing[rel] in allowed
    # The next call sees the settled bytes; it must not carry the pre-write digest forward.
    assert _single(ws, rel) == _sha(new)


def test_concurrent_writer_thread_never_leaves_a_stale_digest(ws):
    rel = _rel(9)
    _write(ws, rel, b"writer-0000")
    stop = threading.Event()

    def writer() -> None:
        index = 0
        while not stop.is_set():
            index += 1
            _write(ws, rel, f"writer-{index:04d}".encode())

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(50):
            DIGEST(_runtime(ws), [rel])
    finally:
        stop.set()
        thread.join()
    final = _reference(ws, rel)
    assert _single(ws, rel) == final


def test_ctime_only_change_keeps_bytes_digest(ws):
    # chmod to the current mode and add/remove a hardlink both bump ctime with no byte change.
    rel = _rel(10)
    _write(ws, rel, b"ctime bytes")
    before = _single(ws, rel)
    mode = (ws / rel).stat().st_mode & 0o777
    os.chmod(ws / rel, mode)
    extra = ws / PKG / "ctime_link.py"
    os.link(ws / rel, extra)
    os.unlink(extra)
    assert _single(ws, rel) == before == _sha(b"ctime bytes")


def test_inode_reuse_with_same_size_and_mtime_reflects_new_bytes(ws):
    rel = _rel(11)
    stamp = 1_700_000_200 * NS_PER_S
    _write(ws, rel, b"reuse-one")
    os.utime(ws / rel, ns=(stamp, stamp))
    assert _single(ws, rel) == _sha(b"reuse-one")

    (ws / rel).unlink()
    _write(ws, rel, b"reuse-two")
    os.utime(ws / rel, ns=(stamp, stamp))
    # The fs may or may not reuse the freed inode; either way the digest must follow the bytes.
    assert _single(ws, rel) == _sha(b"reuse-two")


def test_hardlinked_paths_both_reflect_write_through_one_path(ws):
    first, second = _rel(12), f"{PKG}/f0012_alias.py"
    _write(ws, first, b"linked-old")
    os.link(ws / first, ws / second)
    assert (ws / first).stat().st_ino == (ws / second).stat().st_ino
    result = digests(ws, [first, second])
    assert result is not None and result[first] == result[second] == _sha(b"linked-old")

    (ws / first).write_bytes(b"linked-new")
    result = digests(ws, [first, second])
    assert result is not None
    assert result[first] == result[second] == _sha(b"linked-new")


def test_future_mtime_with_same_size_rewrite_reflects_new_bytes(ws):
    rel = _rel(13)
    future = int((os.stat(ws).st_mtime + 3600) * NS_PER_S)
    _write(ws, rel, b"skew-aaaa")
    os.utime(ws / rel, ns=(future, future))
    assert _single(ws, rel) == _sha(b"skew-aaaa")

    _write(ws, rel, b"skew-bbbb")
    os.utime(ws / rel, ns=(future, future))
    assert _single(ws, rel) == _sha(b"skew-bbbb")


# Verdict parity -------------------------------------------------------------

PARITY_FILES = 50
PARITY_STEPS = 200


def _snapshot(root: Path, paths: list[str]) -> dict[str, object]:
    result = DIGEST(_runtime(root), paths)
    assert result is not None
    return dict(result)


def test_randomized_edit_sequence_matches_reference_at_every_step(ws, restart):
    rng = random.Random(20260721)
    paths = [_rel(index) for index in range(PARITY_FILES)]
    for rel in paths:
        if rng.random() < 0.8:
            _write(ws, rel, bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 64))))
    stamp = 1_700_000_300 * NS_PER_S

    for step in range(PARITY_STEPS):
        present = [rel for rel in paths if os.path.lexists(ws / rel)]
        absent = [rel for rel in paths if not os.path.lexists(ws / rel)]
        action = rng.choice(["write", "same_size", "rename", "delete", "touch"])
        if action == "write" or (action in {"same_size", "touch", "delete", "rename"} and not present):
            rel = rng.choice(paths)
            size = rng.randint(1, 64)
            _write(ws, rel, bytes(rng.getrandbits(8) for _ in range(size)))
        elif action == "same_size":
            rel = rng.choice(present)
            size = (ws / rel).stat().st_size
            _write(ws, rel, bytes(rng.getrandbits(8) for _ in range(size)))
            os.utime(ws / rel, ns=(stamp, stamp))
        elif action == "rename" and absent:
            src, dst = rng.choice(present), rng.choice(absent)
            os.replace(ws / src, ws / dst)
        elif action == "delete":
            (ws / rng.choice(present)).unlink()
        else:
            rel = rng.choice(present)
            touched = rng.randint(1_600_000_000, 1_750_000_000) * NS_PER_S
            os.utime(ws / rel, ns=(touched, touched))

        if step % 37 == 0:
            restart()
        expected = {rel: _reference(ws, rel) for rel in paths}
        assert _snapshot(ws, paths) == expected, f"parity break at step {step} ({action})"
        assert _snapshot(ws, paths) == expected, f"second-call parity break at step {step}"


@pytest.mark.parametrize("bad_kind", ["symlink", "parent_is_file"])
def test_one_bad_path_refuses_the_whole_set(ws, tmp_path, bad_kind):
    # A cache must never return a partial dict for a set that contains one unhashable path.
    good, bad = _rel(20), f"{PKG}/bad_{bad_kind}.py"
    _write(ws, good, b"good bytes")
    if bad_kind == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(b"outside bytes")
        os.symlink(outside, ws / bad)
    else:
        (ws / PKG / "not_a_dir").write_bytes(b"plain file")
        bad = f"{PKG}/not_a_dir/child.py"

    assert digests(ws, [good, bad]) is None
    # A good-only call in between must not change what the bad set returns.
    assert _snapshot(ws, [good]) == {good: _sha(b"good bytes")}
    assert digests(ws, [good, bad]) is None
    assert digests(ws, [bad, good]) is None


# Read budget (RED today: the uncached function re-reads every file) ---------

BUDGET_FILES = 3500


def test_one_file_change_rereads_at_most_five_files_on_second_call(ws, monkeypatch):
    paths = [f"src/mod_{index:05d}.py" for index in range(BUDGET_FILES)]
    for rel in paths:
        _write(ws, rel, f"print({rel!r})\n".encode())
    # Files changed within the racy window (2 s of mtime/ctime) are never memoized, and ctime cannot be
    # backdated, so let the tree settle before the first call; otherwise the cache correctly re-reads everything.
    time.sleep(2.1)
    DIGEST(_runtime(ws), paths)

    _write(ws, paths[1234], b"print('changed')\n")

    reads: list[Path] = []
    real_read = Path.read_bytes
    root_prefix = str(ws.resolve()) + os.sep

    def counting_read(self: Path) -> bytes:
        if str(self.resolve()).startswith(root_prefix):
            reads.append(self)
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read)
    result = DIGEST(_runtime(ws), paths)
    monkeypatch.undo()

    assert result is not None and result[paths[1234]] == _sha(b"print('changed')\n")
    assert len(reads) <= 5, f"second call read {len(reads)} files, budget is 5"
