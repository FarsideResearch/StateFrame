"""Unit tests for the shared stat-identity digest memo (kernel_file_digest_memo)."""
from __future__ import annotations

import builtins
import hashlib
import io
import os
import time
from pathlib import Path

import pytest

memo = pytest.importorskip("kernel_file_digest_memo")
import kernel_live_projection as projection


@pytest.fixture(autouse=True)
def _clean_memo():
    memo.clear()
    yield
    memo.clear()


@pytest.fixture
def opens(monkeypatch):
    """Record every file open of a path during the test."""
    seen: list[str] = []
    real_open = io.open

    def spy(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            seen.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    return seen


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_fresh_file_is_racy_and_reread_every_call(tmp_path, opens):
    target = tmp_path / "fresh.txt"
    target.write_bytes(b"one")
    path = str(target)
    opens.clear()  # drop the write's own open
    assert memo.file_sha256(path) == _sha(b"one")
    assert memo.file_sha256(path) == _sha(b"one")
    assert opens.count(path) == 2


def test_settled_file_is_served_from_memo(tmp_path, opens):
    target = tmp_path / "settled.txt"
    target.write_bytes(b"settled")
    path = str(target)
    time.sleep(2.1)
    assert memo.file_sha256(path) == _sha(b"settled")
    opens.clear()
    assert memo.file_sha256(path) == _sha(b"settled")
    assert opens == []


def test_in_place_edit_with_restored_mtime_is_detected(tmp_path):
    target = tmp_path / "edit.txt"
    target.write_bytes(b"AAAA")
    path = str(target)
    original = os.stat(path)
    time.sleep(2.1)
    assert memo.file_sha256(path) == _sha(b"AAAA")
    target.write_bytes(b"BBBB")  # same size
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert memo.file_sha256(path) == _sha(b"BBBB")


def test_rename_over_path_is_detected_by_inode(tmp_path):
    target = tmp_path / "swap.txt"
    target.write_bytes(b"old-content")
    path = str(target)
    time.sleep(2.1)
    assert memo.file_sha256(path) == _sha(b"old-content")
    staged = tmp_path / "staged.tmp"
    staged.write_bytes(b"new-content")
    os.replace(staged, path)
    assert memo.file_sha256(path) == _sha(b"new-content")


def test_missing_and_non_regular_return_none(tmp_path):
    assert memo.file_sha256(str(tmp_path / "absent")) is None
    assert memo.file_sha256(str(tmp_path)) is None


_NO_SIGNATURE: memo.Signature = (0, 0, 0, 0, 0)


def test_seed_refuses_racy_file_and_stores_nothing(tmp_path, opens):
    target = tmp_path / "racy.txt"
    target.write_bytes(b"seeded")
    path = str(target)
    signature = memo.stat_signature(os.lstat(path))
    assert memo.seed_verified_digest(path, _sha(b"seeded"), signature) is False
    opens.clear()
    assert memo.file_sha256(path) == _sha(b"seeded")
    assert opens.count(path) == 1  # not memoized, so it was read


def test_seed_stores_settled_file_so_next_read_is_skipped(tmp_path, opens):
    target = tmp_path / "old.txt"
    target.write_bytes(b"seeded-old")
    path = str(target)
    time.sleep(2.1)
    signature = memo.stat_signature(os.lstat(path))
    assert memo.seed_verified_digest(path, _sha(b"seeded-old"), signature) is True
    opens.clear()
    assert memo.file_sha256(path) == _sha(b"seeded-old")
    assert opens == []


def test_seed_with_signature_taken_before_later_write_stores_nothing(tmp_path, opens):
    # Caller hashed the old bytes after taking the signature, then the file changed.
    # Seeding the old digest under the new identity would serve stale content.
    target = tmp_path / "late.txt"
    target.write_bytes(b"old-bytes")
    path = str(target)
    time.sleep(2.1)
    signature_before_read = memo.stat_signature(os.lstat(path))
    old_digest = _sha(b"old-bytes")
    target.write_bytes(b"new-bytes-longer")
    time.sleep(2.1)  # settle the write so only the signature mismatch can refuse the seed
    assert memo.seed_verified_digest(path, old_digest, signature_before_read) is False
    opens.clear()
    assert memo.file_sha256(path) == _sha(b"new-bytes-longer")
    assert opens.count(path) == 1


def test_seed_refusal_drops_existing_memo_entry(tmp_path, opens):
    target = tmp_path / "drop.txt"
    target.write_bytes(b"first")
    path = str(target)
    time.sleep(2.1)
    assert memo.file_sha256(path) == _sha(b"first")
    assert path in memo._MEMO
    stale_signature = memo.stat_signature(os.lstat(path))
    target.write_bytes(b"second!")
    time.sleep(2.1)
    assert memo.seed_verified_digest(path, _sha(b"first"), stale_signature) is False
    assert path not in memo._MEMO
    assert memo.file_sha256(path) == _sha(b"second!")


def test_seed_racy_signature_refuses_and_drops_existing_entry(tmp_path):
    target = tmp_path / "racy-drop.txt"
    target.write_bytes(b"settled")
    path = str(target)
    time.sleep(2.1)
    assert memo.file_sha256(path) == _sha(b"settled")
    assert path in memo._MEMO
    # Touching mtime to now keeps the signature matching the caller's but makes it racy.
    os.utime(path)
    racy_signature = memo.stat_signature(os.lstat(path))
    assert memo.seed_verified_digest(path, _sha(b"settled"), racy_signature) is False
    assert path not in memo._MEMO


@pytest.mark.parametrize("bad", ["", "sha256:xyz", "md5:" + "a" * 32, "sha256:" + "g" * 64])
def test_seed_rejects_malformed_digest(tmp_path, bad):
    target = tmp_path / "bad.txt"
    target.write_bytes(b"x")
    time.sleep(2.1)
    signature = memo.stat_signature(os.lstat(str(target)))
    assert memo.seed_verified_digest(str(target), bad, signature) is False


def test_seed_rejects_missing_and_non_regular(tmp_path):
    good = "sha256:" + "0" * 64
    assert memo.seed_verified_digest(str(tmp_path / "absent"), good, _NO_SIGNATURE) is False
    assert memo.seed_verified_digest(str(tmp_path), good, memo.stat_signature(os.lstat(tmp_path))) is False


def test_live_projection_file_value_delegates_to_shared_memo(tmp_path):
    target = tmp_path / "proj.txt"
    target.write_bytes(b"projection")
    time.sleep(2.1)
    assert projection._file_value(str(target)) == _sha(b"projection")
    assert memo._MEMO[str(target)][1] == _sha(b"projection")


def test_seed_rechecks_bytes_when_caller_read_started_in_racy_tick(tmp_path, monkeypatch):
    target = tmp_path / "seed-race.py"
    target.write_bytes(b"old")
    frozen = os.lstat(target)
    signature = memo.stat_signature(frozen)
    old_digest = _sha(b"old")
    target.write_bytes(b"new")
    real_lstat = os.lstat
    monkeypatch.setattr(os, "lstat", lambda path, *a, **kw:
                        frozen if str(path) == str(target) else real_lstat(path, *a, **kw))
    monkeypatch.setattr(time, "time_ns", lambda: max(signature[-2:]) + 3_000_000_000)
    assert memo.seed_verified_digest(str(target), old_digest, signature) is False
    assert memo.file_sha256(str(target)) == _sha(b"new")


@pytest.mark.parametrize("missing", [True, False])
def test_failed_lookup_evicts_identity_before_reuse(tmp_path, monkeypatch, missing):
    target = tmp_path / "reuse.py"
    target.write_bytes(b"old")
    frozen = os.lstat(target)
    real_lstat = os.lstat
    monkeypatch.setattr(time, "time_ns", lambda: max(frozen.st_mtime_ns, frozen.st_ctime_ns) + 3_000_000_000)
    assert memo.file_sha256(str(target)) == _sha(b"old")
    def unavailable(path, *args, **kwargs):
        if str(path) == str(target):
            if missing:
                raise FileNotFoundError(path)
            return real_lstat(tmp_path)
        return real_lstat(path, *args, **kwargs)
    monkeypatch.setattr(os, "lstat", unavailable)
    assert memo.file_sha256(str(target)) is None
    target.write_bytes(b"new")
    monkeypatch.setattr(os, "lstat", lambda path, *a, **kw:
                        frozen if str(path) == str(target) else real_lstat(path, *a, **kw))
    assert memo.file_sha256(str(target)) == _sha(b"new")


def test_clock_rollback_invalidates_other_settled_entries(tmp_path, monkeypatch):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_bytes(b"old")
    second.write_bytes(b"old")
    frozen = {str(path): os.lstat(path) for path in (first, second)}
    stamp = max(stat.st_ctime_ns for stat in frozen.values())
    now = [stamp + 10_000_000_000]
    real_lstat = os.lstat
    monkeypatch.setattr(os, "lstat", lambda path, *args, **kwargs:
                        frozen[str(path)] if str(path) in frozen else real_lstat(path, *args, **kwargs))
    monkeypatch.setattr(time, "time_ns", lambda: now[0])
    assert memo.file_sha256(str(first)) == _sha(b"old")
    assert memo.file_sha256(str(second)) == _sha(b"old")
    second.write_bytes(b"new")
    # Both entries remain outside the racy window: rollback must clear all paths.
    now[0] = stamp + 5_000_000_000
    assert memo.file_sha256(str(first)) == _sha(b"old")
    assert str(second) not in memo._MEMO
    now[0] = stamp + 11_000_000_000
    assert memo.file_sha256(str(second)) == _sha(b"new")
