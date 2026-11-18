"""Per-process file digest memo keyed by lstat identity.

Single shared stat-identity index for "sha256:<hex>" file digests.  A cached
digest is reused only while the path's lstat signature (inode, device, size,
mtime_ns, ctime_ns) is unchanged.  ctime moves on every write even when a tool
restores mtime, so an in-place edit with a preserved mtime still forces a rehash.

Git's racy-timestamp rule applies: a file whose mtime or ctime is within
RACY_NS of now is never memoized, because a second write inside the same
timestamp tick would leave the signature identical.  An entry is stored only
when the signature is unchanged across the read.
"""
from __future__ import annotations

import hashlib
import os
import stat
import time
from pathlib import Path

Signature = tuple[int, int, int, int, int]

RACY_NS = 2_000_000_000
MAX_ENTRIES = 200_000

_MEMO: dict[str, tuple[Signature, str]] = {}
_LAST_CLOCK_NS: int | None = None


def stat_signature(result: os.stat_result) -> Signature:
    """Identity tuple used to decide whether a cached digest is still valid."""
    return (result.st_ino, result.st_dev, result.st_size, result.st_mtime_ns, result.st_ctime_ns)


def _is_racy(result: os.stat_result, now_ns: int) -> bool:
    return now_ns - max(result.st_mtime_ns, result.st_ctime_ns) <= RACY_NS


def _store(path: str, signature: Signature, value: str) -> None:
    if len(_MEMO) >= MAX_ENTRIES:
        _MEMO.clear()
    _MEMO[path] = (signature, value)


def _observe_clock() -> int:
    """Discard observations from a later wall-clock epoch after rollback."""
    global _LAST_CLOCK_NS
    now_ns = time.time_ns()
    if _LAST_CLOCK_NS is not None and now_ns < _LAST_CLOCK_NS:
        _MEMO.clear()
    _LAST_CLOCK_NS = now_ns
    return now_ns


def file_sha256(path: str) -> str | None:
    """Return "sha256:<hex>" for a regular file, or None if missing or not regular.

    Raises OSError if reading fails or identity changes; callers fail closed.
    """
    started_ns = _observe_clock()
    try:
        before = os.lstat(path)
    except OSError:
        _MEMO.pop(path, None)
        return None
    if not stat.S_ISREG(before.st_mode):
        _MEMO.pop(path, None)
        return None
    signature = stat_signature(before)
    racy = _is_racy(before, started_ns)
    memo = _MEMO.get(path)
    if memo is not None and memo[0] == signature and not racy:
        return memo[1]
    # Evict before I/O, including failed reads and clock rollback into the racy window.
    _MEMO.pop(path, None)
    # Keep reads observable at the same seam used by callers and profiling tests.
    value = "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
    after = os.lstat(path)
    if not stat.S_ISREG(after.st_mode) or stat_signature(after) != signature:
        # Returning these bytes would let a view bind them to a newer identity.
        raise OSError("file changed while hashing")
    finished_ns = _observe_clock()
    if not racy and finished_ns >= started_ns and not _is_racy(after, finished_ns):
        _store(path, signature, value)
    return value


def _is_sha256_digest(value: str) -> bool:
    hex_part = value[len("sha256:"):] if value.startswith("sha256:") else ""
    if len(hex_part) != 64:
        return False
    try:
        bytes.fromhex(hex_part)
    except ValueError:
        return False
    return True


def seed(abs_path: str, digest: str, signature: Signature) -> bool:
    """Validate and record a caller's digest and pre-read signature.

    A matching signature at seed time cannot prove that the caller started its
    read outside a coarse timestamp tick. Re-read independently before accepting
    the digest; this deliberately trades one read for a verifiable seed contract.
    Refused seeds evict any older entry and return False.
    """
    if not _is_sha256_digest(digest):
        _MEMO.pop(abs_path, None)
        return False
    try:
        current = os.lstat(abs_path)
    except OSError:
        _MEMO.pop(abs_path, None)
        return False
    fresh = stat_signature(current)
    if (
        not stat.S_ISREG(current.st_mode)
        or tuple(signature) != fresh
        or _is_racy(current, time.time_ns())
    ):
        _MEMO.pop(abs_path, None)
        return False
    _MEMO.pop(abs_path, None)
    try:
        observed = file_sha256(abs_path)
    except OSError:
        return False
    if observed != digest or _MEMO.get(abs_path) != (fresh, digest):
        _MEMO.pop(abs_path, None)
        return False
    return True


def clear() -> None:
    """Drop every memo entry and clock observation (tests and explicit resets)."""
    global _LAST_CLOCK_NS
    _MEMO.clear()
    _LAST_CLOCK_NS = None
