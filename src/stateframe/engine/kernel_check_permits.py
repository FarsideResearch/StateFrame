"""Crash-released per-run check permits backed by advisory flock files."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import time
from typing import Iterator


_installed: tuple[Path, int] | None = None


def install(root: Path, permits: int) -> None:
    global _installed
    if permits < 1:
        raise ValueError('check permits must be positive')
    value = (Path(root), permits)
    if _installed is not None and _installed != value:
        raise RuntimeError('check permits already installed with different settings')
    _installed = value


def active() -> tuple[Path, int] | None:
    return _installed


@contextmanager
def acquire(permit_root: Path, permits: int, *, timeout: float | None = None,
            run_remaining: Callable[[], float | None] | None = None) -> Iterator[int]:
    """Wait for an OS permit, subject to the run budget and process cancellation."""
    if permits < 1:
        raise ValueError('check permits must be positive')
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError('check permit timeout must be nonnegative')
    root = Path(permit_root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout if timeout is not None else math.inf
    while True:
        run_left = run_remaining() if run_remaining is not None else None
        if run_left is not None and run_left <= 0:
            raise TimeoutError("run deadline exhausted while waiting for a check permit")
        for number in range(permits):
            descriptor = os.open(root / f'{number}.lock', os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                run_left = run_remaining() if run_remaining is not None else None
                if run_left is not None and run_left <= 0:
                    raise TimeoutError("run deadline exhausted while acquiring a check permit")
                yield number
                return
            finally:
                os.close(descriptor)
        wait = deadline - time.monotonic()
        if wait <= 0:
            raise TimeoutError(f'no check permit available in {root}')
        time.sleep(min(0.05, wait, run_left if run_left is not None else math.inf))
