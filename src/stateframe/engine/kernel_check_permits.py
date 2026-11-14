"""Crash-released per-run check permits backed by advisory flock files."""
from __future__ import annotations

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
def acquire(permit_root: Path, permits: int, *, timeout: float) -> Iterator[int]:
    if permits < 1:
        raise ValueError('check permits must be positive')
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('check permit timeout must be nonnegative')
    root = Path(permit_root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        for number in range(permits):
            descriptor = os.open(root / f'{number}.lock', os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                yield number
                return
            finally:
                os.close(descriptor)
        if time.monotonic() >= deadline:
            raise TimeoutError(f'no check permit available in {root}')
        time.sleep(min(0.05, deadline - time.monotonic()))
