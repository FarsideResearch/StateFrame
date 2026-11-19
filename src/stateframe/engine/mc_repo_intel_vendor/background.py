"""Bounded, disposable workers for advisory repository intelligence."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")
_MAX_JOBS = 8
_MAX_RUNNING = 4
_lock = threading.Lock()
_running = 0


def enabled() -> bool:
    return os.environ.get("STATEFRAME_TURN_REPO_INTEL", "off") == "on"


@dataclass
class _Job(Generic[T]):
    running: bool = True
    valid: bool = True
    value: T | None = None


_jobs: OrderedDict[tuple[Path, str], _Job] = OrderedDict()


def invalidate(root: Path) -> None:
    """Fence old results when the host observes a workspace mutation."""
    with _lock:
        for key, job in tuple(_jobs.items()):
            if key[0] == root:
                job.valid = False
                del _jobs[key]


def request(
    root: Path,
    kind: str,
    work: Callable[[Callable[[T], None], Callable[[], bool]], None],
) -> T | None:
    """Get the latest published result and start missing work without joining."""
    global _running
    key = (root, kind)
    with _lock:
        existing = _jobs.get(key)
        if existing is not None:
            _jobs.move_to_end(key)
            return existing.value
        if _running >= _MAX_RUNNING:
            return None
        if len(_jobs) >= _MAX_JOBS:
            finished = next(key for key, job in _jobs.items() if not job.running)
            del _jobs[finished]
        job: _Job[T] = _Job()
        _jobs[key] = job
        _running += 1

    def publish(value: T) -> None:
        with _lock:
            if job.valid:
                job.value = value

    def current() -> bool:
        with _lock:
            return job.valid

    def run() -> None:
        global _running
        try:
            work(publish, current)
        except Exception as error:  # advisory index failures are isolated from the engine
            logger.debug("Background repository index %s failed: %s", kind, type(error).__name__)
        finally:
            with _lock:
                _running -= 1
                job.running = False
                if _jobs.get(key) is job and job.value is None:
                    del _jobs[key]

    try:
        threading.Thread(target=run, name=f"repo-index-{kind}", daemon=True).start()
    except RuntimeError:
        with _lock:
            _running -= 1
            if _jobs.get(key) is job:
                del _jobs[key]
    return None


def request_graph(root: Path, use_lsp: bool = True):
    """Share one producer and mutation fence across repository graph consumers."""
    return request(root, "graph" if use_lsp else "static-graph",
                   partial(_produce_graph, root, use_lsp))


def graph_snapshot(root: Path):
    return snapshot(root, "graph")


def snapshot(root: Path, kind: str):
    """Pin a published result and get a cheap check for its workspace generation."""
    key = (root, kind)
    with _lock:
        job = _jobs.get(key)
        if job is None or job.value is None:
            return None, None
        value = job.value

    def current() -> bool:
        with _lock:
            return _jobs.get(key) is job

    return value, current


def _produce_graph(root: Path, use_lsp: bool, publish, current) -> None:
    # Import in the worker, including optional parser and graph libraries.
    from .call_graph import produce_graph

    produce_graph(root, use_lsp, publish, current)


def cache_file(root: Path, name: str) -> Path:
    """Resolve a host-owned cache outside the disposable checkout."""
    from .call_graph import repo_cache_file

    directory = os.environ.get("REPO_INTEL_CACHE_DIR")
    if directory:
        path = Path(directory).resolve()
        if not path.is_relative_to(root):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            return path / name
    return repo_cache_file(root, name)


def indexed_hashes(root: Path, paths: set[str]) -> dict[str, str]:
    """Use the workspace tree's blob digests when that index is available."""
    try:
        from kernel_workspace_index import WorkspaceIndex
        from kernel_workspace_tree import File
    except ImportError:
        return {}
    allowed = {tuple(os.fsencode(part) for part in Path(path).parts) for path in paths}
    prefixes = {path[:length] for path in allowed for length in range(1, len(path) + 1)}
    try:
        index = WorkspaceIndex(root)
        index.track_rows = True
        index.refresh(select=lambda parts, directory: parts in (prefixes if directory else allowed))
    except (OSError, ValueError):
        return {}
    return {
        os.fsdecode(b"/".join(parts)): row.entry.sha256.hex()
        for parts, row in index.rows.items()
        if parts in allowed and isinstance(row.entry, File)
    }
