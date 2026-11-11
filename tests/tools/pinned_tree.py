"""Materialize a pinned source tree from this repository's git objects."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from tools.sandbox_temp import sandbox_visible_temp_dir, shadowed_by_bwrap_dev

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,40}\Z")
_SESSION_TREES: dict[tuple[str, str], Path] = {}
_SESSION_DISK_TEMPS: dict[tuple[str, str], tempfile.TemporaryDirectory[str]] = {}


def _archive_destination(tmp_path_factory, full_commit: str,
                         session_key: tuple[str, str]) -> Path:
    # bwrap's --dev mount hides /dev/shm, including a pinned tree's engine supervisor.
    # A disk tree remains visible to the supervisor while pytest may keep its fast basetemp.
    if shadowed_by_bwrap_dev(Path(tmp_path_factory.getbasetemp())):
        temporary = sandbox_visible_temp_dir()
        _SESSION_DISK_TEMPS[session_key] = temporary
        return Path(temporary.name)
    return tmp_path_factory.mktemp(f"pinned-tree-{full_commit[:12]}")


def pinned_tree(commit: str, tmp_path_factory) -> Path:
    """Extract ``git archive`` for ``commit``, reusing it within this pytest session."""
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(f"pinned commit must be a hexadecimal Git object id: {commit!r}")
    if shutil.which("git") is None:
        pytest.skip(f"git is unavailable, so pinned commit {commit} cannot be materialized")

    resolved = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        pytest.skip(f"pinned commit {commit} is not available in this repository's git objects")
    full_commit = resolved.stdout.strip()
    session_key = (str(Path(tmp_path_factory.getbasetemp()).resolve()), full_commit)
    cached = _SESSION_TREES.get(session_key)
    if cached is not None:
        return cached

    archive = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), "archive", "--format=tar", full_commit],
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        pytest.fail(
            f"could not archive pinned commit {commit}: {archive.stderr.decode(errors='replace')}"
        )

    destination = _archive_destination(tmp_path_factory, full_commit, session_key)
    extracted = subprocess.run(
        ["tar", "-xf", "-", "-C", str(destination)],
        input=archive.stdout,
        capture_output=True,
        check=False,
    )
    if extracted.returncode != 0:
        pytest.fail(
            f"could not extract pinned commit {commit}: {extracted.stderr.decode(errors='replace')}"
        )

    _SESSION_TREES[session_key] = destination
    return destination
