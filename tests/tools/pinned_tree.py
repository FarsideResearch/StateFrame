"""Materialize a pinned source tree from this repository's git objects."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from tools.sandbox_temp import sandbox_visible_temp_dir, shadowed_by_bwrap_dev, timeout_scale

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,40}\Z")
_SESSION_TREES: dict[tuple[str, str], Path] = {}
_SESSION_DISK_TEMPS: dict[tuple[str, str], tempfile.TemporaryDirectory[str]] = {}
_LOCAL_COMMAND_TIMEOUT_SECONDS = 120


def _run_local(arguments: list[str], **options) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            arguments, capture_output=True, check=False,
            timeout=_LOCAL_COMMAND_TIMEOUT_SECONDS * timeout_scale(), **options,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"local pinned-tree command timed out: {arguments[0]}", pytrace=False)


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

    resolved = _run_local(
        ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        text=True,
    )
    if resolved.returncode != 0:
        pytest.skip(f"pinned commit {commit} is not available in this repository's git objects")
    full_commit = resolved.stdout.strip()
    session_key = (str(Path(tmp_path_factory.getbasetemp()).resolve()), full_commit)
    cached = _SESSION_TREES.get(session_key)
    if cached is not None:
        return cached

    archive = _run_local(
        ["git", "-C", str(_REPOSITORY_ROOT), "archive", "--format=tar", full_commit],
    )
    if archive.returncode != 0:
        pytest.fail(
            f"could not archive pinned commit {commit}: {archive.stderr.decode(errors='replace')}"
        )

    destination = _archive_destination(tmp_path_factory, full_commit, session_key)
    extracted = _run_local(
        ["tar", "-xf", "-", "-C", str(destination)],
        input=archive.stdout,
    )
    if extracted.returncode != 0:
        pytest.fail(
            f"could not extract pinned commit {commit}: {extracted.stderr.decode(errors='replace')}"
        )

    _SESSION_TREES[session_key] = destination
    return destination
