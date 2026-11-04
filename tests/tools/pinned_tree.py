"""Materialize a pinned source tree from this repository's git objects."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,40}\Z")
_SESSION_TREES: dict[tuple[str, str], Path] = {}


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

    destination = tmp_path_factory.mktemp(f"pinned-tree-{full_commit[:12]}")
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
