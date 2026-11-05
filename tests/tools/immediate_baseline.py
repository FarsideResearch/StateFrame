"""Resolve the immediate release checkout shared by cross-version tests."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.pinned_tree import pinned_tree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RELEASE_SUBJECT = re.compile(
    r"^(?:StateFrame .+: version and changelog|Release v?\d+(?:\.\d+)+|"
    r"Merge pull request #\d+ from .*/release/v?\d+(?:\.\d+)+)$"
)


def _git_text(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _immediate_base_release() -> str:
    head = _git_text("rev-parse", "--verify", "HEAD^{commit}")
    if head is None:
        pytest.skip("cannot derive the immediate base release because HEAD is unavailable")

    if _git_text("rev-parse", "--verify", "origin/main^{commit}") is not None:
        merge_base = _git_text("merge-base", "HEAD", "origin/main")
        if merge_base is not None and merge_base != head:
            return merge_base

    history = _git_text("log", "--format=%H%x09%s", "HEAD")
    if history is not None:
        for line in history.splitlines():
            commit, separator, subject = line.partition("\t")
            if separator and commit != head and _RELEASE_SUBJECT.fullmatch(subject):
                return commit
    pytest.skip("cannot find an ancestor release commit to use as the hook parity baseline")


@pytest.fixture(scope="module")
def baseline_checkout(tmp_path_factory):
    explicit = os.environ.get("SF_TEST_BASELINE_IMMEDIATE")
    if explicit is not None:
        if not explicit.strip():
            pytest.fail("SF_TEST_BASELINE_IMMEDIATE is set but empty", pytrace=False)
        baseline = Path(explicit).expanduser().resolve()
    else:
        # Must-run gates provision the baseline explicitly; a derived one could silently be the wrong tree.
        if os.environ.get("SF_GATE_MUST_RUN", "").strip().lower() in {"1", "true", "yes", "on"}:
            pytest.fail("SF_TEST_BASELINE_IMMEDIATE is required for must-run parity tests", pytrace=False)
        baseline = pinned_tree(_immediate_base_release(), tmp_path_factory)
    if not (baseline / "src/stateframe/runner.py").is_file():
        pytest.fail(f"immediate base checkout has no src/stateframe/runner.py: {baseline}", pytrace=False)
    return baseline
