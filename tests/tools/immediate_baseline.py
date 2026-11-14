"""Resolve the immediate base checkout shared by cross-version tests."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.pinned_tree import pinned_tree
from tools.sandbox_temp import timeout_scale


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
            timeout=30 * timeout_scale(),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"local git baseline lookup timed out: {arguments[0]}", pytrace=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _nearest_remote_ancestor(head: str) -> str | None:
    """Use the closest remote branch point when the usual origin/main is absent."""
    refs = _git_text("for-each-ref", "--format=%(objectname)", "refs/remotes") or ""
    candidates: list[tuple[int, str]] = []
    for commit in set(refs.splitlines()):
        if commit == head or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            continue
        if _git_text("merge-base", head, commit) != commit:
            continue
        distance = _git_text("rev-list", "--count", f"{commit}..{head}")
        if distance is not None and distance.isdecimal():
            candidates.append((int(distance), commit))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        pytest.fail("multiple equally close remote baselines; set SF_TEST_BASELINE_IMMEDIATE")
    return candidates[0][1]


def _immediate_base_release() -> str:
    head = _git_text("rev-parse", "--verify", "HEAD^{commit}")
    if head is None:
        pytest.skip("cannot derive the immediate base release because HEAD is unavailable")

    if _git_text("rev-parse", "--verify", "origin/main^{commit}") is not None:
        merge_base = _git_text("merge-base", "HEAD", "origin/main")
        if merge_base is not None and merge_base != head:
            return merge_base

    remote_base = _nearest_remote_ancestor(head)
    if remote_base is not None:
        return remote_base

    # A normal clone of a branch-only local repository retains only the checked-out
    # branch's tip as a remote ref. Its first parent is the same immediate ancestor
    # that the source repository's remote refs would select for a one-commit branch.
    parent = _git_text("rev-parse", "--verify", "HEAD^1^{commit}")
    if parent is not None and parent != head:
        return parent

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
