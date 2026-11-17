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
# The released versions, newest first, come from HEAD's own CHANGELOG.md: the in-tree
# source of truth that names the last release. "## Unreleased" is not a release.
_RELEASED_HEADING = re.compile(r"^## v?(\d+(?:\.\d+)+)(?:\s|$)", re.MULTILINE)


def _release_subject(version: str) -> re.Pattern[str]:
    escaped = re.escape(version)
    return re.compile(
        rf"^(?:StateFrame v?{escaped}: version and changelog|Release v?{escaped}|"
        rf"Merge pull request #\d+ from .*/release/v?{escaped})$"
    )


def _git_text(*arguments: str, root: Path | None = None) -> str | None:
    repository = REPOSITORY_ROOT if root is None else root
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
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


def _released_versions(root: Path | None) -> list[str]:
    changelog = _git_text("show", "HEAD:CHANGELOG.md", root=root)
    if changelog is None:
        return []
    return _RELEASED_HEADING.findall(changelog)


def _release_commit(version: str, head: str, root: Path | None) -> str | None:
    """Return release ``version``'s commit in HEAD's own history (inclusive), or None.

    Tag ``v<version>`` wins when it lies in HEAD's history; a same-named tag elsewhere
    (for example a published merge on another line) is not this tree's release. Without
    such a tag, the release commit is found by its subject in HEAD's history, which is
    the same in every non-shallow clone. Remote refs and parents are never consulted.
    """
    tagged = _git_text("rev-parse", "--verify", "--quiet", f"refs/tags/v{version}^{{commit}}", root=root)
    if tagged and (tagged == head or _git_text("merge-base", "--is-ancestor", tagged, head, root=root) is not None):
        return tagged
    history = _git_text("log", "--topo-order", "--format=%H%x09%s", head, root=root)
    subject = _release_subject(version)
    for line in (history or "").splitlines():
        commit, separator, text = line.partition("\t")
        if separator and subject.fullmatch(text):
            return commit
    return None


def _immediate_base_release(root: Path | None = None) -> str:
    """Return the last release strictly before HEAD, as pinned by HEAD's CHANGELOG.md.

    The result depends only on the commit, never on clone layout: no origin/main, no
    nearest remote ref, no first parent. When HEAD is itself the newest release, the
    base is the release before it, so parity never compares a tree with itself. An
    unresolvable pinned release skips rather than falling back to another base.
    """
    head = _git_text("rev-parse", "--verify", "HEAD^{commit}", root=root)
    if head is None:
        pytest.skip("cannot derive the immediate base release because HEAD is unavailable")
    versions = _released_versions(root)
    if not versions:
        pytest.skip("CHANGELOG.md at HEAD names no released version to use as the parity "
                    "baseline; set SF_TEST_BASELINE_IMMEDIATE")
    head_subject = _git_text("log", "-1", "--format=%s", head, root=root) or ""
    for version in versions[:2]:
        # HEAD is this version's release when it is the release commit itself or a release
        # merge for it (whose tag may sit on the merged commit with the same tree).
        if _release_subject(version).fullmatch(head_subject):
            continue
        commit = _release_commit(version, head, root)
        if commit is None:
            pytest.skip(f"pinned base release {version} (CHANGELOG.md at HEAD) has no tag "
                        f"v{version} or release commit in this clone's history; set "
                        "SF_TEST_BASELINE_IMMEDIATE")
        if commit != head:
            return commit
    pytest.skip("HEAD is the only release named by CHANGELOG.md, so it has no earlier "
                "release to use as the parity baseline; set SF_TEST_BASELINE_IMMEDIATE")


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
