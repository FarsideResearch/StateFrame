"""The parity base is the pinned last release, identical in every layout of one commit.

These tests build small throwaway repositories so the resolver runs against real git
history: a full clone with a misleading ``origin/main`` and a detached copy without
remote refs must agree on the base, and that base is the release the tree names.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import immediate_baseline


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True, text=True, check=True, timeout=60,
        env={**os.environ, **_GIT_ENV},
    )
    return result.stdout.strip()


def _commit(root: Path, subject: str, changelog: str | None = None) -> str:
    if changelog is not None:
        (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    marker = root / "work.txt"
    marker.write_text(marker.read_text(encoding="utf-8") + subject + "\n"
                      if marker.exists() else subject + "\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", subject)
    return _git(root, "rev-parse", "HEAD")


def _changelog(*sections: str) -> str:
    return "# Changelog\n\n" + "".join(f"## {section}\n\n- entry\n\n" for section in sections)


def _origin(tmp_path: Path) -> dict[str, str]:
    """origin: 0.1.0 (tagged) -> work -> 0.2.0 (untagged release commit) -> work -> work."""
    root = tmp_path / "origin"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    commits = {"r010": _commit(root, "Release 0.1.0", _changelog("0.1.0 (2026-01-01)"))}
    _git(root, "tag", "v0.1.0")
    commits["work1"] = _commit(root, "Unrelated work")
    commits["r020"] = _commit(root, "Release 0.2.0",
                              _changelog("0.2.0 (2026-01-02)", "0.1.0 (2026-01-01)"))
    commits["open"] = _commit(root, "CHANGELOG: open Unreleased section",
                              _changelog("Unreleased", "0.2.0 (2026-01-02)", "0.1.0 (2026-01-01)"))
    commits["head"] = _commit(root, "Feature work after 0.2.0")
    return commits


def _layouts(tmp_path: Path, head: str, misleading: str) -> tuple[Path, Path]:
    origin = tmp_path / "origin"
    full = tmp_path / "full"
    _git(tmp_path, "clone", "-q", str(origin), str(full))
    _git(full, "checkout", "-q", "--detach", head)
    # A remote main that has moved elsewhere: the old resolver took merge-base(HEAD, origin/main).
    _git(full, "update-ref", "refs/remotes/origin/main", misleading)
    detached = tmp_path / "detached"
    _git(tmp_path, "clone", "-q", "--no-local", str(origin), str(detached))
    _git(detached, "checkout", "-q", "--detach", head)
    _git(detached, "remote", "remove", "origin")
    assert _git(detached, "for-each-ref", "refs/remotes") == ""
    return full, detached


def _resolve(monkeypatch: pytest.MonkeyPatch, root: Path) -> str:
    monkeypatch.setattr(immediate_baseline, "REPOSITORY_ROOT", root)
    return immediate_baseline._immediate_base_release()


def test_same_commit_resolves_to_pinned_release_in_every_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = _origin(tmp_path)
    full, detached = _layouts(tmp_path, commits["head"], misleading=commits["work1"])
    assert _git(full, "rev-parse", "HEAD^1") != commits["r020"]

    assert _resolve(monkeypatch, full) == commits["r020"]
    assert _resolve(monkeypatch, detached) == commits["r020"]


def test_release_head_resolves_to_previous_release_in_every_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = _origin(tmp_path)
    origin = tmp_path / "origin"
    release = _commit(origin, "Release 0.3.0",
                      _changelog("0.3.0 (2026-01-03)", "0.2.0 (2026-01-02)", "0.1.0 (2026-01-01)"))
    _git(origin, "tag", "v0.3.0")
    full, detached = _layouts(tmp_path, release, misleading=commits["r010"])

    # HEAD is itself release 0.3.0 (tagged), so parity must compare against 0.2.0, not HEAD.
    assert _resolve(monkeypatch, full) == commits["r020"]
    assert _resolve(monkeypatch, detached) == commits["r020"]


def test_untagged_release_head_resolves_to_previous_tagged_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = _origin(tmp_path)
    full, detached = _layouts(tmp_path, commits["r020"], misleading=commits["head"])

    assert _resolve(monkeypatch, full) == commits["r010"]
    assert _resolve(monkeypatch, detached) == commits["r010"]


def test_tag_outside_head_history_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = _origin(tmp_path)
    origin = tmp_path / "origin"
    # A same-named tag on a side line (e.g. a published merge) is not HEAD's release.
    _git(origin, "checkout", "-q", "-b", "side", commits["work1"])
    side = _commit(origin, "Side line")
    _git(origin, "tag", "v0.2.0", side)
    _git(origin, "checkout", "-q", "main")
    full, detached = _layouts(tmp_path, commits["head"], misleading=side)

    assert _resolve(monkeypatch, full) == commits["r020"]
    assert _resolve(monkeypatch, detached) == commits["r020"]


def test_unresolvable_pinned_release_skips_instead_of_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits = _origin(tmp_path)
    origin = tmp_path / "origin"
    # The tree names 0.2.5, but no tag or release commit for it exists in this history.
    _commit(origin, "Docs", _changelog("0.2.5 (2026-01-02)", "0.2.0 (2026-01-02)"))
    full, _ = _layouts(tmp_path, _git(origin, "rev-parse", "HEAD"), misleading=commits["r020"])

    monkeypatch.setattr(immediate_baseline, "REPOSITORY_ROOT", full)
    with pytest.raises(pytest.skip.Exception, match="0.2.5"):
        immediate_baseline._immediate_base_release()


def test_explicit_root_argument_matches_repository_root(tmp_path: Path) -> None:
    commits = _origin(tmp_path)
    _, detached = _layouts(tmp_path, commits["head"], misleading=commits["work1"])
    assert immediate_baseline._immediate_base_release(detached) == commits["r020"]
