"""The parity baseline is the pinned last release, never a clone-layout guess."""

from __future__ import annotations

import subprocess

import pytest

from tools import immediate_baseline


HEAD = "a" * 40
BASE = "b" * 40
OLDER = "c" * 40
REMOTE = "d" * 40
CHANGELOG = "# Changelog\n\n## Unreleased\n\n## 0.4.39 (2026-01-02)\n\n## 0.4.38\n"


def _answers(monkeypatch: pytest.MonkeyPatch, answers: dict[tuple[str, ...], str | None]) -> list:
    asked = []

    def fake(*args: str, root=None) -> str | None:
        asked.append(args)
        return answers.get(args)

    monkeypatch.setattr(immediate_baseline, "_git_text", fake)
    return asked


def test_remote_refs_and_parent_are_never_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("show", "HEAD:CHANGELOG.md"): CHANGELOG,
        ("rev-parse", "--verify", "--quiet", "refs/tags/v0.4.39^{commit}"): BASE,
        ("merge-base", "--is-ancestor", BASE, HEAD): "",
        ("rev-parse", "--verify", "origin/main^{commit}"): REMOTE,
        ("for-each-ref", "--format=%(objectname)", "refs/remotes"): REMOTE,
        ("rev-parse", "--verify", "HEAD^1^{commit}"): OLDER,
    })
    assert immediate_baseline._immediate_base_release() == BASE
    assert not any("origin/main" in " ".join(a) or "refs/remotes" in a or "HEAD^1^{commit}" in a
                   for a in asked)


def test_untagged_release_found_by_subject_in_head_history(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("show", "HEAD:CHANGELOG.md"): CHANGELOG,
        ("log", "--topo-order", "--format=%H%x09%s", HEAD):
            f"{HEAD}\twork\n{BASE}\tRelease 0.4.39\n{OLDER}\tRelease 0.4.38",
    })
    assert immediate_baseline._immediate_base_release() == BASE


def test_release_head_uses_previous_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("show", "HEAD:CHANGELOG.md"): "## 0.4.39\n\n## 0.4.38\n",
        ("log", "--topo-order", "--format=%H%x09%s", HEAD):
            f"{HEAD}\tRelease 0.4.39\n{OLDER}\tRelease 0.4.38",
    })
    assert immediate_baseline._immediate_base_release() == OLDER


def test_missing_pinned_release_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("show", "HEAD:CHANGELOG.md"): CHANGELOG,
        ("log", "--topo-order", "--format=%H%x09%s", HEAD): f"{HEAD}\twork",
        ("rev-parse", "--verify", "origin/main^{commit}"): REMOTE,
    })
    with pytest.raises(pytest.skip.Exception, match="0.4.39"):
        immediate_baseline._immediate_base_release()


def test_no_released_version_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("show", "HEAD:CHANGELOG.md"): "# Changelog\n\n## Unreleased\n",
    })
    with pytest.raises(pytest.skip.Exception, match="names no released version"):
        immediate_baseline._immediate_base_release()


def test_local_git_timeout_fails_instead_of_using_stale_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SF_TEST_TIMEOUT_SCALE", "4")
    seen = []

    def expired(*args, **kwargs):
        seen.append((args, kwargs))
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(immediate_baseline.subprocess, "run", expired)
    with pytest.raises(pytest.fail.Exception, match="local git baseline lookup timed out"):
        immediate_baseline._git_text("rev-parse", "HEAD")
    assert seen[0][1]["timeout"] == 120
