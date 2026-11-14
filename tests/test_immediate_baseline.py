"""The parity baseline should follow the branch base in shallow lane clones."""

from __future__ import annotations

import subprocess

import pytest

from tools import immediate_baseline


HEAD = "a" * 40
BASE = "b" * 40
OLDER = "c" * 40
UNRELATED = "d" * 40
RELEASE = "e" * 40


def _answers(monkeypatch: pytest.MonkeyPatch, answers: dict[tuple[str, ...], str | None]) -> None:
    monkeypatch.setattr(immediate_baseline, "_git_text", lambda *args: answers.get(args))


def test_nearest_remote_ancestor_beats_older_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("for-each-ref", "--format=%(objectname)", "refs/remotes"):
            "\n".join((HEAD, OLDER, BASE, UNRELATED, BASE)),
        ("merge-base", HEAD, BASE): BASE,
        ("merge-base", HEAD, OLDER): OLDER,
        ("merge-base", HEAD, UNRELATED): OLDER,
        ("rev-list", "--count", f"{BASE}..{HEAD}"): "3",
        ("rev-list", "--count", f"{OLDER}..{HEAD}"): "20",
        ("log", "--format=%H%x09%s", "HEAD"): f"{RELEASE}\tRelease 0.4.39",
    })
    assert immediate_baseline._immediate_base_release() == BASE


def test_release_fallback_remains_when_no_remote_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("for-each-ref", "--format=%(objectname)", "refs/remotes"): UNRELATED,
        ("merge-base", HEAD, UNRELATED): OLDER,
        ("log", "--format=%H%x09%s", "HEAD"): f"{HEAD}\twork\n{RELEASE}\tRelease 0.4.39",
    })
    assert immediate_baseline._immediate_base_release() == RELEASE


def test_branch_only_clone_uses_parent_before_old_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("for-each-ref", "--format=%(objectname)", "refs/remotes"): HEAD,
        ("rev-parse", "--verify", "HEAD^1^{commit}"): BASE,
        ("log", "--format=%H%x09%s", "HEAD"): f"{RELEASE}\tRelease 0.4.39",
    })
    assert immediate_baseline._immediate_base_release() == BASE


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


def test_origin_main_retains_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("rev-parse", "--verify", "origin/main^{commit}"): OLDER,
        ("merge-base", "HEAD", "origin/main"): OLDER,
        ("for-each-ref", "--format=%(objectname)", "refs/remotes"): BASE,
        ("merge-base", HEAD, BASE): BASE,
        ("rev-list", "--count", f"{BASE}..{HEAD}"): "1",
    })
    assert immediate_baseline._immediate_base_release() == OLDER


def test_ambiguous_remote_ancestors_require_an_explicit_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answers(monkeypatch, {
        ("rev-parse", "--verify", "HEAD^{commit}"): HEAD,
        ("for-each-ref", "--format=%(objectname)", "refs/remotes"): f"{BASE}\n{OLDER}",
        ("merge-base", HEAD, BASE): BASE,
        ("merge-base", HEAD, OLDER): OLDER,
        ("rev-list", "--count", f"{BASE}..{HEAD}"): "1",
        ("rev-list", "--count", f"{OLDER}..{HEAD}"): "1",
    })
    with pytest.raises(pytest.fail.Exception, match="multiple equally close remote baselines"):
        immediate_baseline._immediate_base_release()
