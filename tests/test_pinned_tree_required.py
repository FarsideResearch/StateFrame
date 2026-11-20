"""A release gate must fail when its pinned comparison tree is unavailable."""

from __future__ import annotations

import shutil
import subprocess

import pytest
import tools.pinned_tree as pinned_tree_module

MISSING_COMMIT = "f" * 40


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setattr(pinned_tree_module, "_REPOSITORY_ROOT", repo)
    return repo


def test_missing_commit_skips_without_required_mode(git_repo, tmp_path_factory, monkeypatch):
    monkeypatch.delenv("PINNED_TREE_REQUIRED", raising=False)

    with pytest.raises(pytest.skip.Exception, match=MISSING_COMMIT):
        pinned_tree_module.pinned_tree(MISSING_COMMIT, tmp_path_factory)


def test_missing_commit_fails_in_required_mode(git_repo, tmp_path_factory, monkeypatch):
    monkeypatch.setenv("PINNED_TREE_REQUIRED", "1")

    with pytest.raises(BaseException) as caught:
        pinned_tree_module.pinned_tree(MISSING_COMMIT, tmp_path_factory)
    assert isinstance(caught.value, pytest.fail.Exception)
    assert "gate copy lacks" in str(caught.value)
    assert MISSING_COMMIT in str(caught.value)


def test_present_commit_returns_tree_in_required_mode(git_repo, tmp_path_factory, monkeypatch):
    (git_repo / "example.txt").write_text("pinned content\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "example.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "add example"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "HEAD"],
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    monkeypatch.setenv("PINNED_TREE_REQUIRED", "1")

    tree = pinned_tree_module.pinned_tree(commit, tmp_path_factory)

    assert (tree / "example.txt").read_text(encoding="utf-8") == "pinned content\n"


@pytest.mark.parametrize("required", [False, True])
def test_git_unavailable_matches_mode(git_repo, tmp_path_factory, monkeypatch, required):
    if required:
        monkeypatch.setenv("PINNED_TREE_REQUIRED", "1")
        expected = pytest.fail.Exception
    else:
        monkeypatch.delenv("PINNED_TREE_REQUIRED", raising=False)
        expected = pytest.skip.Exception
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    with pytest.raises(BaseException) as caught:
        pinned_tree_module.pinned_tree(MISSING_COMMIT, tmp_path_factory)
    assert isinstance(caught.value, expected)
    assert MISSING_COMMIT in str(caught.value)
    if required:
        assert "gate copy lacks" in str(caught.value)
