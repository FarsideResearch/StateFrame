"""find_workspace must not follow a task-row cache_key outside the cache."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from build_tasks.swebench_env import WorkspaceLookupError, find_workspace

_VALID_KEY = "0123456789abcdef01234567"


def _materialize(root: Path) -> Path:
    (root / "base").mkdir(parents=True)
    (root / "gold").mkdir()
    return root


def _cache_with_instances(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    (cache / "instances").mkdir(parents=True)
    return cache


def test_cache_key_cannot_escape_with_dotdot(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    outside = _materialize(tmp_path / "outside")
    row = {"cache_key": "../../outside", "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_INVALID"):
        find_workspace(cache, row)
    assert outside.is_dir()  # the bait existed and was still refused


def test_cache_key_cannot_be_absolute(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    outside = _materialize(tmp_path / "elsewhere")
    row = {"cache_key": str(outside), "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_INVALID"):
        find_workspace(cache, row)


def test_cache_key_base_suffix_cannot_redirect(tmp_path: Path) -> None:
    # A key ending in /base used to be accepted and mapped to its parent.
    cache = _cache_with_instances(tmp_path)
    _materialize(cache / "instances" / _VALID_KEY)
    row = {"cache_key": f"{_VALID_KEY}/base", "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_INVALID"):
        find_workspace(cache, row)


@pytest.mark.parametrize(
    "bad_key",
    [
        ".",
        "..",
        "~",
        "~/x",
        "abc",  # too short to be a digest prefix
        "0123456789ABCDEF01234567",  # uppercase is not what the adapter emits
        "0123456789abcdef0123456g",
        f"{_VALID_KEY}\n",
        f" {_VALID_KEY}",
        f"{_VALID_KEY}\x00",
        "a" * 65,
    ],
)
def test_cache_key_non_digest_shapes_are_rejected(tmp_path: Path, bad_key: str) -> None:
    cache = _cache_with_instances(tmp_path)
    row = {"cache_key": bad_key, "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_INVALID"):
        find_workspace(cache, row)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_cache_key_symlink_outside_cache_is_refused(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    outside = _materialize(tmp_path / "victim")
    key = "a" * 24
    (cache / "instances" / key).symlink_to(outside, target_is_directory=True)
    row = {"cache_key": key, "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_ESCAPE"):
        find_workspace(cache, row)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_cache_key_symlink_to_sibling_instance_is_refused(tmp_path: Path) -> None:
    # Even a target inside instances/ is refused: a key must name its own directory.
    cache = _cache_with_instances(tmp_path)
    other = _materialize(cache / "instances" / ("b" * 24))
    key = "c" * 24
    (cache / "instances" / key).symlink_to(other, target_is_directory=True)
    row = {"cache_key": key, "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_ESCAPE"):
        find_workspace(cache, row)


def _git_commit(repo: Path) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required for the SHA scan")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "f.txt"], ["commit", "-q", "-m", "init"]):
        subprocess.run([git, "-C", str(repo), *args], check=True, env=env)
    return subprocess.run(
        [git, "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_sha_scan_skips_instance_symlinked_outside_cache(tmp_path: Path) -> None:
    # No cache_key: the SHA scan must not adopt a planted escaping symlink either.
    cache = _cache_with_instances(tmp_path)
    outside = _materialize(tmp_path / "victim")
    head = _git_commit(outside / "base")
    (cache / "instances" / ("d" * 24)).symlink_to(outside, target_is_directory=True)
    row = {"base_commit": head}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_NOT_FOUND"):
        find_workspace(cache, row)


def test_sha_scan_still_finds_real_instance(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    expected = _materialize(cache / "instances" / _VALID_KEY)
    head = _git_commit(expected / "base")
    assert find_workspace(cache, {"base_commit": head}).resolve() == expected.resolve()


def test_valid_cache_key_still_resolves(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    expected = _materialize(cache / "instances" / _VALID_KEY)
    row = {"cache_key": _VALID_KEY, "base_commit": "deadbeef"}
    assert find_workspace(cache, row) == expected.resolve()


def test_valid_cache_key_resolves_when_cache_root_is_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real-cache"
    expected = _materialize(real / "instances" / _VALID_KEY)
    link = tmp_path / "cache-link"
    link.symlink_to(real, target_is_directory=True)
    row = {"cache_key": _VALID_KEY, "base_commit": "deadbeef"}
    assert find_workspace(link, row) == expected.resolve()


def test_valid_but_missing_cache_key_falls_through_to_sha_scan(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    row = {"cache_key": _VALID_KEY, "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_NOT_FOUND"):
        find_workspace(cache, row)


def test_non_string_or_empty_cache_key_is_ignored(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    for value in ("", None, 123):
        row = {"cache_key": value, "base_commit": "deadbeef"}
        with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_NOT_FOUND"):
            find_workspace(cache, row)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
@pytest.mark.parametrize("member", ["base", "gold"])
def test_cache_key_member_symlinked_outside_is_refused(tmp_path: Path, member: str) -> None:
    # A real instance dir whose base/ or gold/ is a symlink out of the cache.
    cache = _cache_with_instances(tmp_path)
    victim = _materialize(tmp_path / "victim")
    instance = cache / "instances" / _VALID_KEY
    instance.mkdir()
    other = "gold" if member == "base" else "base"
    (instance / other).mkdir()
    (instance / member).symlink_to(victim / member, target_is_directory=True)
    row = {"cache_key": _VALID_KEY, "base_commit": "deadbeef"}
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_CACHE_KEY_ESCAPE"):
        find_workspace(cache, row)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_sha_scan_skips_instance_with_base_symlinked_outside(tmp_path: Path) -> None:
    cache = _cache_with_instances(tmp_path)
    victim = _materialize(tmp_path / "victim")
    head = _git_commit(victim / "base")
    instance = cache / "instances" / _VALID_KEY
    instance.mkdir()
    (instance / "gold").mkdir()
    (instance / "base").symlink_to(victim / "base", target_is_directory=True)
    with pytest.raises(WorkspaceLookupError, match="ENV_WORKSPACE_NOT_FOUND"):
        find_workspace(cache, {"base_commit": head})
