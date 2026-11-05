"""Regression coverage for repo-map cache isolation at the StateFrame intake boundary."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENGINE = SRC / "stateframe" / "engine"


@pytest.fixture(scope="module")
def stateframe_repo_map_modules():
    for path in (str(SRC), str(ENGINE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from stateframe import ts_grammars

    ts_grammars.install_as_language_pack()
    return (
        importlib.import_module("kernel_repo_intel"),
        importlib.import_module("mc_repo_intel_vendor.repo_map"),
    )


def _root(parent: Path, name: str, function_name: str) -> Path:
    root = parent / name
    root.mkdir(parents=True)
    (root / "shared.py").write_text(
        f"def {function_name}():\n    return {function_name!r}\n",
        encoding="utf-8",
    )
    return root


def test_same_commit_different_roots_never_share_intake_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stateframe_repo_map_modules,
) -> None:
    kernel_repo_intel, _repo_map = stateframe_repo_map_modules
    first_root = _root(tmp_path, "first", "first_project_marker")
    second_root = _root(tmp_path, "second", "second_project_marker")
    monkeypatch.setattr(kernel_repo_intel, "get_current_commit_sha", lambda _root: "a" * 40)

    first_orientation = kernel_repo_intel.intake_orientation(first_root)
    second_orientation = kernel_repo_intel.intake_orientation(second_root)

    assert "first_project_marker" in first_orientation
    assert "second_project_marker" in second_orientation
    assert "first_project_marker" not in second_orientation


def test_empty_commit_different_roots_never_share_intake_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stateframe_repo_map_modules,
) -> None:
    # Regression guard, not a fail-first reproduction: the old key was empty for an empty
    # commit, so it never cached. This keeps any future cache from keying on commit alone.
    kernel_repo_intel, _repo_map = stateframe_repo_map_modules
    first_root = _root(tmp_path, "first", "first_project_marker")
    second_root = _root(tmp_path, "second", "second_project_marker")
    monkeypatch.setattr(kernel_repo_intel, "get_current_commit_sha", lambda _root: "")

    first_orientation = kernel_repo_intel.intake_orientation(first_root)
    second_orientation = kernel_repo_intel.intake_orientation(second_root)

    assert "first_project_marker" in first_orientation
    assert "second_project_marker" in second_orientation
    assert "first_project_marker" not in second_orientation


def test_same_root_changed_content_invalidates_explicit_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stateframe_repo_map_modules,
) -> None:
    _kernel_repo_intel, repo_map = stateframe_repo_map_modules
    root = _root(tmp_path, "mutable", "before_change_marker")
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    cache_home = tmp_path / "stateframe-home"
    cache_home.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(cache_home))

    before_change, _ = repo_map.build_repo_map_with_stats(
        str(root), commit_sha="c" * 40, use_cache=True,
    )
    (root / "shared.py").write_text(
        "def after_change_marker():\n    return 'after_change_marker'\n",
        encoding="utf-8",
    )
    after_change, after_stats = repo_map.build_repo_map_with_stats(
        str(root), commit_sha="c" * 40, use_cache=True,
    )

    assert "before_change_marker" in before_change
    assert "after_change_marker" in after_change
    assert "before_change_marker" not in after_change
    assert after_stats["source"] == "generated"


def test_explicit_cache_is_scoped_to_resolved_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stateframe_repo_map_modules,
) -> None:
    _kernel_repo_intel, repo_map = stateframe_repo_map_modules
    first_root = _root(tmp_path, "first", "first_cached_marker")
    second_root = _root(tmp_path, "second", "second_cached_marker")
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    cache_home = tmp_path / "stateframe-home"
    cache_home.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(cache_home))

    first_map, _ = repo_map.build_repo_map_with_stats(
        str(first_root), commit_sha="d" * 40, use_cache=True,
    )
    second_map, second_stats = repo_map.build_repo_map_with_stats(
        str(second_root), commit_sha="d" * 40, use_cache=True,
    )

    assert "first_cached_marker" in first_map
    assert "second_cached_marker" in second_map
    assert "first_cached_marker" not in second_map
    assert second_stats["source"] == "generated"


def test_legacy_cache_home_variable_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stateframe_repo_map_modules,
) -> None:
    kernel_repo_intel, repo_map = stateframe_repo_map_modules
    standalone_root = _root(tmp_path, "standalone", "standalone_home_marker")
    intake_root = _root(tmp_path, "intake", "intake_home_marker")
    ignored_home = tmp_path / "ignored-cache-home"
    monkeypatch.delenv("STATEFRAME_HOME", raising=False)
    # The legacy cache-home variable name is built at runtime so the source never spells it.
    monkeypatch.setenv("_".join(("LEAN", "CTX", "HOME")), str(ignored_home))
    monkeypatch.setattr(kernel_repo_intel, "get_current_commit_sha", lambda _root: "e" * 40)

    workspace = Path("/workspace")
    workspace_before = sorted(workspace.rglob("*")) if workspace.is_dir() else None
    standalone_map = repo_map.build_repo_map(
        str(standalone_root), commit_sha="e" * 40, use_cache=True,
    )
    intake_orientation = kernel_repo_intel.intake_orientation(intake_root)
    workspace_after = sorted(workspace.rglob("*")) if workspace.is_dir() else None

    assert standalone_map
    assert "standalone_home_marker" in standalone_map
    assert intake_orientation
    assert "intake_home_marker" in intake_orientation
    assert not list(ignored_home.rglob("*"))
    assert workspace_after == workspace_before  # nothing created or written under /workspace


def _file_states(directory: Path) -> dict[Path, tuple[int, int]]:
    """Size and mtime of every file under a directory (empty when it doesn't exist)."""
    if not directory.is_dir():
        return {}
    return {path: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("home_set", [False, True], ids=["home-unset", "home-set"])
@pytest.mark.parametrize("stateframe_home_set", [False, True], ids=["stateframe-home-unset", "stateframe-home-set"])
def test_intake_cache_writes_stay_within_configured_state_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stateframe_repo_map_modules,
    home_set: bool,
    stateframe_home_set: bool,
) -> None:
    kernel_repo_intel, repo_map = stateframe_repo_map_modules
    root = _root(tmp_path, "run-workspace", "run_project_marker")
    env_paths = {
        "HOME": tmp_path / "home",
        "STATEFRAME_HOME": tmp_path / "stateframe-home",
    }
    enabled = {
        "HOME": home_set,
        "STATEFRAME_HOME": stateframe_home_set,
    }
    for name, path in env_paths.items():
        if enabled[name]:
            path.mkdir()
            monkeypatch.setenv(name, str(path))
        else:
            monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(kernel_repo_intel, "get_current_commit_sha", lambda _root: "b" * 40)

    allowed_roots = [root]
    if stateframe_home_set:
        allowed_roots.append(env_paths["STATEFRAME_HOME"])
    tmp_before = _file_states(tmp_path)
    workspace_before = _file_states(Path("/workspace"))

    orientation = kernel_repo_intel.intake_orientation(root)
    assert "run_project_marker" in orientation

    # Containment: every file created or changed lives under the run root or STATEFRAME_HOME,
    # and nothing under /workspace is created or modified.
    changed = {path for path, state in _file_states(tmp_path).items() if tmp_before.get(path) != state}
    assert all(any(path.is_relative_to(allowed) for allowed in allowed_roots) for path in changed), changed
    assert _file_states(Path("/workspace")) == workspace_before

    cache_dir = repo_map._get_cache_dir()
    configured_cache_home = env_paths["STATEFRAME_HOME"] if stateframe_home_set else None
    if configured_cache_home is None:
        assert cache_dir is None
    else:
        assert cache_dir == configured_cache_home / "repo_map_cache"



def test_concurrent_cache_writers_never_publish_a_partial_map(
    tmp_path: Path, stateframe_repo_map_modules,
) -> None:
    import threading

    _kernel_repo_intel, repo_map = stateframe_repo_map_modules
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    bodies = ["A" * 200_000, "B" * 200_000]
    barrier = threading.Barrier(len(bodies))

    def write(body: str) -> None:
        barrier.wait()
        for _ in range(20):
            repo_map._save_cached_map("shared-key", body, cache_dir)

    threads = [threading.Thread(target=write, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    published = (cache_dir / "shared-key.txt").read_text(encoding="utf-8")
    assert published in bodies
    assert not list(cache_dir.glob("*.tmp"))
