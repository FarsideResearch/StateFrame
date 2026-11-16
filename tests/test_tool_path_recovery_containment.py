"""Basename recovery must apply the same workspace containment as exact paths.

A repository can ship a symlink whose target lies outside the workspace. Exact
lookups already refuse it; recovery by basename (rglob) must refuse it too, while
still recovering ordinary files and in-workspace symlinks.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import task_runners
from kernel_tool_paths import (
    AmbiguousToolPath, ToolPathEscape, recover_by_basename, resolve_tool_path,
)


@pytest.fixture
def layout(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    (root / "src" / "pkg").mkdir(parents=True)
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("outside the workspace\n")
    return SimpleNamespace(root=root, outside=outside, secret=secret)


def _inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def test_recovery_skips_file_symlink_escaping_root(layout):
    (layout.root / "src" / "secret.txt").symlink_to(layout.secret)
    result = resolve_tool_path(layout.root, "missing/secret.txt")
    assert _inside(layout.root, result)
    assert not result.exists()
    assert result == layout.root / "missing/secret.txt"


def test_exact_escaping_symlink_still_refused(layout):
    (layout.root / "src" / "secret.txt").symlink_to(layout.secret)
    with pytest.raises(ToolPathEscape):
        resolve_tool_path(layout.root, "src/secret.txt")


def test_recovery_skips_files_under_directory_symlink_escaping_root(layout):
    (layout.root / "vendor").symlink_to(layout.outside, target_is_directory=True)
    result = resolve_tool_path(layout.root, "missing/secret.txt")
    assert _inside(layout.root, result)
    assert not result.exists()


def test_recovery_skips_directory_symlink_escaping_root(layout):
    (layout.root / "src" / "leak").symlink_to(layout.outside, target_is_directory=True)
    result = resolve_tool_path(layout.root, "missing/leak")
    assert _inside(layout.root, result)
    assert not result.exists()


def test_escaping_symlink_does_not_hide_unique_legit_match(layout):
    (layout.root / "secret.txt").symlink_to(layout.secret)
    legit = layout.root / "src" / "pkg" / "secret.txt"
    legit.write_text("inside\n")
    assert resolve_tool_path(layout.root, "docs/secret.txt") == legit


def test_legitimate_basename_recovery_still_works(layout):
    target = layout.root / "src" / "pkg" / "module.py"
    target.write_text("x = 1\n")
    assert resolve_tool_path(layout.root, "module.py") == target
    assert resolve_tool_path(layout.root, "wrong/dir/module.py") == target


def test_in_workspace_symlink_is_recoverable(layout):
    real = layout.root / "src" / "pkg" / "real.py"
    real.write_text("x = 1\n")
    (layout.root / "src" / "alias.py").symlink_to(real)
    recovered = resolve_tool_path(layout.root, "lib/alias.py")
    assert recovered == layout.root / "src" / "alias.py"
    assert recovered.resolve() == real.resolve()


def test_duplicate_contained_basenames_stay_ambiguous(layout):
    (layout.root / "src" / "dup.py").write_text("")
    (layout.root / "src" / "pkg" / "dup.py").write_text("")
    with pytest.raises(AmbiguousToolPath):
        resolve_tool_path(layout.root, "dup.py")


def test_recovery_from_unresolved_root_path(layout, monkeypatch):
    target = layout.root / "src" / "pkg" / "module.py"
    target.write_text("")
    (layout.root / "src" / "secret.txt").symlink_to(layout.secret)
    monkeypatch.chdir(layout.root.parent)
    relative_root = Path("repo")
    assert resolve_tool_path(relative_root, "module.py").resolve() == target.resolve()
    assert not resolve_tool_path(relative_root, "x/secret.txt").exists()


def test_basename_is_matched_literally_not_as_glob(layout):
    (layout.root / "src" / "pkg" / "only.py").write_text("")
    assert resolve_tool_path(layout.root, "*.py") == layout.root / "*.py"
    bracket = layout.root / "src" / "pkg" / "a[1].txt"
    bracket.write_text("")
    assert resolve_tool_path(layout.root, "a[1].txt") == bracket


@pytest.mark.parametrize("requested", ["missing/..", "missing/.", "."])
def test_dot_basenames_never_trigger_recovery(layout, requested):
    (layout.root / "src" / "pkg" / "file.py").write_text("")
    result = resolve_tool_path(layout.root, requested)
    assert result == layout.root / requested


def test_recover_by_basename_filters_escapes(layout):
    (layout.root / "secret.txt").symlink_to(layout.secret)
    legit = layout.root / "src" / "secret.txt"
    legit.write_text("")
    assert recover_by_basename(layout.root, "secret.txt") == [legit]


def _no_grep(*_args, **_kwargs):
    raise AssertionError("grep is not used for path selectors")


def test_pytest_selector_recovery_skips_escaping_symlink(layout):
    (layout.outside / "test_outside.py").write_text("def test_x():\n    pass\n")
    (layout.root / "test_outside.py").symlink_to(layout.outside / "test_outside.py")
    targets, _kexpr, error = task_runners._resolve_pytest_targets(
        layout.root.resolve(), "fn", ["tests/test_outside.py"], _no_grep)
    assert targets == []
    assert error == "test path does not exist: `tests/test_outside.py`"


def test_pytest_selector_recovery_keeps_legit_match(layout):
    (layout.root / "tests").mkdir()
    (layout.root / "tests" / "test_inside.py").write_text("def test_x():\n    pass\n")
    targets, _kexpr, error = task_runners._resolve_pytest_targets(
        layout.root.resolve(), "fn", ["test_inside.py::test_x"], _no_grep)
    assert error is None
    assert targets == ["tests/test_inside.py::test_x"]
