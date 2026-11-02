"""Executable authority caching must preserve the exact issued capability."""
from pathlib import Path

import pytest

from stateframe.engine import kernel_exec_capability as capability


@pytest.fixture
def directories(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    run_env = tmp_path / "run_env"
    home = tmp_path / "home"
    external_bin = tmp_path / "external_bin"
    for directory in (workspace, run_env / "bin", home, external_bin):
        directory.mkdir(parents=True)
    system_dirs = {Path(path) for path in ("/usr/bin", "/bin", "/usr/sbin", "/sbin")}
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(()) if path in system_dirs
                        else original_iterdir(path))
    monkeypatch.setattr(capability, "system_runtime_read_roots", lambda: ())
    return workspace, run_env, home, external_bin


def _executable(directory, name):
    path = directory / name
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _build(directories):
    workspace, run_env, home, external_bin = directories
    return capability.shell_capability(workspace=workspace, run_env_dir=run_env,
                                       scratch_home=home, network="none", writable=True,
                                       path_dirs=(external_bin,))


def test_second_call_reuses_external_directory_without_rereading(directories, monkeypatch):
    script = _executable(directories[3], "first")
    calls = []
    original = capability.executable_read_roots

    def counted(path):
        if path == script:
            calls.append(path)
        return original(path)

    monkeypatch.setattr(capability, "executable_read_roots", counted)
    first = _build(directories)
    first_count = len(calls)
    second = _build(directories)
    assert first_count > 0
    assert len(calls) == first_count
    assert second.as_dict() == first.as_dict()
    assert second.digest() == first.digest()


def test_added_external_executable_invalidates_cache(directories, monkeypatch):
    first = _executable(directories[3], "first")
    original = capability.executable_read_roots
    calls = []

    def counted(path):
        if path == first:
            calls.append(path)
        return original(path)

    monkeypatch.setattr(capability, "executable_read_roots", counted)
    before = _build(directories)
    first_count = len(calls)
    assert _build(directories).digest() == before.digest()
    assert len(calls) == first_count
    second = _executable(directories[3], "second")
    after = _build(directories)
    assert first in before.read_roots
    assert second not in before.read_roots
    assert second in after.read_roots


def test_run_environment_bin_is_never_cached(directories, monkeypatch):
    external = _executable(directories[3], "external")
    dynamic = _executable(directories[1] / "bin", "dynamic")
    original = capability.executable_read_roots
    calls = {external: 0, dynamic: 0}

    def counted(path):
        if path in calls:
            calls[path] += 1
        return original(path)

    # A stable external directory reuses its result, while run_env/bin remains live.
    monkeypatch.setattr(capability, "executable_read_roots", counted)
    _build(directories)
    first_count = dict(calls)
    _build(directories)
    assert calls[external] == first_count[external]
    assert calls[dynamic] > first_count[dynamic]
    new = _executable(directories[1] / "bin", "installed_later")
    assert new in _build(directories).read_roots


def test_edited_shebang_invalidates_cache(directories):
    script = _executable(directories[3], "script")
    before = _build(directories)
    script.write_bytes(b"#!/usr/bin/env python3\n")
    after = _build(directories)
    assert after.digest() != before.digest()
    assert Path("/usr/bin/env") in after.read_roots


def test_cached_digest_matches_fresh_expansion(directories):
    _executable(directories[3], "first")
    cached = _build(directories)
    capability.clear_shell_capability_cache()
    fresh = _build(directories)
    assert cached.as_dict() == fresh.as_dict()
    assert cached.digest() == fresh.digest()


def test_directory_scan_admits_exactly_what_is_file_and_x_ok_admit(tmp_path):
    import os
    target = _executable(tmp_path, "real")
    _executable(tmp_path, "tool")
    (tmp_path / "linked").symlink_to(target)
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    (tmp_path / "plain.txt").write_text("not executable")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "dirlink").symlink_to(tmp_path / "subdir")
    expected = tuple(path for path in sorted(tmp_path.iterdir()) if path.is_file() and os.access(path, os.X_OK))
    key, executables = capability._directory_executables(tmp_path)
    assert executables == expected
    assert [name for name, _, _ in key[2]] == ["linked", "real", "tool"]


def test_directory_scan_fingerprints_link_and_target_separately(tmp_path):
    first = _executable(tmp_path, "first")
    second = _executable(tmp_path, "second")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    link = bin_dir / "tool"
    link.symlink_to(first)
    (name, target, own), = capability._directory_executables(bin_dir)[0][2]
    assert name == "tool"
    assert target == capability._file_fingerprint(first.stat())
    assert own == capability._file_fingerprint(link.lstat()) != target
    link.unlink()
    link.symlink_to(second)
    (_, retarget, _), = capability._directory_executables(bin_dir)[0][2]
    assert retarget == capability._file_fingerprint(second.stat()) != target


def test_directory_scan_skips_entries_whose_metadata_is_denied(tmp_path, monkeypatch):
    # macOS protects some /usr/sbin entries: stat raises EACCES and Path.is_file says False.
    # Skipping narrows authority; the entry never becomes a read root.
    import errno
    denied_entry = _executable(tmp_path, "protected")
    kept = _executable(tmp_path, "tool")
    (tmp_path / "linked").symlink_to(denied_entry)
    original_lstat, original_stat = Path.lstat, Path.stat

    def denied_lstat(path):
        if path.name == "protected":
            raise PermissionError(errno.EACCES, "denied", str(path))
        return original_lstat(path)

    def denied_stat(path, **kwargs):
        if path.name in {"protected", "linked"}:
            raise PermissionError(errno.EACCES, "denied", str(path))
        return original_stat(path, **kwargs)
    monkeypatch.setattr(Path, "lstat", denied_lstat)
    monkeypatch.setattr(Path, "stat", denied_stat)
    assert capability._directory_executables(tmp_path)[1] == (kept,)


def test_directory_scan_skips_entries_that_vanish(tmp_path, monkeypatch):
    import errno
    _executable(tmp_path, "gone")
    kept = _executable(tmp_path, "kept")
    original = Path.lstat

    def vanished(path):
        if path.name == "gone":
            raise FileNotFoundError(errno.ENOENT, "gone", str(path))
        return original(path)
    monkeypatch.setattr(Path, "lstat", vanished)
    assert capability._directory_executables(tmp_path)[1] == (kept,)
