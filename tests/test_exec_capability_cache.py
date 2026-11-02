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
