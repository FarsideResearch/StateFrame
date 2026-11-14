"""Sandboxed test fixtures must remain visible with a shared-memory basetemp."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
from tools import pinned_tree as pinned_tree_module
from tools.sandbox_temp import (
    sandbox_visible_temp_dir,
    shadowed_by_bwrap_dev,
    timeout_scale,
)


def test_sandbox_scratch_is_visible_outside_dev_shm():
    with sandbox_visible_temp_dir() as directory:
        path = Path(directory)
        assert path.is_dir()
        assert not shadowed_by_bwrap_dev(path)
    assert not path.exists()


def test_pinned_tree_uses_disk_when_basetemp_is_under_dev_shm():
    if not shadowed_by_bwrap_dev(Path("/dev/shm/pinned-tree-test")):
        pytest.skip("requires Linux bwrap /dev overlay")

    class SharedMemoryFactory:
        def getbasetemp(self) -> Path:
            return Path("/dev/shm/pinned-tree-test")

        def mktemp(self, _prefix: str) -> Path:
            raise AssertionError("shared-memory checkout would be hidden by bwrap")

    key = ("shared-memory-test", "0123456789abcdef")
    destination = pinned_tree_module._archive_destination(
        SharedMemoryFactory(), key[1], key,
    )
    try:
        assert destination.is_dir()
        assert not shadowed_by_bwrap_dev(destination)
    finally:
        temporary = pinned_tree_module._SESSION_DISK_TEMPS.pop(key)
        assert isinstance(temporary, tempfile.TemporaryDirectory)
        temporary.cleanup()
    assert not destination.exists()


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "abc"])
def test_timeout_scale_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("SF_TEST_TIMEOUT_SCALE", raw)
    with pytest.raises(ValueError, match="SF_TEST_TIMEOUT_SCALE"):
        timeout_scale()


def test_timeout_scale_uses_environment(monkeypatch):
    monkeypatch.setenv("SF_TEST_TIMEOUT_SCALE", "4")
    assert timeout_scale() == 4.0


def test_pinned_tree_local_command_has_scaled_timeout(monkeypatch):
    monkeypatch.setenv("SF_TEST_TIMEOUT_SCALE", "4")
    seen = []

    def expired(*args, **kwargs):
        seen.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(pinned_tree_module.subprocess, "run", expired)
    with pytest.raises(pytest.fail.Exception, match="local pinned-tree command timed out"):
        pinned_tree_module._run_local(["git", "archive"])
    assert seen == [480]
