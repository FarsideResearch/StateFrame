"""Cgroup cleanup treats vanished groups as already removed."""
from __future__ import annotations

import errno
import logging
from pathlib import Path

import kernel_execution_policy as policy
import sf_launch


def _cgroup(tmp_path: Path, *, control_file: bool = True) -> Path:
    group = tmp_path / "command"
    group.mkdir()
    if control_file:
        (group / "cgroup.procs").write_text("", encoding="ascii")
    return group


def test_already_removed_group_succeeds_without_retry_or_error_log(
        tmp_path, monkeypatch, caplog):
    group = tmp_path / "already-removed"
    rmdir_calls: list[Path] = []
    sleeps: list[float] = []
    real_rmdir = Path.rmdir

    def track_rmdir(path):
        rmdir_calls.append(path)
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", track_rmdir)
    monkeypatch.setattr(policy.time, "sleep", sleeps.append)
    with caplog.at_level(logging.ERROR, logger=policy._LOGGER.name):
        assert policy._remove_command_cgroup(group)

    assert rmdir_calls == []
    assert sleeps == []
    assert caplog.records == []


def test_rmdir_enoent_after_concurrent_removal_succeeds_without_retry(
        tmp_path, monkeypatch, caplog):
    group = _cgroup(tmp_path)
    rmdir_calls: list[Path] = []
    sleeps: list[float] = []
    real_rmdir = Path.rmdir

    def remove_then_report_enoent(path):
        if path == group:
            rmdir_calls.append(path)
            (path / "cgroup.procs").unlink()
            real_rmdir(path)
            raise FileNotFoundError(errno.ENOENT, "already removed", str(path))
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", remove_then_report_enoent)
    monkeypatch.setattr(policy.time, "sleep", sleeps.append)
    with caplog.at_level(logging.ERROR, logger=policy._LOGGER.name):
        assert policy._remove_command_cgroup(group)

    assert rmdir_calls == [group]
    assert sleeps == []
    assert caplog.records == []


def test_missing_cgroup_procs_is_treated_as_empty(tmp_path):
    group = _cgroup(tmp_path, control_file=False)

    assert policy._remove_command_cgroup(group)
    assert not group.exists()


def test_busy_rmdir_retries_twice_then_succeeds(tmp_path, monkeypatch, caplog):
    group = _cgroup(tmp_path)
    attempts = 0
    sleeps: list[float] = []
    real_rmdir = Path.rmdir

    def busy_twice(path):
        nonlocal attempts
        if path == group:
            attempts += 1
            if attempts <= 2:
                raise OSError(errno.EBUSY, "group is busy", str(path))
            (path / "cgroup.procs").unlink()
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", busy_twice)
    monkeypatch.setattr(policy.time, "sleep", sleeps.append)
    with caplog.at_level(logging.ERROR, logger=policy._LOGGER.name):
        assert policy._remove_command_cgroup(group)

    assert attempts == 3
    assert sleeps == [policy.CGROUP_CLEANUP_INTERVAL_SECONDS] * 2
    assert caplog.records == []
    assert not group.exists()


def test_persistent_ebusy_reports_cleanup_failure(tmp_path, monkeypatch, caplog):
    group = _cgroup(tmp_path)
    attempts = 0
    real_rmdir = Path.rmdir
    monkeypatch.setattr(policy.time, "sleep", lambda _seconds: None)

    def always_busy(path):
        nonlocal attempts
        if path == group:
            attempts += 1
            raise OSError(errno.EBUSY, "group is busy", str(path))
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", always_busy)
    with caplog.at_level(logging.ERROR, logger=policy._LOGGER.name):
        assert not policy._remove_command_cgroup(group)

    assert attempts == policy.CGROUP_CLEANUP_ATTEMPTS
    assert len(caplog.records) == 1
    assert "after 40 attempts (OSError)" in caplog.records[0].getMessage()


def test_leaf_rollback_treats_missing_group_as_removed(tmp_path):
    parent = tmp_path / "delegated"
    parent.mkdir()
    (parent / "cgroup.subtree_control").write_text("", encoding="ascii")

    assert sf_launch._rollback_leaf_pattern(
        parent, parent / "supervisor", set()) == (True, "")
