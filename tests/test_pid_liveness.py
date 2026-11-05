"""Temp sweeps treat a Linux zombie owner as dead; macOS keeps plain signal-zero liveness."""
from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kernel_checkpoint
import kernel_pid_liveness as liveness


def _fake_proc(root: Path, pid: int, state: str, name: str = "python", *, pgrp: int | None = None) -> Path:
    stat = root / str(pid) / "stat"
    stat.parent.mkdir(parents=True)
    process_group_id = pid if pgrp is None else pgrp
    stat.write_text(f"{pid} ({name}) {state} 1 {process_group_id} {pid} 0 -1 4194304\n")
    return root


def _allow_group_probe(monkeypatch):
    def killpg(_pgid, signal_number):
        assert signal_number == 0

    monkeypatch.setattr(liveness.os, "killpg", killpg)


def test_linux_zombie_owner_counts_as_dead(tmp_path):
    proc = _fake_proc(tmp_path, os.getpid(), "Z")
    assert liveness.pid_alive(os.getpid(), platform="linux", proc=proc) is False


def test_linux_running_owner_stays_alive(tmp_path):
    proc = _fake_proc(tmp_path, os.getpid(), "S")
    assert liveness.pid_alive(os.getpid(), platform="linux", proc=proc) is True


def test_macos_keeps_signal_zero_answer_even_if_proc_says_zombie(tmp_path):
    proc = _fake_proc(tmp_path, os.getpid(), "Z")
    assert liveness.pid_alive(os.getpid(), platform="darwin", proc=proc) is True


def test_state_is_read_after_the_last_paren_of_the_command_name(tmp_path):
    proc = _fake_proc(tmp_path, os.getpid(), "Z", name="evil) S (name")
    assert liveness.linux_zombie(os.getpid(), proc) is True


def test_unreadable_stat_is_not_provably_a_zombie(tmp_path):
    assert liveness.linux_zombie(os.getpid(), tmp_path) is False


def test_exited_pid_is_dead():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert liveness.pid_alive(child.pid) is False
    assert liveness.pid_alive(0) is False


def test_checkpoint_sweep_uses_zombie_aware_liveness(monkeypatch, tmp_path):
    proc = _fake_proc(tmp_path, os.getpid(), "Z")
    monkeypatch.setattr(liveness, "PROC", proc)
    monkeypatch.setattr(liveness.pid_alive, "__kwdefaults__", {"platform": "linux", "proc": proc})
    assert kernel_checkpoint._pid_alive(os.getpid()) is False


def test_env_seed_sweep_routes_through_pid_alive():
    import inspect
    import kernel_env_seed
    source = inspect.getsource(kernel_env_seed)
    assert "pid_alive(pid)" in source
    assert "os.kill(pid, 0)" not in source


def test_linux_process_group_with_only_zombies_counts_as_gone(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, 101, "Z", pgrp=73)
    _fake_proc(proc, 102, "Z", pgrp=73)
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is False


def test_linux_process_group_with_running_member_stays_alive(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, 101, "Z", pgrp=73)
    _fake_proc(proc, 102, "S", pgrp=73)
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is True


def test_linux_process_group_without_visible_members_counts_as_gone(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "101" / "stat").parent.mkdir()
    (proc / "101" / "stat").symlink_to(proc / "missing-stat")
    (proc / "102" / "stat").mkdir(parents=True)
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is False


def test_linux_process_group_ignores_non_pid_proc_entries(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, 101, "Z", pgrp=73)
    (proc / "self").mkdir()
    (proc / "self" / "stat").write_text("999 (decoy) S 1 73 999 0 -1 4194304\n")
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is False


@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError])
def test_linux_process_group_ignores_unreadable_or_vanished_stat(monkeypatch, tmp_path, failure):
    proc = _fake_proc(tmp_path / "proc", 101, "S", pgrp=73)
    _fake_proc(proc, 102, "Z", pgrp=73)
    unreadable = proc / "101" / "stat"
    original_read_bytes = Path.read_bytes

    def read_bytes(path):
        if path == unreadable:
            raise failure("proc entry unavailable")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is False


def test_linux_process_group_uses_third_field_after_comm(monkeypatch, tmp_path):
    proc = _fake_proc(tmp_path / "proc", 101, "S", pgrp=73)
    stat = proc / "101" / "stat"
    stat.write_text("101 (worker) S 73 99 101 0 -1 4194304\n")
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is False
    assert liveness.group_alive(99, platform="linux", proc=proc) is True


def test_linux_process_group_uses_pgrp_after_last_comm_parenthesis(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    stat = proc / "101" / "stat"
    stat.parent.mkdir(parents=True)
    # The apparent Z state and pgrp 999 are inside comm; actual fields follow the last ')'.
    stat.write_text("101 (worker) Z 1 999 (decoy) S 1 73 0 -1 4194304\n")
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="linux", proc=proc) is True
    assert liveness.group_alive(999, platform="linux", proc=proc) is False


def test_macos_uses_killpg_answer_without_reading_proc(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, 101, "Z", pgrp=73)
    _allow_group_probe(monkeypatch)

    assert liveness.group_alive(73, platform="darwin", proc=proc) is True


def test_permission_denied_group_probe_stays_alive(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, 101, "Z", pgrp=73)

    def deny_group_probe(_pgid, _signal_number):
        raise PermissionError("process-group probe denied")

    monkeypatch.setattr(liveness.os, "killpg", deny_group_probe)

    assert liveness.group_alive(73, platform="linux", proc=proc) is True


@pytest.mark.parametrize("error, expected", [
    (OSError(errno.EPERM, "probe denied"), True),
    (OSError(errno.ESRCH, "group missing"), False),
])
def test_group_probe_handles_errno_without_scanning_proc(monkeypatch, tmp_path, error, expected):
    def failed_probe(_pgid, _signal_number):
        raise error

    monkeypatch.setattr(liveness.os, "killpg", failed_probe)

    assert liveness.group_alive(73, platform="linux", proc=tmp_path / "missing") is expected


def test_group_probe_propagates_unexpected_os_error(monkeypatch):
    def failed_probe(_pgid, _signal_number):
        raise OSError(errno.EINVAL, "invalid group probe")

    monkeypatch.setattr(liveness.os, "killpg", failed_probe)

    with pytest.raises(OSError, match="invalid group probe"):
        liveness.group_alive(73, platform="linux")


def test_missing_group_is_gone_even_if_proc_contains_a_live_member(monkeypatch, tmp_path):
    proc = _fake_proc(tmp_path / "proc", 101, "S", pgrp=73)

    def missing_group(_pgid, _signal_number):
        raise ProcessLookupError("process group does not exist")

    monkeypatch.setattr(liveness.os, "killpg", missing_group)

    assert liveness.group_alive(73, platform="linux", proc=proc) is False
