import json
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from fixtures.process_signals import signal_run_owned

from kernel_exec_tracker import ProcessTable, process_cmdline_digest
from kernel_mutate_sweep import sweep
import kernel_mutate_sweep


def owner_record(boot_id="test"):
    return {"boot_id": boot_id, "identity": {
        "pid": 99999999, "start": [0, 0], "boot_id": boot_id,
        "ns": "pid:[99999999]", "cmdline_sha256": "0" * 64,
    }}


def launch_identity(pid, *, boot_id="test"):
    table = ProcessTable()
    row = table.identity(pid)
    assert row is not None
    return {"pid": pid, "start": list(row.start), "boot_id": boot_id,
            "ns": table.pid_namespace(pid), "cmdline_sha256": process_cmdline_digest(pid)}


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(kernel_mutate_sweep.os, "pidfd_open")
    or not hasattr(kernel_mutate_sweep.signal, "pidfd_send_signal"),
    reason="Linux pidfd signaling provides identity-safe orphan cleanup",
)
def test_sweep_kills_identity_verified_child(tmp_path):
    root = tmp_path / "run1"
    root.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        identity = launch_identity(child.pid)
        (root / "owner.json").write_text(json.dumps(owner_record()))
        (root / "launches.jsonl").write_text(json.dumps({"schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c", "role": "payload", "state": "REGISTERED", "identity": identity}) + "\n")
        result = sweep(tmp_path, run_locks=lambda name: False, boot_id="test")
        assert "run1" in result.cleaned
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            signal_run_owned(root, child, signal.SIGKILL, deadline=time.monotonic(),
                             what="sweep fixture cleanup", require_checkpoint_saved=False)
            child.wait()


def test_sweep_refuses_identity_mismatch(tmp_path):
    root = tmp_path / "run2"
    root.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (root / "owner.json").write_text(json.dumps(owner_record()))
        (root / "launches.jsonl").write_text(json.dumps({"schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c", "role": "payload", "state": "REGISTERED", "identity": {"pid": child.pid, "start": [0, 0], "boot_id": "test"}}) + "\n")
        result = sweep(tmp_path, run_locks=lambda name: False, boot_id="test")
        assert "run2" in result.unresolved
        assert root.exists()
        assert child.poll() is None
    finally:
        signal_run_owned(root, child, signal.SIGKILL, deadline=time.monotonic(),
                         what="sweep fixture cleanup", require_checkpoint_saved=False)
        child.wait()


def test_sweep_refuses_root_symlink_swap(tmp_path, monkeypatch):
    root = tmp_path / "run3"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps(owner_record()))
    outside = tmp_path / "outside"
    outside.mkdir()
    original = kernel_mutate_sweep._remove_tree
    def swap(fd, *, parent_fd, name, inode):
        root.rename(tmp_path / "moved")
        root.symlink_to(outside, target_is_directory=True)
        return original(fd, parent_fd=parent_fd, name=name, inode=inode)
    monkeypatch.setattr(kernel_mutate_sweep, "_remove_tree", swap)
    result = sweep(tmp_path, run_locks=lambda name: False, boot_id="test")
    assert "run3" in result.unresolved
    assert outside.exists()


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(kernel_mutate_sweep.os, "pidfd_open")
    or not hasattr(kernel_mutate_sweep.signal, "pidfd_send_signal"),
    reason="pidfd is Linux only",
)
def test_sweep_signals_pinned_pidfd(tmp_path, monkeypatch):
    root = tmp_path / "run4"
    root.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        identity = launch_identity(child.pid)
        (root / "owner.json").write_text(json.dumps(owner_record()))
        (root / "launches.jsonl").write_text(json.dumps({"schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c", "role": "payload", "state": "REGISTERED", "identity": identity}) + "\n")
        with monkeypatch.context() as patch:
            patch.setattr(kernel_mutate_sweep.os, "kill", lambda *_: (_ for _ in ()).throw(AssertionError("un-pinned signal")))
            assert "run4" in sweep(tmp_path, run_locks=lambda name: False, boot_id="test").cleaned
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            signal_run_owned(root, child, signal.SIGKILL, deadline=time.monotonic(),
                             what="sweep fixture cleanup", require_checkpoint_saved=False)
            child.wait()


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(kernel_mutate_sweep.os, "pidfd_open")
    or not hasattr(kernel_mutate_sweep.signal, "pidfd_send_signal"),
    reason="Linux pidfd and procfs identity checks are required",
)
def test_sweep_refuses_same_tick_pid_reuse_by_command_identity(tmp_path, monkeypatch):
    root = tmp_path / "same-tick-reuse"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps(owner_record()))
    replacement_pid = 4242
    recorded = {"pid": replacement_pid, "start": [77, 0], "boot_id": "test",
                "ns": "pid:[4242]", "cmdline_sha256": "a" * 64}
    (root / "launches.jsonl").write_text(json.dumps({
        "schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c",
        "role": "payload", "state": "REGISTERED", "identity": recorded,
    }) + "\n")

    class SameTickTable:
        def identity(self, pid):
            if pid == replacement_pid:
                return SimpleNamespace(pid=pid, start=(77, 0), zombie=False)
            return None

        @staticmethod
        def pid_namespace(pid):
            assert pid == replacement_pid
            return "pid:[4242]"

    read_fd, write_fd = kernel_mutate_sweep.os.pipe()
    signals = []

    def open_pidfd(pid):
        assert pid == replacement_pid
        return kernel_mutate_sweep.os.dup(read_fd)

    monkeypatch.setattr(kernel_mutate_sweep, "ProcessTable", SameTickTable)
    monkeypatch.setattr(kernel_mutate_sweep.os, "pidfd_open", open_pidfd)
    monkeypatch.setattr(kernel_mutate_sweep, "process_cmdline_digest", lambda _pid: "b" * 64)
    monkeypatch.setattr(kernel_mutate_sweep.signal, "pidfd_send_signal",
                        lambda *args: signals.append(args))
    try:
        result = sweep(tmp_path, run_locks=lambda _name: False, boot_id="test")
    finally:
        kernel_mutate_sweep.os.close(read_fd)
        kernel_mutate_sweep.os.close(write_fd)

    assert result.unresolved == ("same-tick-reuse",)
    assert signals == []
    assert json.loads((root / "UNRESOLVED.json").read_text()) == {
        "kind": "UNRESOLVED", "reason": "process command identity differs",
    }
    assert root.exists()


def test_previous_boot_root_is_removed_without_signaling_reused_pid(tmp_path):
    root = tmp_path / "old-run"
    root.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (root / "owner.json").write_text(json.dumps(owner_record("previous")))
        (root / "launches.jsonl").write_text(json.dumps({"schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c", "role": "payload", "state": "REGISTERED", "identity": {"pid": child.pid, "start": [0, 0], "boot_id": "previous"}}) + "\n")
        result = sweep(tmp_path, run_locks=lambda name: False, boot_id="current")
        assert "old-run" in result.cleaned
        assert child.poll() is None
    finally:
        signal_run_owned(root, child, signal.SIGKILL, deadline=time.monotonic(),
                         what="sweep fixture cleanup", require_checkpoint_saved=False)
        child.wait()


def test_sweep_missing_parent_is_empty(tmp_path):
    assert sweep(tmp_path / "absent", run_locks=lambda name: False).cleaned == ()


def test_sweep_skips_active_run_lock(tmp_path):
    root = tmp_path / "active-run"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps(owner_record()))
    result = sweep(tmp_path, run_locks=lambda name: True, boot_id="test")
    assert result.cleaned == result.unresolved == ()
    assert root.exists()


def test_sweep_refuses_duplicate_owner_keys(tmp_path):
    root = tmp_path / "corrupt-run"
    root.mkdir()
    (root / "owner.json").write_bytes(b'{"boot_id":"old","boot_id":"test"}')
    result = sweep(tmp_path, run_locks=lambda name: False, boot_id="test")
    assert "corrupt-run" in result.unresolved
    assert root.exists()


def test_sweep_keeps_unidentified_started_launch(tmp_path):
    root = tmp_path / "unknown-launch"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps(owner_record()))
    (root / "launches.jsonl").write_text(json.dumps({"schema": "stateframe-mutation-launch/1", "launch": "l", "call": "c", "role": "payload", "state": "STARTED", "identity": None}) + "\n")
    result = sweep(tmp_path, run_locks=lambda name: False, boot_id="test")
    assert "unknown-launch" in result.unresolved
    assert root.exists()


def test_sweep_opens_records_nonblocking(tmp_path, monkeypatch):
    root = tmp_path / "safe-open"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps(owner_record()))
    (root / "launches.jsonl").write_text("")
    original = kernel_mutate_sweep.os.open
    def guarded(path, flags, *args, **kwargs):
        if path in ("owner.json", "launches.jsonl") and not flags & kernel_mutate_sweep.os.O_NONBLOCK:
            raise AssertionError("record open could block on a special file")
        return original(path, flags, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(kernel_mutate_sweep.os, "open", guarded)
        assert "safe-open" in sweep(tmp_path, run_locks=lambda name: False, boot_id="test").cleaned


def test_sweep_keeps_root_without_supervisor_identity(tmp_path):
    root = tmp_path / "missing-owner-identity"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps({"boot_id": "test"}))
    result = sweep(tmp_path, run_locks=lambda name: False, boot_id="test")
    assert "missing-owner-identity" in result.unresolved
    assert root.exists()


def test_macos_pid_reuse_between_identity_check_and_signal(monkeypatch):
    import kernel_mutate_sweep as module
    from types import SimpleNamespace
    starts = iter([SimpleNamespace(pid=123, start=(1, 2), zombie=False),
                   SimpleNamespace(pid=123, start=(1, 2), zombie=False)])
    class Table:
        def identity(self, _pid):
            return next(starts)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.os, "kill", lambda *_args: pytest.fail("un-pinned signal"))
    with pytest.raises(ValueError, match="pinned"):
        module._kill_recorded(Table(), {"pid": 123, "start": [1, 2]})


def test_macos_boot_identity_is_not_unknown(monkeypatch):
    import kernel_mutate_sweep as module
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda *_args, **_kwargs: b"{ sec = 123456789, usec = 42 }\n")
    assert module._boot_id() == "macos:{ sec = 123456789, usec = 42 }"
