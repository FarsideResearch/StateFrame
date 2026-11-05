"""Process signal fixtures distinguish zombies from reused live pids."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import kernel_pid_liveness
from fixtures import process_signals

pytestmark = pytest.mark.host_sandbox


def _plant_proc_state(monkeypatch, tmp_path: Path, pid: int, state: str) -> None:
    proc_root = tmp_path / "proc"
    stat = proc_root / str(pid) / "stat"
    stat.parent.mkdir(parents=True)
    fields = [state, "1", str(pid), str(pid), "0", "-1", "4194304"]
    fields.extend("0" for _ in range(13))
    fields[19] = "12345"
    stat.write_text(f"{pid} (python) {' '.join(fields)}\n", encoding="ascii")
    cmdline = proc_root / str(pid) / "cmdline"
    cmdline.write_bytes(b"python\0-c\0import time; time.sleep(30)\0")
    boot_id = proc_root / "sys/kernel/random/boot_id"
    boot_id.parent.mkdir(parents=True)
    boot_id.write_text("01234567-89ab-cdef-0123-456789abcdef\n", encoding="ascii")
    monkeypatch.setattr(process_signals, "PROC", proc_root)
    monkeypatch.setattr(
        kernel_pid_liveness.pid_alive,
        "__kwdefaults__",
        {"platform": "linux", "proc": proc_root},
    )


@pytest.fixture
def live_child():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("identity_check", ["record", "inspect"])
def test_zombie_pid_is_gone(monkeypatch, tmp_path, live_child, identity_check):
    _plant_proc_state(monkeypatch, tmp_path, live_child.pid, "Z")

    with pytest.raises(ProcessLookupError, match=f"process {live_child.pid} is no longer running") as error:
        if identity_check == "record":
            # The launch window would fail for a live, mismatched process. Zombie state wins.
            process_signals.record_identity(
                live_child.pid,
                launched_after=time.time() + 120,
                command_fragment="time.sleep(30)",
            )
        else:
            process_signals._process_identity(live_child.pid)
    assert str(error.value) == f"process {live_child.pid} is no longer running"


@pytest.mark.parametrize("identity_check", ["record", "signal"])
def test_live_pid_with_different_identity_remains_mismatch(
        monkeypatch, tmp_path, live_child, identity_check):
    _plant_proc_state(monkeypatch, tmp_path, live_child.pid, "S")

    if identity_check == "record":
        with pytest.raises(RuntimeError, match="does not match its launch identity") as error:
            process_signals.record_identity(
                live_child.pid, launched_after=0, command_fragment="not-the-child-command")
        expected_fragment = "does not match its launch identity"
    else:
        identity = process_signals._process_identity(live_child.pid)
        different_identity = replace(identity, kernel_identity="linux:old-boot:1")
        original_kill = process_signals.os.kill
        original_killpg = process_signals.os.killpg

        def refuse_signal(pid, signal_number):
            if signal_number != 0:
                pytest.fail("a mismatched identity must not be signalled")
            return original_kill(pid, signal_number)

        def refuse_group_signal(pgid, signal_number):
            if signal_number != 0:
                pytest.fail("a mismatched identity must not be signalled")
            return original_killpg(pgid, signal_number)

        with monkeypatch.context() as patch:
            patch.setattr(process_signals.os, "kill", refuse_signal)
            patch.setattr(process_signals.os, "killpg", refuse_group_signal)
            with pytest.raises(RuntimeError, match="no longer matches its recorded identity") as error:
                process_signals.signal_verified(live_child.pid, different_identity, signal.SIGTERM)
        expected_fragment = "no longer matches its recorded identity"
    assert expected_fragment in str(error.value)


def test_live_pid_with_same_identity_passes(monkeypatch, tmp_path, live_child):
    _plant_proc_state(monkeypatch, tmp_path, live_child.pid, "S")

    identity = process_signals.record_identity(
        live_child.pid, launched_after=0, command_fragment="time.sleep(30)")
    binding = process_signals.retain_signal_binding(identity)
    try:
        path = process_signals.signal_bound(binding, signal.SIGCONT)
        if sys.platform == "linux":
            assert path == "pidfd"
        else:
            from kernel_exec_tracker import MAC_FALLBACK_SIGNAL_PATH, MAC_KERNEL_SIGNAL_PATH

            assert path in {MAC_FALLBACK_SIGNAL_PATH, MAC_KERNEL_SIGNAL_PATH}
        assert live_child.poll() is None
    finally:
        process_signals.close_signal_binding(binding)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd signal authority")
def test_retained_signal_binding_uses_pidfd(live_child):
    identity = process_signals._process_identity(live_child.pid)
    binding = process_signals.retain_signal_binding(identity)
    try:
        assert binding.pidfd is not None
        assert process_signals.signal_bound(binding, signal.SIGCONT) == "pidfd"
    finally:
        process_signals.close_signal_binding(binding)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd signal authority")
def test_linux_binding_refuses_without_pidfd(monkeypatch, live_child):
    identity = process_signals._process_identity(live_child.pid)

    def denied(_pid):
        raise PermissionError("pidfd denied by test")

    monkeypatch.setattr(process_signals.os, "pidfd_open", denied)
    with pytest.raises(PermissionError, match="pidfd denied by test") as error:
        process_signals.retain_signal_binding(identity)
    assert str(error.value) == "pidfd denied by test"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux exiting-process cmdline evidence")
def test_exiting_empty_cmdline_is_gone_before_launch_identity_mismatch(
        monkeypatch, tmp_path, live_child):
    _plant_proc_state(monkeypatch, tmp_path, live_child.pid, "R")
    (tmp_path / "proc" / str(live_child.pid) / "cmdline").write_bytes(b"")
    launched_at = time.time()
    started = datetime.fromtimestamp(launched_at).strftime("%a %b %d %H:%M:%S %Y")
    result = subprocess.CompletedProcess(
        ["ps"], 0, stdout=f"{started} python -c import time; time.sleep(30)\n", stderr="")
    monkeypatch.setattr(process_signals.subprocess, "run", lambda *_args, **_kwargs: result)

    with pytest.raises(ProcessLookupError, match=f"process {live_child.pid} is no longer running") as error:
        process_signals.record_identity(
            live_child.pid, launched_after=launched_at - 1, command_fragment="time.sleep(30)")
    assert str(error.value) == f"process {live_child.pid} is no longer running"


def test_nonempty_cmdline_mismatch_remains_a_runtime_error(
        monkeypatch, tmp_path, live_child):
    _plant_proc_state(monkeypatch, tmp_path, live_child.pid, "R")

    with pytest.raises(RuntimeError, match="does not match its launch identity") as error:
        process_signals.record_identity(
            live_child.pid, launched_after=0, command_fragment="not-the-child-command")
    assert "does not match its launch identity" in str(error.value)
