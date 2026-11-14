"""Signal calls in tests are confined to identity-checking helpers."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from e2e_support import interrupt_run
from signal_guard import signal_violations
from process_signals import kernel_start_identity, record_identity, signal_owned, signal_verified


ROOT = Path(__file__).resolve().parents[2]


def test_signal_guard_rejects_new_test_and_source_sites(tmp_path: Path) -> None:
    test_file = tmp_path / "tests/planted.py"
    source_file = tmp_path / "src/stateframe/planted.py"
    test_file.parent.mkdir(parents=True)
    source_file.parent.mkdir(parents=True)
    test_file.write_text("def unsafe(pid):\n    os.kill(pid, 9)\n")
    source_file.write_text("def unsafe(pid):\n    os.killpg(pid, 9)\n")
    (test_file.parent / 'aliased.py').write_text(
        "from os import kill as terminate\n"
        "def unsafe(pid):\n    terminate(pid, 9)\n")
    (test_file.parent / 'embedded.py').write_text(
        "def generated():\n    return 'os.kill(1, 9)'\n")
    (test_file.parent / 'dynamic.py').write_text(
        'import os, signal, importlib\n'
        'def unsafe(pid):\n'
        '    getattr(os, "kill")(pid, 9)\n'
        '    os.__dict__["killpg"](pid, 9)\n'
        '    vars(os)["kill"](pid, 9)\n'
        '    getattr(signal, "pidfd_send_signal")(pid, 9)\n'
        '    getattr(importlib.import_module("os"), "getpgid")(pid)\n'
        '    later = getattr(os, "kill")\n'
        '    later(pid, 9)\n'
        '    module = importlib.import_module("os")\n'
        '    module.kill(pid, 9)\n'
        '    from signal import pidfd_send_signal as send\n'
        '    send(pid, 9)\n')
    violations = signal_violations(tmp_path, review_source=False)
    assert len(violations) == 12
    assert any("tests/planted.py::unsafe: os.kill(pid, 9)" in row for row in violations)
    assert any("src/stateframe/planted.py::unsafe: os.killpg(pid, 9)" in row for row in violations)
    assert any("tests/aliased.py::unsafe: terminate(pid, 9)" in row for row in violations)
    assert any("tests/embedded.py::<text>: embedded process signal reference" in row
               for row in violations)
    assert sum('tests/dynamic.py::unsafe:' in row for row in violations) == 8
    assert any('tests/dynamic.py::unsafe: vars(os)[\'kill\'](pid, 9)' in row
               for row in violations)


def test_whole_tree_signals_only_through_reviewed_sites() -> None:
    assert signal_violations(ROOT) == []


@pytest.mark.host_sandbox
def test_test_signal_reader_agrees_with_worker_kernel_identity() -> None:
    from stateframe.parallel_worker import kernel_start_identity as product_identity

    expected = product_identity(os.getpid())
    assert expected is not None
    assert kernel_start_identity(os.getpid()) == expected


@pytest.mark.host_sandbox
def test_signal_helpers_verify_launch_identity_before_delivery() -> None:
    launched_after = time.time()
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                               start_new_session=True)
    try:
        identity = record_identity(process.pid, launched_after=launched_after,
                                   command_fragment='time.sleep(30)')
        with pytest.raises(RuntimeError, match='no longer matches'):
            signal_verified(process.pid, type(identity)(process.pid,
                            identity.started_epoch, 'different command', identity.kernel_identity), signal.SIGTERM)
        with pytest.raises(RuntimeError, match='no longer matches'):
            signal_verified(process.pid, type(identity)(process.pid,
                            identity.started_epoch, identity.command, 'darwin:0:0'), signal.SIGTERM)
        assert process.poll() is None
        with pytest.raises(ValueError, match='invalid or self'):
            signal_verified(os.getpid(), type(identity)(os.getpid(),
                            identity.started_epoch, identity.command), signal.SIGTERM)
        signal_owned(process, signal.SIGTERM)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            signal_owned(process, signal.SIGKILL)
            process.wait(timeout=10)


def test_interrupt_run_ignores_unrelated_lock_pid(tmp_path: Path, monkeypatch) -> None:
    lock_path = tmp_path / ".engine.lock"
    with lock_path.open("w+") as lock:
        lock.write(str(os.getpid()))
        lock.flush()
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        class OwnedProcess:
            pid = os.getpid() + 100_000
            terminated = False

            def poll(self) -> int | None:
                return -signal.SIGTERM if self.terminated else None

            def terminate(self) -> None:
                self.terminated = True
                fcntl.flock(lock, fcntl.LOCK_UN)

            def wait(self, timeout: float) -> int:
                assert timeout > 0
                return -signal.SIGTERM

        def forbidden_signal(*_args: object) -> None:
            raise AssertionError("a PID read from a lock file was signalled")

        monkeypatch.setattr(os, "kill", forbidden_signal)
        monkeypatch.setattr(os, "killpg", forbidden_signal)
        process = OwnedProcess()
        interrupt_run(process, tmp_path)  # type: ignore[arg-type]
        assert process.terminated
