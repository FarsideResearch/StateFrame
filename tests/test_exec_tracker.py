import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import kernel_exec_tracker as tracker
from kernel_exec_tracker import Descendants, ProcessTable
from fixtures.process_signals import signal_owned


def test_signal_permission_failure_preserves_live_member(tmp_path, monkeypatch):
    script = tmp_path / "wait.py"
    script.write_text("import time\ntime.sleep(10)\n")
    process = subprocess.Popen([sys.executable, "-I", str(script)], start_new_session=True)
    try:
        table = ProcessTable()
        family = Descendants(process.pid, table)
        live = family.update(table.snapshot())
        assert any(row.pid == process.pid for row in live)

        held_fd = os.open(os.devnull, os.O_RDONLY)
        monkeypatch.setattr(tracker, "sys", SimpleNamespace(platform="linux"))

        def open_pidfd(pid):
            assert pid == process.pid
            return os.dup(held_fd)

        monkeypatch.setattr(os, "pidfd_open", open_pidfd, raising=False)

        def denied(_pidfd, sig):
            assert sig == signal.SIGKILL
            raise PermissionError("test boundary denies signal")

        monkeypatch.setattr(signal, "pidfd_send_signal", denied, raising=False)
        try:
            assert family.signal(live, signal.SIGKILL) == set()
        finally:
            os.close(held_fd)
        assert family.signal_errors[process.pid] == "signal permission denied"
        assert any(row.pid == process.pid for row in family.update(table.snapshot()))
    finally:
        signal_owned(process, signal.SIGKILL)
        process.wait(timeout=3)
