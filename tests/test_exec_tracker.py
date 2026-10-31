import os
import signal
import subprocess
import sys

from kernel_exec_tracker import Descendants, ProcessTable


def test_signal_permission_failure_preserves_live_member(tmp_path, monkeypatch):
    script = tmp_path / "wait.py"
    script.write_text("import time\ntime.sleep(10)\n")
    process = subprocess.Popen([sys.executable, "-I", str(script)], start_new_session=True)
    actual_kill = os.kill
    try:
        table = ProcessTable()
        family = Descendants(process.pid, table)
        live = family.update(table.snapshot())
        assert any(row.pid == process.pid for row in live)

        def denied(pid, sig):
            if pid == process.pid:
                raise PermissionError("test boundary denies signal")
            return actual_kill(pid, sig)

        monkeypatch.setattr(os, "kill", denied)
        assert family.signal(live, signal.SIGKILL) == set()
        assert family.signal_errors[process.pid] == "signal permission denied"
        assert any(row.pid == process.pid for row in family.update(table.snapshot()))
    finally:
        actual_kill(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
