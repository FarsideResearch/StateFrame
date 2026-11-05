"""Startup probe failures retain bounded diagnostics before their temp tree is removed."""
from __future__ import annotations

import re
import sys

import pytest

from stateframe import runner
from stateframe.engine import kernel_safe_probe
from stateframe.engine.kernel_exec_session import ExecSession
from stateframe.engine.kernel_safe_probe import ProbeCleanupError
from kernel_exec_session import ExecSession as SmokeExecSession
from kernel_ship_tree_smoke import ImportSmokeCleanupError, _execute_smoke


@pytest.mark.host_sandbox
def test_supervisor_death_before_start_keeps_exit_status_log_tail_and_base_error(
        monkeypatch, tmp_path):
    marker = "planted supervisor log tail"

    class DiedBeforeStartSession(ExecSession):
        def _rpc(self, request):
            if request.get("op") == "start" and not getattr(self, "_death_planted", False):
                self._death_planted = True
                if self._process.poll() is None:
                    self._process.kill()
                self._process.wait(timeout=5)
                self._log.write((marker + "\n").encode())
            return super()._rpc(request)

    monkeypatch.setattr(kernel_safe_probe, "ExecSession", DiedBeforeStartSession)

    with pytest.raises(ProbeCleanupError) as caught:
        runner._startup_probe(sys.executable, "pass", label="engine interpreter test")

    message = str(caught.value)
    assert type(caught.value) is ProbeCleanupError
    assert message.startswith(
        "automatic probe cleanup failed: execution supervisor communication failed:")
    assert "primary error: execution supervisor communication failed:" in message
    assert re.search(r"supervisor exit status: -?\d+", message)
    assert "supervisor.log tail (up to 8192 bytes):" in message
    assert marker in message


def test_missing_supervisor_log_is_reported_as_absent(tmp_path):
    session = ExecSession(tmp_path)

    diagnostic = session.startup_diagnostics()

    assert "supervisor status unknown" in diagnostic
    assert "supervisor.log absent" in diagnostic


def test_symlinked_supervisor_log_is_not_followed(tmp_path):
    session = ExecSession(tmp_path)
    outside = tmp_path / "outside.log"
    outside.write_text("must not be read", encoding="utf-8")
    (session._directory / "supervisor.log").symlink_to(outside)

    diagnostic = session.startup_diagnostics()

    assert "supervisor.log absent" in diagnostic
    assert "must not be read" not in diagnostic


def test_log_tail_is_bounded_and_decodes_invalid_utf8_with_replacement(tmp_path):
    session = ExecSession(tmp_path)
    log = session._directory / "supervisor.log"
    log.write_bytes(b"old prefix " + b"A" * 9000 + b"\xfftail")

    diagnostic = session.startup_diagnostics()
    tail = diagnostic.split("supervisor.log tail (up to 8192 bytes):\n", 1)[1]

    assert tail.count("A") == 8187
    assert "old prefix" not in tail
    assert "\ufffd" in tail
    assert tail.endswith("tail")


@pytest.mark.host_sandbox
def test_import_smoke_supervisor_death_includes_diagnostics(monkeypatch, tmp_path):
    marker = "planted import smoke supervisor log tail"

    class DiedBeforeStartSession(SmokeExecSession):
        def _rpc(self, request):
            if request.get("op") == "start" and not getattr(self, "_death_planted", False):
                self._death_planted = True
                if self._process.poll() is None:
                    self._process.kill()
                self._process.wait(timeout=5)
                self._log.write((marker + "\n").encode())
            return super()._rpc(request)

    monkeypatch.setattr("kernel_exec_session.ExecSession", DiedBeforeStartSession)
    monkeypatch.setenv("STATEFRAME_TEST_PYTHON", sys.executable)
    (tmp_path / "smoke.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ImportSmokeCleanupError) as caught:
        _execute_smoke(tmp_path, ("smoke.py",), (str(tmp_path),), 3)

    message = str(caught.value)
    assert "execution supervisor communication failed:" in message
    assert "startup supervisor diagnostics:" in message
    assert re.search(r"supervisor exit status: -?\d+", message)
    assert marker in message
