"""The Linux engine launch remains identifiable by the gated mutation tests."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from stateframe import runner
from fixtures.process_signals import signal_run_owned
from test_mutation_linux_regressions import _engine_child_identity


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process identity")
def test_linux_engine_launch_is_recognized_as_direct_child(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    captured = []

    class LaunchCaptured(Exception):
        pass

    def capture(command, *args, **kwargs):
        captured.append((command, kwargs))
        raise LaunchCaptured

    monkeypatch.setattr(runner.subprocess, "Popen", capture)
    options = runner.RunOptions(repo=tmp_path, task="identify engine", endpoint=None,
                                python=sys.executable)
    with pytest.raises(LaunchCaptured):
        runner.drive(tmp_path, options)

    command, kwargs = captured[0]
    assert command[1:3] == ["-B", "-c"]
    assert command[3].startswith("'stateframe._child.py';")
    assert command[4:] == ["stateframe._child"]
    monkeypatch.undo()
    probe = command.copy()
    probe[3] = "'stateframe._child.py';import time;time.sleep(10)"
    child = original_popen(probe, executable=kwargs["executable"], cwd=kwargs["cwd"])
    try:
        assert _engine_child_identity(child.pid) == (os.getpid(), "engine")
    finally:
        signal_run_owned(tmp_path, child, signal.SIGTERM, deadline=time.monotonic(),
                         what="identity probe cleanup", require_checkpoint_saved=False)
        child.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process identity")
def test_later_code_marker_does_not_identify_unrelated_child():
    child = subprocess.Popen(
        [sys.executable, "-B", "-c",
         "import time;'stateframe._child.py';time.sleep(10)"],
    )
    try:
        assert _engine_child_identity(child.pid) is None
    finally:
        signal_run_owned(Path.cwd(), child, signal.SIGTERM, deadline=time.monotonic(),
                         what="unrelated child cleanup", require_checkpoint_saved=False)
        child.wait(timeout=5)
