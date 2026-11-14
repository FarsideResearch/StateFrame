"""Portable liveness parsing and bounded waits must fail safely."""
from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest

from stateframe import parallel_host as host
from stateframe.engine import kernel_exec_tracker as tracker


def test_portable_inspection_keeps_live_group_without_command(monkeypatch):
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(host.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=f"123 123 S\n{os.getpid()} {os.getpgrp()} S observer\n"))
    assert host._process_table() == [(123, 123, "S", []),
                                    (os.getpid(), os.getpgrp(), "S", ["observer"])]
    assert host._group_has_live_processes(123)


@pytest.mark.parametrize("row", ["123 123", "invalid 123 S python"])
def test_portable_inspection_refuses_incomplete_process_table(monkeypatch, row):
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(host.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=row))
    with pytest.raises(host.ParallelHostError, match="process table probe returned"):
        host._group_has_live_processes(123)


@pytest.mark.parametrize("setting", ["1e100", "1e308"])
def test_large_finite_wait_cannot_overflow_platform_poll(monkeypatch, setting):
    monkeypatch.setenv("STATEFRAME_PARALLEL_INSPECT_SECONDS", setting)
    monkeypatch.setattr(tracker.os, "getloadavg", lambda: (100, 100, 100))
    monkeypatch.setattr(tracker.os, "cpu_count", lambda: 1)
    seconds = tracker.load_tolerant_seconds("STATEFRAME_PARALLEL_INSPECT_SECONDS")
    assert math.isfinite(seconds)
    assert seconds >= 5
    assert seconds * 1000 < 2**31 - 1
