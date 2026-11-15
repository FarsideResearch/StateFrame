"""Loaded hosts must not turn slow observations into failed subframes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from stateframe import parallel_host as host, parallel_resources as resources, parallel_worker as worker, runner
from stateframe.engine import kernel_exec_tracker as tracker, kernel_parallel as parallel
from stateframe.engine.kernel_parallel_records import HostAnswer


@pytest.mark.parametrize("raw,expected", [(None, 30), ("7", 7), ("nan", 30), ("inf", 30),
                                           ("-1", 30), ("0", 30), ("bad", 30)])
@pytest.mark.parametrize("load,factor", [(0, 1), (4, 1), (16, 4), (1000, 8)])
def test_cleanup_allowance_scales_but_never_shrinks_default(monkeypatch, raw, expected, load, factor):
    monkeypatch.setattr(tracker.os, "getloadavg", lambda: (load, load, load))
    monkeypatch.setattr(tracker.os, "cpu_count", lambda: 4)
    if raw is None:
        monkeypatch.delenv("STATEFRAME_PARALLEL_CLEANUP_SECONDS", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_PARALLEL_CLEANUP_SECONDS", raw)
    assert worker.cleanup_seconds() == expected * factor


def test_runner_and_supervisor_recognize_the_same_timing_settings():
    assert set(runner.PARALLEL_TIMING_SETTINGS) == set(tracker.PARALLEL_TIMING_DEFAULTS)


def test_process_liveness_on_linux_never_spawns_ps(monkeypatch):
    if host.sys.platform != "linux":
        pytest.skip("Linux /proc liveness")

    def forbidden(*args, **kwargs):
        pytest.fail("Linux process inspection spawned a subprocess")

    monkeypatch.setattr(host.subprocess, "run", forbidden)
    assert any(pid == os.getpid() and pgid == os.getpgrp() and state != "Z"
               for pid, pgid, state, _ in host._process_table())


@pytest.mark.parametrize("exhausted", [False, True])
def test_fallback_process_inspection_retries_timeout_without_proving_quiescence(monkeypatch, exhausted):
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(tracker.os, "getloadavg", lambda: (16, 16, 16))
    monkeypatch.setattr(tracker.os, "cpu_count", lambda: 4)
    monkeypatch.setenv("STATEFRAME_PARALLEL_INSPECT_SECONDS", "10")
    calls = []

    def inspect(*args, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1 or exhausted:
            raise subprocess.TimeoutExpired("ps", kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout=f"9876 9876 S child\n{os.getpid()} {os.getpgrp()} S observer\n")

    monkeypatch.setattr(host.subprocess, "run", inspect)
    if exhausted:
        with pytest.raises(host.ProcessInspectionDeferred):
            host._group_has_live_processes(9876)
    else:
        assert host._group_has_live_processes(9876)
    assert calls == [40, 40]


def test_linux_inspection_stays_unknown_when_proc_is_restricted(monkeypatch):
    monkeypatch.setattr(host, "sys", SimpleNamespace(platform="linux"))

    def unavailable():
        raise PermissionError("restricted proc")

    monkeypatch.setattr(host, "ProcessTable", lambda: SimpleNamespace(snapshot=unavailable))
    monkeypatch.setattr(host.subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "failed authoritative scan fell back to weaker authority"))
    with pytest.raises(host.ProcessInspectionDeferred):
        host._group_has_live_processes(9876)


def test_memory_inspection_retries_transient_timeout(monkeypatch):
    monkeypatch.setattr(resources, "sys", SimpleNamespace(platform="darwin"))
    calls = []

    def inspect(*args, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("memory_pressure", kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="System-wide memory free percentage: 70%")

    monkeypatch.setattr(resources.subprocess, "run", inspect)
    assert resources._memory_available() == (70.0, "memory_pressure")
    assert len(calls) == 2 and all(seconds >= 15 for seconds in calls)


def test_slow_worker_gets_watchdog_and_termination_slack(tmp_path, monkeypatch):
    (tmp_path / "spec.json").write_text(json.dumps({"environment": {"STATEFRAME_RUN_WALL_SECONDS": "60"}}))
    clock = [90.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tracker.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.delenv("STATEFRAME_PARALLEL_WATCHDOG_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("STATEFRAME_PARALLEL_CLEANUP_SECONDS", raising=False)
    monkeypatch.setattr(worker.WorkerHandle, "poll", lambda self: None)
    signals = []
    monkeypatch.setattr(worker.WorkerHandle, "_signal_group", lambda self, sig: signals.append(sig))
    handle = worker.WorkerHandle(SimpleNamespace(), 9876, 9876, 1, tmp_path, tmp_path / "log", 0, 0,
                                  SimpleNamespace(), launch_identity=tracker.LaunchIdentity(9876, None, None))
    handle.enforce_deadline()
    handle.escalate()
    assert not signals and not handle.timed_out
    clock[0] = 121
    handle.enforce_deadline()
    assert handle.timed_out and signals == [signal.SIGTERM]
    clock[0] = 150
    handle.escalate()
    assert signals == [signal.SIGTERM]
    clock[0] = 151
    handle.escalate()
    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize("name", runner.PARALLEL_TIMING_SETTINGS)
def test_timing_knobs_are_recorded_validated_and_ignored_when_parallel_off(monkeypatch, name):
    for setting in runner.PARALLEL_TIMING_SETTINGS:
        monkeypatch.delenv(setting, raising=False)
    monkeypatch.setenv(name, " 42.5 ")
    assert runner.resolve_parallel_timing_env(parallel_enabled=True) == {name: "42.5"}
    assert name not in runner.ignored_shell_settings()
    monkeypatch.setenv(name, "nan")
    assert runner.resolve_parallel_timing_env(parallel_enabled=False) == {}
    with pytest.raises(runner.RunError, match="positive finite"):
        runner.resolve_parallel_timing_env(parallel_enabled=True)


@pytest.mark.parametrize("op", ["status", "seal"])
def test_pending_seal_reference_does_not_extend_host_response_deadline(monkeypatch, op):
    clock = [0.0]
    monkeypatch.setattr(parallel.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(tracker.os, "getloadavg", lambda: (0, 0, 0))
    monkeypatch.setenv("STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS", "1")

    class PendingHost:
        def submit(self, *args):
            return "answer"

        def poll(self, ref):
            assert clock[0] < 2
            return HostAnswer(1, "call", 1, op, "PENDING", "seal.result.json", {})

    with pytest.raises(parallel.ParallelError, match=f"host response for {op} a timed out"):
        parallel._host_answer(PendingHost(), op, "a", {}, None)


def test_parent_deadline_still_precedes_response_allowance(monkeypatch):
    monkeypatch.setattr(parallel.time, "monotonic", lambda: 10)
    silent = SimpleNamespace(submit=lambda *args: "answer", poll=lambda *args: pytest.fail("past deadline"))
    with pytest.raises(parallel.ParallelError, match="parent deadline exceeded"):
        parallel._host_answer(silent, "status", "a", {}, 9)


def test_slow_signal_inspection_does_not_consume_worker_termination_grace(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker, "cleanup_seconds", lambda: 30.0)
    delivered = []

    def slow_signal(self, sig):
        if sig == signal.SIGTERM:
            clock[0] += 64
        delivered.append(sig)

    monkeypatch.setattr(worker.WorkerHandle, "_signal_group", slow_signal)
    handle = worker.WorkerHandle(SimpleNamespace(), 9876, 9876, 1, tmp_path, tmp_path / "log", 0, 0,
                                  SimpleNamespace(), launch_identity=tracker.LaunchIdentity(9876, None, None))
    handle.cancel("parent stopped")
    handle.escalate()
    assert delivered == [signal.SIGTERM]
    assert handle.cancel_started == 164
    clock[0] += 30
    handle.escalate()
    assert delivered == [signal.SIGTERM, signal.SIGKILL]


def test_failed_signal_inspection_remains_retryable(tmp_path, monkeypatch):
    handle = worker.WorkerHandle(SimpleNamespace(), 9876, 9876, 1, tmp_path, tmp_path / "log", 0, 0,
                                  SimpleNamespace(), launch_identity=tracker.LaunchIdentity(9876, None, None))
    calls = []

    def inspect(self, sig):
        calls.append(sig)
        if len(calls) in (1, 3):
            raise tracker.ContainmentUnverified("probe unavailable")

    monkeypatch.setattr(worker.WorkerHandle, "_signal_group", inspect)
    with pytest.raises(tracker.ContainmentUnverified):
        handle.cancel("parent stopped")
    assert handle.cancel_started is None
    handle.enforce_deadline()
    with pytest.raises(tracker.ContainmentUnverified):
        handle.escalate(grace=0)
    assert not handle.kill_sent
    handle.escalate(grace=0)
    assert handle.kill_sent


def test_direct_child_uses_adjacent_package_instead_of_other_installation(tmp_path):
    import sys

    child = Path(host.__file__).with_name("_child.py")
    other = tmp_path / "old-install"
    (other / "stateframe").mkdir(parents=True)
    (other / "stateframe" / "__init__.py").write_text("OLD_INSTALL = True\n")
    script = (
        "import runpy, sys\n"
        f"sys.path[:0] = [{str(child.parent)!r}, {str(other)!r}]\n"
        f"runpy.run_path({str(child)!r}, run_name='child_import_check')\n"
        "import stateframe\n"
        f"assert stateframe.__file__ == {str(child.with_name('__init__.py'))!r}, stateframe.__file__\n"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                               timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("method", ["_live_family", "_live_group"])
def test_incomplete_process_snapshot_is_retryable(tmp_path, monkeypatch, method):
    handle = worker.WorkerHandle(SimpleNamespace(), 9876, 9876, 1, tmp_path, tmp_path / "log", 0, 0,
                                  SimpleNamespace(), launch_identity=tracker.LaunchIdentity(9876, None, None))
    calls = []

    def snapshot():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("process table changed beyond enumeration capacity")
        return {}

    table = SimpleNamespace(snapshot=snapshot)
    handle._family = SimpleNamespace(table=table, update=lambda rows: list(rows.values()))
    monkeypatch.setattr(worker, "ProcessTable", lambda: table)
    with pytest.raises(tracker.ContainmentUnverified, match="inspection is unavailable"):
        getattr(handle, method)()
    assert not getattr(handle, method)()
