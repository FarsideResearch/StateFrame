"""Per-tick shared process-table view for subframe worker supervision.

One scan per supervisor tick serves every worker, while containment keeps catching
processes that die or appear between ticks and never trusts the shared view to signal.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

import stateframe.parallel_worker as worker_module
from stateframe.engine.kernel_exec_tracker import ContainmentUnverified, ProcessTable
from stateframe.parallel_worker import TickProcessTable, WorkerHandle

_SLEEPER = "import time\ntime.sleep(120)\n"
# Spawns a detached (new session) child once the trigger file exists.
_SPAWNER = (
    "import pathlib, subprocess, sys, time\n"
    "trigger = pathlib.Path(sys.argv[1])\n"
    "while not trigger.exists():\n"
    "    time.sleep(0.01)\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
    "                         start_new_session=True)\n"
    "pathlib.Path(sys.argv[2]).write_text(str(child.pid))\n"
    "time.sleep(120)\n"
)


def _spawn(tmp_path, code=_SLEEPER, *args):
    return subprocess.Popen([sys.executable, "-c", code, *args], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _handle(tmp_path, process, table=None):
    handle = WorkerHandle(process, process.pid, process.pid, 1, tmp_path, tmp_path / "w.log",
                          time.time(), time.monotonic(), threading.Thread(target=lambda: None))
    if table is not None:
        handle.attach_tick_table(table)
    return handle


@pytest.fixture
def procs():
    started: list[subprocess.Popen] = []
    yield started
    for process in started:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait(timeout=10)


def _wait(predicate, seconds=10.0):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_one_scan_per_tick_serves_every_worker(tmp_path, procs):
    table = TickProcessTable()
    handles = []
    for _ in range(6):
        process = _spawn(tmp_path)
        procs.append(process)
        handles.append(_handle(tmp_path, process, table))
    baseline = table.scans
    with table.tick():
        for _ in range(2):  # enforce_deadline and the host's own poll both observe
            for handle in handles:
                assert handle.poll() is None
    assert table.scans - baseline == 1
    # Outside a tick nothing is shared: each observation scans, exactly as before.
    for handle in handles:
        handle.poll()
    assert table.scans - baseline == 1 + len(handles)
    # A second tick scans again (the view never outlives its tick).
    with table.tick():
        handles[0].poll()
        handles[1].poll()
    assert table.scans - baseline == 2 + len(handles)


def test_shared_view_reports_the_same_live_family_as_a_fresh_scan(tmp_path, procs):
    trigger, pid_file = tmp_path / "go", tmp_path / "child.pid"
    process = _spawn(tmp_path, _SPAWNER, str(trigger), str(pid_file))
    procs.append(process)
    table = TickProcessTable()
    handle = _handle(tmp_path, process, table)
    trigger.write_text("x")
    assert _wait(pid_file.exists) and _wait(lambda: pid_file.read_text() != "")
    child = int(pid_file.read_text())
    with table.tick():
        shared = {row.pid for row in handle._live_family(shared=True)}
    fresh = {row.pid for row in handle._live_family()}
    assert shared == fresh and {process.pid, child} <= fresh
    # Namespace identity checks stay live and exact through the shared table.
    assert table.pid_namespace_inode(child) == ProcessTable.pid_namespace_inode(child)


def test_pid_that_dies_between_ticks_leaves_the_family(tmp_path, procs):
    trigger, pid_file = tmp_path / "go", tmp_path / "child.pid"
    process = _spawn(tmp_path, _SPAWNER, str(trigger), str(pid_file))
    procs.append(process)
    table = TickProcessTable()
    handle = _handle(tmp_path, process, table)
    trigger.write_text("x")
    assert _wait(lambda: pid_file.exists() and pid_file.read_text() != "")
    child = int(pid_file.read_text())
    with table.tick():
        assert child in {row.pid for row in handle._live_family(shared=True)}
    os.kill(child, signal.SIGKILL)
    # Not zombie-reaped by us: its parent (the worker) never waits, so wait for the kernel state.
    def gone():
        with table.tick():
            return child not in {row.pid for row in handle._live_family(shared=True)}
    assert _wait(gone)


def test_pid_that_appears_between_ticks_is_found_next_tick_and_by_any_fresh_scan(tmp_path, procs):
    trigger, pid_file = tmp_path / "go", tmp_path / "child.pid"
    process = _spawn(tmp_path, _SPAWNER, str(trigger), str(pid_file))
    procs.append(process)
    table = TickProcessTable()
    handle = _handle(tmp_path, process, table)
    with table.tick():
        assert {row.pid for row in handle._live_family(shared=True)} == {process.pid}
        trigger.write_text("x")
        assert _wait(lambda: pid_file.exists() and pid_file.read_text() != "")
        child = int(pid_file.read_text())
        # Still inside the tick: the shared view predates the child, a fresh scan does not.
        assert child not in {row.pid for row in handle._live_family(shared=True)}
        assert child in {row.pid for row in handle._live_family()}
    with table.tick():
        assert child in {row.pid for row in handle._live_family(shared=True)}


def test_signalling_never_uses_the_shared_view_and_drops_it(tmp_path, procs):
    trigger, pid_file = tmp_path / "go", tmp_path / "child.pid"
    process = _spawn(tmp_path, _SPAWNER, str(trigger), str(pid_file))
    procs.append(process)
    table = TickProcessTable()
    handle = _handle(tmp_path, process, table)
    with table.tick():
        handle.poll()  # primes the view before the detached child exists
        trigger.write_text("x")
        assert _wait(lambda: pid_file.exists() and pid_file.read_text() != "")
        child = int(pid_file.read_text())
        handle._signal_group(signal.SIGKILL)
        # The detached child lives in another session; only a fresh family scan can reach it.
        assert _wait(lambda: not os.path.exists(f"/proc/{child}") or
                     open(f"/proc/{child}/stat").read().rsplit(")", 1)[1].split()[0] in "ZX")
        scans = table.scans
        handle.poll()
        assert table.scans == scans + 1  # the signal dropped the stale view


def test_failed_scan_is_raised_once_per_tick_not_retried_per_worker(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    handles = []
    for _ in range(3):
        process = _spawn(tmp_path)
        procs.append(process)
        handles.append(_handle(tmp_path, process, table))
    calls = []

    def broken(self):
        calls.append(1)
        raise OSError("procfs unavailable")

    monkeypatch.setattr(ProcessTable, "snapshot", broken)
    with table.tick():
        for handle in handles:
            with pytest.raises(ContainmentUnverified):
                handle.poll()
    assert len(calls) == 1


def test_nested_tick_joins_the_outer_scan(tmp_path, procs):
    table = TickProcessTable()
    process = _spawn(tmp_path)
    procs.append(process)
    handle = _handle(tmp_path, process, table)
    baseline = table.scans
    with table.tick():  # the host tick
        handle.poll()
        with table.tick():  # a status request re-advancing workers inside that tick
            handle.poll()
        handle.poll()
    assert table.scans - baseline == 1


def test_fresh_scans_share_one_scan_that_began_after_each_caller_arrived(monkeypatch):
    table = TickProcessTable()
    begun: list[float] = []
    gate = threading.Event()
    entered = threading.Event()

    def slow(self):
        begun.append(time.monotonic())
        entered.set()
        gate.wait(5)
        return {}

    monkeypatch.setattr(ProcessTable, "snapshot", slow)
    first = threading.Thread(target=table.snapshot)
    first.start()
    assert entered.wait(5)
    arrived: list[float] = []

    def late():
        arrived.append(time.monotonic())
        table.snapshot()

    waiters = [threading.Thread(target=late) for _ in range(4)]
    for waiter in waiters:
        waiter.start()
    time.sleep(0.2)
    gate.set()
    for thread in (first, *waiters):
        thread.join(10)
    # One scan was already in flight when the four arrived: it must not serve them.
    # They share exactly one later scan.
    assert len(begun) == 2
    assert begun[1] >= max(arrived)


def _reference_identity(pid: int):
    """The text-based /proc/<pid>/stat parse the raw read replaced."""
    from pathlib import Path

    from stateframe.engine.kernel_exec_tracker import ProcessIdentity

    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(")") + 2:].split()
    return ProcessIdentity(pid, int(fields[1]), int(fields[2]), int(fields[3]),
                           (int(fields[19]), 0), fields[0] in {"Z", "X", "x"})


@pytest.mark.skipif(sys.platform != "linux", reason="procfs only")
def test_raw_stat_identity_matches_the_text_parse_for_every_process(tmp_path):
    table = ProcessTable()
    checked = 0
    for entry in os.listdir("/proc"):
        if not entry.isdecimal():
            continue
        pid = int(entry)
        try:
            expected = _reference_identity(pid)
        except (PermissionError, UnicodeDecodeError, IndexError, ValueError):
            continue
        got = table.identity(pid)
        if expected is None or got is None:
            continue  # exited between the two reads
        if expected.zombie or got.zombie:
            continue
        assert got == expected, pid
        checked += 1
    assert checked > 5
    assert table.identity(2 ** 22 + 12345) is None


@pytest.mark.skipif(sys.platform != "linux", reason="procfs only")
def test_raw_stat_identity_handles_a_comm_with_parentheses_and_spaces(tmp_path):
    code = ("import ctypes, time\n"
            "ctypes.CDLL(None).prctl(15, b'a) (S 1 x', 0, 0, 0)\n"
            "time.sleep(60)\n")
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        table = ProcessTable()
        assert _wait(lambda: "(S 1" in open(f"/proc/{process.pid}/stat").read())
        row = table.identity(process.pid)
        assert row == _reference_identity(process.pid)
        assert row.ppid == os.getpid() and row.pgid == process.pid and not row.zombie
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def test_no_profiling_leftovers_ship_in_the_engine_sources():
    from pathlib import Path

    root = Path(worker_module.__file__).resolve().parent
    offenders = [str(path.relative_to(root)) for path in root.rglob("*.py")
                 if "TEMPPROF" in path.read_text(errors="replace")
                 or "zz-prof" in path.read_text(errors="replace")]
    assert offenders == []


def test_unexpected_scan_error_never_publishes_an_empty_table(tmp_path, procs, monkeypatch):
    """A scan that dies with a non-OSError (IndexError on a garbled stat read) must reach
    every waiter sharing it as a failure; an empty table would let a seal through."""
    table = TickProcessTable()
    process = _spawn(tmp_path)
    procs.append(process)
    handle = _handle(tmp_path, process, table)

    def garbled(self):
        raise IndexError("list index out of range")

    monkeypatch.setattr(ProcessTable, "snapshot", garbled)
    with table.tick():
        with pytest.raises(ContainmentUnverified):
            handle.poll()
        with pytest.raises(ContainmentUnverified):
            handle._live_group()
    with pytest.raises(ContainmentUnverified):
        handle._live_unit()


def test_coalesced_waiters_share_the_scan_failure_not_an_empty_table(monkeypatch):
    table = TickProcessTable()
    gate = threading.Event()
    entered = threading.Event()
    calls: list[int] = []

    def slow_then_garbled(self):
        calls.append(len(calls))
        if len(calls) == 1:
            entered.set()
            gate.wait(5)
            return {}
        raise IndexError("list index out of range")

    monkeypatch.setattr(ProcessTable, "snapshot", slow_then_garbled)
    first = threading.Thread(target=table.snapshot)
    first.start()
    assert entered.wait(5)
    outcomes: list[object] = []

    def late():
        try:
            outcomes.append(table.snapshot())
        except Exception as error:  # noqa: BLE001 - the outcome type is the assertion
            outcomes.append(error)

    waiters = [threading.Thread(target=late) for _ in range(4)]
    for waiter in waiters:
        waiter.start()
    time.sleep(0.2)
    gate.set()
    for thread in (first, *waiters):
        thread.join(10)
    assert len(calls) == 2  # the four coalesced onto one later scan, which failed
    assert len(outcomes) == 4
    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes), outcomes
    assert all(isinstance(outcome.__cause__, IndexError) for outcome in outcomes)


@pytest.mark.skipif(sys.platform != "linux", reason="procfs only")
def test_empty_stat_read_means_the_process_is_gone(monkeypatch):
    import types

    import stateframe.engine.kernel_exec_tracker as tracker

    # Shadow only the tracker module's `os` name: patching os.read itself would starve
    # the test runner's own pipes.
    shim = types.SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if not name.startswith("__")})
    shim.read = lambda descriptor, size: b""
    monkeypatch.setattr(tracker, "os", shim)
    assert ProcessTable().identity(os.getpid()) is None


def test_tick_held_open_on_another_thread_never_keeps_the_view_stale(tmp_path, procs):
    """Prepare and channel threads poll workers inside their own ticks; a tick left open
    there must not make the supervisor's next tick reuse the previous scan."""
    table = TickProcessTable()
    process = _spawn(tmp_path)
    procs.append(process)
    handle = _handle(tmp_path, process, table)
    holding = threading.Event()
    release = threading.Event()

    def other_thread():
        with table.tick():
            handle.poll()
            holding.set()
            release.wait(10)

    other = threading.Thread(target=other_thread)
    other.start()
    assert holding.wait(5)
    try:
        with table.tick():
            handle.poll()
            first = table.scans
            handle.poll()
            assert table.scans == first  # joined within this tick
        with table.tick():
            handle.poll()
            assert table.scans == first + 1  # a new tick rescans despite the other thread
    finally:
        release.set()
        other.join(10)


def test_view_is_invalidated_after_a_failed_signal(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    process = _spawn(tmp_path)
    procs.append(process)
    handle = _handle(tmp_path, process, table)

    def boom(process, identity, sig):
        raise OSError("partial signal")

    monkeypatch.setattr(worker_module, "signal_launch_group", boom)
    with table.tick():
        handle.poll()
        with pytest.raises(OSError):
            handle._signal_group(signal.SIGTERM)
        scans = table.scans  # after the fresh pre-signal scan the failed signal took
        handle.poll()
        assert table.scans == scans + 1  # the failed signal still dropped the view


def test_signal_during_a_scan_leaves_that_view_unpublished(tmp_path, procs, monkeypatch):
    table = TickProcessTable()
    process = _spawn(tmp_path)
    procs.append(process)
    handle = _handle(tmp_path, process, table)
    real = ProcessTable.snapshot
    calls: list[int] = []
    blocked: list[bool] = []

    def racing(self):
        rows = real(self)
        calls.append(1)
        if len(calls) == 1:
            # A signal lands on another thread while this scan is running: it must not
            # wait behind the scan, and the scan's view must not be reused afterwards.
            signaller = threading.Thread(target=table.invalidate)
            signaller.start()
            signaller.join(2)
            blocked.append(signaller.is_alive())
        return rows

    monkeypatch.setattr(ProcessTable, "snapshot", racing)
    with table.tick():
        handle.poll()
        handle.poll()
    assert blocked == [False]  # invalidate() never waits behind a scan
    assert len(calls) == 2  # the overlapped scan was not reused
