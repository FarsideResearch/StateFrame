"""Race seed 04: finish() must not report success, or wait forever, when the resident-memory supervisor thread died.

Port of y4race seeds/seed04_finish_dead_supervisor.py. Self-contained: the fake OS boundary (an in-memory process
table and a Popen-shaped child) is inlined so the seed runs unchanged from a known-bad archive overlay.

Case a (marked, 04a): the monitoring thread exits at once, verifying nothing, after the child has exited. finish()
must return False promptly. A True result, or no return inside the bound, is a property failure.
Case b (companion, 04b): the real monitoring loop observes the child exit over an empty unit, verifies cleanup and
finish() returns True.

Mutant `finish_ignores_dead_thread` deletes the dead-thread check from _ResidentMemorySupervisor.finish(). It is
active only when ENGINECHECK_RACE_MUTANT names it. The mutated module copy lives in tmp_path and is loaded for that
test only, so the product tree is never touched.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

MUTANT_ENV = "ENGINECHECK_RACE_MUTANT"
MUTANT_NAME = "finish_ignores_dead_thread"
PRODUCT_MODULE = "kernel_execution_policy"
PRODUCT_RELATIVE = Path("src/stateframe/engine/kernel_execution_policy.py")
# Exactly one occurrence of each marker in the product module, or the mutation refuses to run (drift guard).
_BLOCK_START = "            if not self._thread.is_alive():\n"
_BLOCK_END = "            self._leader_exit_processed.wait(0.05)\n"
_DEAD_THREAD_MESSAGE = "resident-memory supervisor thread exited before post-exit verification"

LAUNCH_PID = 4_000_123  # never a live process: the fake table holds no rows, so nothing is ever signalled
LAUNCH_START = (10, 0)
MEMORY_BYTES = 1 << 30
FINISH_BOUND_SECONDS = 5.0
LOOPS = 3


class _FakeChild:
    """Popen-shaped child: pid, scripted returncode and the launch identity the supervisor reads."""

    def __init__(self, policy: Any) -> None:
        self.pid = LAUNCH_PID
        self.returncode: int | None = None
        self._stateframe_launch_identity = policy.LaunchIdentity(pid=LAUNCH_PID, start=LAUNCH_START, boot_id=None)


class _FakeTable:
    """Empty process table offering the union of the product table surfaces the supervisor calls."""

    def identity(self, pid: int) -> None:
        return None

    def snapshot(self) -> dict[int, object]:
        return {}

    def pid_namespace_inode(self, pid: int) -> None:
        return None

    def pid_namespace(self, pid: int) -> None:
        return None

    def resident_memory_limit_breach(self, live: list[object], memory_bytes: int) -> str:
        return ""

    def accounted_memory_bytes(self, row: object) -> int:
        return 0

    def resident_bytes(self, row: object) -> int:
        return 0

    def page_table_bytes(self, row: object) -> int:
        return 0


def _mutate_source(source: str) -> str:
    """Remove the dead-thread block from finish() exactly once. Refuses to run if the product module drifted."""
    for marker in (_BLOCK_START, _BLOCK_END, _DEAD_THREAD_MESSAGE):
        count = source.count(marker)
        if count != 1:
            raise ValueError(f"expected exactly one {marker!r} in product module, found {count}")
    start = source.index(_BLOCK_START)
    end = source.index(_BLOCK_END)
    if end < start:
        raise ValueError("dead-thread block does not precede the wait call; product module drifted")
    mutated = source[:start] + source[end:]
    compile(mutated, f"{PRODUCT_MODULE}.py", "exec")
    return mutated


def _load_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Real product module, or the mutated copy shadowing it for this test when the mutant is selected."""
    if os.environ.get(MUTANT_ENV) != MUTANT_NAME:
        return importlib.import_module(PRODUCT_MODULE)
    repo = Path(__file__).resolve().parents[2]
    source = (repo / PRODUCT_RELATIVE).read_text(encoding="utf-8")
    target = tmp_path / f"{PRODUCT_MODULE}.py"
    target.write_text(_mutate_source(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(PRODUCT_MODULE, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build an import spec for {target}")
    module = importlib.util.module_from_spec(spec)
    # monkeypatch restores sys.modules after this test, so the mutant never leaks into another test.
    monkeypatch.setitem(sys.modules, PRODUCT_MODULE, module)
    spec.loader.exec_module(module)
    return module


def _build_supervisor(policy: Any) -> Any:
    """Construct the real supervisor with the OS process table replaced by the fake, before any thread starts."""
    original = policy.ProcessTable
    policy.ProcessTable = _FakeTable  # type: ignore[misc]
    try:
        child = _FakeChild(policy)
        supervisor = policy._ResidentMemorySupervisor(child, MEMORY_BYTES)
    finally:
        policy.ProcessTable = original
    supervisor.table = policy._ProgressTrackingProcessTable(supervisor, _FakeTable())
    supervisor.family.table = supervisor.table
    return supervisor


def _finish_bounded(supervisor: Any) -> tuple[bool | None, str]:
    """Run the real finish() on a helper thread so a hang becomes a verdict instead of stalling the test."""
    box: dict[str, object] = {}

    def call() -> None:
        box["result"] = supervisor.finish()

    worker = threading.Thread(target=call, name="seed04-finish", daemon=True)
    worker.start()
    worker.join(FINISH_BOUND_SECONDS)
    if worker.is_alive():
        return None, f"finish() did not return within {FINISH_BOUND_SECONDS:g} s"
    return bool(box["result"]), str(supervisor.finish_error)


@pytest.mark.race_seed(
    "04",
    mutant=MUTANT_NAME,
    expect="dead supervisor thread left finish() waiting",
)
def test_04a_dead_supervisor_thread_fails_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _load_policy(tmp_path, monkeypatch)
    for loop in range(LOOPS):
        supervisor = _build_supervisor(policy)
        # The monitoring thread is gone before it verifies anything, and the child has exited.
        dead = threading.Thread(target=lambda: None, name="sandbox-resident-memory", daemon=True)
        dead.start()
        dead.join()
        supervisor._thread = dead
        supervisor.process.returncode = 0
        result, detail = _finish_bounded(supervisor)
        if result is None:
            raise AssertionError(f"loop {loop}: dead supervisor thread left finish() waiting: {detail}")
        if result:
            raise AssertionError(f"loop {loop}: finish() returned True although cleanup was never verified")
        if "thread exited" not in detail:
            raise AssertionError(f"loop {loop}: finish() failed for the wrong reason: {detail}")


def test_04b_verified_supervisor_makes_finish_true() -> None:
    policy = importlib.import_module(PRODUCT_MODULE)
    for loop in range(LOOPS):
        supervisor = _build_supervisor(policy)
        supervisor.process.returncode = 0
        supervisor.start()
        result, detail = _finish_bounded(supervisor)
        if result is None:
            raise AssertionError(f"loop {loop}: verified supervisor did not finish: {detail}")
        if result is not True:
            raise AssertionError(f"loop {loop}: finish() returned False for a verified empty unit: {detail}")
        if not supervisor.containment_verified:
            raise AssertionError(f"loop {loop}: finish() True without containment_verified")
