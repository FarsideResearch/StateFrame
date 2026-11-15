"""Seed 01: a PID-namespace inode freed by a dead sandbox must never name another engine's processes.

Ported from the y4race harness (seeds/seed01_stale_ns.py @ 9c6545d). Self-contained on purpose: the scripted
process table is inlined so this file runs unchanged against the known-bad tree (git archive of the bad sha plus
this file, tests/race_seeds/ and tests/enginecheck/).

Engine A launches a sandbox child in private namespace X. The sandbox ends. Engine B's launcher then gets a new
sandbox namespace. Unpinned, the kernel hands B the freed inode X. Pinned, X stays reserved.
"""
from __future__ import annotations

import importlib
from collections import Counter
from typing import Any

import pytest

KNOWN_BAD_SHA = "40da3c4c5e969097c8f7bdee31b3df26e961881d"

A_LEADER, A_CHILD, A_GENUINE_CHILD = 500, 501, 502
B_LAUNCHER, B_NEIGHBOUR = 890, 900
HOST_INIT = 1  # outside engine A's family: the reparent target for the companion
LAUNCH_START = (10, 0)


class FakeProcessTable:
    """Scripted process table with a PID-namespace lifecycle. Offers the union of product table APIs.

    Kernel model: a namespace inode is freed when the namespace ends and no pin holds it. Inode reuse is modelled
    as reuse of a freed inode: the next new namespace takes the lowest free inode, as Linux allocates namespace
    inums from an IDA. A pin keeps the namespace object alive, so its inode is never reused while pinned.
    """

    def __init__(self, identity_cls: Any, host_inode: int = 1) -> None:
        self._row = identity_cls  # product ProcessIdentity, so rows are the product's own type
        self.rows: dict[int, Any] = {}
        self._ns_of: dict[int, int] = {}
        self._live: set[int] = {host_inode}
        self._refs: Counter[int] = Counter()
        self._freed: list[int] = []
        self._next_fresh = host_inode + 1

    # kernel namespace lifecycle
    def new_namespace(self) -> int:
        """Creates a namespace; reuses the lowest freed inode before allocating a fresh one."""
        if self._freed:
            inode = min(self._freed)
            self._freed.remove(inode)
        else:
            inode = self._next_fresh
            self._next_fresh += 1
        self._live.add(inode)
        return inode

    def end_namespace(self, inode: int) -> None:
        """Ends a namespace; its processes exit. Frees the inode unless a pin still holds it."""
        self._live.discard(inode)
        for pid in [p for p, ns in self._ns_of.items() if ns == inode]:
            self.rows.pop(pid, None)
            self._ns_of.pop(pid, None)
        if self._refs[inode] == 0:
            self._freed.append(inode)

    def exit_process(self, pid: int, *, reparent_to: int) -> None:
        """A non-init process exits while its namespace stays live. Its children are reparented outside the family."""
        if pid not in self.rows:
            raise RuntimeError(f"exit of unknown pid {pid}")
        del self.rows[pid]
        self._ns_of.pop(pid, None)
        for child, row in list(self.rows.items()):
            if row.ppid == pid:
                self.rows[child] = self._row(child, reparent_to, row.pgid, row.sid, row.start, row.zombie)

    def spawn(self, pid: int, ppid: int, start: tuple[int, int], inode: int) -> None:
        if inode not in self._live:
            raise RuntimeError(f"spawn into dead namespace {inode}")
        self.rows[pid] = self._row(pid, ppid, pid, pid, start, False)
        self._ns_of[pid] = inode

    # product ProcessTable surface
    def identity(self, pid: int) -> Any:
        return self.rows.get(pid)

    def snapshot(self) -> dict[int, Any]:
        return dict(self.rows)

    def pid_namespace_inode(self, pid: int) -> int | None:
        return self._ns_of.get(pid) if pid in self.rows else None

    def pin_pid_namespace_inode(self, pid: int, inode: int) -> "_Pin | None":
        """Holds the namespace open only while `pid` is alive in a namespace with `inode` that is still live."""
        if self._ns_of.get(pid) != inode or inode not in self._live:
            return None
        self._refs[inode] += 1
        return _Pin(self, inode)

    def _release(self, inode: int) -> None:
        self._refs[inode] -= 1
        if self._refs[inode] == 0 and inode not in self._live:
            self._freed.append(inode)


class _Pin:
    def __init__(self, host: FakeProcessTable, inode: int) -> None:
        self._host: FakeProcessTable | None = host
        self.inode = inode

    def close(self) -> None:
        if self._host is not None:
            self._host._release(self.inode)
            self._host = None


def _run_wrong_kill(tracker: Any) -> tuple[bool, int]:
    """Engine A tracks its sandbox, the sandbox ends, engine B's sandbox reuses the freed inode.

    Returns (whether B's neighbour was adopted into A's family, the namespace inode B received).
    """
    table = FakeProcessTable(tracker.ProcessIdentity)
    table.spawn(A_LEADER, 1, LAUNCH_START, 1)
    ns_a = table.new_namespace()
    table.spawn(A_CHILD, A_LEADER, (11, 0), ns_a)
    family = tracker.Descendants(A_LEADER, table)
    family.update(table.snapshot())  # a observes family
    table.end_namespace(ns_a)  # a sandbox ends

    table.spawn(B_LAUNCHER, 1, (15, 0), 1)  # b sandbox starts (may reuse freed inode)
    ns_b = table.new_namespace()
    table.spawn(B_NEIGHBOUR, B_LAUNCHER, (20, 0), ns_b)

    live = family.update(table.snapshot())  # a re-observes
    live_pids = {row.pid for row in live}
    adopted = B_NEIGHBOUR in live_pids or B_NEIGHBOUR in family.members
    return adopted, ns_b


def _run_own_child_companion(tracker: Any) -> set[int]:
    """Engine A's own child must be a member through namespace membership alone.

    A's sandbox child 501 is observed (pins private ns X). A's genuine child 502 starts in X under 501. Then 501
    exits while X stays alive (502 keeps it live). 502 is reparented to HOST_INIT, outside the family, so the ppid
    walk from live members cannot reach it. Only namespace X can.
    """
    table = FakeProcessTable(tracker.ProcessIdentity)
    table.spawn(A_LEADER, 1, LAUNCH_START, 1)
    ns_a = table.new_namespace()
    table.spawn(A_CHILD, A_LEADER, (11, 0), ns_a)
    family = tracker.Descendants(A_LEADER, table)
    family.update(table.snapshot())  # a observes family (pins private ns)
    table.spawn(A_GENUINE_CHILD, A_CHILD, (12, 0), ns_a)  # a spawns genuine child in live ns
    table.exit_process(A_CHILD, reparent_to=HOST_INIT)  # sandbox child exits while ns stays live
    if table.identity(A_GENUINE_CHILD).ppid in (A_LEADER, A_CHILD):
        # Scenario isolation: otherwise the ppid walk could find 502 and the seed proves nothing.
        raise RuntimeError("01b isolation broken: genuine child still parented inside engine A's family")
    live = family.update(table.snapshot())  # a re-observes
    return {row.pid for row in live}


@pytest.mark.race_seed(
    "01",
    known_bad=KNOWN_BAD_SHA,
    expect="adopted as engine A member",
)
def test_seed01a_reused_freed_namespace_does_not_adopt_neighbour():
    tracker = importlib.import_module("kernel_exec_tracker")
    adopted, ns_b = _run_wrong_kill(tracker)
    assert not adopted, (
        f"engine B pid {B_NEIGHBOUR} adopted as engine A member (ns inode {ns_b} reused A's freed inode)"
    )


def test_seed01b_companion_own_child_in_live_namespace_is_member():
    tracker = importlib.import_module("kernel_exec_tracker")
    live_pids = _run_own_child_companion(tracker)
    assert A_GENUINE_CHILD in live_pids, (
        f"engine A's own child pid {A_GENUINE_CHILD} in its live sandbox namespace was not a member"
    )
