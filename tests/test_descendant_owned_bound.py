"""Descendants.update: group members orphaned around the leader's reap stay tracked; a reused leader's new group does not."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_exec_tracker as tracker
from stateframe.engine.kernel_exec_tracker import Descendants, ProcessIdentity

LEADER = 500


def row(pid, ppid, start, *, pgid=LEADER, sid=LEADER, zombie=False):
    return ProcessIdentity(pid, ppid, pgid, sid, (start, 0), zombie)


def family(leader_row):
    # The table reports the leader at construction only; afterwards update() sees the snapshots given.
    seen = {"leader": leader_row}
    table = SimpleNamespace(identity=lambda pid: seen.pop("leader", None) if pid == LEADER else None)
    return Descendants(LEADER, table)


def snapshot(*rows):
    return {r.pid: r for r in rows}


@pytest.fixture
def reap_at(monkeypatch):
    def set_clock(start):
        monkeypatch.setattr(tracker, "process_start_now", lambda: (start, 0))
    return set_clock


def test_group_member_is_adopted_while_the_leader_is_unreaped():
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    # A leader PID is reserved until it is reaped, so even a zombie leader anchors its group.
    zombie = row(LEADER, 1, 100, zombie=True)
    orphan = row(601, 1, 9_999)
    live = tracked.update(snapshot(zombie, orphan))
    assert [r.pid for r in live] == [601]


def test_child_forked_after_last_scan_is_adopted_after_reap(reap_at):
    # The fork-limit escapee: born after the last scan that saw the leader, its parent (the
    # leader) then killed and reaped before the next scan, its siblings already gone.
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    tracked.update(snapshot(leader, row(601, LEADER, 110)))
    reap_at(150)
    tracked.leader_reaped()
    escapee = row(602, 1, 130)
    live = tracked.update(snapshot(escapee))
    assert [r.pid for r in live] == [602]


def test_reused_leader_pid_new_group_is_not_adopted(reap_at):
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    tracked.update(snapshot(leader))
    reap_at(150)
    tracked.leader_reaped()
    # PID 500 is reused after the reap; the new process makes its own group and forks.
    reused = row(LEADER, 1, 200)
    child = row(603, LEADER, 201)
    assert tracked.update(snapshot(reused, child)) == []


def test_reused_leader_pid_new_group_is_not_adopted_without_reap_report():
    # Same guarantee through update() alone (no leader_reaped call): green before and after the fix.
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    tracked.update(snapshot(leader, row(900, 1, 140, pgid=900, sid=900)))
    reused = row(LEADER, 1, 200)
    child = row(603, LEADER, 201)
    assert tracked.update(snapshot(reused, child)) == []


def test_start_equal_to_reap_bound_is_adopted_by_design(reap_at):
    # Inclusive bound: a member born in the same clock unit as the reap is adopted. On Linux
    # /proc start times have clock-tick (~10 ms) resolution, so a reused leader whose new group
    # starts in that same tick would also be adopted; KNOWN_LIMITATIONS records this residual.
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    tracked.update(snapshot(leader))
    reap_at(150)
    tracked.leader_reaped()
    assert [r.pid for r in tracked.update(snapshot(row(604, 1, 150)))] == [604]
    assert tracked.update(snapshot(row(605, 1, 151))) == []


def test_unreported_reap_falls_back_to_newest_start_seen_while_leader_was_live():
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    tracked.update(snapshot(leader, row(900, 1, 140, pgid=900, sid=900)))
    # The leader vanished without leader_reaped(): only members born by the last live scan count.
    assert [r.pid for r in tracked.update(snapshot(row(606, 1, 130)))] == [606]
    assert tracked.update(snapshot(row(607, 1, 141))) == []


def test_leader_reaped_is_recorded_once(reap_at):
    tracked = family(row(LEADER, 1, 100))
    reap_at(150)
    tracked.leader_reaped()
    reap_at(999)
    tracked.leader_reaped()
    assert tracked.reaped_bound == (150, 0)


def test_children_of_tracked_members_are_adopted_without_any_bound(reap_at):
    leader = row(LEADER, 1, 100)
    tracked = family(leader)
    member = row(610, LEADER, 110)
    tracked.update(snapshot(leader, member))
    reap_at(150)
    tracked.leader_reaped()
    late = row(611, 610, 300, pgid=611, sid=611)
    assert sorted(r.pid for r in tracked.update(snapshot(member, late))) == [610, 611]


def test_process_start_now_matches_process_table_units():
    import os
    import sys
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("process table units are platform specific")
    me = tracker.ProcessTable().identity(os.getpid())
    assert me is not None
    assert me.start <= tracker.process_start_now()
