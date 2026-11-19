"""Turn-index work budgets on a large scripted workspace.

The profiler discovers every executed engine function from code filenames. Its
line events count loop iterations, including generator filters whose function
call count alone would hide a growing scan.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

import pytest

from kernel_continuation_journal import ContinuationJournal
from test_turn_end_to_end_budget import _qualification_fixture, repository  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
ENGINE = str(ROOT / "src/stateframe/engine") + os.sep
SAMPLES = (10, 50, 100, 200)
FS_NAMES = frozenset({"open", "read", "write", "stat", "lstat", "scandir", "listdir"})


@pytest.fixture
def all_turn_switches(monkeypatch):
    registry = (ROOT / "src/stateframe/engine/switch-registry.json").read_text()
    switches = set(re.findall(r'"(STATEFRAME_TURN_[A-Z_]+)"', registry))
    assert switches, "the shipped switch registry must expose turn switches"
    for switch in switches:
        monkeypatch.setenv(switch, "on")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")


@pytest.fixture
def switched_turn_fixture(all_turn_switches, _qualification_fixture):
    return _qualification_fixture


def _trace_turn(action):
    work: Counter[str] = Counter()

    def trace(frame, event, _arg):
        code = frame.f_code
        if not code.co_filename.startswith(ENGINE):
            return None
        name = f"{Path(code.co_filename).stem}.{code.co_qualname}"
        if event == "call":
            work[f"calls:{name}"] += 1
        elif event == "line":
            work[f"lines:{name}"] += 1
        return trace

    def profile(_frame, event, arg):
        if event == "c_call" and getattr(arg, "__name__", None) in FS_NAMES:
            work[f"fs:{arg.__name__}"] += 1

    previous_trace, previous_profile = sys.gettrace(), sys.getprofile()
    started = perf_counter()
    try:
        sys.settrace(trace)
        sys.setprofile(profile)
        action()
    finally:
        sys.setprofile(previous_profile)
        sys.settrace(previous_trace)
    return perf_counter() - started, work


def _growth_failures(samples, allowlist):
    keys = set().union(*(counts for _, counts in samples.values()))
    failures = []
    for key in sorted(keys):
        if key in allowlist:
            continue
        counts = [samples[step][1][key] for step in SAMPLES]
        # This scripted turn has identical work at each sample. Require exact
        # invariance so even slow logarithmic growth cannot hide in slack.
        if len(set(counts)) != 1:
            failures.append(f"{key}: {dict(zip(SAMPLES, counts))}")
    return failures


def test_scripted_200_turn_prepare_and_send_has_bounded_engine_work(
        tmp_path, switched_turn_fixture, record_property):
    state, port, work, submissions, submit, candidate = switched_turn_fixture
    assert work["reads"] == 5_000
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    journal = ContinuationJournal(journal_dir, "turn-budget", {})
    samples = {}
    elapsed_turns = []
    try:
        for step in range(1, 201):
            state.basis = state.basis._replace(step=step)

            def scripted_provider_turn():
                item = candidate(step)
                approved = port.qualify(state.basis, item, deadline=None)
                port.handoff(approved, submit)
                cursor = journal.receipt_cursor()
                journal.emit("action_receipt", receipt={"receipt_id": f"receipt-{step}"})
                assert f"receipt-{step}" in journal.new_receipts_after(cursor)
                assert submissions[-1] is item.wire

            if step in SAMPLES:
                elapsed, counts = _trace_turn(scripted_provider_turn)
                samples[step] = elapsed, counts
            else:
                started = perf_counter()
                scripted_provider_turn()
                elapsed = perf_counter() - started
            elapsed_turns.append(elapsed)
    finally:
        journal.close()

    allowlist = json.loads(Path(__file__).with_name("turn_perf_allowlist.json").read_text())
    assert all(isinstance(key, str) and isinstance(reason, str) and reason.strip()
               for key, reason in allowlist.items())
    record_property("turn_budget_samples", str({
        step: {"seconds": round(elapsed, 4),
               "calls": sum(value for key, value in counts.items() if key.startswith("calls:")),
               "lines": sum(value for key, value in counts.items() if key.startswith("lines:")),
               "receipt_scan_lines": sum(value for key, value in counts.items()
                                         if "new_receipts_after.<locals>.<genexpr>" in key),
               "fs": {key: value for key, value in counts.items() if key.startswith("fs:")}}
        for step, (elapsed, counts) in samples.items()
    }))
    record_property("turn_budget_seconds", str({
        "send_1": round(elapsed_turns[0], 4),
        "max_turn": round(max(elapsed_turns), 4),
    }))
    print("turn seconds:", {step: round(samples[step][0], 4) for step in SAMPLES},
          "send-1:", round(elapsed_turns[0], 4),
          "max:", round(max(elapsed_turns), 4))
    assert elapsed_turns[0] < 5, "send-1 prepare exceeded five seconds"
    assert max(elapsed_turns) < 5, "a turn prepare exceeded five seconds"
    failures = _growth_failures(samples, allowlist)
    assert not failures, "growing engine work by turn:\n" + "\n".join(failures)


def test_receipt_cursor_index_keeps_gaps_deduplication_and_replay(tmp_path):
    journal = ContinuationJournal(tmp_path, "cursor-index", {})
    try:
        start = journal.receipt_cursor()
        journal.emit("unrelated_event")
        assert journal.new_receipts_after(start) == ()
        before_first = journal.receipt_cursor()
        journal.emit("action_receipt", receipt={"receipt_id": "receipt-1"})
        assert "receipt-1" in journal.new_receipts_after(before_first)
        before_second = journal.receipt_cursor()
        journal.emit("action_receipt", receipt={"receipt_id": "receipt-1"})
        assert "receipt-1" not in journal.new_receipts_after(before_second)
        journal.emit("unrelated_event")
        assert journal.new_receipts_after(journal.receipt_cursor()) == ()
        expected = journal.new_receipts_after(start)
        tuple(journal.records())
        assert journal.new_receipts_after(start) == expected
        assert "receipt-1" in journal.receipt_refs_at()
    finally:
        journal.close()
