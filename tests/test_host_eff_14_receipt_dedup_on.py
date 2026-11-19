"""Receipt enrichment dedup must not change check numbering or any journal consumer.

Drives the real record path (record_check_output) against a real
ContinuationJournal with the dedup mode off and on, over one pinned execution:
first receipt, identical re-record, changed receipt, identical re-record of the
change. Every projection the consumers read must match between the two runs.

Contract:
- A suppressed identical re-record writes no check_receipt row. It journals a
  check_receipt_unchanged row instead, so the journal-owned check_execution_count
  still advances once per execution.
- check_execution_count (which numbers every check execution) matches on and off
  at every step, and so do the execution ids and receipt handles derived from it.
- check_receipt_count (stored rows) may differ, and only on identical re-records.
- Off mode journals the same bytes as the base release (clock and run id pinned).

Normalization (applied identically to both runs, nothing else):
- journal envelope position fields (sequence, event_id, hashes, timestamps) are
  dropped; a suppressed row shifts every later position by design;
- each journal's random run_id is replaced by a placeholder, since two separate
  journal instances can never share one.
The progress view's cache_key and projection_digest are hashes over the raw
event stream, so they are excluded; the rendered view content is compared.
"""
from __future__ import annotations

import copy
import os
import unittest.mock
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import kernel_continuation_journal
from kernel_check_receipts import _next_check_execution_ordinal, record_check_output
from kernel_continuation_journal import ContinuationJournal
from kernel_frame_journal_gate import facts_from_events
from kernel_progress_ledger import ProgressInputs, project_progress
from kernel_vacuity_truth import project_journal_truth

SELECTOR = "tests/test_receipt_dedup.py::test_case"
# One pinned execution for the whole sequence: a re-record of the same check must
# reach the same handle, which is the real re-record path. Without it every call
# allocates a fresh execution and the steps are not re-records at all.
EXECUTION_ID = "check-execution:host-eff-14-receipt-dedup"
TASK_ID = "receipt-dedup"
DEDUP_ENV = "STATEFRAME_CHECK_RECEIPT_ENRICHMENT_DEDUP"
MANIFEST: dict[str, object] = {"origin": "host-eff-14"}
BINDING: dict[str, object] = {}

# Kinds spelled out here (not imported) so this file imports on the base tree too.
CHECK_RECEIPT = "check_receipt"
CHECK_RECEIPT_UNCHANGED = "check_receipt_unchanged"
CHECK_KINDS = frozenset({CHECK_RECEIPT, CHECK_RECEIPT_UNCHANGED})

# Envelope fields that encode journal position; shifted by design when a row is suppressed.
POSITION_KEYS = frozenset({
    "sequence", "seq", "event_id", "event_sequence", "timestamp_ns",
    "previous_hash", "record_hash", "prev_hash", "hash",
})
# Progress-view hashes computed over the raw event stream (includes positions).
RAW_STREAM_DIGEST_KEYS = frozenset({"cache_key", "projection_digest"})
RUN_PLACEHOLDER = "<run>"

# Byte-identity fixture: the off-mode journal of the base release for SEQUENCE, with the
# clock and run id pinned (see _off_journal_bytes).
GOLDEN_OFF_JOURNAL = Path(__file__).resolve().parent / "fixtures" / "host_eff_14_check_journal_off.jsonl"
FIXED_CLOCK_NS = 1_700_000_000_000_000_000
FIXED_RUN_ID_INT = 0x0E14_0E14_0E14_0E14_0E14_0E14_0E14_0E14

PASS_RECEIPT: dict[str, object] = {
    "status": "PASS", "outcome": "PASS", "exit_code": 0,
    "stdout": "1 passed\n", "collected_tests": 1,
}
FAIL_RECEIPT: dict[str, object] = {
    "status": "FAIL", "outcome": "BEHAVIOR_FAIL", "exit_code": 1,
    "stdout": "1 failed\n", "collected_tests": 1,
}
# (label, receipt) in record order. Steps 2 and 4 are identical re-records.
SEQUENCE: tuple[tuple[str, dict[str, object]], ...] = (
    ("first", PASS_RECEIPT),
    ("identical_rerecord", PASS_RECEIPT),
    ("changed", FAIL_RECEIPT),
    ("identical_after_change", FAIL_RECEIPT),
)
# Executions in SEQUENCE; the next auto-allocated execution is number EXECUTION_COUNT + 1.
EXECUTION_COUNT = len(SEQUENCE)


def _normalize(value: object, run_ids: frozenset[str]) -> object:
    if isinstance(value, Mapping):
        return {
            key: _normalize(item, run_ids) for key, item in value.items()
            if key not in POSITION_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, run_ids) for item in value]
    if isinstance(value, str) and value in run_ids:
        return RUN_PLACEHOLDER
    return value


def _kinds(records: list[dict]) -> list[str]:
    return [str(record.get("kind")) for record in records if record.get("kind") in CHECK_KINDS]


def _check_rows(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("kind") == CHECK_RECEIPT]


def _unchanged_rows(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("kind") == CHECK_RECEIPT_UNCHANGED]


def _other_rows(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("kind") not in CHECK_KINDS]


def _projection(runtime: SimpleNamespace, journal: ContinuationJournal) -> dict[str, object]:
    # Read live continuation projections before records() rebuilds them.
    live = {
        # None on a tree without the counter, so the mismatch shows as an assertion, not an AttributeError.
        "check_execution_count": getattr(journal, "check_execution_count", None),
        "vacuity_latest_check": copy.deepcopy(journal._vacuity_latest_check),
        "vacuity_last_edit": copy.deepcopy(journal._vacuity_last_edit),
    }
    stored_check_rows = journal.check_receipt_count
    records = list(journal.records())
    return {
        "host_check_receipts": copy.deepcopy(runtime.host_check_receipts),
        "latest_check": copy.deepcopy(runtime._latest_check_receipt),
        "vacuity_truth": asdict(project_journal_truth(records, BINDING)),
        "continuation_live": live,
        "stored_check_rows": stored_check_rows,
        "frame_gate_facts": asdict(facts_from_events(records)),
        "progress_view": asdict(project_progress(
            ProgressInputs(events=tuple(records), current_binding=BINDING))),
    }


@dataclass(frozen=True)
class _Run:
    returned: list[object]
    steps: list[dict[str, object]]
    records: list[object]
    next_execution_id: str
    journal_path: Path
    run_id: str
    head: str


def _drive(root, mode: str) -> _Run:
    root.mkdir(parents=True, exist_ok=True)
    journal = ContinuationJournal(root, TASK_ID, MANIFEST)
    runtime = SimpleNamespace(
        journal=journal,
        owner=SimpleNamespace(current_step=1),
        stateframe_check_receipt_enrichment_dedup_mode=mode,
        host_check_receipts=[],
    )
    returned: list[dict] = []
    steps: list[dict[str, object]] = []
    try:
        for _label, receipt in SEQUENCE:
            returned.append(record_check_output(
                runtime, selector=SELECTOR, sequence=1, execution_id=EXECUTION_ID, **receipt))
            steps.append(_projection(runtime, journal))
        records = list(journal.records())
        # A check that allocates its own execution id after the sequence. Its id is
        # derived from the execution count, so it exposes any numbering drift.
        fresh = record_check_output(
            runtime, selector="tests/test_receipt_dedup.py::test_fresh", sequence=2,
            **PASS_RECEIPT)
        next_execution_id = str(fresh["execution_id"])
    finally:
        journal.close()
    run_ids = frozenset(str(r["run_id"]) for r in records if isinstance(r.get("run_id"), str))
    assert run_ids, "journal records must carry a run_id"
    return _Run(
        returned=[_normalize(item, run_ids) for item in returned],
        steps=[_normalize(step, run_ids) for step in steps],
        records=_normalize(records, run_ids),
        next_execution_id=next_execution_id,
        journal_path=Path(journal.path),
        run_id=journal.run_id,
        head=str(journal._head),
    )


def _diff(off: Mapping[str, object], on: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    """Field-level delta so a failure names the exact field and both values."""
    return {
        key: (off.get(key), on.get(key))
        for key in set(off) | set(on) if off.get(key) != on.get(key)
    }


def _diff_value(off: object, on: object) -> object:
    if isinstance(off, Mapping) and isinstance(on, Mapping):
        return _diff(off, on)
    return (off, on)


def _comparable(step: dict[str, object]) -> dict[str, object]:
    """Drop the two fields the contract lets differ or that hash the raw stream."""
    view = step["progress_view"]
    assert isinstance(view, dict)
    return {
        **{k: v for k, v in step.items() if k != "stored_check_rows"},
        "progress_view": {k: v for k, v in view.items() if k not in RAW_STREAM_DIGEST_KEYS},
    }


def _off_journal_bytes(root: Path) -> bytes:
    """Raw off-mode journal bytes for SEQUENCE.

    The clock and run id are pinned. The run id feeds every record hash, so a random
    one would change the bytes even with the id text masked; pinning keeps the bytes whole.
    """
    # The runtime mode snapshot records the STATEFRAME_* environment, so the dedup switch is pinned to
    # its shipped default here: the bytes must not depend on how the suite was launched.
    # A constant clock: other threads may read time_ns inside the window, so a counter would not be deterministic.
    with unittest.mock.patch.dict(os.environ, {DEDUP_ENV: "off"}), \
            unittest.mock.patch.object(kernel_continuation_journal.time, "time_ns",
                                       lambda: FIXED_CLOCK_NS), \
            unittest.mock.patch.object(kernel_continuation_journal.uuid, "uuid4",
                                       lambda: uuid.UUID(int=FIXED_RUN_ID_INT)):
        run = _drive(root, "off")
    return run.journal_path.read_bytes()


def test_dedup_on_journals_identical_rerecords_as_unchanged_executions(tmp_path):
    off = _drive(tmp_path / "off", "off")
    on = _drive(tmp_path / "on", "on")

    # Guard against a vacuous pass: the mode must actually have taken effect.
    assert [row["status"] for row in _check_rows(off.records)] == ["PASS", "PASS", "FAIL", "FAIL"]
    assert [row["status"] for row in _check_rows(on.records)] == ["PASS", "FAIL"]
    assert _kinds(off.records) == [CHECK_RECEIPT] * 4
    assert _kinds(on.records) == [
        CHECK_RECEIPT, CHECK_RECEIPT_UNCHANGED, CHECK_RECEIPT, CHECK_RECEIPT_UNCHANGED,
    ]
    assert _unchanged_rows(off.records) == []

    off_checks = _check_rows(off.records)
    on_checks = _check_rows(on.records)
    assert [_diff(off_checks[0], on_checks[0]), _diff(off_checks[2], on_checks[1])] == [{}, {}]
    # Each suppressed re-record names the execution it re-recorded.
    assert [row["execution_id"] for row in _unchanged_rows(on.records)] == [
        off_checks[1]["execution_id"], off_checks[3]["execution_id"],
    ]
    assert len(_other_rows(on.records)) == len(_other_rows(off.records))
    assert [_diff(a, b) for a, b in zip(_other_rows(off.records), _other_rows(on.records))] == \
        [{} for _ in _other_rows(off.records)]


def test_next_auto_allocated_execution_id_is_mode_independent(tmp_path):
    off = _drive(tmp_path / "off", "off")
    on = _drive(tmp_path / "on", "on")
    assert on.next_execution_id == off.next_execution_id


def test_dedup_on_returns_the_same_public_receipt_at_every_step(tmp_path):
    off = _drive(tmp_path / "off", "off")
    on = _drive(tmp_path / "on", "on")
    for index, (label, _receipt) in enumerate(SEQUENCE):
        assert on.returned[index] == off.returned[index], f"returned receipt differs at step {label}"


def test_dedup_on_leaves_every_consumer_projection_identical_at_every_step(tmp_path):
    off = _drive(tmp_path / "off", "off")
    on = _drive(tmp_path / "on", "on")
    mismatches: dict[str, object] = {}
    for index, (label, _receipt) in enumerate(SEQUENCE):
        off_step = _comparable(off.steps[index])
        on_step = _comparable(on.steps[index])
        step_diff = {
            key: _diff_value(off_step.get(key), on_step.get(key))
            for key in set(off_step) | set(on_step) if off_step.get(key) != on_step.get(key)
        }
        if step_diff:
            mismatches[label] = step_diff
    assert not mismatches, f"consumer projections differ (off, on) per step: {mismatches}"


def test_check_execution_count_matches_on_and_off_while_stored_rows_may_not(tmp_path):
    off = _drive(tmp_path / "off", "off")
    on = _drive(tmp_path / "on", "on")
    expected_executions = [index + 1 for index in range(EXECUTION_COUNT)]
    assert [s["continuation_live"]["check_execution_count"] for s in off.steps] == expected_executions
    assert [s["continuation_live"]["check_execution_count"] for s in on.steps] == expected_executions
    assert [s["stored_check_rows"] for s in off.steps] == [1, 2, 3, 4]
    assert [s["stored_check_rows"] for s in on.steps] == [1, 1, 2, 2]


def test_on_mode_final_projection_follows_the_changed_receipt(tmp_path):
    on = _drive(tmp_path / "on", "on")
    final = on.steps[-1]
    assert final["latest_check"]["status"] == "FAIL"
    assert final["vacuity_truth"]["last_status"] == "FAIL"
    assert final["vacuity_truth"]["changed_since_check"] is False
    assert final["stored_check_rows"] == 2


def test_resume_rebuilds_the_same_execution_count_in_both_modes(tmp_path):
    for mode in ("off", "on"):
        live = _drive(tmp_path / mode, mode)
        resumed = ContinuationJournal.resume(
            live.journal_path, task_id=TASK_ID, manifest=MANIFEST, expected_head=live.head)
        try:
            # records() replays the stream and rebuilds the projections from the journal.
            # The fresh allocation in _drive is one more execution and one more stored row.
            list(resumed.records())
            executions = EXECUTION_COUNT + 1
            assert getattr(resumed, "check_execution_count", None) == executions, mode
            assert resumed.check_receipt_count == len(_check_rows(live.records)) + 1, mode
            assert _next_check_execution_ordinal(resumed) == executions + 1, mode
        finally:
            resumed.close()


def test_off_mode_journal_bytes_match_base(tmp_path):
    actual = _off_journal_bytes(tmp_path / "off")
    assert actual == GOLDEN_OFF_JOURNAL.read_bytes()
