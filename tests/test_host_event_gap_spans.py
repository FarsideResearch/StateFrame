from __future__ import annotations

import hashlib

import kernel_measurement
from kernel_continuation_journal import ContinuationJournal
from kernel_measurement import RequestCapture, canonical_bytes


_GAP_SPAN = "vacuity_verdict_to_tokenlab_ledger"
_EXECUTION_NS = 100
_ENGINE_BEFORE_NS = 11
_ENGINE_AFTER_NS = 17
_JOURNAL_ENVELOPE_FIELDS = frozenset({
    "schema", "run_id", "task_id", "event_id", "sequence", "timestamp_ns",
    "journal_format", "previous_hash", "record_hash",
})


class _ControllableClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000

    def read(self) -> int:
        return self.now_ns

    def advance(self, duration_ns: int) -> None:
        self.now_ns += duration_ns


def _new_warm_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOST_SPANS", "on")
    clock = _ControllableClock()
    monkeypatch.setattr(kernel_measurement.time, "perf_counter_ns", clock.read)
    journal = ContinuationJournal(tmp_path, "event-gap", {"origin": "fixture"})
    # A submission closes startup spans so subsequent event spans use the warm path.
    journal.emit("request_submitted", request_id="warmup", step=0)
    return journal, clock


def _start_gap(journal: ContinuationJournal) -> None:
    journal.emit("vacuity_verdict", producer="event_gap_fixture", verdict="pass")


def _end_gap(journal: ContinuationJournal) -> None:
    journal.emit("tokenlab_v22_unified_ledger", producer="event_gap_fixture")


def _gap_elapsed_ns(journal: ContinuationJournal) -> int:
    rows = journal.host_spans.trace_snapshot(names=(_GAP_SPAN,))
    return sum(int(row["elapsed_ns"]) for row in rows)


def _dispatch_boundary(journal: ContinuationJournal) -> None:
    # This is the engine's journal boundary immediately before host execution.
    journal.emit(
        "step_dispatch_binding", producer="stateframe_runtime",
        accepted=True, request_id="synthetic-request", step=1,
    )


def _emit_test_run_events(
        journal: ContinuationJournal, *, started_ns: int, duration_ns: int) -> None:
    finished_ns = started_ns + duration_ns
    journal.emit(
        "test_run_profile", event_schema="stateframe-test-run-profile/1",
        execution_id="synthetic-execution", profile_mode="on",
        monotonic_started_ns=started_ns, monotonic_finished_ns=finished_ns,
        total_host_window_ns=duration_ns, spans=[], markers=[], child_runs=[],
        shadow_mismatches=[], producer="kernel_test_run_spans",
    )
    journal.emit("action_receipt", receipt={
        "receipt_id": "synthetic-receipt", "request_id": "synthetic-request",
        "turn": 1, "kind": "test", "result_class": "GREEN",
    })
    journal.emit(
        "test_subprocess_completed", producer="test_subprocess",
        selectors=[], exit_code=0, request_id="synthetic-request",
    )
    journal.emit("action_outcome_closed", record={
        "step": 1, "result": "GREEN", "receipt_ids": ["synthetic-receipt"],
    })
    journal.emit(
        "observation_committed", producer="kernel_progress_ledger",
        observation_id="synthetic-receipt",
    )


def _emit_host_action_completion(journal: ContinuationJournal) -> None:
    journal.emit("action_receipt", receipt={
        "receipt_id": "synthetic-action-receipt",
        "request_id": "synthetic-request", "turn": 1,
        "kind": "host_action", "result_class": "COMPLETED",
    })
    journal.emit("action_outcome_closed", record={
        "step": 1, "result": "COMPLETED",
        "receipt_ids": ["synthetic-action-receipt"],
    })
    journal.emit(
        "observation_committed", producer="kernel_progress_ledger",
        observation_id="synthetic-action-receipt",
    )


def test_warm_event_gap_excludes_test_execution_and_counts_both_sides(
        tmp_path, monkeypatch):
    journal, clock = _new_warm_journal(tmp_path, monkeypatch)
    try:
        _start_gap(journal)
        clock.advance(_ENGINE_BEFORE_NS)
        _dispatch_boundary(journal)

        started_ns = clock.read()
        clock.advance(_EXECUTION_NS)
        _emit_test_run_events(
            journal, started_ns=started_ns, duration_ns=_EXECUTION_NS)

        clock.advance(_ENGINE_AFTER_NS)
        _end_gap(journal)

        assert _gap_elapsed_ns(journal) == _ENGINE_BEFORE_NS + _ENGINE_AFTER_NS
    finally:
        journal.close()


def test_warm_event_gap_excludes_executor_span_and_counts_both_sides(
        tmp_path, monkeypatch):
    journal, clock = _new_warm_journal(tmp_path, monkeypatch)
    try:
        _start_gap(journal)
        clock.advance(_ENGINE_BEFORE_NS)
        _dispatch_boundary(journal)

        executor = journal.host_spans.start_span(
            "executor", boundary="request_preparation", force=True)
        assert executor is not None
        clock.advance(_EXECUTION_NS)
        journal.host_spans.finish_span(executor, ended_ns=clock.read())

        _emit_host_action_completion(journal)
        clock.advance(_ENGINE_AFTER_NS)
        _end_gap(journal)

        executor_rows = journal.host_spans.trace_snapshot(names=("executor",))
        assert sum(int(row["elapsed_ns"]) for row in executor_rows) == _EXECUTION_NS
        assert _gap_elapsed_ns(journal) == _ENGINE_BEFORE_NS + _ENGINE_AFTER_NS
    finally:
        journal.close()


def test_warm_event_gap_without_execution_counts_the_full_gap(
        tmp_path, monkeypatch):
    journal, clock = _new_warm_journal(tmp_path, monkeypatch)
    try:
        _start_gap(journal)
        full_gap_ns = _ENGINE_BEFORE_NS + _ENGINE_AFTER_NS
        clock.advance(full_gap_ns)
        _end_gap(journal)

        assert _gap_elapsed_ns(journal) == full_gap_ns
    finally:
        journal.close()


def test_request_submitted_still_forces_open_warm_gap_closed(
        tmp_path, monkeypatch):
    journal, clock = _new_warm_journal(tmp_path, monkeypatch)
    try:
        _start_gap(journal)
        clock.advance(_ENGINE_BEFORE_NS)
        journal.emit("request_submitted", request_id="next-request", step=1)
        closed_elapsed_ns = _gap_elapsed_ns(journal)

        clock.advance(_EXECUTION_NS)
        _end_gap(journal)

        gap_rows = journal.host_spans.trace_snapshot(names=(_GAP_SPAN,))
        assert len(gap_rows) == 1
        assert closed_elapsed_ns == _ENGINE_BEFORE_NS
        assert _gap_elapsed_ns(journal) == closed_elapsed_ns
    finally:
        journal.close()


def _event_payloads_without_profile(parent, monkeypatch, *, enabled: bool):
    parent.mkdir()
    monkeypatch.setenv("STATEFRAME_HOST_SPANS", "on" if enabled else "off")
    journal = ContinuationJournal(parent, "profile-only", {"origin": "fixture"})
    try:
        journal.emit("vacuity_verdict", producer="parity_fixture", verdict="pass")
        journal.emit("step_dispatch_binding", producer="stateframe_runtime", accepted=True)
        journal.emit(
            "observation_committed", producer="kernel_progress_ledger",
            observation_id="stable-observation",
        )

        body = {"messages": [{"role": "user", "content": "stable request"}]}
        body_before = canonical_bytes(body)
        capture = RequestCapture(
            journal, 1, "stable-snapshot", call_purpose="profile-parity",
            identity_content={"request": "stable"},
        )
        if enabled:
            profile = journal.host_spans.begin_request_profile(
                capture.request_id, capture.step)
            assert profile is not None
            with journal.host_spans.request_scope(capture.request_id):
                handle = journal.host_spans.start_span("profile_probe")
                journal.host_spans.finish_span(handle)

        capture._emit_host_span_profile(body, body_before)
        assert canonical_bytes(body) == body_before
        assert body_before == canonical_bytes(
            {"messages": [{"role": "user", "content": "stable request"}]})

        records = list(journal.records())
        profile_rows = [row for row in records if row.get("kind") == "host_span_profile"]
        assert bool(profile_rows) is enabled
        if enabled:
            manifest = profile_rows[0]["final_wire_manifest"]
            assert manifest["body_sha256"] == hashlib.sha256(body_before).hexdigest()
            assert manifest["body_bytes"] == len(body_before)

        return [
            (
                row["kind"],
                {key: value for key, value in row.items()
                 if key not in _JOURNAL_ENVELOPE_FIELDS and key != "kind"},
            )
            for row in records
            if row.get("kind") != "host_span_profile"
        ], body_before
    finally:
        journal.close()


def test_profile_only_events_and_request_bytes_are_unchanged(
        tmp_path, monkeypatch):
    enabled_events, enabled_wire = _event_payloads_without_profile(
        tmp_path / "enabled", monkeypatch, enabled=True)
    disabled_events, disabled_wire = _event_payloads_without_profile(
        tmp_path / "disabled", monkeypatch, enabled=False)

    assert enabled_events == disabled_events
    assert enabled_wire == disabled_wire
