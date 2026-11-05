"""Pause/resume controls are dispatch-local and never cross checkpoints."""
from __future__ import annotations

import asyncio
import contextvars
import types

import pytest

import kernel_checkpoint
import kernel_measurement
from kernel_continuation_journal import ContinuationJournal


_GAP_SPAN = "vacuity_verdict_to_tokenlab_ledger"
_BEFORE_NS = 11
_EXECUTION_NS = 100
_AFTER_NS = 17
_BASELINE_CHECKPOINTED_JOURNAL_FIELDS = frozenset({
    "path", "_lock", "_fd", "_poisoned", "_poison_cause", "_observer_failures",
    "_reporting_observer_failures", "_sequence", "_head", "_size", "_blob_cache",
    "_record_cache", "_record_cache_start", "_record_cache_bytes", "_record_cache_sizes",
    "_record_offsets", "_record_hashes", "_record_cache_limit", "_framefit_receipt_records",
    "_framefit_receipt_history", "_framefit_pending_coverage_session",
    "_framefit_receipt_cursor", "_framefit_receipt_overflow", "_framefit_latest_check",
    "_framefit_last_applied_edit_after_check", "_action_receipt_ordinal_step",
    "_action_receipt_ordinal_step_set", "_action_receipt_ordinal_counts",
    "_framefit_journal_origin_mode", "_framefit_origin_shadow_mismatches", "_replaying",
    "_notifying", "_subscribers", "_subscriber_kinds", "_subscriber_cursors",
    "_publication_stats", "host_spans", "_cold_start_root", "_cold_start_span_closed",
    "_host_event_spans", "_host_event_span_seen", "_receipt_refs", "_receipt_ref_set",
    "_response_request_ids", "_request_sequences", "_tool_reconciliation_records",
    "_journal_applied_edit_rows", "_vacuity_latest_check", "_vacuity_last_edit",
    "check_receipt_count", "_receipt_generation", "_outline_capability_reports",
    "_outline_capability_listener_token", "run_id", "task_id", "manifest", "config_hash",
})


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000

    def read(self) -> int:
        return self.now_ns

    def advance(self, duration_ns: int) -> None:
        self.now_ns += duration_ns


def _new_warm_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOST_SPANS", "on")
    clock = _Clock()
    monkeypatch.setattr(kernel_measurement.time, "perf_counter_ns", clock.read)
    journal = ContinuationJournal(tmp_path, "span-robustness", {"origin": "fixture"})
    journal.emit("request_submitted", request_id="warmup", step=0)
    return journal, clock


def _start_gap(journal: ContinuationJournal) -> None:
    journal.emit("vacuity_verdict", producer="span_robustness", verdict="pass")


def _end_gap(journal: ContinuationJournal) -> None:
    journal.emit("tokenlab_v22_unified_ledger", producer="span_robustness")


def _dispatch(journal: ContinuationJournal, effect_id: str) -> None:
    journal.emit(
        "step_dispatch_binding", producer="stateframe_runtime", accepted=True,
        continuation_effect_id=effect_id, request_id="synthetic-request", step=1,
    )


def _close_dispatch(journal: ContinuationJournal, effect_id: str) -> None:
    journal.emit("action_outcome_closed", record={
        "step": 1, "continuation_effect_id": effect_id, "result": "COMPLETED",
    })


def _gap_elapsed_ns(journal: ContinuationJournal) -> int:
    rows = journal.host_spans.trace_snapshot(names=(_GAP_SPAN,))
    return sum(int(row["elapsed_ns"]) for row in rows)


def _assert_missing_outcome_finalizes(tmp_path, monkeypatch, error_type):
    journal, clock = _new_warm_journal(tmp_path, monkeypatch)
    try:
        _start_gap(journal)
        clock.advance(_BEFORE_NS)
        _dispatch(journal, f"effect:{error_type.__name__}")
        with pytest.raises(error_type):
            raise error_type("dispatch stopped before an outcome")

        clock.advance(_EXECUTION_NS)
        journal.emit("request_submitted", request_id="after-interruption", step=2)
        assert journal._warm_host_span_pause_counts == {}
        assert journal._warm_host_span_resume_counts == {}

        _start_gap(journal)
        clock.advance(_AFTER_NS)
        _end_gap(journal)
        assert _gap_elapsed_ns(journal) == _BEFORE_NS + _AFTER_NS
    finally:
        journal.close()


def test_raised_dispatch_pause_is_cleared_at_next_submission(tmp_path, monkeypatch):
    _assert_missing_outcome_finalizes(tmp_path, monkeypatch, RuntimeError)


def test_cancelled_dispatch_pause_is_cleared_at_next_submission(tmp_path, monkeypatch):
    _assert_missing_outcome_finalizes(tmp_path, monkeypatch, asyncio.CancelledError)


def test_nested_dispatches_resume_only_after_matching_closures(tmp_path, monkeypatch):
    journal, clock = _new_warm_journal(tmp_path, monkeypatch)
    try:
        _start_gap(journal)
        clock.advance(_BEFORE_NS)
        _dispatch(journal, "effect:outer")
        _dispatch(journal, "effect:outer")
        _dispatch(journal, "effect:inner")
        clock.advance(_EXECUTION_NS)

        _close_dispatch(journal, "effect:outer")
        assert journal._warm_host_span_pause_counts == {
            ("effect", "effect:outer"): 1,
            ("effect", "effect:inner"): 1,
        }
        _close_dispatch(journal, "effect:unrelated")
        assert journal._warm_host_span_pause_counts == {
            ("effect", "effect:outer"): 1,
            ("effect", "effect:inner"): 1,
        }
        _close_dispatch(journal, "effect:inner")
        assert journal._warm_host_span_pause_counts == {
            ("effect", "effect:outer"): 1,
        }

        clock.advance(_AFTER_NS)
        _close_dispatch(journal, "effect:outer")
        assert journal._warm_host_span_pause_counts == {}
        clock.advance(_AFTER_NS)
        _end_gap(journal)
        assert _gap_elapsed_ns(journal) == _BEFORE_NS + _AFTER_NS
    finally:
        journal.close()


def test_pause_state_is_dropped_and_reinitialized_by_checkpoint_restore(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    parent = run_dir / "worker" / "measurement-runs"
    parent.mkdir(parents=True)
    journal, clock = _new_warm_journal(parent, monkeypatch)
    _start_gap(journal)
    clock.advance(_BEFORE_NS)
    _dispatch(journal, "effect:checkpoint")
    clock.advance(_EXECUTION_NS)

    frame = types.SimpleNamespace(
        f_code=test_pause_state_is_dropped_and_reinitialized_by_checkpoint_restore.__code__,
        f_locals={"journal": journal, "step": 1},
    )
    old_state = kernel_checkpoint._STATE.copy()
    old_key = kernel_checkpoint._SIGNING_KEY.copy()
    old_secrets = kernel_checkpoint._SECRETS.copy()
    old_scans = kernel_checkpoint._SCANS.copy()
    try:
        monkeypatch.setattr(kernel_checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(kernel_checkpoint, "_context_items", lambda: [])
        monkeypatch.setattr(kernel_checkpoint.contextvars, "copy_context", contextvars.Context)
        monkeypatch.setattr(kernel_checkpoint, "install_continuous_clock", lambda _saved: None)
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE["setup"] = {
            "code": frame.f_code, "line": 0, "funcs": frozenset(),
        }
        kernel_checkpoint.register_signing_key(b"span-checkpoint-test-key-0123456789")

        checkpoint = kernel_checkpoint.save(frame, 1, run_dir)
        journal.close()
        kernel_checkpoint.restore(frame, checkpoint, run_dir)

        restored = frame.f_locals["journal"]
        assert restored._warm_host_span_pause_counts == {}
        assert restored._warm_host_span_resume_counts == {}
        _close_dispatch(restored, "effect:checkpoint")
        _start_gap(restored)
        clock.advance(_AFTER_NS)
        _end_gap(restored)
        assert _gap_elapsed_ns(restored) == _BEFORE_NS + _AFTER_NS
    finally:
        active = frame.f_locals.get("journal")
        if active is not None:
            active.close()
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(old_state)
        kernel_checkpoint._SIGNING_KEY[:] = old_key
        kernel_checkpoint._SECRETS[:] = old_secrets
        kernel_checkpoint._SCANS.clear()
        kernel_checkpoint._SCANS.update(old_scans)


_SPAN_PAUSE_FIELDS = frozenset({
    "_warm_host_span_pause_counts", "_warm_host_span_resume_counts",
})


def test_checkpointed_journal_attributes_exclude_span_pause_state(tmp_path):
    # Pin only what this change adds: the pause bookkeeping must never be checkpointed. A frozen copy of the whole
    # field set broke on the next release that added an unrelated journal field (v0.4.26 _subscriber_cursor_kinds).
    journal = ContinuationJournal(tmp_path, "checkpoint-attributes", {"origin": "fixture"})
    try:
        state = set(kernel_checkpoint._journal_state(journal))
        assert _SPAN_PAUSE_FIELDS <= set(vars(journal))
        assert not (state & _SPAN_PAUSE_FIELDS)
        assert _BASELINE_CHECKPOINTED_JOURNAL_FIELDS - {"_subscriber_cursor_kinds"} <= state
    finally:
        journal.close()
