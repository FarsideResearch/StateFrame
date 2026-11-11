"""Ownership drift must be observed before lazy journal rows escape to consumers.

The four ``fails today`` cases intentionally have no xfail marker.  They capture
the consumer-visible race on base 6003aba; the StateFrame lane runs them against
the candidate fix.
"""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from _timeouts import scaled

from kernel_continuation_journal import ContinuationJournal, JournalError


_APPEND_TORN_TAIL = (
    "import os, sys\n"
    "fd = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND)\n"
    "try:\n"
    "    os.write(fd, b'x')\n"
    "finally:\n"
    "    os.close(fd)\n"
)


def _external_append(path) -> None:
    """Change the journal from a real child process without taking its flock."""
    subprocess.run(
        [sys.executable, "-c", _APPEND_TORN_TAIL, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=scaled(10),
    )


def _journal(tmp_path, task_id: str = "lazy-records") -> ContinuationJournal:
    journal = ContinuationJournal(tmp_path, task_id, {"origin": "lazy-iteration-contract"})
    journal.emit("contract_row", ordinal=1)
    return journal


def _mutate_after_first_yield(monkeypatch, journal, method: str = "records") -> None:
    """Append after a consumer has processed row one, before it requests row two."""
    original = getattr(journal, method)

    def racing(*args, **kwargs):
        for index, row in enumerate(original(*args, **kwargs)):
            yield row
            if index == 0:
                _external_append(journal.path)

    monkeypatch.setattr(journal, method, racing)


def _mutate_class_records_after_first_yield(monkeypatch, path) -> None:
    original = ContinuationJournal.records

    def racing(self, *, rebuild: bool = True):
        for index, row in enumerate(original(self, rebuild=rebuild)):
            yield row
            if index == 0:
                _external_append(path)

    monkeypatch.setattr(ContinuationJournal, "records", racing)


def test_subscribe_does_not_publish_rows_after_midstream_owner_drift(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    published: list[int] = []
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            journal.subscribe("contract-projection", lambda row: published.append(row["sequence"]))

        # Row one may have been delivered before the second process changed the
        # file. No later row may be published or installed as the replay cursor.
        assert all(sequence == 1 for sequence in published)
        assert journal.subscriber_cursor("contract-projection") <= 1
        assert journal.publication_stats()["replayed"] <= 1
        assert "contract-projection" not in journal._subscribers
    finally:
        journal.close()


def test_records_does_not_advance_temporary_coverage_index_after_owner_drift(
        tmp_path, monkeypatch):
    import kernel_stateframe_call_coverage as coverage
    from kernel_continuation_journal import MEMBOUND_ENV

    monkeypatch.setenv(MEMBOUND_ENV, "bounded")
    journal = _journal(tmp_path, "coverage-index-race")
    handle = coverage._sha(["coverage-index", journal.run_id, str(journal.path)])
    context = {coverage.CONTEXT_KEY: handle}
    assert coverage.context_index(context) is None
    rows = journal.records()
    try:
        assert next(rows)["sequence"] == 1
        _external_append(journal.path)
        # A partial rebuild must remain private after the competing process
        # changes ownership, even before the consumer asks for row two.
        assert coverage.context_index(context) is None
    finally:
        rows.close()
        journal.close()


def test_parent_meter_does_not_refuse_from_a_row_after_owner_drift(tmp_path, monkeypatch):
    from kernel_pilot import ParentMeter

    allocation = ParentMeter(attempts=1, steps=None, enforce=False)
    manifest = {"parent_budget": {"limits": dict(allocation.limits), "enforced": False}}
    journal = ContinuationJournal(tmp_path, "parent-meter-race", manifest)
    journal.emit("contract_row", ordinal=1)
    journal.emit("attempt_terminated", reason="semantic refusal row")
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            ParentMeter.restore(journal, attempts=1, steps=None, enforce=False)
    finally:
        journal.close()


def test_external_effect_replay_does_not_refuse_from_a_row_after_owner_drift(
        tmp_path, monkeypatch):
    from kernel_unknown_outcomes import DISPATCH_EVENT, ExternalEffectAdapter

    journal = _journal(tmp_path, "external-effect-race")
    journal.emit(DISPATCH_EVENT)
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            ExternalEffectAdapter(journal)
    finally:
        journal.close()


def test_resume_does_not_append_resumed_event_after_drift_during_replay(tmp_path, monkeypatch):
    journal = _journal(tmp_path, "resume-race")
    path, task_id, manifest, head = journal.path, journal.task_id, journal.manifest, journal.head
    journal.close()
    _mutate_class_records_after_first_yield(monkeypatch, path)

    with pytest.raises(JournalError):
        ContinuationJournal.resume(path, task_id=task_id, manifest=manifest, expected_head=head)

    assert b'"kind":"continuation_writer_resumed"' not in path.read_bytes()


def test_checkpoint_validation_does_not_return_after_drift_during_scan(tmp_path, monkeypatch):
    journal = _journal(tmp_path, "checkpoint-validation-race")
    before_offsets = list(journal._record_offsets)
    before_hashes = list(journal._record_hashes)
    _mutate_after_first_yield(monkeypatch, journal, "_records")
    try:
        with pytest.raises(JournalError):
            journal.validate_checkpoint_journal()

        assert journal._replaying is False
        assert journal._record_offsets == before_offsets
        assert journal._record_hashes == before_hashes
    finally:
        journal.close()


@pytest.mark.parametrize(
    ("consumer", "invoke"),
    [
        ("action_receipt_ordinal", lambda journal: journal.action_receipt_ordinal(
            request_id="request", step=1, action_kind="edit")),
        ("journal_applied_edit_rows", lambda journal: journal.journal_applied_edit_rows()),
        ("vacuity_truth_records", lambda journal: journal.vacuity_truth_records()),
        ("request_records", lambda journal: journal.request_records("request")),
        ("tool_reconciliation_records", lambda journal: journal.tool_reconciliation_records()),
    ],
)
def test_stale_projection_accessors_do_not_return_after_drift(
        tmp_path, monkeypatch, consumer, invoke):
    journal = _journal(tmp_path, f"projection-{consumer}")
    journal._receipt_generation = -1
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            invoke(journal)
        assert journal._receipt_generation == -1
    finally:
        journal.close()


def test_cached_record_index_rebuild_does_not_return_a_view_after_drift(tmp_path, monkeypatch):
    journal = _journal(tmp_path, "index-rebuild-race")
    journal._record_offsets.clear()
    journal._record_hashes.clear()
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            journal.cached_records()
        assert journal._record_offsets == []
        assert journal._record_hashes == []
    finally:
        journal.close()


@pytest.mark.parametrize("consumer", ["cached_records", "cached_records_since", "_borrow_records_since"])
def test_legacy_cache_rebuild_does_not_return_after_drift(tmp_path, monkeypatch, consumer):
    journal = _journal(tmp_path, f"legacy-cache-{consumer}")
    journal._record_cache_limit = None
    journal._record_cache.clear()
    journal._record_cache_sizes.clear()
    journal._record_cache_start = 0
    journal._record_cache_bytes = 0
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            if consumer == "cached_records":
                journal.cached_records()
            else:
                getattr(journal, consumer)(0)
        assert journal._record_cache == []
    finally:
        journal.close()


@pytest.mark.parametrize("consumer", ["cached_records_since", "_borrow_records_since"])
def test_bounded_suffix_materializers_check_each_row_before_return(
        tmp_path, monkeypatch, consumer):
    journal = _journal(tmp_path, f"bounded-suffix-{consumer}")
    _mutate_after_first_yield(monkeypatch, journal, "_record_range")
    try:
        with pytest.raises(JournalError):
            getattr(journal, consumer)(0)
        assert journal._records_since_cache == {}
    finally:
        journal.close()


def test_lazy_cached_view_checks_owner_before_the_next_indexed_row(tmp_path, monkeypatch):
    journal = _journal(tmp_path, "cached-view-race")
    view = journal.cached_records()
    journal._record_cache_start = journal._sequence
    journal._record_cache.clear()
    journal._record_cache_sizes.clear()
    journal._record_cache_bytes = 0
    _mutate_after_first_yield(monkeypatch, journal, "_record_range")
    try:
        with pytest.raises(JournalError):
            list(view)
    finally:
        journal.close()


def test_candidate_restart_reconcile_materializes_before_index_or_effect_work(
        tmp_path, monkeypatch):
    import kernel_candidate_restart_reconcile as reconcile_module

    journal = _journal(tmp_path, "candidate-reconcile-race")
    index_calls: list[bool] = []
    monkeypatch.setattr(reconcile_module, "enabled", lambda: True)
    monkeypatch.setattr(reconcile_module, "_RestartIndex", lambda _rows: index_calls.append(True))
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            reconcile_module.reconcile(SimpleNamespace(journal=journal))
        assert index_calls == []
    finally:
        journal.close()


def test_completion_owner_journal_helper_materializes_before_return(tmp_path, monkeypatch):
    import kernel_host_cache
    from kernel_completion_owner import _journal_records

    journal = _journal(tmp_path, "completion-owner-race")
    monkeypatch.setattr(kernel_host_cache, "cache_enabled", lambda _name: False)
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            _journal_records(SimpleNamespace(journal=journal))
    finally:
        journal.close()


def test_guarded_effect_state_materializes_records_before_projection(tmp_path, monkeypatch):
    from kernel_guarded_effects import GuardedEffectSequence

    journal = _journal(tmp_path, "guarded-effects-race")
    sequence = GuardedEffectSequence(journal, enabled_override=True)
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            sequence._state("effect")
    finally:
        journal.close()


def test_restart_restore_materializes_before_checkpoint_decisions(tmp_path, monkeypatch):
    import kernel_candidate_restart_reconcile
    import kernel_continuation_restart as restart_module
    import kernel_true_resume

    journal = _journal(tmp_path, "restore-runtime-race")
    runtime = SimpleNamespace(
        journal=journal,
        continuation=SimpleNamespace(engine=SimpleNamespace(
            state=SimpleNamespace(intentions={}),
        )),
    )
    checkpoint_calls: list[bool] = []
    monkeypatch.setattr(kernel_true_resume, "require_enabled", lambda: None)
    monkeypatch.setattr(restart_module, "_reconcile_external_effects", lambda _runtime: set())
    monkeypatch.setattr(kernel_candidate_restart_reconcile, "enabled", lambda: False)
    monkeypatch.setattr(restart_module, "latest_checkpoint",
                        lambda _rows: checkpoint_calls.append(True))
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            restart_module.restore_runtime(runtime)
        assert checkpoint_calls == []
    finally:
        journal.close()


def test_model_router_checkpoint_scan_finishes_before_replay_decision(tmp_path, monkeypatch):
    import kernel_stateframe_model_router_replay as replay_module
    from kernel_stateframe_model_router import (
        EndpointConfig,
        ModelRouter,
        ModelRouterConfig,
    )

    journal = _journal(tmp_path, "model-router-race")
    replay_calls: list[bool] = []
    config = ModelRouterConfig(endpoints=(EndpointConfig(name="default", model_id="test-model"),))
    monkeypatch.setattr(replay_module, "latest_state", lambda *_args, **_kwargs: replay_calls.append(True))
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            ModelRouter(mode="on", config=config, journal=journal,
                        replay_enabled=True, resume=True)
        assert replay_calls == []
    finally:
        journal.close()


def test_tool_call_audit_does_not_emit_summary_from_partial_replay(tmp_path, monkeypatch):
    import kernel_host_cache
    from kernel_stateframe_runtime import StateFrameRuntime

    journal = _journal(tmp_path, "tool-call-audit-race")
    runtime = object.__new__(StateFrameRuntime)
    journal_view = SimpleNamespace(
        path=journal.path,
        events=None,
        records=journal.records,
        emit=journal.emit,
    )
    runtime.journal = journal_view
    runtime.stateframe_tool_call_loop_breaker_mode = "on"
    runtime._tool_call_cap_audit_emitted = False
    initial_sequence = journal._sequence
    monkeypatch.setattr(kernel_host_cache, "cache_enabled", lambda _name: False)
    _mutate_after_first_yield(monkeypatch, journal_view)
    try:
        # Safe consumer: it materializes with list(records()) before building the summary, so the end-of-scan
        # ownership check raises before any emit. The audit is diagnostic and fail-open (it swallows the
        # JournalError), so the contract is "no summary written", not "error propagates".
        runtime._emit_tool_call_cap_audit()
        assert runtime._tool_call_cap_audit_emitted is False
        assert journal._sequence == initial_sequence
    finally:
        journal.close()


def test_checkpoint_rehydration_does_not_publish_partial_indexes(tmp_path, monkeypatch):
    from kernel_checkpoint import CheckpointError, _rebuild_journal

    journal = _journal(tmp_path, "checkpoint-rebuild-race")
    path = journal.path
    state = journal.__dict__.copy()
    state["_fd"] = "OPEN"
    for name in (
        "_record_offsets", "_record_hashes", "_request_sequences",
        "_tool_reconciliation_records", "_journal_applied_edit_rows",
        "_vacuity_latest_check", "_vacuity_last_edit",
    ):
        state.pop(name, None)
    journal.close()
    obj = object.__new__(ContinuationJournal)
    _mutate_class_records_after_first_yield(monkeypatch, path)

    with pytest.raises(CheckpointError):
        _rebuild_journal(obj, state)

    assert obj._fd is None
    assert not hasattr(obj, "_record_offsets")
    assert not hasattr(obj, "_record_hashes")


def test_restart_journal_binding_list_completes_before_workspace_decision(
        tmp_path, monkeypatch):
    import kernel_continuation_restart as restart_module

    journal = _journal(tmp_path, "resume-binding-race")
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    workspace_dir = tmp_path / "workspace"
    initial = {
        "journal_path": str(journal.path),
        "run_id": journal.run_id,
        "task_id": journal.task_id,
        "config_hash": journal.config_hash,
    }
    reads: list[str] = []

    class Workspace:
        directory = workspace_dir

        def read(self, name):
            reads.append(name)
            return initial if name == "initialized.json" else {}

    runtime = SimpleNamespace(
        workspace=Workspace(),
        owner=SimpleNamespace(root=root, task_id=journal.task_id),
        target=target,
    )
    descriptor = SimpleNamespace(expected_journal_head=journal.head)
    monkeypatch.setattr(restart_module, "execution_paths", lambda _workspace: (root, target))
    monkeypatch.setattr(
        ContinuationJournal,
        "resume",
        classmethod(lambda _cls, *_args, **_kwargs: journal),
    )
    _mutate_after_first_yield(monkeypatch, journal)
    try:
        with pytest.raises(JournalError):
            restart_module.resume_journal(runtime, tmp_path, journal.manifest, descriptor)
        assert reads == ["initialized.json"]
    finally:
        journal.close()
