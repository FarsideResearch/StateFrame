"""Continuation evidence joins consume each durable journal row once."""
import json
from time import perf_counter
from types import SimpleNamespace

import pytest

from kernel_continuation_journal import ContinuationJournal
from kernel_continuation_runtime import LiveContinuation, _ContinuationJournalView
from kernel_guarded_effects import GuardedEffectSequence


def _live(journal):
    live = LiveContinuation.__new__(LiveContinuation)
    live.journal = journal
    return live


def _renderable_live(journal, *, guarded):
    live = _live(journal)
    live.engine = SimpleNamespace(
        state=SimpleNamespace(intentions={}, active_id=None, revision=0),
        unfinished=lambda: False, retention_summary=lambda: None)
    live.runtime = SimpleNamespace(continuation_source_refresh=False)
    live.certified = False
    live.context_scope = None
    live._stranded_blocker = None
    live._unknown_outcome_adapter = lambda: None
    live.guarded_effects = (GuardedEffectSequence(journal, enabled_override=True)
                            if guarded else None)
    return live


def _forty_steps(journal, live, *, view_type=_ContinuationJournalView):
    live._journal_view = view_type(journal)
    live._view()
    counts = {}
    timings = {}
    for step in range(1, 41):
        # Appends use the ordinary journal path, without notifying the view.
        journal.emit("guarded_effect_intent", effect_id=f"effect-{step}")
        journal.emit("action_outcome_closed", record={
            "continuation_effect_id": f"effect-{step}", "tool": "shell",
            "result": "NO_OP", "step": step,
        })
        before_scanned = live._journal_view.scanned_total
        start = perf_counter()
        view = live._view()
        context = json.loads(live.context().split("\n", 1)[1])
        elapsed = perf_counter() - start
        if live.guarded_effects is None:
            assert "guarded_effects" not in context
        else:
            assert len(context["guarded_effects"]["effects"]) == step
        if step in {5, 20, 40}:
            counts[step] = view.scanned_total - before_scanned
            timings[step] = elapsed
            assert len(view.closures_by_effect) == step
            assert view.closures_by_effect[f"effect-{step}"][-1]["sequence"] == view.cursor
    return counts, timings


def _assert_incremental(counts):
    assert counts == {5: 2, 20: 2, 40: 2}


@pytest.mark.parametrize("guarded", [False, True])
def test_forty_step_live_view_resolves_only_new_records_and_keeps_journal_bytes(
        tmp_path, monkeypatch, guarded):
    import kernel_candidate_restart_reconcile

    monkeypatch.setattr(kernel_candidate_restart_reconcile, "enabled", lambda: False)
    journal = ContinuationJournal(tmp_path, "view-perf", {"origin": "test"})
    try:
        live = _renderable_live(journal, guarded=guarded)
        live._view()  # The initial validated scan occurs once.
        counts, _timings = _forty_steps(journal, live)
        _assert_incremental(counts)
        assert live._view().last_sync_count == 0
        assert live._journal_view.scanned_total == journal._sequence
        before = journal.path.read_bytes()
        live._view()
        assert journal.path.read_bytes() == before
    finally:
        journal.close()


@pytest.mark.parametrize("guarded", [False, True])
def test_full_scan_control_fails_incremental_counter(tmp_path, monkeypatch, guarded):
    import kernel_candidate_restart_reconcile

    monkeypatch.setattr(kernel_candidate_restart_reconcile, "enabled", lambda: False)
    class FullScanView(_ContinuationJournalView):
        def _new_records(self):
            return self.journal.records()

    journal = ContinuationJournal(tmp_path, "view-control", {"origin": "test"})
    try:
        counts, _ = _forty_steps(journal, _renderable_live(journal, guarded=guarded),
                                view_type=FullScanView)
        assert counts[5] < counts[20] < counts[40]
        with pytest.raises(AssertionError):
            _assert_incremental(counts)
    finally:
        journal.close()


def test_static_journal_double_is_read_once():
    calls = []

    def records():
        calls.append(True)
        return ({"sequence": 1, "kind": "unrelated_evidence"},)

    view = _ContinuationJournalView(SimpleNamespace(records=records))
    for _ in range(20):
        view.sync()
    assert len(calls) == 1
    assert view.scanned_total == 1


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("planted_context_read", [False, True])
def test_forty_step_prepare_and_context_consume_only_new_rows(
        tmp_path, monkeypatch, guarded, planted_context_read):
    import kernel_candidate_restart_reconcile

    monkeypatch.setattr(kernel_candidate_restart_reconcile, "enabled", lambda: False)
    journal = ContinuationJournal(tmp_path, "reconcile-perf", {"origin": "test"})
    state = SimpleNamespace(effects={}, intentions={"root": SimpleNamespace(parent_id=None)},
                            revision=0, active_id=None)
    live = _live(journal)
    live.engine = SimpleNamespace(
        state=state, settle_external_effect=lambda **_kwargs: None,
        retention_checkpoint=lambda: None,
        retention_summary=lambda: None, unfinished=lambda: False,
        choose_next=lambda _current: object())
    live.runtime = SimpleNamespace(binding=lambda: {}, continuation_source_refresh=False)
    live.completed = False
    live.certified = False
    live.context_scope = None
    live._stranded_blocker = None
    live._unknown_outcome_adapter = lambda: None
    live.guarded_effects = (GuardedEffectSequence(journal, enabled_override=True)
                            if guarded else None)
    if guarded and planted_context_read:
        sequence = live.guarded_effects
        original_context = sequence.context_dict

        def old_context():
            tuple(sequence._records())
            return original_context()

        sequence.context_dict = old_context
    live._reconcile_unknown_outcomes = lambda: set()
    settled = []

    def settle_effect(effect_id, *_args, **_kwargs):
        settled.append(effect_id)
        state.effects[effect_id].status = "not_applied"

    live._settle_durable_effect = settle_effect
    processed = 0
    original_records = journal.records

    def counted_records(*, rebuild=True):
        nonlocal processed
        for row in original_records(rebuild=rebuild):
            processed += 1
            yield row

    journal.records = counted_records
    counts, replay_counts = {}, {}
    try:
        live._view()
        for step in range(1, 41):
            effect_id = f"effect-{step}"
            journal.emit("guarded_effect_intent", effect_id=effect_id, destination="sink")
            closure = journal.emit("action_outcome_closed", record={
                "continuation_effect_id": effect_id, "tool": "shell", "result": "NO_OP",
            })
            state.effects[effect_id] = SimpleNamespace(
                effect_id=effect_id, status="pending", tool="shell",
                receipt_ref=closure["event_id"], intention_id="root")
            journal_bytes = journal.path.read_bytes()
            before_scanned = live._journal_view.scanned_total
            live.prepare([])
            context = json.loads(live.context().split("\n", 1)[1])
            assert journal.path.read_bytes() == journal_bytes
            if guarded:
                assert [effect["effect_id"] for effect in context["guarded_effects"]["effects"]] == [
                    f"effect-{index}" for index in range(1, step + 1)]
            else:
                assert "guarded_effects" not in context
            if step in {5, 20, 40}:
                counts[step] = live._journal_view.scanned_total - before_scanned
                replay_counts[step] = processed
        assert counts == {5: 2, 20: 2, 40: 2}
        print(f"guarded={guarded} planted={planted_context_read} "
              f"steps scanned={counts} replayed={replay_counts}")
        if guarded and planted_context_read:
            assert replay_counts[5] > 0
            assert replay_counts[5] < replay_counts[20] < replay_counts[40]
            with pytest.raises(AssertionError):
                assert replay_counts == {5: 0, 20: 0, 40: 0}
        else:
            assert replay_counts == {5: 0, 20: 0, 40: 0}
        assert settled == [f"effect-{step}" for step in range(1, 41)]
    finally:
        journal.close()


def _unresolved_prepare_counts(tmp_path, effect_count, *, guarded,
                               planted_full_read=False, planted_guarded_recover=False):
    tmp_path.mkdir()
    journal = ContinuationJournal(tmp_path, "unresolved-perf", {"origin": "test"})
    try:
        for index in range(60):
            journal.emit("unrelated_evidence", index=index)
        effects = {}
        for index in range(effect_count):
            effect_id = f"effect-{index}"
            closure = journal.emit("action_outcome_closed", record={
                "continuation_effect_id": effect_id, "tool": "shell", "result": "NO_OP",
            })
            effects[effect_id] = SimpleNamespace(
                effect_id=effect_id, status="unknown", tool="shell",
                receipt_ref=None, intention_id="root",
                expected_receipt=closure["event_id"],
            )
        initial_records = journal._sequence
        state = SimpleNamespace(effects=effects,
                                intentions={"root": SimpleNamespace(parent_id=None)},
                                revision=0, active_id="root")

        def settle_host_effect(*, effect_id, receipt_ref, **_kwargs):
            effect = effects[effect_id]
            assert effect.status == "unknown"
            assert receipt_ref == effect.expected_receipt
            effect.status = "not_applied"
            effect.receipt_ref = receipt_ref
            state.revision += 1
            journal.emit("continuation_transition", operation="settle_effect",
                         data={"effect_id": effect_id, "receipt_ref": receipt_ref})
            return effect.status

        live = _live(journal)
        live.engine = SimpleNamespace(
            state=state, settle_host_effect=settle_host_effect,
            settle_external_effect=lambda **_kwargs: None,
            retention_checkpoint=lambda: None,
            choose_next=lambda _current, **_kwargs: object(),
        )
        live.runtime = SimpleNamespace(binding=lambda: {})
        live.completed = False
        live.guarded_effects = (GuardedEffectSequence(journal, enabled_override=True)
                                if guarded else None)
        live._reconcile_unknown_outcomes = lambda: set()
        def settle_durable_effect(effect_id, _current, closures, **_kwargs):
            return settle_host_effect(effect_id=effect_id,
                                      receipt_ref=closures[0]["event_id"])

        live._settle_durable_effect = settle_durable_effect
        processed = 0
        original_records = journal.records

        def counted_records(*, rebuild=True):
            nonlocal processed
            for row in original_records(rebuild=rebuild):
                processed += 1
                yield row

        journal.records = counted_records
        if planted_guarded_recover:
            sequence = live.guarded_effects
            original_state = sequence._state

            def old_state(effect_id):
                tuple(sequence._records())
                return original_state(effect_id)

            sequence._state = old_state
        if planted_full_read:
            original_settle = live._settle_durable_effect

            def old_settle(*args, **kwargs):
                tuple(journal.records())
                return original_settle(*args, **kwargs)

            live._settle_durable_effect = old_settle

        live.prepare([])
        assert {effect.status for effect in effects.values()} == {"not_applied"}
        live._view()  # Include the final settlement appended during prepare.
        scanned_first = live._journal_view.scanned_total
        journal.emit("unrelated_evidence", index="new")
        live.prepare([])
        scanned_second = live._journal_view.scanned_total - scanned_first
        return processed, scanned_first, scanned_second, initial_records
    finally:
        journal.close()


@pytest.mark.parametrize("effect_count", [1, 5, 20])
@pytest.mark.parametrize("guarded", [False, True])
def test_unresolved_prepare_processes_only_new_rows_independent_of_effect_count(
        tmp_path, monkeypatch, effect_count, guarded):
    import kernel_candidate_restart_reconcile

    monkeypatch.setattr(kernel_candidate_restart_reconcile, "enabled", lambda: False)
    actual = _unresolved_prepare_counts(tmp_path / "actual", effect_count, guarded=guarded)
    planted = _unresolved_prepare_counts(tmp_path / "planted", effect_count,
                                         guarded=guarded, planted_full_read=True)
    assert actual == (0, actual[3] + effect_count, 1, actual[3])
    print(f"guarded={guarded} K={effect_count} journal_rows={actual[3]} "
          f"processed={actual[0]} initial_scan={actual[1]} appended_scan={actual[2]}")
    assert planted[0] >= effect_count * actual[3]
    print(f"guarded planted K={effect_count} processed={planted[0]}")
    with pytest.raises(AssertionError):
        assert planted[0] == 0


@pytest.mark.parametrize("effect_count", [1, 5, 20])
def test_old_guarded_recover_per_effect_fails_processed_record_budget(
        tmp_path, monkeypatch, effect_count):
    import kernel_candidate_restart_reconcile

    monkeypatch.setattr(kernel_candidate_restart_reconcile, "enabled", lambda: False)
    actual = _unresolved_prepare_counts(tmp_path / "actual", effect_count, guarded=True)
    planted = _unresolved_prepare_counts(tmp_path / "planted", effect_count,
                                         guarded=True, planted_guarded_recover=True)
    assert actual[0] == 0
    assert planted[0] >= effect_count * actual[3]
    with pytest.raises(AssertionError):
        assert planted[0] == 0


@pytest.mark.parametrize("effect_count", [1, 5, 20])
def test_candidate_resume_reconcile_reads_only_new_rows_per_effect(
        tmp_path, monkeypatch, effect_count):
    import kernel_candidate_restart_reconcile as candidate

    journal = ContinuationJournal(tmp_path, "candidate-perf", {"origin": "test"})
    try:
        for index in range(60):
            journal.emit("unrelated_evidence", index=index)
        effects = {}
        for index in range(effect_count):
            effect_id = f"effect-{index}"
            journal.emit("candidate_restart_intent", effect_id=effect_id)
            effects[effect_id] = SimpleNamespace(
                effect_id=effect_id, tool=candidate.EDIT_TOOL, status="unknown")
        runtime = SimpleNamespace(
            journal=journal,
            continuation=SimpleNamespace(engine=SimpleNamespace(
                state=SimpleNamespace(effects=effects))),
        )
        full_reads = 0
        original_records = journal.records

        def counted_records(*, rebuild=True):
            nonlocal full_reads
            full_reads += 1
            return original_records(rebuild=rebuild)

        initial_sequence = journal._sequence

        def reconcile_effect(_runtime, effect, _intent, index):
            assert index.last_sequence == (initial_sequence
                                           + int(effect.effect_id.split("-")[1]))
            event = journal.emit("candidate_restart_reconciled", effect_id=effect.effect_id)
            return {event["event_id"]}

        restored = []
        monkeypatch.setattr(journal, "records", counted_records)
        monkeypatch.setattr(candidate, "enabled", lambda: True)
        monkeypatch.setattr(candidate, "_reconcile_effect", reconcile_effect)
        monkeypatch.setattr(candidate, "_restore_latest_recovery",
                            lambda _runtime, rows: restored.extend(rows))
        assert len(candidate.reconcile(runtime)) == effect_count
        assert full_reads == 1
        assert sum(row.get("kind") == "candidate_restart_reconciled"
                   for row in restored) == effect_count
    finally:
        journal.close()


def _candidate_restart_processed(tmp_path, effect_count, monkeypatch, *, planted_linear=False):
    import kernel_candidate_restart_reconcile as candidate

    journal = ContinuationJournal(tmp_path, "restart-scan", {"origin": "test"})
    try:
        for number in range(60):
            journal.emit("step_dispatch_binding", request_id=f"other-{number}",
                         step=number, continuation_effect_id=f"other-effect-{number}")
        effects = {}
        for number in range(effect_count):
            effect_id = f"effect-{number}"
            journal.emit("candidate_restart_intent", effect_id=effect_id,
                         path="missing.txt", candidate={})
            effects[effect_id] = SimpleNamespace(
                effect_id=effect_id, tool=candidate.EDIT_TOOL, status="unknown",
                dispatch_ref=f"dispatch-{number}", intention_id="root",
                request_id=f"request-{number}", step=number)
        processed = 0
        original_records = journal.records

        class CountedRecord(dict):
            def get(self, key, default=None):
                nonlocal processed
                processed += 1
                return super().get(key, default)

        def counted_records(*, rebuild=True):
            for row in original_records(rebuild=rebuild):
                yield CountedRecord(row)

        journal.records = counted_records
        initial_sequence = journal._sequence
        if planted_linear:
            original_index = candidate._RestartIndex

            class LinearIntents:
                def __init__(self, rows):
                    self.rows = rows

                def get(self, effect_id):
                    result = None
                    for row in self.rows:
                        if (row.get("kind") == "candidate_restart_intent"
                                and row.get("effect_id") == effect_id):
                            result = row
                    return result

            class LinearIndex(original_index):
                def __init__(self, rows):
                    super().__init__(rows)
                    self.intents = LinearIntents(rows)

            monkeypatch.setattr(candidate, "_RestartIndex", LinearIndex)
        monkeypatch.setattr(candidate, "enabled", lambda: True)
        runtime = SimpleNamespace(
            journal=journal, owner=SimpleNamespace(root=tmp_path),
            continuation=SimpleNamespace(engine=SimpleNamespace(
                state=SimpleNamespace(effects=effects))),
        )
        candidate.reconcile(runtime)
        assert len(journal.cached_records_since(initial_sequence)) == effect_count
        return processed
    finally:
        journal.close()


@pytest.mark.parametrize("effect_count", [1, 5, 20])
def test_restart_reconcile_record_work_scales_with_rows_and_effects(
        tmp_path, monkeypatch, effect_count):
    processed = _candidate_restart_processed(tmp_path, effect_count, monkeypatch)
    print(f"restart K={effect_count} processed_lookups={processed}")
    assert processed <= 8 * (60 + effect_count) + 20 * effect_count


def test_linear_restart_intent_search_fails_processed_record_budget(tmp_path, monkeypatch):
    planted = _candidate_restart_processed(tmp_path, 20, monkeypatch,
                                           planted_linear=True)
    print(f"restart planted K=20 processed_lookups={planted}")
    with pytest.raises(AssertionError):
        assert planted <= 8 * 80 + 20 * 20


def test_guarded_view_matches_journal_replay_after_new_receipt(tmp_path):
    journal = ContinuationJournal(tmp_path, "guarded-view", {"origin": "test"})
    try:
        journal.emit("guarded_effect_intent", effect_id="received", destination="sink")
        journal.emit("guarded_effect_dispatch", effect_id="received")
        journal.emit("guarded_effect_receipt", effect_id="received", outcome="PRESENT",
                     receipt={"done": True})
        journal.emit("guarded_effect_intent", effect_id="pending", destination="sink")
        sequence = GuardedEffectSequence(journal, enabled_override=True)
        cases = ("received", "pending", "absent")
        expected = {effect_id: sequence.recover(effect_id) for effect_id in cases}
        before = journal.path.read_bytes()
        sequence._journal_view = _ContinuationJournalView(journal)
        assert {effect_id: sequence.recover(effect_id) for effect_id in cases} == expected
        assert journal.path.read_bytes() == before
        journal.emit("guarded_effect_dispatch", effect_id="pending")
        journal.emit("guarded_effect_receipt", effect_id="pending", outcome="PRESENT",
                     receipt={"done": True})
        before = journal.path.read_bytes()
        replay = GuardedEffectSequence(journal, enabled_override=True)
        assert sequence.recover("pending") == replay.recover("pending")
        assert journal.path.read_bytes() == before
    finally:
        journal.close()


def test_guarded_context_keeps_first_intent_order_and_latest_state(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "guarded-context", {"origin": "test"})
    try:
        journal.emit("guarded_effect_dispatch", effect_id="late-intent")
        journal.emit("guarded_effect_intent", effect_id="first-intent")
        journal.emit("guarded_effect_intent", effect_id="late-intent")
        journal.emit("guarded_effect_intent", effect_id="first-intent")
        sequence = GuardedEffectSequence(journal, enabled_override=True)

        def replay_context():
            effects = {}
            for event in sequence._records():
                if (event.get("kind") == "guarded_effect_intent"
                        and isinstance(event.get("effect_id"), str)):
                    effects[event["effect_id"]] = sequence._snapshot(event["effect_id"])
            return {"schema": "guarded-effect-sequence/1",
                    "event_schema": "guarded-effect-sequence/1",
                    "effects": list(effects.values())}

        expected = replay_context()
        assert sequence.context_dict() == expected
        journal.emit("guarded_effect_receipt", effect_id="late-intent",
                     outcome="PRESENT", receipt={"done": True})
        expected = replay_context()
        monkeypatch.setattr(journal, "records", lambda **_kwargs: pytest.fail("full replay"))
        rendered = sequence.context_dict()
        assert rendered == expected
        live = _renderable_live(journal, guarded=True)
        live.guarded_effects = sequence
        actual_context = live.context()
        original_context = sequence.context_dict
        sequence.context_dict = lambda: expected
        assert live.context() == actual_context
        sequence.context_dict = original_context
        assert [effect["effect_id"] for effect in rendered["effects"]] == [
            "first-intent", "late-intent"]
        assert rendered["effects"][1]["has_receipt"] is True
        assert rendered["effects"][1]["receipt"] == {"done": True}
    finally:
        journal.close()


def test_candidate_promotion_uses_restored_link_without_prepare_replay(tmp_path, monkeypatch):
    import kernel_candidate_restart_reconcile as candidate

    journal = ContinuationJournal(tmp_path, "candidate-link", {"origin": "test"})
    try:
        journal.emit("candidate_restart_reconciled", outcome="APPLIED",
                     candidate_digest="prior")
        journal.emit("candidate_restart_portfolio_linked", candidate_digest="prior",
                     candidate_id="existing")
        preserved = []
        runtime = SimpleNamespace(
            journal=journal, portfolio=object(), owner=SimpleNamespace(last_test_exit=0),
            portfolio_preserve_candidate=lambda **_kwargs: (
                preserved.append(True) or SimpleNamespace(candidate_id="new")),
        )
        candidate._restore_latest_recovery(runtime, list(journal.cached_records_since(0)))
        assert runtime._candidate_restart_recovery["portfolio_candidate_id"] == "existing"
        monkeypatch.setattr(candidate, "enabled", lambda: True)
        monkeypatch.setattr(journal, "records", lambda **_kwargs: pytest.fail("full journal replay"))
        candidate.maybe_promote_portfolio(runtime)
        assert preserved == []

        runtime._candidate_restart_recovery = {"outcome": "APPLIED",
                                               "candidate_digest": "new-digest"}
        before = journal._sequence
        candidate.maybe_promote_portfolio(runtime)
        candidate.maybe_promote_portfolio(runtime)
        assert preserved == [True]
        assert runtime._candidate_restart_recovery["portfolio_candidate_id"] == "new"
        linked = journal.cached_records_since(before)
        assert len(linked) == 1
        assert linked[0]["kind"] == "candidate_restart_portfolio_linked"
    finally:
        journal.close()


def test_resume_rebuilds_view_once_then_consumes_only_appends(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "view-resume", {"origin": "test"})
    for step in range(20):
        journal.emit("action_outcome_closed", record={
            "continuation_effect_id": f"effect-{step}", "tool": "shell", "result": "NO_OP",
        })
    path, head = journal.path, journal.head
    journal.close()
    full_reads = 0
    original_records = ContinuationJournal.records

    def counted_records(self, *, rebuild=True):
        nonlocal full_reads
        full_reads += 1
        return original_records(self, rebuild=rebuild)

    monkeypatch.setattr(ContinuationJournal, "records", counted_records)
    resumed = ContinuationJournal.resume(
        path, task_id="view-resume", manifest={"origin": "test"}, expected_head=head)
    try:
        assert full_reads == 1  # Trusted-head validation at resume.
        live = _live(resumed)
        view = live._view()
        assert view.last_sync_count == resumed._sequence
        assert len(view.closures_by_effect) == 20
        resumed.emit("action_outcome_closed", record={
            "continuation_effect_id": "effect-20", "tool": "shell", "result": "NO_OP",
        })
        assert live._view().last_sync_count == 1
        assert live._view().last_sync_count == 0
        assert full_reads == 1
    finally:
        resumed.close()


def test_external_receipt_lookup_uses_one_indexed_record(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "receipt-lookup", {"origin": "test"})
    try:
        for step in range(40):
            row = journal.emit("guarded_effect_receipt", effect_id=f"effect-{step}")
        monkeypatch.setattr(journal, "records", lambda: pytest.fail("full journal replay"))
        assert journal.record_by_event_id(row["event_id"]) == row
        assert journal.record_by_event_id(f"{journal.run_id}:9999") is None
        assert journal.record_by_event_id(f"{journal.run_id}:01") is None
    finally:
        journal.close()
