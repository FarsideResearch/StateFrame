"""F4 handoff must not present an edit-invalidated decision as freshly verified."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kernel_decision_continuity_slots import SlotDecisionContinuityStore
from kernel_stateframe_f4 import render_f4_handoff
from kernel_stateframe_runtime import StateFrameRuntime

R1 = "sha256:" + "a1" * 32
R2 = "sha256:" + "b2" * 32
R3 = "sha256:" + "c3" * 32
S1, S3 = R1[7:19], R3[7:19]
EDITED = "src/app/core.py"
TIGHT = (40, 60, 40, 40, 40)


def _delta(sentence="the read of core.py shows the slice", refs=("receipt:1",)):
    return {
        "hypothesis": {"text": "the parser drops trailing tokens", "falsifiable": True},
        "basis": {"support_refs": list(refs), "sentence": sentence},
        "rejected": [{"text": "locale bug", "disposition": "contradicted", "scope": "parser",
                      "reconsider_condition": "locale appears in a trace"}],
    }


def _store(*, invalidate: bool):
    store = SlotDecisionContinuityStore()
    store.set_ref_dependencies({"receipt:1": [EDITED]})
    assert store.apply_delta(_delta(), existing_refs={"receipt:1"}, step=1, revision=R1)["accepted"]
    if invalidate:
        assert store.invalidate_after_edit(EDITED, step=2)["invalidated_refs"] == ["receipt:1"]
    return store


def _handoff(store, **kw):
    return render_f4_handoff(slots=store.snapshot(), host={"revision": R3}, **kw)["content"]


def test_stale_basis_keeps_written_revision_and_says_stale():
    text = _handoff(_store(invalidate=True))
    assert S1 in text and S3 not in text
    assert "STALE" in text and EDITED in text and "recheck" in text


def test_stale_basis_never_shows_current_revision_without_bound_revision():
    # Slots written by a pre-fix snapshot carry no bound_revision: unknown, never current.
    store = _store(invalidate=True)
    snap = store.snapshot()
    for name in ("basis", "hypothesis"):
        snap["slots"][name].pop("bound_revision")
        snap[name].pop("bound_revision")
    text = render_f4_handoff(slots=snap, host={"revision": R3})["content"]
    assert "revision unknown" in text and S3 not in text and "STALE" in text


def test_current_basis_shows_bound_revision_not_envelope_revision():
    text = _handoff(_store(invalidate=False))
    assert f"revision {S1}" in text
    assert S3 not in text and "STALE" not in text


def test_snapshot_restore_keeps_bound_revision_and_stale_status():
    store = _store(invalidate=True)
    snap = json.loads(json.dumps(store.snapshot()))
    restored = SlotDecisionContinuityStore.from_state(snap, existing_refs={"receipt:1"})
    for name in ("basis", "hypothesis"):
        assert restored.slots[name]["bound_revision"] == R1
        assert restored.slots[name]["status"] == "stale"
    assert _handoff(restored) == _handoff(store)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_json_block_keeps_status_and_short_revision_at_every_compaction_level(level):
    store = _store(invalidate=True)
    payload = store._payload(store.slots, store._host_envelope, level)
    basis, hyp = payload["slots"]["basis"], payload["slots"]["hypothesis"]
    assert basis["status"] == "stale" and basis["bound_revision"] == S1
    assert hyp["status"] == "stale" and hyp["bound_revision"] == S1
    assert store.snapshot()["basis"]["bound_revision"] == R1  # stored in full


def test_unknown_bound_revision_is_not_replaced_by_envelope_revision():
    store = SlotDecisionContinuityStore()
    store.apply_delta(_delta(), existing_refs={"receipt:1"}, step=1)
    text = _handoff(store)
    assert "revision unknown" in text and S3 not in text


def test_non_string_revision_is_not_stamped():
    store = SlotDecisionContinuityStore()
    store.apply_delta(_delta(), existing_refs={"receipt:1"}, revision=7)
    assert "bound_revision" not in store.slots["basis"]


def test_hypothesis_only_delta_leaves_basis_stamp_and_stale_intact():
    store = _store(invalidate=True)
    store.apply_delta({"hypothesis": {"text": "new idea", "falsifiable": True}},
                      existing_refs={"receipt:1"}, step=3, revision=R3)
    assert store.slots["basis"]["bound_revision"] == R1
    assert store.slots["basis"]["status"] == "stale"
    assert store.slots["hypothesis"]["bound_revision"] == R3
    assert store.slots["hypothesis"]["status"] == "unverified"


def test_unrelated_delta_keeps_original_bound_revision():
    store = _store(invalidate=False)
    store.apply_delta({"step": {"text": "rerun", "done_when": "green"}},
                      existing_refs={"receipt:1"}, step=3, revision=R3)
    assert store.slots["basis"]["bound_revision"] == R1


def test_changed_basis_sentence_restamps_when_not_stale():
    store = _store(invalidate=False)
    store.apply_delta({"basis": {"support_refs": ["receipt:1"], "sentence": "new reading"}},
                      existing_refs={"receipt:1"}, step=3, revision=R3)
    assert store.slots["basis"]["bound_revision"] == R3


def test_resent_stale_basis_stays_stale_at_old_revision():
    store = _store(invalidate=True)
    store.apply_delta({"basis": {"support_refs": ["receipt:1"], "sentence": "same claim"}},
                      existing_refs={"receipt:1"}, step=3, revision=R3)
    basis = store.slots["basis"]
    assert basis["status"] == "stale" and basis["bound_revision"] == R1
    assert basis["stale_refs"] == ["receipt:1"] and basis["invalidated_by"] == [EDITED]
    text = _handoff(store)
    assert "STALE" in text and S3 not in text


def test_basis_citing_a_newer_receipt_is_fresh():
    store = _store(invalidate=True)
    store.apply_delta({"basis": {"support_refs": ["receipt:1", "receipt:9"], "sentence": "reread"}},
                      existing_refs={"receipt:1", "receipt:9"}, step=3, revision=R3)
    basis = store.slots["basis"]
    assert basis["status"] == "unverified" and basis["bound_revision"] == R3


@pytest.mark.parametrize("limits", [(240, 300, 220, 220, 220), (80, 100, 70, 70, 80), TIGHT])
def test_stale_markers_survive_every_limit_level(limits):
    from kernel_stateframe_f4 import _handoff_text
    store = _store(invalidate=True)
    text = _handoff_text(store.snapshot(), {"revision": R3}, None, limits=limits)
    hypothesis_line = text.split("\n")[1]
    evidence_line = text.split("\n")[2]
    assert "STALE" in hypothesis_line and "STALE" in evidence_line


def test_stale_without_paths_uses_generic_wording():
    store = _store(invalidate=True)
    snap = store.snapshot()
    snap["slots"]["basis"].pop("invalidated_by")
    snap["basis"].pop("invalidated_by")
    assert "STALE since a later edit" in render_f4_handoff(slots=snap, host={})["content"]


def test_legacy_on_mode_store_accepts_revision_kwarg(monkeypatch):
    monkeypatch.setenv("STATEFRAME_DECISION_CONTINUITY", "on")
    from kernel_decision_continuity import DecisionContinuityStore
    store = DecisionContinuityStore()
    assert not isinstance(store, SlotDecisionContinuityStore)
    store.apply_delta({}, revision=R1)
    store.sync_organ_projection({}, revision=R1)


def test_sync_organ_projection_passes_revision_through():
    store = SlotDecisionContinuityStore()
    store.set_ref_dependencies({"receipt:1": [EDITED]})
    store.sync_organ_projection({"slots": _delta()}, existing_refs={"receipt:1"}, revision=R2)
    assert store.slots["basis"]["bound_revision"] == R2


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


def _runtime_stub(hashes, binding=None):
    journal = _Journal()
    stub = SimpleNamespace(
        decision_continuity=SlotDecisionContinuityStore(), journal=journal,
        owner=SimpleNamespace(current_step=4), artifact_hashes=hashes,
        _target_relative=lambda: "core.py",
        binding=lambda: binding if binding is not None else {},
        _decision_continuity_receipt_refs=lambda: {"receipt:1"},
    )
    for name in ("_decision_continuity_revision", "_decision_continuity_persist"):
        setattr(stub, name, getattr(StateFrameRuntime, name).__get__(stub))
    return stub, journal


def test_runtime_ingest_stamps_f4_revision_and_journals_it_and_restores():
    stub, journal = _runtime_stub({"core.py": R1})
    result = StateFrameRuntime.ingest_decision_continuity_delta(stub, _delta())
    assert result["accepted"]
    assert stub.decision_continuity.slots["basis"]["bound_revision"] == R1
    state = [f["state"] for kind, f in journal.events if kind == "decision_continuity_state"][-1]
    assert state["basis"]["bound_revision"] == R1
    assert state["hypothesis"]["bound_revision"] == R1
    restored = SlotDecisionContinuityStore.from_state(
        json.loads(json.dumps(state)), existing_refs={"receipt:1"})
    assert restored.slots["basis"]["bound_revision"] == R1


def test_runtime_revision_falls_back_to_binding_then_reports_unavailable():
    stub, journal = _runtime_stub({}, binding={"product_revision": R2})
    assert stub._decision_continuity_revision() == R2
    stub, journal = _runtime_stub({}, binding={"product_revision": "", "artifact": None,
                                                "files": {"core.py": R1}})
    assert stub._decision_continuity_revision() is None
    assert [k for k, _ in journal.events] == ["decision_continuity_revision_unavailable"]


def test_runtime_stale_decision_is_not_rendered_at_current_revision():
    """Real-path scenario: written at R1, edit lands, handoff built at envelope R3."""
    stub, _ = _runtime_stub({"core.py": R1})
    stub.decision_continuity.set_ref_dependencies({"receipt:1": [EDITED]})
    StateFrameRuntime.ingest_decision_continuity_delta(stub, _delta())
    stub.decision_continuity.invalidate_after_edit(EDITED)
    text = render_f4_handoff(slots=stub.decision_continuity.snapshot(),
                             host={"revision": R3})["content"]
    assert S3 not in text and "STALE" in text


def test_stale_basis_not_rendered_at_current_revision_via_public_store_api():
    """Uses only the store API that exists before the fix: assertion-level regression."""
    store = SlotDecisionContinuityStore()
    store.set_ref_dependencies({"receipt:1": [EDITED]})
    store.set_host_envelope({"revision": R1})
    assert store.apply_delta(_delta(), existing_refs={"receipt:1"}, step=1)["accepted"]
    store.invalidate_after_edit(EDITED, step=2)
    text = render_f4_handoff(slots=store.snapshot(), host={"revision": R3})["content"]
    assert S3 not in text and R3 not in text
    assert "STALE" in text


def test_multi_part_revision_is_unavailable_not_stamped():
    stub, journal = _runtime_stub({"a.py": R1, "b.py": R2})
    stub._target_relative = lambda: ""
    assert stub._decision_continuity_revision() is None
    assert journal.events[-1][0] == "decision_continuity_revision_unavailable"


def test_runtime_error_in_revision_lookup_is_non_fatal():
    def boom():
        raise RuntimeError("no binding")
    stub, journal = _runtime_stub({})
    stub.binding = boom
    assert stub._decision_continuity_revision() is None
    assert journal.events[-1][0] == "decision_continuity_revision_unavailable"


def test_fabricated_ref_cannot_launder_a_stale_basis():
    store = _store(invalidate=True)
    store.apply_delta({"basis": {"support_refs": ["receipt:NOPE"], "sentence": "same claim"}},
                      existing_refs={"receipt:1"}, step=3, revision=R3)
    basis = store.slots["basis"]
    assert basis["status"] == "stale" and basis["bound_revision"] == R1
    store.apply_delta({"basis": {"support_refs": ["receipt:1", "receipt:NOPE"], "sentence": "x"}},
                      existing_refs={"receipt:1"}, step=4, revision=R3)
    assert store.slots["basis"]["status"] == "stale"


def test_fresh_basis_clears_hypothesis_stale_even_with_identical_text():
    store = _store(invalidate=True)
    both = _delta(refs=("receipt:1", "receipt:9"))
    store.apply_delta({"hypothesis": both["hypothesis"], "basis": both["basis"]},
                      existing_refs={"receipt:1", "receipt:9"}, step=3, revision=R3)
    assert store.slots["hypothesis"]["status"] == "unverified"
    assert "invalidated_by" not in store.slots["hypothesis"]
    assert store.slots["hypothesis"]["bound_revision"] == R3
    assert "STALE" not in _handoff(store)


def test_basis_only_fresh_delta_revalidates_standing_hypothesis():
    store = _store(invalidate=True)
    store.apply_delta({"basis": {"support_refs": ["receipt:9"], "sentence": "reread"}},
                      existing_refs={"receipt:1", "receipt:9"}, step=3, revision=R3)
    hyp = store.slots["hypothesis"]
    assert hyp["status"] == "unverified" and hyp["bound_revision"] == R3
    assert "invalidated_by" not in hyp


def test_resent_identical_hypothesis_stays_stale_without_fresh_basis():
    store = _store(invalidate=True)
    store.apply_delta({"hypothesis": _delta()["hypothesis"]}, existing_refs={"receipt:1"},
                      step=3, revision=R3)
    assert store.slots["hypothesis"]["status"] == "stale"
    assert store.slots["hypothesis"]["bound_revision"] == R1


def test_long_paths_do_not_eat_the_hypothesis_at_default_limits():
    from kernel_stateframe_f4 import _handoff_text
    long_a = "src/" + "a" * 111
    long_b = "src/" + "b" * 111
    snap = _store(invalidate=True).snapshot()
    for key in ("slots", None):
        target = snap["slots"] if key else snap
        target["hypothesis"]["invalidated_by"] = [long_a, long_b]
        target["basis"]["invalidated_by"] = [long_a, long_b]
    text = _handoff_text(snap, {"revision": R3}, None, limits=(240, 300, 220, 220, 220))
    line = text.split("\n")[1]
    assert "STALE" in line and "the parser drops trailing tokens" in line


_V059_GOLDEN = "[Decision handoff]\nHypothesis/approach: drops tokens; approach: rerun the parser test.\nDecisive evidence: host recorded run_tests -> FAILED on revision sha256:c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3; interpretation is not yet supplied (revision sha256:c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3); it establishes only the bounded observation, not a final guarantee.\nRejected alternative: locale bug (contradicted); reopen if locale appears in a trace.\nNext decision: rerun the parser test; expected outcome: it is green.\nDefeasible: Revisit if relevant code changes or contrary evidence."


def test_handoff_without_model_basis_is_byte_identical_to_v0_5_9():
    from kernel_stateframe_f4 import _handoff_text
    slots = {"rejected": [{"text": "locale bug", "disposition": "contradicted",
                           "scope": "parser", "reconsider_condition": "locale appears in a trace"}],
             "step": {"text": "rerun the parser test", "done_when": "it is green"}}
    host = {"revision": R3, "last_action": "run_tests", "last_result": "FAILED"}
    assert _handoff_text(slots, host, "drops tokens",
                         limits=(240, 300, 220, 220, 220)) == _V059_GOLDEN
