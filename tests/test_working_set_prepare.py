"""Exact floor/core adapter and bounded complete-candidate token admission."""
from dataclasses import replace
import hashlib
import json

import pytest

from kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
from kernel_working_set_allocator import FULL, DEFERRED
from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
from kernel_working_set_render import render_assignment
from kernel_working_set_prepare import (
    Blocked, Measurement, PreparedCandidate, prepare_candidate,
)
from kernel_working_set_stop import ReasonCode


def source(key, text="x" * 1000 + "\n", state="RESIDENT", ordinal=1):
    identity = ReceiptIdentity(key + ".py", "sha256:" + "1" * 64, 1, 1,
        0, len(text.encode()), hashlib.sha256(text.encode()).hexdigest(), None)
    digests, numbers = line_digests_with_numbers(text, 1)
    return SourceSlot(key, (key,), identity, text, "[read page]\n" + text,
        digests, numbers, json.dumps({"path": key + ".py", "start_line": 1,
        "end_line": 1}).encode(), ordinal, ordinal, state)


def snapshot(*sources, **kwargs):
    body = {"model": "local", "messages": [{"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "opaque suffix"}], "max_tokens": 100}
    return PreparationSnapshot(json.dumps(body).encode(), tuple(sources),
        "request", 1, kwargs.pop("byte_cap", 100_000), **kwargs)


def test_floor_infeasible_uses_exact_wire_and_never_measures_tokens():
    snap = snapshot(source("a", state="DUE"), byte_cap=1)
    calls = []
    result = prepare_candidate(snap, measure=lambda wire: calls.append(wire))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    assert result.floor_bytes == len(result.proof.mandatory_wire)
    result.proof.validate()
    assert not calls


def test_unproven_policy_minimum_cannot_certify_mandatory_impossibility(monkeypatch):
    import kernel_working_set_prepare as adapter
    original = snapshot(source("a", "v\n"))
    full = render_assignment(original, {"a": FULL})
    deferred = render_assignment(original, {"a": DEFERRED})
    assert len(full.wire) < len(deferred.wire)
    snap = replace(original, byte_cap=len(full.wire))
    # Simulate a coupled local minimum hiding a smaller assignment. Both
    # candidates carry real valid certificates; only the policy search errs.
    monkeypatch.setattr(adapter, "_minimum_completion",
        lambda _snapshot, _slots: ({"a": DEFERRED}, deferred))
    result = prepare_candidate(snap)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.BYTE_POLICY_DOMAIN_EXHAUSTED
    assert result.proof is None


def test_immutable_envelope_can_prove_floor_failure_with_optional_slots():
    result = prepare_candidate(snapshot(source("a"), byte_cap=1))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    result.proof.validate()


def test_shared_full_carrier_can_be_cheaper_than_independent_notice_costs():
    a = source("a", "v\n")
    b = replace(a, key="b", obligation_ids=("b",), admission_ordinal=2)
    original = snapshot(a, b)
    full = render_assignment(original, {"a": FULL, "b": FULL}, "plain")
    snap = replace(original, byte_cap=len(full.wire))
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate)
    assert result.total_bytes <= snap.byte_cap


def test_floor_considers_simultaneous_removal_of_shared_notice_shell(monkeypatch):
    from types import SimpleNamespace
    import kernel_working_set_prepare as adapter
    from kernel_working_set_allocator import DEFERRED, PriorityTier, Slot

    snap = snapshot(source("a"), source("b", ordinal=2))
    slots = tuple(Slot(s.key, "product", PriorityTier.OPTIONAL,
        s.admission_ordinal, s.last_use_ordinal, ((FULL, 1), (DEFERRED, 0)))
        for s in snap.sources)
    # Two independent promotions each grow the wire. Together they eliminate
    # the aggregate omission shell and make the whole request smaller.
    def render(_snapshot, modes):
        full = sum(mode == FULL for mode in modes.values())
        return SimpleNamespace(wire=b"x" * {0: 100, 1: 105, 2: 90}[full])
    monkeypatch.setattr(adapter, "_best_render", render)
    modes, floor = adapter._minimum_completion(snap, slots)
    assert set(modes.values()) == {FULL}
    assert len(floor.wire) == 90


def test_whole_wire_cost_never_admits_over_cap_at_count_digit_boundary():
    sources = tuple(source(str(i), ordinal=i + 1) for i in range(12))
    snap = snapshot(*sources)
    full = render_assignment(snap, {s.key: FULL for s in sources}, "plain")
    snap = replace(snap, byte_cap=len(full.wire) - 1)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate)
    assert result.total_bytes == len(result.rendered.wire) <= snap.byte_cap
    assert not result.fits_without_degrade


def test_smallest_candidate_is_measured_even_after_ratio_skips_below_it():
    snap = snapshot(*(source(str(i), ordinal=i + 1) for i in range(15)), prompt_limit=100)
    calls = []
    def measure(wire):
        calls.append(wire)
        # The ratio collapses the search cap below every possible candidate.
        return (1_000_000, 0, "test")
    result = prepare_candidate(snap, measure=measure)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED
    assert 1 <= len(calls) <= 8
    assert len(calls[-1]) == result.floor_bytes
    assert result.proof is None


def test_verified_byte_bound_is_never_a_successful_token_measurement():
    result = prepare_candidate(snapshot(source("a")),
        measure=lambda wire: (1, 0, "verified_byte_bound"))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED


def test_frame_and_output_context_limits_are_admitted_together():
    snap = snapshot(source("a", state="DUE"), prompt_limit=100,
                    frame_limit=10, context_limit=200)
    result = prepare_candidate(snap, measure=lambda wire: Measurement(60, 11, "test"))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED
    result = prepare_candidate(snap, measure=lambda wire: Measurement(101, 9, "test"))
    assert isinstance(result, Blocked)  # 101 prompt + 100 allocated output > 200
    result = prepare_candidate(snap, measure=lambda wire: Measurement(100, 10, "test"))
    assert isinstance(result, PreparedCandidate)
    assert json.loads(result.wire)["max_tokens"] == 100
    # Normalize output before cost/certification, then prove that exact body.
    smaller = prepare_candidate(replace(snap, context_limit=150),
        measure=lambda wire: Measurement(100, 10, "test"))
    assert isinstance(smaller, PreparedCandidate)
    assert json.loads(smaller.wire)["max_tokens"] == 50


def test_contract_failure_does_not_try_to_repair_source_or_identity():
    calls = []
    def measure(wire):
        calls.append(wire)
        return Measurement(1, 0, "test", contract_valid=False)
    result = prepare_candidate(snapshot(source("a")), measure=measure)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.PREPARATION_CONTRACT_FAILURE
    assert len(calls) == 1


def test_measurement_identity_can_change_once_without_resetting_budget():
    calls = []
    def measure(wire):
        calls.append(wire)
        return Measurement(1, 0, "test", tokenizer_identity=str(len(calls)))
    snap = replace(snapshot(source("a")), tokenizer_identity="initial")
    result = prepare_candidate(snap, measure=measure)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.PREPARATION_CONTRACT_FAILURE
    assert len(calls) == 2


@pytest.mark.parametrize("initial,expected_calls", [("initial", 2), ("", 1)])
def test_admitted_certificate_binds_the_selected_tokenizer(initial, expected_calls):
    calls = []
    def measure(wire):
        calls.append(wire)
        return Measurement(1, 0, "test", tokenizer_identity="new-authority")
    snap = replace(snapshot(source("a")), tokenizer_identity=initial)
    result = prepare_candidate(snap, measure=measure)
    assert isinstance(result, PreparedCandidate)
    assert len(calls) == expected_calls
    assert result.rendered.certificate.snapshot.tokenizer_identity == "new-authority"
    assert result.measurement.tokenizer_identity == "new-authority"


def test_offline_measurement_is_labelled_calibrated_bytes():
    result = prepare_candidate(snapshot(source("a")))
    assert isinstance(result, PreparedCandidate)
    assert result.measurement.method == "calibrated_bytes"


@pytest.mark.parametrize("bad", [(-1, 0, "test"), (True, 0, "test"), (1, -1, "test")])
def test_invalid_measurements_fail_closed(bad):
    result = prepare_candidate(snapshot(source("a")), measure=lambda wire: bad)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.PREPARATION_CONTRACT_FAILURE
