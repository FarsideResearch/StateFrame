"""Tokenizer-identity rebuilds must not redo the preparation search.

The live path measures with the provider tokenizer, so the first measurement
binds a tokenizer identity and prepare_candidate replaces the frozen snapshot.
Every identity-keyed memo used to miss after that: the whole coordinate
descent and every projected-layout trial were rendered again (and once more
when the identity changed), which is where the switch-ON request cost went.
Provider bytes cannot depend on the tokenizer identity, so a sibling snapshot
shares trial wires and the policy floor; only certificates are re-rendered.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/stateframe/engine"), str(ROOT / "src"), str(ROOT / "tests")]

import kernel_working_set_prepare as prepare_module
import kernel_working_set_render as render_module
from kernel_context import token_estimate
from kernel_working_set_prepare import Measurement, PreparedCandidate, prepare_candidate
from kernel_working_set_proof import verify_certificate
from working_set_benchmark import recorded_snapshot


def _measure(identities):
    """A tokenizer reporting the identities in order, holding the last one."""
    calls = []

    def measure(wire):
        value = json.loads(wire)
        estimate = token_estimate(value)
        identity = identities[min(len(calls), len(identities) - 1)]
        calls.append(identity)
        return Measurement(estimate, estimate, "tokenize", identity)
    return measure


@pytest.fixture
def counted(monkeypatch):
    counts = {"full": 0, "preview": 0, "descents": 0}
    original_render = render_module._render_assignment
    original_best = prepare_module._best_render

    def render(*args, **kwargs):
        counts["preview" if kwargs.get("wire_only") else "full"] += 1
        return original_render(*args, **kwargs)

    def best(*args, **kwargs):
        counts["descents"] += 1
        return original_best(*args, **kwargs)
    monkeypatch.setattr(render_module, "_render_assignment", render)
    monkeypatch.setattr(prepare_module, "_best_render", best)
    return counts


def _reset(counts):
    for key in counts:
        counts[key] = 0


def test_identity_bind_reuses_the_search(counted):
    snap = replace(recorded_snapshot(), layouts=("plain", "final_cache"))
    assert len(snap.sources) == 98 and not snap.tokenizer_identity
    offline = prepare_candidate(snap)
    assert isinstance(offline, PreparedCandidate)
    baseline = dict(counted)

    _reset(counted)
    bound = prepare_candidate(snap, measure=_measure(["tok-a"]))
    assert isinstance(bound, PreparedCandidate)
    assert bound.wire == offline.wire
    assert bound.rendered.certificate.assignment == offline.rendered.certificate.assignment
    assert bound.rendered.certificate.snapshot.tokenizer_identity == "tok-a"
    assert verify_certificate(bound.wire, bound.rendered.certificate)
    # One identity bind: the floor, the rich candidate and the admitted
    # candidate are re-rendered for their certificates; no trial is redone.
    assert counted["full"] <= baseline["full"] + 8
    assert counted["preview"] <= baseline["preview"] + 4
    assert counted["descents"] <= baseline["descents"] + 4


def test_rebound_previews_materialize_against_their_own_snapshot(monkeypatch):
    from kernel_working_set_cache import _ACTIVE, _Cache, alias, family, trial_render
    from kernel_working_set_render import (
        _ProjectedPreview, _WirePreview, materialize, preview_assignment,
    )
    snap = replace(recorded_snapshot(), layouts=("plain", "final_cache"))
    sibling = replace(snap, tokenizer_identity="tok-z")
    modes = {slot.key: render_module._slot_modes(snap, slot)[-1] for slot in snap.sources}
    token = _ACTIVE.set(_Cache())
    try:
        assert family(sibling) != family(snap)
        alias(sibling, snap)
        assert family(sibling) == family(snap)
        for layout, kind in (("plain", _WirePreview), ("final_cache", _ProjectedPreview)):
            first = trial_render(snap, modes, layout, preview_assignment)
            second = trial_render(sibling, modes, layout, preview_assignment)
            assert isinstance(second, kind) and second.snapshot is sibling
            assert second.wire == first.wire
            candidate = materialize(second)
            assert candidate.wire == first.wire
            assert candidate.certificate.snapshot is sibling
            assert candidate.certificate.layout == layout
            assert verify_certificate(candidate.wire, candidate.certificate)
            # The rebound preview is now the cached entry for this family.
            assert trial_render(sibling, modes, layout, preview_assignment) is second
    finally:
        _ACTIVE.reset(token)
    # Outside a transaction nothing is aliased and nothing is cached.
    assert family(sibling) == id(sibling)
    assert alias(sibling, snap) is sibling
