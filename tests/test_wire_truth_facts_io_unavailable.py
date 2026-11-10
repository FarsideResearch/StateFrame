"""Deferred I/O failures produce typed wire-fact gaps and one run event."""
from __future__ import annotations

from types import SimpleNamespace

import kernel_wire_truth_facts as facts
import pytest


def _runtime(binding):
    events = []
    journal = SimpleNamespace(emit=lambda kind, **data: events.append((kind, data)))
    return SimpleNamespace(binding=binding, journal=journal), events


def _assert_gap(result, events, code):
    assert isinstance(result, facts.UnavailableFactEvidence)
    assert result.reason_code == code
    assert events == [("wire_truth_fact_evidence_failure", {
        "producer": "kernel_wire_truth_facts", "dispatch": False,
        "reason_code": code,
    })]


def test_planted_live_capture_binding_io_emits_event():
    def broken():
        raise OSError("planted live binding failure")

    runtime, events = _runtime(broken)
    capture = SimpleNamespace(request_id="request-1")
    _assert_gap(facts.capture_context(capture, runtime, {"tools": [{"name": "done"}]}),
                events, "BINDING_UNAVAILABLE")
    assert not hasattr(capture, "wire_truth_fact_store")


def test_public_check_io_returns_gap(monkeypatch):
    import kernel_green_lock

    runtime, events = _runtime(lambda: {"product_revision": "r1"})

    def broken(_runtime):
        raise OSError("public selector unavailable")

    monkeypatch.setattr(kernel_green_lock, "_declared_selectors", broken)
    _assert_gap(facts._current_public_check(runtime, {"product_revision": "r1"}),
                events, "PUBLIC_CHECK_UNAVAILABLE")


def test_product_revision_io_returns_gap(monkeypatch):
    import kernel_done_readiness_authority
    import kernel_green_lock

    runtime, events = _runtime(lambda: {"product_revision": "r1"})
    capture = SimpleNamespace(request_id="request-1")
    monkeypatch.setattr(kernel_green_lock, "_declared_selectors", lambda _runtime: ())
    monkeypatch.setattr(kernel_done_readiness_authority, "cached_decision", lambda _runtime: None)

    def broken(_runtime):
        raise OSError("revision unavailable")

    monkeypatch.setattr(kernel_green_lock, "product_revision_change_from_parent", broken)
    _assert_gap(facts.capture_context(capture, runtime, {"tools": [{"name": "done"}]}),
                events, "PRODUCT_REVISION_UNAVAILABLE")


def test_unexpected_binding_failure_propagates():
    runtime, events = _runtime(lambda: (_ for _ in ()).throw(LookupError("unexpected")))
    with pytest.raises(LookupError, match="unexpected"):
        facts.capture_context(SimpleNamespace(request_id="request-1"), runtime,
                              {"tools": [{"name": "done"}]})
    assert events == []
