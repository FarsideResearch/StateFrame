"""A replayed send reproduces the stable-wire decision of the run that recorded it.

A run with stable wire off was killed after a file move and resumed. Replay read every
plain recorded request as a stable-wire fallback ("plain"), which put the replayed sends
on the stable-wire path: final source ordering was skipped, the window order was never
saved, and the first live request after replay laid out its source windows differently
from the request that was in flight, so the resume was reported unfaithful.
"""
from __future__ import annotations

import json

import pytest

from stateframe._child import recorded_stable_wire_decision


def _request(*messages):
    return json.dumps({"model": "m", "messages": list(messages)}).encode()


PLAIN = _request({"role": "system", "content": "system"},
                 {"role": "user", "content": "Task: x"},
                 {"role": "user", "content": "[StateFrame]\n{}"})


@pytest.mark.parametrize("value", [None, "off"])
def test_plain_request_from_a_run_without_stable_wire_has_no_decision(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("STATEFRAME_STABLE_WIRE", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_STABLE_WIRE", value)
    assert recorded_stable_wire_decision(PLAIN) is None


def test_plain_request_is_a_fallback_only_when_stable_wire_is_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    assert recorded_stable_wire_decision(PLAIN) == "plain"


def test_a_recorded_projection_is_applied_whatever_the_switch(monkeypatch):
    import stateframe._child as child
    monkeypatch.setattr("kernel_stable_wire.is_stable_wire_projection", lambda messages, recorded=False: recorded)
    for value in ("on", "off"):
        monkeypatch.setenv("STATEFRAME_STABLE_WIRE", value)
        assert child.recorded_stable_wire_decision(PLAIN) == "applied"


@pytest.mark.parametrize("recorded", [None, b"not json", b"[]", b'{"messages": "x"}'])
def test_unreadable_recordings_have_no_decision(monkeypatch, recorded):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    assert recorded_stable_wire_decision(recorded) is None


def test_no_decision_keeps_final_source_ordering_on_the_replayed_send(monkeypatch):
    # The render path reads "plain" as stable wire enabled, which turns ordering off.
    import kernel_cache_layout_candidate as layout
    monkeypatch.delenv("STATEFRAME_STABLE_WIRE", raising=False)
    layout.install_replay_stable_wire_decision_provider(
        lambda: recorded_stable_wire_decision(PLAIN))
    try:
        assert layout.replay_stable_wire_decision() is None
        assert layout._final_source_order_enabled_for_fitting() is True
    finally:
        layout.install_replay_stable_wire_decision_provider(None)
