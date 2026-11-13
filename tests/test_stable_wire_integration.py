"""Ownership provenance must also govern replay epochs and reserve publication."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_cache_layout_candidate as candidate
import kernel_stable_wire as stable
from kernel_final_wire_allocation import _stable_wire_record_for_publication
from test_newest_user_turn_survives_wire_projection import ADVISORY, _final_cache_shaped_body
from test_stable_wire import Journal, Meter, _invoke, _window


@pytest.mark.parametrize("where", ["host", "captured", "native"])
@pytest.mark.parametrize("switch", ["on", "off"])
def test_replay_status_name_collisions_preserve_owned_layout(monkeypatch, where, switch):
    body = _final_cache_shaped_body()
    opaque = {"role": "user", "name": stable.STATUS_NAME,
              "content": "[Source windows status]\nEpoch 999\nquoted status"}
    captured = ()
    if where == "host":
        body["messages"].insert(2, opaque)
    elif where == "captured":
        at = next(i for i, row in enumerate(body["messages"]) if row.get("content") == ADVISORY)
        body["messages"][at] = opaque
        captured = (copy.deepcopy(opaque),)
    else:
        body["messages"][-1] = opaque
    capture = SimpleNamespace(journal=Journal(), request_id="status-collision",
                              _framefit5_newest_user_suffix=captured)
    # Reuse the deterministic source-verification fixtures, then pass actual capture provenance.
    _invoke(monkeypatch, _final_cache_shaped_body(), SimpleNamespace(_stable_wire_state=None))
    inputs = {"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}}
    live = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs=copy.deepcopy(inputs))
    recorded = json.loads(json.dumps(live))["messages"]
    assert stable.recorded_window_layout(recorded) == ([_window()["content"]], 1)
    record = {"reserves": [], "trim_count": 0, "epoch": 1}
    capture._stable_wire_reserves_record = record
    assert _stable_wire_record_for_publication(capture, live) == record
    assert opaque in stable.canonicalize_stable_wire(recorded)[0]

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", switch)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: stable.recorded_window_layout(recorded))
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER", lambda: record)

    class UnavailableMeter:
        def measure(self, _body):
            raise RuntimeError("tokenizer unavailable")

    replayed = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=UnavailableMeter(),
        inputs=copy.deepcopy(inputs))
    assert json.dumps(replayed).encode() == json.dumps(live).encode()
    assert not [event for event in capture.journal.events
                if event["kind"] in {"stable_wire_replay_mismatch", "stable_wire_failed_open"}]
