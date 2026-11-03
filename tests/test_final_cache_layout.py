"""Final cache layout keeps current authority on the submitted tail."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from kernel_cache_layout_candidate import render_final_cache_candidate
from kernel_cache_layout_view import CacheLayoutError, semantic_index_map, semantic_messages
from kernel_frame_allocation_capture import _obligations_and_locations
from kernel_stateframe_page_log import _partition_authority, _restore_authority


class Meter:
    def measure(self, _body):
        return 100, "tokenize"


def _body(exchange=False):
    payload = {"state": {"framefit5": {
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1",
                           "continuation_limits": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2",
                           "evidence_pin": "[EVIDENCEPIN]current[/EVIDENCEPIN]"},
        "next_site": {"path": "pkg/a.py"},
        "last_verdict": {"line": "LAST FAIL"},
        "action_ledger": {"reads": [], "edits": []},
    }}}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix it"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))},
        {"role": "user", "content": "[Current source: pkg/a.py]\nrevision=sha256:123\nsource bytes"},
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
    ]
    if exchange:
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                                  "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ])
    messages.append({"role": "user", "content": "[WIRE VOLATILE TAIL]"})
    return {"messages": messages, "tools": []}


def _project(monkeypatch, exchange=False):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    capture = SimpleNamespace(request_source_final_delivery=None)
    original = _body(exchange)
    projected = render_final_cache_candidate(
        capture, None, original, meter=Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    return original, projected


@pytest.mark.parametrize("exchange", [False, True])
def test_physical_layout_round_trip_and_authority(monkeypatch, exchange):
    original, projected = _project(monkeypatch, exchange)
    rows = projected["messages"]
    assert rows[2]["content"] == original["messages"][3]["content"]
    assert rows[3]["content"].startswith("[StateFrame]\n")
    assert "continuation_limits" not in rows[3]["content"]
    assert "next_site" not in rows[3]["content"]
    assert sum(row.get("content", "").count("[CONTINUATION LIMITS |") for row in rows) == 1
    assert "[CONTINUATION LIMITS |" in rows[-1]["content"]
    assert semantic_index_map(rows)[2] == 3
    view = semantic_messages(rows)
    assert view[2]["content"] == original["messages"][2]["content"]
    assert view[3]["content"] == original["messages"][3]["content"]
    assert [row for row in rows if row.get("role") in {"assistant", "tool"}] == [
        row for row in original["messages"] if row.get("role") in {"assistant", "tool"}]


def test_planted_early_authority_violation_fails(monkeypatch):
    _original, projected = _project(monkeypatch)
    broken = copy.deepcopy(projected["messages"])
    broken[2]["content"] += "\n[CONTINUATION LIMITS | stale]"
    with pytest.raises(CacheLayoutError, match="physically final"):
        semantic_messages(broken)


def test_current_host_instruction_moves_without_changing_its_bytes(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    body = _body()
    instruction = "[Read lockout] current refusal\nRecover with the read route."
    body["messages"].insert(4, {"role": "user", "content": instruction})
    projected = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body, meter=Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert instruction in projected["messages"][-1]["content"]
    assert sum(instruction in str(row.get("content")) for row in projected["messages"]) == 1


def test_partition_restores_current_authority_exactly():
    payload = json.loads(_body()["messages"][2]["content"].split("\n", 1)[1])
    legacy_stable, _legacy_fields = _partition_authority(payload)
    assert legacy_stable["state"]["framefit5"]["next_site"] == {"path": "pkg/a.py"}
    stable, fields = _partition_authority(payload, final=True)
    assert _restore_authority(stable, fields) == payload
    assert set(stable["state"]["framefit5"]["owner_protocol"]) == {"schema"}
    assert "next_site" not in stable["state"]["framefit5"]


def test_off_preserves_input_bytes(monkeypatch):
    # The layout is on by default; this checks the explicit off switch.
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    body = _body()
    before = json.dumps(body).encode()
    assert render_final_cache_candidate(None, None, body, meter=None, inputs=None) is body
    assert json.dumps(body).encode() == before


def test_invalid_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "unexpected")
    with pytest.raises(ValueError, match="invalid final cache layout mode"):
        render_final_cache_candidate(None, None, _body(), meter=None, inputs=None)


class _Journal:
    run_id = "run"

    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


class _OverMeter:
    def __init__(self, tokens, method="tokenize"):
        self.tokens, self.method = tokens, method

    def measure(self, _body):
        return self.tokens, self.method


@pytest.mark.parametrize("meter,ceilings", [
    (_OverMeter(1001), (1000, 100000)),       # projected layout adds tokens past the ceiling
    (_OverMeter(100), (1000, 10)),            # or bytes past the byte ceiling
    (_OverMeter(100, "estimate"), (1000, 100000)),  # or cannot be measured exactly
])
def test_envelope_overflow_falls_open_to_the_fitted_body(monkeypatch, meter, ceilings):
    # JOB 4 on_r1/t2: an overflow at request 10 raised and ended the run.
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: pytest.fail("source proof must not run on fallback"))
    journal = _Journal()
    capture = SimpleNamespace(request_source_final_delivery=None, journal=journal, request_id="r10", step=10)
    body = _body()
    before = json.dumps(body).encode()
    inputs = {"token_ceiling": ceilings[0], "byte_ceiling": ceilings[1],
              "source_plan": {}, "sources": {}, "receipt": None}
    assert render_final_cache_candidate(capture, None, body, meter=meter, inputs=inputs) is body
    assert json.dumps(body).encode() == before
    assert capture.request_source_final_delivery is None and inputs["receipt"] is None
    kinds = [kind for kind, _fields in journal.events]
    assert "final_cache_layout_failed_open" in kinds


def test_final_layout_fail_open_counts_in_run_vitals_and_protocol_spec():
    from kernel_protocol_spec import _event_rows
    from kernel_run_vitals import RunVitalsAccumulator

    vitals = RunVitalsAccumulator()
    vitals.observe({"kind": "final_cache_layout_failed_open", "reason": "exceeds_measured_envelope"})
    vitals.observe({"kind": "stateframe_page_log_failed_open"})
    assert vitals.snapshot()["cache_layout_fail_open"] == 2
    rows = {row.get("kind"): row for row in _event_rows()}
    assert "final_cache_layout_failed_open" in rows


def test_source_proof_failure_cannot_fall_back(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "INCOMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: False)
    with pytest.raises(ValueError, match="source proof failed"):
        render_final_cache_candidate(
            SimpleNamespace(request_source_final_delivery=None), None, _body(), meter=Meter(),
            inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                    "source_plan": {}, "sources": {}, "receipt": None})


def test_physical_source_proof_maps_into_canonical_allocation(monkeypatch):
    _original, projected = _project(monkeypatch)
    mapping = semantic_index_map(projected["messages"])
    source = "source bytes\n"
    plan = {"units": [{"path": "pkg/a.py", "unit_handle": "unit:a",
                       "start_line": 1, "end_line": 1}],
            "revisions": {"pkg/a.py": "sha256:123"}}
    receipt = {"deferred_paths": [], "delivery_proofs": {"pkg/a.py": [
        {"key": {"byte_start": 0, "byte_end": len(source.encode())},
         "wire_contains_exact": True, "final_message_location": "$.messages[2].content"}]}}
    obligations, locations = _obligations_and_locations(
        plan, receipt, {"pkg/a.py": source}, {3: "prepared_source"}, mapping)
    assert obligations[0]["disposition"] == "RETAINED"
    assert locations == {3: [{"handle": "unit:a", "kind": "prepared_source"}]}


def test_shell_setting_reaches_the_engine_spec(monkeypatch):
    # A shell-set switch only reaches the engine through the runner's opt-in allowlist; without this entry the host
    # guard ran with the layout silently off (StateFrame still message #2 on the sealed wire).
    from stateframe import runner
    name = "STATEFRAME_FINAL_CACHE_LAYOUT"
    assert runner.OPT_IN_SWITCHES[name] == ("on", "off")
    assert name in runner.engine_switches()
    monkeypatch.setenv(name, "on")
    assert runner.resolve_opt_in_switches()[name] == "on"
    monkeypatch.delenv(name)
    assert name not in runner.resolve_opt_in_switches()
