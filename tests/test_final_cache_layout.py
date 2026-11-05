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


def test_overlay_references_unique_authority_substrings_in_moved_rows(monkeypatch):
    from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    body = _body()
    evidence = "[EVIDENCEPIN]current[/EVIDENCEPIN]"
    body["messages"].insert(4, {"role": "user",
                                "content": "[COMPLETION OWNER]\nCurrent evidence: " + evidence})
    projected = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body, meter=Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    encoded = projected["messages"][-1]["content"].split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[1]
    overlay, _consumed = json.JSONDecoder().raw_decode(encoded)
    field = next(row for row in overlay["fields"]
                 if row.get("path") == ["state", "framefit5", "owner_protocol", "evidence_pin"])
    assert set(field) == {"path", "message", "span"}
    moved = next(row for row in overlay["order"]["moved"] if row["index"] == field["message"])
    final_prefix = projected["messages"][-1]["content"].split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[0]
    moved_content = final_prefix[moved["start"]:moved["end"]]
    start, end = field["span"]
    assert moved_content[start:end] == evidence
    assert semantic_messages(projected["messages"]) == body["messages"]


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
    failure = next(fields for kind, fields in journal.events if kind == "final_cache_layout_failed_open")
    assert failure["reason"] == "exceeds_measured_envelope"


def test_estimate_meter_applies_layout_when_it_fits(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    body = _body()
    projected = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body,
        meter=_OverMeter(100, "estimate"),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert projected is not body
    assert projected["messages"][-1]["name"] == "stateframe_final_cache_1"


@pytest.mark.parametrize("meter,ceilings", [
    (_OverMeter(1001, "estimate"), (1000, 100000)),
    (_OverMeter(100, "estimate"), (1000, 10)),
])
def test_estimate_meter_over_either_ceiling_fails_open_with_reason(monkeypatch, meter, ceilings):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: pytest.fail("source proof must not run on fallback"))
    journal = _Journal()
    body = _body()
    capture = SimpleNamespace(request_source_final_delivery=None, journal=journal,
                              request_id="estimate-over", step=1)
    result = render_final_cache_candidate(
        capture, None, body, meter=meter,
        inputs={"token_ceiling": ceilings[0], "byte_ceiling": ceilings[1],
                "source_plan": {}, "sources": {}, "receipt": None})
    assert result is body
    failure = next(fields for kind, fields in journal.events if kind == "final_cache_layout_failed_open")
    assert failure["reason"] == "exceeds_measured_envelope"


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


def _coverage_body(text):
    body = _body()
    body["messages"].insert(2, {"role": "user", "content": text})
    return body


def _project_with_source_coverage(monkeypatch, body):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    return render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body, meter=Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})


def _assert_equal_stable_prefix(first, second):
    def prefix(body):
        messages = body["messages"]
        frame_index = next(index for index, row in enumerate(messages)
                           if row.get("content", "").startswith("[StateFrame]\n"))
        return messages[:frame_index]

    assert prefix(first) == prefix(second)


def test_source_coverage_moves_after_stable_prefix_and_rebuilds_input(monkeypatch):
    source_coverage = "[Source coverage]\nCurrent source windows are complete."
    original = _coverage_body(source_coverage)
    projected = _project_with_source_coverage(monkeypatch, original)
    rows = projected["messages"]
    frame_index = next(index for index, row in enumerate(rows)
                       if row.get("content", "").startswith("[StateFrame]\n"))
    assert all(not str(row.get("content", "")).startswith("[Source coverage]")
               for row in rows[:frame_index])
    assert semantic_messages(rows) == original["messages"]


def test_source_coverage_changes_do_not_change_stable_message_prefix(monkeypatch):
    first = _project_with_source_coverage(
        monkeypatch, _coverage_body("[Source coverage]\nCoverage pass 1."))
    second = _project_with_source_coverage(
        monkeypatch, _coverage_body("[Source coverage]\nCoverage pass 2."))
    _assert_equal_stable_prefix(first, second)


def test_planted_source_coverage_left_in_prefix_is_caught(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    monkeypatch.setattr(candidate, "SOURCE_COVERAGE_PREFIX", "[unmatched coverage]", raising=False)
    first = _project_with_source_coverage(
        monkeypatch, _coverage_body("[Source coverage]\nCoverage pass 1."))
    second = _project_with_source_coverage(
        monkeypatch, _coverage_body("[Source coverage]\nCoverage pass 2."))
    with pytest.raises(AssertionError):
        _assert_equal_stable_prefix(first, second)


class _ProjectionMeter:
    """Deterministic token estimate with a planted projection overhead."""

    def measure(self, body):
        from kernel_context import token_estimate
        from kernel_cache_layout_view import FINAL_LAYOUT_NAME

        count = token_estimate(body)
        if body["messages"][-1].get("name") == FINAL_LAYOUT_NAME:
            count += 100
        return count, "estimate"


def _budget_body():
    body = _body()
    body["messages"].insert(3, {"role": "user", "content": "[Optional source]\n" + "x" * 1200})
    return body


def _trim_optional_message(body, token_reserve, byte_reserve):
    assert token_reserve > 0 or byte_reserve > 0
    body["messages"] = [row for row in body["messages"]
                        if not str(row.get("content", "")).startswith("[Optional source]")]


def test_exact_plain_budget_reserves_layout_overhead_and_trims_once(monkeypatch):
    from kernel_context import token_estimate

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    body = _budget_body()
    ceiling = token_estimate(body)
    byte_ceiling = len(json.dumps(body).encode())
    inputs = {"token_ceiling": ceiling, "byte_ceiling": byte_ceiling,
              "source_plan": {}, "sources": {}, "receipt": None,
              "_final_layout_trim": lambda tokens, bytes_: _trim_optional_message(body, tokens, bytes_)}
    projected = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body,
        meter=_ProjectionMeter(), inputs=inputs)
    assert projected is not body
    assert _ProjectionMeter().measure(projected)[0] <= ceiling
    assert len(json.dumps(projected).encode()) <= byte_ceiling
    assert not any(str(row.get("content", "")).startswith("[Optional source]")
                   for row in body["messages"])


def test_planted_zero_projection_reserve_fails_open(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: pytest.fail("source proof must not run on fallback"))
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    monkeypatch.setattr(candidate, "_final_layout_reserve", lambda *_args, **_kwargs: (0, 0), raising=False)
    body = _budget_body()
    from kernel_context import token_estimate

    ceiling = token_estimate(body)
    byte_ceiling = len(json.dumps(body).encode())
    journal = _Journal()
    capture = SimpleNamespace(request_source_final_delivery=None, journal=journal,
                              request_id="zero-reserve", step=1)
    inputs = {"token_ceiling": ceiling, "byte_ceiling": byte_ceiling,
              "source_plan": {}, "sources": {}, "receipt": None,
              "_final_layout_trim": lambda tokens, bytes_: None}
    assert render_final_cache_candidate(capture, None, body,
                                        meter=_ProjectionMeter(), inputs=inputs) is body
    failure = next(fields for kind, fields in journal.events if kind == "final_cache_layout_failed_open")
    assert failure["reason"] == "exceeds_measured_envelope"


def test_unfittable_projection_reserve_fails_open(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: pytest.fail("source proof must not run on fallback"))
    body = _body()
    journal = _Journal()
    capture = SimpleNamespace(request_source_final_delivery=None, journal=journal,
                              request_id="unfittable-layout", step=1)
    projected_body = copy.deepcopy(body)
    from kernel_context import token_estimate

    ceiling = token_estimate(body)
    result = render_final_cache_candidate(
        capture, None, body, meter=_ProjectionMeter(),
        inputs={"token_ceiling": ceiling, "byte_ceiling": len(json.dumps(body).encode()),
                "source_plan": {}, "sources": {}, "receipt": None,
                "_final_layout_trim": lambda _tokens, _bytes: (
                    body["messages"].append({"role": "user", "content": "partial trim"}) or False
                )})
    assert result is body
    assert json.dumps(body).encode() == json.dumps(projected_body).encode()
    failure = next(fields for kind, fields in journal.events if kind == "final_cache_layout_failed_open")
    assert failure["reason"] == "exceeds_measured_envelope"


def _recorded_request_replayer(tmp_path, wire):
    import hashlib

    from stateframe.replay_tape import Replayer

    request = tmp_path / "recorded.request.json"
    response = tmp_path / "recorded.response.json"
    response_bytes = b"{}"
    request.write_bytes(wire)
    response.write_bytes(response_bytes)
    return Replayer([{"send": 1, "request": str(request), "response": str(response),
                      "response_sha256": hashlib.sha256(response_bytes).hexdigest()}])


def test_recorded_layout_decision_overrides_new_meter_and_ceiling(monkeypatch, tmp_path):
    import kernel_cache_layout_candidate as candidate

    original, expected = _project(monkeypatch)
    recorded_wire = json.dumps(expected).encode()
    replay = _recorded_request_replayer(tmp_path, recorded_wire)
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "layout", raising=False)
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    rebuilt = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, original,
        meter=_OverMeter(5000, "estimate"),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    rebuilt_wire = json.dumps(rebuilt).encode()
    assert rebuilt_wire == recorded_wire
    assert replay.answer(rebuilt_wire) == b"{}"


def test_recorded_plain_decision_overrides_a_fitting_current_meter(monkeypatch, tmp_path):
    import kernel_cache_layout_candidate as candidate

    body = _body()
    recorded_wire = json.dumps(body).encode()
    replay = _recorded_request_replayer(tmp_path, recorded_wire)
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "plain", raising=False)
    rebuilt = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, body,
        meter=Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert rebuilt is body
    assert replay.answer(json.dumps(rebuilt).encode()) == b"{}"


def test_planted_legacy_method_gate_is_detected_by_replay(monkeypatch, tmp_path):
    import kernel_cache_layout_candidate as candidate

    original, expected = _project(monkeypatch)
    recorded_wire = json.dumps(expected).encode()
    replay = _recorded_request_replayer(tmp_path, recorded_wire)
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "layout", raising=False)
    real_projection = candidate._render_final_cache_layout

    def planted_old_gate(capture, runtime, body, **kwargs):
        meter = kwargs["meter"]
        projected = real_projection(capture, runtime, body, **kwargs)
        _tokens, method = meter.measure(projected)
        return body if method != "tokenize" else projected

    monkeypatch.setattr(candidate, "_render_final_cache_layout", planted_old_gate)
    rebuilt = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, original,
        meter=_OverMeter(100, "estimate"),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert replay.answer(json.dumps(rebuilt).encode()) is None
    assert replay.diverged is not None


def test_no_replay_decision_hook_preserves_request_bytes(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    original, baseline = _project(monkeypatch)
    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: None, raising=False)
    with_hook = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, original,
        meter=Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert json.dumps(with_hook).encode() == json.dumps(baseline).encode()


@pytest.mark.parametrize("recorded_layout", [False, True])
def test_child_installs_layout_decision_from_recorded_wire(tmp_path, monkeypatch, recorded_layout):
    import kernel_bounded_http
    import kernel_cache_layout_candidate as candidate
    from stateframe import _child

    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    body = _body()
    if recorded_layout:
        monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
        monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                            lambda *_args, **_kwargs: {"status": "COMPLETE"})
        monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
        body = render_final_cache_candidate(
            SimpleNamespace(request_source_final_delivery=None), None, body, meter=Meter(),
            inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                    "source_plan": {}, "sources": {}, "receipt": None})
    wire = json.dumps(body).encode()
    replay = _recorded_request_replayer(tmp_path, wire)
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", kernel_bounded_http.request_bytes)
    run = tmp_path / "child-run"
    run.mkdir()
    _child.install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions", replay)
    assert candidate.replay_final_layout_decision() == ("layout" if recorded_layout else "plain")


def test_pending_replay_without_request_body_has_no_layout_decision(tmp_path, monkeypatch):
    import kernel_bounded_http
    import kernel_cache_layout_candidate as candidate
    from stateframe import _child
    from stateframe.replay_tape import Replayer

    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", kernel_bounded_http.request_bytes)
    replay = Replayer([], pending={"send": 1})
    run = tmp_path / "child-run"
    run.mkdir()
    _child.install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions", replay)
    assert candidate.replay_final_layout_decision() is None


def test_physical_send_records_layout_disposition_from_capture(monkeypatch, tmp_path):
    import kernel_bounded_http
    import kernel_measurement
    from stateframe import _child

    endpoint = "http://127.0.0.1:1/v1/chat/completions"
    response_bytes = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
    submitted_wires = []

    def request_bytes(submitted, *_args, **_kwargs):
        submitted_wires.append(submitted)
        return response_bytes

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", request_bytes)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    _child.install_wire_recorder(tmp_path, endpoint)

    body = {
        "model": "scripted",
        "messages": [{"role": "user", "content": "request"}],
        "tools": [],
    }
    wire = json.dumps(body).encode()
    journal = kernel_measurement.AuditJournal(tmp_path, "layout-record", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    capture._final_cache_layout_record = {"final_layout": "applied"}
    assert kernel_measurement.ACTIVE_REQUEST.get() is None

    response, _submission, _started = capture._send_transport_with_retries(
        body, wire, endpoint, "", admission=None, socket_timeout=5,
        deadline=kernel_measurement.time.monotonic() + 10, wall_timeout=5,
        backend_identity={}, request_extensions={})

    assert response["choices"]
    assert submitted_wires == [wire]
    record = json.loads((tmp_path / "sends.jsonl").read_text())
    assert record["final_layout"] == "applied"
    assert kernel_measurement.ACTIVE_REQUEST.get() is None
