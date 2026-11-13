"""Stable wire keeps its layout when the fresh layout lands near the request envelope.

Live run 73111484 fell back to the plain layout on 5 of 16 sends. Each case here
reproduces one way the fit can miss, through the real project_stable_wire and its caller.
"""
from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest

import kernel_cache_layout_candidate as candidate
from kernel_stable_wire import (
    STATUS_NAME,
    canonicalize_stable_wire,
    is_stable_wire_projection,
    project_stable_wire,
    recorded_window_contents,
    recorded_window_layout,
)
from test_stable_wire import (
    Journal,
    Meter,
    _body,
    _final_layout_body,
    _invoke,
    _revision,
    _window,
)

FAILED_OPEN = "stable_wire_failed_open"
# Event fields may only be numbers, flags and short enum words. Never request content.
ALLOWED_FIELDS = {
    "kind", "request_id", "reason", "vital", "producer", "dispatch", "cause", "prompt_tokens",
    "token_ceiling", "bytes", "byte_ceiling", "measurement", "trim_count",
    "trim_attempted", "trim_available", "reserve_available",
}


def _frame_row(body):
    return next(row for row in body["messages"]
                if str(row.get("content", "")).startswith("[StateFrame]\n"))


def _write_frame(row, payload):
    row["content"] = "[StateFrame]\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _layout_body(optional_rows=0):
    """The layout body with optional notes in the frame, which sits behind the windows."""
    body = _final_layout_body()
    row = _frame_row(body)
    payload = json.loads(row["content"].split("\n", 1)[1])
    payload["state"]["framefit5"]["optional_notes"] = [f"{n}:" + "x" * 1000
                                                        for n in range(optional_rows)]
    _write_frame(row, payload)
    return body


def _render(monkeypatch, body, *, meter, inputs, trim=None, runtime=None):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", "16000")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda _plan, _messages, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    if trim is not None:
        inputs["_final_layout_trim"] = trim
    capture = SimpleNamespace(journal=Journal(), request_id="fit", step=1,
                              request_source_final_delivery=None)
    runtime = runtime or SimpleNamespace(_stable_wire_state=None)
    result = candidate.render_final_cache_candidate(
        capture, runtime, body, meter=meter, inputs=inputs)
    return result, capture


def _inputs(token_ceiling):
    return {"token_ceiling": token_ceiling, "byte_ceiling": 10_000_000,
            "source_plan": {}, "sources": {}, "receipt": None}


class SizeMeter:
    """One token per 100 serialized bytes, so a trim that drops rows really frees tokens."""

    def measure(self, body):
        return Meter().measure(body)


class UnderTrimmer:
    """A trim that frees only a fraction of the reserve it is asked for, like a coarse allocator."""

    def __init__(self, body, *, rows_per_reserve):
        self.body = body
        self.rows_per_reserve = rows_per_reserve
        self.calls = []

    def __call__(self, token_reserve, byte_reserve):
        self.calls.append((token_reserve, byte_reserve))
        row = _frame_row(self.body)
        payload = json.loads(row["content"].split("\n", 1)[1])
        notes = payload["state"]["framefit5"]["optional_notes"]
        drop = min(len(notes), int(token_reserve * self.rows_per_reserve))
        payload["state"]["framefit5"]["optional_notes"] = notes[:len(notes) - drop]
        _write_frame(row, payload)
        return True


def _failed_open_events(capture):
    return [e for e in capture.journal.events if e["kind"] == FAILED_OPEN]


def test_a_trim_that_lands_short_is_retried_with_the_running_reserve(monkeypatch):
    """Cause (d): the retry after a trim used to carry no callback, so a short trim failed open."""
    probe_body = _layout_body(optional_rows=6)
    probe, _ = _render(monkeypatch, probe_body, meter=SizeMeter(), inputs=_inputs(100_000))
    assert is_stable_wire_projection(probe["messages"])
    ceiling = SizeMeter().measure(probe)[0] - 12   # the fresh layout is a little over
    body = _layout_body(optional_rows=6)
    # 0.1 rows per reserved token: the first request (28 tokens) frees nothing, the next frees one row.
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(ceiling),
                              trim=trimmer)
    assert _failed_open_events(capture) == [], "stable wire fell back to the plain layout"
    assert is_stable_wire_projection(result["messages"])
    assert len(trimmer.calls) >= 2
    # The callback sizes limits from the untrimmed base, so each call asks for the running total.
    assert all(later[0] > earlier[0] for earlier, later in zip(trimmer.calls, trimmer.calls[1:]))
    assert SizeMeter().measure(result)[0] <= ceiling
    assert capture._final_cache_layout_record["stable_wire"] == "applied"


def test_replay_repeats_live_trims_with_the_same_cumulative_reserves(monkeypatch):
    live_body = _layout_body(6)
    live_trim = UnderTrimmer(live_body, rows_per_reserve=0.05)
    live, live_capture = _render(
        monkeypatch, live_body, meter=SizeMeter(), inputs=_inputs(70), trim=live_trim)
    assert live_trim.calls == [(28, 256), (46, 512)]
    assert is_stable_wire_projection(live["messages"])
    assert _failed_open_events(live_capture) == []

    recorded = recorded_window_contents(live["messages"])
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
                        lambda: live_capture._stable_wire_reserves_record)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: recorded_window_layout(live["messages"]))
    replay_body = _layout_body(6)
    replay_trim = UnderTrimmer(replay_body, rows_per_reserve=0.05)
    replay, capture = _render(
        monkeypatch, replay_body, meter=SizeMeter(), inputs=_inputs(70), trim=replay_trim)

    assert replay_trim.calls == live_trim.calls
    assert json.dumps(replay["messages"]).encode() == json.dumps(live["messages"]).encode()
    assert recorded_window_contents(replay["messages"]) == recorded
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    assert _failed_open_events(capture) == []
    assert not [e for e in capture.journal.events if e["kind"] == "stable_wire_replay_mismatch"]


@pytest.mark.parametrize("replay_meter", ["lower", "unavailable", "metadata", "fallback"])
def test_recorded_trims_ignore_replay_meter_and_intermediate_metadata(monkeypatch, replay_meter):
    """Only the final, trimmed layout is replay authority, regardless of the current meter."""
    def setup():
        body = _layout_body(6)
        trimmer = UnderTrimmer(body, rows_per_reserve=0.05)

        def trim(tokens, byte_count):
            trimmer(tokens, byte_count)
            if replay_meter == "metadata":
                window = next(row for row in body["messages"]
                              if str(row.get("content", "")).startswith("[Current source:"))
                window["content"] = window["content"].replace('"max_lines":1000',
                                                              '"max_lines":999')
            return True
        return body, trimmer, trim

    body, live_trim, trim = setup()
    live, live_capture = _render(monkeypatch, body, meter=SizeMeter(),
                                 inputs=_inputs(70), trim=trim)
    assert live_trim.calls == [(28, 256), (46, 512)]
    # Explicit numbers make this test fail on missing capture as well as wrong replay policy.
    record = {"reserves": [[28, 256], [46, 512]], "trim_count": 2, "epoch": 1}
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
                        lambda: record, raising=False)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: recorded_window_layout(live["messages"]))

    class ReplayMeter:
        def measure(self, body):
            if replay_meter == "unavailable":
                raise RuntimeError("tokenizer unavailable")
            if replay_meter == "metadata":
                return SizeMeter().measure(body)
            if replay_meter == "fallback":
                return 1_000_000, "calibrated_bytes"
            return 1, "tokenize"

    body, replay_trim, trim = setup()
    rebuilt, capture = _render(monkeypatch, body, meter=ReplayMeter(),
                                inputs=_inputs(70), trim=trim)
    assert replay_trim.calls == live_trim.calls
    assert json.dumps(rebuilt).encode() == json.dumps(live).encode()
    assert live_capture._stable_wire_reserves_record == record
    assert capture._stable_wire_reserves_record == record
    assert _failed_open_events(capture) == []
    assert not [e for e in capture.journal.events if e["kind"] == "stable_wire_replay_mismatch"]


@pytest.mark.parametrize("mismatch_after_trim", [1, 2])
def test_replay_rejects_final_window_mismatch_after_recorded_trims(monkeypatch, mismatch_after_trim):
    live_body = _layout_body(6)
    live_trim = UnderTrimmer(live_body, rows_per_reserve=0.05)
    live, live_capture = _render(
        monkeypatch, live_body, meter=SizeMeter(), inputs=_inputs(70), trim=live_trim)
    assert live_trim.calls == [(28, 256), (46, 512)]
    recorded = recorded_window_contents(live["messages"])
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
                        lambda: live_capture._stable_wire_reserves_record)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: recorded_window_layout(live["messages"]))

    body = _layout_body(6)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
    runtime = SimpleNamespace(_stable_wire_state=None)

    def trim(token_reserve, byte_reserve):
        trimmer(token_reserve, byte_reserve)
        if len(trimmer.calls) == mismatch_after_trim:
            # The change survives to the final projection, which must reject it.
            window = next(row for row in body["messages"]
                          if str(row.get("content", "")).startswith("[Current source:"))
            window["content"] = window["content"].replace("value = 1", "value = 2")
        return True

    result, capture = _render(
        monkeypatch, body, meter=SizeMeter(), inputs=_inputs(70), trim=trim, runtime=runtime)

    assert trimmer.calls == live_trim.calls
    mismatch, = [e for e in capture.journal.events
                 if e["kind"] == "stable_wire_replay_mismatch"]
    failed, = _failed_open_events(capture)
    assert failed["reason"] == "replay_layout_mismatch"
    assert mismatch["recorded_window_count"] == len(recorded)
    assert mismatch["projected_window_count"] == len(recorded)
    assert mismatch["shared_prefix_windows"] == 0
    assert not any("[Current source:" in str(value) for value in mismatch.values())
    assert not is_stable_wire_projection(result["messages"])
    assert runtime._stable_wire_state is None


@pytest.mark.parametrize("provider", [None, lambda: None])
def test_replay_applied_without_windows_explicitly_fails_open(monkeypatch, provider):
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", provider)
    body = _layout_body(6)
    before = copy.deepcopy(body)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
    result, capture = _render(
        monkeypatch, body, meter=SizeMeter(), inputs=_inputs(70), trim=trimmer)

    assert trimmer.calls == []
    assert body == before
    assert not is_stable_wire_projection(result["messages"])
    assert capture._final_cache_layout_record["stable_wire"] == "failed_open"
    assert len([e for e in capture.journal.events if e["kind"] == "stable_wire_replay_mismatch"]) == 1
    assert _failed_open_events(capture)[0]["reason"] == "replay_layout_mismatch"


def test_replay_applies_the_full_three_trim_bound_exactly(monkeypatch):
    body = _layout_body(6)
    live_trim = UnderTrimmer(body, rows_per_reserve=0.02)
    live, live_capture = _render(monkeypatch, body, meter=SizeMeter(),
                                 inputs=_inputs(70), trim=live_trim)
    assert live_trim.calls == [(28, 256), (56, 512), (74, 768)]
    assert live_capture._stable_wire_reserves_record["trim_count"] == candidate.MAX_STABLE_TRIMS
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: recorded_window_layout(live["messages"]))
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
                        lambda: live_capture._stable_wire_reserves_record)
    body = _layout_body(6)
    replay_trim = UnderTrimmer(body, rows_per_reserve=0.02)
    replay, capture = _render(monkeypatch, body, meter=SizeMeter(),
                               inputs=_inputs(70), trim=replay_trim)
    assert replay_trim.calls == live_trim.calls
    assert json.dumps(replay).encode() == json.dumps(live).encode()
    assert capture._stable_wire_reserves_record == live_capture._stable_wire_reserves_record


@pytest.mark.parametrize("refuse", [False, True])
def test_replay_cannot_skip_an_unavailable_recorded_trim(monkeypatch, refuse):
    body = _layout_body(6)
    live_trim = UnderTrimmer(body, rows_per_reserve=0.05)
    live, live_capture = _render(monkeypatch, body, meter=SizeMeter(),
                                 inputs=_inputs(70), trim=live_trim)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: recorded_window_layout(live["messages"]))
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
                        lambda: live_capture._stable_wire_reserves_record)
    body = _layout_body(6)
    before = copy.deepcopy(body)
    calls = []

    def trim(tokens, byte_count):
        calls.append((tokens, byte_count))
        body["messages"].clear()  # a refusing callback's edits must not escape
        return False

    runtime = SimpleNamespace(_stable_wire_state=None)
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(70),
                               trim=trim if refuse else None, runtime=runtime)
    assert calls == ([(28, 256)] if refuse else [])
    assert body == before
    assert not is_stable_wire_projection(result["messages"])
    assert _failed_open_events(capture)[0]["reason"] == "recorded_trim_unavailable"
    assert len([e for e in capture.journal.events if e["kind"] == "stable_wire_replay_mismatch"]) == 1
    assert runtime._stable_wire_state is None


def test_trim_retries_are_bounded_and_the_failed_open_reports_measured_fields(monkeypatch):
    body = _layout_body(optional_rows=6)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.0)   # a trim that never frees anything
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(60),
                              trim=trimmer)
    assert len(trimmer.calls) == 3
    event, = _failed_open_events(capture)
    assert event["reason"] == "fresh_epoch_exceeds_measured_envelope"
    assert event["trim_attempted"] is True and event["trim_count"] == 3
    assert event["cause"] == "tokens"
    assert event["token_ceiling"] == 60 and event["prompt_tokens"] > 60
    assert event["byte_ceiling"] == 10_000_000 and 0 < event["bytes"] < 10_000_000
    assert event["measurement"] == "tokenize"
    assert set(event) <= ALLOWED_FIELDS
    assert not is_stable_wire_projection(result["messages"])


def test_failed_open_without_a_trim_callback_says_so(monkeypatch):
    prior = _window(revision=_revision("1"), text="prior = 1\n")
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([prior]), runtime, headroom=6_000)
    huge = _window(revision=_revision("2"), text="large = 2\n" * 500)
    result, _capture, journal = _invoke(
        monkeypatch, _body([huge]), runtime, headroom=6_000,
        inputs={"token_ceiling": 50, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}, "receipt": None})
    event, = [e for e in journal.events if e["kind"] == FAILED_OPEN]
    assert event["trim_attempted"] is False and event["trim_available"] is False
    assert event["trim_count"] == 0 and event["cause"] == "tokens"
    assert set(event) <= ALLOWED_FIELDS


def test_a_trim_that_returns_false_fails_open_and_restores_the_body(monkeypatch):
    """Cause (c): the allocator cannot trim any further."""
    body = _layout_body(optional_rows=6)
    before = copy.deepcopy(body)
    calls = []

    def refuse(token_reserve, byte_reserve):
        calls.append((token_reserve, byte_reserve))
        _write_frame(_frame_row(body), {"state": {"framefit5": {}}})
        return False

    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(60),
                              trim=refuse)
    event, = _failed_open_events(capture)
    assert len(calls) == 1 and event["trim_attempted"] is True and event["trim_count"] == 1
    assert body["messages"] == before["messages"], "a refused trim must leave the body untouched"
    assert not is_stable_wire_projection(result["messages"])


def test_an_unmeasurable_projection_reports_it_without_trimming(monkeypatch):
    """Cause (b): no measurement means no reserve and no trim."""
    class BlindMeter:
        def measure(self, body):
            if is_stable_wire_projection(body.get("messages", [])):
                raise RuntimeError("tokenizer unavailable")
            return Meter().measure(body)

    called = []
    body = _layout_body(optional_rows=1)
    result, capture = _render(monkeypatch, body, meter=BlindMeter(), inputs=_inputs(10_000),
                              trim=lambda *args: called.append(args) or True)
    event, = _failed_open_events(capture)
    assert called == []
    assert event["cause"] == "measurement_unavailable"
    assert event["trim_attempted"] is False and event["reserve_available"] is False
    assert event["trim_available"] is True
    assert set(event) <= ALLOWED_FIELDS
    assert not is_stable_wire_projection(result["messages"])


def test_a_measurement_method_change_is_named_in_the_failed_open(monkeypatch):
    class DriftMeter:
        def measure(self, body):
            return Meter().measure(body)[0], "estimate"

    prior = _window(revision=_revision("1"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([prior]), runtime, headroom=6_000)
    _result, _capture, journal = _invoke(
        monkeypatch, _body([_window(revision=_revision("2"), text="v = 2\n")]), runtime,
        headroom=6_000, meter=DriftMeter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000, "measurement_identity": "tokenize",
                "source_plan": {}, "sources": {}, "receipt": None})
    event, = [e for e in journal.events if e["kind"] == FAILED_OPEN]
    assert event["cause"] == "measurement_method_mismatch"
    assert event["measurement"] == "estimate"


def _status(messages):
    return next(row["content"] for row in messages if row.get("name") == STATUS_NAME)


def test_status_names_current_windows_without_repeating_their_revision(monkeypatch):
    """Overhead cause: the status repeated each window's 64 character revision and ranges."""
    windows = [_window(path=f"pkg/mod{n}.py", revision=_revision(format(n, "x")), end=40)
               for n in range(1, 13)]
    result, _capture, _journal = _invoke(
        monkeypatch, _body(windows), SimpleNamespace(_stable_wire_state=None))
    status = _status(result["messages"])
    for n in range(1, 13):
        assert f"pkg/mod{n}.py: current" in status
    assert not re.search(r"[0-9a-f]{64}", status), "current windows carry their own revision"
    assert "lines 1 to" not in status
    # Contract kept: every window message still states its own revision on the wire.
    wire_windows = [r["content"] for r in result["messages"]
                    if str(r.get("content", "")).startswith("[Current source:")]
    assert len(wire_windows) == 12 and all("revision" in w for w in wire_windows)
    # Overhead over the plain layout stays small and grows by a path per window, not a hash.
    plain, _index = canonicalize_stable_wire(result["messages"])
    overhead = (len(json.dumps(result)) - len(json.dumps({**result, "messages": plain})))
    assert overhead < 900, overhead


def test_retained_windows_keep_their_revision_and_ranges_in_the_status(monkeypatch):
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([_window(revision=_revision("1"), end=7)]), runtime)
    result, _capture, _journal = _invoke(
        monkeypatch, _body([_window(path="pkg/b.py", revision=_revision("2"))]), runtime)
    status = _status(result["messages"])
    retained = status.split("Not current, do not edit from these:\n", 1)[1]
    assert "pkg/a.py, revision sha256:" + "1" * 64 + ", lines 1 to 7" in retained
    assert "pkg/b.py: current" in status


def test_projection_arguments_unchanged_for_callers_without_a_trim_callback(monkeypatch):
    """project_stable_wire keeps its signature: trim_callback stays optional."""
    runtime = SimpleNamespace(_stable_wire_state=None)
    result, _capture, _journal = _invoke(monkeypatch, _body([_window()]), runtime)
    assert is_stable_wire_projection(result["messages"])
    assert project_stable_wire.__kwdefaults__["trim_callback"] is None


@pytest.mark.parametrize("cause", ["measurement_method_mismatch", "ceiling_unavailable"])
def test_a_non_size_cause_fails_open_without_asking_for_a_trim(monkeypatch, cause):
    """Trimming cannot cure a method mismatch or a missing ceiling, so no trim is requested."""
    import kernel_stable_wire as wire
    real = wire._fit_report

    def report(*args, **kwargs):
        out = dict(real(*args, **kwargs))
        if not out["fits"]:
            out["cause"] = cause
        return out

    monkeypatch.setattr(wire, "_fit_report", report)
    called = []
    body = _layout_body(optional_rows=6)
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(60),
                              trim=lambda *args: called.append(args) or True)
    event, = _failed_open_events(capture)
    assert called == []
    assert event["reason"] == "fresh_epoch_exceeds_measured_envelope"
    assert event["cause"] == cause and event["trim_count"] == 0
    assert event["trim_attempted"] is False
    assert not is_stable_wire_projection(result["messages"])
