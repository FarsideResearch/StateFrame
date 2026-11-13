"""Qualification must preserve recorded stable layouts despite a larger replay meter."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import kernel_cache_layout_candidate as layout
import kernel_frame_allocation_capture as allocation
from kernel_final_wire_allocation import _recover_post_fit_growth
from kernel_stable_wire import is_stable_wire_projection, recorded_window_layout
from test_stable_wire_trim_selection import _install_replay, allocator as allocator
from tools.pinned_tree import pinned_tree


def _snapshot(prepared):
    body, capture, _runtime, _meter = prepared
    return (copy.deepcopy(body), capture._frame_candidate_counter,
            {name: copy.deepcopy(value) for name, value in capture._frame_candidate_inputs.items()
             if name != "_final_layout_trim"},
            copy.deepcopy(capture.request_source_final_delivery))


def _recover(prepared, projected, snapshot):
    _body, capture, runtime, meter = prepared
    return _recover_post_fit_growth(
        capture, runtime, projected, *snapshot, meter=meter, endpoint=None, key="")


@pytest.mark.parametrize("legacy", [False, True])
def test_zero_trim_recorded_layout_survives_qualification(monkeypatch, tmp_path, allocator, legacy):
    prepare, render = allocator
    live_prepared = prepare(replay=True)  # A low live meter admits a zero-trim stable layout.
    live_snapshot = _snapshot(live_prepared)
    live = render(live_prepared)
    live, changed = _recover(live_prepared, live, live_snapshot)
    assert not changed
    wire = json.dumps(live).encode()
    record = live_prepared[1]._stable_wire_reserves_record
    assert record == {"reserves": [], "trim_count": 0, "epoch": 1,
                      "initial_selection": live_prepared[1]._stable_wire_initial_selection}
    # Old tapes have no reserve sidecar, but their recorded windows/epoch still
    # supply layout authority. A tape with no layout at all is tested separately.
    _install_replay(monkeypatch, live, None if legacy else record)
    replay = prepare()
    snapshot = _snapshot(replay)
    assert json.dumps(replay[0]).encode() == json.dumps(live_snapshot[0]).encode()
    rebuilt = render(replay)
    assert json.dumps(rebuilt).encode() == wire
    assert replay[3].measure(replay[0]) == (33320, "tokenize")
    assert replay[3].measure(rebuilt) == (33398, "tokenize")

    def forbidden_measurement(*args, **kwargs):
        pytest.fail("recorded qualification consulted the replay meter")

    monkeypatch.setattr(replay[3], "measure", forbidden_measurement)
    qualified, changed = _recover(replay, rebuilt, snapshot)
    assert json.dumps(qualified).encode() == wire
    assert not changed

    # Qualification calls recovery again after output normalization. The final
    # budget proof must also retain the admitted counter on these exact bytes.
    repeated, changed = _recover(replay, qualified, snapshot)
    assert not changed and json.dumps(repeated).encode() == wire
    assert allocation.final_request_tokens(repeated, replay[3], replay[1]) == (
        33320 if legacy else 100, "tokenize")
    assert not [event for event in replay[1].journal.events
                if event["kind"] == "stable_wire_replay_mismatch"]
    # Use the real pending-send guard that raised ReplayUnfaithful in the report.
    import kernel_bounded_http
    from kernel_measurement import ACTIVE_REQUEST
    from stateframe import _child
    from stateframe.replay_tape import Replayer, pending_request
    from test_stable_wire_record import ENDPOINT, RESPONSE

    wire_dir = tmp_path / "wire"
    wire_dir.mkdir()
    (wire_dir / "0001.request.json").write_bytes(wire)
    if not legacy:
        (wire_dir / "0001.stable-wire.json").write_text(json.dumps(record))
    pending = pending_request(wire_dir)
    for matches, sent in ((True, json.dumps(repeated).encode()),
                          (False, json.dumps(snapshot[0]).encode())):
        with monkeypatch.context() as transport_patch:
            transport_patch.setattr(_child, "HOST_STATE", {})
            transport_patch.setattr(_child, "emit", lambda *a, **k: None)
            transported = []

            def transport(value, *args, **kwargs):
                transported.append(value)
                return RESPONSE

            transport_patch.setattr(kernel_bounded_http, "request_bytes", transport)
            resumed = tmp_path / ("faithful" if matches else "divergent")
            resumed.mkdir()
            replayer = Replayer([], pending=pending)
            _child.install_wire_recorder(resumed, ENDPOINT, replayer)
            active = ACTIVE_REQUEST.set(replay[1])
            try:
                if matches:
                    assert kernel_bounded_http.request_bytes(sent, ENDPOINT, "") == RESPONSE
                    assert replayer.pending_verified is not None
                    assert replayer.diverged is None
                    assert transported == [wire]
                else:
                    # v0.5.17 retires an unanswered divergent request and sends
                    # the new bytes live, without lending them replay authority.
                    assert kernel_bounded_http.request_bytes(sent, ENDPOINT, "") == RESPONSE
                    assert transported == [sent]
                    assert replayer.pending is None and replayer.pending_mismatch
                    assert replayer.diverged is not None
                    assert replayer.pending_verified is None
                    assert replayer.reserves_for_layout_decision() is None
                    assert layout.replay_stable_wire_windows() is None
            finally:
                ACTIVE_REQUEST.reset(active)


def test_same_post_fit_growth_still_recovers_on_live_send(monkeypatch, allocator):
    prepare, render = allocator
    projected = render(prepare(replay=True))
    assert is_stable_wire_projection(projected["messages"])
    # A live meter can discover growth after rendering. It has no tape authority.
    prepared = prepare()
    snapshot = _snapshot(prepared)
    recovered, changed = _recover(prepared, projected, snapshot)
    assert changed
    assert json.dumps(recovered).encode() == json.dumps(snapshot[0]).encode()
    assert not is_stable_wire_projection(recovered["messages"])


def test_replay_without_recorded_layout_keeps_capacity_recovery(monkeypatch, allocator):
    prepare, render = allocator
    live = render(prepare(replay=True))
    _install_replay(monkeypatch, live, None)
    monkeypatch.setattr(layout, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", lambda: None)
    prepared = prepare()
    snapshot = _snapshot(prepared)
    # Exercise the qualification seam with a stable-shaped body but no record.
    recovered, changed = _recover(prepared, live, snapshot)
    assert changed
    assert json.dumps(recovered).encode() == json.dumps(snapshot[0]).encode()


@pytest.mark.parametrize("mismatch", ["window", "epoch"])
def test_recorded_layout_mismatch_fails_open_during_qualification(monkeypatch, allocator, mismatch):
    prepare, render = allocator
    live_prepared = prepare(replay=True)
    live = render(live_prepared)
    _install_replay(monkeypatch, live, live_prepared[1]._stable_wire_reserves_record)
    prepared = prepare(replay=True)  # Even a fitting mismatch must fail open.
    snapshot = _snapshot(prepared)
    rebuilt = render(prepared)
    windows, epoch = recorded_window_layout(live["messages"])
    if mismatch == "window":
        windows = [window + " changed" for window in windows]
    else:
        epoch += 1
    monkeypatch.setattr(layout, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", lambda: (windows, epoch))
    recovered, changed = _recover(prepared, rebuilt, snapshot)
    assert changed
    assert json.dumps(recovered).encode() == json.dumps(snapshot[0]).encode()
    assert not is_stable_wire_projection(recovered["messages"])
    assert [event for event in prepared[1].journal.events
            if event["kind"] == "stable_wire_replay_mismatch"]


def test_both_switches_off_qualification_and_tape_match_base(tmp_path, tmp_path_factory):
    baseline = pinned_tree("37f0ac81", tmp_path_factory)
    root = Path(__file__).resolve().parents[1]
    script = r'''
import copy, json, sys
from pathlib import Path
from types import SimpleNamespace
import kernel_bounded_http
from kernel_cache_layout_candidate import render_final_cache_candidate
from kernel_final_wire_allocation import _recover_post_fit_growth
from kernel_frame_allocation_capture import FrozenRequestCounter
from kernel_measurement import ACTIVE_REQUEST
from kernel_request_diet import RequestTokenMeter
from stateframe import _child
from test_stable_wire import Journal, _final_layout_body
body = _final_layout_body()
meter = RequestTokenMeter(tokenizer=lambda text: 100, template_overhead_tokens=0)
meter.bind_request("off", "off")
inputs = {"token_ceiling":33344, "byte_ceiling":1000000,
          "source_plan":{}, "sources":{}, "receipt":None}
counter = FrozenRequestCounter.capture(body, meter, 100, "tokenize", "off")
capture = SimpleNamespace(request_id="off", journal=Journal(),
    request_source_final_delivery=None, _frame_candidate_counter=counter,
    _frame_candidate_inputs=copy.deepcopy(inputs))
runtime = SimpleNamespace()
projected = render_final_cache_candidate(capture, runtime, body, meter=meter,
                                         inputs=capture._frame_candidate_inputs)
qualified, changed = _recover_post_fit_growth(capture, runtime, projected,
    body, counter, inputs, None, meter=meter, endpoint=None, key="")
assert not changed
wire = json.dumps(qualified).encode()
assert wire == json.dumps(body).encode()
kernel_bounded_http.request_bytes = lambda *a, **k: b'{"choices":[{}]}'
_child.time.monotonic = lambda: 1.0
_child.emit = lambda *a, **k: None
run = Path(sys.argv[1]); run.mkdir()
_child.install_wire_recorder(run, "http://localhost/chat")
active = ACTIVE_REQUEST.set(capture)
try:
    kernel_bounded_http.request_bytes(wire, "http://localhost/chat", "")
finally:
    ACTIVE_REQUEST.reset(active)
print(json.dumps({str(p.relative_to(run)):p.read_bytes().hex()
                  for p in run.rglob("*") if p.is_file()}, sort_keys=True))
'''
    outputs = []
    for checkout, name in ((baseline, "baseline"), (root, "current")):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("STATEFRAME_", "KERNEL_", "DS_", "REACT_", "PILOT_", "UNIT_PLAN_"))}
        env.update(STATEFRAME_STABLE_WIRE="off", STATEFRAME_FINAL_CACHE_LAYOUT="off",
                   STATEFRAME_PAGE_LOG="off",
                   PYTHONPATH=os.pathsep.join(str(path) for path in (
                       checkout / "src/stateframe/engine", checkout / "src", root / "tests")))
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path / name)],
                                env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
    assert set(outputs[1]) == {"wire/0001.request.json", "wire/0001.response.json", "sends.jsonl"}
    assert "stable_wire_reserves" not in bytes.fromhex(outputs[1]["sends.jsonl"]).decode()
