"""Durable numeric trim authority across completed, pending and legacy replay tapes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import kernel_bounded_http
import kernel_cache_layout_candidate as candidate
from kernel_measurement import ACTIVE_REQUEST
from kernel_stable_wire import is_stable_wire_projection, recorded_window_layout
from kernel_stable_wire_record import validate_reserves_record
from stateframe import _child
from stateframe.replay_tape import ReplayUnfaithful, Replayer, TapeError, build, pending_request
from test_stable_wire_fit import SizeMeter, UnderTrimmer, _inputs, _layout_body, _render
from tools.pinned_tree import pinned_tree

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
RESPONSE = b'{"choices":[{"message":{"content":"ok"}}]}'
RECORD = {"reserves": [[28, 256], [46, 512]], "trim_count": 2, "epoch": 1}


def _isolate_recorder(monkeypatch):
    monkeypatch.setattr(_child, "HOST_STATE", {})
    monkeypatch.setattr(_child, "emit", lambda *_args, **_kwargs: None)
    for name in ("_REPLAY_LAYOUT_DECISION_PROVIDER", "_REPLAY_STABLE_WIRE_DECISION_PROVIDER",
                 "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER"):
        monkeypatch.setattr(candidate, name, None)


@pytest.mark.parametrize("pending", [False, True])
def test_trim_record_is_durable_before_transport_and_restored_for_replay(monkeypatch, tmp_path, pending):
    _isolate_recorder(monkeypatch)
    run = tmp_path / "live"
    run.mkdir()
    body = _layout_body(6)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
    live, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(70), trim=trimmer)
    wire = json.dumps(live).encode()
    assert capture._stable_wire_reserves_record == RECORD

    class Interrupted(BaseException):
        pass

    def transport(sent, *_args, **_kwargs):
        assert sent == wire
        assert json.loads((run / "wire/0001.stable-wire.json").read_bytes()) == RECORD
        assert (run / "wire/0001.request.json").read_bytes() == wire
        assert not (run / "sends.jsonl").exists()
        # No completed-send log can help recover an interrupted transport.
        if pending:
            raise Interrupted
        return RESPONSE

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", transport)
    publications = []
    atomic = _child.atomic_bytes

    def publish(path, data):
        atomic(path, data)
        publications.append(path.name)

    monkeypatch.setattr(_child, "atomic_bytes", publish)
    _child.install_wire_recorder(run, ENDPOINT)
    active = ACTIVE_REQUEST.set(capture)
    try:
        if pending:
            with pytest.raises(Interrupted):
                kernel_bounded_http.request_bytes(wire, ENDPOINT, "")
        else:
            assert kernel_bounded_http.request_bytes(wire, ENDPOINT, "") == RESPONSE
    finally:
        ACTIVE_REQUEST.reset(active)
    assert publications[:2] == ["0001.stable-wire.json", "0001.request.json"]
    tape = build(run / "wire", run / "sends.jsonl")
    unfinished = pending_request(run / "wire")
    item = unfinished if pending else tape[0]
    assert item["stable_wire_reserves"] == RECORD
    if not pending:
        assert json.loads((run / "sends.jsonl").read_bytes())["stable_wire_reserves"] == RECORD
    replay = Replayer(tape, pending=unfinished)
    assert replay.reserves_for_layout_decision() == RECORD
    copy = replay.reserves_for_layout_decision()
    copy["reserves"].clear()
    assert replay.reserves_for_layout_decision() == RECORD

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *_args, **_kwargs: RESPONSE)
    _child.install_wire_recorder(resumed, ENDPOINT, replay)

    class UnavailableMeter:
        def measure(self, _body):
            raise RuntimeError("unavailable")

    body = _layout_body(6)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
    rebuilt, capture = _render(monkeypatch, body, meter=UnavailableMeter(),
                                inputs=_inputs(70), trim=trimmer)
    assert trimmer.calls == [(28, 256), (46, 512)]
    assert json.dumps(rebuilt).encode() == wire
    assert candidate.replay_stable_wire_windows() == recorded_window_layout(live["messages"])
    active = ACTIVE_REQUEST.set(capture)
    try:
        assert kernel_bounded_http.request_bytes(json.dumps(rebuilt).encode(), ENDPOINT, "") == RESPONSE
    finally:
        ACTIVE_REQUEST.reset(active)
    assert replay.diverged is None
    assert replay.pending is None
    assert replay.reserves_for_layout_decision() is None
    assert json.loads((resumed / "sends.jsonl").read_bytes())["stable_wire_reserves"] == RECORD


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("mismatch", [False, True])
def test_old_format_tape_without_reserves_loads_and_checks_final_windows(
        monkeypatch, tmp_path, pending, mismatch):
    _isolate_recorder(monkeypatch)
    live, _ = _render(monkeypatch, _layout_body(), meter=SizeMeter(), inputs=_inputs(1000))
    wire = json.dumps(live).encode()
    folder = tmp_path / "old-wire"
    folder.mkdir()
    (folder / "0001.request.json").write_bytes(wire)
    if not pending:
        (folder / "0001.response.json").write_bytes(RESPONSE)
    sends = tmp_path / "sends.jsonl"
    sends.write_text(json.dumps({"send": 1, "request_sha256": hashlib.sha256(wire).hexdigest()}) + "\n")
    tape = build(folder, sends)
    unfinished = pending_request(folder)
    assert "stable_wire_reserves" not in (unfinished if pending else tape[0])
    replay = Replayer(tape, pending=unfinished)
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *_args, **_kwargs: RESPONSE)
    _child.install_wire_recorder(resumed, ENDPOINT, replay)
    assert replay.reserves_for_layout_decision() is None
    body = _layout_body()
    if mismatch:
        for row in body["messages"]:
            row["content"] = row.get("content", "").replace("value = 1", "value = 2")
    runtime = SimpleNamespace(_stable_wire_state=None)
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(1000), runtime=runtime)
    events = [event for event in capture.journal.events if event["kind"] == "stable_wire_replay_mismatch"]
    if mismatch:
        assert len(events) == 1
        assert not is_stable_wire_projection(result["messages"])
        assert runtime._stable_wire_state is None
        assert capture._final_cache_layout_record["stable_wire_reason"] == "replay_layout_mismatch"
    else:
        assert events == []
        assert json.dumps(result).encode() == wire


@pytest.mark.parametrize("bad", [
    {**RECORD, "secret": "must not be accepted"},
    {**RECORD, "trim_count": True},
    {**RECORD, "epoch": 0},
    {**RECORD, "reserves": [[28, 256], [27, 512]]},
    {**RECORD, "reserves": [[28, 256], [46, "path"]]},
    {**RECORD, "reserves": [[1, 1]] * 4, "trim_count": 4},
])
def test_numeric_record_rejects_invalid_or_unbounded_trim_history(tmp_path, bad):
    with pytest.raises(ValueError):
        validate_reserves_record(bad)
    (tmp_path / "0001.request.json").write_bytes(b"{}")
    (tmp_path / "0001.stable-wire.json").write_text(json.dumps(bad))
    with pytest.raises(TapeError, match="stable wire reserves"):
        pending_request(tmp_path)


@pytest.mark.parametrize("flag", [None, "off"])
@pytest.mark.parametrize("final_layout", [None, "on", "off"])
def test_stable_wire_off_request_and_tape_bytes_match_0517(
        tmp_path, tmp_path_factory, flag, final_layout):
    baseline = pinned_tree("9f07117d53b8edf6a02ce2f48a8ff706ccad11f3", tmp_path_factory)
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
import kernel_bounded_http
import kernel_request_source_window as source
from kernel_cache_layout_candidate import render_final_cache_candidate
from kernel_measurement import ACTIVE_REQUEST
from stateframe import _child
from test_stable_wire import Meter, _final_layout_body
source.verify_final_source_delivery = lambda *a, **k: {"status": "COMPLETE"}
source.source_delivery_satisfied = lambda receipt: True
capture = SimpleNamespace(request_id="off", request_source_final_delivery=None)
body = render_final_cache_candidate(capture, SimpleNamespace(), _final_layout_body(),
    meter=Meter(), inputs={"token_ceiling":1000, "byte_ceiling":1000000,
                          "source_plan":{}, "sources":{}, "receipt":None})
wire = json.dumps(body).encode()
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
        if flag is not None:
            env["STATEFRAME_STABLE_WIRE"] = flag
        if final_layout is not None:
            env["STATEFRAME_FINAL_CACHE_LAYOUT"] = final_layout
        env.update(STATEFRAME_PAGE_LOG="off",
                   PYTHONPATH=os.pathsep.join(str(path) for path in (
                       checkout / "src/stateframe/engine", checkout / "src", root / "tests")))
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path / name)],
                                env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
    assert set(outputs[1]) == {"wire/0001.request.json", "wire/0001.response.json", "sends.jsonl"}
    assert "stable_wire_reserves" not in bytes.fromhex(outputs[1]["sends.jsonl"]).decode()


@pytest.mark.parametrize("final_layout", ["on", "off"])
def test_allocator_off_request_and_tape_bytes_match_immediate_base(
        tmp_path, tmp_path_factory, final_layout):
    baseline = pinned_tree("37f0ac81", tmp_path_factory)
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json, sys
from pathlib import Path
import pytest
import kernel_bounded_http
from kernel_measurement import ACTIVE_REQUEST
from stateframe import _child
from test_stable_wire_trim_selection import allocator
patch = pytest.MonkeyPatch()
prepare, render = allocator.__wrapped__(patch, Path(sys.argv[1]))
patch.setenv("STATEFRAME_STABLE_WIRE", "off")
patch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", sys.argv[2])
prepared = prepare()
body = render(prepared)
assert getattr(prepared[1], "_stable_wire_reserves_record", None) is None
assert "kernel_stable_wire_initial" not in sys.modules
wire = json.dumps(body).encode()
kernel_bounded_http.request_bytes = lambda *a, **k: b'{"choices":[{}]}'
_child.time.monotonic = lambda: 1.0
_child.emit = lambda *a, **k: None
run = Path(sys.argv[1]); run.mkdir()
_child.install_wire_recorder(run, "http://localhost/chat")
active = ACTIVE_REQUEST.set(prepared[1])
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
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in (
            checkout / "src/stateframe/engine", checkout / "src", root / "tests"))
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path / name), final_layout],
                                env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
    assert set(outputs[1]) == {"wire/0001.request.json", "wire/0001.response.json", "sends.jsonl"}
    assert "stable_wire_reserves" not in bytes.fromhex(outputs[1]["sends.jsonl"]).decode()


def test_publication_copies_only_surviving_layout_authority(monkeypatch):
    from kernel_final_wire_allocation import _stable_wire_record_for_publication
    from kernel_request_seal import RequestSealError

    live, capture = _render(monkeypatch, _layout_body(), meter=SizeMeter(), inputs=_inputs(1000))
    record = _stable_wire_record_for_publication(capture, live)
    assert record == {"reserves": [], "trim_count": 0, "epoch": 1}
    record["epoch"] = 2
    assert capture._stable_wire_reserves_record["epoch"] == 1
    assert _stable_wire_record_for_publication(capture, _layout_body()) is None
    capture._stable_wire_reserves_record["epoch"] = 2
    with pytest.raises(RequestSealError, match="trim epoch"):
        _stable_wire_record_for_publication(capture, live)


@pytest.mark.parametrize("pending", [False, True])
def test_recorded_trim_authority_preserves_0517_divergence_policy(monkeypatch, tmp_path, pending):
    _isolate_recorder(monkeypatch)
    recorded = b'{"model": "scripted", "task": "original"}'
    rebuilt = recorded.replace(b"original", b"different")
    request = tmp_path / "old-request.json"
    response = tmp_path / "old-response.json"
    request.write_bytes(recorded)
    response.write_bytes(RESPONSE)
    item = {"send": 1, "request": str(request), "response": str(response),
            "response_sha256": hashlib.sha256(RESPONSE).hexdigest(),
            "stable_wire_reserves": RECORD}
    player = Replayer([], pending=item) if pending else Replayer([item])
    assert player.reserves_for_layout_decision() == RECORD
    live_calls = []
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda wire, *_args, **_kwargs: live_calls.append(wire) or RESPONSE)
    _child.install_wire_recorder(tmp_path, ENDPOINT, player)
    if pending:
        assert kernel_bounded_http.request_bytes(rebuilt, ENDPOINT, "") == RESPONSE
        assert live_calls == [rebuilt]
        assert player.pending is None and player.pending_mismatch
    else:
        with pytest.raises(ReplayUnfaithful, match="recorded request"):
            kernel_bounded_http.request_bytes(rebuilt, ENDPOINT, "")
        assert live_calls == []
        assert player.cursor == 0
    assert player.reserves_for_layout_decision() is None
