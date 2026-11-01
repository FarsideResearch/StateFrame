"""Routed wire records are written before submission and constrain replay."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import _child, replay_tape


URL = "http://127.0.0.1:8090/v1/chat/completions"
ROUTE = {"role": "read", "scope": "run", "unit_class": None, "endpoint": URL,
         "endpoint_host": "127.0.0.1:8090", "model": "reader", "escalated_from": None}
WIRE = b'{"model": "reader", "messages": []}'
RESPONSE = b'{"choices":[{"message":{"tool_calls":[]}}]}'


def test_route_record_precedes_request_and_submission(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(_child.__file__).parent / "engine"))
    import kernel_bounded_http

    run = tmp_path / "run"
    run.mkdir()
    seen = []
    replace = _child.os.replace

    def observed_replace(source, target):
        seen.append(Path(target).name)
        replace(source, target)

    def physical(wire, endpoint, key, **kwargs):
        assert (run / "wire" / "0001.route.json").is_file()
        assert (run / "wire" / "0001.request.json").read_bytes() == WIRE
        assert endpoint == URL and kwargs["route_meta"]["model"] == "reader"
        return RESPONSE

    monkeypatch.setattr(_child.os, "replace", observed_replace)
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", physical)
    _child.install_wire_recorder(run, URL, routed_endpoints={URL})
    assert kernel_bounded_http.request_bytes(WIRE, URL, "", deadline=1000000,
                                             socket_timeout=10, route_meta=ROUTE) == RESPONSE
    assert seen[:2] == ["0001.route.json", "0001.request.json"]
    assert json.loads((run / "wire" / "0001.route.json").read_text()) == ROUTE


def test_orphan_route_record_becomes_pending_and_requires_same_destination(tmp_path):
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0001.route.json").write_text(json.dumps(ROUTE))
    assert replay_tape.build(wire, tmp_path / "sends.jsonl") == []
    pending = replay_tape.pending_request(wire)
    assert pending == {"send": 1, "route": ROUTE, "orphan_route": True}
    replay = replay_tape.Replayer([], pending=pending)
    replay.answer(WIRE, route={**ROUTE, "model": "wrong"})
    assert replay.diverged["reason"] == "pending routed send destination or model differs"
    replay = replay_tape.Replayer([], pending=pending)
    assert replay.answer(WIRE, route=ROUTE) is None
    assert replay.diverged is None and replay.pending is None


def test_completed_route_replay_and_per_role_rename(tmp_path):
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0001.route.json").write_text(json.dumps(ROUTE))
    (wire / "0001.request.json").write_bytes(WIRE)
    (wire / "0001.response.json").write_bytes(RESPONSE)
    tape = replay_tape.build(wire, tmp_path / "sends.jsonl")
    assert tape[0]["route"] == ROUTE
    replay = replay_tape.Replayer(tape)
    assert replay.answer(WIRE, route=ROUTE) == RESPONSE
    replay = replay_tape.Replayer(tape)
    assert replay.answer(WIRE, route={**ROUTE, "endpoint": "http://127.0.0.1:9000/v1/chat/completions"}) is None
    assert replay.diverged["reason"] == "routed send destination or model differs"
    moved = {**ROUTE, "endpoint": "http://127.0.0.1:9000/v1/chat/completions",
             "endpoint_host": "127.0.0.1:9000", "model": "renamed"}
    replay = replay_tape.Replayer(tape, route_changes=[
        {"role": "read", "field": "endpoint", "from": "http://127.0.0.1:8090/v1",
         "to": "http://127.0.0.1:9000/v1"},
        {"role": "read", "field": "model", "from": "reader", "to": "renamed"}])
    assert replay.answer(b'{"model": "renamed", "messages": []}', route=moved) == RESPONSE


def test_pending_route_with_request_checks_bytes(tmp_path):
    wire = tmp_path / "wire"
    wire.mkdir()
    (wire / "0001.route.json").write_text(json.dumps(ROUTE))
    (wire / "0001.request.json").write_bytes(WIRE)
    pending = replay_tape.pending_request(wire)
    assert pending["request_digest"] == replay_tape.normalized_digest(WIRE)
    replay = replay_tape.Replayer([], pending=pending)
    replay.answer(b'{"model": "reader", "messages": [1]}', route=ROUTE)
    assert replay.diverged is not None


def test_routed_recorder_refuses_unlisted_destination_before_writing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(_child.__file__).parent / "engine"))
    import kernel_bounded_http

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", lambda *args, **kwargs: RESPONSE)
    _child.install_wire_recorder(run, URL, routed_endpoints={URL})
    with pytest.raises(ValueError, match="authorized destination"):
        kernel_bounded_http.request_bytes(WIRE, "http://127.0.0.1:9999/v1/chat/completions", "",
                                          deadline=1000000, socket_timeout=10, route_meta=ROUTE)
    assert not list((run / "wire").iterdir())


@pytest.mark.parametrize("change", [
    {"role": "plan"},
    {"model": "other"},
    {"endpoint": "http://127.0.0.1:9000/v1/chat/completions",
     "endpoint_host": "127.0.0.1:9000"},
])
@pytest.mark.parametrize("completed", [False, True])
def test_routed_replay_mismatch_refuses_before_physical_send(tmp_path, monkeypatch, change, completed):
    monkeypatch.syspath_prepend(str(Path(_child.__file__).parent / "engine"))
    import kernel_bounded_http

    run = tmp_path / "run"
    run.mkdir()
    pending = {"send": 1, "route": ROUTE, "orphan_route": True}
    recorded = tmp_path / "recorded.request.json"
    recorded.write_bytes(WIRE)
    tape = [{"send": 1, "route": ROUTE, "request": str(recorded),
             "request_digest": replay_tape.normalized_digest(WIRE)}] if completed else []
    actual = {**ROUTE, **change}
    endpoint = actual["endpoint"]
    wire = json.dumps({"model": actual["model"], "messages": []}).encode()
    sent = []
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda *args, **kwargs: sent.append(args) or RESPONSE)
    _child.install_wire_recorder(run, URL, replayer=replay_tape.Replayer(
        tape, pending=None if completed else pending),
                                 routed_endpoints={URL, endpoint})
    with pytest.raises(replay_tape.CheckpointUnfaithful, match="routed send"):
        kernel_bounded_http.request_bytes(wire, endpoint, "", deadline=1000000,
                                          socket_timeout=10, route_meta=actual)
    with pytest.raises(replay_tape.CheckpointUnfaithful, match="earlier routed replay"):
        kernel_bounded_http.request_bytes(wire, endpoint, "", deadline=1000000,
                                          socket_timeout=10, route_meta=actual)
    assert sent == []
    assert (run / "resume-divergence.json").is_file()
