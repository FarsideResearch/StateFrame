"""Regression checks for route and subframe records sharing a run."""
from __future__ import annotations

import json

from stateframe import parallel_resources, replay_tape
from stateframe.runner import auto_parallel_concurrency


def test_auto_concurrency_reads_routes_from_recorded_run_spec(monkeypatch):
    primary = "http://127.0.0.1:8080/v1"
    secondary = "http://127.0.0.1:8081/v1"
    seen = []

    def idle(endpoint):
        seen.append(endpoint)
        return [0, 1, 2] if endpoint == primary else [0, 1, 2, 3]

    monkeypatch.setattr(parallel_resources, "idle_slots", idle)
    spec = {"chat_url": primary + "/chat/completions",
            "routes": {"schema": "stateframe-routes/1",
                       "roles": {"unit": {"endpoint": secondary}}}}

    assert auto_parallel_concurrency(spec, 8) == 7
    assert seen == [primary, secondary]
    seen.clear()
    assert auto_parallel_concurrency({"chat_url": spec["chat_url"]}, 8) == 3
    assert seen == [primary]


def test_routed_subframe_replay_checks_both_records(tmp_path):
    wire = tmp_path / "wire"
    wire.mkdir()
    request = b'{"model":"unit-model","messages":[]}'
    response = b'{"choices":[{"message":{"tool_calls":[]}}]}'
    route = {"role": "unit", "scope": "unit", "unit_class": "default",
             "endpoint": "http://127.0.0.1:8081/v1/chat/completions",
             "endpoint_host": "127.0.0.1:8081", "model": "unit-model",
             "escalated_from": None}
    identity = {"url": "http://127.0.0.1:8081/v1", "model": "unit-model"}
    (wire / "0001.request.json").write_bytes(request)
    (wire / "0001.response.json").write_bytes(response)
    (wire / "0001.route.json").write_text(json.dumps(route))
    (wire / "0001.endpoint.json").write_text(json.dumps(identity))

    tape = replay_tape.build(wire, tmp_path / "sends.jsonl", routes_required=True)
    assert tape[0]["route"] == route
    assert tape[0]["endpoint_identity"] == identity
    assert replay_tape.Replayer(tape).answer(request, route=route,
                                              endpoint_identity=identity) == response

    wrong_identity = {**identity, "model": "other-model"}
    rejected = replay_tape.Replayer(tape)
    assert rejected.answer(request, route=route, endpoint_identity=wrong_identity) is None
    assert rejected.diverged["recorded_endpoint_identity"] == identity
    assert rejected.diverged["replayed_endpoint_identity"] == wrong_identity

    wrong_route = {**route, "endpoint": "http://127.0.0.1:8090/v1/chat/completions"}
    rejected = replay_tape.Replayer(tape)
    assert rejected.answer(request, route=wrong_route, endpoint_identity=identity) is None
    assert rejected.diverged["reason"] == "routed send destination or model differs"
