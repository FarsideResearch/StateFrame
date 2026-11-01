"""Routed wire records are written before submission and constrain replay."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import _child, replay_tape, resume, runner
from stateframe import profiles


URL = "http://127.0.0.1:8090/v1/chat/completions"
ROUTE = {"role": "read", "scope": "run", "unit_class": None, "endpoint": URL,
         "endpoint_host": "127.0.0.1:8090", "model": "reader", "escalated_from": None}
WIRE = b'{"model": "reader", "messages": []}'
RESPONSE = b'{"choices":[{"message":{"tool_calls":[]}}]}'


def _write_send(wire, number, *, route=ROUTE, completed=True):
    wire.mkdir(parents=True, exist_ok=True)
    stem = f"{number:04d}"
    (wire / f"{stem}.route.json").write_text(json.dumps(route))
    (wire / f"{stem}.request.json").write_bytes(WIRE)
    if completed:
        (wire / f"{stem}.response.json").write_bytes(RESPONSE)


def _interrupted_routed_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "parent").mkdir()
    spec = {"schema": "stateframe-run/1", "environment": {"REACT_STEPS": "8"},
            "public_row": {"msg": "task"}, "parent_snapshot": str(run / "parent"),
            "chat_url": URL, "model": "reader", "max_turns": 8, "repo": str(tmp_path),
            "engine_source": profiles.engine_source_summary(), "routes": {"roles": {"read": ROUTE}}}
    (run / "spec.json").write_text(json.dumps(spec))
    return run, spec


def test_second_resume_keeps_route_move_for_carried_tape(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2)
    moved = {**ROUTE, "endpoint": "http://127.0.0.1:9000/v1/chat/completions",
             "endpoint_host": "127.0.0.1:9000"}
    _write_send(run / "wire", 1, route=moved)
    first_move = {"role": "read", "field": "endpoint", "from": "http://127.0.0.1:8090/v1",
                  "to": "http://127.0.0.1:9000/v1"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "route_changes": [first_move]}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 2
    assert resumed["route_changes"] == [first_move]
    second_move = {**first_move, "from": first_move["to"], "to": "http://127.0.0.1:9100/v1"}
    runner._compose_resume_route_changes({"resume": resumed}, [second_move])
    assert resumed["route_changes"] == [first_move, second_move]
    runner._compose_resume_route_changes({"resume": resumed}, [second_move])
    assert resumed["route_changes"] == [first_move, second_move]
    latest = {**moved, "endpoint": "http://127.0.0.1:9100/v1/chat/completions",
              "endpoint_host": "127.0.0.1:9100"}
    replay = replay_tape.Replayer(resumed["tape"], route_changes=resumed["route_changes"])
    assert replay.answer(WIRE, route=latest) == RESPONSE
    assert replay.answer(WIRE, route=latest) == RESPONSE


def test_later_model_move_wins_when_replaying_earlier_route(tmp_path):
    wire = tmp_path / "wire"
    _write_send(wire, 1)
    tape = replay_tape.build(wire, tmp_path / "sends.jsonl")
    changes = [{"role": "read", "field": "model", "from": "reader", "to": "middle"},
               {"role": "read", "field": "model", "from": "middle", "to": "latest"}]
    replay = replay_tape.Replayer(tape, route_changes=changes)

    assert replay.answer(b'{"model": "latest", "messages": []}',
                         route={**ROUTE, "model": "latest"}) == RESPONSE
    assert replay.diverged is None


def test_carried_tape_uses_its_own_pending_send(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2)
    _write_send(old_wire, 3, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire)}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 2
    assert resumed["pending"]["send"] == 3
    assert resumed["pending"]["request"] == str(old_wire / "0003.request.json")


def test_equal_length_replay_keeps_prior_pending_and_route_changes(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    move = {"role": "read", "field": "endpoint", "from": "http://127.0.0.1:8090/v1",
            "to": "http://127.0.0.1:9000/v1"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": [move]}
    prior_pending = spec["resume"]["pending"]
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert resumed["pending"] == prior_pending
    assert resumed["route_changes"] == [move]


def test_unverified_rebuilt_pending_does_not_replace_the_original_check(tmp_path):
    # Killed after the child wrote its rebuilt request 2 but before the replayer checked it against the
    # carried pending send: the rebuilt copy is unverified, so the original check and its moves stay.
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    (run / "wire" / "0002.request.json").write_bytes(b'{"changed": "request bytes"}')
    move = {"role": "read", "field": "endpoint", "from": "http://127.0.0.1:8090/v1",
            "to": "http://127.0.0.1:9000/v1"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": [move]}
    prior_pending = spec["resume"]["pending"]
    assert replay_tape.pending_request(run / "wire", routes_required=True)["send"] == prior_pending["send"]
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert resumed["pending"] == prior_pending
    assert resumed["route_changes"] == [move]


def test_equal_length_replay_drops_prior_pending_after_later_failed_send(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    (run / "wire" / "0002.error.txt").write_text("send failed")
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire),
                      "route_changes": [{"role": "read", "field": "model", "from": "reader", "to": "new"}]}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert "pending" not in resumed and "route_changes" not in resumed


@pytest.mark.parametrize("divergence", ["marker", "request"])
def test_diverged_replay_drops_prior_pending_and_route_changes(tmp_path, divergence):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    if divergence == "marker":
        (run / "resume-divergence.json").write_text("{}")
    else:
        (run / "wire" / "0001.request.json").write_bytes(b'{"model": "reader", "messages": [1]}')
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire),
                      "route_changes": [{"role": "read", "field": "model", "from": "reader", "to": "new"}]}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert "pending" not in resumed and "route_changes" not in resumed


def test_live_restart_discards_carried_route_changes_and_pending(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire),
                      "route_changes": [{"role": "read", "field": "model", "from": "reader",
                                         "to": "renamed"}]}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 9}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert resumed["restart_live"] is True
    assert resumed["tape"] == []
    assert "pending" not in resumed and "route_changes" not in resumed
    runner._compose_resume_route_changes({"resume": resumed}, [
        {"role": "read", "field": "model", "from": "renamed", "to": "latest"}])
    assert "route_changes" not in resumed


@pytest.mark.parametrize("damage", ["missing_completed_route", "malformed_pending_route"])
def test_routed_tape_damage_refused_before_archive(tmp_path, damage):
    run, _ = _interrupted_routed_run(tmp_path)
    if damage == "missing_completed_route":
        _write_send(run / "wire", 1)
        (run / "wire" / "0001.route.json").unlink()
    else:
        (run / "wire").mkdir()
        (run / "wire" / "0001.route.json").write_text("{broken")
    before = {str(path.relative_to(run)): (path.read_bytes() if path.is_file() else None)
              for path in run.rglob("*")}

    with pytest.raises(resume.ResumeRefused, match="recorded model exchanges are damaged.*route"):
        resume.assess(run)

    after = {str(path.relative_to(run)): (path.read_bytes() if path.is_file() else None)
             for path in run.rglob("*")}
    assert after == before
    assert not (run / "attempts").exists()


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
