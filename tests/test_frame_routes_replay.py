"""Routed wire records are written before submission and constrain replay."""
from __future__ import annotations

import hashlib
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


def _write_send(wire, number, *, route=ROUTE, request=WIRE, completed=True):
    wire.mkdir(parents=True, exist_ok=True)
    stem = f"{number:04d}"
    if route is not None:
        (wire / f"{stem}.route.json").write_text(json.dumps(route))
    (wire / f"{stem}.request.json").write_bytes(request)
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


def _mark_pending_verified(run, pending, *, route=ROUTE):
    request = pending.get("request")
    marker = {"send": pending["send"],
              "request_sha256": hashlib.sha256(Path(request).read_bytes()).hexdigest() if request else None,
              "recorded_route": pending["route"], "route": route}
    (run / "resume-pending-verified.json").write_text(json.dumps(marker))


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
    assert "pending" not in resumed
    assert resumed["pending_check"] == {"status": "skipped", "reason": "no_pending_check"}
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


def test_model_moved_replayed_prefix_keeps_tail_but_skips_unverified_pending(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2)
    _write_send(old_wire, 3, completed=False)
    moved_route = {**ROUTE, "model": "latest"}
    moved_request = b'{"model": "latest", "messages": []}'
    _write_send(run / "wire", 1, route=moved_route, request=moved_request)
    _write_send(run / "wire", 2, route=moved_route, request=moved_request, completed=False)
    moves = [{"role": "read", "field": "model", "from": "reader", "to": "middle"},
             {"role": "read", "field": "model", "from": "middle", "to": "latest"}]
    spec["resume"] = {"tape": replay_tape.build(old_wire, old_wire.parent / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": moves}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert [item["send"] for item in resumed["tape"]] == [1, 2]
    assert "pending" not in resumed
    assert resumed["pending_check"] == {"status": "skipped", "reason": "unverified_pending"}
    assert resumed["route_changes"] == moves


@pytest.mark.parametrize("attempt_request", [
    b'{"model": "unauthorized", "messages": []}',
    b'{"model": "latest", "messages": [1]}',
])
def test_model_moved_prefix_with_other_difference_drops_carried_state(tmp_path, attempt_request):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2)
    _write_send(old_wire, 3, completed=False)
    _write_send(run / "wire", 1, route={**ROUTE, "model": "latest"}, request=attempt_request)
    moves = [{"role": "read", "field": "model", "from": "reader", "to": "latest"}]
    spec["resume"] = {"tape": replay_tape.build(old_wire, old_wire.parent / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": moves}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert "pending" not in resumed
    assert resumed["route_changes"] == moves
    assert resumed["pending_check"]["reason"] == "unverified_pending"


def test_unrouted_model_name_difference_does_not_carry_prior_tape(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    spec.pop("routes")
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, route=None)
    _write_send(old_wire, 2, route=None)
    _write_send(run / "wire", 1, route=None,
                request=b'{"model": "latest", "messages": []}')
    spec["resume"] = {"tape": replay_tape.build(old_wire, old_wire.parent / "sends.jsonl"),
                      "model_swap": {"served": "latest", "recorded": ["reader"]}}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert resumed["tape"][0]["request_digest"] == replay_tape.normalized_digest(
        b'{"model": "latest", "messages": []}')
    assert "pending_check" not in resumed


def test_unrouted_resume_keeps_previous_faithful_tail_without_pending_check(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    spec.pop("routes")
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, route=None)
    _write_send(old_wire, 2, route=None)
    _write_send(run / "wire", 1, route=None)
    spec["resume"] = {"tape": replay_tape.build(old_wire, old_wire.parent / "sends.jsonl")}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)
    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert [item["send"] for item in resumed["tape"]] == [1, 2]
    assert resumed["tape"][1]["request"] == str(old_wire / "0002.request.json")
    assert "pending_check" not in resumed


def test_skipped_check_event_only_for_routed_runs_and_never_in_the_task_row(monkeypatch):
    events = []
    monkeypatch.setattr(_child, "emit", lambda kind, **fields: events.append((kind, fields)))
    skipped = {"status": "skipped", "reason": "unverified_pending"}
    _child.emit_resume_pending_check({"routes": {"roles": {"read": {}}}}, {"pending_check": skipped})
    _child.emit_resume_pending_check({}, {"pending_check": skipped})
    _child.emit_resume_pending_check({"routes": {"roles": {"read": {}}}}, {"attempt": 2})
    _child.emit_resume_pending_check({"routes": {"roles": {"read": {}}}}, None)
    assert events == [("routes_resume_pending_check_skipped", skipped)]
    # The engine's task row feeds request building; the skipped record must never be put there.
    source = Path(_child.__file__).read_text()
    assert "row[" not in source.split("def emit_resume_pending_check", 1)[1].split("\ndef ", 1)[0]


def test_carried_tape_skips_unverified_own_pending_send(tmp_path):
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
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"


def test_resume_without_carried_check_skips_its_own_pending(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(run / "wire", 1, completed=False)
    spec["resume"] = {"tape": replay_tape.build(old_wire, old_wire.parent / "sends.jsonl")}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)
    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert [item["send"] for item in resumed["tape"]] == [1]
    assert "pending" not in resumed
    assert resumed["pending_check"] == {"status": "skipped", "reason": "no_pending_check"}


def test_first_archive_killed_after_wire_move_replays_without_pending(tmp_path, monkeypatch):
    run, spec = _interrupted_routed_run(tmp_path)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    (run / "requests").mkdir()
    real_replace = resume.os.replace

    def killed_after_wire(src, dst):
        if Path(src).name == "requests":
            raise KeyboardInterrupt("killed after wire moved")
        real_replace(src, dst)

    monkeypatch.setattr(resume.os, "replace", killed_after_wire)
    with pytest.raises(KeyboardInterrupt):
        resume.archive_attempt({"run": run, "spec": spec, "attempt": 1, "max_turns": 8}, tmp_path,
                               use_checkpoint=False)
    monkeypatch.setattr(resume.os, "replace", real_replace)
    assert "resume" not in json.loads((run / "spec.json").read_text())
    assert resume.assess(run)["tape_count"] == 1

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)
    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert [item["send"] for item in resumed["tape"]] == [1]
    assert "pending" not in resumed
    assert resumed["pending_check"] == {"status": "skipped", "reason": "interrupted_archive"}


def test_refused_routed_mismatch_never_adopts_its_pending_request(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    refused_request = b'{"model": "reader", "messages": ["refused"]}'
    _write_send(run / "wire", 1, request=refused_request, completed=False)
    (run / "resume-divergence.json").write_text('{"reason":"route mismatch"}')
    spec["resume"] = {"tape": replay_tape.build(old_wire, old_wire.parent / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire)}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)
    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert resumed["tape"] == []
    assert "pending" not in resumed
    assert resumed["pending_check"] == {"status": "skipped", "reason": "diverged"}


def test_verified_resume_uses_its_own_pending_request(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, request=b'{"model": "reader", "messages": ["live"]}',
                completed=False)
    spec["resume"] = {"tape": [], "pending": replay_tape.pending_request(old_wire)}
    _mark_pending_verified(run, spec["resume"]["pending"])
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)
    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert [item["send"] for item in resumed["tape"]] == [1]
    assert resumed["pending"]["send"] == 2
    assert resumed["pending"]["request"] == str(run / "attempts" / "2" / "wire" / "0002.request.json")
    assert "pending_check" not in resumed


def test_failed_send_renumbering_skips_unverified_pending_and_keeps_route_move(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, completed=False)
    (old_wire / "0001.error.txt").write_text("send failed")
    _write_send(old_wire, 2)
    _write_send(old_wire, 3, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    (run / "wire" / "0002.request.json").write_bytes(b'{"model": "reader", "messages": [1]}')
    move = {"role": "read", "field": "endpoint", "from": "http://127.0.0.1:8090/v1",
            "to": "http://127.0.0.1:9000/v1"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": [move]}
    prior_pending = spec["resume"]["pending"]
    assert [record["send"] for record in spec["resume"]["tape"]] == [2]
    assert prior_pending["send"] == 3
    assert replay_tape.pending_request(run / "wire", routes_required=True)["send"] == 2
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"
    assert resumed["route_changes"] == [move]


def test_failed_send_renumbering_retires_verified_pending(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, completed=False)
    (old_wire / "0001.error.txt").write_text("send failed")
    _write_send(old_wire, 2)
    _write_send(old_wire, 3, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    move = {"role": "read", "field": "model", "from": "reader", "to": "renamed"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": [move]}
    _mark_pending_verified(run, spec["resume"]["pending"], route={**ROUTE, "model": "renamed"})
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert resumed["pending"]["send"] == 2
    assert resumed["pending"]["request"] == str(run / "attempts" / "2" / "wire" / "0002.request.json")
    assert resumed["route_changes"] == [move]
    assert "pending_check" not in resumed
    assert (run / "attempts" / "2" / "resume-pending-verified.json").is_file()
    assert not (run / "resume-pending-verified.json").exists()


@pytest.mark.parametrize("bad_send", [True, 1.0, "1", None])
def test_malformed_pending_marker_skips_the_check(tmp_path, bad_send):
    # JSON true == 1 in Python: the marker's send must be an exact int to prove anything.
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, completed=False)
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire)}
    prior_pending = spec["resume"]["pending"]
    _mark_pending_verified(run, prior_pending)
    marker = json.loads((run / "resume-pending-verified.json").read_text())
    (run / "resume-pending-verified.json").write_text(json.dumps({**marker, "send": bad_send}))
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"


def test_archive_killed_between_marker_and_wire_skips_the_check(tmp_path, monkeypatch):
    # Attempt 2 verified the carried pending send (marker written) and was killed live; its archive is
    # then killed after the first move. The next archive has no marker beside its wire.
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1, completed=False)
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire)}
    prior_pending = spec["resume"]["pending"]
    _mark_pending_verified(run, prior_pending)
    _write_send(run / "wire", 1, completed=False)  # attempt 2's rebuilt request, verified, then killed
    (run / "spec.json").write_text(json.dumps(spec))
    real_replace, moved = resume.os.replace, []

    def killed_after_first_move(src, dst):
        if moved:
            raise KeyboardInterrupt("killed while archiving")
        moved.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(resume.os, "replace", killed_after_first_move)
    with pytest.raises(KeyboardInterrupt):
        resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                               use_checkpoint=False)
    monkeypatch.setattr(resume.os, "replace", real_replace)
    assert moved == ["resume-pending-verified.json"]
    assert not (run / "resume-pending-verified.json").exists()

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 3, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"


def test_equal_length_replay_skips_unverified_pending_and_keeps_route_changes(tmp_path):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    move = {"role": "read", "field": "endpoint", "from": "http://127.0.0.1:8090/v1",
            "to": "http://127.0.0.1:9000/v1"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": [move]}
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"
    assert resumed["route_changes"] == [move]


def test_unverified_rebuilt_pending_skips_the_original_check(tmp_path):
    # Killed after the child wrote its rebuilt request 2 but before the replayer checked it.
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
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"
    assert resumed["route_changes"] == [move]


@pytest.mark.parametrize("stale_field", ["request_sha256", "route"])
def test_stale_pending_verification_marker_skips_check(tmp_path, stale_field):
    run, spec = _interrupted_routed_run(tmp_path)
    old_wire = run / "attempts" / "1" / "wire"
    _write_send(old_wire, 1)
    _write_send(old_wire, 2, completed=False)
    _write_send(run / "wire", 1)
    _write_send(run / "wire", 2, completed=False)
    move = {"role": "read", "field": "model", "from": "reader", "to": "renamed"}
    spec["resume"] = {"tape": replay_tape.build(old_wire, run / "attempts" / "1" / "sends.jsonl"),
                      "pending": replay_tape.pending_request(old_wire), "route_changes": [move]}
    _mark_pending_verified(run, spec["resume"]["pending"], route={**ROUTE, "model": "renamed"})
    marker = run / "resume-pending-verified.json"
    stale = json.loads(marker.read_text())
    if stale_field == "request_sha256":
        stale["request_sha256"] = "0" * 64
    else:
        stale["route"]["model"] = "wrong"
    marker.write_text(json.dumps(stale))
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert "pending" not in resumed
    assert resumed["pending_check"]["reason"] == "unverified_pending"
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
    _mark_pending_verified(run, spec["resume"]["pending"], route={**ROUTE, "model": "new"})
    (run / "spec.json").write_text(json.dumps(spec))

    resume.archive_attempt({"run": run, "spec": spec, "attempt": 2, "max_turns": 8}, tmp_path,
                           use_checkpoint=False)

    resumed = json.loads((run / "spec.json").read_text())["resume"]
    assert len(resumed["tape"]) == 1
    assert "pending" not in resumed
    assert resumed["route_changes"][0]["to"] == "new"
    assert "pending_check" not in resumed


@pytest.mark.parametrize("divergence", ["marker", "request"])
def test_diverged_replay_skips_prior_pending_but_keeps_route_changes(tmp_path, divergence):
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
    assert "pending" not in resumed
    assert resumed["route_changes"][0]["to"] == "new"
    assert resumed["pending_check"]["reason"] == (
        "diverged" if divergence == "marker" else "unverified_pending")


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


@pytest.mark.parametrize("case", ["passing", "model_move", "failing", "orphan", "unrouted"])
def test_child_records_pending_verification_before_live_send(tmp_path, monkeypatch, case):
    monkeypatch.syspath_prepend(str(Path(_child.__file__).parent / "engine"))
    import kernel_bounded_http

    run = tmp_path / "run"
    run.mkdir()
    old_wire = tmp_path / "old" / "wire"
    if case == "orphan":
        old_wire.mkdir(parents=True)
        (old_wire / "0003.route.json").write_text(json.dumps(ROUTE))
    else:
        _write_send(old_wire, 3, completed=False)
    pending = replay_tape.pending_request(old_wire) if case != "unrouted" else None
    move = {"role": "read", "field": "model", "from": "reader", "to": "renamed"}
    replay = replay_tape.Replayer([], pending=pending,
                                  route_changes=[move] if case == "model_move" else None)
    marker = run / "resume-pending-verified.json"
    wire = (b'{"model": "reader", "messages": [1]}' if case == "failing" else
            b'{"model": "renamed", "messages": []}' if case == "model_move" else WIRE)
    route = {**ROUTE, "model": "renamed"} if case == "model_move" else ROUTE
    replaced = []
    replace = _child.os.replace

    def observed_replace(source, target):
        replaced.append(Path(target).name)
        replace(source, target)

    def physical(request, endpoint, key, **kwargs):
        assert request == wire and endpoint == URL
        assert marker.exists() is (case in {"passing", "model_move", "orphan"})
        if case in {"passing", "model_move", "orphan"}:
            assert replaced[-1] == "resume-pending-verified.json"
        return RESPONSE

    monkeypatch.setattr(_child.os, "replace", observed_replace)
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", physical)
    _child.install_wire_recorder(run, URL, replayer=replay,
                                 routed_endpoints={URL} if case != "unrouted" else None)
    kwargs = {"route_meta": route} if case != "unrouted" else {}
    assert kernel_bounded_http.request_bytes(wire, URL, "", deadline=1000000,
                                             socket_timeout=10, **kwargs) == RESPONSE
    if case in {"passing", "model_move", "orphan"}:
        assert json.loads(marker.read_text()) == {
            "send": 3, "request_sha256": None if case == "orphan" else hashlib.sha256(WIRE).hexdigest(),
            "recorded_route": ROUTE, "route": route}
        assert replay.diverged is None
    else:
        assert not marker.exists()
        assert "resume-pending-verified.json" not in replaced
        assert (run / "resume-divergence.json").exists() is (case == "failing")


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
