"""Checkpoint restore must preserve the resumed child's layout replay authority."""
from __future__ import annotations

import contextvars
import io
import json
from types import SimpleNamespace

import pytest

import kernel_bounded_http
import kernel_cache_layout_candidate as layout
import kernel_checkpoint as checkpoint
from kernel_measurement import ACTIVE_REQUEST
from stateframe import _child
from stateframe.replay_tape import Replayer, pending_request
from test_stable_wire_record import ENDPOINT, RESPONSE
from test_stable_wire_trim_selection import allocator as allocator


PROVIDERS = (
    "_REPLAY_LAYOUT_DECISION_PROVIDER",
    "_REPLAY_STABLE_WIRE_DECISION_PROVIDER",
    "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
    "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
)


@pytest.fixture
def checkpoint_run(monkeypatch, tmp_path):
    # Snapshot the real layout module without serializing this pytest process's
    # unrelated engine patches, contexts or registered host participants.
    for name in ("_STATE", "_SAVE", "_SCANS", "_PARTICIPANTS"):
        monkeypatch.setattr(checkpoint, name, {})
    for name in ("_SIGNING_KEY", "_SECRETS"):
        monkeypatch.setattr(checkpoint, name, [])
    monkeypatch.setattr(checkpoint, "_is_state_module", lambda name, module: module is layout)
    monkeypatch.setattr(checkpoint, "_patched_callables", lambda: {})
    monkeypatch.setattr(checkpoint, "_context_items", lambda: [])
    monkeypatch.setattr(checkpoint.contextvars, "copy_context", contextvars.Context)
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda saved: None)
    monkeypatch.setattr(_child, "HOST_STATE", {})
    monkeypatch.setattr(_child, "emit", lambda *args, **kwargs: None)
    checkpoint.register_signing_key(b"checkpoint-layout-test-key-0123456789")
    run = tmp_path / "engine-state"
    run.mkdir()
    frame = SimpleNamespace(f_code=checkpoint.save.__code__, f_locals={"step": 1})
    checkpoint._STATE["setup"] = {"code": frame.f_code, "line": 0, "funcs": frozenset()}
    return run, frame


def _saved_payload(saved):
    _, raw = checkpoint._validate(saved)
    return checkpoint._Unpickler(io.BytesIO(raw), {"framecells": {}, "framefuncs": {}}).load()


@pytest.mark.parametrize("installed", [False, True])
def test_checkpoint_payload_has_no_layout_replay_provider(monkeypatch, checkpoint_run, installed):
    run, frame = checkpoint_run
    for name in PROVIDERS:
        monkeypatch.setattr(layout, name, (lambda: None) if installed else None)
    saved = checkpoint.save(frame, 1, run)
    state = _saved_payload(saved)["modules"][layout.__name__]
    assert not set(PROVIDERS).intersection(state)


@pytest.mark.parametrize("checkpoint_during_replay", [False, True])
def test_checkpoint_repeated_resume_replays_recorded_trim_with_lower_meter(
        monkeypatch, tmp_path, allocator, checkpoint_run, checkpoint_during_replay):
    prepare, render = allocator
    run, frame = checkpoint_run
    prepared = prepare()
    starting = json.dumps(prepared[0]).encode()
    frame.f_locals["runtime"] = prepared[2]
    saved = checkpoint.save(frame, 1, run)
    live = render(prepared)
    wire = json.dumps(live).encode()
    record = prepared[1]._stable_wire_reserves_record
    assert record["reserves"] == [[70, 256]]
    assert not any("[Optional context]" in row.get("content", "") for row in live["messages"])

    class Interrupted(BaseException):
        pass

    def interrupt(sent, *args, **kwargs):
        assert sent == wire
        raise Interrupted

    recorded = tmp_path / "recorded"
    recorded.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", interrupt)
    _child.install_wire_recorder(recorded, ENDPOINT)
    active = ACTIVE_REQUEST.set(prepared[1])
    try:
        with pytest.raises(Interrupted):
            kernel_bounded_http.request_bytes(wire, ENDPOINT, "")
    finally:
        ACTIVE_REQUEST.reset(active)

    # Interrupt the first resumed send too, then resume its checkpoint again.
    # Each attempt installs fresh providers before the actual disk restore.
    for attempt in (1, 2):
        player = Replayer([], strict=True, pending=pending_request(recorded / "wire"))
        resumed = tmp_path / f"resumed-{attempt}"
        resumed.mkdir()
        monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                            interrupt if attempt == 1 else lambda *args, **kwargs: RESPONSE)
        _child.install_wire_recorder(resumed, ENDPOINT, player)
        providers = {name: getattr(layout, name) for name in PROVIDERS}
        checkpoint.restore(frame, saved, run)
        replay = prepare(replay=True)
        assert json.dumps(replay[0]).encode() == starting
        replay = (replay[0], replay[1], frame.f_locals["runtime"], replay[3])
        rebuilt = render(replay)
        active = ACTIVE_REQUEST.set(replay[1])
        try:
            if attempt == 1:
                with pytest.raises(Interrupted):
                    kernel_bounded_http.request_bytes(json.dumps(rebuilt).encode(), ENDPOINT, "")
            else:
                assert kernel_bounded_http.request_bytes(
                    json.dumps(rebuilt).encode(), ENDPOINT, "") == RESPONSE
        finally:
            ACTIVE_REQUEST.reset(active)
        assert json.dumps(rebuilt).encode() == wire
        assert player.diverged is None
        assert player.pending is None
        assert all(getattr(layout, name) is provider for name, provider in providers.items())
        assert replay[1]._stable_wire_reserves_record == record
        assert json.loads((resumed / "wire/0001.stable-wire.json").read_bytes()) == record
        if attempt == 1 and checkpoint_during_replay:
            # Checkpoints taken during replay must not capture provider closures
            # or their bound Replayer either.
            frame.f_locals["runtime"] = prepare(replay=True)[2]
            saved = checkpoint.save(frame, 2, run)
        recorded = resumed
