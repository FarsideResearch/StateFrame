"""Checkpoint, initial selection, qualification and marker handling compose."""
from __future__ import annotations

import json

import kernel_bounded_http
import kernel_cache_layout_candidate as layout
import kernel_checkpoint as checkpoint
import kernel_frame_allocation_capture as allocation
import kernel_stable_wire as stable
import pytest
from kernel_measurement import ACTIVE_REQUEST
from test_stable_wire_checkpoint import PROVIDERS
from test_stable_wire_checkpoint import checkpoint_run as checkpoint_run  # noqa: PLC0414 - imported pytest fixture
from test_stable_wire_qualification import _recover, _snapshot
from test_stable_wire_record import ENDPOINT, RESPONSE
from test_stable_wire_trim_selection import allocator as allocator  # noqa: PLC0414 - imported pytest fixture

from stateframe import _child
from stateframe.replay_tape import Replayer, build, pending_request


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("zero_trim", [False, True])
def test_all_four_fixes_survive_repeated_checkpoint_replay(
        monkeypatch, tmp_path, allocator, checkpoint_run, pending, zero_trim):
    prepare, render = allocator
    run, frame = checkpoint_run
    marker = {"role": "user", "content": "[Current source: pkg/echo.py] ordinary quoted text"}
    prepared = prepare(replay=zero_trim, marker=marker)
    frame.f_locals["runtime"] = prepared[2]
    saved = checkpoint.save(frame, 1, run)
    live = render(prepared)
    wire = json.dumps(live).encode()
    record = prepared[1]._stable_wire_reserves_record
    assert stable.is_stable_wire_projection(live["messages"])
    assert record["initial_selection"] == prepared[1]._stable_wire_initial_selection
    assert record["trim_count"] == (0 if zero_trim else 1)
    assert record["reserves"] == ([] if zero_trim else [[70, 256]])
    assert bool(record.get("selections")) is not zero_trim
    canonical, _ = stable.canonicalize_stable_wire(live["messages"])
    assert canonical.count(marker) == 1
    assert canonical.index(marker) < stable._frame_index(canonical)
    windows, epoch = stable.recorded_window_layout(live["messages"])
    assert len(windows) == 1 and epoch == record["epoch"]
    assert marker["content"] not in windows

    class Interrupted(BaseException):
        pass

    def interrupt(sent, *args, **kwargs):
        assert sent == wire
        raise Interrupted

    recorded = tmp_path / "recorded"
    recorded.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        interrupt if pending else lambda *args, **kwargs: RESPONSE)
    _child.install_wire_recorder(recorded, ENDPOINT)
    active = ACTIVE_REQUEST.set(prepared[1])
    try:
        if pending:
            with pytest.raises(Interrupted):
                kernel_bounded_http.request_bytes(wire, ENDPOINT, "")
        else:
            assert kernel_bounded_http.request_bytes(wire, ENDPOINT, "") == RESPONSE
    finally:
        ACTIVE_REQUEST.reset(active)

    for attempt in (1, 2):
        player = Replayer(build(recorded / "wire", recorded / "sends.jsonl"), strict=True,
                          pending=pending_request(recorded / "wire"))
        assert player.reserves_for_layout_decision() == record
        resumed = tmp_path / f"resumed-{attempt}"
        resumed.mkdir()
        transported = []

        def transport(sent, *args, _transported=transported, _attempt=attempt, **kwargs):
            _transported.append(sent)
            assert sent == wire
            if pending and _attempt == 1:
                raise Interrupted
            return RESPONSE

        monkeypatch.setattr(kernel_bounded_http, "request_bytes", transport)
        _child.install_wire_recorder(resumed, ENDPOINT, player)
        providers = {name: getattr(layout, name) for name in PROVIDERS}
        checkpoint.restore(frame, saved, run)
        assert all(getattr(layout, name) is provider for name, provider in providers.items())
        replay = prepare(token_delta=30, marker=marker)
        replay = (replay[0], replay[1], frame.f_locals["runtime"], replay[3])
        snapshot = _snapshot(replay)
        rebuilt = render(replay)
        assert json.dumps(rebuilt).encode() == wire
        assert replay[1]._stable_wire_reserves_record == record

        def forbidden(*args, **kwargs):
            pytest.fail("recorded qualification consulted a current measurement")

        with monkeypatch.context() as qualification_patch:
            qualification_patch.setattr(replay[3], "measure", forbidden)
            for _ in range(2):
                rebuilt, changed = _recover(replay, rebuilt, snapshot)
                assert not changed
                assert json.dumps(rebuilt).encode() == wire
                assert allocation.final_request_tokens(rebuilt, replay[3], replay[1]) == (
                    snapshot[1].tokens, snapshot[1].method)
        assert not [event for event in replay[1].journal.events
                    if event["kind"] in {"stable_wire_replay_mismatch", "stable_wire_failed_open"}]

        # Snapshot while the fresh providers still point at the unconsumed send.
        frame.f_locals["runtime"] = replay[2]
        saved = checkpoint.save(frame, attempt + 1, run)
        active = ACTIVE_REQUEST.set(replay[1])
        try:
            if pending and attempt == 1:
                with pytest.raises(Interrupted):
                    kernel_bounded_http.request_bytes(json.dumps(rebuilt).encode(), ENDPOINT, "")
            else:
                assert kernel_bounded_http.request_bytes(
                    json.dumps(rebuilt).encode(), ENDPOINT, "") == RESPONSE
        finally:
            ACTIVE_REQUEST.reset(active)
        assert transported == ([wire] if pending else [])
        assert player.diverged is None and not player.pending_mismatch
        if pending:
            assert player.pending_verified is not None and player.pending is None
        else:
            assert player.cursor == 1
        assert player.reserves_for_layout_decision() is None
        assert (resumed / "wire/0001.request.json").read_bytes() == wire
        assert json.loads((resumed / "wire/0001.stable-wire.json").read_bytes()) == record
        recorded = resumed
