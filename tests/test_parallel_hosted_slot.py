"""Subframe slot leases, provider pins, and historical replay identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from stateframe._child import configure_subframe_slot
from stateframe.replay_tape import Replayer


def test_unpinned_subframe_keeps_lease_identity_out_of_provider_environment(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACT_SLOT_ID", "99")
    live_slot = configure_subframe_slot(tmp_path, 3, pinned=False, replayer=None)
    assert live_slot is None
    assert "REACT_SLOT_ID" not in os.environ
    assert json.loads((tmp_path / "parallel-slot.json").read_bytes()) == {
        "slot": 3, "pinned": False}


def test_slot_capable_subframe_pins_the_leased_slot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REACT_SLOT_ID", raising=False)
    assert configure_subframe_slot(tmp_path, 5, pinned=True, replayer=None) == 5
    assert os.environ["REACT_SLOT_ID"] == "5"
    assert json.loads((tmp_path / "parallel-slot.json").read_bytes()) == {
        "slot": 5, "pinned": True}


def test_legacy_pinned_slot_replays_exact_request_then_continues_unpinned(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REACT_SLOT_ID", raising=False)
    (tmp_path / "parallel-slot.json").write_text('{"slot": 7}')
    request = b'{"model":"scripted","id_slot":7,"cache_prompt":false}'
    response = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
    request_path = tmp_path / "old.request.json"
    response_path = tmp_path / "old.response.json"
    request_path.write_bytes(request)
    response_path.write_bytes(response)
    replay = Replayer([{
        "send": 1, "request": str(request_path), "response": str(response_path),
        "response_sha256": hashlib.sha256(response).hexdigest(),
    }], strict=True)

    assert configure_subframe_slot(tmp_path, 2, pinned=False, replayer=replay) is None
    assert os.environ["REACT_SLOT_ID"] == "7"
    rebuilt = json.dumps({"model": "scripted", "id_slot": int(os.environ["REACT_SLOT_ID"]),
                          "cache_prompt": False}, separators=(",", ":")).encode()
    assert rebuilt == request
    assert replay.answer(rebuilt) == response
    assert replay.active is False
    assert configure_subframe_slot(tmp_path, 2, pinned=False, replayer=replay) is None
    assert "REACT_SLOT_ID" not in os.environ
    assert json.loads((tmp_path / "parallel-slot.json").read_bytes()) == {"slot": 7}


def test_legacy_pending_request_keeps_historical_pin(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REACT_SLOT_ID", raising=False)
    (tmp_path / "parallel-slot.json").write_text('{"slot": 7}')
    replay = Replayer([], pending={"send": 1, "request": str(tmp_path / "old.request.json")})
    assert configure_subframe_slot(tmp_path, 2, pinned=False, replayer=replay) is None
    assert os.environ["REACT_SLOT_ID"] == "7"


def test_new_unpinned_tape_stays_unpinned_during_replay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACT_SLOT_ID", "99")
    (tmp_path / "parallel-slot.json").write_text('{"slot": 3, "pinned": false}')
    replay = Replayer([], pending={"send": 1, "request": str(tmp_path / "old.request.json")})
    assert configure_subframe_slot(tmp_path, 2, pinned=False, replayer=replay) is None
    assert "REACT_SLOT_ID" not in os.environ


@pytest.mark.parametrize("record", ['{"slot": 3, "pinned": "yes"}',
                                    '{"slot": 3, "unexpected": true}'])
def test_historical_slot_rejects_untrusted_pin_record(
        tmp_path: Path, record: str) -> None:
    (tmp_path / "parallel-slot.json").write_text(record)
    with pytest.raises(RuntimeError, match="historical slot .*invalid"):
        configure_subframe_slot(tmp_path, 2, pinned=False, replayer=None)


def test_historical_slot_rejects_dangling_symlink(tmp_path: Path) -> None:
    (tmp_path / "parallel-slot.json").symlink_to(tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="historical slot artifact is unsafe"):
        configure_subframe_slot(tmp_path, 2, pinned=False, replayer=None)


def test_historical_slot_rejects_oversized_record(tmp_path: Path) -> None:
    (tmp_path / "parallel-slot.json").write_text('{"slot": 3}' + " " * 257)
    with pytest.raises(RuntimeError, match="exceeds 256 bytes"):
        configure_subframe_slot(tmp_path, 2, pinned=False, replayer=None)


def test_resume_refuses_missing_historical_slot_even_after_replay(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="historical slot artifact is missing for replay"):
        configure_subframe_slot(tmp_path, 2, pinned=False, replayer=Replayer([]))
